"""
FastAPI server: one photo in, depth + occupancy grid + path + a plain verdict out.

    uvicorn backend.app:app --port 7860          (from the project root)

Two kinds of input, deliberately treated differently:
  sample  a bundled KITTI frame with its real calibration and camera height.
          The numbers shown for it (LiDAR grid metrics, depth AbsRel) are measured
          live against KITTI ground truth.
  upload  any photo. Intrinsics come from the user's FOV, else EXIF, else 70 deg;
          camera height from the user. Marked experimental: accuracy on cameras
          other than KITTI's is not measured (see the README, Limitations).

Both go through pipeline.fit_to_size to 294 x 392 (portrait photos: 392 x 294),
the size benchmark.py chose, and then pipeline.run with ONNX Runtime models.
No torch anywhere in this process.
"""
import base64
import io
import json
import math
import os
import threading
import time
from typing import Literal

import cv2
import numpy as np
import onnxruntime as ort
import pillow_heif
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

from freespace.bev import UNKNOWN
from freespace.metrics import compute_depth_metrics
from freespace.pipeline import PRESETS, counts, fit_to_size, onnx_fn, ratios, run

MODEL_DIR = os.environ.get("MODEL_DIR", "checkpoints/onnx")
SAMPLE_DIR = os.environ.get("SAMPLE_DIR", "backend/samples")
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
THREADS = int(os.environ.get("ORT_THREADS", "2"))   # what benchmark.py measured; a free Space has 2 vCPUs
MODEL_HW = (294, 392)           # benchmark.py's choice: provisional, chosen on KITTI only
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 40_000_000         # a small file can still decode to a huge image ("decompression bomb")
DEFAULT_HFOV = 70.0
F35_DIAGONAL_MM = 43.27         # the 35 mm-equivalent focal length is defined by the frame diagonal
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
# iPhones save HEIC by default. pillow-heif teaches Pillow to read it, EXIF included;
# it returns the pixels already rotated upright, with EXIF orientation reset to 1.
pillow_heif.register_heif_opener()
# Pillow format name -> what users call it. iPhone JPEGs open as "MPO" (a JPEG with
# extra embedded images), so it must count as JPEG.
ACCEPTED = {"JPEG": "JPEG", "MPO": "JPEG", "PNG": "PNG", "WEBP": "WebP", "HEIF": "HEIC/HEIF"}
ACCEPTED_TEXT = "JPEG, PNG, WebP or HEIC/HEIF"

so = ort.SessionOptions()
so.intra_op_num_threads, so.inter_op_num_threads = THREADS, 1
cv2.setNumThreads(THREADS)
FNS = []
for name in ("depth", "segformer"):
    with open(os.path.join(MODEL_DIR, f"{name}.json")) as fh:
        meta = json.load(fh)
    FNS.append(onnx_fn(ort.InferenceSession(os.path.join(MODEL_DIR, f"{name}.onnx"), so,
                                            providers=["CPUExecutionProvider"]), meta["mean"], meta["std"]))
DEPTH_FN, SEG_FN = FNS
# A 2-vCPU host cannot run two requests at once without both getting slower
LOCK = threading.Lock()

# What the host really gives us. Inside a container os.cpu_count() reports the
# HOST machine's cores, not our share; the cgroup CPU quota / period is the actual
# vCPU limit. cgroup v2 keeps it in cpu.max ("quota period", quota "max" = none),
# v1 (e.g. Docker Desktop on WSL2) in two files (quota -1 = none).
# Both stay None where the files don't exist (Windows).
CPU_MODEL = CPU_LIMIT = None
if os.path.exists("/proc/cpuinfo"):
    with open("/proc/cpuinfo") as fh:
        CPU_MODEL = next((l.split(":", 1)[1].strip() for l in fh if l.startswith("model name")), None)
if os.path.exists("/sys/fs/cgroup/cpu.max"):
    with open("/sys/fs/cgroup/cpu.max") as fh:
        quota, period = fh.read().split()
elif os.path.exists("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
    with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fq, open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
        quota, period = fq.read().strip(), fp.read().strip()
else:
    quota = period = None
if quota not in (None, "max", "-1"):
    CPU_LIMIT = int(quota) / int(period)

SAMPLES = {}
if os.path.exists(os.path.join(SAMPLE_DIR, "samples.json")):
    with open(os.path.join(SAMPLE_DIR, "samples.json")) as fh:
        for s in json.load(fh)["samples"]:
            d = os.path.join(SAMPLE_DIR, s["id"])
            s["image"] = cv2.cvtColor(cv2.imread(os.path.join(d, "image.png")), cv2.COLOR_BGR2RGB)
            s["gt_depth"] = cv2.imread(os.path.join(d, "gt_depth.png"), cv2.IMREAD_UNCHANGED) / 256.0
            s["lidar_grid"] = cv2.imread(os.path.join(d, "lidar_grid.png"), cv2.IMREAD_UNCHANGED)
            s["K"] = np.array(s["K"])
            SAMPLES[s["id"]] = s

app = FastAPI(title="Monocular free-space perception")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def png(rgb):
    return base64.b64encode(cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1]).decode()


def grid_json(g, P):
    return {"data": b64(g.astype(np.uint8)), "shape": list(g.shape), "cell_m": P["cell"],
            "x_max_m": P["x_max"], "y_half_m": P["y_half"], "codes": {"0": "free", "1": "occupied", "2": "unknown"}}


def hfov(K, w):
    return math.degrees(2 * math.atan(w / 2 / K[0, 0]))


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True, "input_hw": list(MODEL_HW), "threads": THREADS,
            "onnxruntime": ort.__version__, "samples": len(SAMPLES),
            "cpu_model": CPU_MODEL, "cpu_limit": CPU_LIMIT, "host_cpu_count": os.cpu_count()}


@app.get("/samples")
def samples():
    out = []
    for s in SAMPLES.values():
        thumb = cv2.resize(s["image"], (196, 147), interpolation=cv2.INTER_AREA)
        jpg = cv2.imencode(".jpg", cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])[1]
        out.append({"id": s["id"], "drive": s["drive"], "frame": s["frame"],
                    "thumbnail_jpg": base64.b64encode(jpg).decode(), "camera_height_m": s["camera_height_m"],
                    "hfov_deg": round(hfov(s["K"], s["image"].shape[1]), 1), "has_lidar": True})
    return out


def read_upload(file):
    """Bytes -> upright RGB array + EXIF 35 mm focal length, with every failure a clear HTTP error."""
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"file larger than {MAX_BYTES // (1024 * 1024)} MB")
    try:
        im = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError:
        raise HTTPException(413, f"image larger than {MAX_PIXELS // 1_000_000} megapixels")
    except UnidentifiedImageError:
        raise HTTPException(415, f"not a readable image; send {ACCEPTED_TEXT}")
    if im.format not in ACCEPTED:
        raise HTTPException(415, f"unsupported image format ({im.format}); send {ACCEPTED_TEXT}")
    if im.width * im.height > MAX_PIXELS:
        raise HTTPException(413, f"image larger than {MAX_PIXELS // 1_000_000} megapixels")
    try:
        f35 = im.getexif().get_ifd(0x8769).get(0xA405)      # FocalLengthIn35mmFilm
    except Exception:                                       # broken EXIF is not a reason to refuse the photo
        f35 = None
    try:
        # orientation first: phones store portrait photos sideways plus a rotate flag
        im = ImageOps.exif_transpose(im).convert("RGB")
    except (OSError, SyntaxError, ValueError) as e:
        raise HTTPException(400, f"image could not be decoded ({e})")
    return np.asarray(im), (float(f35) if f35 else None)


@app.post("/predict")
def predict(file: UploadFile | None = File(None),
            sample_id: str | None = Form(None),
            camera_height: float = Form(1.5, ge=0.1, le=5.0),
            hfov_deg: float | None = Form(None, ge=20.0, le=150.0),
            f35_mm: float | None = Form(None, gt=0.0, le=2000.0),
            drivable: Literal["none", "road", "road+sidewalk"] | None = Form(None),
            preset: Literal["outdoor", "indoor"] = Form("outdoor")):
    t_start = time.perf_counter()
    if (file is None) == (sample_id is None):
        raise HTTPException(400, "send exactly one of sample_id or file")

    if sample_id is not None:
        s = SAMPLES.get(sample_id)
        if s is None:
            raise HTTPException(404, f"unknown sample_id {sample_id!r}")
        img, K = s["image"], s["K"]
        # a sample is "measured" only with its real camera: calibration, height, KITTI grid
        camera_height, preset, source = s["camera_height_m"], "outdoor", "kitti_calibration"
        full_hfov = hfov(K, img.shape[1])
    else:
        img, f35 = read_upload(file)
        # the browser shrinks photos before uploading, which drops their EXIF, so it
        # reads FocalLengthIn35mmFilm from the original and sends it here; it is used
        # exactly like the EXIF value (a proportional resize leaves the formula valid)
        f35 = f35_mm or f35
        H, W = img.shape[:2]
        if min(H, W) < min(MODEL_HW):
            raise HTTPException(400, f"image too small: {W}x{H}, need at least {min(MODEL_HW)} px on the short side")
        # an FOV the user typed wins (the UI pre-fills it from EXIF); then EXIF; then the default
        if hfov_deg is not None:
            fx, source = (W / 2) / math.tan(math.radians(hfov_deg) / 2), "user"
        elif f35:
            fx, source = f35 * math.hypot(W, H) / F35_DIAGONAL_MM, "exif_35mm"
        else:
            fx, source = (W / 2) / math.tan(math.radians(DEFAULT_HFOV) / 2), "default_70"
        K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])
        full_hfov = hfov(K, W)
    original_hw = img.shape[:2]
    h, w = MODEL_HW if original_hw[1] >= original_hw[0] else MODEL_HW[::-1]   # portrait stays portrait
    img_m, K_m, box = fit_to_size(img, K, h, w)
    t_decode = time.perf_counter() - t_start

    with LOCK:
        out = run(img_m, K_m, camera_height, DEPTH_FN, SEG_FN, drivable=drivable, preset=preset)
    t0 = time.perf_counter()
    P = PRESETS[preset]
    ok = out["status"] == "ok"
    d = out["depth"]
    lo, hi = (float(np.percentile(d, 2)), float(np.percentile(d, 98)))
    # near = warm colours; the legend range is only metres when the scale was accepted
    norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    depth_rgb = cv2.cvtColor(cv2.applyColorMap((255 * (1 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO),
                             cv2.COLOR_BGR2RGB)
    is_sample = sample_id is not None
    res = {"status": out["status"], "reason": out["reason"], "source": "sample" if is_sample else "upload",
           "sample_id": sample_id, "experimental": not is_sample,
           "input": {"original_hw": list(original_hw), "model_hw": [h, w],
                     "crop_box": dict(zip(("y0", "x0", "h", "w"), map(int, box)))},
           "camera": {"height_m": camera_height, "fx": K_m[0, 0], "fy": K_m[1, 1], "cx": K_m[0, 2], "cy": K_m[1, 2],
                      "hfov_deg": round(full_hfov, 1), "model_hfov_deg": round(hfov(K_m, w), 1),
                      "fov_source": source},
           "preset": preset, "drivable": out["drivable"], "robot_radius_m": out["robot_radius"],
           "scale": out["scale"], "tilt_deg": out["tilt_deg"], "depth_scaled": out["depth_scaled"],
           "depth_png": png(depth_rgb), "depth_range_m": [lo, hi] if ok else None,
           "assumptions": ([f"Measured sample: KITTI's own calibration and camera height ({camera_height} m)."]
                           if is_sample else
                           ["Experimental: accuracy on cameras other than KITTI's has not been measured.",
                            f"Camera height {camera_height} m is user-supplied and sets the metric scale.",
                            f"Horizontal field of view {full_hfov:.0f} deg, from: {source}."])
                          + ["The ground in front of the camera is assumed flat.",
                             "The depth model was fine-tuned on German streets (KITTI)."]}
    if ok:
        res.update(painted_png=png(out["painted"]), grid=grid_json(out["grid"], P),
                   grid_inflated=grid_json(out["grid_inflated"], P),
                   path_m=None if out["path_m"] is None else np.round(out["path_m"], 3).tolist(),
                   verdict=out["verdict"], ribbon_px=out["ribbon_px"],
                   distance_lines_px=out["distance_lines_px"], distance_marks_m=list(P["marks"]))
        if is_sample:                       # measured against KITTI ground truth, for THIS request
            gt = s["lidar_grid"]
            known = gt != UNKNOWN
            m = ratios(counts(out["grid"], gt, known))
            c_inf = counts(out["grid_inflated"], gt, known)
            m["false_free_rate_inflated"] = c_inf[2] / max(c_inf[3], 1)
            pred = cv2.resize(d, s["gt_depth"].shape[::-1], interpolation=cv2.INTER_LINEAR)
            res["lidar"] = {"grid": grid_json(gt, P), "metrics": {k: float(v) for k, v in m.items()},
                            "depth_abs_rel": compute_depth_metrics(pred, s["gt_depth"])["abs_rel"]}
    else:
        res["verdict"] = {"level": "refused", "text": f"No floor found: {out['reason']}"}
    t = out["timings"]
    res["timings_ms"] = {"decode_resize": 1000 * t_decode, "depth": 1000 * t["depth"], "seg": 1000 * t["seg"],
                         "geometry": 1000 * t["geometry"], "encode": 1000 * (time.perf_counter() - t0),
                         "total": 1000 * (time.perf_counter() - t_start)}
    return res

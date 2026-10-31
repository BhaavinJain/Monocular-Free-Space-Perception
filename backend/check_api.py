"""
M4 check. Start the server first (uvicorn backend.app:app --port 7860), then:

    python -m backend.check_api

Endpoint checks (against the running server):
  health, samples; every sample measured live (its depth AbsRel must equal
  report/benchmark.json for that frame: proof the server runs the measured
  pipeline); a full KITTI frame UPLOADED with h = 1.65, hfov = 80 deg must
  give a sensible grid (scale within 5% and LiDAR free-space IoU within 0.05
  of the same frame as a sample); EXIF orientation / focal length / FOV
  source; bad inputs rejected with the right status code.
Pipeline-level check (local, same ONNX models): crops of the UPPER part of
  KITTI frames (no road) must be refused by the plane fit. They are smaller
  than the upload minimum, which is why this runs below the endpoint. A crop
  that is NOT refused is reported as a finding, not "fixed" by tuning.
Optional: backend/test_images/wall_with_floor.jpg is sent through /predict and
  its result recorded (borderline case, not asserted).
Results -> report/api_check.json, crops -> report/image/m4_refusal_crops.png
"""
import argparse
import base64
import io
import json
import math
import os

import cv2
import numpy as np
import onnxruntime as ort
import requests
from PIL import Image, ImageOps

from freespace.bev import UNKNOWN
from freespace.kitti_loader import KittiRawDrive
from freespace.pipeline import CITYSCAPES, counts, fit_to_size, onnx_fn, ratios, run

# (frame, x0): top-of-image crops of drive 0023, 150 x 200 (4:3) — rows 0-149
# are above KITTI's horizon (cy ~ 173), so the lower 40% holds no road
REFUSAL_CROPS = [(5, 60), (107, 330), (210, 600), (262, 850), (365, 1000), (468, 200)]
CROP_H, CROP_W = 150, 200


def jpeg(rgb, **kw):
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, "JPEG", quality=90, **kw)
    return buf.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:7860")
    p.add_argument("--root", default="data")
    p.add_argument("--onnx_dir", default="checkpoints/onnx")
    p.add_argument("--samples", default="backend/samples")
    p.add_argument("--test_images", default="backend/test_images")
    p.add_argument("--benchmark", default="report/benchmark.json")
    p.add_argument("--report_dir", default="report")
    p.add_argument("--phone_height", type=float, default=1.1, help="camera height of the owner's phone photos (m)")
    a = p.parse_args()
    checks, findings = [], []

    def check(name, passed, **info):
        checks.append({"name": name, "passed": bool(passed), **info})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}  {info if info else ''}")

    def post(files=None, **data):
        return requests.post(f"{a.url}/predict", files=files, data=data, timeout=120)

    def grid(g):
        return np.frombuffer(base64.b64decode(g["data"]), np.uint8).reshape(g["shape"])

    # ---- basics ----
    r = requests.get(f"{a.url}/health", timeout=30)
    check("GET /health", r.status_code == 200 and r.json()["models_loaded"], body=r.json())
    samples = requests.get(f"{a.url}/samples", timeout=30).json()
    check("GET /samples lists the gallery", len(samples) == 6, ids=[s["id"] for s in samples])
    with open(os.path.join(a.onnx_dir, "segformer.json")) as fh:
        check("pipeline.CITYSCAPES == segformer.json labels", json.load(fh)["labels"] == CITYSCAPES)

    # ---- every sample: measured live, must reproduce benchmark.json ----
    with open(a.benchmark) as fh:
        bench = json.load(fh)["accuracy"]["294x392"]["per_frame"]
    sample_res = {}
    for s in samples:
        res = post(sample_id=s["id"]).json()
        sample_res[s["id"]] = res
        ref = bench[str(s["frame"])]["height"]
        ok = (res["status"] == "ok" and grid(res["grid"]).shape == (200, 200) and "lidar" in res
              and abs(res["lidar"]["depth_abs_rel"] - ref) < 1e-6)
        check(f"sample {s['id']}", ok, abs_rel=res.get("lidar", {}).get("depth_abs_rel"), benchmark_abs_rel=ref,
              lidar_metrics=res.get("lidar", {}).get("metrics"), scale=res["scale"],
              verdict=res["verdict"]["text"], total_ms=round(res["timings_ms"]["total"]))

    # ---- M4 check: a full KITTI frame uploaded, h = 1.65, hfov = 80 deg ----
    s = samples[0]
    drive = KittiRawDrive(a.root, "2011_09_26", "0023")
    full = drive.load_image(s["frame"])
    buf = io.BytesIO()
    Image.fromarray(full).save(buf, "PNG")
    res = post(files={"file": ("kitti.png", buf.getvalue(), "image/png")}, camera_height=1.65, hfov_deg=80).json()
    ref = sample_res[s["id"]]
    gt = cv2.imread(os.path.join(a.samples, s["id"], "lidar_grid.png"), cv2.IMREAD_UNCHANGED)
    iou = lambda g: float(ratios(counts(g, gt, gt != UNKNOWN))["free_iou"])
    up_iou = iou(grid(res["grid"])) if res["status"] == "ok" else None
    ok = (res["status"] == "ok" and res["experimental"] and res["camera"]["fov_source"] == "user"
          and abs(res["scale"] / ref["scale"] - 1) < 0.05 and up_iou >= iou(grid(ref["grid"])) - 0.05)
    check("upload full KITTI frame, h=1.65, hfov=80: sensible grid", ok, frame=s["frame"],
          scale_upload=res["scale"], scale_sample=ref["scale"], iou_upload=up_iou,
          iou_sample=iou(grid(ref["grid"])), verdict=res["verdict"]["text"],
          camera=res["camera"], total_ms=round(res["timings_ms"]["total"]))

    # ---- EXIF: orientation and focal length ----
    small = cv2.resize(full[:, 371:871], (400, 300), interpolation=cv2.INTER_AREA)   # landscape 4:3
    exif = Image.Exif()
    exif[0x0112] = 6                                        # "rotate 90 deg CW to view"
    exif.get_ifd(0x8769)[0xA405] = 26                       # FocalLengthIn35mmFilm = 26 mm
    res = post(files={"file": ("rot.jpg", jpeg(small, exif=exif.tobytes()), "image/jpeg")}).json()
    fx_expected = 26 * math.hypot(300, 400) / 43.27 * (294 / 300)    # upright 300x400 -> 392x294 model
    check("EXIF orientation 6 + f35: portrait, fx from the diagonal",
          res["input"]["original_hw"] == [400, 300] and res["input"]["model_hw"] == [392, 294]
          and res["camera"]["fov_source"] == "exif_35mm" and abs(res["camera"]["fx"] / fx_expected - 1) < 1e-9,
          input=res["input"], fx=res["camera"]["fx"], fx_expected=fx_expected, status=res["status"])
    res = post(files={"file": ("rot.jpg", jpeg(small, exif=exif.tobytes()), "image/jpeg")}, hfov_deg=60).json()
    check("user FOV wins over EXIF", res["camera"]["fov_source"] == "user" and res["camera"]["hfov_deg"] == 60.0,
          camera=res["camera"])
    res = post(files={"file": ("plain.jpg", jpeg(small), "image/jpeg")}).json()
    check("no EXIF, no FOV -> default 70 deg",
          res["camera"]["fov_source"] == "default_70" and res["camera"]["hfov_deg"] == 70.0, camera=res["camera"])

    # ---- bad inputs ----
    good = jpeg(small)
    bmp = io.BytesIO()
    Image.fromarray(small).save(bmp, "BMP")
    bmp = bmp.getvalue()
    buf = io.BytesIO()
    Image.new("1", (10000, 5000)).save(buf, "PNG")          # 50 MP, a few kB on disk
    bad = [("11 MB file", {"file": ("big.jpg", os.urandom(11 * 1024 * 1024), "image/jpeg")}, {}, 413),
           ("decompression bomb (50 MP PNG)", {"file": ("bomb.png", buf.getvalue(), "image/png")}, {}, 413),
           ("text file named .jpg", {"file": ("x.jpg", b"hello, not an image", "image/jpeg")}, {}, 415),
           ("BMP (readable, but not an accepted format)", {"file": ("x.bmp", bmp, "image/bmp")}, {}, 415),
           ("truncated JPEG", {"file": ("t.jpg", jpeg(full)[:20000], "image/jpeg")}, {}, 400),
           ("image too small (200x150)", {"file": ("s.jpg", jpeg(small[::2, ::2]), "image/jpeg")}, {}, 400),
           ("both sample_id and file", {"file": ("g.jpg", good, "image/jpeg")}, {"sample_id": samples[0]["id"]}, 400),
           ("neither sample_id nor file", None, {}, 400),
           ("camera_height = -1", {"file": ("g.jpg", good, "image/jpeg")}, {"camera_height": -1}, 422),
           ("drivable = foo", {"file": ("g.jpg", good, "image/jpeg")}, {"drivable": "foo"}, 422),
           ("hfov_deg = 5", {"file": ("g.jpg", good, "image/jpeg")}, {"hfov_deg": 5}, 422),
           ("unknown sample_id", None, {"sample_id": "nope"}, 404)]
    for name, files, data, code in bad:
        r = post(files=files, **data)
        detail = r.json().get("detail")
        ok = r.status_code == code and (code != 415 or "HEIC/HEIF" in str(detail))   # 415 must name the formats
        check(f"reject: {name} -> {code}", ok, got=r.status_code, detail=detail)

    # ---- phone photos (owner's, git-ignored; skipped when absent) ----
    # Every photo must be read with its EXIF focal length; a JPEG and a HEIC from the
    # same phone and orientation must give the same intrinsics. Results are recorded.
    phones = {}
    for fn in sorted(os.listdir(a.test_images)) if os.path.isdir(a.test_images) else []:
        ext = os.path.splitext(fn)[1].lower()
        if ext not in (".jpg", ".jpeg", ".heic", ".heif"):
            continue
        with open(os.path.join(a.test_images, fn), "rb") as fh:
            data = fh.read()
        mime = "image/heic" if ext in (".heic", ".heif") else "image/jpeg"
        r = post(files={"file": (fn, data, mime)}, camera_height=a.phone_height)
        b = r.json()
        phones[fn] = {"http": r.status_code, "file_mb": round(len(data) / 1e6, 2), "status": b.get("status"),
                      "reason": b.get("reason"), "input": b.get("input"), "camera": b.get("camera"),
                      "scale": b.get("scale"), "tilt_deg": b.get("tilt_deg"), "verdict": (b.get("verdict") or {}).get("text"),
                      "server_ms": (b.get("timings_ms") or {}).get("total")}
        check(f"phone photo {fn}: read, FOV from EXIF", r.status_code == 200 and b["camera"]["fov_source"] == "exif_35mm",
              **{k: phones[fn][k] for k in ("status", "reason", "scale", "verdict")}, fx=b.get("camera", {}).get("fx"))
    jpgs = [v for k, v in phones.items() if k.lower().endswith((".jpg", ".jpeg"))]
    heics = [v for k, v in phones.items() if k.lower().endswith((".heic", ".heif"))]
    if jpgs and heics:
        j, h = jpgs[0], heics[0]
        check("HEIC and JPEG: same model size and intrinsics",
              j["input"]["model_hw"] == h["input"]["model_hw"] and abs(j["camera"]["fx"] / h["camera"]["fx"] - 1) < 1e-9,
              jpeg_fx=j["camera"]["fx"], heic_fx=h["camera"]["fx"], model_hw=h["input"]["model_hw"])
    # the browser uploads phone JPEGs shrunk to 1280 px on the long side, re-encoded
    # WITHOUT EXIF, and sends the original's FocalLengthIn35mmFilm as f35_mm: that
    # must give the same intrinsics at model size as the original file
    for fn, v in phones.items():
        if not fn.lower().endswith((".jpg", ".jpeg")):
            continue
        im = Image.open(os.path.join(a.test_images, fn))
        f35 = im.getexif().get_ifd(0x8769).get(0xA405)
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((1280, 1280))                          # proportional, never upscales
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)                    # no exif= argument: EXIF dropped, as in the browser
        small = buf.getvalue()
        r0 = post(files={"file": (fn, small, "image/jpeg")}, camera_height=a.phone_height)
        r = post(files={"file": (fn, small, "image/jpeg")}, camera_height=a.phone_height, f35_mm=float(f35))
        b0, b = r0.json(), r.json()
        v["resized_upload"] = {"file_mb": round(len(small) / 1e6, 3), "size": list(im.size), "f35_mm": float(f35),
                               "fx": b["camera"]["fx"], "fov_source": b["camera"]["fov_source"],
                               "status": b["status"], "scale": b.get("scale"), "server_ms": b["timings_ms"]["total"]}
        check(f"phone photo {fn} resized to {im.size[0]}x{im.size[1]} + f35_mm: same fx as the original",
              b["camera"]["fov_source"] == "exif_35mm" and b["input"]["model_hw"] == v["input"]["model_hw"]
              and abs(b["camera"]["fx"] / v["camera"]["fx"] - 1) < 1e-6 and b["status"] == v["status"],
              original_fx=v["camera"]["fx"], resized_fx=b["camera"]["fx"], resized_mb=len(small) / 1e6,
              without_f35_source=b0["camera"]["fov_source"])

    # ---- refused plane fit, at pipeline level ----
    fns = []
    for k in ("depth", "segformer"):
        with open(os.path.join(a.onnx_dir, f"{k}.json")) as fh:
            meta = json.load(fh)
        fns.append(onnx_fn(ort.InferenceSession(os.path.join(a.onnx_dir, f"{k}.onnx"),
                                                providers=["CPUExecutionProvider"]), meta["mean"], meta["std"]))
    crops, tiles = [], []
    for f, x0 in REFUSAL_CROPS:
        crop = drive.load_image(f)[:CROP_H, x0:x0 + CROP_W]
        K = drive.K.copy()
        K[0, 2] -= x0                                       # y0 = 0, so cy is unchanged
        img, K_m, _ = fit_to_size(crop, K, 294, 392)
        out = run(img, K_m, 1.65, *fns)
        crops.append({"frame": f, "x0": x0, "status": out["status"], "reason": out["reason"],
                      "tilt_deg": out["tilt_deg"]})
        if out["status"] != "refused":
            findings.append(f"crop frame {f} x0 {x0} NOT refused (scale {out['scale']:.3f})")
        tile = img.copy()
        cv2.line(tile, (0, int(0.6 * 294)), (391, int(0.6 * 294)), (255, 255, 0), 1)   # top of the fit region
        cv2.putText(tile, f"{f}/{x0}: {out['status']}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        tiles.append(tile)
    sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in (0, 3)])
    os.makedirs(os.path.join(a.report_dir, "image"), exist_ok=True)
    cv2.imwrite(os.path.join(a.report_dir, "image", "m4_refusal_crops.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    check("upper-image KITTI crops are refused (pipeline level)",
          all(c["status"] == "refused" for c in crops), crops=crops)

    # ---- optional borderline photo, recorded only ----
    borderline = None
    path = os.path.join(a.test_images, "wall_with_floor.jpg")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            r = post(files={"file": ("wall_with_floor.jpg", fh.read(), "image/jpeg")})
        b = r.json()
        borderline = {"http": r.status_code, "status": b.get("status"), "reason": b.get("reason"),
                      "scale": b.get("scale"), "tilt_deg": b.get("tilt_deg"), "camera": b.get("camera"),
                      "verdict": b.get("verdict")}
        print(f"[INFO] wall_with_floor.jpg: {borderline}")

    report = {"passed": all(c["passed"] for c in checks), "findings": findings,
              "checks": checks, "borderline_wall_with_floor": borderline, "phone_photos": phones}
    path = os.path.join(a.report_dir, "api_check.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\nM4 {'PASSED' if report['passed'] else 'FAILED'}, findings: {findings or 'none'} -> {path}")


if __name__ == "__main__":
    main()

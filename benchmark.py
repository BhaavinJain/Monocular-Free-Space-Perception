"""
CPU latency AND accuracy of the full pipeline (ONNX Runtime) at a few input
sizes, to choose the size the server ships with.

Run twice, once per thread setting; each run adds its section to
report/benchmark.json, and once both exist the speed-vs-accuracy table and
the chosen size are written too:

    python benchmark.py --threads 2    # approximates a free Hugging Face Space (2 vCPU)
    python benchmark.py --threads 0    # this laptop, all cores (labelled as such)

Two laptop cores are still faster than two shared Space vCPUs, so even the
2-thread numbers are optimistic: M5 must re-measure on the real Space.

Test images: KITTI val frames, bottom-centre cropped to each size's aspect
ratio, then resized, with K updated for both. Accuracy = depth AbsRel in
height mode (scale from the camera height, exactly as the server does it)
against the LiDAR ground truth, with the prediction resized back to the crop's
resolution. 350x1218 is the reference: the crop every Phase 2-4 number used.
The oracle-scale AbsRel and s/s_oracle split each size's error into "wrong
metric scale" vs "wrong depth shape"; two diagnostic sizes (the same crops at
~native resolution) split "crop" from "resize". All of it is KITTI only — it
says nothing about accuracy on other cameras.
"""
import argparse
import os

p = argparse.ArgumentParser()
p.add_argument("--threads", type=int, default=2, help="0 = no limit (all cores)")
p.add_argument("--root", default="data")
p.add_argument("--depth_root", default="data/kitti_depth")
p.add_argument("--drive", default="0023")
p.add_argument("--frames", type=int, default=10)
p.add_argument("--onnx_dir", default="checkpoints/onnx")
p.add_argument("--report_dir", default="report")
p.add_argument("--budget_s", type=float, default=2.0, help="max 2-thread median total for the shipping size")
a = p.parse_args()
# numpy's BLAS (the SVD in the plane fit) reads its thread count once, at
# import — so the limit must be set before numpy is imported
if a.threads:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(a.threads)

import json
import platform
import time

import cv2
import numpy as np
import onnxruntime as ort

from kitti_loader import KittiRawDrive
from metrics import compute_depth_metrics, valid_mask
from pipeline import onnx_fn, run

KITTI_CAMERA_HEIGHT = 1.65
# (H, W, crop height). Crop width follows from the aspect ratio. KITTI images
# are 375 rows tall, so 518x392 and 532x700 are UPSAMPLED from KITTI.
SIZES = [(294, 392, 375), (518, 392, 375), (532, 700, 375),
         (378, 504, 375), (378, 280, 375),   # diagnostic: the same two crops at ~native size
         (350, 1218, 350)]      # reference: same crop as eval_depth.crop_bottom_centre
REFERENCE = "350x1218"
# Diagnostic rows separate "crop (field of view)" from "resolution" in the
# accuracy numbers; they and the reference are never shipping candidates.
NOT_CANDIDATES = {REFERENCE, "378x504", "378x280"}
STAGES = ("depth", "seg", "geometry", "total")


def prepare(full, K, gt, h, w, ch):
    """Bottom-centre crop to the aspect of (h, w), resize to (h, w); K follows both."""
    cw = round(ch * w / h)
    y0, x0 = full.shape[0] - ch, (full.shape[1] - cw) // 2
    img, gt = full[y0:y0 + ch, x0:x0 + cw], gt[y0:y0 + ch, x0:x0 + cw]
    sx, sy = w / cw, h / ch
    Kr = K.copy()
    Kr[0, 0], Kr[1, 1] = K[0, 0] * sx, K[1, 1] * sy
    Kr[0, 2] = (K[0, 2] - x0 + 0.5) * sx - 0.5        # pixel centres stay aligned
    Kr[1, 2] = (K[1, 2] - y0 + 0.5) * sy - 0.5
    if (h, w) != (ch, cw):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR)
    return img, Kr, gt


def main():
    mode = "space_approx_2threads" if a.threads == 2 else (
        "laptop_unrestricted" if a.threads == 0 else f"{a.threads}_threads")
    so = ort.SessionOptions()
    if a.threads:
        so.intra_op_num_threads, so.inter_op_num_threads = a.threads, 1
        cv2.setNumThreads(a.threads)

    t0 = time.perf_counter()
    fns = []
    for k in ("depth", "segformer"):
        sess = ort.InferenceSession(os.path.join(a.onnx_dir, f"{k}.onnx"), so,
                                    providers=["CPUExecutionProvider"])
        with open(os.path.join(a.onnx_dir, f"{k}.json")) as fh:
            meta = json.load(fh)
        fns.append(onnx_fn(sess, meta["mean"], meta["std"]))
    load_s = time.perf_counter() - t0
    depth_fn, seg_fn = fns

    drive = KittiRawDrive(a.root, "2011_09_26", a.drive, depth_root=a.depth_root)
    gt_dir = os.path.join(a.depth_root, "val", drive.drive_name, "proj_depth", "groundtruth", "image_02")
    gt_frames = sorted(int(f[:-4]) for f in os.listdir(gt_dir))
    frames = [gt_frames[i] for i in np.linspace(0, len(gt_frames) - 1, a.frames).astype(int)]
    raw = [(drive.load_image(f), drive.load_depth_gt(f)) for f in frames]

    speed, accuracy = {}, {}
    for h, w, ch in SIZES:
        name = f"{h}x{w}"
        data = [prepare(img, drive.K, gt, h, w, ch) for img, gt in raw]
        run(data[0][0], data[0][1], KITTI_CAMERA_HEIGHT, depth_fn, seg_fn)   # warm-up, not counted
        times, abs_rel, refused = [], {}, []
        for f, (img, K, gt) in zip(frames, data):
            out = run(img, K, KITTI_CAMERA_HEIGHT, depth_fn, seg_fn, drivable="road+sidewalk")
            if out["status"] != "ok":        # a refusal skips seg + grid: its time would mislead
                refused.append({"frame": f, "reason": out["reason"]})
                continue
            t = out["timings"]
            times.append([t["depth"], t["seg"], t["geometry"], t["depth"] + t["seg"] + t["geometry"]])
            # unscaled prediction at the GT's resolution; the oracle scale (median
            # ratio to the LiDAR) splits the error into "wrong scale s" vs "wrong shape"
            raw_up = cv2.resize(out["depth"] / out["scale"], gt.shape[::-1], interpolation=cv2.INTER_LINEAR)
            m = valid_mask(gt)
            s_oracle = np.median(gt[m]) / np.median(raw_up[m])
            abs_rel[f] = {"height": compute_depth_metrics(raw_up * out["scale"], gt)["abs_rel"],
                          "oracle": compute_depth_metrics(raw_up * s_oracle, gt)["abs_rel"],
                          "s_over_oracle": float(out["scale"] / s_oracle)}
        times = np.array(times)
        speed[name] = {s: {"median_s": float(np.median(times[:, i])),
                           "p90_s": float(np.percentile(times[:, i], 90))} for i, s in enumerate(STAGES)}
        speed[name]["runs"] = len(times)
        v, vo = (np.array([r[k] for r in abs_rel.values()]) for k in ("height", "oracle"))
        accuracy[name] = {"abs_rel_mean": float(v.mean()), "abs_rel_se": float(v.std(ddof=1) / np.sqrt(len(v))),
                          "abs_rel_oracle_mean": float(vo.mean()),
                          "s_over_oracle_median": float(np.median([r["s_over_oracle"] for r in abs_rel.values()])),
                          "candidate": name not in NOT_CANDIDATES,
                          "eval_hw": list(data[0][2].shape), "frames_refused": refused,
                          "per_frame": {str(f): r for f, r in abs_rel.items()}}
        print(f"{name:>9s}: total median {speed[name]['total']['median_s']:.3f} s "
              f"(depth {speed[name]['depth']['median_s']:.3f}, seg {speed[name]['seg']['median_s']:.3f}, "
              f"geometry {speed[name]['geometry']['median_s']:.3f})   AbsRel {v.mean():.4f} +- {accuracy[name]['abs_rel_se']:.4f}"
              f"  oracle {vo.mean():.4f}  s/s_oracle {accuracy[name]['s_over_oracle_median']:.3f}"
              f"   refused {len(refused)}")

    path = os.path.join(a.report_dir, "benchmark.json")
    report = {}
    if os.path.exists(path):
        with open(path) as fh:
            report = json.load(fh)
    report.update(machine={"cpu": platform.processor(), "logical_cores": os.cpu_count(),
                           "os": platform.platform(), "onnxruntime": ort.__version__,
                           "numpy": np.__version__, "opencv": cv2.__version__},
                  setup={"frames": frames, "drive": a.drive, "camera_height_m": KITTI_CAMERA_HEIGHT,
                         "preset": "outdoor", "drivable": "road+sidewalk", "warmup_runs": 1,
                         "stages": "depth / seg include numpy normalisation; geometry = plane fit, "
                                   "back-projection, labels, grid, inflation, A*, photo painting",
                         "accuracy": "depth AbsRel, height mode, vs KITTI LiDAR GT at the crop resolution",
                         "note": "2-thread laptop numbers are optimistic for a free Space; re-measure in M5"},
                  accuracy=accuracy)
    report.setdefault("speed", {})[mode] = {"threads": a.threads or "all", "session_load_s": load_s,
                                            "sizes": speed}

    sp = report["speed"]
    if {"space_approx_2threads", "laptop_unrestricted"} <= sp.keys():
        two, lap = sp["space_approx_2threads"]["sizes"], sp["laptop_unrestricted"]["sizes"]
        report["table"] = [{"size": s, "pixels": int(s.split("x")[0]) * int(s.split("x")[1]),
                            "abs_rel": accuracy[s]["abs_rel_mean"], "abs_rel_se": accuracy[s]["abs_rel_se"],
                            "abs_rel_oracle": accuracy[s]["abs_rel_oracle_mean"],
                            "s_over_oracle": accuracy[s]["s_over_oracle_median"],
                            "total_s_2threads": two[s]["total"]["median_s"],
                            "total_s_laptop": lap[s]["total"]["median_s"],
                            "candidate": accuracy[s]["candidate"]} for s in accuracy]

        # most accurate size within budget; if others are within noise of it
        # (paired difference on the same frames < 2 standard errors), the fastest of those
        fits = [s for s in accuracy if accuracy[s]["candidate"] and two[s]["total"]["median_s"] <= a.budget_s]
        if fits:
            best = min(fits, key=lambda s: accuracy[s]["abs_rel_mean"])
            noise = {}
            for s in fits:
                pf, pb = accuracy[s]["per_frame"], accuracy[best]["per_frame"]
                d = np.array([pf[f]["height"] - pb[f]["height"] for f in pf.keys() & pb.keys()])
                noise[s] = {"paired_diff_mean": float(d.mean()),
                            "paired_diff_se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0}
                noise[s]["within_noise"] = bool(s == best or abs(d.mean()) < 2 * noise[s]["paired_diff_se"])
            tied = [s for s in fits if noise[s]["within_noise"]]
            pick = min(tied, key=lambda s: two[s]["total"]["median_s"])
            why = (f"{best} is the most accurate size within the {a.budget_s} s 2-thread budget"
                   + (f"; {', '.join(t for t in tied if t != best)} within noise of it, so the fastest "
                      f"of those ({pick}) is chosen" if len(tied) > 1 else ""))
        else:
            pick = min((s for s in accuracy if accuracy[s]["candidate"]), key=lambda s: two[s]["total"]["median_s"])
            noise, why = {}, f"no size fits the {a.budget_s} s budget; fastest size chosen"
        report["choice"] = {"size": pick, "budget_s": a.budget_s, "reason": why, "vs_best": noise,
                            "displayed_latency_s": two[pick]["total"]["median_s"],
                            "caveat": "measured on a laptop limited to 2 threads; M5 must re-measure on the Space"}
        print(f"\nchoice: {pick} — {why}")

    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[{mode}] session load {load_s:.2f} s -> {path}")


if __name__ == "__main__":
    main()

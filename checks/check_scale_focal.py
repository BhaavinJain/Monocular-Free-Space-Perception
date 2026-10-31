"""
Is the scale plausibility check fooled by focal length? (M6 investigation)

A metric depth network learns its training camera's focal length: with a shorter
fx (in pixels) things look smaller, so it predicts them farther, by about
fx_train / fx. If so, the camera-height scale s = h / d is expected to be about
fx / fx_train even when everything is right — and the fixed band [0.5, 2.0] would
refuse good photos from short-focal cameras (phones) for the wrong reason.

1. Measure fx_train from data: over the M3 size sweep (10 KITTI frames x 6 input
   sizes, i.e. 6 different focal lengths), the ORACLE scale (LiDAR median /
   prediction median) is what the network is really off by, so
   fx_train = median(fx / s_oracle). Also check s_oracle really follows fx.
2. For every case below, report s and the focal-normalised s / (fx / fx_train):
   all 47 KITTI val frames at 294 x 392, the 6 gallery samples among them, the 6
   no-floor crops (each with its real fx after cropping and resizing), and the
   owner's phone photos (fx from EXIF).
3. The normalised check is adoptable only if every KITTI frame and sample is
   accepted and every no-floor crop refused. Nothing is tuned.

-> report/scale_focal.json
"""
import argparse
import json
import math
import os

import cv2
import numpy as np
import onnxruntime as ort
import pillow_heif
from PIL import Image, ImageOps

from freespace.camera import backproject
from freespace.ground_plane import fit_ground, tilt_deg
from freespace.kitti_loader import KittiRawDrive
from freespace.metrics import valid_mask
from freespace.pipeline import fit_to_size, onnx_fn
from freespace.scale import S_MAX, S_MIN

KITTI_H = 1.65
SWEEP = [(294, 392), (518, 392), (532, 700), (378, 504), (378, 280), (350, 1218)]   # benchmark.py's sizes
CROPS = [(5, 60), (107, 330), (210, 600), (262, 850), (365, 1000), (468, 200)]       # check_api.py's
CROP_H, CROP_W = 150, 200


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--depth_root", default="data/kitti_depth")
    p.add_argument("--onnx_dir", default="checkpoints/onnx")
    p.add_argument("--benchmark", default="report/benchmark.json")
    p.add_argument("--samples", default="backend/samples/samples.json")
    p.add_argument("--test_images", default="backend/test_images")
    p.add_argument("--phone_height", type=float, default=1.1)
    p.add_argument("--report_dir", default="report")
    a = p.parse_args()

    with open(os.path.join(a.onnx_dir, "depth.json")) as fh:
        meta = json.load(fh)
    depth = onnx_fn(ort.InferenceSession(os.path.join(a.onnx_dir, "depth.onnx"), providers=["CPUExecutionProvider"]),
                    meta["mean"], meta["std"])
    drive = KittiRawDrive(a.root, "2011_09_26", "0023", depth_root=a.depth_root)

    def fit(img, K, h):
        """s = h / d from the plane fit, BEFORE the plausibility band (None if tilt/inliers refuse)."""
        pred = depth(img)
        g = fit_ground(backproject(pred, K), pred)
        return pred, (h / g["d"] if g["valid"] else None), g

    def oracle(pred, gt, box):
        y0, x0, ch, cw = box
        gt = gt[y0:y0 + ch, x0:x0 + cw]
        up = cv2.resize(pred, (cw, ch), interpolation=cv2.INTER_LINEAR)
        m = valid_mask(gt)
        return float(np.median(gt[m]) / np.median(up[m]))

    # ---- 1. fx_train from the size sweep ----
    with open(a.benchmark) as fh:
        bench_frames = json.load(fh)["setup"]["frames"]
    sweep = []
    for f in bench_frames:
        full, gt = drive.load_image(f), drive.load_depth_gt(f)
        for h, w in SWEEP:
            img, K, box = fit_to_size(full, drive.K, h, w)
            pred, s, _ = fit(img, K, KITTI_H)
            sweep.append({"frame": f, "size": f"{h}x{w}", "fx": K[0, 0], "s": s, "s_oracle": oracle(pred, gt, box)})
    fx_train = float(np.median([r["fx"] / r["s_oracle"] for r in sweep]))
    by_size = {}
    for h, w in SWEEP:
        rows = [r for r in sweep if r["size"] == f"{h}x{w}"]
        fx = rows[0]["fx"]
        by_size[f"{h}x{w}"] = {"fx": fx, "fx_over_fx_train": fx / fx_train,
                               "s_oracle_median": float(np.median([r["s_oracle"] for r in rows])),
                               "s_median": float(np.median([r["s"] for r in rows if r["s"] is not None]))}
    # how strongly does the oracle scale follow fx? slope of log(s_oracle) on log(fx):
    # 1 = proportional (the fx_train / fx model), 0 = focal length makes no difference
    slope = float(np.polyfit(np.log([r["fx"] for r in sweep]), np.log([r["s_oracle"] for r in sweep]), 1)[0])
    print(f"KITTI calibration fx = {drive.K[0, 0]:.1f} px; measured fx_train = median(fx / s_oracle) = {fx_train:.1f} px")
    print(f"log-log slope of s_oracle on fx: {slope:.2f}  (1 = proportional, 0 = no focal effect)")
    print(f"{'size':>9s} {'fx':>7s} {'fx/fx_tr':>9s} {'s_oracle':>9s} {'s':>7s}")
    for k, v in by_size.items():
        print(f"{k:>9s} {v['fx']:7.1f} {v['fx_over_fx_train']:9.3f} {v['s_oracle_median']:9.3f} {v['s_median']:7.3f}")

    norm = lambda s, fx: None if s is None else s / (fx / fx_train)
    inside = lambda x: x is not None and S_MIN <= x <= S_MAX

    # ---- 2a. all KITTI val frames at the shipping size; the samples are among them ----
    gt_dir = os.path.join(a.depth_root, "val", drive.drive_name, "proj_depth", "groundtruth", "image_02")
    frames = sorted(int(x[:-4]) for x in os.listdir(gt_dir))[::10]
    with open(a.samples) as fh:
        sample_frames = {s["frame"] for s in json.load(fh)["samples"]}
    # the samples come from benchmark.py's frame list, so most are NOT among the
    # every-10th evaluation frames: evaluate them too
    kitti = []
    for f in sorted(set(frames) | sample_frames):
        img, K, _ = fit_to_size(drive.load_image(f), drive.K, 294, 392)
        _, s, _ = fit(img, K, KITTI_H)
        kitti.append({"frame": f, "sample": f in sample_frames, "eval_frame": f in frames,
                      "fx": K[0, 0], "s": s, "normalised": norm(s, K[0, 0])})

    # ---- 2b. the no-floor crops, each with its real fx ----
    crops = []
    for f, x0 in CROPS:
        K = drive.K.copy()
        K[0, 2] -= x0
        img, K_m, _ = fit_to_size(drive.load_image(f)[:CROP_H, x0:x0 + CROP_W], K, 294, 392)
        _, s, g = fit(img, K_m, KITTI_H)
        crops.append({"frame": f, "x0": x0, "fx": K_m[0, 0], "plane_fit": g["reason"],
                      "tilt_deg": None if g["n"] is None else tilt_deg(g["n"]), "s": s, "normalised": norm(s, K_m[0, 0])})

    # ---- 2c. the owner's phone photos (fx from EXIF, as the server computes it) ----
    pillow_heif.register_heif_opener()
    phones = []
    for fn in sorted(os.listdir(a.test_images)) if os.path.isdir(a.test_images) else []:
        if os.path.splitext(fn)[1].lower() not in (".jpg", ".jpeg", ".heic", ".heif"):
            continue
        im = Image.open(os.path.join(a.test_images, fn))
        f35 = im.getexif().get_ifd(0x8769).get(0xA405)
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        H, W = rgb.shape[:2]
        fx = f35 * math.hypot(W, H) / 43.27
        K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1.0]])
        h, w = (294, 392) if W >= H else (392, 294)
        img, K_m, _ = fit_to_size(rgb, K, h, w)
        _, s, _ = fit(img, K_m, a.phone_height)
        phones.append({"file": fn, "fx": K_m[0, 0], "s": s, "normalised": norm(s, K_m[0, 0])})

    # ---- 3. would the focal-aware check be adoptable? ----
    crit = {"kitti_all_accepted_now": all(inside(r["s"]) for r in kitti),
            "kitti_all_accepted_normalised": all(inside(r["normalised"]) for r in kitti),
            "samples_all_accepted_normalised": all(inside(r["normalised"]) for r in kitti if r["sample"]),
            "crops_all_refused_now": not any(inside(r["s"]) for r in crops),
            "crops_all_refused_normalised": not any(inside(r["normalised"]) for r in crops)}
    crit["adoptable"] = (crit["kitti_all_accepted_normalised"] and crit["samples_all_accepted_normalised"]
                         and crit["crops_all_refused_normalised"])

    ev = [r for r in kitti if r["eval_frame"]]
    print(f"\n{len(ev)} KITTI val frames at 294x392 (fx {ev[0]['fx']:.1f}): s {min(r['s'] for r in ev):.3f}-{max(r['s'] for r in ev):.3f}, "
          f"normalised {min(r['normalised'] for r in ev):.3f}-{max(r['normalised'] for r in ev):.3f}")
    print("samples:", ", ".join(f"{r['frame']}: s {r['s']:.3f} / norm {r['normalised']:.3f}" for r in kitti if r["sample"]))
    print("\nno-floor crops:")
    for r in crops:
        s = "-" if r["s"] is None else f"{r['s']:.3f}"
        n = "-" if r["normalised"] is None else f"{r['normalised']:.3f}"
        print(f"  frame {r['frame']:3d} x0 {r['x0']:4d}: fx {r['fx']:7.1f}  plane fit: {r['plane_fit'][:40]:40s}  s {s:>6s}  normalised {n:>6s}"
              f"  -> now {'ACCEPTED' if inside(r['s']) else 'refused'}, normalised {'ACCEPTED' if inside(r['normalised']) else 'refused'}")
    print("\nphone photos:")
    for r in phones:
        print(f"  {r['file']}: fx {r['fx']:.1f}  s {r['s']:.3f}  normalised {r['normalised']:.3f}"
              f"  -> now {'ACCEPTED' if inside(r['s']) else 'refused'}, normalised {'ACCEPTED' if inside(r['normalised']) else 'refused'}")
    print("\ncriteria:", crit)

    report = {"band": [S_MIN, S_MAX], "kitti_calibration_fx": drive.K[0, 0], "fx_train_measured": fx_train,
              "fx_train_method": "median over the M3 size sweep (10 frames x 6 sizes) of fx / s_oracle",
              "loglog_slope_s_oracle_on_fx": slope,
              "sweep_by_size": by_size, "sweep": sweep, "kitti_294x392": kitti, "crops": crops, "phones": phones,
              "criteria": crit}
    with open(os.path.join(a.report_dir, "scale_focal.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"-> {os.path.join(a.report_dir, 'scale_focal.json')}")


if __name__ == "__main__":
    main()

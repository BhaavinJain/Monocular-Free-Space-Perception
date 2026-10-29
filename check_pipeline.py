"""
M1 check: pipeline.run() must build EXACTLY the grid eval_freespace.py builds.

One KITTI frame, cropped as in eval_depth, KITTI K, h = 1.65, PyTorch models.
The reference below is a verbatim copy of eval_freespace.py's per-frame steps.
Both paths get the SAME cached network outputs, so a difference can only come
from the geometry code, not from GPU run-to-run noise.
"""
import argparse
import json
import os

import cv2
import numpy as np

from bev import build_grid, inflate
from camera import backproject, transform_points
from depth_model import DepthModel
from eval_depth import crop_bottom_centre
from eval_freespace import CELL, KITTI_CAMERA_HEIGHT, ROBOT_RADIUS, X_MAX, Y_HALF
from kitti_loader import KittiRawDrive
from pipeline import DRIVABLE_IDS, UPSAMPLE, fuse_labels, run, upsample_rows
from planner import astar, pick_start_goal
from pointcloud import classify_heights, floor_frame
from scale import estimate_metric_scale
from segmenter import Segmenter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--drive", default="0023")
    p.add_argument("--frame", type=int, default=10)
    p.add_argument("--weights", default="checkpoints/depth_best.pt")
    p.add_argument("--report_dir", default="report")
    a = p.parse_args()

    drive = KittiRawDrive(a.root, "2011_09_26", a.drive)
    img, K = crop_bottom_centre(drive.load_image(a.frame), drive.K)
    H, W = img.shape[:2]
    model, seg = DepthModel(weights=a.weights), Segmenter()
    pred, sem = model.predict(img), seg.predict(img)
    depth_fn, seg_fn = (lambda im: pred), (lambda im: sem)

    results = {"drive": a.drive, "frame": a.frame, "image_hw": [H, W]}
    for drivable in ("none", "road+sidewalk"):
        ids = DRIVABLE_IDS[drivable]
        if ids is not None:
            assert ids == seg.ids(drivable.split("+")), "pipeline's Cityscapes ids are wrong"

        # ---- reference: eval_freespace.py, lines for one frame ----
        fit = estimate_metric_scale(pred, K, KITTI_CAMERA_HEIGHT)
        depth, T = pred * fit["scale"], floor_frame(fit["n"], KITTI_CAMERA_HEIGHT)
        depth_k, K_k = upsample_rows(depth, K)
        pts = transform_points(backproject(depth_k, K_k).reshape(-1, 3), T)
        labels = classify_heights(pts)
        if ids is not None:
            sem_k = cv2.resize(sem, (W, UPSAMPLE * H), interpolation=cv2.INTER_NEAREST)
            labels = fuse_labels(labels, sem_k.reshape(-1), ids)
        ref_grid = build_grid(pts, labels, CELL, X_MAX, Y_HALF)
        ref_inf = inflate(ref_grid, ROBOT_RADIUS, CELL)
        start, goal = pick_start_goal(ref_inf)
        ref_path = astar(ref_inf, start, goal) if start is not None else None

        # ---- candidate: pipeline.run ----
        out = run(img, K, KITTI_CAMERA_HEIGHT, depth_fn, seg_fn, drivable=drivable, preset="outdoor")
        r = {"status": out["status"],
             "scale_equal": bool(out["scale"] == fit["scale"]),
             "grid_equal": bool(np.array_equal(out["grid"], ref_grid)),
             "inflated_equal": bool(np.array_equal(out["grid_inflated"], ref_inf)),
             "path_equal": out["path_cells"] == ref_path,
             "scale": out["scale"], "tilt_deg": out["tilt_deg"],
             "path_cells": None if ref_path is None else len(ref_path),
             "timings_s": out["timings"]}
        r["passed"] = all(r[k] for k in ("scale_equal", "grid_equal", "inflated_equal", "path_equal"))
        results[drivable] = r
        print(drivable, {k: r[k] for k in ("scale_equal", "grid_equal", "inflated_equal",
                                           "path_equal", "passed")})

        os.makedirs(os.path.join(a.report_dir, "image"), exist_ok=True)
        path = os.path.join(a.report_dir, "image", f"m1_pipeline_frame{a.frame}_{drivable.replace('+', '_')}.png")
        cv2.imwrite(path, cv2.cvtColor(out["painted"], cv2.COLOR_RGB2BGR))

    results["passed"] = all(results[d]["passed"] for d in ("none", "road+sidewalk"))
    path = os.path.join(a.report_dir, "m1_check.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nM1 {'PASSED' if results['passed'] else 'FAILED'} -> {path}")


if __name__ == "__main__":
    main()

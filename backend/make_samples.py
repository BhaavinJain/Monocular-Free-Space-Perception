"""
Build the gallery the server shows by default: a few KITTI val frames whose
results are backed by measurements. Dev-only (needs the KITTI data); the
output folder is git-ignored and goes to the Hugging Face Hub in M5.

Per frame, in backend/samples/<id>/:
  image.png       the 375 x 500 bottom-centre 4:3 crop — the server resizes it to
                  294 x 392 with pipeline.fit_to_size, which reproduces exactly the
                  input benchmark.py measured
  gt_depth.png    KITTI depth ground truth for the same crop (16-bit, metres * 256)
  lidar_grid.png  LiDAR occupancy grid (0 free, 1 occupied, 2 unknown), built as in
                  eval_freespace.py: LiDAR's own floor plane, restricted to what the
                  camera sees at 294 x 392
plus backend/samples/samples.json (K of the crop, camera height, source frame).

The frames are taken from benchmark.py's frame list, so each sample's depth
AbsRel can be checked against report/benchmark.json.
"""
import argparse
import json
import os

import cv2
import numpy as np

from bev import lidar_to_bev
from camera import backproject, lidar_to_depth_map
from ground_plane import fit_ground
from kitti_loader import KittiRawDrive
from pipeline import PRESETS, fit_to_size
from pointcloud import floor_frame

KITTI_CAMERA_HEIGHT = 1.65
MODEL_HW = (294, 392)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--depth_root", default="data/kitti_depth")
    p.add_argument("--drive", default="0023")
    p.add_argument("--count", type=int, default=6)
    p.add_argument("--out_dir", default="backend/samples")
    p.add_argument("--benchmark", default="report/benchmark.json")
    a = p.parse_args()

    with open(a.benchmark) as fh:
        bench_frames = json.load(fh)["setup"]["frames"]
    frames = [bench_frames[i] for i in np.linspace(0, len(bench_frames) - 1, a.count).round().astype(int)]
    drive = KittiRawDrive(a.root, "2011_09_26", a.drive, depth_root=a.depth_root)
    P = PRESETS["outdoor"]
    samples = []
    for f in frames:
        full, gt = drive.load_image(f), drive.load_depth_gt(f)
        H, W = full.shape[:2]
        cw = round(H * MODEL_HW[1] / MODEL_HW[0])               # 4:3 at full height
        x0 = (W - cw) // 2
        img, gt_crop = full[:, x0:x0 + cw], gt[:, x0:x0 + cw]
        K = drive.K.copy()
        K[0, 2] -= x0

        scan = drive.load_velodyne(f)
        lid = lidar_to_depth_map(scan, drive.K, drive.T_cam_velo, H, W)
        gfit = fit_ground(backproject(lid, drive.K), lid)
        if not gfit["valid"]:
            print(f"  frame {f}: LiDAR ground fit refused ({gfit['reason']}), skipped")
            continue
        _, K_model, _ = fit_to_size(img, K, *MODEL_HW)
        grid = lidar_to_bev(scan, drive.T_cam_velo, floor_frame(gfit["n"], gfit["d"]),
                            K_model, *MODEL_HW, P["cell"], P["x_max"], P["y_half"])

        sid = f"{a.drive}_{f:06d}"
        d = os.path.join(a.out_dir, sid)
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, "image.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(d, "gt_depth.png"), np.round(gt_crop * 256).astype(np.uint16))
        cv2.imwrite(os.path.join(d, "lidar_grid.png"), grid)
        samples.append({"id": sid, "drive": f"2011_09_26_drive_{a.drive}_sync", "frame": f,
                        "K": K.tolist(), "camera_height_m": KITTI_CAMERA_HEIGHT,
                        "crop": {"x0": x0, "y0": 0, "w": cw, "h": H}})
        print(f"  {sid}: image {img.shape[:2]}, LiDAR grid {grid.shape}")

    with open(os.path.join(a.out_dir, "samples.json"), "w") as fh:
        json.dump({"source": "KITTI raw (Geiger et al. 2013), CC BY-NC-SA 3.0",
                   "samples": samples}, fh, indent=2)
    print(f"{len(samples)} samples -> {a.out_dir}")


if __name__ == "__main__":
    main()

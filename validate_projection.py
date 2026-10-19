"""
Three independent checks. If all pass, your geometry is correct and every
later phase can trust it.
 
  CHECK 1 (visual)   : project LiDAR onto the RGB image. Points must sit on
                       cars, poles, road — not shifted or skewed.
  CHECK 2 (math)     : back-project a depth map, then project it again.
                       Every point must land back on its own pixel.
                       Proves project() and backproject() are exact inverses.
  CHECK 3 (vs KITTI) : compare YOUR LiDAR-projected depth against KITTI's
                       official depth ground truth. Only runs if you pass
                       --depth_root. Proves your calibration chain matches
                       the one the KITTI authors used.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")            # render to file, no window needed
import matplotlib.pyplot as plt
import contextlib
import os

from camera import (transform_points, project, in_image_mask, backproject, lidar_to_depth_map)

from kitti_loader import KittiRawDrive

def check_1_overlay(drive, frame, out_path):
    img = drive.load_image(frame)
    H,W = img.shape[:2]
    scan = drive.load_velodyne(frame)

    # Step 1: LiDAR frame -> camera frame  (drop reflectance column)
    pts_cam = transform_points(scan[:,:3], drive.T_cam_velo)

    # Step 2: camera frame -> pixels
    uv, z = project(pts_cam, drive.K)

    # Step 3: throw away points behind the camera or outside the image
    keep = in_image_mask(uv, z, H, W)
    uv, z = uv[keep], z[keep]

    print(f"[check 1] {scan.shape[0]} LiDAR points, "
          f"{keep.sum()} land in the image "
          f"({100 * keep.mean():.1f}%) — expect roughly 15-25%, "
          f"since the LiDAR sees 360 deg and the camera ~90 deg")

    # Step 4: draw, coloured by depth (near = red/yellow, far = blue)
    fig = plt.figure(figsize = (W/100, H/100), dpi = 100)
    ax = fig.add_axes([0,0,1,1])
    ax.imshow(img)
    ax.scatter(uv[:,0], uv[:,1], c = z, s = 1.5, cmap = "turbo_r", vmin = 0, vmax = 50)
    ax.set_xlim(0,W); ax.set_ylim(H,0)                             # image coords: v grows down
    ax.axis("off")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"[check 1] saved overlay -> {out_path}")

def check_2_roundtrip(drive, frame):
    img =drive.load_image(frame)
    H, W = img.shape[:2]
    depth = lidar_to_depth_map(drive.load_velodyne(frame), drive.K, drive.T_cam_velo, H , W)
    valid = depth > 0                           # (H, W) array of True/False: True where the depth map has a value

    # pixel -> 3D
    cloud = backproject(depth, drive.K)         # (H, W, 3)
    pts = cloud[valid]                          # (M, 3)

    # 3D -> pixel again
    uv, _=  project(pts, drive.K)

    # where each point SHOULD land : its original pixel coordinates
    v_true, u_true = np.nonzero(valid)        
    err = np.hypot(uv[:, 0] - u_true, uv[:, 1] - v_true) # for every point, how far it landed : horizontally and vertically from its original pixel coordinates
    print(f"[check 2] round-trip on {valid.sum()} pixels: "
          f"max error = {err.max():.2e} px")
    ok = err.max() < 1e-6
    print("[check 2] PASS" if ok else "[check 2] FAIL — project/backproject "
          "are not inverses (check fx/fy/cx/cy indexing)")
    return ok

def check_3_vs_kitti_gt(drive, frame):                         # The comparision against KITTI
    gt = drive.load_depth_gt(frame)
    if gt is None:
        print("[check 3] skipped (no --depth_root, or no GT for this frame; "
              "GT starts at frame 5)")
        return None
    # building out own depth map
    H,W = gt.shape
    ours = lidar_to_depth_map(drive.load_velodyne(frame), drive.K, drive.T_cam_velo, H, W)
    both = (ours > 0) & (gt > 0)          # where both depth maps have a value
    diff = np.abs(ours[both] - gt[both])
    print(f"[check 3] {both.sum()} pixels where both have depth")
    print(f"[check 3] median |ours - GT| = {np.median(diff):.3f} m, "
          f"{100 * np.mean(diff < 0.5):.1f}% within 0.5 m")
    return np.median(diff)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data",
                   help="folder containing the day folders")
    p.add_argument("--date", default="2011_09_26")
    p.add_argument("--drive", default="23")
    p.add_argument("--frame", type=int, default=10)
    p.add_argument("--depth_root", default=None)
    p.add_argument("--report_dir", default="report")
    a = p.parse_args()

    tag = f"phase1_frame{a.frame:03d}"
    txt_path = os.path.join(a.report_dir, f"validation_results_{a.frame:03d}.txt")
    img_path = os.path.join(a.report_dir, "image", f"check1_overlay_{a.frame:03d}.png")
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
 
    with open(txt_path, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        drive = KittiRawDrive(a.root, a.date, a.drive, depth_root=a.depth_root)
        print("calibration read from:", drive.calib_dir)
        print("K =\n", np.round(drive.K, 3))
        print("T_cam_velo =\n", np.round(drive.T_cam_velo, 4))
 
        check_1_overlay(drive, a.frame, img_path)
        check_2_roundtrip(drive, a.frame)
        check_3_vs_kitti_gt(drive, a.frame)
 
    # Then show the saved results on screen too.
    print(open(txt_path, encoding="utf-8").read())
 
 
if __name__ == "__main__":
    main()
    
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
import matplotlib as plt

from camera import (transform_points, project, in_image_mask, backproject, lidar_to_depth_map)

from Kitti_loader import KittiRawDrive

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

def check_3_vs_kitti_gt(drive, frame):
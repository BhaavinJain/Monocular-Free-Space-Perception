import argparse
import contextlib
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from freespace.camera import backproject, lidar_to_depth_map, project
from freespace.ground_plane import fit_ground, lower_region_points, tilt_deg
from freespace.kitti_loader import KittiRawDrive
 
KITTI_CAMERA_HEIGHT = 1.65
 
 
def open3d_plane(points, rel_thresh=0.01):
    """Open3D's RANSAC on the same points, converted to our (n, d>0) convention."""
    try:
        import open3d as o3d
    except ImportError:
        return None
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    thresh = rel_thresh * np.median(np.linalg.norm(points, axis=1))
    (a, b, c, d), _ = pcd.segment_plane(distance_threshold=thresh,
                                        ransac_n=3, num_iterations=200)
    n = np.array([a, b, c]); norm = np.linalg.norm(n)
    n, d = n / norm, d / norm
    if d < 0:
        n, d = -n, -d
    return n, d
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--date", default="2011_09_26")
    p.add_argument("--drive", default="23")
    p.add_argument("--frames", type=int, nargs="+", default=[10, 50, 100])
    p.add_argument("--report_dir", default="report")
    a = p.parse_args()
 
    os.makedirs(os.path.join(a.report_dir, "image"), exist_ok=True)
    txt_path = os.path.join(a.report_dir, "check_ground_plane.txt")
 
    # Every print() inside this block goes into the .txt file.
    with open(txt_path, "w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        drive = KittiRawDrive(a.root, a.date, a.drive)
        print(f"expected camera height: {KITTI_CAMERA_HEIGHT} m\n")
 
        for f in a.frames:
            img = drive.load_image(f)
            H, W = img.shape[:2]
            depth = lidar_to_depth_map(drive.load_velodyne(f), drive.K,
                                       drive.T_cam_velo, H, W)
            cloud = backproject(depth, drive.K)
 
            fit = fit_ground(cloud, depth)
            print(f"frame {f}: valid={fit['valid']} ({fit['reason']})")
            if fit["n"] is None:
                continue
            print(f"   ours    : height d = {fit['d']:.3f} m, "
                  f"tilt = {tilt_deg(fit['n']):.2f} deg, "
                  f"inliers = {100 * fit['inlier_ratio']:.0f}% of {fit['n_points']} pts")
 
            pts = lower_region_points(cloud, depth)   # same points the fit used
            ref = open3d_plane(pts)
            planes = [("ours", fit["n"], fit["d"])]
            if ref is None:
                print("   open3d  : not installed (pip install open3d) — skipped")
            else:
                n_ref, d_ref = ref
                angle = np.degrees(np.arccos(np.clip(abs(n_ref @ fit["n"]), -1, 1)))
                print(f"   open3d  : height d = {d_ref:.3f} m, "
                      f"normal differs from ours by {angle:.2f} deg "
                      f"(tilt from up: {tilt_deg(n_ref):.1f} deg)")
                planes.append(("open3d", n_ref, d_ref))
 
            # Picture: each point drawn where it sits in the image. Both planes
            # are judged with the SAME tolerance, so any difference between the
            # panels comes from the planes, not from different thresholds.
            uv, _ = project(pts, drive.K)
            thresh = 0.02 * np.median(np.linalg.norm(pts, axis=1))
            fig, axes = plt.subplots(len(planes), 1, squeeze=False,
                                     figsize=(W / 100, len(planes) * H / 100 + 0.6))
            for ax, (name, n, d) in zip(axes[:, 0], planes):
                on = np.abs(pts @ n + d) < thresh
                ax.imshow(img)
                ax.scatter(uv[~on, 0], uv[~on, 1], s=1, color="red")
                ax.scatter(uv[on, 0], uv[on, 1], s=1, color="lime")
                ax.set_title(f"{name}: d = {d:.3f} m, {100 * on.mean():.0f}% on plane")
                ax.axis("off")
            fig.tight_layout()
            img_path = os.path.join(a.report_dir, "image", f"ground_plane_{f:03d}.png")
            fig.savefig(img_path, dpi=100)
            plt.close(fig)
            print(f"   picture -> {img_path}")
 
    print(open(txt_path, encoding="utf-8").read())
 
 
if __name__ == "__main__":
    main()
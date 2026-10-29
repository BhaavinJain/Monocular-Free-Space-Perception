import argparse
import json
import os
 
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from matplotlib.patches import Circle, Patch
 
from bev import FREE, OCCUPIED, UNKNOWN, build_grid, cell_centres, cell_index, lidar_to_bev
from camera import backproject, lidar_to_depth_map, transform_points
from depth_model import DepthModel, METRIC_CKPT
from eval_depth import crop_bottom_centre, val_drives
from ground_plane import fit_ground
from kitti_loader import KittiRawDrive
from pointcloud import IGNORE, classify_heights, floor_frame
from scale import estimate_metric_scale
 
KITTI_CAMERA_HEIGHT = 1.65
# Grid for KITTI: the camera sees no road closer than ~5.9 m, so the
# blueprint's indoor 10 m x 10 m grid would be mostly blind spot here.
CELL, X_MAX, Y_HALF = 0.10, 20.0, 10.0          # metres
BANDS = ((0, 10), (10, 15), (15, 20))           # distance from the robot, metres
UPSAMPLE = 4    # rows per image row before back-projection (see upsample_rows)
 
 
def upsample_rows(depth, K, k=UPSAMPLE):
    """
    One image row covers a strip of road about distance^2 / (fy * h) long:
    8 cm at 10 m but 34 cm at 20 m. With 10 cm cells, far rows leave gaps of
    UNKNOWN cells between them even when the depth is perfect. Interpolating
    k rows between each pair of real rows (and scaling K to match) puts the
    points ~8 cm apart at 20 m.
 
    K must change because the new rows are k times denser: fy -> k*fy, and
    cy moves so that pixel CENTRES still line up (new row v' sits at old
    row (v' + 0.5)/k - 0.5).
    """
    H, W = depth.shape
    Kk = K.copy()
    Kk[1, 1] = k * K[1, 1]
    Kk[1, 2] = k * (K[1, 2] + 0.5) - 0.5
    return cv2.resize(depth, (W, k * H), interpolation=cv2.INTER_LINEAR), Kk
 
 
def counts(pred, gt, region):
    """Cell counts needed for the three metrics, inside `region`."""
    pf, gf, go = (pred == FREE) & region, (gt == FREE) & region, (gt == OCCUPIED) & region
    return np.array([(pf & gf).sum(), (pf | gf).sum(), (pf & go).sum(), pf.sum(), go.sum()])
 
 
def ratios(c):
    inter, union, false_free, pred_free, gt_occ = c
    return {"free_iou": inter / max(union, 1),
            "false_free_rate": false_free / max(pred_free, 1),
            "missed_obstacle_rate": false_free / max(gt_occ, 1),
            "cells_evaluated": int(union + gt_occ - false_free)}
 
 
GREEN, RED, GREY, MAGENTA = (0.2, 0.8, 0.2), (0.9, 0.1, 0.1), (0.8, 0.8, 0.8), (1.0, 0.0, 1.0)
 
 
def colour(grid, false_free=None):
    rgb = np.array([GREEN, RED, GREY])[grid]          # indexed by FREE, OCCUPIED, UNKNOWN
    if false_free is not None:
        rgb[false_free] = MAGENTA
    return rgb
 
 
def paint_photo(img, pts, labels, pred_grid, gt_grid, alpha=0.45):
    """
    Tint every pixel by the state of the grid cell its OWN 3D point falls in:
    road pixels show where the grid says free, car pixels where it says
    occupied, and magenta marks pixels whose cell was called free although
    the LiDAR says occupied. Pixels whose point is ignored (sky, treetops,
    anything above 2 m) or outside the grid keep their original colour.
    """
    ix, iy, ok = cell_index(pts, CELL, X_MAX, Y_HALF)
    ok &= labels != IGNORE
    p = np.full(len(pts), -1); g = np.full(len(pts), -1)
    p[ok], g[ok] = pred_grid[ix[ok], iy[ok]], gt_grid[ix[ok], iy[ok]]
    tint, use = np.zeros((len(pts), 3)), np.zeros(len(pts), bool)
    for mask, rgb in ((p == FREE, GREEN), (p == OCCUPIED, RED),
                      ((p == FREE) & (g == OCCUPIED), MAGENTA)):
        tint[mask], use = rgb, use | mask
    out = img.reshape(-1, 3) / 255.0
    out[use] = (1 - alpha) * out[use] + alpha * tint[use]
    return out.reshape(img.shape)
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--depth_root", default="data/kitti_depth")
    p.add_argument("--checkpoint", default=METRIC_CKPT)
    p.add_argument("--weights", default=None, help="fine-tuned .pt (optional)")
    p.add_argument("--scale", default="height", choices=["height", "raw"])
    p.add_argument("--frame_step", type=int, default=10)
    p.add_argument("--tag", default="pretrained")
    p.add_argument("--report_dir", default="report")
    p.add_argument("--images", type=int, default=3)
    a = p.parse_args()
 
    model = DepthModel(a.checkpoint, weights=a.weights)
    drives, _ = val_drives(a.root, a.depth_root)
    xs, ys = cell_centres(CELL, X_MAX, Y_HALF)
    dist = np.hypot(xs, ys)
    band_masks = {"all": np.ones_like(dist, bool)}
    band_masks.update({f"{lo}-{hi}m": (dist >= lo) & (dist < hi) for lo, hi in BANDS})
    totals = {k: np.zeros(5, int) for k in band_masks}
    refused, worst = [], []
 
    for date, num, name in drives:
        drive = KittiRawDrive(a.root, date, num, depth_root=a.depth_root)
        gt_dir = os.path.join(a.depth_root, "val", name, "proj_depth", "groundtruth", "image_02")
        frames = sorted(int(f[:-4]) for f in os.listdir(gt_dir))[::a.frame_step]
 
        for f in frames:
            full = drive.load_image(f)
            img, K = crop_bottom_centre(full, drive.K)
            H, W = img.shape[:2]
 
            # ---- predicted grid: camera only ----
            pred = model.predict(img)
            fit = estimate_metric_scale(pred, K, KITTI_CAMERA_HEIGHT)
            if not fit["valid"]:
                refused.append((name, f, fit["reason"]))     # a real system outputs
                continue                                     # nothing metric here
            if a.scale == "height":
                depth, T = pred * fit["scale"], floor_frame(fit["n"], KITTI_CAMERA_HEIGHT)
            else:                                            # network's own metres
                depth, T = pred, floor_frame(fit["n"], fit["d"])
            depth_k, K_k = upsample_rows(depth, K)
            pts = transform_points(backproject(depth_k, K_k).reshape(-1, 3), T)
            pred_grid = build_grid(pts, classify_heights(pts), CELL, X_MAX, Y_HALF)
 
            # ---- ground-truth grid: LiDAR, with its own floor plane ----
            scan = drive.load_velodyne(f)
            lid = lidar_to_depth_map(scan, drive.K, drive.T_cam_velo, *full.shape[:2])
            gfit = fit_ground(backproject(lid, drive.K), lid)
            if not gfit["valid"]:
                refused.append((name, f, "LiDAR ground fit: " + gfit["reason"]))
                continue
            gt_grid = lidar_to_bev(scan, drive.T_cam_velo, floor_frame(gfit["n"], gfit["d"]),
                                   K, H, W, CELL, X_MAX, Y_HALF)
 
            known = gt_grid != UNKNOWN
            for k, m in band_masks.items():
                totals[k] += counts(pred_grid, gt_grid, known & m)
 
            n_ff = int(((pred_grid == FREE) & (gt_grid == OCCUPIED)).sum())
            if len(worst) < a.images or n_ff > worst[-1][0]:
                # the photo overlay uses one 3D point per original pixel
                px = transform_points(backproject(depth, K).reshape(-1, 3), T)
                painted = paint_photo(img, px, classify_heights(px), pred_grid, gt_grid)
                worst.append((n_ff, name, f, painted, pred_grid, gt_grid))
                worst.sort(key=lambda w: -w[0])
                del worst[a.images:]
        print(f"  {name}: {len(frames)} frames done")
 
    # ---------------- report ----------------
    report = {"tag": a.tag, "weights": a.weights, "scale": a.scale,
              "cell_m": CELL, "x_max_m": X_MAX, "y_half_m": Y_HALF,
              "frames_refused": len(refused), "refused": refused[:20],
              "metrics": {k: ratios(c) for k, c in totals.items()}}
    print(f"\nframes refused: {len(refused)}")
    print(f"\n{'band':>8s}  {'free IoU':>9s}  {'false-free':>10s}  {'missed obst.':>12s}  {'cells':>9s}")
    for k, m in report["metrics"].items():
        print(f"{k:>8s}  {m['free_iou']:9.3f}  {m['false_free_rate']:10.3f}  "
              f"{m['missed_obstacle_rate']:12.3f}  {m['cells_evaluated']:9d}")
 
    os.makedirs(os.path.join(a.report_dir, "image"), exist_ok=True)
    print(f"\nframes with the most false-free cells:")
    legend = [Patch(color=GREEN, label="free"), Patch(color=RED, label="occupied"),
              Patch(color=GREY, label="unknown (not seen)"),
              Patch(color=MAGENTA, label="called free, LiDAR says occupied")]
    for n_ff, name, f, painted, pg, gg in worst:
        fig = plt.figure(figsize=(12, 10))
        ax0 = fig.add_subplot(2, 1, 1); ax0.imshow(painted); ax0.axis("off")
        ax0.set_title(f"{name} frame {f}: what the camera-only grid decided, painted on the photo "
                      f"({n_ff} false-free cells)", fontsize=10)
        for i, (g, title, ff) in enumerate([
                (pg, "predicted grid (camera only), seen from above", (pg == FREE) & (gg == OCCUPIED)),
                (gg, "LiDAR ground truth, seen from above", None)]):
            ax = fig.add_subplot(2, 2, 3 + i)
            ax.imshow(colour(g, ff), origin="lower", extent=(-Y_HALF, Y_HALF, 0, X_MAX))
            for r in (5, 10, 15, 20):                      # distance rings around the camera
                ax.add_patch(Circle((0, 0), r, fill=False, ls="--", lw=0.6, alpha=0.6))
                ax.text(0.3, r - 0.9, f"{r} m", fontsize=7, alpha=0.8)
            ax.plot(0, 0.3, marker="^", markersize=12, color="black")   # the camera
            ax.set_xlim(-Y_HALF, Y_HALF); ax.set_ylim(0, X_MAX)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("sideways (m), left negative"); ax.set_ylabel("forward (m)")
        fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        path = os.path.join(a.report_dir, "image", f"bev_{a.tag}_{name[17:21]}_{f:03d}.png")
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(f"  frame {f}: {n_ff} false-free cells -> {path}")
 
    path = os.path.join(a.report_dir, f"freespace_{a.tag}.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=int)
    print(f"\nreport -> {path}")
 
 
if __name__ == "__main__":
    main()
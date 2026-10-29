"""
The whole camera-only pipeline for ONE image, with no KITTI and no LiDAR:

    image -> depth_fn -> ground plane -> scale s = h / d -> floor-frame points
          -> height labels (+ semantic fusion) -> BEV grid -> inflate -> A* path

The networks are passed in as plain functions (depth_fn, seg_fn), so the same
code runs with PyTorch models (evaluation) or ONNX Runtime (the server). This
file must never import torch: the server imports it.

If the ground-plane fit is refused, nothing metric is returned — only the
reason, and the network's raw (unscaled) depth so the user can see why.
"""
import time

import cv2
import numpy as np

from bev import FREE, OCCUPIED, build_grid, cell_index, inflate
from camera import backproject, project, transform_points
from ground_plane import tilt_deg
from planner import astar, pick_start_goal
from pointcloud import GROUND, IGNORE, OBSTACLE, classify_heights, floor_frame
from scale import estimate_metric_scale

UPSAMPLE = 4    # rows per image row before back-projection (see upsample_rows)
# outdoor = the KITTI setting the Phase 3-4 numbers were measured with.
# indoor defaults to no semantics: SegFormer only knows Cityscapes street
# classes and has not been tested indoors.
PRESETS = {
    "outdoor": {"cell": 0.10, "x_max": 20.0, "y_half": 10.0, "robot_radius": 1.0,
                "drivable": "road+sidewalk"},
    "indoor": {"cell": 0.05, "x_max": 10.0, "y_half": 5.0, "robot_radius": 0.3,
               "drivable": "none"},
}
# Cityscapes train ids (road = 0, sidewalk = 1). Fixed here because the
# server has no Segmenter to look the names up; check_pipeline.py asserts them.
DRIVABLE_IDS = {"none": None, "road": [0], "road+sidewalk": [0, 1]}

GREEN, RED, GREY, MAGENTA = (0.2, 0.8, 0.2), (0.9, 0.1, 0.1), (0.8, 0.8, 0.8), (1.0, 0.0, 1.0)


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


def fuse_labels(labels, semantic, drivable_ids):
    """
    labels   : (N,) height labels from classify_heights
    semantic : (N,) class id of the pixel each point came from
    Flat points whose class is not drivable become obstacles.
    """
    out = labels.copy()
    out[(labels == GROUND) & ~np.isin(semantic, drivable_ids)] = OBSTACLE
    return out


def paint_photo(img, pts, labels, pred_grid, cell, x_max, y_half, gt_grid=None, alpha=0.45):
    """
    Tint every pixel by the state of the grid cell its OWN 3D point falls in:
    road pixels show where the grid says free, car pixels where it says
    occupied. With a LiDAR gt_grid, magenta marks pixels whose cell was called
    free although the LiDAR says occupied. Pixels whose point is ignored (sky,
    treetops, anything above 2 m) or outside the grid keep their original colour.
    """
    ix, iy, ok = cell_index(pts, cell, x_max, y_half)
    ok &= labels != IGNORE
    p = np.full(len(pts), -1); g = np.full(len(pts), -1)
    p[ok] = pred_grid[ix[ok], iy[ok]]
    if gt_grid is not None:
        g[ok] = gt_grid[ix[ok], iy[ok]]
    tint, use = np.zeros((len(pts), 3)), np.zeros(len(pts), bool)
    for mask, rgb in ((p == FREE, GREEN), (p == OCCUPIED, RED),
                      ((p == FREE) & (g == OCCUPIED), MAGENTA)):
        tint[mask], use = rgb, use | mask
    out = img.reshape(-1, 3) / 255.0
    out[use] = (1 - alpha) * out[use] + alpha * tint[use]
    return out.reshape(img.shape)


def onnx_fn(sess, mean, std):
    """
    Wrap an ONNX Runtime session as a depth_fn / seg_fn: uint8 RGB (H, W, 3)
    -> the model's (H, W) output. The normalisation lives here, in numpy, with
    the mean/std saved next to each model by export_onnx.py.
    """
    mean, std = np.reshape(mean, (1, 1, 3)), np.reshape(std, (1, 1, 3))
    def fn(img):
        x = ((img / 255.0 - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
        return sess.run(None, {"pixel_values": x})[0][0]
    return fn


def run(image, K, camera_height, depth_fn, seg_fn=None, drivable=None, preset="outdoor"):
    """
    image : uint8 (H, W, 3) RGB; K must already match this exact image.
    drivable : "none" | "road" | "road+sidewalk"; None = the preset's default.
    Returns a dict (see the keys below). Grid row = forward, column 0 = far left.
    """
    P = PRESETS[preset]
    cell, x_max, y_half = P["cell"], P["x_max"], P["y_half"]
    drivable = drivable or P["drivable"]
    ids = DRIVABLE_IDS[drivable]
    if ids is not None and seg_fn is None:
        raise ValueError(f"drivable={drivable!r} needs a seg_fn")
    H, W = image.shape[:2]
    out = {"preset": preset, "drivable": drivable, "cell": cell, "x_max": x_max,
           "y_half": y_half, "robot_radius": P["robot_radius"],
           "timings": {"depth": 0.0, "seg": 0.0, "geometry": 0.0}}

    t0 = time.perf_counter()
    pred = depth_fn(image)
    out["timings"]["depth"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    fit = estimate_metric_scale(pred, K, camera_height)
    out["tilt_deg"] = tilt_deg(fit["n"]) if fit["n"] is not None else None
    if not fit["valid"]:                    # hard rule: no metric output on a refused fit
        out["timings"]["geometry"] = time.perf_counter() - t0
        out.update(status="refused", reason=fit["reason"], scale=None,
                   depth=pred, depth_scaled=False)
        return out

    depth, T = pred * fit["scale"], floor_frame(fit["n"], camera_height)
    depth_k, K_k = upsample_rows(depth, K)
    pts = transform_points(backproject(depth_k, K_k).reshape(-1, 3), T)
    labels = classify_heights(pts)
    # the photo overlay uses one 3D point per original pixel
    px = transform_points(backproject(depth, K).reshape(-1, 3), T)
    px_labels = classify_heights(px)
    if ids is not None:
        ts = time.perf_counter()
        sem = seg_fn(image)
        seg_time = time.perf_counter() - ts
        sem_k = cv2.resize(sem, (W, UPSAMPLE * H), interpolation=cv2.INTER_NEAREST)
        labels = fuse_labels(labels, sem_k.reshape(-1), ids)
        px_labels = fuse_labels(px_labels, sem.reshape(-1), ids)
    grid = build_grid(pts, labels, cell, x_max, y_half)
    inflated = inflate(grid, P["robot_radius"], cell)
    start, goal = pick_start_goal(inflated)
    path = astar(inflated, start, goal) if start is not None else None

    painted = (paint_photo(image, px, px_labels, grid, cell, x_max, y_half) * 255).round().astype(np.uint8)
    path_m = None
    if path:                                # path cells -> metres on the floor (x forward, y left)
        ix, iy = np.array(path).T
        path_m = np.stack([(ix + 0.5) * cell, y_half - (iy + 0.5) * cell], 1)
        floor = np.column_stack([path_m, np.zeros(len(path_m))])
        uv, _ = project(transform_points(floor, np.linalg.inv(T)), K)
        cv2.polylines(painted, [np.round(uv).astype(np.int32)], False, (0, 0, 255), 2, cv2.LINE_AA)

    out["timings"]["seg"] = seg_time if ids is not None else 0.0
    out["timings"]["geometry"] = time.perf_counter() - t0 - out["timings"]["seg"]
    out.update(status="ok", reason="ok", scale=fit["scale"], depth=depth, depth_scaled=True,
               grid=grid, grid_inflated=inflated, path_cells=path, path_m=path_m,
               painted=painted)
    return out

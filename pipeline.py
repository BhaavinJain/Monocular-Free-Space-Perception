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

from bev import FREE, OCCUPIED, build_grid, cell_centres, cell_index, inflate, visible_mask
from camera import backproject, in_image_mask, project, transform_points
from ground_plane import tilt_deg
from planner import astar, pick_start_goal
from pointcloud import GROUND, HOLE, IGNORE, OBSTACLE, classify_heights, floor_frame
from scale import estimate_metric_scale

UPSAMPLE = 4    # rows per image row before back-projection (see upsample_rows)
# outdoor = the KITTI setting the Phase 3-4 numbers were measured with.
# indoor defaults to no semantics: SegFormer only knows Cityscapes street
# classes and has not been tested indoors. marks = forward distances (m) of
# the distance lines drawn on the photo and the BEV map.
PRESETS = {
    "outdoor": {"cell": 0.10, "x_max": 20.0, "y_half": 10.0, "robot_radius": 1.0,
                "drivable": "road+sidewalk", "marks": (5.0, 10.0, 15.0, 20.0)},
    "indoor": {"cell": 0.05, "x_max": 10.0, "y_half": 5.0, "robot_radius": 0.3,
               "drivable": "none", "marks": (2.5, 5.0, 7.5, 10.0)},
}
# Cityscapes train ids (road = 0, sidewalk = 1). Fixed here because the
# server has no Segmenter to look the names up; check_pipeline.py asserts the
# ids and backend/check_api.py asserts the names against segformer.json.
DRIVABLE_IDS = {"none": None, "road": [0], "road+sidewalk": [0, 1]}
CITYSCAPES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
              "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
              "truck", "bus", "train", "motorcycle", "bicycle"]
SIDE_MARGIN = 1.0   # m: the verdict looks for obstacles this far beyond the robot's width

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


def fit_to_size(img, K, h, w):
    """
    Crop to the aspect ratio of (h, w), then resize to exactly (h, w); K follows.

    The crop is the largest one ANCHORED AT THE BOTTOM and centred sideways:
    the floor near the robot is at the bottom of the photo, so only sky or the
    sides are lost. Returns the image, its K and the crop box (y0, x0, ch, cw).
    """
    H, W = img.shape[:2]
    if W * h > H * w:                      # wider than the target: trim the sides
        ch, cw = H, round(H * w / h)
    else:                                  # taller: trim the top
        ch, cw = round(W * h / w), W
    y0, x0 = H - ch, (W - cw) // 2
    sx, sy = w / cw, h / ch
    Kr = K.copy()
    Kr[0, 0], Kr[1, 1] = K[0, 0] * sx, K[1, 1] * sy
    Kr[0, 2] = (K[0, 2] - x0 + 0.5) * sx - 0.5          # pixel centres stay aligned
    Kr[1, 2] = (K[1, 2] - y0 + 0.5) * sy - 0.5
    out = img[y0:y0 + ch, x0:x0 + cw]
    if (ch, cw) != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR)
    return out, Kr, (y0, x0, ch, cw)


def fuse_labels(labels, semantic, drivable_ids):
    """
    labels   : (N,) height labels from classify_heights
    semantic : (N,) class id of the pixel each point came from
    Flat points whose class is not drivable become obstacles.
    """
    out = labels.copy()
    out[(labels == GROUND) & ~np.isin(semantic, drivable_ids)] = OBSTACLE
    return out


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


def describe(grid, pts, labels, sem_k, path_m, T, K, H, W, P):
    """
    What a result means, in words and in lines on the photo (specified in
    CLAUDE.md, "M4 definitions"):

      verdict           is the robot-wide corridor straight ahead free, counted from
                        the first row the camera can see (the blind zone in front
                        of the robot is never seen); the nearest obstacle within
                        SIDE_MARGIN of that corridor, with its class
      ribbon_px         the A* path as a band as wide as the robot, in photo pixels
      distance_lines_px floor lines at the preset's forward distances, in photo pixels
    """
    cell, x_max, y_half, r = P["cell"], P["x_max"], P["y_half"], P["robot_radius"]
    T_inv = np.linalg.inv(T)

    def to_px(xy):                          # floor points (forward, left) -> photo pixels
        return project(transform_points(np.column_stack([xy, np.zeros(len(xy))]), T_inv), K)

    def obstacle(i, j):
        y = float(ys[i, j])
        name = "obstacle"
        if sem_k is not None:               # majority class of the points that made the cell OCCUPIED
            ci, cj, ok = cell_index(pts, cell, x_max, y_half)
            m = ok & (ci == i) & (cj == j) & ((labels == OBSTACLE) | (labels == HOLE))
            name = CITYSCAPES[np.bincount(sem_k[m]).argmax()]
        return {"distance_m": round(float(i * cell), 2), "lateral_m": round(y, 2), "class": name,
                "side": "ahead" if abs(y) <= r else ("left" if y > 0 else "right")}

    _, ys = cell_centres(cell, x_max, y_half)
    lane = np.abs(ys[0]) <= r                       # columns the robot's body sweeps
    seen = np.nonzero(visible_mask(T, K, H, W, cell, x_max, y_half)[:, lane].all(1))[0]
    v = {"visible_from_m": None, "corridor_clear_m": None, "nearest_obstacle": None,
         "path_reach_m": None, "path_length_m": None}

    # nearest row first, then nearest the centre line
    ii, jj = np.nonzero((grid == OCCUPIED) & (np.abs(ys) <= r + SIDE_MARGIN))
    if len(ii):
        k = np.lexsort((np.abs(ys[ii, jj]), ii))[0]
        v["nearest_obstacle"] = obstacle(ii[k], jj[k])
    blocker = None
    if len(seen):
        r0 = seen[0]
        bad = np.nonzero((grid[r0:, lane] != FREE).any(1))[0]
        r1 = r0 + bad[0] if len(bad) else grid.shape[0]
        v["visible_from_m"], v["corridor_clear_m"] = r0 * cell, r1 * cell
        if len(bad) and (grid[r1, lane] == OCCUPIED).any():   # the free run ends at a real obstacle
            cols = np.nonzero(lane)[0]
            cols = cols[grid[r1, cols] == OCCUPIED]
            blocker = obstacle(r1, cols[np.argmin(np.abs(ys[r1, cols]))])
    if path_m is not None:
        v["path_reach_m"] = float(path_m[:, 0].max())
        v["path_length_m"] = float(np.linalg.norm(np.diff(path_m, axis=0), axis=1).sum())

    vf, cc, ob = v["visible_from_m"], v["corridor_clear_m"], v["nearest_obstacle"]
    if blocker:
        level, text = "blocked", f"Stop: {blocker['class']} {blocker['distance_m']:.1f} m ahead"
    elif vf is None:
        level, text = "unknown", "The corridor straight ahead is not in the camera's view."
    elif cc <= vf:
        level, text = "unknown", (f"Nothing straight ahead is known to be free; "
                                  f"the floor in the first {vf:.0f} m is outside the camera's view.")
    else:
        # "the floor": an object's upper part can be seen inside that distance
        level, text = "clear", (f"Clear from {vf:.0f} m to {cc:.0f} m ahead; "
                                f"the floor in the first {vf:.0f} m is outside the camera's view.")
    if ob and level != "blocked":
        text += (f" Nearest obstacle: {ob['class']} {ob['distance_m']:.1f} m ahead"
                 + ("." if ob["side"] == "ahead" else f", {abs(ob['lateral_m']):.1f} m to the {ob['side']}."))
    v.update(level=level, text=text, blocker=blocker)

    ribbon = None
    if path_m is not None and len(path_m) > 1:
        d = np.gradient(path_m, axis=0)                       # direction of travel
        n = np.column_stack([-d[:, 1], d[:, 0]]) / np.linalg.norm(d, axis=1, keepdims=True)
        uv, z = to_px(np.vstack([path_m + r * n, (path_m - r * n)[::-1]]))   # left edge, right edge back
        ribbon = np.round(uv, 1).tolist() if (z > 0).all() else None

    lines, side = [], np.linspace(-y_half, y_half, 81)
    for dm in P["marks"]:
        uv, z = to_px(np.column_stack([np.full_like(side, dm), side]))
        lines.append({"d_m": dm, "points": np.round(uv[in_image_mask(uv, z, H, W)], 1).tolist()})
    return {"verdict": v, "ribbon_px": ribbon, "distance_lines_px": lines}


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
    sem = sem_k = None
    if ids is not None:
        ts = time.perf_counter()
        sem = seg_fn(image)
        seg_time = time.perf_counter() - ts
        sem_k = cv2.resize(sem, (W, UPSAMPLE * H), interpolation=cv2.INTER_NEAREST).reshape(-1)
        labels = fuse_labels(labels, sem_k, ids)
        px_labels = fuse_labels(px_labels, sem.reshape(-1), ids)
    grid = build_grid(pts, labels, cell, x_max, y_half)
    inflated = inflate(grid, P["robot_radius"], cell)
    start, goal = pick_start_goal(inflated)
    path = astar(inflated, start, goal) if start is not None else None

    # the path is no longer burned into the photo: describe() returns it as a
    # robot-wide ribbon in pixels, for the frontend to draw
    painted = (paint_photo(image, px, px_labels, grid, cell, x_max, y_half) * 255).round().astype(np.uint8)
    path_m = None
    if path:                                # path cells -> metres on the floor (x forward, y left)
        ix, iy = np.array(path).T
        path_m = np.stack([(ix + 0.5) * cell, y_half - (iy + 0.5) * cell], 1)
    desc = describe(grid, pts, labels, sem_k, path_m, T, K, H, W, P)

    out["timings"]["seg"] = seg_time if ids is not None else 0.0
    out["timings"]["geometry"] = time.perf_counter() - t0 - out["timings"]["seg"]
    out.update(status="ok", reason="ok", scale=fit["scale"], depth=depth, depth_scaled=True,
               grid=grid, grid_inflated=inflated, path_cells=path, path_m=path_m,
               painted=painted, T_floor_cam=T, px_labels=px_labels.reshape(H, W), sem=sem, **desc)
    return out

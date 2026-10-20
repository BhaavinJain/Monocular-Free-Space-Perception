import numpy as np

def valid_mask(gt, min_depth=1e-3, max_depth=80.0):
    """
    Pixels that count. 0 means "no LiDAR return". The 80 m cap is the
    standard KITTI protocol: beyond it, LiDAR returns are sparse and noisy,
    so evaluating there measures the sensor more than the model.
    """
    return (gt > min_depth) & (gt < max_depth)


def compute_depth_metrics(pred, gt, min_depth=1e-3, max_depth=80.0):
    """
    pred, gt : (H, W) arrays in METRES, same shape.
    Returns a dict of metrics plus n = number of pixels evaluated.
    """
    m = valid_mask(gt, min_depth, max_depth)
    if m.sum() == 0:
        raise ValueError("no valid ground-truth pixels to evaluate")

    # Predictions are clipped to the evaluation range — standard protocol.
    # A prediction of 0 would make log() explode; one of 500 m on a 30 m object would dominate rmse for no useful reason.
    p = np.clip(pred[m], min_depth, max_depth).astype(np.float64)
    g = gt[m].astype(np.float64)

    ratio = np.maximum(p / g, g / p)          # >= 1 always; 1 = perfect
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "sq_rel": float(np.mean((p - g) ** 2 / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "d1": float(np.mean(ratio < 1.25)),
        "d2": float(np.mean(ratio < 1.25 ** 2)),
        "d3": float(np.mean(ratio < 1.25 ** 3)),
        "n": int(m.sum()),
    }


def range_stratified(pred, gt, bands=((0, 5), (5, 10), (10, 20), (20, 80))):
    """
    The same metrics, computed separately per distance band of the GROUND
    TRUTH. Monocular depth error grows roughly with distance squared, so one
    aggregate number hides where the system is usable. This table shows it.

    Bands with no pixels (e.g. 0-5 m: KITTI's LiDAR cannot see the road
    right in front of the car) are reported as None.
    """
    out = {}
    for lo, hi in bands:
        band = (gt > lo) & (gt <= hi)
        if band.sum() == 0:
            out[f"{lo}-{hi}m"] = None
            continue
        gt_band = np.where(band, gt, 0.0)     # zero = "ignore" outside band
        out[f"{lo}-{hi}m"] = compute_depth_metrics(pred, gt_band)
    return out


def average_metrics(per_frame):
    """
    Mean of each metric over frames (the standard way papers report KITTI:
    compute per image, then average). per_frame: list of metric dicts.
    """
    keys = [k for k in per_frame[0] if k != "n"]
    avg = {k: float(np.mean([f[k] for f in per_frame])) for k in keys}
    avg["frames"] = len(per_frame)
    return avg
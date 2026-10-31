from freespace.camera import backproject
from freespace.ground_plane import fit_ground

def recover_scale(plane_d, camera_height):   # s = h / d. Example: h = 1.65 m, d = 0.85 units -> s = 1.94 m/unit
    if plane_d is None or plane_d <= 0:
        raise ValueError("plane distance must be positive")
    return camera_height / plane_d


def to_metric(depth, s):  # Multiply the whole map by one number. That's all scaling is.
    return depth * s

# Plausibility band for s. The depth network is already roughly metric (KITTI
# frames give s ~ 0.87-0.89), so s far from 1 means the plane RANSAC found is not
# the floor (e.g. a row of balconies, found in M4) or the camera height is wrong.
# This check RELIES on the network being roughly metric: with a relative-depth
# model, or a scene where the metric model is off by more than 2x, it would refuse
# valid floors.
S_MIN, S_MAX = 0.5, 2.0

def estimate_metric_scale(depth, K, camera_height, **fit_kwargs):
    cloud = backproject(depth, K)                                                               # backproject--> 3D points
    fit = fit_ground(cloud, depth, **fit_kwargs)                                                # fit_ground----> plane (n, d)
    fit["scale"] = recover_scale(fit["d"], camera_height) if fit["valid"] else None             # h / d---------> scale s
    if fit["valid"] and not S_MIN <= fit["scale"] <= S_MAX:
        fit["valid"], fit["reason"] = False, (
            f"camera-height scale ({fit['scale']:.1f}x) disagrees with the network's own metres: "
            f"the detected floor is probably not the floor, or the camera height is wrong")
        fit["scale"] = None                 # a refused fit carries no metric output
    return fit
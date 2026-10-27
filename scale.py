from camera import backproject
from ground_plane import fit_ground

def recover_scale(plane_d, camera_height):   # s = h / d. Example: h = 1.65 m, d = 0.85 units -> s = 1.94 m/unit
    if plane_d is None or plane_d <= 0:
        raise ValueError("plane distance must be positive")
    return camera_height / plane_d


def to_metric(depth, s):  # Multiply the whole map by one number. That's all scaling is.
    return depth * s

def estimate_metric_scale(depth, K, camera_height, **fit_kwargs):
    cloud = backproject(depth, K)                                                               # backproject--> 3D points 
    fit = fit_ground(cloud, depth, **fit_kwargs)                                                # fit_ground----> plane (n, d)
    fit["scale"] = recover_scale(fit["d"], camera_height) if fit["valid"] else None             # h / d---------> scale s
    return fit
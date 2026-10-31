import numpy as np
UP_CAM = np.array([0.0, -1.0, 0.0])

def lower_region_points(cloud, depth, lower_frac = 0.4, max_points = 20000, rng = None):     
    rng = rng or np.random.default_rng(0)
    H = depth.shape[0]
    rows = np.arange(H)[:, None] >= int(H * (1 - lower_frac))  # (H, 1) boolean mask for the bottom `lower_frac` of the image
    mask = rows & (depth > 0)                                   # (H, W) boolean mask for the bottom `lower_frac` of the image with valid depth
    pts = cloud[mask]                                        # (N, 3) array of 3D points in the bottom `lower_frac` of the image with valid depth
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace = False)]
    return pts
 
def tilt_deg(n):    # Angle between a plane normal and camera 'up', in degrees
    return float(np.degrees(np.arccos(np.clip(n @ UP_CAM, -1.0, 1.0))))

def fit_plane_ransac(points, n_iters=200, rel_thresh=0.02, max_tilt=15.0,refine_iters=5, rng=None):
    rng = rng or np.random.default_rng(0)
    N = len(points)
    if N < 3:
        return None, None, None
 
    scale = np.median(np.linalg.norm(points, axis=1))
    thresh = rel_thresh * scale
 
    best_count, best_inliers = 0, None
    for _ in range(n_iters):
        # 1. three random points define a candidate plane
        p1, p2, p3 = points[rng.choice(N, 3, replace=False)]
        n = np.cross(p2 - p1, p3 - p1)            # perpendicular to both edges
        norm = np.linalg.norm(n)
        if norm < 1e-12:                          # points (nearly) collinear
            continue
        n = n / norm
        d = -n @ p1                               # plane passes through p1
        if d < 0:                                 # orient: camera side positive
            n, d = -n, -d
 
        # 2. reject planes that cannot be a floor
        if tilt_deg(n) > max_tilt:
            continue
 
        # 3. count the points that agree with it
        inliers = np.abs(points @ n + d) < thresh         # Vote
        count = inliers.sum()
        if count > best_count:
            best_count, best_inliers = count, inliers
 
    if best_inliers is None:
        return None, None, None
    # Refine the best plane by iteratively re-fitting to the inliers and re-evaluating the inlier set. 
    # This is a common practice in RANSAC to improve the estimate after the initial random sampling.
    min_sigma = 0.002 * scale
    inliers = best_inliers
    for _ in range(refine_iters):
        n, d = fit_plane_lstsq(points[inliers])
        r = points @ n + d
        band = r[np.abs(r) < thresh]
        sigma = max(1.4826 * np.median(np.abs(band - np.median(band))), min_sigma)
        inliers = np.abs(r) < 3.0 * sigma
        if inliers.sum() < 3:
            return None, None, None
    n, d = fit_plane_lstsq(points[inliers])
 
    # Report support at the ORIGINAL tolerance: "what fraction of the lower
    # region agrees with this floor?" — that is what the validity check needs.
    support = np.abs(points @ n + d) < thresh
    return n, d, support
 
 
def fit_plane_lstsq(points):
    c = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - c, full_matrices=False)
    n = Vt[-1]
    d = -n @ c
    if d < 0:
        n, d = -n, -d
    return n, d
 
 
def is_valid_fit(n, inlier_ratio, max_tilt=15.0, min_inlier_ratio=0.3):
    if n is None:
        return False, "no floor-like plane found in the lower image region"
    if tilt_deg(n) > max_tilt:
        return False, f"plane tilted {tilt_deg(n):.1f} deg from vertical normal"
    if inlier_ratio < min_inlier_ratio:
        return False, (f"only {100 * inlier_ratio:.0f}% of lower-region points "
                       f"lie on the plane (need {100 * min_inlier_ratio:.0f}%)")
    return True, "ok"
 
def fit_ground(cloud, depth, lower_frac=0.4, rel_thresh=0.02, max_tilt=15.0,
               min_inlier_ratio=0.3, rng=None):
    """
    Full ground-plane step: select lower region -> RANSAC -> validate.
 
    Returns a dict:
      valid, reason, n, d, inlier_ratio, pitch_deg, roll_deg, n_points
    """
    pts = lower_region_points(cloud, depth, lower_frac, rng=rng)
    n, d, inliers = fit_plane_ransac(pts, rel_thresh=rel_thresh,
                                     max_tilt=max_tilt, rng=rng)
    ratio = float(inliers.mean()) if inliers is not None else 0.0
    ok, reason = is_valid_fit(n, ratio, max_tilt, min_inlier_ratio)
 
    result = {"valid": ok, "reason": reason, "n": n, "d": d,
             "inlier_ratio": ratio, "n_points": len(pts)}
    return result
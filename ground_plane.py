"""Find the floor in a 3D point cloud
    Finds the floor in a 3D point cloud and reports two things: 
    i) how high the camera is above it (in whatever units the points are in) and 
    ii) whether the fit can be trusted. 
    The height is the d in s = h / d, so everything about metric scale depends on this file.
"""

import numpy as np
UP_CAM = np.array([0.0, -1.0, 0.0])

def lower_region_points(cloud, depth, lower_frac = 0.4, max_points = 20000, rng = None):     
    """ cloud : (H, W, 3) from camera.backproject()
        depth : (H, W) the depth map that cloud was built from (0 = invalid)

        Pick the 3D points most likely to be floor, from the bottom 40% of the image.
        Select 3D points from the bottom `lower_frac` of the image — where floor dominates — that have a valid depth"""
    
    rng = rng or np.random.default_rng(0)
    H = depth.shape[0]
    rows = np.arrange(H)[:, None] >= int(H * (1 - lower_frac))  # (H, 1) boolean mask for the bottom `lower_frac` of the image
    mask = rows & (depth > 0)                                   # (H, W) boolean mask for the bottom `lower_frac` of the image with valid depth
    pts = cloud[mask]                                        # (N, 3) array of 3D points in the bottom `lower_frac` of the image with valid depth
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace = False)]
    return pts
 
def tilt_deg(n):    # Angle between a plane normal and camera 'up', in degrees
    return float(np.degrees(np.arccos(np.clip(n @ UP_CAM, -1.0, 1.0))))
import numpy as np

def _read_calib_file(path):                 # turn a KITTI text calibration file into a dictionary of NumPy arrays.
    data = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":",1)                   # split at the first colon
            try:
                data[key.strip()] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass                                        # Handle the case where conversion to float fails
    return data

def load_calib(cam_to_cam_path, velo_to_cam_path, cam_id = 2):           
    c2c = _read_calib_file(cam_to_cam_path)
    v2c = _read_calib_file(velo_to_cam_path)

    # (a) LiDAR to camera 0 (unrectified)
    T_cam0_velo = np.eye(4)                              # 4x4 identity matrix
    T_cam0_velo[:3 , :3] = v2c["R"].reshape(3,3)         # Rotation
    T_cam0_velo[:3 , 3] = v2c["T"].reshape(3)            # Translation
 
    # (b) Camera 0 rectification
    T_rect0_cam0 = np.eye(4)
    T_rect0_cam0[:3 , :3] = c2c["R_rect_00"].reshape(3,3)

    # (c) Splitting P_rect_02 into K and baseline translation b
    P = c2c[f"P_rect_0{cam_id}"].reshape(3,4)        # 3x4 projection matrix, building the key name "P_rect_02" (P = K @ [ I | b ])
    K = P[:, :3].copy()                              # 3x3 camera intrinsics matrix
    b = np.linalg.solve(K, P[:, 3])                  # baseline translation (camera 2 offset) b = K^-1 @ P[:,3], in metre

    T_cam_rect0 = np.eye(4)                              # 4x4 identity matrix
    T_cam_rect0[:3, 3] = b

    # Finally mulitplying the LiDAR to camera 0 (unectified) 
    # with the rectification rotation and the baseline translation to get LiDAR to camera 2 (rectified)
    T_cam_velo = T_cam_rect0 @ T_rect0_cam0 @ T_cam0_velo

    return K, T_cam_velo

def transform_points(points, T):   # PERFORMS THE VELO TO CAM CONVERSION
    points = np.asarray(points, dtype=np.float64)
    ones = np.ones((points.shape[0], 1))
    homo = np.hstack([points, np.ones((points.shape[0], 1))])      # (N, 4),turning [x, y, z] into [x, y, z, 1] (horizontal stacking)
    return (T@ homo.T).T[:, :3]          

def project(points_cam, K):              # 3D camera to 2D pixel,    
    points_cam = np.asarray(points_cam, dtype = np.float64)
    z = points_cam[: , 2]
    proj = (K @ points_cam.T).T
    with np.errstate(divide = "ignore", invalid = "ignore"):
        uv = proj[:, :2]/proj[:, 2:3]
    return uv,z

def in_image_mask(uv , z, height, width, min_depth = 0.1):
    u = np.round(uv[: , 0])
    v = np.round(uv[: , 1])
    return ((z > min_depth) & (u >= 0) & (u < width) & (v >= 0) & (v < height))

def backproject(depth_map, K):                # Inverse of project , rebuild the 3D point for each pixel (for depth network)
    H, W = depth_map.shape                    
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # u = column index, v = row index, shape = (H, W)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth_map.astype(np.float64)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.stack([X,Y,Z], axis = -1)

def lidar_to_depth_map(points_velo, K, T_cam_velo, height, width):
    # Take x, y, z only (the Velodyne's 4th column is reflectance, which isn't geometry) and move them into the camera frame.
    pts_cam = transform_points(points_velo[:, :3], T_cam_velo)     
    uv, z = project(pts_cam, K)                                 
    keep = in_image_mask(uv, z, height, width)                  
    u = np.round(uv[keep, 0]).astype(int)
    v = np.round(uv[keep, 1]).astype(int)
    z = z[keep]

    depth = np.full((height, width), np.inf)    # Create a depth map of shape (height, width) filled with infinity values. This will be used to store the minimum depth for each pixel.
    # Two LiDAR points can land on the same pixel, 
    # for example the edge of a car at 8 m and the wall behind it at 20 m. The camera sees only the nearer one,
    # so the pixel should get 8 m.
    np.minimum.at(depth, (v, u), z) 
    depth[np.isinf(depth)] = 0.0            # Replace all remaining infinity values with 0.0, indicating that no LiDAR point projected to those pixels.
    return depth.astype(np.float32)
"""Pinhole camera geometry for KITTI : moving points between 3D and the image plane.
    velo --> Velodyne LiDAR frame: 3D points in the velodyne coordinate system (X_forward, Y_left, Z_up) (METRES)
    cam --> RECTIFIED camera-2 frame: 3D points in the left color camera coordinate  (X_right, Y_down, Z_forward) (METRES)
        (example: "5m ahead" means x=5 in velo and z=5 in cam)

    (note: the camera frame is right-handed, while the velodyne frame is left-handed)

    A 4x4 matrix called 'T_a_b' takes a point written in frame b and returns the same physical point written in frame a:
 
        p_a = T_a_b @ p_b          (@ is matrix multiplication, p_a and p_b are 4x1 homogeneous coordinates)
        
        Chains read right-to-left and the inner letters must cancel:
 
        T_a_c = T_a_b @ T_b_c        (b cancels -> correct)
""" 

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
    """Building 2 things every later phase needs:
        1) K : 3x3 camera intrinsics matrix of rectified camera
        2) T_cam_velo : 4x4 rigid body transformation matrix, LiDAR frame -> rectified camera frame
               
        a) R,T in calib_velo_to_cam.txt are the from velodyne to camera frame 0 (unrectified) = T_cam0_velo
        b) R_rect_00 in calib_cam_to_cam.txt is the small rectification rotation for camera 0 = T_rect0_cam0
        c) P_rect_02 in calib_cam_to_cam.txt, the 3x4 projection matrix rectified cam-0 frame -> pixels of camera 2""" 

    c2c = _read_calib_file(cam_to_cam_path)
    v2c = _read_calib_file(velo_to_cam_path)

    # (a) LiDAR to camera 0 (unrectified)
    T_cam0_velo = np.eye(4)                              # 4x4 identity matrix
    T_cam0_velo[:3 , :3] = v2c["R"].reshape(3,3)         # Rotation
    T_cam0_velo[:3 , 3] = v2c["T"].reshape(3)            # Translation
 
    # (b) Camera 0 rectification           (pure rotation)
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

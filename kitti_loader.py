# kitti_loader.py — load one KITTI raw drive.

import os 
import numpy as np
import cv2

from camera import load_calib

class KittiRawDrive:
    CALIB_FOLDER = "caliberation files"

    def __init__(self, root, date, drive, depth_root = None, cam_id = 2):
        """
        root       : folder containing the day folders (e.g. data/)
        date       : "2011_09_26"
        drive      : 23 or "0023"
        depth_root : folder containing train/ and val/ of the depth benchmark
        """
        self.cam_id = cam_id
        self.date = date
        self.drive_name = f"{date}_drive_{int(drive):04d}_sync"               # 2011_09_26_drive_0023_sync
        self.date_dir = os.path.join(root, date)                              # data/2011_09_26
        self.drive_dir = os.path.join(self.date_dir, self.drive_name)         # data/2011_09_26/2011_09_26_drive_0023_sync
        self.calib_dir = os.path.join(self.date_dir, self.CALIB_FOLDER)      # data/2011_09_26//caliberation files
        self.depth_root = depth_root

        if not os.path.exists(self.drive_dir):
            raise FileNotFoundError(f"Drive directory {self.drive_dir} does not exist.")
        for name in ("calib_cam_to_cam.txt", "calib_velo_to_cam.txt"):
            path = os.path.join(self.calib_dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Calibration file {path} does not exist.")

        self.K, self.T_cam_velo = load_calib(
            os.path.join(self.calib_dir, "calib_cam_to_cam.txt"),
            os.path.join(self.calib_dir, "calib_velo_to_cam.txt"),
            cam_id = self.cam_id
        )

    @staticmethod
    def _name(idx):
        return f"{idx:010d}"

    def __len__(self):
        img_dir = os.path.join(self.drive_dir, f"image_0{self.cam_id}", "data")
        return len([f for f in os.listdir(img_dir) if f.endswith(".png")])

    def load_image(self, idx):
        # Load an image by index.
        # RGB uint8 image, shape (H, W, 3). KITTI is ~375 x 1242
        path = os.path.join(self.drive_dir, f"image_0{self.cam_id}", "data", self._name(idx) + ".png")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_velodyne(self, idx):
        """
        LiDAR scan as (N, 4) float32: x, y, z, reflectance.
        The .bin file is a raw dump of float32 numbers with no header,
        so we read it flat and reshape into rows of 4.
        N is roughly 100-120k points per scan.
        """
        path = os.path.join(self.drive_dir, "velodyne_points", "data", self._name(idx) + ".bin")
        return np.fromfile(path, dtype = np.float32).reshape(-1,4)

    def load_depth_gt(self, idx):
        """
        loads KITTI's official depth ground truth for one frame: 
        an (H, W) array where each pixel holds the true distance in metres, or 0 where there's no measurement

        Rebuild the depth map yourself from raw LiDAR, and 
        compare it pixel-by-pixel against KITTI's official version to prove your geometry matches theirs.
 
        Encoding: 16-bit PNG, depth_in_metres = pixel_value / 256.
        IMREAD_UNCHANGED is essential — the default flag would squash the
        16-bit values to 8-bit and silently destroy the depth.
 
        Note: the benchmark skips the first and last 5 frames of each
        drive, so frame 0 has no GT. Returns None if the file is missing.
        """
        if self.depth_root is None:
            return None
        for split in ("train", "val"):
            path = os.path.join(self.depth_root, split, self.drive_name,
                                "proj_depth", "groundtruth",
                                f"image_0{self.cam_id}", self._name(idx) + ".png")
            if os.path.exists(path):
                raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                assert raw.dtype == np.uint16, "depth PNG must be 16-bit"
                return raw.astype(np.float32) / 256.0
        return None
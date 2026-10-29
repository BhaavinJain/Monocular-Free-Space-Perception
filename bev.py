import cv2
import numpy as np
 
from camera import transform_points, project, in_image_mask
from pointcloud import GROUND, IGNORE, classify_heights
 
FREE, OCCUPIED, UNKNOWN = 0, 1, 2
 
 
def grid_shape(cell, x_max, y_half):
    return int(round(x_max / cell)), int(round(2 * y_half / cell))
 
 
def cell_index(points_floor, cell, x_max, y_half):
    """Row (forward) and column (sideways, col 0 = far left) of each point's cell,
    and whether the point lies inside the grid at all."""
    nx, ny = grid_shape(cell, x_max, y_half)
    ix = np.floor(points_floor[:, 0] / cell).astype(int)
    iy = np.floor((y_half - points_floor[:, 1]) / cell).astype(int)
    return ix, iy, (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
 
 
def build_grid(points_floor, labels, cell, x_max, y_half):
    """
    Drop every labelled point into its cell, then decide each cell:
 
        any obstacle or hole point  -> OCCUPIED   (obstacle wins: safety)
        else any ground point       -> FREE
        else                        -> UNKNOWN
 
    Counting uses np.bincount on a flattened cell index: one vectorised pass
    over ~400k points instead of a Python loop.
    """
    nx, ny = grid_shape(cell, x_max, y_half)
    ix, iy, inside = cell_index(points_floor, cell, x_max, y_half)
    ok = inside & (labels != IGNORE)
    flat = ix[ok] * ny + iy[ok]
    lab = labels[ok]
 
    ground = np.bincount(flat[lab == GROUND], minlength=nx * ny).reshape(nx, ny)
    blocked = np.bincount(flat[lab != GROUND], minlength=nx * ny).reshape(nx, ny)
 
    grid = np.full((nx, ny), UNKNOWN, dtype=np.uint8)
    grid[ground > 0] = FREE
    grid[blocked > 0] = OCCUPIED
    return grid
 
 
def cell_centres(cell, x_max, y_half):
    """(nx, ny) arrays of each cell centre's forward x and sideways y, in metres."""
    nx, ny = grid_shape(cell, x_max, y_half)
    x = (np.arange(nx) + 0.5) * cell
    y = y_half - (np.arange(ny) + 0.5) * cell
    return np.meshgrid(x, y, indexing="ij")
 
 
def visible_mask(T_floor_cam, K, H, W, cell, x_max, y_half):
    """
    Which floor cells can the camera see at all?
 
    Take each cell's centre ON THE FLOOR (z = 0), move it into the camera
    frame, project it with K. If it lands inside the H x W image, the camera
    looks at that patch of floor; if not, it can't. This one test gives both
    the sideways field-of-view wedge AND the blind zone in front of the robot
    (on KITTI: no road closer than ~5.9 m is in the image).
 
    It ignores occlusion: floor hidden behind a car still counts as "in the
    image". That is handled by the points themselves — no floor points reach
    such cells, so they stay UNKNOWN.
    """
    xs, ys = cell_centres(cell, x_max, y_half)
    pts = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    uv, z = project(transform_points(pts, np.linalg.inv(T_floor_cam)), K)
    return in_image_mask(uv, z, H, W).reshape(xs.shape)
 
 
def inflate(grid, robot_radius, cell):
    """
    Configuration-space inflation: grow every OCCUPIED cell by the robot's
    radius. A planner can then treat the robot as a single point — if the
    point stays in FREE cells, the whole robot clears every obstacle.
    UNKNOWN stays UNKNOWN (a planner should not enter it either).
    """
    r = int(np.ceil(robot_radius / cell))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    grown = cv2.dilate((grid == OCCUPIED).astype(np.uint8), disk) > 0
    out = grid.copy()
    out[grown & (grid == FREE)] = OCCUPIED
    return out
 
 
def lidar_to_bev(scan, T_cam_velo, T_floor_cam, K, H, W, cell, x_max, y_half,
                 **height_kwargs):
    """
    Ground-truth grid from a LiDAR scan, built the SAME way as the predicted
    one (same frame convention, same height labels, same cell rule), then
    restricted to what the camera can see — the camera is what's being
    graded, so cells outside its view must not count against it.
    """
    pts_floor = transform_points(transform_points(scan[:, :3], T_cam_velo), T_floor_cam)
    grid = build_grid(pts_floor, classify_heights(pts_floor, **height_kwargs),
                      cell, x_max, y_half)
    grid[~visible_mask(T_floor_cam, K, H, W, cell, x_max, y_half)] = UNKNOWN
    return grid
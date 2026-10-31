import numpy as np

GROUND, OBSTACLE, HOLE, IGNORE = 0, 1, 2, 3


def floor_frame(n, d):
    z = n
    forward = np.array([0.0, 0.0, 1.0])            # camera's viewing axis
    x = forward - (forward @ z) * z                # flatten onto the floor
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)                             # left
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z])                # rows = new axes
    T[:3, 3] = [0.0, 0.0, d]                       # camera sits d above origin
    return T


def classify_heights(points_floor, ground_eps=0.15, hole_eps=0.25, max_height=2.0):
    """
    Label each point (N, 3) in the floor frame by its height z:

        |z| <= ground_eps          GROUND    drivable surface
        ground_eps < z <= max_h    OBSTACLE  something the robot would hit
        z < -hole_eps              HOLE      a drop: kerb edge, step, pit
        z > max_height             IGNORE    above the robot (tree canopy, sky,
                                             signs) — can't be hit, so it must
                                             not block the map
        -hole_eps <= z < -ground_eps IGNORE  too uncertain to call either way """
    z = points_floor[:, 2]
    labels = np.full(len(z), IGNORE, dtype=np.int8)
    labels[np.abs(z) <= ground_eps] = GROUND
    labels[(z > ground_eps) & (z <= max_height)] = OBSTACLE
    labels[z < -hole_eps] = HOLE
    return labels
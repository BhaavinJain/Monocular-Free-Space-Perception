import heapq

import cv2
import numpy as np

from bev import FREE

STEPS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def octile(a, b):
    """Exact cost between two cells on an empty 8-connected grid: the heuristic."""
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)


def astar(grid, start, goal):
    """Shortest 8-connected path of FREE cells from start to goal, or None."""
    nx, ny = grid.shape
    queue = [(octile(start, goal), 0.0, start)]       # (g + h, g, cell)
    best = {start: 0.0}
    came_from = {start: None}
    while queue:
        _, g, cell = heapq.heappop(queue)
        if cell == goal:                               # walk back to the start
            path = []
            while cell is not None:
                path.append(cell)
                cell = came_from[cell]
            return path[::-1]
        if g > best[cell]:                             # stale queue entry
            continue
        for di, dj in STEPS:
            nxt = (cell[0] + di, cell[1] + dj)
            if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny) or grid[nxt] != FREE:
                continue
            g2 = g + (np.sqrt(2) if di and dj else 1.0)
            if g2 < best.get(nxt, np.inf):
                best[nxt], came_from[nxt] = g2, cell
                heapq.heappush(queue, (g2 + octile(nxt, goal), g2, nxt))
    return None


def pick_start_goal(grid):
    """
    A demo task, one frame at a time:
      start = the FREE cell nearest the robot (the robot itself stands in the
              camera's blind zone, which is UNKNOWN)
      goal  = the reachable FREE cell farthest ahead, preferring the centre line
    "Reachable" = in the same connected patch of free cells as the start, so
    A* is never asked for an impossible route.
    """
    free = (grid == FREE).astype(np.uint8)
    if not free.any():
        return None, None
    rows, cols = np.nonzero(free)
    centre = grid.shape[1] / 2
    i = np.argmin(rows ** 2 + (cols - centre) ** 2)
    start = (int(rows[i]), int(cols[i]))
    _, comp = cv2.connectedComponents(free, connectivity=8)
    rows, cols = np.nonzero(comp == comp[start])
    i = np.argmax(rows - 0.5 * np.abs(cols - centre))
    return start, (int(rows[i]), int(cols[i]))
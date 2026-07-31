"""A* pathfinding on a 2D occupancy grid.

Package-safe (stdlib + numpy only).
"""

from __future__ import annotations

import heapq
import math

import numpy as np

NEIGHBORS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


def astar(
    safe: np.ndarray,
    clearance_m: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    scale: float,
) -> list[tuple[int, int]] | None:
    height, width = safe.shape
    g_score = np.full((height, width), np.inf, dtype=np.float32)
    parent_row = np.full((height, width), -1, dtype=np.int32)
    parent_col = np.full((height, width), -1, dtype=np.int32)
    closed = np.zeros((height, width), dtype=bool)

    def heuristic(row: int, col: int) -> float:
        return math.hypot(row - goal[0], col - goal[1]) * scale

    g_score[start] = 0.0
    queue: list[tuple[float, float, int, int]] = [
        (heuristic(*start), 0.0, start[0], start[1])
    ]

    while queue:
        _, current_g, row, col = heapq.heappop(queue)
        if closed[row, col]:
            continue
        closed[row, col] = True
        if (row, col) == goal:
            path = []
            cursor = goal
            while cursor != (-1, -1):
                path.append(cursor)
                parent = (
                    int(parent_row[cursor]),
                    int(parent_col[cursor]),
                )
                cursor = parent
            return path[::-1]

        for d_row, d_col, step_factor in NEIGHBORS:
            n_row, n_col = row + d_row, col + d_col
            if not (0 <= n_row < height and 0 <= n_col < width):
                continue
            if not safe[n_row, n_col] or closed[n_row, n_col]:
                continue
            if d_row and d_col:
                # Prevent diagonal corner cutting.
                if not safe[row + d_row, col] or not safe[row, col + d_col]:
                    continue
            clearance = max(float(clearance_m[n_row, n_col]), 0.05)
            clearance_multiplier = 1.0 + 0.12 / clearance
            tentative = (
                current_g + step_factor * scale * clearance_multiplier
            )
            if tentative >= float(g_score[n_row, n_col]):
                continue
            g_score[n_row, n_col] = tentative
            parent_row[n_row, n_col] = row
            parent_col[n_row, n_col] = col
            heapq.heappush(
                queue,
                (
                    tentative + heuristic(n_row, n_col),
                    tentative,
                    n_row,
                    n_col,
                ),
            )
    return None
"""Point cloud utilities (numpy only)."""

from __future__ import annotations

import numpy as np


def voxel_downsample(
    points: np.ndarray, voxel_size: float, max_points: int, seed: int
) -> np.ndarray:
    """Voxel-downsample ``points`` preserving the legacy algorithm exactly.

    Mirrors ``generate_sage3d_trajectories.py::voxel_downsample``:
    floor-divide by ``voxel_size`` to integer voxel coords, keep the first
    point per voxel (sorted by first-seen index), then random-subsample to
    ``max_points`` with the given seed when needed. Output is ``float32``.
    """
    voxel = np.floor(points / voxel_size).astype(np.int64)
    _, indices = np.unique(voxel, axis=0, return_index=True)
    sampled = points[np.sort(indices)]
    if len(sampled) > max_points:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(sampled), max_points, replace=False))
        sampled = sampled[selected]
    return sampled.astype(np.float32)
"""Geometry primitives: MapTransform, coordinate conversion, path metrics.

Package-safe (stdlib + numpy only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MapTransform:
    height: int
    width: int
    scale: float
    lower_x: float
    lower_y: float

    def pixel_to_world(self, row: int, col: int) -> tuple[float, float]:
        x = self.lower_x + (col + 0.5) * self.scale
        # Raw InteriorGS occupancy maps use row 0 at the lower world-Y bound.
        # SAGE3D's semantic-map export flips the raw occupancy image for
        # visualization, but that flip must not be applied while planning
        # directly on occupancy.png.
        y = self.lower_y + (row + 0.5) * self.scale
        return x, y

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        col = int(round((x - self.lower_x) / self.scale - 0.5))
        row = int(round((y - self.lower_y) / self.scale - 0.5))
        return row, col


def pixels_to_world(
    pixels: Iterable[tuple[int, int]], transform: MapTransform
) -> np.ndarray:
    return np.asarray(
        [transform.pixel_to_world(row, col) for row, col in pixels],
        dtype=np.float64,
    )


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
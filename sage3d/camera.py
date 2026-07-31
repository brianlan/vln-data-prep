"""OpenCV fisheye camera calibration (numpy + stdlib).

``CameraCalibration`` wraps ``fisheye_camera.opencv_fisheye_parameters`` and
adds width/height/FOV/finite-value validation per the Phase 1 contract.
``extrinsic_matrix`` delegates to :func:`sage3d.frames.camera_extrinsic` so the
extrinsic formula has exactly one implementation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from fisheye_camera import opencv_fisheye_parameters

from sage3d.frames import camera_extrinsic


class CameraCalibration:
    """Validated OpenCV fisheye calibration for one resolution + FOV."""

    def __init__(
        self,
        width: int,
        height: int,
        horizontal_fov_deg: float,
        fisheye_coefficients: Sequence[float],
    ) -> None:
        if not isinstance(width, int) or width <= 0:
            raise ValueError(f"width must be a positive int, got {width!r}")
        if not isinstance(height, int) or height <= 0:
            raise ValueError(f"height must be a positive int, got {height!r}")
        if not math.isfinite(horizontal_fov_deg) or not (
            0.0 < horizontal_fov_deg <= 360.0
        ):
            raise ValueError(
                "horizontal_fov_deg must be finite and in (0, 360] degrees, "
                f"got {horizontal_fov_deg!r}"
            )
        coefficients = tuple(float(value) for value in fisheye_coefficients)
        if len(coefficients) != 4 or not all(
            math.isfinite(value) for value in coefficients
        ):
            raise ValueError(
                "fisheye_coefficients must be four finite floats, "
                f"got {fisheye_coefficients!r}"
            )

        params = opencv_fisheye_parameters(
            width, height, horizontal_fov_deg, coefficients
        )
        self.width = width
        self.height = height
        self.horizontal_fov_deg = float(horizontal_fov_deg)
        self.fisheye_coefficients: tuple[float, float, float, float] = coefficients
        self.cx: float = float(params["cx"])
        self.cy: float = float(params["cy"])
        self.fx: float = float(params["fx"])
        self.fy: float = float(params["fy"])
        self.vertical_fov_deg: float = float(params["vertical_fov_deg"])
        self.forward_mask_radius_pixels: float = float(
            params["forward_mask_radius_pixels"]
        )

    def intrinsic_matrix(self) -> np.ndarray:
        return np.asarray(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def extrinsic_matrix(self, camera_height: float) -> np.ndarray:
        return camera_extrinsic(camera_height)

    def distortion_vector(self) -> np.ndarray:
        return np.asarray(self.fisheye_coefficients, dtype=np.float32)
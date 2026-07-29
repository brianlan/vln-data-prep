#!/usr/bin/env python3
"""Shared OpenCV fisheye camera calculations."""

from __future__ import annotations

import math
from collections.abc import Sequence


def validate_fisheye_coefficients(
    coefficients: Sequence[float],
) -> tuple[float, float, float, float]:
    if len(coefficients) != 4:
        raise ValueError("OpenCV fisheye requires exactly four coefficients")
    result = tuple(float(value) for value in coefficients)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("Fisheye coefficients must be finite")
    return result


def distort_theta(theta: float, coefficients: Sequence[float]) -> float:
    """Apply OpenCV's fisheye angular polynomial."""
    k1, k2, k3, k4 = validate_fisheye_coefficients(coefficients)
    theta2 = theta * theta
    return theta * (
        1.0
        + k1 * theta2
        + k2 * theta2**2
        + k3 * theta2**3
        + k4 * theta2**4
    )


def focal_length_for_horizontal_fov(
    width: int,
    horizontal_fov_deg: float,
    coefficients: Sequence[float],
) -> float:
    """Return square-pixel focal length for the requested full horizontal FOV."""
    if width <= 0:
        raise ValueError("Image width must be positive")
    if not 0.0 < horizontal_fov_deg <= 360.0:
        raise ValueError("Horizontal FOV must be in (0, 360] degrees")
    half_angle = math.radians(horizontal_fov_deg) / 2.0
    distorted_half_angle = distort_theta(half_angle, coefficients)
    if distorted_half_angle <= 0.0:
        raise ValueError(
            "Fisheye polynomial is non-positive at the horizontal image edge"
        )
    return (width / 2.0) / distorted_half_angle


def angle_for_image_radius(
    radius_pixels: float,
    focal_pixels: float,
    coefficients: Sequence[float],
    *,
    max_angle_rad: float = math.pi,
) -> float:
    """Invert the monotonic OpenCV fisheye angular mapping by bisection."""
    if radius_pixels < 0.0 or focal_pixels <= 0.0:
        raise ValueError("Radius must be non-negative and focal length positive")
    target = radius_pixels / focal_pixels
    low = 0.0
    high = max_angle_rad
    if distort_theta(high, coefficients) < target:
        raise ValueError("Requested image radius is outside the fisheye model")
    for _ in range(80):
        middle = (low + high) / 2.0
        if distort_theta(middle, coefficients) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def opencv_fisheye_parameters(
    width: int,
    height: int,
    horizontal_fov_deg: float,
    coefficients: Sequence[float],
) -> dict[str, float | list[float]]:
    """Calculate centered, square-pixel OpenCV fisheye calibration metadata."""
    if height <= 0:
        raise ValueError("Image height must be positive")
    distortion = validate_fisheye_coefficients(coefficients)
    focal = focal_length_for_horizontal_fov(
        width, horizontal_fov_deg, distortion
    )
    vertical_half_angle = angle_for_image_radius(
        height / 2.0,
        focal,
        distortion,
    )
    forward_radius = focal * distort_theta(math.pi / 2.0, distortion)
    return {
        "cx": width / 2.0,
        "cy": height / 2.0,
        "fx": focal,
        "fy": focal,
        "horizontal_fov_deg": float(horizontal_fov_deg),
        "vertical_fov_deg": math.degrees(2.0 * vertical_half_angle),
        "forward_mask_radius_pixels": forward_radius,
        "fisheye_coefficients": list(distortion),
    }

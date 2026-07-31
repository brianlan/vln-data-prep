"""Focused tests for issue #10 render/package wiring migration call sites.

Each migrated call site in ``render_fisheye_sage3d.py`` and
``package_lerobot_sage3d.py`` is asserted to produce identical output to the
legacy inlined implementation it replaced. Package-safe: numpy + stdlib only.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sage3d.camera import CameraCalibration  # noqa: E402
from sage3d.frames import camera_extrinsic, yaw_to_quaternion  # noqa: E402
from sage3d.naming import frame_stem  # noqa: E402
from sage3d.render_processing import (  # noqa: E402
    RawDepthSummaryAccumulator,
    build_forward_mask,
    encode_depth,
    mask_rgb,
)


# --- legacy inlined implementations (verbatim from pre-migration scripts) -----


def _legacy_camera_quaternion(yaw: float) -> np.ndarray:
    return np.asarray(
        [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
        dtype=np.float32,
    )


def _legacy_fisheye_intrinsic(fx, fy, cx, cy) -> np.ndarray:
    return np.asarray(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _legacy_camera_extrinsic(camera_height: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[2, 3] = camera_height
    return transform


def _legacy_build_circular_mask(width, height, cx, cy, radius) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def _legacy_mask_rgb(rgb, circular_mask) -> np.ndarray:
    rgb = rgb.copy()
    rgb[~circular_mask] = 0
    return rgb


def _legacy_encode_depth(depth, circular_mask, min_depth_m, max_depth_m, depth_scale) -> np.ndarray:
    depth = depth.copy()
    finite = np.isfinite(depth) & (depth >= min_depth_m)
    depth = np.nan_to_num(depth, nan=max_depth_m, posinf=max_depth_m, neginf=max_depth_m)
    depth[~finite] = max_depth_m
    depth[~circular_mask] = max_depth_m
    depth = np.clip(depth, 0.0, max_depth_m)
    return np.rint(depth * depth_scale).astype(np.uint16)


def _legacy_frame_stem(episode_index, frame_index) -> str:
    return f"episode_{episode_index:06d}_{frame_index:03d}"


def _sample_calibration(width=600, height=450, fov=180.0, coeffs=(0.1, 0.0, 0.0, 0.0)):
    return CameraCalibration(width, height, fov, coeffs)


# --- migrated call site: yaw_to_quaternion -----------------------------------


def test_yaw_to_quaternion_matches_legacy_camera_quaternion():
    for yaw in (0.0, 0.5, 1.0, -1.0, 3.14159, 2 * math.pi):
        legacy = _legacy_camera_quaternion(yaw)
        prod = yaw_to_quaternion(yaw)
        assert np.array_equal(legacy, prod)
        assert prod.dtype == np.float32


# --- migrated call site: CameraCalibration intrinsic/extrinsic --------------


def test_calibration_intrinsic_matches_legacy_fisheye_intrinsic():
    cal = _sample_calibration()
    legacy = _legacy_fisheye_intrinsic(cal.fx, cal.fy, cal.cx, cal.cy)
    prod = cal.intrinsic_matrix()
    assert np.array_equal(legacy, prod)
    assert prod.dtype == np.float32


def test_calibration_extrinsic_matches_legacy_camera_extrinsic():
    cal = _sample_calibration()
    for height in (0.0, 0.6, 1.5):
        legacy = _legacy_camera_extrinsic(height)
        prod = cal.extrinsic_matrix(height)
        assert np.array_equal(legacy, prod)
        assert prod.dtype == np.float32
    # Also verify the module-level frames.camera_extrinsic matches.
    for height in (0.0, 0.6, 1.5):
        assert np.array_equal(_legacy_camera_extrinsic(height), camera_extrinsic(height))


def test_calibration_distortion_is_float32_tuple():
    cal = _sample_calibration(coeffs=(0.2, -0.1, 0.05, 0.0))
    dist = cal.distortion_vector()
    assert np.array_equal(dist, np.asarray((0.2, -0.1, 0.05, 0.0), dtype=np.float32))


# --- migrated call site: build_forward_mask ----------------------------------


def test_build_forward_mask_matches_legacy_circular_mask():
    cal = _sample_calibration()
    legacy = _legacy_build_circular_mask(
        cal.width, cal.height, cal.cx, cal.cy, cal.forward_mask_radius_pixels
    )
    prod = build_forward_mask(
        cal.width, cal.height, cal.cx, cal.cy, cal.forward_mask_radius_pixels
    )
    assert np.array_equal(legacy, prod)
    assert prod.dtype == np.bool_


# --- migrated call site: mask_rgb --------------------------------------------


def test_mask_rgb_matches_legacy_rgb_zeroing():
    cal = _sample_calibration()
    mask = build_forward_mask(
        cal.width, cal.height, cal.cx, cal.cy, cal.forward_mask_radius_pixels
    )
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, (cal.height, cal.width, 3), dtype=np.uint8)
    rgb_original = rgb.copy()
    legacy = _legacy_mask_rgb(rgb, mask)
    prod = mask_rgb(rgb, mask)
    assert np.array_equal(legacy, prod)
    # mask_rgb must not mutate the caller's array.
    assert np.array_equal(rgb, rgb_original)


# --- migrated call site: encode_depth ---------------------------------------


def test_encode_depth_matches_legacy_encoder_on_random_depth():
    cal = _sample_calibration()
    mask = build_forward_mask(
        cal.width, cal.height, cal.cx, cal.cy, cal.forward_mask_radius_pixels
    )
    rng = np.random.default_rng(99)
    for min_d, max_d, scale in (
        (0.05, 6.0, 10000.0),
        (0.1, 10.0, 5000.0),
    ):
        depth = rng.uniform(-1.0, 12.0, (cal.height, cal.width)).astype(np.float32)
        # Inject NaN and inf.
        depth[0, 0] = np.nan
        depth[1, 1] = np.inf
        depth[2, 2] = -np.inf
        legacy = _legacy_encode_depth(depth, mask, min_d, max_d, scale)
        prod = encode_depth(depth, mask, min_d, max_d, scale)
        assert np.array_equal(legacy, prod), f"mismatch for ({min_d},{max_d},{scale})"


# --- migrated call site: RawDepthSummaryAccumulator -------------------------


def test_accumulator_matches_legacy_per_frame_reduction():
    cal = _sample_calibration()
    mask = build_forward_mask(
        cal.width, cal.height, cal.cx, cal.cy, cal.forward_mask_radius_pixels
    )
    min_d = 0.05
    rng = np.random.default_rng(7)
    # Simulate per-frame depth accumulation (legacy approach).
    legacy_fractions = []
    legacy_minima = []
    legacy_maxima = []
    accumulator = RawDepthSummaryAccumulator(mask, min_d)
    for _ in range(10):
        depth = rng.uniform(0.0, 10.0, (cal.height, cal.width)).astype(np.float32)
        finite = np.isfinite(depth) & (depth >= min_d)
        valid_inside = finite & mask
        legacy_fractions.append(float(valid_inside.sum() / mask.sum()))
        legacy_minima.append(float(depth[valid_inside].min()))
        legacy_maxima.append(float(depth[valid_inside].max()))
        accumulator.add(depth)
    summary = accumulator.finish()
    assert math.isclose(
        summary["finite_depth_fraction_mean"], float(np.mean(legacy_fractions))
    )
    assert math.isclose(
        summary["finite_depth_fraction_min"], float(np.min(legacy_fractions))
    )
    assert math.isclose(
        summary["finite_depth_min_m"], float(min(legacy_minima))
    )
    assert math.isclose(
        summary["finite_depth_max_m"], float(max(legacy_maxima))
    )


# --- migrated call site: frame_stem -----------------------------------------


def test_frame_stem_matches_legacy_fstring():
    for ep in (0, 1, 42, 999999):
        for fr in (0, 1, 25, 103, 999):
            assert frame_stem(ep, fr) == _legacy_frame_stem(ep, fr)
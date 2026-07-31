"""Tests for sage3d.render_processing Phase 1 module.

Covers build_forward_mask, mask_rgb, encoded_depth_sentinel, encode_depth
(the synthetic matrix from the issue Tests Required), and
RawDepthSummaryAccumulator (zero/no-valid/shape/squeeze/dtype/order tests).
Package-safe: numpy only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sage3d.render_processing import (  # noqa: E402
    RawDepthSummaryAccumulator,
    build_forward_mask,
    encode_depth,
    encoded_depth_sentinel,
    mask_rgb,
)


# --- build_forward_mask -----------------------------------------------------


def test_build_forward_mask_shape_and_dtype():
    mask = build_forward_mask(6, 4, cx=3.0, cy=2.0, radius=2.0)
    assert mask.shape == (4, 6)
    assert mask.dtype == np.bool_
    # Center inside, corner outside.
    assert mask[2, 3]
    assert not mask[0, 0]


def test_build_forward_mask_matches_legacy_formula():
    width, height = 600, 450
    cx, cy, radius = 300.0, 225.0, 225.0
    yy, xx = np.ogrid[:height, :width]
    expected = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    got = build_forward_mask(width, height, cx, cy, radius)
    assert np.array_equal(got, expected)


# --- mask_rgb ---------------------------------------------------------------


def test_mask_rgb_zeroes_outside_mask_and_does_not_mutate():
    rgb = np.full((4, 5, 3), 200, dtype=np.uint8)
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:4] = True
    out = mask_rgb(rgb, mask)
    assert out[~mask].sum() == 0
    # Inside-mask pixels unchanged.
    assert np.all(out[mask] == 200)
    # Original not mutated.
    assert np.all(rgb == 200)


def test_mask_rgb_rejects_shape_mismatch():
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    mask = np.zeros((3, 5), dtype=bool)
    with pytest.raises(ValueError, match="circular_mask shape"):
        mask_rgb(rgb, mask)


# --- encoded_depth_sentinel -------------------------------------------------


def test_sentinel_matches_encoder_formula():
    assert encoded_depth_sentinel(6.0, 10000.0) == np.uint16(60000)


def test_sentinel_float32_half_step_rounds_to_501():
    # Pinned: max_depth_m=1.001, depth_scale=500 -> float32 product rounds to 501
    # (Python float64 round produces 500). See contract pin.
    assert encoded_depth_sentinel(1.001, 500) == np.uint16(501)


def test_sentinel_overflow_raises_at_float32_threshold():
    # Pinned: max_depth_m=131.07, depth_scale=500 -> float64 product is 65535,
    # float32 product exceeds 65535 and must raise.
    with pytest.raises(ValueError, match="exceeds 65535"):
        encoded_depth_sentinel(131.07, 500)


def test_sentinel_non_finite_rejected():
    with pytest.raises(ValueError):
        encoded_depth_sentinel(float("inf"), 10000.0)
    with pytest.raises(ValueError):
        encoded_depth_sentinel(6.0, float("nan"))


def test_sentinel_non_positive_rejected():
    with pytest.raises(ValueError):
        encoded_depth_sentinel(-1.0, 10000.0)
    with pytest.raises(ValueError):
        encoded_depth_sentinel(6.0, 0.0)


# --- encode_depth: synthetic matrix -----------------------------------------


def _mask(w: int = 4, h: int = 4) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[1:3, 1:3] = True
    return m


def test_encode_depth_basic_valid_value():
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    mask = _mask()
    out = encode_depth(depth, mask, min_depth_m=0.05, max_depth_m=6.0, depth_scale=10000.0)
    assert out.dtype == np.uint16
    assert out.shape == depth.shape
    sentinel = encoded_depth_sentinel(6.0, 10000.0)
    assert np.all(out[~mask] == sentinel)
    assert np.all(out[mask] == np.uint16(20000))


def test_encode_depth_nan_becomes_sentinel():
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    mask = _mask()
    depth[1, 1] = np.nan
    out = encode_depth(depth, mask, 0.05, 6.0, 10000.0)
    sentinel = encoded_depth_sentinel(6.0, 10000.0)
    assert out[1, 1] == sentinel


def test_encode_depth_posinf_and_neginf_become_sentinel():
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    mask = _mask()
    depth[1, 2] = np.inf
    depth[2, 1] = -np.inf
    out = encode_depth(depth, mask, 0.05, 6.0, 10000.0)
    sentinel = encoded_depth_sentinel(6.0, 10000.0)
    assert out[1, 2] == sentinel
    assert out[2, 1] == sentinel


def test_encode_depth_below_min_becomes_sentinel():
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    mask = _mask()
    depth[1, 1] = 0.01  # below min 0.05
    out = encode_depth(depth, mask, 0.05, 6.0, 10000.0)
    sentinel = encoded_depth_sentinel(6.0, 10000.0)
    assert out[1, 1] == sentinel


def test_encode_depth_exactly_at_min_is_valid():
    depth = np.full((4, 4), 0.05, dtype=np.float32)
    mask = _mask()
    out = encode_depth(depth, mask, 0.05, 6.0, 10000.0)
    assert np.all(out[mask] == np.uint16(500))


def test_encode_depth_above_max_clipped_to_sentinel():
    depth = np.full((4, 4), 7.0, dtype=np.float32)  # above max 6.0
    mask = _mask()
    out = encode_depth(depth, mask, 0.05, 6.0, 10000.0)
    sentinel = encoded_depth_sentinel(6.0, 10000.0)
    # Clipped to max_depth_m then encoded == sentinel.
    assert np.all(out[mask] == sentinel)


def test_encode_depth_exactly_at_max_is_valid():
    depth = np.full((4, 4), 6.0, dtype=np.float32)
    mask = _mask()
    out = encode_depth(depth, mask, 0.05, 6.0, 10000.0)
    sentinel = encoded_depth_sentinel(6.0, 10000.0)
    assert np.all(out[mask] == sentinel)


def test_encode_depth_rint_half_step_bankers_rounding():
    # 2.00005 * 10000 = 20000.5 -> np.rint (half-to-even) -> 20000.
    depth = np.full((4, 4), 2.00005, dtype=np.float32)
    mask = _mask()
    out = encode_depth(depth, mask, 0.05, 6.0, 10000.0)
    assert np.all(out[mask] == np.uint16(20000))


def test_encode_depth_integral_scale():
    depth = np.full((4, 4), 1.5, dtype=np.float32)
    mask = _mask()
    out = encode_depth(depth, mask, 0.05, 6.0, depth_scale=1000.0)
    assert np.all(out[mask] == np.uint16(1500))


def test_encode_depth_non_integral_scale():
    depth = np.full((4, 4), 1.5, dtype=np.float32)
    mask = _mask()
    out = encode_depth(depth, mask, 0.05, 6.0, depth_scale=333.3)
    expected = np.uint16(np.rint(np.float32(1.5) * np.float32(333.3)))
    assert np.all(out[mask] == expected)


def test_encode_depth_overflow_raises_before_output():
    # max_depth_m * depth_scale overflows uint16 in float32 -> helper raises.
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    mask = _mask()
    with pytest.raises(ValueError, match="exceeds 65535"):
        encode_depth(depth, mask, 0.05, max_depth_m=131.07, depth_scale=500.0)


def test_encode_depth_rejects_non_2d_depth():
    depth = np.zeros((4,), dtype=np.float32)
    mask = np.zeros(4, dtype=bool)
    with pytest.raises(ValueError, match="depth must be 2-D"):
        encode_depth(depth, mask, 0.05, 6.0, 10000.0)


def test_encode_depth_rejects_non_bool_mask():
    depth = np.zeros((4, 4), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="circular_mask must be bool"):
        encode_depth(depth, mask, 0.05, 6.0, 10000.0)


def test_encode_depth_rejects_shape_mismatch():
    depth = np.zeros((4, 4), dtype=np.float32)
    mask = np.zeros((3, 4), dtype=bool)
    with pytest.raises(ValueError, match="depth shape"):
        encode_depth(depth, mask, 0.05, 6.0, 10000.0)


def test_encode_depth_rejects_invalid_min_max_range():
    depth = np.zeros((4, 4), dtype=np.float32)
    mask = _mask()
    with pytest.raises(ValueError, match="min_depth_m"):
        encode_depth(depth, mask, min_depth_m=-1.0, max_depth_m=6.0, depth_scale=10000.0)
    with pytest.raises(ValueError, match="must be < max_depth_m"):
        encode_depth(depth, mask, min_depth_m=6.0, max_depth_m=6.0, depth_scale=10000.0)


def test_encode_depth_does_not_mutate_inputs():
    depth = np.array([[0.05, 2.0], [np.nan, 7.0]], dtype=np.float32)
    mask = np.array([[True, True], [True, False]], dtype=bool)
    depth_copy = depth.copy()
    mask_copy = mask.copy()
    encode_depth(depth, mask, 0.05, 6.0, 10000.0)
    assert np.array_equal(depth, depth_copy, equal_nan=True)
    assert np.array_equal(mask, mask_copy)


def test_encode_depth_float64_input_preserves_float32_precision():
    # float64 inputs must go through the same float32 path as float32 inputs.
    depth32 = np.full((4, 4), 2.00005, dtype=np.float32)
    depth64 = np.full((4, 4), 2.00005, dtype=np.float64)
    mask = _mask()
    out32 = encode_depth(depth32, mask, 0.05, 6.0, 10000.0)
    out64 = encode_depth(depth64, mask, 0.05, 6.0, 10000.0)
    assert np.array_equal(out32, out64)


def test_encode_depth_outside_mask_exactly_sentinel_even_for_valid_pixels():
    # A valid depth outside the mask must still be the sentinel.
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    mask = _mask()
    out = encode_depth(depth, mask, 0.05, 6.0, 10000.0)
    sentinel = encoded_depth_sentinel(6.0, 10000.0)
    assert np.all(out[~mask] == sentinel)
    assert np.all(out[mask] != sentinel)


# --- RawDepthSummaryAccumulator ---------------------------------------------


def _frame_mask(h: int = 4, w: int = 4) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[1:3, 1:3] = True
    return m


def test_accumulator_basic_summary():
    mask = _frame_mask()
    acc = RawDepthSummaryAccumulator(mask, 0.05)
    f = np.full((4, 4), 2.0, dtype=np.float32)
    acc.add(f)
    acc.add(f)
    result = acc.finish()
    assert result["finite_depth_fraction_mean"] == pytest.approx(1.0)
    assert result["finite_depth_fraction_min"] == pytest.approx(1.0)
    assert result["finite_depth_min_m"] == pytest.approx(2.0)
    assert result["finite_depth_max_m"] == pytest.approx(2.0)


def test_accumulator_finish_raises_when_zero_frames_added():
    mask = _frame_mask()
    acc = RawDepthSummaryAccumulator(mask, 0.05)
    with pytest.raises(ValueError, match="no frames"):
        acc.finish()


def test_accumulator_add_raises_when_no_valid_depth():
    mask = _frame_mask()
    acc = RawDepthSummaryAccumulator(mask, 0.05)
    # All NaN -> no valid.
    f = np.full((4, 4), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="no valid depth"):
        acc.add(f)
    # All below min -> no valid.
    f = np.full((4, 4), 0.01, dtype=np.float32)
    with pytest.raises(ValueError, match="no valid depth"):
        acc.add(f)


def test_accumulator_squeezes_singleton_dimension():
    mask = _frame_mask()
    acc = RawDepthSummaryAccumulator(mask, 0.05)
    # (1, 4, 4) squeezes to (4, 4).
    f = np.full((1, 4, 4), 2.0, dtype=np.float32)
    acc.add(f)
    result = acc.finish()
    assert result["finite_depth_min_m"] == pytest.approx(2.0)


def test_accumulator_rejects_shape_mismatch_after_squeeze():
    mask = _frame_mask()
    acc = RawDepthSummaryAccumulator(mask, 0.05)
    f = np.full((3, 4), 2.0, dtype=np.float32)
    with pytest.raises(ValueError, match="frame shape"):
        acc.add(f)


def test_accumulator_rejects_non_bool_mask():
    mask = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="circular_mask must be bool"):
        RawDepthSummaryAccumulator(mask, 0.05)


def test_accumulator_rejects_empty_mask():
    mask = np.zeros((4, 4), dtype=bool)
    with pytest.raises(ValueError, match="non-empty"):
        RawDepthSummaryAccumulator(mask, 0.05)


def test_accumulator_rejects_invalid_min_depth():
    mask = _frame_mask()
    with pytest.raises(ValueError, match="min_depth_m"):
        RawDepthSummaryAccumulator(mask, -1.0)


def test_accumulator_dtype_is_float32_after_add():
    # float64 input straddling min after float32 conversion.
    mask = np.zeros((4, 4), dtype=bool)
    mask[0:2, 0:2] = True  # top-left 2x2 inside
    acc = RawDepthSummaryAccumulator(mask, 0.05)
    f = np.array(
        [[0.04, 0.05, 1.0, 1.0],
         [2.0,  1.0,  1.0, 1.0],
         [5.0,  5.0,  5.0, 5.0],
         [5.0,  5.0,  5.0, 5.0]],
        dtype=np.float64,
    )
    acc.add(f)
    result = acc.finish()
    # Inside-mask (0:2,0:2): 0.04 -> invalid (<min), 0.05/2.0/1.0 -> valid.
    # Min 0.05, max 2.0, fraction 3/4.
    assert result["finite_depth_min_m"] == pytest.approx(0.05)
    assert result["finite_depth_max_m"] == pytest.approx(2.0)
    assert result["finite_depth_fraction_mean"] == pytest.approx(0.75)


def test_accumulator_input_non_mutation():
    mask = _frame_mask()
    acc = RawDepthSummaryAccumulator(mask, 0.05)
    f = np.full((4, 4), 2.0, dtype=np.float32)
    f_copy = f.copy()
    acc.add(f)
    assert np.array_equal(f, f_copy)


def test_accumulator_preserves_legacy_reduction_order():
    # Replicate the legacy per-frame-list reduction from render_fisheye_sage3d.py.
    mask = _frame_mask()
    frames = [
        np.array([[0.1, 1.0, 1.0, 1.0],
                  [0.1, 1.0, 2.0, 2.0],
                  [0.1, 1.0, 2.0, 2.0],
                  [0.1, 0.1, 0.1, 0.1]], dtype=np.float32),
        np.array([[5.0, 5.0, 5.0, 5.0],
                  [5.0, 3.0, 3.0, 5.0],
                  [5.0, 3.0, 3.0, 5.0],
                  [5.0, 5.0, 5.0, 5.0]], dtype=np.float32),
    ]
    acc = RawDepthSummaryAccumulator(mask, 0.05)
    legacy_fractions = []
    legacy_min = []
    legacy_max = []
    for depth in frames:
        frame = np.asarray(depth, dtype=np.float32).squeeze()
        finite = np.isfinite(frame) & (frame >= 0.05)
        valid_inside = finite & mask
        legacy_fractions.append(float(valid_inside.sum()) / float(mask.sum()))
        legacy_min.append(float(frame[valid_inside].min()))
        legacy_max.append(float(frame[valid_inside].max()))
        acc.add(depth)
    result = acc.finish()
    assert result["finite_depth_fraction_mean"] == pytest.approx(float(np.mean(legacy_fractions)))
    assert result["finite_depth_fraction_min"] == pytest.approx(float(np.min(legacy_fractions)))
    assert result["finite_depth_min_m"] == pytest.approx(float(min(legacy_min)))
    assert result["finite_depth_max_m"] == pytest.approx(float(max(legacy_max)))


def test_accumulator_above_max_depth_included_in_max():
    # Valid depths above max_depth_m are included in finite_depth_max_m (clipping
    # is for PNG encoding only, not the summary).
    mask = _frame_mask()
    acc = RawDepthSummaryAccumulator(mask, 0.05)
    f = np.full((4, 4), 7.0, dtype=np.float32)  # above max 6.0 but still valid
    acc.add(f)
    result = acc.finish()
    assert result["finite_depth_max_m"] == pytest.approx(7.0)
    assert result["finite_depth_min_m"] == pytest.approx(7.0)
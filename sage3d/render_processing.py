"""Render processing primitives (numpy only).

Package-safe: stdlib + numpy. No Isaac, cv2, scipy, or trimesh.

Owns the float32 depth encoder, the shared uint16 sentinel, the circular
forward mask, RGB masking, and the streaming raw-depth accumulator. The
contracts are specified in ``SAGE3D_REFACTOR_PLAN.md`` revision 8
(``encode_depth`` contract and Streaming raw-depth summary contract).
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


# --- forward mask -----------------------------------------------------------


def build_forward_mask(
    width: int, height: int, cx: float, cy: float, radius: float
) -> np.ndarray:
    """Circular forward mask from center and radius (bool, shape ``(height, width)``).

    Mirrors the inline formula in ``render_fisheye_sage3d.py`` and the Phase 0b
    ``build_circular_mask`` helper in ``check_render.py``.
    """
    yy, xx = np.ogrid[:height, :width]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def mask_rgb(rgb: np.ndarray, circular_mask: np.ndarray) -> np.ndarray:
    """Return a copy of ``rgb`` with outside-mask pixels set to zero.

    Mirrors ``rgb[~circular_mask] = 0`` from ``render_fisheye_sage3d.py`` without
    mutating the caller's array.
    """
    if circular_mask.shape != rgb.shape[:2]:
        raise ValueError(
            f"circular_mask shape {circular_mask.shape} != rgb HxW {rgb.shape[:2]}"
        )
    out = np.array(rgb, copy=True)
    out[~circular_mask] = 0
    return out


# --- depth sentinel ---------------------------------------------------------


def encoded_depth_sentinel(max_depth_m: float, depth_scale: float) -> np.uint16:
    """Return the uint16 sentinel for out-of-range depth.

    Mirrors the production encoder: ``np.rint(max_depth_m * depth_scale)``
    computed in float32, checked against 65535 before conversion.
    """
    if not (np.isfinite(max_depth_m) and max_depth_m > 0):
        raise ValueError(f"max_depth_m must be finite positive, got {max_depth_m}")
    if not (np.isfinite(depth_scale) and depth_scale > 0):
        raise ValueError(f"depth_scale must be finite positive, got {depth_scale}")
    scaled = np.asarray([max_depth_m], dtype=np.float32) * np.float32(depth_scale)
    if not np.isfinite(scaled[0]):
        raise ValueError("scaled sentinel is not finite")
    if float(scaled[0]) > 65535:
        raise ValueError(f"scaled sentinel {float(scaled[0])} exceeds 65535")
    return np.rint(scaled).astype(np.uint16)[0]


# --- depth encoder ----------------------------------------------------------


def encode_depth(
    depth: np.ndarray,
    circular_mask: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    depth_scale: float,
) -> np.ndarray:
    """Encode raw float depth to ``uint16`` PNG units.

    See the ``encode_depth`` contract in ``SAGE3D_REFACTOR_PLAN.md`` revision 8.
    Never mutates ``depth`` or ``circular_mask``.
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2-D, got ndim={depth.ndim}")
    if circular_mask.dtype != np.bool_:
        raise ValueError(f"circular_mask must be bool, got {circular_mask.dtype}")
    if depth.shape != circular_mask.shape:
        raise ValueError(
            f"depth shape {depth.shape} != circular_mask shape {circular_mask.shape}"
        )
    if not np.isfinite(min_depth_m) or min_depth_m < 0:
        raise ValueError(f"min_depth_m must be finite non-negative, got {min_depth_m}")
    # max_depth_m, depth_scale positivity/finite/overflow validated by the helper.
    if not (min_depth_m < max_depth_m):
        raise ValueError(
            f"min_depth_m {min_depth_m} must be < max_depth_m {max_depth_m}"
        )
    # Sentinel validates max_depth_m/depth_scale and preflights overflow before
    # any output allocation (intentional Phase 1 guard).
    sentinel = encoded_depth_sentinel(max_depth_m, depth_scale)

    # Private float32 copy preserves current operation precision and leaves the
    # caller's array untouched.
    work = np.array(depth, dtype=np.float32)
    finite = np.isfinite(work) & (work >= np.float32(min_depth_m))
    valid_inside = finite & circular_mask
    # NaN, +/-inf, below-min, and outside-mask pixels become max_depth_m.
    work = np.where(valid_inside, work, np.float32(max_depth_m))
    # Clip valid-above-max to max_depth_m for encoding only (legacy clip 0..max).
    work = np.clip(work, np.float32(0.0), np.float32(max_depth_m))
    # Encode with float32 multiply + NumPy rint (half-to-even) + uint16 cast.
    encoded = np.rint(work * np.float32(depth_scale)).astype(np.uint16)
    # Outside-mask pixels are exactly the sentinel (max_depth_m encoded).
    # Guard against any stray rounding drift outside the mask.
    encoded[~circular_mask] = sentinel
    return encoded


# --- streaming raw-depth accumulator ----------------------------------------


class RawDepthEpisodeSummary(TypedDict):
    finite_depth_fraction_mean: float
    finite_depth_fraction_min: float
    finite_depth_min_m: float
    finite_depth_max_m: float


class RawDepthSummaryAccumulator:
    """Streaming per-episode raw-depth summary (never retains raw frames).

    See the Streaming raw-depth summary contract in ``SAGE3D_REFACTOR_PLAN.md``
    revision 8. Preserves the legacy per-frame-list reduction order exactly.
    """

    def __init__(self, circular_mask: np.ndarray, min_depth_m: float) -> None:
        if circular_mask.dtype != np.bool_:
            raise ValueError(
                f"circular_mask must be bool, got {circular_mask.dtype}"
            )
        if not np.isfinite(min_depth_m) or min_depth_m < 0:
            raise ValueError(
                f"min_depth_m must be finite non-negative, got {min_depth_m}"
            )
        mask_total = int(circular_mask.sum())
        if mask_total == 0:
            raise ValueError("circular_mask must be non-empty")
        self._mask = circular_mask
        self._min = np.float32(min_depth_m)
        self._mask_total = mask_total
        self._fractions: list[float] = []
        self._minima: list[float] = []
        self._maxima: list[float] = []

    def add(self, depth: np.ndarray) -> None:
        """Accumulate one frame. Raises if no valid depth in the frame."""
        frame = np.asarray(depth, dtype=np.float32).squeeze()
        if frame.ndim != 2:
            raise ValueError(f"frame must be 2-D after squeeze, got ndim={frame.ndim}")
        if frame.shape != self._mask.shape:
            raise ValueError(
                f"frame shape {frame.shape} != circular_mask shape {self._mask.shape}"
            )
        valid = np.isfinite(frame) & (frame >= self._min) & self._mask
        if not valid.any():
            raise ValueError("frame contains no valid depth")
        self._fractions.append(float(valid.sum()) / float(self._mask_total))
        self._minima.append(float(frame[valid].min()))
        self._maxima.append(float(frame[valid].max()))

    def finish(self) -> RawDepthEpisodeSummary:
        """Return the episode summary. Raises if no frames were added."""
        if not self._fractions:
            raise ValueError("no frames were added")
        # Legacy reduction order: np.mean/np.min over fractions, min/max over
        # per-frame extrema (not a running sum).
        return RawDepthEpisodeSummary(
            finite_depth_fraction_mean=float(np.mean(self._fractions)),
            finite_depth_fraction_min=float(np.min(self._fractions)),
            finite_depth_min_m=float(min(self._minima)),
            finite_depth_max_m=float(max(self._maxima)),
        )
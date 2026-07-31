#!/usr/bin/env python3
"""Read-only package-Python checker for rendered RGB/depth artifacts.

Two modes per the plan (Checker execution contract, revision 8):

- ``validate``: baseline-independent inventory, calibration, trajectory
  linkage, encoded-depth structure, and summary shapes/ranges/counts.
- ``compare-golden``: runs validate, then tolerant RGB/depth metrics on
  selected frames (first/middle/last per episode, de-duplicated).

Phase 0b temporarily owns ``encoded_depth_sentinel`` as a standalone helper.
Phase 1 moves it to ``sage3d.render_processing`` and rewires the checker.

Package-safe: stdlib + numpy + PIL. No forbidden imports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make the repo root and package_safe test helpers importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "tests" / "package_safe") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tests" / "package_safe"))

from artifact_parsers import parse_render_summary, parse_trajectory_manifest  # noqa: E402
from sage3d_canonical.provenance import _atomic_write_json  # noqa: E402


# ---------------------------------------------------------------------------
# Depth sentinel (Phase 0b standalone; Phase 1 moves to render_processing)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Circular mask (NumPy-only, same formula as render_fisheye_sage3d.py)
# ---------------------------------------------------------------------------


def build_circular_mask(width: int, height: int, cx: float, cy: float, radius: float) -> np.ndarray:
    """Build the circular forward mask from center and radius."""
    yy, xx = np.ogrid[:height, :width]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def _selected_frame_indices(frame_count: int) -> list[int]:
    """First/middle/last, de-duplicated, preserving order."""
    if frame_count <= 0:
        return []
    indices = {0, frame_count - 1, frame_count // 2}
    return sorted(indices)


# ---------------------------------------------------------------------------
# Validate mode
# ---------------------------------------------------------------------------


def validate(rendered_dir: Path, trajectory_dir: Path) -> dict[str, Any]:
    """Baseline-independent validation of rendered artifacts."""
    errors: list[str] = []

    # Load trajectory manifest for linkage.
    try:
        manifest = parse_trajectory_manifest(trajectory_dir)
    except Exception as e:
        return {"eligible": False, "errors": [f"trajectory manifest: {e}"], "warnings": []}

    # Load both render summaries (rgb + depth).
    try:
        rgb_summary = parse_render_summary(rendered_dir, "rgb_render_summary.json")
    except Exception as e:
        errors.append(f"rgb_render_summary: {e}")
        rgb_summary = None

    try:
        depth_summary = parse_render_summary(rendered_dir, "depth_render_summary.json")
    except Exception as e:
        errors.append(f"depth_render_summary: {e}")
        depth_summary = None

    # Also check the generic render_summary.json.
    try:
        parse_render_summary(rendered_dir, "render_summary.json")
    except Exception as e:
        errors.append(f"render_summary: {e}")

    scene_id = manifest.get("scene_id")

    # Cross-artifact: scene_id must match across summaries and manifest.
    for name, summary in (("rgb_render_summary", rgb_summary), ("depth_render_summary", depth_summary)):
        if summary is not None and summary.get("scene_id") != scene_id:
            errors.append(f"{name} scene_id {summary.get('scene_id')} != manifest {scene_id}")

    # Inventory: check RGB/depth file counts match manifest episodes.
    rgb_dir = rendered_dir / "observation.images.rgb"
    depth_dir = rendered_dir / "observation.images.depth"
    expected_frames = sum(ep["frame_count"] for ep in manifest["episodes"])

    if rgb_summary is not None:
        rgb_files = sorted(rgb_dir.glob("*.jpg"))
        if len(rgb_files) != expected_frames:
            errors.append(f"RGB file count {len(rgb_files)} != manifest total_frames {expected_frames}")
        if rgb_summary.get("total_frames") != expected_frames:
            errors.append(f"rgb summary total_frames {rgb_summary.get('total_frames')} != manifest {expected_frames}")

    if depth_summary is not None:
        depth_files = sorted(depth_dir.glob("*.png"))
        if len(depth_files) != expected_frames:
            errors.append(f"depth file count {len(depth_files)} != manifest total_frames {expected_frames}")
        if depth_summary.get("total_frames") != expected_frames:
            errors.append(f"depth summary total_frames {depth_summary.get('total_frames')} != manifest {expected_frames}")

    # Encoded-depth structure: check a sample depth PNG for uint16 dtype and
    # sentinel value presence outside the mask.
    if depth_summary is not None:
        from PIL import Image

        width, height = depth_summary["resolution"]
        cx, cy = depth_summary.get("principal_point", [width / 2.0, height / 2.0])
        radius = depth_summary["forward_mask_radius_pixels"]
        mask = build_circular_mask(width, height, cx, cy, radius)
        sentinel = encoded_depth_sentinel(depth_summary["max_depth_m"], depth_summary["depth_scale"])

        depth_files_check = sorted(depth_dir.glob("*.png"))
        if depth_files_check:
            # Check first frame for dtype and sentinel.
            first = np.array(Image.open(depth_files_check[0]))
            if first.dtype != np.uint16:
                errors.append(f"depth PNG dtype {first.dtype} != uint16")
            if first.shape != (height, width):
                errors.append(f"depth PNG shape {first.shape} != ({height}, {width})")
            outside = first[~mask]
            if outside.size > 0:
                if not np.all(outside == sentinel):
                # Outside-mask pixels should be the sentinel value.
                    errors.append(
                        f"depth outside-mask pixels != sentinel {sentinel}; "
                        f"got unique={np.unique(outside)[:5]}"
                    )

    eligible = len(errors) == 0
    return {
        "eligible": eligible,
        "errors": errors,
        "warnings": [],
        "scene_id": scene_id,
        "episode_count": manifest.get("episode_count"),
    }


# ---------------------------------------------------------------------------
# Compare-golden mode
# ---------------------------------------------------------------------------


def _decode_rgb(path: Path) -> np.ndarray:
    """Decode a JPEG RGB frame as float64 in [0,1]."""
    from PIL import Image

    return np.asarray(Image.open(path), dtype=np.float64) / 255.0


def _decode_depth(path: Path) -> np.ndarray:
    """Decode a depth PNG as uint16."""
    from PIL import Image

    return np.asarray(Image.open(path), dtype=np.uint16)


def _rgb_mask_leakage(actual: np.ndarray, baseline: np.ndarray, dilated_mask: np.ndarray) -> float:
    """Max per-channel mean normalized intensity outside the dilated mask.

    Returns the worst (max) of the three per-channel means.
    """
    outside = ~dilated_mask
    if not outside.any():
        return 0.0
    worst = 0.0
    for ch in range(actual.shape[2]):
        diff = np.abs(actual[..., ch] - baseline[..., ch])
        mean_outside = float(diff[outside].mean())
        if mean_outside > worst:
            worst = mean_outside
    return worst


def _rgb_masked_rmse(actual: np.ndarray, baseline: np.ndarray, mask: np.ndarray) -> float:
    """RMSE over masked pixels, all three channels, float64 [0,1]."""
    diff = actual - baseline
    return float(np.sqrt(np.mean(diff[mask] ** 2, dtype=np.float64)))


def _rgb_masked_abs_error_p99(actual: np.ndarray, baseline: np.ndarray, mask: np.ndarray) -> float:
    """p99 of abs error over masked pixels, all three channels, float64 [0,1]."""
    diff = np.abs(actual - baseline)
    return float(np.percentile(diff[mask], 99, method="linear"))


def _depth_non_max_mask(depth: np.ndarray, mask: np.ndarray, sentinel: np.uint16) -> np.ndarray:
    """Boolean mask of non-max (non-sentinel) pixels within the circular mask."""
    return mask & (depth != sentinel)


def _depth_iou(actual_nonmax: np.ndarray, baseline_nonmax: np.ndarray) -> tuple[int, int]:
    """Return (intersection_count, union_count) for IoU."""
    intersection = int(np.count_nonzero(actual_nonmax & baseline_nonmax))
    union = int(np.count_nonzero(actual_nonmax | baseline_nonmax))
    return intersection, union


def _depth_error_percentiles(
    actual: np.ndarray,
    baseline: np.ndarray,
    actual_nonmax: np.ndarray,
    baseline_nonmax: np.ndarray,
    pcts: list[int],
) -> dict[int, float]:
    """Percentiles of abs error on the intersection of non-max masks.

    If the intersection is empty, returns 0.0 for all percentiles (both
    frames agree on having no non-max pixels). If only one side has non-max
    pixels, the intersection is empty but it's a mismatch caught by IoU.
    """
    intersection = actual_nonmax & baseline_nonmax
    if not intersection.any():
        return {p: 0.0 for p in pcts}
    errors = np.abs(actual[intersection].astype(np.float64) - baseline[intersection].astype(np.float64))
    return {p: float(np.percentile(errors, p, method="linear")) for p in pcts}


def _dilate_mask(mask: np.ndarray, dilation_pixels: int) -> np.ndarray:
    """Dilate a boolean mask by dilation_pixels using a square structuring element.

    NumPy-only: no SciPy. For dilation_pixels=0, returns the original mask.
    """
    if dilation_pixels <= 0:
        return mask

    h, w = mask.shape
    padded = np.pad(mask, dilation_pixels, constant_values=False)
    result = np.zeros_like(mask)
    for dy in range(-dilation_pixels, dilation_pixels + 1):
        for dx in range(-dilation_pixels, dilation_pixels + 1):
            result |= padded[dilation_pixels + dy : dilation_pixels + dy + h,
                             dilation_pixels + dx : dilation_pixels + dx + w]
    return result


def compare_golden(
    rendered_dir: Path,
    trajectory_dir: Path,
    baseline_dir: Path,
    *,
    tolerance_policy: Path | None = None,
    run_provenance: Path | None = None,
    baseline_provenance: Path | None = None,
) -> dict[str, Any]:
    """Run validate, then tolerant RGB/depth metrics on selected frames."""
    # Validate candidate.
    val = validate(rendered_dir, trajectory_dir)
    if not val["eligible"]:
        return {
            "eligible": False,
            "errors": ["validate failed"] + val["errors"],
            "warnings": [],
            "metrics": {},
        }

    # Validate baseline.
    base_val = validate(baseline_dir, trajectory_dir)
    if not base_val["eligible"]:
        return {
            "eligible": False,
            "errors": ["baseline validate failed"] + base_val["errors"],
            "warnings": [],
            "metrics": {},
        }

    errors: list[str] = []
    metrics: dict[str, Any] = {}

    # Load tolerance policy if provided.
    policy = None
    if tolerance_policy:
        try:
            with tolerance_policy.open() as f:
                policy = json.load(f)
        except Exception as e:
            errors.append(f"tolerance policy load failed: {e}")

    # Load render summaries for mask construction.
    depth_summary = parse_render_summary(rendered_dir, "depth_render_summary.json")

    width, height = depth_summary["resolution"]
    cx, cy = depth_summary.get("principal_point", [width / 2.0, height / 2.0])
    radius = depth_summary["forward_mask_radius_pixels"]
    mask = build_circular_mask(width, height, cx, cy, radius)

    sentinel = encoded_depth_sentinel(depth_summary["max_depth_m"], depth_summary["depth_scale"])

    dilation_pixels = 0
    if policy is not None:
        dilation_pixels = policy.get("rgb_mask_dilation_pixels", 0)
    dilated_mask = _dilate_mask(mask, dilation_pixels)

    # Selected frames: first/middle/last per episode, de-duplicated.
    manifest = parse_trajectory_manifest(trajectory_dir)
    selected = []
    for ep in manifest["episodes"]:
        for fi in _selected_frame_indices(ep["frame_count"]):
            selected.append((ep["episode_index"], fi))

    # Default thresholds: exact match (zero error, perfect IoU).
    # When a tolerance policy provides explicit thresholds, those override.
    default_thresholds = {
        "rgb_mask_leakage_mean_max": 0.0,
        "rgb_masked_rmse": 0.0,
        "rgb_masked_abs_error_p99": 0.0,
        "depth_non_max_mask_iou": 1.0,
        "depth_error_p50": 0.0,
        "depth_error_p95": 0.0,
        "depth_error_p99": 0.0,
    }
    thresholds = default_thresholds
    if policy is not None and "thresholds" in policy:
        thresholds = policy["thresholds"]

    per_frame_metrics: list[dict] = []

    for ep_idx, frame_idx in selected:
        stem = f"episode_{ep_idx:06d}_{frame_idx:03d}"
        fm: dict[str, Any] = {"episode": ep_idx, "frame": frame_idx}

        # RGB metrics.
        rgb_path = rendered_dir / "observation.images.rgb" / f"{stem}.jpg"
        base_rgb_path = baseline_dir / "observation.images.rgb" / f"{stem}.jpg"
        if rgb_path.is_file() and base_rgb_path.is_file():
            actual_rgb = _decode_rgb(rgb_path)
            baseline_rgb = _decode_rgb(base_rgb_path)
            if actual_rgb.shape != baseline_rgb.shape:
                errors.append(f"frame {stem} RGB shape mismatch")
            else:
                leakage = _rgb_mask_leakage(actual_rgb, baseline_rgb, dilated_mask)
                rmse = _rgb_masked_rmse(actual_rgb, baseline_rgb, mask)
                p99 = _rgb_masked_abs_error_p99(actual_rgb, baseline_rgb, mask)
                fm.update({
                    "rgb_mask_leakage_mean": leakage,
                    "rgb_masked_rmse": rmse,
                    "rgb_masked_abs_error_p99": p99,
                })

                # Check thresholds if available.
                for metric_name, value in [
                    ("rgb_mask_leakage_mean_max", leakage),
                    ("rgb_masked_rmse", rmse),
                    ("rgb_masked_abs_error_p99", p99),
                ]:
                    if metric_name in thresholds:
                        if value > thresholds[metric_name]:
                            errors.append(f"frame {stem} {metric_name}={value:.6f} > {thresholds[metric_name]}")

        # Depth metrics.
        depth_path = rendered_dir / "observation.images.depth" / f"{stem}.png"
        base_depth_path = baseline_dir / "observation.images.depth" / f"{stem}.png"
        if depth_path.is_file() and base_depth_path.is_file():
            actual_depth = _decode_depth(depth_path)
            baseline_depth = _decode_depth(base_depth_path)
            if actual_depth.shape != baseline_depth.shape:
                errors.append(f"frame {stem} depth shape mismatch")
            else:
                actual_nonmax = _depth_non_max_mask(actual_depth, mask, sentinel)
                baseline_nonmax = _depth_non_max_mask(baseline_depth, mask, sentinel)
                intersection, union = _depth_iou(actual_nonmax, baseline_nonmax)
                if union == 0:
                    # Both frames have no non-max pixels. This is a match
                    # (both all-sentinel inside mask), not an error.
                    iou = 1.0
                elif intersection == 0:
                    iou = 0.0
                    errors.append(f"frame {stem} depth IoU empty intersection")
                else:
                    iou = intersection / union
                pct_results = _depth_error_percentiles(
                    actual_depth, baseline_depth, actual_nonmax, baseline_nonmax, [50, 95, 99]
                )
                fm.update({
                    "depth_non_max_mask_iou": iou,
                    "depth_error_p50": pct_results[50],
                    "depth_error_p95": pct_results[95],
                    "depth_error_p99": pct_results[99],
                })

                for metric_name, value in [
                    ("depth_non_max_mask_iou", iou),
                    ("depth_error_p50", pct_results[50]),
                    ("depth_error_p95", pct_results[95]),
                    ("depth_error_p99", pct_results[99]),
                ]:
                    if metric_name in thresholds:
                        # IoU is a lower bound; others are upper bounds.
                        if metric_name == "depth_non_max_mask_iou":
                            if value < thresholds[metric_name]:
                                errors.append(f"frame {stem} {metric_name}={value:.6f} < {thresholds[metric_name]}")
                        else:
                            if value > thresholds[metric_name]:
                                errors.append(f"frame {stem} {metric_name}={value:.6f} > {thresholds[metric_name]}")

        per_frame_metrics.append(fm)

    metrics["per_frame"] = per_frame_metrics

    # Provenance binding.
    if run_provenance and baseline_provenance:
        try:
            with run_provenance.open() as f:
                rp = json.load(f)
            with baseline_provenance.open() as f:
                bp = json.load(f)
            for field in ("plan_revision", "baseline_id"):
                if rp.get(field) != bp.get(field):
                    errors.append(f"provenance {field} mismatch: {rp.get(field)} != {bp.get(field)}")
        except Exception as e:
            errors.append(f"provenance load failed: {e}")

    eligible = len(errors) == 0
    return {
        "eligible": eligible,
        "errors": errors,
        "warnings": [],
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAGE3D render artifact checker")
    sub = p.add_subparsers(dest="mode", required=True)

    pv = sub.add_parser("validate", help="Baseline-independent validation")
    pv.add_argument("--rendered-dir", type=Path, required=True)
    pv.add_argument("--trajectory-dir", type=Path, required=True)
    pv.add_argument("--result-path", type=Path, default=None)

    pg = sub.add_parser("compare-golden", help="Compare against a golden baseline")
    pg.add_argument("--rendered-dir", type=Path, required=True)
    pg.add_argument("--trajectory-dir", type=Path, required=True)
    pg.add_argument("--baseline-dir", type=Path, required=True)
    pg.add_argument("--baseline-provenance", type=Path, default=None)
    pg.add_argument("--run-provenance", type=Path, default=None)
    pg.add_argument("--tolerance-policy", type=Path, default=None)
    pg.add_argument("--result-path", type=Path, default=None)

    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.mode == "validate":
        result = validate(args.rendered_dir, args.trajectory_dir)
    elif args.mode == "compare-golden":
        result = compare_golden(
            args.rendered_dir,
            args.trajectory_dir,
            args.baseline_dir,
            tolerance_policy=args.tolerance_policy,
            run_provenance=args.run_provenance,
            baseline_provenance=args.baseline_provenance,
        )
    else:
        print(f"unknown mode: {args.mode}", file=sys.stderr)
        return 2

    result["checker"] = "check_render"
    result["mode"] = args.mode

    if args.result_path:
        _atomic_write_json(args.result_path, result)

    status = "ELIGIBLE" if result["eligible"] else "INELIGIBLE"
    print(f"[check_render:{args.mode}] {status}")
    for err in result.get("errors", []):
        print(f"  ERROR: {err}")

    return 0 if result["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""Read-only package-Python checker for rendered RGB/depth artifacts.

Two modes per the plan (Checker execution contract, revision 8):

- ``validate``: baseline-independent inventory, calibration, trajectory
  linkage, encoded-depth structure, and summary shapes/ranges/counts.
- ``compare-golden``: runs validate, then tolerant RGB/depth metrics on
  selected frames (first/middle/last per episode, de-duplicated).

Phase 1 rewired the sentinel and forward mask to
``sage3d.render_processing``; no independent sentinel formula remains.

Package-safe: stdlib + numpy + PIL. No forbidden imports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make the repo root importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sage3d.render_processing import (  # noqa: E402
    build_forward_mask as build_circular_mask,
    encoded_depth_sentinel,
)
from sage3d_canonical.digest import digest_directory  # noqa: E402
from sage3d_canonical.parsers import parse_render_summary, parse_trajectory_manifest  # noqa: E402
from sage3d_canonical.provenance import _atomic_write_json  # noqa: E402


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

    # Encoded-depth structure: every frame must preserve dtype, shape, and the
    # outside-mask sentinel. Sampling here would let a single stale/corrupt
    # frame escape the canonical mutation gate.
    if depth_summary is not None:
        from PIL import Image

        width, height = depth_summary["resolution"]
        cx, cy = depth_summary.get("principal_point", [width / 2.0, height / 2.0])
        radius = depth_summary["forward_mask_radius_pixels"]
        mask = build_circular_mask(width, height, cx, cy, radius)
        sentinel = encoded_depth_sentinel(depth_summary["max_depth_m"], depth_summary["depth_scale"])

        depth_files_check = sorted(depth_dir.glob("*.png"))
        for depth_path in depth_files_check:
            depth = np.array(Image.open(depth_path))
            if depth.dtype != np.uint16:
                errors.append(
                    f"{depth_path.name} dtype {depth.dtype} != uint16"
                )
                break
            if depth.shape != (height, width):
                errors.append(
                    f"{depth_path.name} shape {depth.shape} "
                    f"!= ({height}, {width})"
                )
                break
            outside = depth[~mask]
            if outside.size > 0 and not np.all(outside == sentinel):
                errors.append(
                    f"{depth_path.name} outside-mask pixels != sentinel "
                    f"{sentinel}; got unique={np.unique(outside)[:5]}"
                )
                break

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


def _rgb_mask_leakage(actual: np.ndarray, dilated_mask: np.ndarray) -> float:
    """Max per-channel mean normalized intensity outside the dilated mask."""
    outside = ~dilated_mask
    if not outside.any():
        return 0.0
    return max(
        float(actual[..., channel][outside].mean())
        for channel in range(actual.shape[2])
    )


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
    threshold_report: Path | None = None,
    run_provenance: Path | None = None,
    baseline_provenance: Path | None = None,
    enforce_thresholds: bool = True,
    include_all_frames: bool = False,
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

    # Load the immutable pre-observation policy and separately derived report.
    policy = None
    if tolerance_policy:
        try:
            with tolerance_policy.open() as f:
                policy = json.load(f)
        except Exception as e:
            errors.append(f"tolerance policy load failed: {e}")
    derived = None
    if threshold_report:
        try:
            with threshold_report.open() as f:
                derived = json.load(f)
            if policy and derived.get("baseline_id") != policy.get("baseline_id"):
                errors.append("threshold report baseline_id does not match policy")
        except Exception as e:
            errors.append(f"threshold report load failed: {e}")

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

    # Exact-match defaults preserve the original checker API. Canonical
    # characterization uses enforce_thresholds=False; held-outs consume the
    # separately derived threshold report.
    default_thresholds = {
        "rgb_masked_rmse": 0.0,
        "rgb_masked_abs_error_p99": 0.0,
        "depth_non_max_mask_iou": 1.0,
        "depth_error_p50": 0.0,
        "depth_error_p95": 0.0,
        "depth_error_p99": 0.0,
    }
    thresholds: dict[str, float] = default_thresholds if enforce_thresholds else {}
    if derived is not None:
        thresholds = derived["thresholds"]
    elif policy is not None and "thresholds" in policy:
        thresholds = policy["thresholds"]

    def measure_frame(
        ep_idx: int,
        frame_idx: int,
        *,
        check_thresholds: bool,
    ) -> tuple[dict[str, Any], list[str]]:
        stem = f"episode_{ep_idx:06d}_{frame_idx:03d}"
        fm: dict[str, Any] = {"episode": ep_idx, "frame": frame_idx}
        frame_errors: list[str] = []

        # RGB metrics.
        rgb_path = rendered_dir / "observation.images.rgb" / f"{stem}.jpg"
        base_rgb_path = baseline_dir / "observation.images.rgb" / f"{stem}.jpg"
        if rgb_path.is_file() and base_rgb_path.is_file():
            actual_rgb = _decode_rgb(rgb_path)
            baseline_rgb = _decode_rgb(base_rgb_path)
            if actual_rgb.shape != baseline_rgb.shape:
                frame_errors.append(f"frame {stem} RGB shape mismatch")
            else:
                leakage = _rgb_mask_leakage(actual_rgb, dilated_mask)
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
                    if check_thresholds and metric_name in thresholds:
                        if value > thresholds[metric_name]:
                            frame_errors.append(
                                f"frame {stem} {metric_name}={value:.6f} "
                                f"> {thresholds[metric_name]}"
                            )
        else:
            frame_errors.append(f"frame {stem} RGB file missing")

        # Depth metrics.
        depth_path = rendered_dir / "observation.images.depth" / f"{stem}.png"
        base_depth_path = baseline_dir / "observation.images.depth" / f"{stem}.png"
        if depth_path.is_file() and base_depth_path.is_file():
            actual_depth = _decode_depth(depth_path)
            baseline_depth = _decode_depth(base_depth_path)
            if actual_depth.shape != baseline_depth.shape:
                frame_errors.append(f"frame {stem} depth shape mismatch")
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
                    frame_errors.append(f"frame {stem} depth IoU empty intersection")
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
                    if check_thresholds and metric_name in thresholds:
                        # IoU is a lower bound; others are upper bounds.
                        if metric_name == "depth_non_max_mask_iou":
                            if value < thresholds[metric_name]:
                                frame_errors.append(
                                    f"frame {stem} {metric_name}={value:.6f} "
                                    f"< {thresholds[metric_name]}"
                                )
                        else:
                            if value > thresholds[metric_name]:
                                frame_errors.append(
                                    f"frame {stem} {metric_name}={value:.6f} "
                                    f"> {thresholds[metric_name]}"
                                )
        else:
            frame_errors.append(f"frame {stem} depth file missing")

        return fm, frame_errors

    per_frame_metrics: list[dict] = []
    for ep_idx, frame_idx in selected:
        fm, frame_errors = measure_frame(
            ep_idx, frame_idx, check_thresholds=True
        )
        per_frame_metrics.append(fm)
        errors.extend(frame_errors)

    metrics["per_frame"] = per_frame_metrics
    if include_all_frames:
        all_frame_metrics = []
        for ep in manifest["episodes"]:
            for frame_idx in range(ep["frame_count"]):
                fm, frame_errors = measure_frame(
                    ep["episode_index"],
                    frame_idx,
                    check_thresholds=False,
                )
                all_frame_metrics.append(fm)
                errors.extend(frame_errors)
        distributions = {}
        frame_metric_names = (
            "rgb_mask_leakage_mean",
            "rgb_masked_rmse",
            "rgb_masked_abs_error_p99",
            "depth_non_max_mask_iou",
            "depth_error_p50",
            "depth_error_p95",
            "depth_error_p99",
        )
        for name in frame_metric_names:
            values = [float(frame[name]) for frame in all_frame_metrics if name in frame]
            distributions[name] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": float(np.mean(values, dtype=np.float64)),
            }
        metrics["all_frame_distributions"] = distributions

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
        "artifact_digests": {
            "rendered_root": digest_directory("rendered_root", rendered_dir),
            "trajectory_root": digest_directory("trajectory", trajectory_dir),
        },
        "thresholds_applied": enforce_thresholds,
        "binding": enforce_thresholds,
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
    pg.add_argument("--threshold-report", type=Path, default=None)
    pg.add_argument("--all-frames", action="store_true")
    pg.add_argument("--result-path", type=Path, default=None)

    pm = sub.add_parser(
        "measure-golden",
        help="Measure candidate against baseline without applying thresholds",
    )
    pm.add_argument("--rendered-dir", type=Path, required=True)
    pm.add_argument("--trajectory-dir", type=Path, required=True)
    pm.add_argument("--baseline-dir", type=Path, required=True)
    pm.add_argument("--baseline-provenance", type=Path, default=None)
    pm.add_argument("--run-provenance", type=Path, default=None)
    pm.add_argument("--tolerance-policy", type=Path, default=None)
    pm.add_argument("--all-frames", action="store_true")
    pm.add_argument("--result-path", type=Path, default=None)

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
            threshold_report=args.threshold_report,
            run_provenance=args.run_provenance,
            baseline_provenance=args.baseline_provenance,
            include_all_frames=args.all_frames,
        )
    elif args.mode == "measure-golden":
        result = compare_golden(
            args.rendered_dir,
            args.trajectory_dir,
            args.baseline_dir,
            tolerance_policy=args.tolerance_policy,
            run_provenance=args.run_provenance,
            baseline_provenance=args.baseline_provenance,
            enforce_thresholds=False,
            include_all_frames=args.all_frames,
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

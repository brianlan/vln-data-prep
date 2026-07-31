"""Tests for scripts/check_render.py validate and compare-golden modes.

Covers the Phase 0b render artifact checker: inventory/calibration/depth
structure validation, tolerant RGB/depth metrics on selected frames, depth
sentinel helper, mutation detection, and provenance binding.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fixtures import build_rendered_dir, build_trajectory_dir

# Make the checker importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKER_DIR = _REPO_ROOT / "scripts"
if str(_CHECKER_DIR) not in sys.path:
    sys.path.insert(0, str(_CHECKER_DIR))
_TESTS_DIR = _REPO_ROOT / "tests" / "package_safe"
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import check_render


# --- helpers -----------------------------------------------------------------

def _make_pair(tmp_path: Path, frame_counts=(3, 5), depth_fill="gradient"):
    """Build trajectory + rendered dirs, return (rendered, trajectory)."""
    traj = tmp_path / "traj"
    manifest = build_trajectory_dir(traj, episode_frame_counts=frame_counts)
    rendered = tmp_path / "rendered"
    build_rendered_dir(rendered, trajectory_manifest=manifest, depth_fill=depth_fill)
    return rendered, traj, manifest


# --- depth sentinel helper ---------------------------------------------------

def test_sentinel_matches_encoder_formula():
    sentinel = check_render.encoded_depth_sentinel(6.0, 10000.0)
    assert sentinel == np.uint16(60000)


def test_sentinel_half_step_rounding():
    # 6.00005 * 10000 = 60000.5 -> np.rint rounds to 60000 (banker's rounding)
    sentinel = check_render.encoded_depth_sentinel(6.00005, 10000.0)
    assert sentinel == np.uint16(60000) or sentinel == np.uint16(60001)


def test_sentinel_overflow_rejected():
    with pytest.raises(ValueError, match="exceeds 65535"):
        check_render.encoded_depth_sentinel(7.0, 10000.0)


def test_sentinel_non_finite_rejected():
    with pytest.raises(ValueError):
        check_render.encoded_depth_sentinel(float("inf"), 10000.0)
    with pytest.raises(ValueError):
        check_render.encoded_depth_sentinel(6.0, float("nan"))


def test_sentinel_non_positive_rejected():
    with pytest.raises(ValueError):
        check_render.encoded_depth_sentinel(-1.0, 10000.0)
    with pytest.raises(ValueError):
        check_render.encoded_depth_sentinel(6.0, 0.0)


# --- validate: positive ------------------------------------------------------

def test_validate_passes_for_valid_rendered(tmp_path):
    rendered, traj, _ = _make_pair(tmp_path)
    result = check_render.validate(rendered, traj)
    assert result["eligible"] is True
    assert result["errors"] == []
    assert result["scene_id"] == "839920"
    assert result["episode_count"] == 2


def test_validate_cli_passes(tmp_path):
    rendered, traj, _ = _make_pair(tmp_path)
    result_path = tmp_path / "result.json"
    proc = subprocess.run(
        [sys.executable, str(_CHECKER_DIR / "check_render.py"),
         "validate", "--rendered-dir", str(rendered),
         "--trajectory-dir", str(traj),
         "--result-path", str(result_path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "ELIGIBLE" in proc.stdout
    with result_path.open() as f:
        result = json.load(f)
    assert result["eligible"] is True
    assert result["checker"] == "check_render"


# --- validate: negative ------------------------------------------------------

def test_validate_fails_for_missing_rgb_summary(tmp_path):
    rendered, traj, _ = _make_pair(tmp_path)
    (rendered / "rgb_render_summary.json").unlink()
    result = check_render.validate(rendered, traj)
    assert result["eligible"] is False
    assert any("rgb_render_summary" in e for e in result["errors"])


def test_validate_fails_for_missing_depth_summary(tmp_path):
    rendered, traj, _ = _make_pair(tmp_path)
    (rendered / "depth_render_summary.json").unlink()
    result = check_render.validate(rendered, traj)
    assert result["eligible"] is False


def test_validate_fails_for_wrong_rgb_count(tmp_path):
    rendered, traj, _ = _make_pair(tmp_path)
    # Delete one RGB file.
    first_rgb = sorted((rendered / "observation.images.rgb").glob("*.jpg"))[0]
    first_rgb.unlink()
    result = check_render.validate(rendered, traj)
    assert result["eligible"] is False
    assert any("RGB file count" in e for e in result["errors"])


def test_validate_fails_for_wrong_depth_count(tmp_path):
    rendered, traj, _ = _make_pair(tmp_path)
    # Delete one depth file.
    first_depth = sorted((rendered / "observation.images.depth").glob("*.png"))[0]
    first_depth.unlink()
    result = check_render.validate(rendered, traj)
    assert result["eligible"] is False
    assert any("depth file count" in e for e in result["errors"])


def test_validate_fails_for_scene_id_mismatch(tmp_path):
    rendered, traj, _ = _make_pair(tmp_path)
    # Corrupt the scene_id in rgb summary.
    with (rendered / "rgb_render_summary.json").open() as f:
        summary = json.load(f)
    summary["scene_id"] = "wrong"
    with (rendered / "rgb_render_summary.json").open("w") as f:
        json.dump(summary, f)
    result = check_render.validate(rendered, traj)
    assert result["eligible"] is False
    assert any("scene_id" in e for e in result["errors"])


def test_validate_fails_for_depth_dtype_not_uint16(tmp_path):
    rendered, traj, _ = _make_pair(tmp_path)
    # Overwrite a depth PNG with uint8.
    first_depth = sorted((rendered / "observation.images.depth").glob("*.png"))[0]
    Image.new("L", (600, 450), 0).save(first_depth)
    result = check_render.validate(rendered, traj)
    assert result["eligible"] is False
    assert any("dtype" in e for e in result["errors"])


# --- compare-golden: positive ------------------------------------------------

def test_compare_golden_passes_for_identical_copy(tmp_path):
    import shutil
    rendered, traj, _ = _make_pair(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(rendered, baseline)
    result = check_render.compare_golden(rendered, traj, baseline)
    assert result["eligible"] is True, f"errors: {result['errors']}"
    assert result["errors"] == []


def test_compare_golden_cli_passes(tmp_path):
    import shutil
    rendered, traj, _ = _make_pair(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(rendered, baseline)
    result_path = tmp_path / "result.json"
    proc = subprocess.run(
        [sys.executable, str(_CHECKER_DIR / "check_render.py"),
         "compare-golden", "--rendered-dir", str(rendered),
         "--trajectory-dir", str(traj),
         "--baseline-dir", str(baseline),
         "--result-path", str(result_path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "ELIGIBLE" in proc.stdout


# --- compare-golden: RGB mutations -------------------------------------------

def _mutate_rgb_file(path: Path, mutator):
    img = Image.open(path)
    arr = np.array(img)
    arr = mutator(arr)
    Image.fromarray(arr).save(path, quality=95)


def test_compare_golden_fails_for_all_black_rgb(tmp_path):
    import shutil
    rendered, traj, _ = _make_pair(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(rendered, baseline)
    # Black out the first selected frame RGB.
    rgb_path = rendered / "observation.images.rgb" / "episode_000000_000.jpg"
    _mutate_rgb_file(rgb_path, lambda a: np.zeros_like(a))
    result = check_render.compare_golden(rendered, traj, baseline)
    assert result["eligible"] is False


def test_compare_golden_fails_for_channel_swap(tmp_path):
    import shutil
    rendered, traj, _ = _make_pair(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(rendered, baseline)
    rgb_path = rendered / "observation.images.rgb" / "episode_000000_000.jpg"
    _mutate_rgb_file(rgb_path, lambda a: a[..., [2, 1, 0]])  # swap R<->B
    result = check_render.compare_golden(rendered, traj, baseline)
    assert result["eligible"] is False


def test_compare_golden_fails_for_horizontal_flip(tmp_path):
    import shutil
    rendered, traj, _ = _make_pair(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(rendered, baseline)
    rgb_path = rendered / "observation.images.rgb" / "episode_000000_000.jpg"
    _mutate_rgb_file(rgb_path, lambda a: a[:, ::-1, :])
    result = check_render.compare_golden(rendered, traj, baseline)
    assert result["eligible"] is False


def test_compare_golden_fails_for_vertical_flip(tmp_path):
    import shutil
    rendered, traj, _ = _make_pair(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(rendered, baseline)
    rgb_path = rendered / "observation.images.rgb" / "episode_000000_000.jpg"
    _mutate_rgb_file(rgb_path, lambda a: a[::-1, :, :])
    result = check_render.compare_golden(rendered, traj, baseline)
    assert result["eligible"] is False


def test_compare_golden_fails_for_block_corruption(tmp_path):
    import shutil
    rendered, traj, _ = _make_pair(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(rendered, baseline)
    rgb_path = rendered / "observation.images.rgb" / "episode_000000_000.jpg"

    def corrupt(a):
        # Corrupt a block > 1% of pixels (e.g. 60x60 in 600x450 = ~1.3%).
        a[100:160, 100:160] = 255
        return a

    _mutate_rgb_file(rgb_path, corrupt)
    result = check_render.compare_golden(rendered, traj, baseline)
    assert result["eligible"] is False


def test_compare_golden_fails_for_multi_code_intensity_shift(tmp_path):
    import shutil
    rendered, traj, _ = _make_pair(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(rendered, baseline)
    rgb_path = rendered / "observation.images.rgb" / "episode_000000_000.jpg"

    def shift(a):
        a = np.clip(a.astype(np.int16) + 5, 0, 255).astype(np.uint8)
        return a

    _mutate_rgb_file(rgb_path, shift)
    result = check_render.compare_golden(rendered, traj, baseline)
    assert result["eligible"] is False


# --- compare-golden: depth mutations ----------------------------------------

def _mutate_depth_file(path: Path, mutator):
    arr = np.array(Image.open(path))
    arr = mutator(arr)
    Image.fromarray(arr).save(path)


def test_compare_golden_fails_for_all_sentinel_depth(tmp_path):
    import shutil
    rendered, traj, _ = _make_pair(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(rendered, baseline)
    sentinel_val = int(np.rint(np.float32(6.0) * np.float32(10000.0)))
    depth_path = rendered / "observation.images.depth" / "episode_000000_000.png"
    _mutate_depth_file(depth_path, lambda a: np.full_like(a, sentinel_val))
    result = check_render.compare_golden(rendered, traj, baseline)
    assert result["eligible"] is False
    assert any("IoU" in e for e in result["errors"])


def test_compare_golden_fails_for_wrong_depth_scale(tmp_path):
    import shutil
    rendered, traj, _ = _make_pair(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(rendered, baseline)
    # Halve all depth values (wrong scale).
    depth_path = rendered / "observation.images.depth" / "episode_000000_000.png"
    _mutate_depth_file(depth_path, lambda a: (a // 2).astype(np.uint16))
    result = check_render.compare_golden(rendered, traj, baseline)
    assert result["eligible"] is False


def test_compare_golden_fails_for_depth_horizontal_flip(tmp_path):
    import shutil
    rendered, traj, _ = _make_pair(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(rendered, baseline)
    depth_path = rendered / "observation.images.depth" / "episode_000000_000.png"
    _mutate_depth_file(depth_path, lambda a: a[:, ::-1])
    result = check_render.compare_golden(rendered, traj, baseline)
    assert result["eligible"] is False


def test_compare_golden_fails_for_constant_depth_offset(tmp_path):
    import shutil
    rendered, traj, _ = _make_pair(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(rendered, baseline)
    depth_path = rendered / "observation.images.depth" / "episode_000000_000.png"

    def add_offset(a):
        return np.clip(a.astype(np.int32) + 50, 0, 65535).astype(np.uint16)

    _mutate_depth_file(depth_path, add_offset)
    result = check_render.compare_golden(rendered, traj, baseline)
    assert result["eligible"] is False


def test_compare_golden_fails_for_changed_outside_mask_value(tmp_path):
    import shutil
    rendered, traj, _ = _make_pair(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(rendered, baseline)
    sentinel_val = int(np.rint(np.float32(6.0) * np.float32(10000.0)))
    depth_path = rendered / "observation.images.depth" / "episode_000000_000.png"
    # Change outside-mask pixels from sentinel to something else.
    arr = np.array(Image.open(depth_path))
    arr[arr == sentinel_val] = sentinel_val - 1
    Image.fromarray(arr).save(depth_path)
    result = check_render.compare_golden(rendered, traj, baseline)
    assert result["eligible"] is False


def test_compare_golden_fails_for_one_frame_offset(tmp_path):
    import shutil
    rendered, traj, _ = _make_pair(tmp_path, frame_counts=(5,))
    baseline = tmp_path / "baseline"
    shutil.copytree(rendered, baseline)
    # Swap frame 0 and frame 1 depth files.
    d0 = rendered / "observation.images.depth" / "episode_000000_000.png"
    d1 = rendered / "observation.images.depth" / "episode_000000_001.png"
    arr0 = np.array(Image.open(d0))
    arr1 = np.array(Image.open(d1))
    Image.fromarray(arr1).save(d0)
    Image.fromarray(arr0).save(d1)
    result = check_render.compare_golden(rendered, traj, baseline)
    assert result["eligible"] is False


# --- compare-golden: with tolerance policy -----------------------------------

def test_compare_golden_with_tolerance_policy_passes(tmp_path):
    import shutil
    rendered, traj, _ = _make_pair(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(rendered, baseline)
    policy = {
        "schema_version": 1,
        "baseline_id": "test",
        "rgb_mask_dilation_pixels": 0,
        "thresholds": {
            "rgb_mask_leakage_mean_max": 1.0,
            "rgb_masked_rmse": 1.0,
            "rgb_masked_abs_error_p99": 1.0,
            "depth_non_max_mask_iou": 0.0,
            "depth_error_p50": 65535.0,
            "depth_error_p95": 65535.0,
            "depth_error_p99": 65535.0,
        },
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    result = check_render.compare_golden(
        rendered, traj, baseline, tolerance_policy=policy_path
    )
    assert result["eligible"] is True, f"errors: {result['errors']}"


def test_compare_golden_with_strict_threshold_fails_for_mutation(tmp_path):
    """Very strict RMSE threshold should fail when there's any RGB difference."""
    import shutil
    rendered, traj, _ = _make_pair(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(rendered, baseline)
    # Mutate one RGB frame slightly.
    rgb_path = rendered / "observation.images.rgb" / "episode_000000_000.jpg"
    _mutate_rgb_file(rgb_path, lambda a: np.clip(a.astype(np.int16) + 3, 0, 255).astype(np.uint8))
    policy = {
        "thresholds": {
            "rgb_masked_rmse": 0.0,  # exact match required
        },
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    result = check_render.compare_golden(
        rendered, traj, baseline, tolerance_policy=policy_path
    )
    assert result["eligible"] is False


def test_measure_golden_reports_differences_without_threshold_failure(tmp_path):
    import shutil
    rendered, traj, _ = _make_pair(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(rendered, baseline)
    rgb_path = rendered / "observation.images.rgb" / "episode_000000_000.jpg"
    _mutate_rgb_file(
        rgb_path,
        lambda a: np.clip(a.astype(np.int16) + 3, 0, 255).astype(np.uint8),
    )
    result = check_render.compare_golden(
        rendered,
        traj,
        baseline,
        enforce_thresholds=False,
        include_all_frames=True,
    )
    assert result["eligible"] is True
    assert result["binding"] is False
    assert result["metrics"]["per_frame"][0]["rgb_masked_rmse"] > 0
    assert result["metrics"]["all_frame_distributions"]["rgb_masked_rmse"]["count"] == 8


def test_leakage_is_absolute_outside_mask_intensity():
    actual = np.zeros((4, 4, 3), dtype=np.float64)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    actual[~mask, 1] = 0.25
    assert check_render._rgb_mask_leakage(actual, mask) == pytest.approx(0.25)


def test_threshold_report_is_applied_separately_from_policy(tmp_path):
    import shutil
    rendered, traj, _ = _make_pair(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(rendered, baseline)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({"baseline_id": "b", "rgb_mask_dilation_pixels": 0}))
    report_path = tmp_path / "threshold_report.json"
    report_path.write_text(json.dumps({
        "baseline_id": "b",
        "thresholds": {
            "rgb_mask_leakage_mean_max": 1.0,
            "rgb_masked_rmse": 1.0,
            "rgb_masked_abs_error_p99": 1.0,
            "depth_non_max_mask_iou": 0.0,
            "depth_error_p50": 65535.0,
            "depth_error_p95": 65535.0,
            "depth_error_p99": 65535.0,
        },
    }))
    result = check_render.compare_golden(
        rendered,
        traj,
        baseline,
        tolerance_policy=policy_path,
        threshold_report=report_path,
    )
    assert result["eligible"] is True


# --- provenance binding ------------------------------------------------------

def test_compare_golden_fails_for_provenance_mismatch(tmp_path):
    import shutil
    rendered, traj, _ = _make_pair(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(rendered, baseline)
    rp = tmp_path / "rp.json"
    bp = tmp_path / "bp.json"
    rp.write_text(json.dumps({"plan_revision": 8, "baseline_id": "b1"}))
    bp.write_text(json.dumps({"plan_revision": 7, "baseline_id": "b1"}))
    result = check_render.compare_golden(
        rendered, traj, baseline, run_provenance=rp, baseline_provenance=bp
    )
    assert result["eligible"] is False
    assert any("plan_revision" in e for e in result["errors"])


def test_compare_golden_passes_with_matching_provenance(tmp_path):
    import shutil
    rendered, traj, _ = _make_pair(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(rendered, baseline)
    rp = tmp_path / "rp.json"
    bp = tmp_path / "bp.json"
    rp.write_text(json.dumps({"plan_revision": 8, "baseline_id": "b1"}))
    bp.write_text(json.dumps({"plan_revision": 8, "baseline_id": "b1"}))
    result = check_render.compare_golden(
        rendered, traj, baseline, run_provenance=rp, baseline_provenance=bp
    )
    assert result["eligible"] is True

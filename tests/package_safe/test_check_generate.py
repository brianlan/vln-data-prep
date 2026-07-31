"""Tests for scripts/check_generate.py validate and compare-golden modes.

Covers the Phase 0b generation artifact checker: schema/inventory validation,
cross-artifact consistency, exact array/manifest/PLY/viz comparison, path
normalization (scene_dir/collision_usd only), and negative mutation detection.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fixtures import build_trajectory_dir

# Make the checker importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKER_DIR = _REPO_ROOT / "scripts"
if str(_CHECKER_DIR) not in sys.path:
    sys.path.insert(0, str(_CHECKER_DIR))
_TESTS_DIR = _REPO_ROOT / "tests" / "package_safe"
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import check_generate


# --- helpers -----------------------------------------------------------------

def _make_trajectory(tmp_path: Path, frame_counts=(3, 5)) -> Path:
    traj = tmp_path / "traj"
    build_trajectory_dir(traj, episode_frame_counts=frame_counts)
    return traj


def _mutate_manifest(traj: Path, mutator):
    path = traj / "trajectory_manifest.json"
    with path.open() as f:
        manifest = json.load(f)
    mutator(manifest)
    with path.open("w") as f:
        json.dump(manifest, f, indent=2)


def _mutate_npz(traj: Path, episode_idx: int, mutator):
    path = traj / f"episode_{episode_idx:06d}.npz"
    data = np.load(path)
    arrays = {key: data[key] for key in data.files}
    mutator(arrays)
    np.savez_compressed(path, **arrays)


# --- validate: positive ------------------------------------------------------

def test_validate_passes_for_valid_trajectory(tmp_path):
    traj = _make_trajectory(tmp_path)
    result = check_generate.validate(traj)
    assert result["eligible"] is True
    assert result["episode_count"] == 2
    assert result["scene_id"] == "839920"
    assert result["errors"] == []


def test_validate_cli_passes(tmp_path):
    traj = _make_trajectory(tmp_path)
    result_path = tmp_path / "result.json"
    proc = subprocess.run(
        [sys.executable, str(_CHECKER_DIR / "check_generate.py"),
         "validate", "--trajectory-dir", str(traj),
         "--result-path", str(result_path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "ELIGIBLE" in proc.stdout
    with result_path.open() as f:
        result = json.load(f)
    assert result["eligible"] is True
    assert result["checker"] == "check_generate"
    assert result["mode"] == "validate"


# --- validate: negative ------------------------------------------------------

def test_validate_fails_for_missing_manifest(tmp_path):
    traj = tmp_path / "traj"
    traj.mkdir()
    result = check_generate.validate(traj)
    assert result["eligible"] is False
    assert len(result["errors"]) == 1


def test_validate_fails_for_missing_npz_keys(tmp_path):
    traj = _make_trajectory(tmp_path, (3,))
    _mutate_npz(traj, 0, lambda a: a.pop("yaw"))
    result = check_generate.validate(traj)
    assert result["eligible"] is False
    assert any("npz keys" in e for e in result["errors"])


def test_validate_fails_for_frame_count_mismatch(tmp_path):
    traj = _make_trajectory(tmp_path, (3,))
    # Truncate ALL time-series arrays to 2 frames so parse_episode_npz passes
    # internal consistency, but manifest still says frame_count=3.
    _mutate_npz(traj, 0, lambda a: a.update(
        points=a["points"][:-1],
        actions=a["actions"][:-1],
        camera_positions=a["camera_positions"][:-1],
        yaw=a["yaw"][:-1],
        point_goal=a["point_goal"][:-1],
    ))
    result = check_generate.validate(traj)
    assert result["eligible"] is False
    assert any("frame_count" in e for e in result["errors"])


def test_validate_fails_for_missing_ply(tmp_path):
    traj = _make_trajectory(tmp_path, (3,))
    (traj / "pointcloud.ply").unlink()
    result = check_generate.validate(traj)
    assert result["eligible"] is False
    assert any("PLY" in e for e in result["errors"])


def test_validate_fails_for_missing_viz(tmp_path):
    traj = _make_trajectory(tmp_path, (3,))
    (traj / "navigation_map.png").unlink()
    result = check_generate.validate(traj)
    assert result["eligible"] is False
    assert any("navigation_map" in e for e in result["errors"])


def test_validate_fails_for_episode_count_mismatch(tmp_path):
    traj = _make_trajectory(tmp_path, (3,))
    _mutate_manifest(traj, lambda m: m.update(episode_count=99))
    result = check_generate.validate(traj)
    assert result["eligible"] is False


def test_validate_fails_for_missing_manifest_key(tmp_path):
    traj = _make_trajectory(tmp_path, (3,))
    _mutate_manifest(traj, lambda m: m.pop("seed"))
    result = check_generate.validate(traj)
    assert result["eligible"] is False


# --- compare-golden: positive ------------------------------------------------

def test_compare_golden_passes_for_identical_copy(tmp_path):
    traj = _make_trajectory(tmp_path, (3, 5))
    import shutil
    baseline = tmp_path / "baseline"
    shutil.copytree(traj, baseline)
    result = check_generate.compare_golden(traj, baseline)
    assert result["eligible"] is True
    assert result["errors"] == []
    assert "manifest" in result["artifact_digests"]
    assert "pointcloud_ply" in result["artifact_digests"]
    assert "episode_000000" in result["artifact_digests"]
    assert "episode_000001" in result["artifact_digests"]


def test_compare_golden_cli_passes(tmp_path):
    traj = _make_trajectory(tmp_path, (3, 5))
    import shutil
    baseline = tmp_path / "baseline"
    shutil.copytree(traj, baseline)
    result_path = tmp_path / "result.json"
    proc = subprocess.run(
        [sys.executable, str(_CHECKER_DIR / "check_generate.py"),
         "compare-golden", "--trajectory-dir", str(traj),
         "--baseline-dir", str(baseline),
         "--result-path", str(result_path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "ELIGIBLE" in proc.stdout
    with result_path.open() as f:
        result = json.load(f)
    assert result["eligible"] is True


# --- compare-golden: negative (mutation detection) ---------------------------

def test_compare_golden_fails_for_array_value_change(tmp_path):
    traj = _make_trajectory(tmp_path, (3,))
    import shutil
    baseline = tmp_path / "baseline"
    shutil.copytree(traj, baseline)
    # Mutate candidate array.
    _mutate_npz(traj, 0, lambda a: a.update(actions=a["actions"] + 0.001))
    result = check_generate.compare_golden(traj, baseline)
    assert result["eligible"] is False
    assert any("array digest" in e for e in result["errors"])


def test_compare_golden_fails_for_array_shape_change(tmp_path):
    traj = _make_trajectory(tmp_path, (3,))
    import shutil
    baseline = tmp_path / "baseline"
    shutil.copytree(traj, baseline)
    _mutate_npz(traj, 0, lambda a: a.update(actions=a["actions"][:-1]))
    result = check_generate.compare_golden(traj, baseline)
    assert result["eligible"] is False
    # Truncating only actions breaks internal consistency (parse_episode_npz
    # catches it), so validate fails before reaching array digest comparison.
    assert result["errors"]


def test_compare_golden_fails_for_array_dtype_change(tmp_path):
    traj = _make_trajectory(tmp_path, (3,))
    import shutil
    baseline = tmp_path / "baseline"
    shutil.copytree(traj, baseline)
    _mutate_npz(traj, 0, lambda a: a.update(actions=a["actions"].astype("float64")))
    result = check_generate.compare_golden(traj, baseline)
    assert result["eligible"] is False
    assert any("array digest" in e for e in result["errors"])


def test_compare_golden_fails_for_ply_bytes_change(tmp_path):
    traj = _make_trajectory(tmp_path, (3,))
    import shutil
    baseline = tmp_path / "baseline"
    shutil.copytree(traj, baseline)
    # Corrupt candidate PLY.
    with (traj / "pointcloud.ply").open("ab") as f:
        f.write(b"\x00\x00\x00")
    result = check_generate.compare_golden(traj, baseline)
    assert result["eligible"] is False
    assert any("PLY" in e for e in result["errors"])


def test_compare_golden_fails_for_viz_pixel_change(tmp_path):
    traj = _make_trajectory(tmp_path, (3,))
    import shutil
    baseline = tmp_path / "baseline"
    shutil.copytree(traj, baseline)
    # Overwrite candidate viz with different pixels.
    from PIL import Image
    Image.new("RGB", (8, 8), (200, 200, 200)).save(traj / "navigation_map.png")
    result = check_generate.compare_golden(traj, baseline)
    assert result["eligible"] is False
    assert any("viz" in e for e in result["errors"])


def test_compare_golden_fails_for_non_path_manifest_change(tmp_path):
    traj = _make_trajectory(tmp_path, (3,))
    import shutil
    baseline = tmp_path / "baseline"
    shutil.copytree(traj, baseline)
    # Change a non-path field in candidate manifest.
    _mutate_manifest(traj, lambda m: m.update(seed=999))
    result = check_generate.compare_golden(traj, baseline)
    assert result["eligible"] is False
    assert any("manifest" in e.lower() for e in result["errors"])


def test_compare_golden_fails_for_episode_order_change(tmp_path):
    traj = _make_trajectory(tmp_path, (3, 5))
    import shutil
    baseline = tmp_path / "baseline"
    shutil.copytree(traj, baseline)
    # Swap episode order in candidate manifest.
    _mutate_manifest(traj, lambda m: m.update(episodes=list(reversed(m["episodes"]))))
    result = check_generate.compare_golden(traj, baseline)
    assert result["eligible"] is False


# --- compare-golden: path normalization invariance ---------------------------

def test_compare_golden_invariant_to_scene_dir_path_spelling(tmp_path):
    traj = _make_trajectory(tmp_path, (3,))
    import shutil
    baseline = tmp_path / "baseline"
    shutil.copytree(traj, baseline)
    # Change candidate scene_dir trailing slash and backslashes.
    _mutate_manifest(traj, lambda m: m.update(
        scene_dir=m["scene_dir"] + "/",
        collision_usd=m["collision_usd"].replace("/", "\\"),
    ))
    result = check_generate.compare_golden(traj, baseline)
    # Should pass because the paths are normalized.
    assert result["eligible"] is True, f"errors: {result['errors']}"


# --- compare-golden: provenance binding --------------------------------------

def test_compare_golden_fails_for_provenance_plan_revision_mismatch(tmp_path):
    traj = _make_trajectory(tmp_path, (3,))
    import shutil
    baseline = tmp_path / "baseline"
    shutil.copytree(traj, baseline)
    rp = tmp_path / "run_provenance.json"
    bp = tmp_path / "baseline_provenance.json"
    rp.write_text(json.dumps({"plan_revision": 8, "baseline_id": "b1"}))
    bp.write_text(json.dumps({"plan_revision": 7, "baseline_id": "b1"}))
    result = check_generate.compare_golden(
        traj, baseline, run_provenance=rp, baseline_provenance=bp
    )
    assert result["eligible"] is False
    assert any("plan_revision" in e for e in result["errors"])


def test_compare_golden_passes_with_matching_provenance(tmp_path):
    traj = _make_trajectory(tmp_path, (3,))
    import shutil
    baseline = tmp_path / "baseline"
    shutil.copytree(traj, baseline)
    rp = tmp_path / "run_provenance.json"
    bp = tmp_path / "baseline_provenance.json"
    rp.write_text(json.dumps({"plan_revision": 8, "baseline_id": "b1"}))
    bp.write_text(json.dumps({"plan_revision": 8, "baseline_id": "b1"}))
    result = check_generate.compare_golden(
        traj, baseline, run_provenance=rp, baseline_provenance=bp
    )
    assert result["eligible"] is True
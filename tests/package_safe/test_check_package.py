"""Tests for scripts/check_package.py validate and compare-golden modes.

Covers the Phase 0b package artifact checker: inventory/schema/copied-file
validation, calibration/extrinsics, depth metadata, Arrow schema/values,
JSON/JSONL order, copied checksums, and negative mutation matrix.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fixtures import build_packaged_dataset, build_rendered_dir, build_trajectory_dir

# Make the checker importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKER_DIR = _REPO_ROOT / "scripts"
if str(_CHECKER_DIR) not in sys.path:
    sys.path.insert(0, str(_CHECKER_DIR))
_TESTS_DIR = _REPO_ROOT / "tests" / "package_safe"
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import check_package


# --- helpers -----------------------------------------------------------------

def _make_triple(tmp_path: Path, frame_counts=(3, 5)):
    """Build trajectory + rendered + packaged dirs."""
    traj = tmp_path / "traj"
    manifest = build_trajectory_dir(traj, episode_frame_counts=frame_counts)
    rendered = tmp_path / "rendered"
    build_rendered_dir(rendered, trajectory_manifest=manifest)
    pkg = tmp_path / "pkg"
    build_packaged_dataset(pkg, trajectory_dir=traj, rendered_dir=rendered, scene_id="839920")
    return pkg, traj, rendered, manifest


def _mutate_info(pkg: Path, mutator):
    path = pkg / "meta" / "info.json"
    with path.open() as f:
        info = json.load(f)
    mutator(info)
    with path.open("w") as f:
        json.dump(info, f, indent=2)


# --- validate: positive ------------------------------------------------------

def test_validate_passes_for_valid_package(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    result = check_package.validate(pkg, traj, rendered)
    assert result["eligible"] is True
    assert result["errors"] == []
    assert result["scene_id"] == "839920"
    assert result["episode_count"] == 2


def test_validate_cli_passes(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    result_path = tmp_path / "result.json"
    proc = subprocess.run(
        [sys.executable, str(_CHECKER_DIR / "check_package.py"),
         "validate", "--dataset-dir", str(pkg),
         "--trajectory-dir", str(traj),
         "--rendered-dir", str(rendered),
         "--result-path", str(result_path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "ELIGIBLE" in proc.stdout
    with result_path.open() as f:
        result = json.load(f)
    assert result["eligible"] is True
    assert result["checker"] == "check_package"


# --- validate: negative ------------------------------------------------------

def test_validate_fails_for_missing_meta_file(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    (pkg / "meta" / "tasks.jsonl").unlink()
    result = check_package.validate(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("tasks.jsonl" in e for e in result["errors"])


def test_validate_fails_for_wrong_scene_id(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    _mutate_info(pkg, lambda i: i.update(scene_id="wrong"))
    result = check_package.validate(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("scene_id" in e for e in result["errors"])


def test_validate_fails_for_wrong_episode_count(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    _mutate_info(pkg, lambda i: i.update(total_episodes=99))
    result = check_package.validate(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("total_episodes" in e for e in result["errors"])


def test_validate_fails_for_missing_parquet(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    (pkg / "data" / "chunk-000" / "episode_000000.parquet").unlink()
    result = check_package.validate(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("parquet" in e.lower() for e in result["errors"])


def test_validate_fails_for_extra_rgb_files(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    # Add an extra RGB file.
    (pkg / "videos" / "chunk-000" / "observation.images.rgb" / "extra.jpg").write_bytes(b"\x00")
    result = check_package.validate(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("RGB file count" in e for e in result["errors"])


def test_validate_fails_for_ply_checksum_mismatch(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    # Corrupt the copied PLY.
    with (pkg / "meta" / "pointcloud.ply").open("ab") as f:
        f.write(b"\x00\x00\x00")
    result = check_package.validate(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("pointcloud.ply" in e for e in result["errors"])


def test_validate_fails_for_manifest_checksum_mismatch(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    # Corrupt the copied manifest.
    path = pkg / "meta" / "trajectory_manifest.json"
    with path.open("w") as f:
        json.dump({"corrupted": True}, f)
    result = check_package.validate(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("manifest" in e for e in result["errors"])


def test_validate_fails_for_depth_clip_mismatch(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    _mutate_info(pkg, lambda i: i.update(depth_clip_m=99.0))
    result = check_package.validate(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("depth_clip_m" in e for e in result["errors"])


def test_validate_fails_for_camera_height_mismatch(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    _mutate_info(pkg, lambda i: i.update(camera_height_m=99.0))
    result = check_package.validate(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("camera_height_m" in e or "extrinsic" in e.lower() for e in result["errors"])


def test_validate_fails_for_episodes_jsonl_order_mismatch(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    # Reverse the episodes.jsonl order.
    path = pkg / "meta" / "episodes.jsonl"
    lines = path.read_text().strip().split("\n")
    path.write_text("\n".join(reversed(lines)) + "\n")
    result = check_package.validate(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("episodes.jsonl" in e for e in result["errors"])


def test_validate_fails_for_stale_rgb_file(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    # Corrupt one RGB file in the package.
    rgb_path = sorted((pkg / "videos" / "chunk-000" / "observation.images.rgb").glob("*.jpg"))[0]
    Image.new("RGB", (600, 450), (0, 0, 0)).save(rgb_path, quality=95)
    result = check_package.validate(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("RGB" in e for e in result["errors"])


def test_validate_fails_for_stale_depth_file(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    # Corrupt one depth file in the package.
    depth_path = sorted((pkg / "videos" / "chunk-000" / "observation.images.depth").glob("*.png"))[0]
    Image.new("L", (600, 450), 0).save(depth_path)
    result = check_package.validate(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("depth" in e for e in result["errors"])


# --- compare-golden: positive -------------------------------------------------

def test_compare_golden_passes_for_identical_copy(tmp_path):
    import shutil
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(pkg, baseline)
    result = check_package.compare_golden(pkg, traj, rendered, baseline)
    assert result["eligible"] is True, f"errors: {result['errors']}"
    assert "info" in result["artifact_digests"]
    assert "episodes_jsonl" in result["artifact_digests"]
    assert "episode_000000" in result["artifact_digests"]


def test_compare_golden_cli_passes(tmp_path):
    import shutil
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(pkg, baseline)
    result_path = tmp_path / "result.json"
    proc = subprocess.run(
        [sys.executable, str(_CHECKER_DIR / "check_package.py"),
         "compare-golden", "--dataset-dir", str(pkg),
         "--trajectory-dir", str(traj),
         "--rendered-dir", str(rendered),
         "--baseline-dir", str(baseline),
         "--result-path", str(result_path)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0
    assert "ELIGIBLE" in proc.stdout


def test_compare_golden_validates_baseline_against_its_own_render_root(tmp_path):
    import shutil

    traj = tmp_path / "traj"
    manifest = build_trajectory_dir(traj, episode_frame_counts=(3,))
    baseline_rendered = tmp_path / "baseline_rendered"
    build_rendered_dir(baseline_rendered, trajectory_manifest=manifest)
    candidate_rendered = tmp_path / "candidate_rendered"
    shutil.copytree(baseline_rendered, candidate_rendered)

    candidate_rgb = sorted(
        (candidate_rendered / "observation.images.rgb").glob("*.jpg")
    )[0]
    Image.new("RGB", (600, 450), (20, 40, 60)).save(candidate_rgb, quality=95)

    baseline_pkg = tmp_path / "baseline_pkg"
    candidate_pkg = tmp_path / "candidate_pkg"
    build_packaged_dataset(
        baseline_pkg,
        trajectory_dir=traj,
        rendered_dir=baseline_rendered,
        scene_id="839920",
    )
    build_packaged_dataset(
        candidate_pkg,
        trajectory_dir=traj,
        rendered_dir=candidate_rendered,
        scene_id="839920",
    )

    result = check_package.compare_golden(
        candidate_pkg,
        traj,
        candidate_rendered,
        baseline_pkg,
        baseline_trajectory_dir=traj,
        baseline_rendered_dir=baseline_rendered,
    )
    assert result["eligible"] is True, result["errors"]


# --- compare-golden: negative -------------------------------------------------

def test_compare_golden_fails_for_info_mismatch(tmp_path):
    import shutil
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(pkg, baseline)
    _mutate_info(pkg, lambda i: i.update(fps=60))
    result = check_package.compare_golden(pkg, traj, rendered, baseline)
    assert result["eligible"] is False
    assert any("info.json" in e for e in result["errors"])


def test_compare_golden_fails_for_episodes_jsonl_mismatch(tmp_path):
    import shutil
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(pkg, baseline)
    # Change a field in the candidate episodes.jsonl.
    path = pkg / "meta" / "episodes.jsonl"
    lines = path.read_text().strip().split("\n")
    rec = json.loads(lines[0])
    rec["frame_count"] = 99
    lines[0] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    result = check_package.compare_golden(pkg, traj, rendered, baseline)
    assert result["eligible"] is False
    assert any("episodes.jsonl" in e for e in result["errors"])


def test_compare_golden_fails_for_tasks_jsonl_mismatch(tmp_path):
    import shutil
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(pkg, baseline)
    # Change tasks.jsonl in candidate.
    path = pkg / "meta" / "tasks.jsonl"
    path.write_text(json.dumps({"task_index": 99, "task": {"type": "wrong"}}) + "\n")
    result = check_package.compare_golden(pkg, traj, rendered, baseline)
    assert result["eligible"] is False
    assert any("tasks.jsonl" in e for e in result["errors"])


def test_compare_golden_fails_for_parquet_value_mismatch(tmp_path):
    import shutil
    import pyarrow as pa
    import pyarrow.parquet as pq
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(pkg, baseline)
    # Mutate a parquet value in the candidate.
    path = pkg / "data" / "chunk-000" / "episode_000000.parquet"
    table = pq.read_table(path)
    # Change index column values by rebuilding the table.
    cols = {name: table[name] for name in table.column_names}
    cols["index"] = pa.array([100 + i for i in range(table.num_rows)], type=pa.int64())
    new_table = pa.table(cols)
    pq.write_table(new_table, path)
    result = check_package.compare_golden(pkg, traj, rendered, baseline)
    assert result["eligible"] is False
    assert any("index" in e or "column" in e or "values mismatch" in e for e in result["errors"])


def test_compare_golden_fails_for_ply_bytes_mismatch(tmp_path):
    import shutil
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(pkg, baseline)
    # Corrupt candidate PLY.
    with (pkg / "meta" / "pointcloud.ply").open("ab") as f:
        f.write(b"\x00")
    result = check_package.compare_golden(pkg, traj, rendered, baseline)
    assert result["eligible"] is False
    assert any("pointcloud.ply" in e for e in result["errors"])


def test_compare_golden_fails_for_summary_bytes_mismatch(tmp_path):
    import shutil
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(pkg, baseline)
    # Corrupt candidate render_summary.json in meta.
    path = pkg / "meta" / "render_summary.json"
    with path.open("w") as f:
        json.dump({"corrupted": True}, f)
    result = check_package.compare_golden(pkg, traj, rendered, baseline)
    assert result["eligible"] is False
    assert any("render_summary" in e for e in result["errors"])


# --- provenance binding ------------------------------------------------------

def test_compare_golden_fails_for_provenance_mismatch(tmp_path):
    import shutil
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(pkg, baseline)
    rp = tmp_path / "rp.json"
    bp = tmp_path / "bp.json"
    rp.write_text(json.dumps({"plan_revision": 8, "baseline_id": "b1"}))
    bp.write_text(json.dumps({"plan_revision": 7, "baseline_id": "b1"}))
    result = check_package.compare_golden(
        pkg, traj, rendered, baseline, run_provenance=rp, baseline_provenance=bp
    )
    assert result["eligible"] is False
    assert any("plan_revision" in e for e in result["errors"])


def test_compare_golden_passes_with_matching_provenance(tmp_path):
    import shutil
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    baseline = tmp_path / "baseline"
    shutil.copytree(pkg, baseline)
    rp = tmp_path / "rp.json"
    bp = tmp_path / "bp.json"
    rp.write_text(json.dumps({"plan_revision": 8, "baseline_id": "b1"}))
    bp.write_text(json.dumps({"plan_revision": 8, "baseline_id": "b1"}))
    result = check_package.compare_golden(
        pkg, traj, rendered, baseline, run_provenance=rp, baseline_provenance=bp
    )
    assert result["eligible"] is True

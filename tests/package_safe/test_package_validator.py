"""Tests for sage3d.lerobot_dataset.validate_packaged_dataset (Phase 5b).

Covers the production staged-package validator: inventory, Arrow/JSON content,
copied-input checksums, calibration/extrinsics, depth metadata, optional
PackageConfig compatibility assertions, and parity with the Phase 0b
``check_package.py`` oracle.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from fixtures import build_packaged_dataset, build_rendered_dir, build_trajectory_dir
from sage3d.config import PackageConfig
from sage3d.lerobot_dataset import validate_packaged_dataset

# Make the checker importable for the parity oracle.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKER_DIR = _REPO_ROOT / "scripts"
if str(_CHECKER_DIR) not in sys.path:
    sys.path.insert(0, str(_CHECKER_DIR))
_TESTS_DIR = _REPO_ROOT / "tests" / "package_safe"
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import check_package  # noqa: E402


# --- helpers -----------------------------------------------------------------

def _make_triple(tmp_path: Path, frame_counts=(3, 5)):
    """Build trajectory + rendered + packaged dirs."""
    traj = tmp_path / "traj"
    manifest = build_trajectory_dir(traj, episode_frame_counts=frame_counts)
    rendered = tmp_path / "rendered"
    build_rendered_dir(rendered, trajectory_manifest=manifest)
    pkg = tmp_path / "pkg"
    build_packaged_dataset(
        pkg, trajectory_dir=traj, rendered_dir=rendered, scene_id="839920"
    )
    return pkg, traj, rendered, manifest


def _mutate_info(pkg: Path, mutator):
    path = pkg / "meta" / "info.json"
    with path.open() as f:
        info = json.load(f)
    mutator(info)
    with path.open("w") as f:
        json.dump(info, f, indent=2)


def _result(pkg, traj, rendered, **config_kwargs):
    config = PackageConfig(
        fps=30, trajectory_dir=traj, rendered_dir=rendered,
        output_dir=pkg, scene_id="839920", **config_kwargs,
    )
    return validate_packaged_dataset(pkg, traj, rendered, config)


# --- positive ----------------------------------------------------------------

def test_validate_passes_for_valid_package(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    result = validate_packaged_dataset(pkg, traj, rendered)
    assert result["eligible"] is True
    assert result["errors"] == []
    assert result["scene_id"] == "839920"
    assert result["episode_count"] == 2


def test_validate_accepts_supported_config_assertions(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    result = _result(
        pkg, traj, rendered,
        width=600, height=450, horizontal_fov_deg=180.0,
        fisheye_coefficients=(0.1, 0.0, 0.0, 0.0), camera_height=0.6,
    )
    assert result["eligible"] is True, result["errors"]


# --- negative matrix ---------------------------------------------------------

def test_validate_fails_for_missing_meta_file(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    (pkg / "meta" / "tasks.jsonl").unlink()
    result = validate_packaged_dataset(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("tasks.jsonl" in e for e in result["errors"])


def test_validate_fails_for_wrong_scene_id(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    _mutate_info(pkg, lambda i: i.update(scene_id="wrong"))
    result = validate_packaged_dataset(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("scene_id" in e for e in result["errors"])


def test_validate_fails_for_wrong_episode_count(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    _mutate_info(pkg, lambda i: i.update(total_episodes=99))
    result = validate_packaged_dataset(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("total_episodes" in e for e in result["errors"])


def test_validate_fails_for_missing_parquet(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    (pkg / "data" / "chunk-000" / "episode_000000.parquet").unlink()
    result = validate_packaged_dataset(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("parquet" in e.lower() for e in result["errors"])


def test_validate_fails_for_extra_rgb_files(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    (pkg / "videos" / "chunk-000" / "observation.images.rgb" / "extra.jpg").write_bytes(b"\x00")
    result = validate_packaged_dataset(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("RGB file count" in e for e in result["errors"])


def test_validate_fails_for_ply_checksum_mismatch(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    with (pkg / "meta" / "pointcloud.ply").open("ab") as f:
        f.write(b"\x00\x00\x00")
    result = validate_packaged_dataset(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("pointcloud.ply" in e for e in result["errors"])


def test_validate_fails_for_manifest_checksum_mismatch(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    path = pkg / "meta" / "trajectory_manifest.json"
    with path.open("w") as f:
        json.dump({"corrupted": True}, f)
    result = validate_packaged_dataset(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("manifest" in e for e in result["errors"])


def test_validate_fails_for_depth_clip_mismatch(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    _mutate_info(pkg, lambda i: i.update(depth_clip_m=99.0))
    result = validate_packaged_dataset(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("depth_clip_m" in e for e in result["errors"])


def test_validate_fails_for_camera_height_mismatch(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    _mutate_info(pkg, lambda i: i.update(camera_height_m=99.0))
    result = validate_packaged_dataset(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("camera_height_m" in e or "extrinsic" in e.lower() for e in result["errors"])


def test_validate_fails_for_episodes_jsonl_order_mismatch(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    path = pkg / "meta" / "episodes.jsonl"
    lines = path.read_text().strip().split("\n")
    path.write_text("\n".join(reversed(lines)) + "\n")
    result = validate_packaged_dataset(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("episodes.jsonl" in e for e in result["errors"])


def test_validate_fails_for_stale_rgb_file(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    rgb_path = sorted((pkg / "videos" / "chunk-000" / "observation.images.rgb").glob("*.jpg"))[0]
    Image.new("RGB", (600, 450), (0, 0, 0)).save(rgb_path, quality=95)
    result = validate_packaged_dataset(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("RGB" in e for e in result["errors"])


def test_validate_fails_for_stale_depth_file(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    depth_path = sorted((pkg / "videos" / "chunk-000" / "observation.images.depth").glob("*.png"))[0]
    Image.new("L", (600, 450), 0).save(depth_path)
    result = validate_packaged_dataset(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("depth" in e for e in result["errors"])


def test_validate_fails_for_parquet_column_type_mismatch(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    path = pkg / "data" / "chunk-000" / "episode_000000.parquet"
    table = pq.read_table(path)
    cols = {name: table[name] for name in table.column_names}
    cols["action"] = pa.array(
        [[0.0] * 16] * table.num_rows, type=pa.list_(pa.float64())
    )
    pq.write_table(pa.table(cols), path)
    result = validate_packaged_dataset(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("action" in e and "float32" in e for e in result["errors"])


def test_validate_fails_for_parquet_row_count_mismatch(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    path = pkg / "data" / "chunk-000" / "episode_000000.parquet"
    table = pq.read_table(path)
    # Drop the last row from every column so the row count no longer matches
    # the manifest frame_count.
    new_n = table.num_rows - 1
    cols = {
        name: pa.array(table[name].to_pylist()[:new_n], type=table[name].type)
        for name in table.column_names
    }
    pq.write_table(pa.table(cols), path)
    result = validate_packaged_dataset(pkg, traj, rendered)
    assert result["eligible"] is False
    assert any("rows" in e for e in result["errors"])


# --- PackageConfig compatibility assertions ----------------------------------

def test_validate_fails_for_config_width_mismatch(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    result = _result(pkg, traj, rendered, width=640)
    assert result["eligible"] is False
    assert any("width" in e for e in result["errors"])


def test_validate_fails_for_config_fov_mismatch(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    result = _result(pkg, traj, rendered, horizontal_fov_deg=90.0)
    assert result["eligible"] is False
    assert any("horizontal_fov_deg" in e for e in result["errors"])


def test_validate_fails_for_config_coefficients_mismatch(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    result = _result(pkg, traj, rendered, fisheye_coefficients=(1.0, 0.0, 0.0, 0.0))
    assert result["eligible"] is False
    assert any("fisheye_coefficients" in e for e in result["errors"])


def test_validate_fails_for_config_camera_height_mismatch(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    result = _result(pkg, traj, rendered, camera_height=1.2)
    assert result["eligible"] is False
    assert any("camera_height" in e for e in result["errors"])


# --- checker parity (Phase 0b oracle behavior unchanged) ---------------------

def test_checker_validate_parity_with_production_validator(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    prod = validate_packaged_dataset(pkg, traj, rendered)
    checker = check_package.validate(pkg, traj, rendered)
    assert prod["eligible"] == checker["eligible"]
    assert prod["errors"] == checker["errors"]
    assert prod["scene_id"] == checker["scene_id"]
    assert prod["episode_count"] == checker["episode_count"]


def test_checker_validate_parity_on_mutated_package(tmp_path):
    pkg, traj, rendered, _ = _make_triple(tmp_path)
    _mutate_info(pkg, lambda i: i.update(depth_clip_m=99.0))
    prod = validate_packaged_dataset(pkg, traj, rendered)
    checker = check_package.validate(pkg, traj, rendered)
    assert prod["eligible"] is False
    assert prod["errors"] == checker["errors"]


def test_checker_cli_validate_still_works(tmp_path):
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

"""Tests for the legacy artifact parsers and synthetic package-success fixture.

These prove the Phase 0a parsers correctly read the legacy artifact contract,
and that the synthetic fixture builder produces an internally-consistent
package-success tree. They import no target production module (``sage3d.*``).
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from artifact_parsers import (
    parse_binary_ply,
    parse_episode_npz,
    parse_packaged_dataset,
    parse_render_summary,
    parse_trajectory_manifest,
)
from fixtures import (
    NPZ_KEYS,
    build_packaged_dataset,
    build_rendered_dir,
    build_trajectory_dir,
)


def test_parse_trajectory_manifest_success(tmp_path):
    manifest = build_trajectory_dir(tmp_path / "traj", episode_frame_counts=(3, 5))
    parsed = parse_trajectory_manifest(tmp_path / "traj")
    assert parsed["scene_id"] == manifest["scene_id"]
    assert parsed["episode_count"] == 2 == len(parsed["episodes"])


def test_parse_trajectory_manifest_rejects_missing_keys(tmp_path):
    build_trajectory_dir(tmp_path / "traj", episode_frame_counts=(3,))
    manifest_path = tmp_path / "traj" / "trajectory_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    del manifest["seed"]
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with pytest.raises(ValueError, match="missing keys"):
        parse_trajectory_manifest(tmp_path / "traj")


def test_parse_trajectory_manifest_rejects_count_mismatch(tmp_path):
    build_trajectory_dir(tmp_path / "traj", episode_frame_counts=(3,))
    manifest_path = tmp_path / "traj" / "trajectory_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["episode_count"] = 99
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with pytest.raises(ValueError, match="episode_count"):
        parse_trajectory_manifest(tmp_path / "traj")


def test_parse_episode_npz_success(tmp_path):
    build_trajectory_dir(tmp_path / "traj", episode_frame_counts=(4, 6))
    arrays = parse_episode_npz(tmp_path / "traj", 0)
    assert set(arrays) == set(NPZ_KEYS)
    assert arrays["actions"].shape == (4, 4, 4)
    assert arrays["point_goal"].shape == (4, 2)


def test_parse_episode_npz_rejects_missing_file(tmp_path):
    build_trajectory_dir(tmp_path / "traj", episode_frame_counts=(3,))
    with pytest.raises(FileNotFoundError):
        parse_episode_npz(tmp_path / "traj", 99)


def test_parse_episode_npz_rejects_key_mismatch(tmp_path):
    build_trajectory_dir(tmp_path / "traj", episode_frame_counts=(3,))
    path = tmp_path / "traj" / "episode_000000.npz"
    data = np.load(path)
    arrays = {key: data[key] for key in data.files}
    del arrays["yaw"]
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="npz keys"):
        parse_episode_npz(tmp_path / "traj", 0)


def test_parse_episode_npz_rejects_shape_mismatch(tmp_path):
    build_trajectory_dir(tmp_path / "traj", episode_frame_counts=(3,))
    path = tmp_path / "traj" / "episode_000000.npz"
    data = np.load(path)
    arrays = {key: data[key] for key in data.files}
    arrays["actions"] = arrays["actions"][:-1]
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="length"):
        parse_episode_npz(tmp_path / "traj", 0)


def test_parse_binary_ply_success(tmp_path):
    build_trajectory_dir(tmp_path / "traj", episode_frame_counts=(3,))
    ply = parse_binary_ply(tmp_path / "traj" / "pointcloud.ply")
    assert ply["vertex_count"] == 3
    assert ply["points"].shape == (3, 3)
    assert ply["colors"].dtype == np.uint8
    assert np.all(ply["colors"] == 160)


def test_parse_binary_ply_rejects_wrong_format(tmp_path):
    path = tmp_path / "bad.ply"
    path.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    with pytest.raises(ValueError, match="format"):
        parse_binary_ply(path)


def test_parse_render_summary_success(tmp_path):
    manifest = build_trajectory_dir(tmp_path / "traj", episode_frame_counts=(3, 5))
    build_rendered_dir(tmp_path / "render", trajectory_manifest=manifest)
    summary = parse_render_summary(tmp_path / "render")
    assert summary["render_mode"] == "depth"
    assert summary["total_frames"] == 8


def test_parse_render_summary_rejects_missing_keys(tmp_path):
    manifest = build_trajectory_dir(tmp_path / "traj", episode_frame_counts=(3,))
    build_rendered_dir(tmp_path / "render", trajectory_manifest=manifest)
    path = tmp_path / "render" / "render_summary.json"
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    del summary["focal_length_pixels"]
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f)
    with pytest.raises(ValueError, match="missing keys"):
        parse_render_summary(tmp_path / "render")


def test_parse_packaged_dataset_success(tmp_path):
    manifest = build_trajectory_dir(
        tmp_path / "traj", episode_frame_counts=(3, 5), scene_id="839920"
    )
    build_rendered_dir(tmp_path / "render", trajectory_manifest=manifest)
    info = build_packaged_dataset(
        tmp_path / "packaged",
        trajectory_dir=tmp_path / "traj",
        rendered_dir=tmp_path / "render",
        scene_id="839920",
    )
    dataset = parse_packaged_dataset(tmp_path / "packaged")
    assert dataset["info"]["scene_id"] == "839920"
    assert len(dataset["parquet_files"]) == 2
    assert len(dataset["rgb_files"]) == 8
    assert len(dataset["depth_files"]) == 8
    assert "info.json" in dataset["meta_files"]
    assert "trajectory_manifest.json" in dataset["meta_files"]
    assert "render_summary.json" in dataset["meta_files"]
    assert info["total_frames"] == 8


def test_parse_packaged_dataset_rejects_missing_info(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_packaged_dataset(tmp_path / "empty")
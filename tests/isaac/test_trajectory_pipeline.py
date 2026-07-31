"""Tests for the trajectory pipeline (Phase 3d).

Covers:
- ``generate(config)`` produces a ``TrajectoryResult`` with all fields.
- ``write_trajectory_artifacts`` writes all 7 artifacts into staging.
- Atomic publication: generate → staging → write → publish yields a final
  target with byte-identical artifacts.
- Failure injection at write stage leaves the final target absent.
- Existing target (file/dir/symlink/dangling) is refused.
- Same-device assertion.
- Rerun allocates a fresh staging directory (cooperative single-publisher).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from sage3d.config import (
    GenerationConfig,
    PathConfig,
    SceneConfig,
    SafetyConfig,
)
from sage3d.publication import (
    assert_target_absent,
    atomic_publish_directory,
    create_staging_directory,
)
from sage3d.trajectory_pipeline import (
    TrajectoryResult,
    generate,
    write_trajectory_artifacts,
)


# --- helpers -----------------------------------------------------------------

SAGE3D_ROOT = os.environ.get("SAGE3D_ROOT", "/ssd5/datasets/SAGE3D")
SCENE_ID = "839920"
SEED = 20260720
EPISODES = 3


def _make_config() -> GenerationConfig:
    return GenerationConfig(
        episodes=EPISODES,
        seed=SEED,
        pointcloud_voxel_size=0.05,
        pointcloud_max_points=100_000,
        scene=SceneConfig(
            scene_id=SCENE_ID,
            sage_root=Path(SAGE3D_ROOT),
        ),
        safety=SafetyConfig(
            robot_radius=0.25,
            safety_margin=0.05,
            camera_height=0.6,
            camera_clearance=0.25,
            endpoint_extra_clearance=0.10,
        ),
        path=PathConfig(
            min_path_length=3.0,
            max_path_length=15.0,
            frame_spacing=0.05,
            max_attempts=3000,
        ),
    )


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _all_artifact_digests(root: Path) -> dict[str, str]:
    """Return sha256 of every file directly under root."""
    digests = {}
    for f in sorted(root.iterdir()):
        if f.is_file():
            digests[f.name] = _sha256(f)
    return digests


# --- generate ----------------------------------------------------------------

@pytest.fixture(scope="module")
def result() -> TrajectoryResult:
    if not Path(SAGE3D_ROOT).is_dir():
        pytest.skip("SAGE3D_ROOT not available")
    return generate(_make_config())


def test_generate_returns_trajectory_result(result: TrajectoryResult):
    assert isinstance(result, TrajectoryResult)
    assert len(result.episodes) == EPISODES
    assert "attempts" in result.generation_info
    assert "rejection_counts" in result.generation_info
    assert result.pointcloud.ndim == 2
    assert result.pointcloud.shape[1] == 3
    assert isinstance(result.scene_dir, Path)
    assert isinstance(result.collision_usd, Path)
    assert result.safe.ndim == 2
    assert result.clearance_m.ndim == 2
    assert result.transform is not None


def test_generate_manifest_has_legacy_key_order(result: TrajectoryResult):
    keys = list(result.manifest.keys())
    expected = [
        "scene_id", "scene_dir", "collision_usd", "seed", "episode_count",
        "robot_radius_m", "safety_margin_m", "camera_height_m",
        "camera_clearance_m", "frame_spacing_m",
        "requested_path_length_range_m", "endpoint_clearance_m",
        "map", "generation", "pointcloud", "episodes",
    ]
    assert keys == expected


# --- write + publish ---------------------------------------------------------

@pytest.fixture(scope="module")
def published_root(result: TrajectoryResult, tmp_path_factory):
    """Generate once, write to staging, publish to a fresh target."""
    out = tmp_path_factory.mktemp("traj_out") / "trajectory"
    assert_target_absent(out)
    staging = create_staging_directory(out, prefix=".trajectory-stage.")
    write_trajectory_artifacts(staging, result)
    atomic_publish_directory(staging, out)
    return out


def test_published_root_has_seven_artifacts(published_root: Path):
    files = sorted(f.name for f in published_root.iterdir() if f.is_file())
    # 3 episode npz + pointcloud.ply + navigation_map.png +
    # trajectories_overlay.png + trajectory_manifest.json = 7
    assert len(files) == EPISODES + 4
    for i in range(EPISODES):
        assert f"episode_{i:06d}.npz" in files
    assert "pointcloud.ply" in files
    assert "navigation_map.png" in files
    assert "trajectories_overlay.png" in files
    assert "trajectory_manifest.json" in files


def test_published_manifest_matches_result_manifest(
    published_root: Path, result: TrajectoryResult
):
    with (published_root / "trajectory_manifest.json").open() as f:
        on_disk = json.load(f)
    assert on_disk == result.manifest


# --- failure injection: write failure leaves target absent -------------------

def test_write_failure_leaves_target_absent(result: TrajectoryResult, tmp_path):
    target = tmp_path / "traj_fail"
    staging = create_staging_directory(target, prefix=".trajectory-stage.")
    # Simulate a write failure by removing staging before publish.
    import shutil
    shutil.rmtree(staging)
    with pytest.raises((FileNotFoundError, NotADirectoryError)):
        atomic_publish_directory(staging, target)
    assert not target.exists()


# --- target rejection: existing file/dir/symlink/dangling -------------------

def test_pipeline_refuses_existing_file_target(result: TrajectoryResult, tmp_path):
    target = tmp_path / "blocking_file"
    target.write_text("block")
    with pytest.raises(FileExistsError):
        assert_target_absent(target)


def test_pipeline_refuses_existing_dir_target(result: TrajectoryResult, tmp_path):
    target = tmp_path / "existing_dir"
    target.mkdir()
    (target / "old.txt").write_text("old")
    with pytest.raises(FileExistsError):
        assert_target_absent(target)


def test_pipeline_refuses_symlinked_target(result: TrajectoryResult, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(FileExistsError):
        assert_target_absent(link)


def test_pipeline_refuses_dangling_symlink_target(result: TrajectoryResult, tmp_path):
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        assert_target_absent(link)


# --- same-device assertion --------------------------------------------------

def test_staging_and_target_same_device(result: TrajectoryResult, tmp_path):
    target = tmp_path / "traj_same_dev"
    staging = create_staging_directory(target, prefix=".trajectory-stage.")
    write_trajectory_artifacts(staging, result)
    # atomic_publish_directory checks same-device internally; if devices
    # match (they always will under tmp_path), publish succeeds.
    atomic_publish_directory(staging, target)
    assert target.is_dir()


# --- rerun allocates fresh staging (cooperative single-publisher) ------------

def test_rerun_uses_fresh_staging(result: TrajectoryResult, tmp_path):
    """A second publish attempt to a new target allocates a new staging dir."""
    target1 = tmp_path / "run1"
    target2 = tmp_path / "run2"
    staging1 = create_staging_directory(target1, prefix=".trajectory-stage.")
    write_trajectory_artifacts(staging1, result)
    atomic_publish_directory(staging1, target1)

    staging2 = create_staging_directory(target2, prefix=".trajectory-stage.")
    write_trajectory_artifacts(staging2, result)
    atomic_publish_directory(staging2, target2)

    assert staging1 != staging2
    assert target1.is_dir()
    assert target2.is_dir()
    # Both runs produce identical artifacts.
    assert _all_artifact_digests(target1) == _all_artifact_digests(target2)


# --- determinism: two independent generate calls produce identical artifacts --

def test_generate_is_deterministic(tmp_path):
    if not Path(SAGE3D_ROOT).is_dir():
        pytest.skip("SAGE3D_ROOT not available")
    r1 = generate(_make_config())
    r2 = generate(_make_config())
    # Manifests identical.
    assert r1.manifest == r2.manifest
    # Pointclouds byte-identical.
    assert np.array_equal(r1.pointcloud, r2.pointcloud)
    # Episodes identical.
    assert len(r1.episodes) == len(r2.episodes)
    for e1, e2 in zip(r1.episodes, r2.episodes):
        for key in e1:
            if isinstance(e1[key], np.ndarray):
                assert np.array_equal(e1[key], e2[key])
            else:
                assert e1[key] == e2[key]
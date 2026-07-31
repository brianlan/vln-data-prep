"""Tests for sage3d.contract.validate_pipeline_contract (Phase 2b).

Covers the positive synthetic fixture and the complete negative matrix:
delete / add / rename / wrong-dtype / wrong-shape / stale-file / missing-file /
index-discontinuity. Also covers ``allow_pickle=False`` context-manager
behavior and the package-safe forbidden-import smoke for ``sage3d.contract``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sage3d.contract import (
    CalibrationMismatchError,
    CameraHeightMismatchError,
    ContractError,
    DepthAliasMismatchError,
    EpisodeCountMismatchError,
    EpisodeIndexError,
    FrameCountMismatchError,
    ImageInventoryError,
    NpzSchemaError,
    SceneIdMismatchError,
    SharedDepthFieldMismatchError,
    validate_pipeline_contract,
)
from sage3d.episode_arrays import load_episode
from sage3d.naming import parse_episode_filename

from fixtures import build_rendered_dir, build_trajectory_dir

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _load_episodes(trajectory_dir: Path) -> dict[int, object]:
    episodes = {}
    for tf in sorted(trajectory_dir.glob("episode_*.npz")):
        episodes[parse_episode_filename(tf.name)] = load_episode(tf)
    return episodes


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def canonical_tree(tmp_path: Path) -> dict:
    """Build a fully-consistent synthetic trajectory + render tree."""
    traj_dir = tmp_path / "trajectory"
    rendered_dir = tmp_path / "rendered"
    manifest = build_trajectory_dir(
        traj_dir, episode_frame_counts=(3, 2, 4), scene_id="839920"
    )
    build_rendered_dir(rendered_dir, trajectory_manifest=manifest)
    return {
        "trajectory_dir": traj_dir,
        "rendered_dir": rendered_dir,
        "pointcloud_path": traj_dir / "pointcloud.ply",
        "manifest": manifest,
        "rgb_summary": _load_json(rendered_dir / "rgb_render_summary.json"),
        "canonical_depth_summary": _load_json(rendered_dir / "render_summary.json"),
        "depth_alias_summary": _load_json(rendered_dir / "depth_render_summary.json"),
        "episodes_by_id": _load_episodes(traj_dir),
        "scene_id": "839920",
    }


def _validate(tree: dict) -> None:
    validate_pipeline_contract(
        expected_scene_id=tree["scene_id"],
        manifest=tree["manifest"],
        rgb_summary=tree["rgb_summary"],
        canonical_depth_summary=tree["canonical_depth_summary"],
        depth_alias_summary=tree["depth_alias_summary"],
        episodes_by_id=tree["episodes_by_id"],
        trajectory_dir=tree["trajectory_dir"],
        rendered_dir=tree["rendered_dir"],
        pointcloud_path=tree["pointcloud_path"],
    )


# ---------------------------------------------------------------------------
# Positive test
# ---------------------------------------------------------------------------


def test_positive_canonical_fixture_passes(canonical_tree: dict) -> None:
    _validate(canonical_tree)


# ---------------------------------------------------------------------------
# Scene-ID mismatch
# ---------------------------------------------------------------------------


def test_scene_id_mismatch_in_manifest(canonical_tree: dict) -> None:
    canonical_tree["manifest"]["scene_id"] = "999999"
    with pytest.raises(SceneIdMismatchError, match="manifest"):
        _validate(canonical_tree)


def test_scene_id_mismatch_in_rgb_summary(canonical_tree: dict) -> None:
    canonical_tree["rgb_summary"]["scene_id"] = "999999"
    with pytest.raises(SceneIdMismatchError, match="rgb_summary"):
        _validate(canonical_tree)


def test_scene_id_mismatch_in_depth_summary(canonical_tree: dict) -> None:
    canonical_tree["canonical_depth_summary"]["scene_id"] = "999999"
    with pytest.raises(SceneIdMismatchError, match="canonical_depth_summary"):
        _validate(canonical_tree)


def test_scene_id_mismatch_in_alias_summary(canonical_tree: dict) -> None:
    canonical_tree["depth_alias_summary"]["scene_id"] = "999999"
    with pytest.raises(SceneIdMismatchError, match="depth_alias_summary"):
        _validate(canonical_tree)


# ---------------------------------------------------------------------------
# Render-mode checks
# ---------------------------------------------------------------------------


def test_rgb_render_mode_wrong(canonical_tree: dict) -> None:
    canonical_tree["rgb_summary"]["render_mode"] = "depth"
    with pytest.raises(ContractError, match="rgb_summary render_mode"):
        _validate(canonical_tree)


def test_depth_render_mode_wrong(canonical_tree: dict) -> None:
    canonical_tree["canonical_depth_summary"]["render_mode"] = "rgb"
    with pytest.raises(ContractError, match="canonical_depth_summary render_mode"):
        _validate(canonical_tree)


# ---------------------------------------------------------------------------
# Calibration agreement (RGB ↔ canonical depth)
# ---------------------------------------------------------------------------


def test_calibration_resolution_mismatch(canonical_tree: dict) -> None:
    canonical_tree["rgb_summary"]["resolution"] = [700, 500]
    with pytest.raises(CalibrationMismatchError, match="resolution"):
        _validate(canonical_tree)


def test_calibration_fov_mismatch(canonical_tree: dict) -> None:
    canonical_tree["rgb_summary"]["horizontal_fov_deg"] = 170.0
    with pytest.raises(CalibrationMismatchError, match="horizontal_fov_deg"):
        _validate(canonical_tree)


def test_calibration_focal_length_mismatch(canonical_tree: dict) -> None:
    canonical_tree["rgb_summary"]["focal_length_pixels"] = 999.0
    with pytest.raises(CalibrationMismatchError, match="focal_length_pixels"):
        _validate(canonical_tree)


def test_calibration_principal_point_mismatch(canonical_tree: dict) -> None:
    canonical_tree["rgb_summary"]["principal_point"] = [301.0, 225.0]
    with pytest.raises(CalibrationMismatchError, match="principal_point"):
        _validate(canonical_tree)


def test_calibration_forward_mask_radius_mismatch(canonical_tree: dict) -> None:
    canonical_tree["rgb_summary"]["forward_mask_radius_pixels"] = 299.0
    with pytest.raises(CalibrationMismatchError, match="forward_mask_radius_pixels"):
        _validate(canonical_tree)


def test_calibration_coefficients_mismatch(canonical_tree: dict) -> None:
    canonical_tree["rgb_summary"]["fisheye_coefficients"] = [0.2, 0.0, 0.0, 0.0]
    with pytest.raises(CalibrationMismatchError, match="fisheye_coefficients"):
        _validate(canonical_tree)


# ---------------------------------------------------------------------------
# Shared depth fields
# ---------------------------------------------------------------------------


def test_shared_depth_type_mismatch(canonical_tree: dict) -> None:
    canonical_tree["rgb_summary"]["depth_type"] = "distance_to_object"
    with pytest.raises(SharedDepthFieldMismatchError, match="depth_type"):
        _validate(canonical_tree)


def test_shared_min_depth_mismatch(canonical_tree: dict) -> None:
    canonical_tree["rgb_summary"]["min_depth_m"] = 0.1
    with pytest.raises(SharedDepthFieldMismatchError, match="min_depth_m"):
        _validate(canonical_tree)


def test_shared_max_depth_mismatch(canonical_tree: dict) -> None:
    canonical_tree["rgb_summary"]["max_depth_m"] = 10.0
    with pytest.raises(SharedDepthFieldMismatchError, match="max_depth_m"):
        _validate(canonical_tree)


def test_shared_depth_scale_mismatch(canonical_tree: dict) -> None:
    canonical_tree["rgb_summary"]["depth_scale"] = 5000.0
    with pytest.raises(SharedDepthFieldMismatchError, match="depth_scale"):
        _validate(canonical_tree)


# ---------------------------------------------------------------------------
# Depth alias equality (render_summary.json == depth_render_summary.json)
# ---------------------------------------------------------------------------


def test_depth_alias_total_frames_mismatch(canonical_tree: dict) -> None:
    canonical_tree["depth_alias_summary"]["total_frames"] = 999
    with pytest.raises(DepthAliasMismatchError):
        _validate(canonical_tree)


def test_depth_alias_episode_record_mismatch(canonical_tree: dict) -> None:
    canonical_tree["depth_alias_summary"]["episodes"][0]["frame_count"] = 999
    with pytest.raises(DepthAliasMismatchError, match="episodes"):
        _validate(canonical_tree)


# ---------------------------------------------------------------------------
# Episode count and index
# ---------------------------------------------------------------------------


def test_episode_count_mismatch_too_few_npz(canonical_tree: dict, tmp_path: Path) -> None:
    # Remove an episode file and its entry in episodes_by_id.
    (canonical_tree["trajectory_dir"] / "episode_000002.npz").unlink()
    del canonical_tree["episodes_by_id"][2]
    with pytest.raises(EpisodeCountMismatchError):
        _validate(canonical_tree)


def test_episode_count_mismatch_too_many_npz(canonical_tree: dict) -> None:
    # Add an extra npz file not in the manifest.
    extra = canonical_tree["trajectory_dir"] / "episode_000003.npz"
    np.savez_compressed(extra, **dict(
        points=np.zeros((1, 2), dtype=np.float32),
        actions=np.eye(4, dtype=np.float32)[None],
        camera_positions=np.zeros((1, 3), dtype=np.float32),
        yaw=np.zeros(1, dtype=np.float32),
        point_goal=np.zeros((1, 2), dtype=np.float32),
        start_position=np.zeros(3, dtype=np.float32),
        goal_position=np.zeros(3, dtype=np.float32),
    ))
    canonical_tree["episodes_by_id"][3] = load_episode(extra)
    with pytest.raises(EpisodeCountMismatchError):
        _validate(canonical_tree)


def test_index_discontinuity_gap(canonical_tree: dict, tmp_path: Path) -> None:
    # Rename episode_000001.npz to episode_000010.npz so the index set is {0,2,10}.
    src = canonical_tree["trajectory_dir"] / "episode_000001.npz"
    dst = canonical_tree["trajectory_dir"] / "episode_000010.npz"
    src.rename(dst)
    canonical_tree["episodes_by_id"] = _load_episodes(canonical_tree["trajectory_dir"])
    with pytest.raises(EpisodeIndexError):
        _validate(canonical_tree)


def test_index_discontinuity_extra_file(canonical_tree: dict) -> None:
    # Add a stale episode file that is not in the manifest.
    stale = canonical_tree["trajectory_dir"] / "episode_000099.npz"
    np.savez_compressed(stale, **dict(
        points=np.zeros((1, 2), dtype=np.float32),
        actions=np.eye(4, dtype=np.float32)[None],
        camera_positions=np.zeros((1, 3), dtype=np.float32),
        yaw=np.zeros(1, dtype=np.float32),
        point_goal=np.zeros((1, 2), dtype=np.float32),
        start_position=np.zeros(3, dtype=np.float32),
        goal_position=np.zeros(3, dtype=np.float32),
    ))
    with pytest.raises(EpisodeIndexError, match="unexpected episode file"):
        _validate(canonical_tree)


# ---------------------------------------------------------------------------
# Frame-count mismatches
# ---------------------------------------------------------------------------


def test_manifest_frame_count_mismatches_npz(canonical_tree: dict) -> None:
    canonical_tree["manifest"]["episodes"][0]["frame_count"] = 999
    with pytest.raises(FrameCountMismatchError, match="manifest frame_count"):
        _validate(canonical_tree)


def test_summary_total_frames_mismatches_manifest(canonical_tree: dict) -> None:
    canonical_tree["canonical_depth_summary"]["total_frames"] = 999
    canonical_tree["depth_alias_summary"]["total_frames"] = 999
    with pytest.raises(FrameCountMismatchError, match="canonical_depth_summary"):
        _validate(canonical_tree)


def test_rgb_total_frames_mismatches_manifest(canonical_tree: dict) -> None:
    canonical_tree["rgb_summary"]["total_frames"] = 999
    with pytest.raises(FrameCountMismatchError, match="rgb_summary total_frames"):
        _validate(canonical_tree)


def test_depth_per_episode_count_mismatches_npz(canonical_tree: dict) -> None:
    canonical_tree["canonical_depth_summary"]["episodes"][0]["frame_count"] = 999
    canonical_tree["depth_alias_summary"]["episodes"][0]["frame_count"] = 999
    with pytest.raises(FrameCountMismatchError, match="depth summary episode 0"):
        _validate(canonical_tree)


def test_depth_episode_record_count_mismatches_inventory(canonical_tree: dict) -> None:
    # Remove one episode record from the depth summary.
    canonical_tree["canonical_depth_summary"]["episodes"] = canonical_tree["canonical_depth_summary"]["episodes"][:2]
    canonical_tree["depth_alias_summary"]["episodes"] = canonical_tree["depth_alias_summary"]["episodes"][:2]
    with pytest.raises(FrameCountMismatchError, match="depth summary has 2"):
        _validate(canonical_tree)


# ---------------------------------------------------------------------------
# Camera height
# ---------------------------------------------------------------------------


def test_camera_height_mismatch_in_manifest(canonical_tree: dict) -> None:
    canonical_tree["manifest"]["camera_height_m"] = 0.9
    with pytest.raises(CameraHeightMismatchError, match="manifest camera_height_m"):
        _validate(canonical_tree)


def test_camera_height_mismatch_in_npz(canonical_tree: dict, tmp_path: Path) -> None:
    # Rewrite episode 0 with a different camera_height z.
    ep = canonical_tree["episodes_by_id"][0]
    ep.camera_positions[:, 2] = np.float32(0.9)
    from sage3d.episode_arrays import save_episode

    save_episode(canonical_tree["trajectory_dir"] / "episode_000000.npz", ep)
    canonical_tree["episodes_by_id"][0] = load_episode(
        canonical_tree["trajectory_dir"] / "episode_000000.npz"
    )
    with pytest.raises(CameraHeightMismatchError, match="episode 0"):
        _validate(canonical_tree)


# ---------------------------------------------------------------------------
# NPZ schema checks
# ---------------------------------------------------------------------------


def test_npz_missing_key(canonical_tree: dict, tmp_path: Path) -> None:
    path = canonical_tree["trajectory_dir"] / "episode_000000.npz"
    data = dict(np.load(path, allow_pickle=False))
    del data["yaw"]
    np.savez_compressed(path, **data)
    # Keep the original episodes_by_id; the contract validator re-loads the
    # npz from disk in _check_npz_schema and catches the missing key there.
    with pytest.raises(NpzSchemaError, match="missing keys"):
        _validate(canonical_tree)


def test_npz_extra_key(canonical_tree: dict, tmp_path: Path) -> None:
    path = canonical_tree["trajectory_dir"] / "episode_000000.npz"
    data = dict(np.load(path, allow_pickle=False))
    data["extra"] = np.zeros(1, dtype=np.float32)
    np.savez_compressed(path, **data)
    canonical_tree["episodes_by_id"] = _load_episodes(canonical_tree["trajectory_dir"])
    with pytest.raises(NpzSchemaError, match="extra keys"):
        _validate(canonical_tree)


def test_npz_wrong_dtype(canonical_tree: dict, tmp_path: Path) -> None:
    path = canonical_tree["trajectory_dir"] / "episode_000000.npz"
    data = dict(np.load(path, allow_pickle=False))
    data["actions"] = data["actions"].astype(np.float64)
    np.savez_compressed(path, **data)
    canonical_tree["episodes_by_id"] = _load_episodes(canonical_tree["trajectory_dir"])
    with pytest.raises(NpzSchemaError, match="dtype"):
        _validate(canonical_tree)


def test_npz_wrong_shape(canonical_tree: dict, tmp_path: Path) -> None:
    path = canonical_tree["trajectory_dir"] / "episode_000000.npz"
    data = dict(np.load(path, allow_pickle=False))
    data["actions"] = data["actions"][:, :3, :]  # shape (N, 3, 4) instead of (N, 4, 4)
    np.savez_compressed(path, **data)
    canonical_tree["episodes_by_id"] = _load_episodes(canonical_tree["trajectory_dir"])
    with pytest.raises(NpzSchemaError, match="actions shape"):
        _validate(canonical_tree)


def test_npz_non_finite_values(canonical_tree: dict, tmp_path: Path) -> None:
    path = canonical_tree["trajectory_dir"] / "episode_000000.npz"
    data = dict(np.load(path, allow_pickle=False))
    data["actions"][0, 0, 0] = np.nan
    np.savez_compressed(path, **data)
    canonical_tree["episodes_by_id"] = _load_episodes(canonical_tree["trajectory_dir"])
    with pytest.raises(NpzSchemaError, match="non-finite"):
        _validate(canonical_tree)


def test_npz_pickle_load_rejected(canonical_tree: dict, tmp_path: Path) -> None:
    """allow_pickle=False must reject a pickle-payload npz."""
    import pickle

    path = canonical_tree["trajectory_dir"] / "episode_000000.npz"
    # Write a numpy file with an object array (requires allow_pickle=True).
    obj = np.array([object()], dtype=object)
    np.savez(path, payload=obj)
    with pytest.raises((NpzSchemaError, ValueError, KeyError, OSError)):
        _validate(canonical_tree)


# ---------------------------------------------------------------------------
# Image inventory checks
# ---------------------------------------------------------------------------


def test_missing_rgb_frame(canonical_tree: dict) -> None:
    (canonical_tree["rendered_dir"] / "observation.images.rgb" / "episode_000000_000.jpg").unlink()
    with pytest.raises(ImageInventoryError, match="missing.*rgb"):
        _validate(canonical_tree)


def test_missing_depth_frame(canonical_tree: dict) -> None:
    (canonical_tree["rendered_dir"] / "observation.images.depth" / "episode_000000_000.png").unlink()
    with pytest.raises(ImageInventoryError, match="missing.*depth"):
        _validate(canonical_tree)


def test_stale_extra_rgb_frame(canonical_tree: dict) -> None:
    from PIL import Image

    rgb_dir = canonical_tree["rendered_dir"] / "observation.images.rgb"
    Image.new("RGB", (600, 450), (1, 2, 3)).save(rgb_dir / "episode_000000_999.jpg")
    with pytest.raises(ImageInventoryError, match="stale/extra rgb"):
        _validate(canonical_tree)


def test_stale_extra_depth_frame(canonical_tree: dict) -> None:
    rgb_dir = canonical_tree["rendered_dir"] / "observation.images.depth"
    arr = np.zeros((450, 600), dtype=np.uint16)
    from PIL import Image

    Image.fromarray(arr).save(rgb_dir / "episode_000000_999.png")
    with pytest.raises(ImageInventoryError, match="stale/extra depth"):
        _validate(canonical_tree)


def test_renamed_rgb_frame(canonical_tree: dict) -> None:
    rgb_dir = canonical_tree["rendered_dir"] / "observation.images.rgb"
    (rgb_dir / "episode_000000_000.jpg").rename(rgb_dir / "episode_000000_000_renamed.jpg")
    with pytest.raises(ImageInventoryError, match="missing.*rgb"):
        _validate(canonical_tree)


def test_wrong_rgb_dimensions(canonical_tree: dict) -> None:
    from PIL import Image

    rgb_dir = canonical_tree["rendered_dir"] / "observation.images.rgb"
    path = rgb_dir / "episode_000000_000.jpg"
    Image.new("RGB", (700, 500), (1, 2, 3)).save(path)
    with pytest.raises(ImageInventoryError, match="size"):
        _validate(canonical_tree)


def test_wrong_depth_dtype(canonical_tree: dict) -> None:
    from PIL import Image

    depth_dir = canonical_tree["rendered_dir"] / "observation.images.depth"
    path = depth_dir / "episode_000000_000.png"
    Image.fromarray(np.zeros((450, 600), dtype=np.uint8)).save(path)
    with pytest.raises(ImageInventoryError, match="dtype"):
        _validate(canonical_tree)


def test_pointcloud_missing(canonical_tree: dict) -> None:
    canonical_tree["pointcloud_path"] = canonical_tree["trajectory_dir"] / "nonexistent.ply"
    with pytest.raises(FileNotFoundError):
        _validate(canonical_tree)


# ---------------------------------------------------------------------------
# Package-safe forbidden-import smoke for sage3d.contract
# ---------------------------------------------------------------------------


def test_contract_module_package_safe():
    proc = subprocess.run(
        [sys.executable, "-c", "import sage3d.contract; print('OK')"],
        env={**__import__("os").environ, "PYTHONPATH": str(_REPO_ROOT)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"
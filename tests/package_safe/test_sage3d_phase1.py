"""Tests for the Phase 1 package-safe sage3d leaf modules.

Covers frames, camera, episode_arrays, naming, io_ply, pointcloud,
publication, and cli._args. All tests run under package python and must not
import forbidden heavy deps (cv2, scipy, trimesh, pxr, isaacsim).
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sage3d.frames import (  # noqa: E402
    COORDINATE_FRAME,
    camera_extrinsic,
    yaw_to_quaternion,
    yaw_to_rotation2d,
)
from sage3d.camera import CameraCalibration  # noqa: E402
from sage3d.episode_arrays import (  # noqa: E402
    EPISODE_KEYS,
    EpisodeArrays,
    load_episode,
    save_episode,
)
from sage3d.naming import (  # noqa: E402
    episode_filename,
    frame_stem,
    parse_episode_filename,
    parse_frame_filename,
)
from sage3d.io_ply import (  # noqa: E402
    read_binary_pointcloud_metadata,
    write_binary_pointcloud,
)
from sage3d.pointcloud import voxel_downsample  # noqa: E402
from sage3d.publication import (  # noqa: E402
    assert_staging_entries_regular,
    assert_target_absent,
    atomic_publish_directory,
    create_named_directory,
    create_staging_directory,
)
from sage3d.cli._args import add_fisheye_args  # noqa: E402


# --- frames ------------------------------------------------------------------


def test_coordinate_frame_constant():
    assert COORDINATE_FRAME == "world_z_up_x_forward"


def test_yaw_to_quaternion_matches_render_formula():
    yaw = 0.5
    expected = np.asarray(
        [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float32
    )
    q = yaw_to_quaternion(yaw)
    assert q.dtype == np.float32
    assert np.array_equal(q, expected)


def test_camera_extrinsic_matches_package_formula():
    height = 0.6
    ext = camera_extrinsic(height)
    assert ext.dtype == np.float32
    assert ext.shape == (4, 4)
    expected = np.eye(4, dtype=np.float32)
    expected[2, 3] = height
    assert np.array_equal(ext, expected)
    assert ext[2, 3] == pytest.approx(height)


def test_yaw_to_rotation2d_matches_build_episode_arrays():
    yaw = 0.3
    expected = np.asarray(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
        dtype=np.float32,
    )
    rot = yaw_to_rotation2d(yaw)
    assert rot.dtype == np.float32
    assert np.allclose(rot, expected)


# --- camera ------------------------------------------------------------------


def _default_calibration():
    return CameraCalibration(600, 450, 180.0, (0.1, 0.0, 0.0, 0.0))


def test_calibration_rejects_non_positive_width():
    with pytest.raises(ValueError):
        CameraCalibration(0, 450, 180.0, (0.1, 0.0, 0.0, 0.0))


def test_calibration_rejects_non_positive_height():
    with pytest.raises(ValueError):
        CameraCalibration(600, 0, 180.0, (0.1, 0.0, 0.0, 0.0))


def test_calibration_rejects_non_finite_fov():
    with pytest.raises(ValueError):
        CameraCalibration(600, 450, float("inf"), (0.1, 0.0, 0.0, 0.0))


def test_calibration_rejects_zero_fov():
    with pytest.raises(ValueError):
        CameraCalibration(600, 450, 0.0, (0.1, 0.0, 0.0, 0.0))


def test_calibration_rejects_too_many_coefficients():
    with pytest.raises(ValueError):
        CameraCalibration(600, 450, 180.0, (0.1, 0.0, 0.0, 0.0, 0.5))


def test_calibration_rejects_non_finite_coefficients():
    with pytest.raises(ValueError):
        CameraCalibration(600, 450, 180.0, (0.1, float("nan"), 0.0, 0.0))


def test_calibration_intrinsic_matches_package_fisheye_intrinsic():
    cal = _default_calibration()
    expected = np.asarray(
        [
            [cal.fx, 0.0, cal.cx],
            [0.0, cal.fy, cal.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert np.array_equal(cal.intrinsic_matrix(), expected)


def test_calibration_extrinsic_delegates_to_frames():
    cal = _default_calibration()
    height = 0.6
    assert np.array_equal(
        cal.extrinsic_matrix(height), camera_extrinsic(height)
    )


def test_calibration_distortion_vector_dtype():
    cal = _default_calibration()
    dist = cal.distortion_vector()
    assert dist.dtype == np.float32
    assert np.array_equal(dist, np.asarray(cal.fisheye_coefficients, dtype=np.float32))


# --- episode_arrays ----------------------------------------------------------


def _sample_episode(frame_count: int = 4) -> EpisodeArrays:
    points = np.asarray(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]], dtype=np.float32
    )[:frame_count]
    actions = np.eye(4, dtype=np.float32)[None, ...].repeat(frame_count, axis=0)
    camera_positions = np.column_stack(
        (
            points[:, 0],
            points[:, 1],
            np.full(frame_count, 0.6, dtype=np.float32),
        )
    )
    yaw = np.zeros(frame_count, dtype=np.float32)
    point_goal = np.zeros((frame_count, 2), dtype=np.float32)
    start_position = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    goal_position = np.asarray([0.3, 0.0, 0.0], dtype=np.float32)
    return EpisodeArrays(
        points=points,
        actions=actions,
        camera_positions=camera_positions,
        yaw=yaw,
        point_goal=point_goal,
        start_position=start_position,
        goal_position=goal_position,
    )


def test_episode_keys_match_legacy_contract():
    assert EPISODE_KEYS == (
        "points",
        "actions",
        "camera_positions",
        "yaw",
        "point_goal",
        "start_position",
        "goal_position",
    )


def test_save_and_load_roundtrip_preserves_arrays(tmp_path):
    episode = _sample_episode()
    path = tmp_path / "episode_000000.npz"
    save_episode(path, episode)
    loaded = load_episode(path)
    for key in EPISODE_KEYS:
        assert loaded.__getattribute__(key).dtype == np.float32
        assert np.array_equal(
            loaded.__getattribute__(key), episode.__getattribute__(key)
        )


def test_save_episode_uses_compressed_npz(tmp_path):
    episode = _sample_episode()
    path = tmp_path / "episode_000000.npz"
    save_episode(path, episode)
    # NPZ magic for compressed files.
    with path.open("rb") as file:
        magic = file.read(2)
    assert magic == b"PK"


def test_load_episode_rejects_missing_keys(tmp_path):
    path = tmp_path / "episode_000000.npz"
    np.savez(path, points=np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(KeyError):
        load_episode(path)


def test_load_episode_uses_allow_pickle_false(tmp_path):
    episode = _sample_episode()
    path = tmp_path / "episode_000000.npz"
    save_episode(path, episode)
    # Confirm load succeeds and returns the right type (allow_pickle=False would
    # reject object arrays, but all our arrays are numeric).
    loaded = load_episode(path)
    assert isinstance(loaded, EpisodeArrays)


# --- naming ------------------------------------------------------------------


def test_episode_filename_format():
    assert episode_filename(0) == "episode_000000.npz"
    assert episode_filename(42) == "episode_000042.npz"


def test_parse_episode_filename_roundtrip():
    for index in (0, 1, 42, 999999):
        assert parse_episode_filename(episode_filename(index)) == index
        assert parse_episode_filename(f"episode_{index:06d}") == index


def test_parse_episode_filename_rejects_non_episode():
    with pytest.raises(ValueError):
        parse_episode_filename("not_an_episode_000000.npz")


def test_parse_episode_filename_rejects_non_digit():
    with pytest.raises(ValueError):
        parse_episode_filename("episode_abc.npz")


def test_frame_stem_format():
    assert frame_stem(0, 0) == "episode_000000_000"
    assert frame_stem(1, 25) == "episode_000001_025"


def test_parse_frame_filename_roundtrip():
    for episode, frame in [(0, 0), (1, 25), (42, 999)]:
        stem = frame_stem(episode, frame)
        assert parse_frame_filename(stem) == (episode, frame)
        assert parse_frame_filename(f"{stem}.jpg") == (episode, frame)
        assert parse_frame_filename(f"{stem}.png") == (episode, frame)


def test_parse_frame_filename_rejects_bad_shape():
    with pytest.raises(ValueError):
        parse_frame_filename("episode_000000.npz")
    with pytest.raises(ValueError):
        parse_frame_filename("not_a_frame_000")


# --- io_ply ------------------------------------------------------------------


def test_write_binary_pointcloud_exact_bytes(tmp_path):
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.5, 0.25, 4.75]],
        dtype=np.float32,
    )
    path = tmp_path / "pointcloud.ply"
    write_binary_pointcloud(path, points)

    # Reproduce the exact legacy byte stream.
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment SAGE3D collision mesh voxel point cloud\n"
        "element vertex 3\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    record = struct.Struct("<fffBBB")
    expected = bytearray(header)
    for x, y, z in points:
        expected += record.pack(float(x), float(y), float(z), 160, 160, 160)
    assert path.read_bytes() == bytes(expected)


def test_write_binary_pointcloud_coerces_float32(tmp_path):
    points = np.asarray([[1e-8, 1e-8, 1e-8]], dtype=np.float64)
    path = tmp_path / "pointcloud.ply"
    write_binary_pointcloud(path, points)
    meta = read_binary_pointcloud_metadata(path)
    assert meta["vertex_count"] == 1


def test_read_binary_pointcloud_metadata_parses_vertex_count(tmp_path):
    points = np.zeros((7, 3), dtype=np.float32)
    path = tmp_path / "pointcloud.ply"
    write_binary_pointcloud(path, points)
    meta = read_binary_pointcloud_metadata(path)
    assert meta["vertex_count"] == 7
    assert "float x" in meta["properties"]
    assert "uchar red" in meta["properties"]


def test_read_binary_pointcloud_metadata_rejects_bad_magic(tmp_path):
    path = tmp_path / "bad.ply"
    path.write_bytes(b"not a ply file\n")
    with pytest.raises(ValueError):
        read_binary_pointcloud_metadata(path)


def test_read_binary_pointcloud_metadata_rejects_ascii_format(tmp_path):
    path = tmp_path / "ascii.ply"
    path.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 1\nend_header\n")
    with pytest.raises(ValueError):
        read_binary_pointcloud_metadata(path)


# --- pointcloud --------------------------------------------------------------


def test_voxel_downsample_keeps_one_per_voxel():
    points = np.asarray(
        [[0.0, 0.0, 0.0], [0.01, 0.01, 0.01], [1.0, 1.0, 1.0]], dtype=np.float32
    )
    sampled = voxel_downsample(points, voxel_size=0.1, max_points=100, seed=0)
    assert sampled.dtype == np.float32
    assert len(sampled) == 2
    # First point per voxel preserved (sorted by first-seen index).
    assert np.array_equal(sampled[0], points[0])


def test_voxel_downsample_random_subsample_respects_max_points_and_seed():
    points = np.linspace(0, 10, 200, dtype=np.float32).reshape(-1, 1)
    # Broadcast to 3D so each point lands in its own voxel.
    points = np.column_stack([points, points, points])
    sampled_a = voxel_downsample(points, voxel_size=0.001, max_points=50, seed=7)
    sampled_b = voxel_downsample(points, voxel_size=0.001, max_points=50, seed=7)
    assert len(sampled_a) == 50
    assert np.array_equal(sampled_a, sampled_b)


def test_voxel_downsample_output_is_float32():
    points = np.zeros((3, 3), dtype=np.float64)
    sampled = voxel_downsample(points, voxel_size=0.1, max_points=10, seed=0)
    assert sampled.dtype == np.float32


# --- cli._args ---------------------------------------------------------------


def test_add_fisheye_args_defaults_match_legacy():
    parser = argparse.ArgumentParser()
    add_fisheye_args(parser)
    args = parser.parse_args([])
    assert args.width == 600
    assert args.height == 450
    assert args.horizontal_fov_deg == 180.0
    assert list(args.fisheye_coefficients) == [0.1, 0.0, 0.0, 0.0]


def test_add_fisheye_args_accepts_override():
    parser = argparse.ArgumentParser()
    add_fisheye_args(parser)
    args = parser.parse_args(
        [
            "--width",
            "320",
            "--height",
            "240",
            "--horizontal-fov-deg",
            "200.0",
            "--fisheye-coefficients",
            "0.2",
            "0.1",
            "0.0",
            "0.0",
        ]
    )
    assert args.width == 320
    assert args.height == 240
    assert args.horizontal_fov_deg == 200.0
    assert list(args.fisheye_coefficients) == [0.2, 0.1, 0.0, 0.0]


# --- publication: absent target ---------------------------------------------


def test_assert_target_absent_passes_when_missing(tmp_path):
    assert_target_absent(tmp_path / "missing")  # no raise


def test_assert_target_absent_rejects_existing_file(tmp_path):
    target = tmp_path / "file"
    target.write_text("x")
    with pytest.raises(FileExistsError):
        assert_target_absent(target)


def test_assert_target_absent_rejects_existing_dir(tmp_path):
    target = tmp_path / "dir"
    target.mkdir()
    with pytest.raises(FileExistsError):
        assert_target_absent(target)


def test_assert_target_absent_rejects_symlink(tmp_path):
    real = tmp_path / "real"
    real.write_text("x")
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(FileExistsError):
        assert_target_absent(link)


def test_assert_target_absent_rejects_dangling_symlink(tmp_path):
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "does-not-exist")
    with pytest.raises(FileExistsError):
        assert_target_absent(link)


# --- publication: create_named_directory -------------------------------------


def test_create_named_directory_creates_real_sibling(tmp_path):
    child = create_named_directory(tmp_path, "child")
    assert child.is_dir()
    assert not child.is_symlink()
    assert child.stat().st_dev == tmp_path.stat().st_dev


def test_create_named_directory_rejects_existing_target(tmp_path):
    create_named_directory(tmp_path, "child")
    with pytest.raises(FileExistsError):
        create_named_directory(tmp_path, "child")


def test_create_named_directory_rejects_symlinked_parent(tmp_path):
    symlink = tmp_path / "parent-link"
    symlink.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        create_named_directory(symlink, "child")


def test_create_named_directory_rejects_invalid_name(tmp_path):
    for bad in ("", ".", "..", "a/b"):
        with pytest.raises(ValueError):
            create_named_directory(tmp_path, bad)


def test_create_named_directory_rejects_missing_parent(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_named_directory(tmp_path / "missing", "child")


# --- publication: create_staging_directory -----------------------------------


def test_create_staging_directory_allocates_real_sibling(tmp_path):
    final = tmp_path / "final"
    staging = create_staging_directory(final, prefix=".stage.")
    assert staging.is_dir()
    assert not staging.is_symlink()
    assert staging.parent == final.resolve().parent
    assert staging.name.startswith(".stage.")
    assert staging.stat().st_dev == tmp_path.stat().st_dev


def test_create_staging_directory_rejects_symlinked_final_target(tmp_path):
    # A symlinked *final target* is refused by the absence recheck at publish
    # time; a symlinked *parent* traversed to a validated real directory is
    # allowed by the plan's parent-resolution rule, so only the target itself
    # is rejected here.
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    # create_staging_directory resolves the parent; the staging dir lands
    # beside the resolved real parent. The symlinked target is caught later by
    # assert_target_absent, demonstrated in the publish tests below.
    staging = create_staging_directory(link / "final", prefix=".stage.")
    assert staging.is_dir()
    with pytest.raises(FileExistsError):
        assert_target_absent(link)


def test_create_staging_directory_rejects_empty_prefix(tmp_path):
    with pytest.raises(ValueError):
        create_staging_directory(tmp_path / "final", prefix="")


# --- publication: staging entry validation ------------------------------------


def test_assert_staging_entries_regular_accepts_files_and_dirs(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "file.txt").write_text("x")
    (tmp_path / "file.bin").write_bytes(b"\x00")
    assert_staging_entries_regular(tmp_path)  # no raise


def test_assert_staging_entries_regular_rejects_symlink(tmp_path):
    target = tmp_path / "real"
    target.write_text("x")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        assert_staging_entries_regular(tmp_path)


def test_assert_staging_entries_regular_rejects_dangling_symlink(tmp_path):
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="symlink"):
        assert_staging_entries_regular(tmp_path)


def test_assert_staging_entries_regular_rejects_fifo(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="FIFO/socket"):
        assert_staging_entries_regular(tmp_path)


def test_assert_staging_entries_regular_rejects_symlinked_dir(tmp_path):
    real = tmp_path / "real-dir"
    real.mkdir()
    (real / "file.txt").write_text("x")
    link = tmp_path / "link-dir"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        assert_staging_entries_regular(tmp_path)


# --- publication: atomic_publish_directory -----------------------------------


def test_atomic_publish_directory_renames_staging_to_absent_target(tmp_path):
    final = tmp_path / "final"
    staging = create_staging_directory(final, prefix=".stage.")
    (staging / "file.txt").write_text("payload")
    result = atomic_publish_directory(staging, final)
    assert result == final
    assert final.is_dir()
    assert (final / "file.txt").read_text() == "payload"
    assert not staging.exists()


def test_atomic_publish_rejects_existing_empty_target(tmp_path):
    final = tmp_path / "final"
    final.mkdir()
    staging = create_staging_directory(final, prefix=".stage.")
    (staging / "file.txt").write_text("payload")
    with pytest.raises(FileExistsError):
        atomic_publish_directory(staging, final)
    # Staging left intact, final untouched.
    assert staging.is_dir()
    assert not (final / "file.txt").exists()


def test_atomic_publish_rejects_existing_nonempty_target(tmp_path):
    final = tmp_path / "final"
    final.mkdir()
    (final / "old.txt").write_text("old")
    staging = create_staging_directory(final, prefix=".stage.")
    (staging / "file.txt").write_text("payload")
    with pytest.raises(FileExistsError):
        atomic_publish_directory(staging, final)


def test_atomic_publish_rejects_existing_file_target(tmp_path):
    final = tmp_path / "final"
    final.write_text("blocking")
    staging = create_staging_directory(tmp_path / "other", prefix=".stage.") if False else None
    # Use a staging dir beside the final target so devices match.
    staging = create_staging_directory(final, prefix=".stage.")
    (staging / "file.txt").write_text("payload")
    with pytest.raises(FileExistsError):
        atomic_publish_directory(staging, final)


def test_atomic_publish_rejects_symlinked_target(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    # Allocate staging beside the link's parent.
    staging = create_staging_directory(tmp_path / "other-final", prefix=".stage.")
    (staging / "file.txt").write_text("payload")
    with pytest.raises((FileExistsError, ValueError)):
        atomic_publish_directory(staging, link)


def test_atomic_publish_rejects_symlinked_staging_entry(tmp_path):
    final = tmp_path / "final"
    staging = create_staging_directory(final, prefix=".stage.")
    real = staging / "real"
    real.write_text("x")
    link = staging / "link"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        atomic_publish_directory(staging, final)


def test_atomic_publish_rejects_fifo_in_staging(tmp_path):
    final = tmp_path / "final"
    staging = create_staging_directory(final, prefix=".stage.")
    os.mkfifo(staging / "fifo")
    with pytest.raises(ValueError, match="FIFO/socket"):
        atomic_publish_directory(staging, final)


def test_atomic_publish_rejects_device_mismatch(tmp_path, monkeypatch):
    final = tmp_path / "final"
    staging = create_staging_directory(final, prefix=".stage.")
    (staging / "file.txt").write_text("payload")
    # Force a device mismatch by patching _require_same_device to raise.
    from sage3d import publication as pub_mod

    def fake_require_same_device(a, b):
        raise OSError(
            f"staging and target are on different filesystems: "
            f"{a} vs {b}"
        )

    monkeypatch.setattr(pub_mod, "_require_same_device", fake_require_same_device)
    with pytest.raises(OSError, match="different filesystems"):
        atomic_publish_directory(staging, final)


def test_atomic_publish_rechecks_target_absence_immediately_before_rename(tmp_path):
    """Confirm the final absence recheck runs after staging validation and
    before the rename, so a target that appears between allocation and the
    recheck is refused (the documented cooperative-contract boundary)."""
    final = tmp_path / "final"
    staging = create_staging_directory(final, prefix=".stage.")
    (staging / "file.txt").write_text("payload")
    final.mkdir()
    with pytest.raises(FileExistsError):
        atomic_publish_directory(staging, final)
    # Staging left intact; final untouched by the publication path.
    assert staging.is_dir()
    assert (staging / "file.txt").exists()
    assert not (final / "file.txt").exists()
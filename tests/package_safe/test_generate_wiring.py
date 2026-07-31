"""Focused tests for issue #9 generate-wiring migration call sites.

Each migrated call site in ``generate_sage3d_trajectories.py`` is asserted to
produce byte-for-byte / array-for-array identical output to the legacy inlined
implementation it replaced. Package-safe: numpy + stdlib only.
"""

from __future__ import annotations

import io
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sage3d.episode_arrays import EPISODE_KEYS, EpisodeArrays, save_episode  # noqa: E402
from sage3d.io_ply import write_binary_pointcloud  # noqa: E402
from sage3d.naming import episode_filename  # noqa: E402
from sage3d.pointcloud import voxel_downsample  # noqa: E402


# --- legacy inlined implementations (verbatim from pre-migration generate_*) -


def _legacy_voxel_downsample(points, voxel_size, max_points, seed):
    voxel = np.floor(points / voxel_size).astype(np.int64)
    _, indices = np.unique(voxel, axis=0, return_index=True)
    sampled = points[np.sort(indices)]
    if len(sampled) > max_points:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(sampled), max_points, replace=False))
        sampled = sampled[selected]
    return sampled.astype(np.float32)


_PLY_RECORD = struct.Struct("<fffBBB")


def _legacy_write_binary_pointcloud(path: Path, points: np.ndarray) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment SAGE3D collision mesh voxel point cloud\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as file:
        file.write(header)
        for x, y, z in points:
            file.write(_PLY_RECORD.pack(float(x), float(y), float(z), 160, 160, 160))


def _legacy_save_episode(path: Path, episode: dict) -> None:
    np.savez_compressed(
        path,
        points=episode["points"],
        actions=episode["actions"],
        camera_positions=episode["camera_positions"],
        yaw=episode["yaw"],
        point_goal=episode["point_goal"],
        start_position=np.asarray(episode["start_position"], dtype=np.float32),
        goal_position=np.asarray(episode["goal_position"], dtype=np.float32),
    )


def _legacy_episode_filename(episode_index: int) -> str:
    return f"episode_{episode_index:06d}.npz"


# --- helpers ----------------------------------------------------------------


def _sample_episode_dict(n: int = 10) -> dict:
    rng = np.random.default_rng(123)
    points = rng.standard_normal((n, 2)).astype(np.float32)
    actions = rng.standard_normal((n, 4, 4)).astype(np.float32)
    camera_positions = rng.standard_normal((n, 3)).astype(np.float32)
    yaw = rng.standard_normal(n).astype(np.float32)
    point_goal = rng.standard_normal((n, 2)).astype(np.float32)
    start_position = rng.standard_normal(2).astype(np.float32)
    goal_position = rng.standard_normal(2).astype(np.float32)
    return {
        "points": points,
        "actions": actions,
        "camera_positions": camera_positions,
        "yaw": yaw,
        "point_goal": point_goal,
        "start_position": start_position,
        "goal_position": goal_position,
    }


def _npz_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _npz_entries(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


# --- migrated call site: episode filename -----------------------------------


def test_episode_filename_matches_legacy_format_string():
    for index in (0, 1, 42, 999999):
        assert episode_filename(index) == _legacy_episode_filename(index)


# --- migrated call site: save_episode via EpisodeArrays ---------------------


def test_save_episode_npz_bytes_match_legacy_savez_compressed(tmp_path):
    ep = _sample_episode_dict(12)
    legacy_path = tmp_path / "legacy.npz"
    prod_path = tmp_path / "prod.npz"
    _legacy_save_episode(legacy_path, ep)
    save_episode(
        prod_path,
        EpisodeArrays(
            points=ep["points"],
            actions=ep["actions"],
            camera_positions=ep["camera_positions"],
            yaw=ep["yaw"],
            point_goal=ep["point_goal"],
            start_position=ep["start_position"],
            goal_position=ep["goal_position"],
        ),
    )
    # Byte-for-byte identical NPZ.
    assert _npz_bytes(legacy_path) == _npz_bytes(prod_path)
    # Same internal zip entries.
    assert _npz_entries(legacy_path) == _npz_entries(prod_path)


def test_save_episode_keys_match_legacy_contract():
    ep = _sample_episode_dict(5)
    legacy_path = Path("/tmp/_gen_wiring_legacy_check.npz")
    prod_path = Path("/tmp/_gen_wiring_prod_check.npz")
    try:
        _legacy_save_episode(legacy_path, ep)
        save_episode(
            prod_path,
            EpisodeArrays(**{k: ep[k] for k in EPISODE_KEYS}),
        )
        with np.load(legacy_path) as a, np.load(prod_path) as b:
            assert set(a.files) == set(b.files) == set(EPISODE_KEYS)
    finally:
        legacy_path.unlink(missing_ok=True)
        prod_path.unlink(missing_ok=True)


# --- migrated call site: voxel_downsample -----------------------------------


def test_voxel_downsample_matches_legacy_algorithm():
    rng = np.random.default_rng(7)
    points = rng.standard_normal((500, 3)).astype(np.float64)
    for voxel_size, max_points, seed in (
        (0.05, 100_000, 20260720),
        (0.1, 50, 99),
        (1.0, 1000, 1),
    ):
        legacy = _legacy_voxel_downsample(points, voxel_size, max_points, seed)
        prod = voxel_downsample(points, voxel_size, max_points, seed)
        assert np.array_equal(legacy, prod), f"mismatch for ({voxel_size},{max_points},{seed})"
        assert prod.dtype == np.float32


def test_voxel_downsample_preserves_legacy_subsample_order():
    # When subsampling is triggered, the legacy uses a default_rng with the
    # given seed and sorts the choice; the production helper must match exactly.
    rng = np.random.default_rng(3)
    points = rng.standard_normal((2000, 3)).astype(np.float64)
    legacy = _legacy_voxel_downsample(points, 0.05, 100, 42)
    prod = voxel_downsample(points, 0.05, 100, 42)
    assert np.array_equal(legacy, prod)
    assert len(prod) == 100


# --- migrated call site: write_binary_pointcloud ----------------------------


def test_write_binary_pointcloud_bytes_match_legacy_writer(tmp_path):
    rng = np.random.default_rng(11)
    points = rng.standard_normal((120, 3)).astype(np.float32)
    legacy_path = tmp_path / "legacy.ply"
    prod_path = tmp_path / "prod.ply"
    _legacy_write_binary_pointcloud(legacy_path, points)
    write_binary_pointcloud(prod_path, points)
    assert legacy_path.read_bytes() == prod_path.read_bytes()


def test_write_binary_pointcloud_matches_legacy_with_float64_input(tmp_path):
    # The legacy writer received float32 (voxel_downsample output). Verify the
    # production writer's float32 cast does not change bytes vs a legacy call
    # given the same float32 input.
    rng = np.random.default_rng(5)
    points_f32 = rng.standard_normal((50, 3)).astype(np.float32)
    points_f64 = points_f32.astype(np.float64)
    legacy_path = tmp_path / "legacy.ply"
    prod_path = tmp_path / "prod.ply"
    _legacy_write_binary_pointcloud(legacy_path, points_f32)
    write_binary_pointcloud(prod_path, points_f64)
    assert legacy_path.read_bytes() == prod_path.read_bytes()
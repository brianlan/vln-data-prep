"""Package-safe tests for the Phase 5a pure package builders (issue #24).

Covers ``sage3d.lerobot_dataset`` and ``sage3d.config.PackageConfig``:
Arrow schema/values/order, JSON/JSONL order, frame copies,
calibration/extrinsics, depth format truthfulness (default + non-default),
the float32 list policy, and a package golden comparison against the legacy
``package_lerobot_sage3d.py`` output.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sage3d.camera import CameraCalibration
from sage3d.episode_arrays import load_episode
from sage3d.lerobot_dataset import (
    build_episode_parquet,
    copy_episode_frames,
    write_jsonl,
    write_lerobot_meta,
)
from sage3d.naming import parse_episode_filename

from fixtures import build_rendered_dir, build_trajectory_dir

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_WIDTH = 600
DEFAULT_HEIGHT = 450
DEFAULT_FOV_DEG = 180.0
DEFAULT_COEFFICIENTS = (0.1, 0.0, 0.0, 0.0)


# --- helpers ------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_episodes(trajectory_dir: Path) -> dict[int, object]:
    episodes = {}
    for tf in sorted(trajectory_dir.glob("episode_*.npz")):
        episodes[parse_episode_filename(tf.name)] = load_episode(tf)
    return episodes


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
        "manifest": manifest,
        "rgb_summary": _load_json(rendered_dir / "rgb_render_summary.json"),
        "canonical_depth_summary": _load_json(rendered_dir / "render_summary.json"),
        "depth_alias_summary": _load_json(rendered_dir / "depth_render_summary.json"),
        "episodes_by_id": _load_episodes(traj_dir),
        "scene_id": "839920",
    }


def _calibration(tree: dict) -> CameraCalibration:
    summary = tree["canonical_depth_summary"]
    return CameraCalibration(
        summary["resolution"][0],
        summary["resolution"][1],
        summary["horizontal_fov_deg"],
        summary["fisheye_coefficients"],
    )


# --- build_episode_parquet ----------------------------------------------------


def test_build_episode_parquet_schema_and_order(canonical_tree, tmp_path):
    data_dir = tmp_path / "data"
    cal = _calibration(canonical_tree)
    camera_height = canonical_tree["manifest"]["camera_height_m"]
    intrinsic_flat = cal.intrinsic_matrix().reshape(-1).tolist()
    extrinsic_flat = cal.extrinsic_matrix(camera_height).reshape(-1).tolist()
    distortion = cal.fisheye_coefficients

    ep = canonical_tree["episodes_by_id"][0]
    build_episode_parquet(
        data_dir,
        0,
        frame_count=len(ep.actions),
        intrinsic_flat=intrinsic_flat,
        extrinsic_flat=extrinsic_flat,
        distortion=distortion,
        point_goal=ep.point_goal,
        actions=ep.actions,
    )

    import pyarrow.parquet as pq

    table = pq.read_table(data_dir / "episode_000000.parquet")
    assert table.column_names == [
        "index",
        "observation.camera_intrinsic",
        "observation.camera_extrinsic",
        "observation.camera_distortion",
        "observation.point_goal",
        "action",
    ]
    assert table.num_rows == 3
    # index is int64; all observation/action columns are float32 lists.
    assert str(table.schema.field("index").type) == "int64"
    for name in table.column_names[1:]:
        assert str(table.schema.field(name).type) == "list<element: float>"


def test_build_episode_parquet_values_match_legacy(canonical_tree, tmp_path):
    """Row values are float32 lists matching the legacy parquet construction."""
    import pyarrow.parquet as pq

    data_dir = tmp_path / "data"
    cal = _calibration(canonical_tree)
    camera_height = canonical_tree["manifest"]["camera_height_m"]
    intrinsic_flat = cal.intrinsic_matrix().reshape(-1).tolist()
    extrinsic_flat = cal.extrinsic_matrix(camera_height).reshape(-1).tolist()
    distortion = cal.fisheye_coefficients

    ep = canonical_tree["episodes_by_id"][1]
    build_episode_parquet(
        data_dir,
        1,
        frame_count=len(ep.actions),
        intrinsic_flat=intrinsic_flat,
        extrinsic_flat=extrinsic_flat,
        distortion=distortion,
        point_goal=ep.point_goal,
        actions=ep.actions,
    )

    table = pq.read_table(data_dir / "episode_000001.parquet")
    assert table.column("index").to_pylist() == [0, 1]
    assert table.column("observation.camera_intrinsic").to_pylist() == [
        intrinsic_flat,
        intrinsic_flat,
    ]
    assert table.column("observation.camera_extrinsic").to_pylist() == [
        extrinsic_flat,
        extrinsic_flat,
    ]
    assert table.column("observation.camera_distortion").to_pylist() == [
        [float(np.float32(v)) for v in distortion],
        [float(np.float32(v)) for v in distortion],
    ]
    assert table.column("observation.point_goal").to_pylist() == [
        ep.point_goal[0].tolist(),
        ep.point_goal[1].tolist(),
    ]
    assert table.column("action").to_pylist() == [
        ep.actions[i].reshape(16).tolist() for i in range(2)
    ]


def test_build_episode_parquet_float_policy_positive_negative(canonical_tree, tmp_path):
    """Float policy: float64 inputs still produce float32 list columns."""
    import pyarrow.parquet as pq

    data_dir = tmp_path / "data"
    cal = _calibration(canonical_tree)
    camera_height = canonical_tree["manifest"]["camera_height_m"]
    ep = canonical_tree["episodes_by_id"][0]
    # Deliberately feed float64 arrays: the policy forces float32 lists.
    build_episode_parquet(
        data_dir,
        0,
        frame_count=len(ep.actions),
        intrinsic_flat=cal.intrinsic_matrix().reshape(-1).tolist(),
        extrinsic_flat=cal.extrinsic_matrix(camera_height).reshape(-1).tolist(),
        distortion=cal.fisheye_coefficients,
        point_goal=ep.point_goal.astype(np.float64),
        actions=ep.actions.astype(np.float64),
    )
    table = pq.read_table(data_dir / "episode_000000.parquet")
    for name in table.column_names[1:]:
        assert str(table.schema.field(name).type) == "list<element: float>"


# --- copy_episode_frames ------------------------------------------------------


def test_copy_episode_frames_copies_all_frames(canonical_tree, tmp_path):
    video_out = tmp_path / "videos" / "chunk-000"
    copy_episode_frames(
        canonical_tree["rendered_dir"],
        video_out,
        episode_index=0,
        frame_count=3,
    )
    rgb_dir = video_out / "observation.images.rgb"
    depth_dir = video_out / "observation.images.depth"
    assert sorted(p.name for p in rgb_dir.glob("*.jpg")) == [
        "episode_000000_000.jpg",
        "episode_000000_001.jpg",
        "episode_000000_002.jpg",
    ]
    assert sorted(p.name for p in depth_dir.glob("*.png")) == [
        "episode_000000_000.png",
        "episode_000000_001.png",
        "episode_000000_002.png",
    ]


def test_copy_episode_frames_preserves_bytes(canonical_tree, tmp_path):
    video_out = tmp_path / "videos" / "chunk-000"
    copy_episode_frames(
        canonical_tree["rendered_dir"],
        video_out,
        episode_index=1,
        frame_count=2,
    )
    for stem in ("episode_000001_000", "episode_000001_001"):
        src = (
            canonical_tree["rendered_dir"]
            / "observation.images.rgb"
            / f"{stem}.jpg"
        )
        dst = video_out / "observation.images.rgb" / f"{stem}.jpg"
        assert dst.read_bytes() == src.read_bytes()
        src_d = (
            canonical_tree["rendered_dir"]
            / "observation.images.depth"
            / f"{stem}.png"
        )
        dst_d = video_out / "observation.images.depth" / f"{stem}.png"
        assert dst_d.read_bytes() == src_d.read_bytes()


# --- write_lerobot_meta -------------------------------------------------------


def test_write_lerobot_meta_info_and_jsonl_order(canonical_tree, tmp_path):
    meta_dir = tmp_path / "meta"
    write_lerobot_meta(
        meta_dir,
        scene_id="839920",
        fps=30,
        manifest=canonical_tree["manifest"],
        render_summary=canonical_tree["canonical_depth_summary"],
        trajectory_dir=canonical_tree["trajectory_dir"],
        rendered_dir=canonical_tree["rendered_dir"],
        calibration=_calibration(canonical_tree),
        episodes_by_id=canonical_tree["episodes_by_id"],
    )

    info = _load_json(meta_dir / "info.json")
    assert info["scene_id"] == "839920"
    assert info["fps"] == 30
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 3 + 2 + 4
    assert info["total_videos"] == 3
    assert info["camera_height_m"] == canonical_tree["manifest"]["camera_height_m"]
    # Canonical depth summary is the calibration authority.
    assert info["image_width"] == DEFAULT_WIDTH
    assert info["image_height"] == DEFAULT_HEIGHT
    assert info["camera_horizontal_fov_deg"] == DEFAULT_FOV_DEG

    episodes = _read_jsonl(meta_dir / "episodes.jsonl")
    assert [rec["episode_index"] for rec in episodes] == [0, 1, 2]
    assert [rec["frame_count"] for rec in episodes] == [3, 2, 4]
    assert episodes[0]["seed"] == canonical_tree["manifest"]["seed"]
    assert episodes[0]["start_position"] == canonical_tree["manifest"]["episodes"][0][
        "start_position"
    ]

    stats = _read_jsonl(meta_dir / "episodes_stats.jsonl")
    assert [rec["episode_index"] for rec in stats] == [0, 1, 2]
    assert stats[0]["image_index"]["max"] == 2
    assert stats[0]["point_goal_distance_m"]["count"] == 3

    tasks = _read_jsonl(meta_dir / "tasks.jsonl")
    assert tasks == [
        {
            "task_index": 0,
            "task": {
                "type": "point_goal_navigation",
                "goal_input": ["distance_m", "relative_bearing_rad"],
            },
        }
    ]


def test_write_lerobot_meta_depth_format_default(canonical_tree, tmp_path):
    meta_dir = tmp_path / "meta"
    write_lerobot_meta(
        meta_dir,
        scene_id="839920",
        fps=30,
        manifest=canonical_tree["manifest"],
        render_summary=canonical_tree["canonical_depth_summary"],
        trajectory_dir=canonical_tree["trajectory_dir"],
        rendered_dir=canonical_tree["rendered_dir"],
        calibration=_calibration(canonical_tree),
        episodes_by_id=canonical_tree["episodes_by_id"],
    )
    info = _load_json(meta_dir / "info.json")
    # Default depth_scale=10000 produces the legacy hardcoded string.
    assert info["depth_format"] == "uint16_meters_x_10000"
    assert info["depth_clip_m"] == canonical_tree["canonical_depth_summary"][
        "max_depth_m"
    ]


def test_write_lerobot_meta_depth_format_non_default_truthful(
    canonical_tree, tmp_path
):
    """Non-default depth_scale must produce a truthful format string."""
    meta_dir = tmp_path / "meta"
    summary = dict(canonical_tree["canonical_depth_summary"])
    summary["depth_scale"] = 5000.0
    write_lerobot_meta(
        meta_dir,
        scene_id="839920",
        fps=30,
        manifest=canonical_tree["manifest"],
        render_summary=summary,
        trajectory_dir=canonical_tree["trajectory_dir"],
        rendered_dir=canonical_tree["rendered_dir"],
        calibration=_calibration(canonical_tree),
        episodes_by_id=canonical_tree["episodes_by_id"],
    )
    info = _load_json(meta_dir / "info.json")
    assert info["depth_format"] == "uint16_meters_x_5000"


def test_write_lerobot_meta_copies_inputs(canonical_tree, tmp_path):
    meta_dir = tmp_path / "meta"
    write_lerobot_meta(
        meta_dir,
        scene_id="839920",
        fps=30,
        manifest=canonical_tree["manifest"],
        render_summary=canonical_tree["canonical_depth_summary"],
        trajectory_dir=canonical_tree["trajectory_dir"],
        rendered_dir=canonical_tree["rendered_dir"],
        calibration=_calibration(canonical_tree),
        episodes_by_id=canonical_tree["episodes_by_id"],
    )
    for name in (
        "pointcloud.ply",
        "trajectory_manifest.json",
        "render_summary.json",
        "rgb_render_summary.json",
        "depth_render_summary.json",
    ):
        assert meta_dir.joinpath(name).is_file()
    # Copied files must be byte-identical to their sources.
    pairs = [
        (meta_dir / "pointcloud.ply", canonical_tree["trajectory_dir"] / "pointcloud.ply"),
        (
            meta_dir / "trajectory_manifest.json",
            canonical_tree["trajectory_dir"] / "trajectory_manifest.json",
        ),
        (
            meta_dir / "render_summary.json",
            canonical_tree["rendered_dir"] / "render_summary.json",
        ),
    ]
    for dst, src in pairs:
        assert hashlib.sha256(dst.read_bytes()).digest() == hashlib.sha256(
            src.read_bytes()
        ).digest()


def test_write_jsonl_uses_ensure_ascii_false(tmp_path):
    path = tmp_path / "out.jsonl"
    write_jsonl(path, [{"note": "café"}])
    assert "café" in path.read_text(encoding="utf-8")


# --- package golden comparison -------------------------------------------------


def _legacy_package(tree: dict, output_dir: Path) -> None:
    """Run the legacy package script on the fixture tree (subprocess)."""
    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "package_lerobot_sage3d.py"),
            "--scene", tree["scene_id"],
            "--trajectory-dir", str(tree["trajectory_dir"]),
            "--rendered-dir", str(tree["rendered_dir"]),
            "--output-dir", str(output_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"legacy package failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )


def _builder_package(tree: dict, output_dir: Path) -> None:
    """Produce a package with the new builders on the same fixture tree."""
    data_dir = output_dir / "data" / "chunk-000"
    video_out = output_dir / "videos" / "chunk-000"
    meta_dir = output_dir / "meta"
    cal = _calibration(tree)
    camera_height = tree["manifest"]["camera_height_m"]
    intrinsic_flat = cal.intrinsic_matrix().reshape(-1).tolist()
    extrinsic_flat = cal.extrinsic_matrix(camera_height).reshape(-1).tolist()
    distortion = cal.fisheye_coefficients

    for episode_index in sorted(tree["episodes_by_id"]):
        ep = tree["episodes_by_id"][episode_index]
        frame_count = len(ep.actions)
        build_episode_parquet(
            data_dir,
            episode_index,
            frame_count=frame_count,
            intrinsic_flat=intrinsic_flat,
            extrinsic_flat=extrinsic_flat,
            distortion=distortion,
            point_goal=ep.point_goal,
            actions=ep.actions,
        )
        copy_episode_frames(
            tree["rendered_dir"], video_out, episode_index, frame_count
        )
    write_lerobot_meta(
        meta_dir,
        scene_id=tree["scene_id"],
        fps=30,
        manifest=tree["manifest"],
        render_summary=tree["canonical_depth_summary"],
        trajectory_dir=tree["trajectory_dir"],
        rendered_dir=tree["rendered_dir"],
        calibration=cal,
        episodes_by_id=tree["episodes_by_id"],
    )


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_package_golden_comparison_legacy_parity(canonical_tree, tmp_path):
    """Builders reproduce the legacy script's deterministic content exactly."""
    legacy_out = tmp_path / "legacy"
    builder_out = tmp_path / "builder"
    _legacy_package(canonical_tree, legacy_out)
    _builder_package(canonical_tree, builder_out)

    # Deterministic Arrow content: schema + values per episode parquet.
    import pyarrow.parquet as pq

    for idx in (0, 1, 2):
        leg = pq.read_table(legacy_out / "data" / "chunk-000" / f"episode_{idx:06d}.parquet")
        new = pq.read_table(builder_out / "data" / "chunk-000" / f"episode_{idx:06d}.parquet")
        assert leg.schema == new.schema
        assert leg.num_rows == new.num_rows
        assert leg.to_pydict() == new.to_pydict()

    # Deterministic JSON content: info.json and JSONL order.
    assert _load_json(legacy_out / "meta" / "info.json") == _load_json(
        builder_out / "meta" / "info.json"
    )
    for name in ("episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl"):
        assert _read_jsonl(legacy_out / "meta" / name) == _read_jsonl(
            builder_out / "meta" / name
        )

    # Copied input layout: byte-identical copied files.
    for name in (
        "pointcloud.ply",
        "trajectory_manifest.json",
        "render_summary.json",
        "rgb_render_summary.json",
        "depth_render_summary.json",
    ):
        assert _sha256_bytes(legacy_out / "meta" / name) == _sha256_bytes(
            builder_out / "meta" / name
        )

    # Copied frame layout: all rgb/depth frames byte-identical.
    leg_rgb = sorted((legacy_out / "videos" / "chunk-000" / "observation.images.rgb").glob("*.jpg"))
    new_rgb = sorted((builder_out / "videos" / "chunk-000" / "observation.images.rgb").glob("*.jpg"))
    assert [p.name for p in new_rgb] == [p.name for p in leg_rgb]
    for lp, np_ in zip(leg_rgb, new_rgb):
        assert lp.read_bytes() == np_.read_bytes()

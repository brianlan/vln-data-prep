"""Pure package-safe LeRobot-style dataset builders and staged validator.

Phase 5a extracts the deterministic dataset construction from
``package_lerobot_sage3d.py`` into package-safe builders so the package
boundary can be validated independently; Phase 5b adds the staged package
dataset validator (:func:`validate_packaged_dataset`) plus the shared
packaged-artifact primitives that ``scripts/check_package.py`` reuses:

- :func:`build_episode_parquet` writes one episode's Arrow parquet with the
  exact legacy schema and column order.
- :func:`copy_episode_frames` copies one episode's RGB/depth frames from the
  finalized render root.
- :func:`write_lerobot_meta` writes the meta directory: copied pointcloud /
  manifest / render summaries, ``info.json``, and the episodes / tasks /
  episodes_stats JSONL records.
- :func:`validate_packaged_dataset` validates a complete package staging tree
  against the current trajectory/render inputs before publication.
- :func:`package` orchestrates publication: source contract validation,
  internally allocated sibling staging, complete build, staged validation, and
  absent-target atomic rename.

The builders consume only finalized render artifacts and the authoritative
depth/calibration summaries. They are non-destructive: they never overwrite an
existing publication target (the staging/rename contract is owned by
:func:`package`). The validator checks inventory, Arrow/JSON content,
copied-input checksums, calibration/extrinsics, and depth metadata without
invoking CLI or publication behavior. The project-specific LeRobot-style
layout is preserved exactly; standard LeRobot compatibility is not claimed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sage3d.camera import CameraCalibration
from sage3d.config import PackageConfig
from sage3d.contract import validate_pipeline_contract
from sage3d.episode_arrays import EpisodeArrays, load_episode
from sage3d.naming import frame_stem, parse_episode_filename
from sage3d.publication import (
    atomic_publish_directory,
    create_staging_directory,
)

# Parquet columns that must be present with float32 list type in every episode.
PARQUET_REQUIRED_COLUMNS = (
    "index",
    "observation.camera_intrinsic",
    "observation.camera_extrinsic",
    "observation.camera_distortion",
    "observation.point_goal",
    "action",
)

# Meta files that must exist in a packaged dataset.
REQUIRED_META_FILES = {
    "info.json",
    "episodes.jsonl",
    "tasks.jsonl",
    "trajectory_manifest.json",
    "render_summary.json",
    "rgb_render_summary.json",
    "depth_render_summary.json",
    "pointcloud.ply",
}


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write JSONL records with ``ensure_ascii=False``, matching legacy output."""
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# --- episode parquet ----------------------------------------------------------


def build_episode_parquet(
    data_dir: Path,
    episode_index: int,
    *,
    frame_count: int,
    intrinsic_flat: Sequence[float],
    extrinsic_flat: Sequence[float],
    distortion: Sequence[float],
    point_goal: np.ndarray,
    actions: np.ndarray,
) -> None:
    """Write one episode's Arrow parquet with the exact legacy schema/order.

    ``data_dir`` is ``<output>/data/chunk-000``. The table columns, order,
    and float32 list policy match ``package_lerobot_sage3d.py`` exactly.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "index": pa.array(range(frame_count), type=pa.int64()),
            "observation.camera_intrinsic": pa.array(
                [intrinsic_flat] * frame_count,
                type=pa.list_(pa.float32()),
            ),
            "observation.camera_extrinsic": pa.array(
                [extrinsic_flat] * frame_count,
                type=pa.list_(pa.float32()),
            ),
            "observation.camera_distortion": pa.array(
                [distortion] * frame_count,
                type=pa.list_(pa.float32()),
            ),
            "observation.point_goal": pa.array(
                point_goal.tolist(),
                type=pa.list_(pa.float32()),
            ),
            "action": pa.array(
                actions.reshape(frame_count, 16).tolist(),
                type=pa.list_(pa.float32()),
            ),
        }
    )
    pq.write_table(table, data_dir / f"episode_{episode_index:06d}.parquet")


# --- frame copies -------------------------------------------------------------


def copy_episode_frames(
    rendered_dir: Path,
    video_output_dir: Path,
    episode_index: int,
    frame_count: int,
) -> None:
    """Copy one episode's RGB JPEGs and depth PNGs from the finalized render root.

    ``video_output_dir`` is ``<output>/videos/chunk-000``; the
    ``observation.images.rgb`` / ``observation.images.depth`` subdirectories
    are created under it.
    """
    rgb_output_dir = video_output_dir / "observation.images.rgb"
    depth_output_dir = video_output_dir / "observation.images.depth"
    rgb_output_dir.mkdir(parents=True, exist_ok=True)
    depth_output_dir.mkdir(parents=True, exist_ok=True)
    for frame_index in range(frame_count):
        stem = frame_stem(episode_index, frame_index)
        shutil.copy2(
            rendered_dir / "observation.images.rgb" / f"{stem}.jpg",
            rgb_output_dir / f"{stem}.jpg",
        )
        shutil.copy2(
            rendered_dir / "observation.images.depth" / f"{stem}.png",
            depth_output_dir / f"{stem}.png",
        )


# --- meta ---------------------------------------------------------------------


def write_lerobot_meta(
    meta_dir: Path,
    *,
    scene_id: str,
    fps: int,
    manifest: dict,
    render_summary: dict,
    trajectory_dir: Path,
    rendered_dir: Path,
    calibration: CameraCalibration,
    episodes_by_id: dict[int, EpisodeArrays],
) -> None:
    """Write the meta directory from finalized render artifacts.

    Copies ``pointcloud.ply`` / ``trajectory_manifest.json`` and the three
    render summaries into ``meta_dir``, then writes ``info.json``,
    ``episodes.jsonl``, ``tasks.jsonl``, and ``episodes_stats.jsonl`` with the
    exact legacy key order and values. ``render_summary.json`` is the
    authoritative calibration/depth summary.
    """
    meta_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trajectory_dir / "pointcloud.ply", meta_dir / "pointcloud.ply")
    shutil.copy2(
        trajectory_dir / "trajectory_manifest.json",
        meta_dir / "trajectory_manifest.json",
    )
    shutil.copy2(rendered_dir / "render_summary.json", meta_dir / "render_summary.json")
    shutil.copy2(
        rendered_dir / "rgb_render_summary.json", meta_dir / "rgb_render_summary.json"
    )
    shutil.copy2(
        rendered_dir / "depth_render_summary.json",
        meta_dir / "depth_render_summary.json",
    )

    summary_width, summary_height = render_summary["resolution"]
    summary_fov = render_summary["horizontal_fov_deg"]
    distortion = calibration.fisheye_coefficients
    camera_height = manifest["camera_height_m"]

    episode_records = []
    episode_stats_records = []
    total_frames = 0
    for episode_index in sorted(episodes_by_id):
        episode = episodes_by_id[episode_index]
        point_goal = episode.point_goal
        frame_count = len(episode.actions)
        manifest_episode = manifest["episodes"][episode_index]

        episode_records.append(
            {
                "episode_index": episode_index,
                "task_index": 0,
                "task_type": "point_goal_navigation",
                "coordinate_frame": "world_z_up_x_forward",
                "point_goal_representation": [
                    "distance_m",
                    "relative_bearing_rad",
                ],
                "start_position": manifest_episode["start_position"],
                "goal_position": manifest_episode["goal_position"],
                "path_length_m": manifest_episode["path_length_m"],
                "minimum_clearance_m": manifest_episode["minimum_clearance_m"],
                "frame_count": frame_count,
                "frame_indexes": [0, frame_count - 1],
                "seed": manifest["seed"],
            }
        )
        episode_stats_records.append(
            {
                "episode_index": episode_index,
                "task_index": {"min": 0, "max": 0, "count": 1},
                "image_index": {
                    "min": 0,
                    "max": frame_count - 1,
                    "count": frame_count,
                },
                "point_goal_distance_m": {
                    "min": float(point_goal[:, 0].min()),
                    "max": float(point_goal[:, 0].max()),
                    "count": frame_count,
                },
            }
        )
        total_frames += frame_count

    info = {
        "codebase_version": "v2.1",
        "robot_type": "sage3d_pointgoal_fisheye",
        "scene_id": scene_id,
        "total_episodes": len(episodes_by_id),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(episodes_by_id),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:1"},
        "data_path": (
            "data/chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.camera_intrinsic": {
                "dtype": "float32",
                "shape": [3, 3],
            },
            "observation.camera_extrinsic": {
                "dtype": "float32",
                "shape": [4, 4],
            },
            "observation.camera_distortion": {
                "dtype": "float32",
                "shape": [4],
                "names": ["k1", "k2", "k3", "k4"],
            },
            "observation.point_goal": {
                "dtype": "float32",
                "shape": [2],
                "names": ["distance_m", "relative_bearing_rad"],
            },
            "action": {"dtype": "float32", "shape": [4, 4]},
        },
        "action_semantics": (
            "robot-base-to-world pose; +X forward, +Z up, translation z=0"
        ),
        "camera_extrinsic_semantics": (
            "camera-to-robot-base pose; identity rotation and +Z camera height"
        ),
        "camera_height_m": camera_height,
        "camera_model": "opencv_fisheye",
        "camera_fov_deg": summary_fov,
        "camera_horizontal_fov_deg": summary_fov,
        "camera_vertical_fov_deg": calibration.vertical_fov_deg,
        "camera_fisheye_coefficients": distortion,
        "camera_pitch_deg": 0.0,
        "camera_forward_mask_radius_pixels": calibration.forward_mask_radius_pixels,
        "image_width": summary_width,
        "image_height": summary_height,
        "depth_type": "distance_to_camera",
        "depth_format": f"uint16_meters_x_{int(render_summary['depth_scale'])}",
        "depth_clip_m": render_summary["max_depth_m"],
        "depth_min_m": render_summary["min_depth_m"],
        "trajectory_seed": manifest["seed"],
        "robot_radius_m": manifest["robot_radius_m"],
        "frame_spacing_m": manifest["frame_spacing_m"],
    }
    with (meta_dir / "info.json").open("w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)
    write_jsonl(meta_dir / "episodes.jsonl", episode_records)
    write_jsonl(
        meta_dir / "tasks.jsonl",
        [
            {
                "task_index": 0,
                "task": {
                    "type": "point_goal_navigation",
                    "goal_input": [
                        "distance_m",
                        "relative_bearing_rad",
                    ],
                },
            }
        ],
    )
    write_jsonl(meta_dir / "episodes_stats.jsonl", episode_stats_records)


# --- staged validation --------------------------------------------------------


def _parse_packaged_dataset(dataset_dir: Path) -> dict[str, Any]:
    """Read the packaged LeRobot-style dataset tree structure (inventory only)."""
    info_path = dataset_dir / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as file:
        info = json.load(file)
    data_dir = dataset_dir / "data" / "chunk-000"
    parquet_files = sorted(data_dir.glob("episode_*.parquet"))
    rgb_dir = dataset_dir / "videos" / "chunk-000" / "observation.images.rgb"
    depth_dir = dataset_dir / "videos" / "chunk-000" / "observation.images.depth"
    rgb_files = sorted(rgb_dir.glob("*.jpg"))
    depth_files = sorted(depth_dir.glob("*.png"))
    meta_files = {p.name for p in (dataset_dir / "meta").iterdir()}
    return {
        "info": info,
        "parquet_files": parquet_files,
        "rgb_files": rgb_files,
        "depth_files": depth_files,
        "meta_files": meta_files,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _check_config_assertions(
    info: dict[str, Any],
    config: PackageConfig | None,
    errors: list[str],
) -> None:
    """Apply the optional PackageConfig compatibility assertions.

    Mirrors the legacy CLI camera assertions: when a config field is set it is
    an expected value that the packaged dataset must match.
    """
    if config is None:
        return
    if config.width is not None and info.get("image_width") != config.width:
        errors.append(
            f"info image_width {info.get('image_width')} != config width {config.width}"
        )
    if config.height is not None and info.get("image_height") != config.height:
        errors.append(
            f"info image_height {info.get('image_height')} != config height {config.height}"
        )
    if config.horizontal_fov_deg is not None and not np.isclose(
        info.get("camera_horizontal_fov_deg"), config.horizontal_fov_deg
    ):
        errors.append(
            f"info camera_horizontal_fov_deg {info.get('camera_horizontal_fov_deg')} "
            f"!= config horizontal_fov_deg {config.horizontal_fov_deg}"
        )
    if config.fisheye_coefficients is not None and not np.allclose(
        info.get("camera_fisheye_coefficients"), config.fisheye_coefficients
    ):
        errors.append(
            f"info camera_fisheye_coefficients {info.get('camera_fisheye_coefficients')} "
            f"!= config fisheye_coefficients {list(config.fisheye_coefficients)}"
        )
    if config.camera_height is not None and not np.isclose(
        info.get("camera_height_m"), config.camera_height
    ):
        errors.append(
            f"info camera_height_m {info.get('camera_height_m')} "
            f"!= config camera_height {config.camera_height}"
        )


def validate_packaged_dataset(
    dataset_dir: Path,
    trajectory_dir: Path,
    rendered_dir: Path,
    config: PackageConfig | None = None,
) -> dict[str, Any]:
    """Validate a complete package staging tree before publication.

    Baseline-independent production validator (Phase 5b): checks inventory,
    Arrow/JSON content, copied-input checksums, calibration/extrinsics, and
    depth metadata against the current trajectory/render inputs, without
    invoking CLI or publication behavior. ``config`` supplies the optional
    legacy camera compatibility assertions.

    Returns ``{"eligible", "errors", "warnings", "scene_id", "episode_count"}``.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # --- Load packaged dataset structure ---
    try:
        pkg = _parse_packaged_dataset(dataset_dir)
    except Exception as e:
        return {
            "eligible": False,
            "errors": [f"packaged dataset: {e}"],
            "warnings": [],
            "scene_id": None,
            "episode_count": None,
        }

    info = pkg["info"]

    # --- Load trajectory manifest and render summaries ---
    try:
        manifest = _load_json(trajectory_dir / "trajectory_manifest.json")
    except Exception as e:
        return {
            "eligible": False,
            "errors": [f"trajectory manifest: {e}"],
            "warnings": [],
            "scene_id": None,
            "episode_count": None,
        }

    try:
        rgb_summary = _load_json(rendered_dir / "rgb_render_summary.json")
    except Exception as e:
        errors.append(f"rgb_render_summary: {e}")
        rgb_summary = None

    try:
        depth_summary = _load_json(rendered_dir / "depth_render_summary.json")
    except Exception as e:
        errors.append(f"depth_render_summary: {e}")
        depth_summary = None

    scene_id = manifest.get("scene_id")

    # --- Meta files inventory ---
    meta_files = pkg["meta_files"]
    for name in REQUIRED_META_FILES:
        if name not in meta_files:
            errors.append(f"missing meta file: {name}")

    # --- Scene ID consistency ---
    if info.get("scene_id") != scene_id:
        errors.append(f"info scene_id {info.get('scene_id')} != manifest {scene_id}")
    if depth_summary is not None and depth_summary.get("scene_id") != scene_id:
        errors.append(
            f"depth summary scene_id {depth_summary.get('scene_id')} != manifest {scene_id}"
        )
    if rgb_summary is not None and rgb_summary.get("scene_id") != scene_id:
        errors.append(
            f"rgb summary scene_id {rgb_summary.get('scene_id')} != manifest {scene_id}"
        )

    # --- Episode count consistency ---
    expected_episodes = manifest.get("episode_count")
    if info.get("total_episodes") != expected_episodes:
        errors.append(
            f"info total_episodes {info.get('total_episodes')} != manifest {expected_episodes}"
        )

    # --- Parquet file count ---
    parquet_files = pkg["parquet_files"]
    if len(parquet_files) != expected_episodes:
        errors.append(
            f"parquet count {len(parquet_files)} != manifest episodes {expected_episodes}"
        )

    # --- RGB/depth file counts ---
    expected_frames = sum(ep["frame_count"] for ep in manifest.get("episodes", []))
    if len(pkg["rgb_files"]) != expected_frames:
        errors.append(
            f"RGB file count {len(pkg['rgb_files'])} != manifest total_frames {expected_frames}"
        )
    if len(pkg["depth_files"]) != expected_frames:
        errors.append(
            f"depth file count {len(pkg['depth_files'])} != manifest total_frames {expected_frames}"
        )

    # --- info.json total_frames ---
    if info.get("total_frames") != expected_frames:
        errors.append(
            f"info total_frames {info.get('total_frames')} != manifest {expected_frames}"
        )

    # --- Parquet schema and values ---
    for ep in manifest.get("episodes", []):
        idx = ep["episode_index"]
        parquet_path = dataset_dir / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"
        if not parquet_path.is_file():
            errors.append(f"missing parquet: episode_{idx:06d}.parquet")
            continue
        try:
            table = pq.read_table(parquet_path)
            schema = table.schema

            col_names = set(schema.names)
            for col in PARQUET_REQUIRED_COLUMNS:
                if col not in col_names:
                    errors.append(f"episode_{idx:06d} parquet missing column: {col}")

            row_count = table.num_rows
            if row_count != ep["frame_count"]:
                errors.append(
                    f"episode_{idx:06d} parquet rows {row_count} != manifest frame_count {ep['frame_count']}"
                )

            for col_name in PARQUET_REQUIRED_COLUMNS[1:]:
                if col_name in col_names:
                    field = schema.field(col_name)
                    type_str = str(field.type)
                    if "float" not in type_str:
                        errors.append(
                            f"episode_{idx:06d} column {col_name} type {field.type} != list<float32>"
                        )

            if "index" in col_names:
                field = schema.field("index")
                if str(field.type) != "int64":
                    errors.append(f"episode_{idx:06d} index type {field.type} != int64")

        except Exception as e:
            errors.append(f"episode_{idx:06d} parquet read failed: {e}")

    # --- Calibration/extrinsics: float32 camera_intrinsic/extrinsic ---
    if parquet_files:
        try:
            first_table = pq.read_table(parquet_files[0])
            if "observation.camera_intrinsic" in first_table.column_names:
                intrinsic = np.array(
                    first_table["observation.camera_intrinsic"][0].as_py(),
                    dtype=np.float32,
                )
                if intrinsic.shape != (9,):
                    errors.append(f"camera_intrinsic flat shape {intrinsic.shape} != (9,)")
            if "observation.camera_extrinsic" in first_table.column_names:
                extrinsic = np.array(
                    first_table["observation.camera_extrinsic"][0].as_py(),
                    dtype=np.float32,
                )
                if extrinsic.shape != (16,):
                    errors.append(f"camera_extrinsic flat shape {extrinsic.shape} != (16,)")
                ext_mat = extrinsic.reshape(4, 4)
                if abs(float(ext_mat[2, 3]) - float(info.get("camera_height_m", 0.0))) > 1e-6:
                    errors.append(
                        f"extrinsic z={float(ext_mat[2, 3])} != info camera_height_m {info.get('camera_height_m')}"
                    )
        except Exception as e:
            errors.append(f"calibration check failed: {e}")

    # --- Depth metadata authority ---
    if depth_summary is not None:
        if info.get("depth_clip_m") != depth_summary.get("max_depth_m"):
            errors.append(
                f"info depth_clip_m {info.get('depth_clip_m')} != depth summary max_depth_m {depth_summary.get('max_depth_m')}"
            )
        if info.get("depth_min_m") != depth_summary.get("min_depth_m"):
            errors.append(
                f"info depth_min_m {info.get('depth_min_m')} != depth summary min_depth_m {depth_summary.get('min_depth_m')}"
            )

    # --- Copied files: PLY and manifest checksums match source ---
    pkg_ply = dataset_dir / "meta" / "pointcloud.ply"
    src_ply = trajectory_dir / "pointcloud.ply"
    if pkg_ply.is_file() and src_ply.is_file():
        if _sha256_file(pkg_ply) != _sha256_file(src_ply):
            errors.append("meta/pointcloud.ply checksum != trajectory pointcloud.ply")

    pkg_manifest = dataset_dir / "meta" / "trajectory_manifest.json"
    src_manifest = trajectory_dir / "trajectory_manifest.json"
    if pkg_manifest.is_file() and src_manifest.is_file():
        if _sha256_file(pkg_manifest) != _sha256_file(src_manifest):
            errors.append("meta/trajectory_manifest.json checksum != trajectory manifest")

    # --- Copied render summaries match source ---
    for summary_name in (
        "render_summary.json",
        "rgb_render_summary.json",
        "depth_render_summary.json",
    ):
        pkg_summary = dataset_dir / "meta" / summary_name
        src_summary = rendered_dir / summary_name
        if pkg_summary.is_file() and src_summary.is_file():
            if _sha256_file(pkg_summary) != _sha256_file(src_summary):
                errors.append(f"meta/{summary_name} checksum != rendered/{summary_name}")

    # --- Copied RGB/depth files match source ---
    for rgb_file in pkg["rgb_files"][:3]:
        src = rendered_dir / "observation.images.rgb" / rgb_file.name
        if src.is_file():
            if _sha256_file(rgb_file) != _sha256_file(src):
                errors.append(f"RGB file {rgb_file.name} checksum != rendered source")
                break

    for depth_file in pkg["depth_files"][:3]:
        src = rendered_dir / "observation.images.depth" / depth_file.name
        if src.is_file():
            if _sha256_file(depth_file) != _sha256_file(src):
                errors.append(f"depth file {depth_file.name} checksum != rendered source")
                break

    # --- Episodes JSONL order matches manifest ---
    try:
        episodes_jsonl = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
        if len(episodes_jsonl) != expected_episodes:
            errors.append(
                f"episodes.jsonl count {len(episodes_jsonl)} != manifest {expected_episodes}"
            )
        for i, (ep_rec, ep_manifest) in enumerate(
            zip(episodes_jsonl, manifest.get("episodes", []))
        ):
            if ep_rec.get("episode_index") != ep_manifest["episode_index"]:
                errors.append(f"episodes.jsonl[{i}] episode_index mismatch")
                break
            if ep_rec.get("frame_count") != ep_manifest["frame_count"]:
                errors.append(f"episodes.jsonl[{i}] frame_count mismatch")
                break
    except Exception as e:
        errors.append(f"episodes.jsonl read failed: {e}")

    # --- Optional PackageConfig compatibility assertions ---
    _check_config_assertions(info, config, errors)

    return {
        "eligible": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "scene_id": scene_id,
        "episode_count": expected_episodes,
    }


# --- publication orchestration -------------------------------------------------


def _load_episodes(trajectory_dir: Path) -> dict[int, EpisodeArrays]:
    episodes: dict[int, EpisodeArrays] = {}
    for npz_file in sorted(trajectory_dir.glob("episode_*.npz")):
        episodes[parse_episode_filename(npz_file.name)] = load_episode(npz_file)
    return episodes


def package(config: PackageConfig) -> Path:
    """Build, validate, and atomically publish a packaged dataset.

    Orchestration per the revision-8 plan (Phase 5c):

    1. Validate the source pipeline contract (pre-package invariants).
    2. Build the complete dataset into an internally allocated sibling staging
       directory next to ``config.output_dir``.
    3. Validate the staged tree with :func:`validate_packaged_dataset`.
    4. Atomically rename the staging directory onto the absent final target
       only after validation succeeds.

    Non-destructive: the final output is never pre-created or deleted in
    library code; a rerun requires caller/shell cleanup of an existing target.
    Cross-filesystem copy fallback is prohibited by
    :func:`sage3d.publication.atomic_publish_directory` (rename on one device
    only). Any failure before the rename leaves ``config.output_dir`` absent
    and the staging directory intact for diagnosis.
    """
    manifest = _load_json(config.trajectory_dir / "trajectory_manifest.json")
    rgb_summary = _load_json(config.rendered_dir / "rgb_render_summary.json")
    canonical_depth_summary = _load_json(config.rendered_dir / "render_summary.json")
    depth_alias_summary = _load_json(
        config.rendered_dir / "depth_render_summary.json"
    )
    episodes_by_id = _load_episodes(config.trajectory_dir)

    # 1. Source pipeline contract (raises ContractError subclass on violation).
    validate_pipeline_contract(
        expected_scene_id=config.scene_id,
        manifest=manifest,
        rgb_summary=rgb_summary,
        canonical_depth_summary=canonical_depth_summary,
        depth_alias_summary=depth_alias_summary,
        episodes_by_id=episodes_by_id,
        trajectory_dir=config.trajectory_dir,
        rendered_dir=config.rendered_dir,
        pointcloud_path=config.trajectory_dir / "pointcloud.ply",
    )

    # 2. Build into an internally allocated sibling staging directory.
    staging = create_staging_directory(config.output_dir, prefix=".pkg.")
    try:
        summary_width, summary_height = canonical_depth_summary["resolution"]
        summary_fov = canonical_depth_summary["horizontal_fov_deg"]
        summary_coeffs = tuple(canonical_depth_summary["fisheye_coefficients"])
        calibration = CameraCalibration(
            summary_width, summary_height, summary_fov, summary_coeffs
        )
        camera_height = manifest["camera_height_m"]
        intrinsic_flat = calibration.intrinsic_matrix().reshape(-1).tolist()
        extrinsic_flat = calibration.extrinsic_matrix(camera_height).reshape(-1).tolist()
        distortion = calibration.fisheye_coefficients

        data_dir = staging / "data" / "chunk-000"
        video_output_dir = staging / "videos" / "chunk-000"
        meta_dir = staging / "meta"

        for episode_index in sorted(episodes_by_id):
            episode = episodes_by_id[episode_index]
            frame_count = len(episode.actions)
            build_episode_parquet(
                data_dir,
                episode_index,
                frame_count=frame_count,
                intrinsic_flat=intrinsic_flat,
                extrinsic_flat=extrinsic_flat,
                distortion=distortion,
                point_goal=episode.point_goal,
                actions=episode.actions,
            )
            copy_episode_frames(
                config.rendered_dir, video_output_dir, episode_index, frame_count
            )
        write_lerobot_meta(
            meta_dir,
            scene_id=config.scene_id,
            fps=config.fps,
            manifest=manifest,
            render_summary=canonical_depth_summary,
            trajectory_dir=config.trajectory_dir,
            rendered_dir=config.rendered_dir,
            calibration=calibration,
            episodes_by_id=episodes_by_id,
        )

        # 3. Staged validation before publication.
        validation = validate_packaged_dataset(
            staging, config.trajectory_dir, config.rendered_dir, config
        )
        if not validation["eligible"]:
            raise RuntimeError(
                "staged package validation failed: " + "; ".join(validation["errors"])
            )

        # 4. Atomic publish onto the absent final target.
        return atomic_publish_directory(staging, config.output_dir)
    except Exception:
        # Leave the incomplete staging directory intact for diagnosis; never
        # clean or reuse ambiguous partial staging state.
        raise

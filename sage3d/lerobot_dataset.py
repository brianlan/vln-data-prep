"""Pure package-safe LeRobot-style dataset builders (numpy + pyarrow + stdlib).

Phase 5a extracts the deterministic dataset construction from
``package_lerobot_sage3d.py`` into package-safe builders so the package
boundary can be validated independently:

- :func:`build_episode_parquet` writes one episode's Arrow parquet with the
  exact legacy schema and column order.
- :func:`copy_episode_frames` copies one episode's RGB/depth frames from the
  finalized render root.
- :func:`write_lerobot_meta` writes the meta directory: copied pointcloud /
  manifest / render summaries, ``info.json``, and the episodes / tasks /
  episodes_stats JSONL records.

The builders consume only finalized render artifacts and the authoritative
depth/calibration summaries. They are non-destructive: they never overwrite an
existing publication target (that is the Phase 5c publication flow's job) and
never validate a staged tree (that is the Phase 5b validator's job). The
project-specific LeRobot-style layout is preserved exactly; standard LeRobot
compatibility is not claimed.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sage3d.camera import CameraCalibration
from sage3d.episode_arrays import EpisodeArrays
from sage3d.naming import frame_stem


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
    rgb_summary: dict,
    depth_summary: dict,
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

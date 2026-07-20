#!/usr/bin/env python3
"""Package generated SAGE3D PointGoal trajectories as LeRobot v2.1."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--rendered-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fov-deg", type=float, default=195.0)
    parser.add_argument("--camera-height", type=float, default=0.6)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def fisheye_intrinsic(width: int, height: int, fov_deg: float) -> np.ndarray:
    focal = width / math.radians(fov_deg)
    return np.asarray(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def camera_extrinsic(camera_height: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[2, 3] = camera_height
    return transform


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_image_pair(
    rgb_path: Path, depth_path: Path, width: int, height: int
) -> None:
    if not rgb_path.is_file():
        raise FileNotFoundError(rgb_path)
    if not depth_path.is_file():
        raise FileNotFoundError(depth_path)
    with Image.open(rgb_path) as rgb:
        if rgb.size != (width, height) or rgb.mode != "RGB":
            raise RuntimeError(
                f"Invalid RGB image {rgb_path}: size={rgb.size}, mode={rgb.mode}"
            )
    with Image.open(depth_path) as depth:
        array = np.asarray(depth)
        if depth.size != (width, height) or array.dtype != np.uint16:
            raise RuntimeError(
                f"Invalid depth image {depth_path}: size={depth.size}, "
                f"dtype={array.dtype}"
            )


def main() -> None:
    args = parse_args()
    manifest_path = args.trajectory_dir / "trajectory_manifest.json"
    pointcloud_path = args.trajectory_dir / "pointcloud.ply"
    render_summary_path = args.rendered_dir / "render_summary.json"
    rgb_summary_path = args.rendered_dir / "rgb_render_summary.json"
    depth_summary_path = args.rendered_dir / "depth_render_summary.json"
    for path in (
        manifest_path,
        pointcloud_path,
        render_summary_path,
        rgb_summary_path,
        depth_summary_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    with render_summary_path.open("r", encoding="utf-8") as file:
        render_summary = json.load(file)
    with rgb_summary_path.open("r", encoding="utf-8") as file:
        rgb_summary = json.load(file)

    trajectory_files = sorted(args.trajectory_dir.glob("episode_*.npz"))
    if len(trajectory_files) != manifest["episode_count"]:
        raise RuntimeError(
            f"Trajectory file count {len(trajectory_files)} does not match "
            f"manifest {manifest['episode_count']}"
        )
    if render_summary["total_frames"] != sum(
        episode["frame_count"] for episode in manifest["episodes"]
    ):
        raise RuntimeError("Rendered frame count does not match trajectory manifest")
    if rgb_summary["total_frames"] != render_summary["total_frames"]:
        raise RuntimeError("RGB/depth rendered frame counts do not match")

    data_dir = args.output_dir / "data" / "chunk-000"
    meta_dir = args.output_dir / "meta"
    rgb_output_dir = (
        args.output_dir
        / "videos"
        / "chunk-000"
        / "observation.images.rgb"
    )
    depth_output_dir = (
        args.output_dir
        / "videos"
        / "chunk-000"
        / "observation.images.depth"
    )
    for directory in (data_dir, meta_dir, rgb_output_dir, depth_output_dir):
        directory.mkdir(parents=True, exist_ok=True)

    intrinsic = fisheye_intrinsic(args.width, args.height, args.fov_deg)
    extrinsic = camera_extrinsic(args.camera_height)
    intrinsic_flat = intrinsic.reshape(-1).tolist()
    extrinsic_flat = extrinsic.reshape(-1).tolist()

    episode_records = []
    episode_stats_records = []
    total_frames = 0
    for episode_index, trajectory_file in enumerate(trajectory_files):
        trajectory = np.load(trajectory_file)
        actions = trajectory["actions"].astype(np.float32)
        point_goal = trajectory["point_goal"].astype(np.float32)
        if len(actions) != len(point_goal):
            raise RuntimeError(
                f"Action/PointGoal count mismatch in {trajectory_file}"
            )
        frame_count = len(actions)

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
        pq.write_table(
            table, data_dir / f"episode_{episode_index:06d}.parquet"
        )

        for frame_index in range(frame_count):
            stem = f"episode_{episode_index:06d}_{frame_index:03d}"
            rgb_source = (
                args.rendered_dir
                / "observation.images.rgb"
                / f"{stem}.jpg"
            )
            depth_source = (
                args.rendered_dir
                / "observation.images.depth"
                / f"{stem}.png"
            )
            validate_image_pair(
                rgb_source, depth_source, args.width, args.height
            )
            shutil.copy2(rgb_source, rgb_output_dir / rgb_source.name)
            shutil.copy2(depth_source, depth_output_dir / depth_source.name)

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
                "minimum_clearance_m": manifest_episode[
                    "minimum_clearance_m"
                ],
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

    shutil.copy2(pointcloud_path, meta_dir / "pointcloud.ply")
    shutil.copy2(manifest_path, meta_dir / "trajectory_manifest.json")
    shutil.copy2(render_summary_path, meta_dir / "render_summary.json")
    shutil.copy2(rgb_summary_path, meta_dir / rgb_summary_path.name)
    shutil.copy2(depth_summary_path, meta_dir / depth_summary_path.name)

    info = {
        "codebase_version": "v2.1",
        "robot_type": "sage3d_pointgoal_fisheye",
        "scene_id": args.scene,
        "total_episodes": len(trajectory_files),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(trajectory_files),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": args.fps,
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
        "camera_height_m": args.camera_height,
        "camera_model": "fisheye_equidistant",
        "camera_fov_deg": args.fov_deg,
        "image_width": args.width,
        "image_height": args.height,
        "depth_type": "distance_to_camera",
        "depth_format": "uint16_meters_x_10000",
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

    print(
        f"[package] Wrote {len(trajectory_files)} episodes and "
        f"{total_frames} frames to {args.output_dir}"
    )


if __name__ == "__main__":
    main()

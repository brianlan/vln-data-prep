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

from sage3d.camera import CameraCalibration
from sage3d.contract import validate_pipeline_contract
from sage3d.episode_arrays import load_episode
from sage3d.naming import frame_stem, parse_episode_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--rendered-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--horizontal-fov-deg", type=float, default=None)
    parser.add_argument(
        "--fisheye-coefficients",
        type=float,
        nargs=4,
        metavar=("K1", "K2", "K3", "K4"),
        default=None,
    )
    parser.add_argument("--camera-height", type=float, default=None)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    with depth_summary_path.open("r", encoding="utf-8") as file:
        depth_summary = json.load(file)

    trajectory_files = sorted(args.trajectory_dir.glob("episode_*.npz"))
    episodes_by_id: dict[int, object] = {}
    for tf in trajectory_files:
        episodes_by_id[parse_episode_filename(tf.name)] = load_episode(tf)

    # Cross-artifact contract: validate all pre-package invariants before any
    # output write. Raises a ContractError subclass on the first violation.
    validate_pipeline_contract(
        expected_scene_id=args.scene,
        manifest=manifest,
        rgb_summary=rgb_summary,
        canonical_depth_summary=render_summary,
        depth_alias_summary=depth_summary,
        episodes_by_id=episodes_by_id,
        trajectory_dir=args.trajectory_dir,
        rendered_dir=args.rendered_dir,
        pointcloud_path=pointcloud_path,
    )

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

    # Canonical depth summary (render_summary.json) is the calibration authority.
    summary_width, summary_height = render_summary["resolution"]
    summary_fov = render_summary["horizontal_fov_deg"]
    summary_coeffs = tuple(render_summary["fisheye_coefficients"])
    calibration = CameraCalibration(
        summary_width, summary_height, summary_fov, summary_coeffs
    )

    # Legacy CLI camera values are optional expected-value assertions.
    if args.width is not None and args.width != summary_width:
        raise RuntimeError(
            f"--width {args.width} does not match depth summary {summary_width}"
        )
    if args.height is not None and args.height != summary_height:
        raise RuntimeError(
            f"--height {args.height} does not match depth summary {summary_height}"
        )
    if args.horizontal_fov_deg is not None and not math.isclose(
        args.horizontal_fov_deg, summary_fov, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise RuntimeError(
            f"--horizontal-fov-deg {args.horizontal_fov_deg} does not match "
            f"depth summary {summary_fov}"
        )
    if args.fisheye_coefficients is not None and not np.allclose(
        args.fisheye_coefficients, summary_coeffs, rtol=1e-5, atol=1e-8
    ):
        raise RuntimeError(
            f"--fisheye-coefficients {args.fisheye_coefficients} do not match "
            f"depth summary {summary_coeffs}"
        )

    # Manifest camera_height_m is authoritative; CLI --camera-height is optional.
    camera_height = manifest["camera_height_m"]
    if args.camera_height is not None and not math.isclose(
        args.camera_height, camera_height, rel_tol=1e-6, abs_tol=1e-6
    ):
        raise RuntimeError(
            f"--camera-height {args.camera_height} does not match "
            f"manifest camera_height_m {camera_height}"
        )

    intrinsic = calibration.intrinsic_matrix()
    extrinsic = calibration.extrinsic_matrix(camera_height)
    intrinsic_flat = intrinsic.reshape(-1).tolist()
    extrinsic_flat = extrinsic.reshape(-1).tolist()
    distortion = calibration.fisheye_coefficients

    episode_records = []
    episode_stats_records = []
    total_frames = 0
    for episode_index in sorted(episodes_by_id):
        episode = episodes_by_id[episode_index]
        actions = episode.actions
        point_goal = episode.point_goal
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
        pq.write_table(
            table, data_dir / f"episode_{episode_index:06d}.parquet"
        )

        for frame_index in range(frame_count):
            stem = frame_stem(episode_index, frame_index)
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

    print(
        f"[package] Wrote {len(trajectory_files)} episodes and "
        f"{total_frames} frames to {args.output_dir}"
    )


if __name__ == "__main__":
    main()

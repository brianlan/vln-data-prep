#!/usr/bin/env python3
"""Generate deterministic, collision-aware PointGoal trajectories for SAGE3D."""

from __future__ import annotations

import argparse
from pathlib import Path

from sage3d.cli._args import add_scene_args
from sage3d.config import (
    GenerationConfig,
    PathConfig,
    SceneConfig,
    SafetyConfig,
)
from sage3d.publication import (
    atomic_publish_directory,
    assert_target_absent,
    create_staging_directory,
)
from sage3d.trajectory_pipeline import generate, write_trajectory_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_scene_args(parser)
    parser.add_argument(
        "--interiorgs-root",
        type=Path,
        default=None,
        help="Override InteriorGS root; defaults to <sage-root>/InteriorGS",
    )
    parser.add_argument(
        "--collision-usd",
        type=Path,
        default=None,
        help="Override collision USD; defaults to <sage-root>/Collision_Mesh/...",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--robot-radius", type=float, default=0.25)
    parser.add_argument("--safety-margin", type=float, default=0.05)
    parser.add_argument("--camera-height", type=float, default=0.6)
    parser.add_argument(
        "--camera-clearance",
        type=float,
        default=None,
        help="Minimum 3D collision-mesh distance at the camera center; defaults to robot radius",
    )
    parser.add_argument("--min-path-length", type=float, default=3.0)
    parser.add_argument("--max-path-length", type=float, default=15.0)
    parser.add_argument("--frame-spacing", type=float, default=0.05)
    parser.add_argument("--endpoint-extra-clearance", type=float, default=0.10)
    parser.add_argument("--max-attempts", type=int, default=3000)
    parser.add_argument("--pointcloud-voxel-size", type=float, default=0.05)
    parser.add_argument("--pointcloud-max-points", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    camera_clearance = (
        args.robot_radius
        if args.camera_clearance is None
        else args.camera_clearance
    )

    config = GenerationConfig(
        episodes=args.episodes,
        seed=args.seed,
        pointcloud_voxel_size=args.pointcloud_voxel_size,
        pointcloud_max_points=args.pointcloud_max_points,
        scene=SceneConfig(
            scene_id=args.scene,
            sage_root=args.sage_root,
            interiorgs_root=args.interiorgs_root,
            collision_usd=args.collision_usd,
        ),
        safety=SafetyConfig(
            robot_radius=args.robot_radius,
            safety_margin=args.safety_margin,
            camera_height=args.camera_height,
            camera_clearance=camera_clearance,
            endpoint_extra_clearance=args.endpoint_extra_clearance,
        ),
        path=PathConfig(
            min_path_length=args.min_path_length,
            max_path_length=args.max_path_length,
            frame_spacing=args.frame_spacing,
            max_attempts=args.max_attempts,
        ),
    )

    result = generate(config)

    assert_target_absent(args.output_dir)
    staging = create_staging_directory(args.output_dir, prefix=".trajectory-stage.")
    write_trajectory_artifacts(staging, result)
    atomic_publish_directory(staging, args.output_dir)

    print(
        f"Generated {len(result.episodes)} episodes for {config.scene.scene_id}: "
        f"{sum(ep['frame_count'] for ep in result.episodes)} frames"
    )
    for episode in result.episodes:
        print(
            f"  episode {episode['episode_index']:06d}: "
            f"{episode['path_length_m']:.2f} m, "
            f"{episode['frame_count']} frames, "
            f"clearance >= {episode['minimum_clearance_m']:.2f} m, "
            f"camera clearance >= "
            f"{episode['minimum_camera_clearance_m']:.2f} m, "
            f"{episode['smoothing_method']}"
        )
    print(f"Artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
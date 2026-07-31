#!/usr/bin/env python3
"""Generate deterministic, collision-aware PointGoal trajectories for SAGE3D."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

from sage3d.artifacts import resolve_generation_assets
from sage3d.collision import (
    apply_camera_clearance,
    extract_collision_geometry,
)
from sage3d.cli._args import add_scene_args
from sage3d.config import (
    GenerationConfig,
    PathConfig,
    SceneConfig,
    SafetyConfig,
)
from sage3d.episode_arrays import EpisodeArrays, save_episode
from sage3d.episode_generation import generate_episodes
from sage3d.geometry import path_length
from sage3d.io_ply import write_binary_pointcloud
from sage3d.naming import episode_filename
from sage3d.navigation_map import MapInfo, connected_components, load_navigation_map
from sage3d.pointcloud import voxel_downsample
from sage3d.schemas import build_trajectory_manifest, manifest_to_json
from sage3d.viz import save_navigation_visualizations


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
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    assets = resolve_generation_assets(
        config.scene.scene_id,
        config.scene.sage_root,
        interiorgs_root=config.scene.interiorgs_root,
        collision_usd=config.scene.collision_usd,
    )
    scene_dir = assets.scene_dir
    collision_usd = assets.collision_usd

    collision_points, collision_faces = extract_collision_geometry(collision_usd)
    collision_mesh = trimesh.Trimesh(
        vertices=collision_points,
        faces=collision_faces,
        process=False,
    )
    safe, clearance_m, transform, raw_map_info = load_navigation_map(
        scene_dir, config.safety.robot_radius, config.safety.safety_margin
    )
    safe, camera_clearance_info = apply_camera_clearance(
        safe=safe,
        mesh=collision_mesh,
        transform=transform,
        camera_height=config.safety.camera_height,
        camera_clearance=config.safety.camera_clearance,
    )
    component_labels, components = connected_components(safe, transform.scale)
    map_info = MapInfo(
        shape=raw_map_info["shape"],
        scale_m_per_pixel=raw_map_info["scale_m_per_pixel"],
        robot_radius_m=raw_map_info["robot_radius_m"],
        safety_margin_m=raw_map_info["safety_margin_m"],
        required_path_clearance_m=raw_map_info["required_path_clearance_m"],
        room_count=raw_map_info["room_count"],
        raw_free_area_m2=raw_map_info["raw_free_area_m2"],
        safe_free_area_m2=float(safe.sum() * transform.scale**2),
        occupancy_values=raw_map_info["occupancy_values"],
        components=components,
        camera_collision_filter=camera_clearance_info,
    )
    endpoint_clearance = (
        config.safety.robot_radius
        + config.safety.safety_margin
        + config.safety.endpoint_extra_clearance
    )
    episodes, generation_info = generate_episodes(
        safe=safe,
        clearance_m=clearance_m,
        component_labels=component_labels,
        transform=transform,
        episode_count=config.episodes,
        seed=config.seed,
        min_path_length=config.path.min_path_length,
        max_path_length=config.path.max_path_length,
        frame_spacing=config.path.frame_spacing,
        endpoint_clearance=endpoint_clearance,
        max_attempts=config.path.max_attempts,
        camera_height=config.safety.camera_height,
        collision_mesh=collision_mesh,
        camera_clearance=config.safety.camera_clearance,
    )

    for episode in episodes:
        episode_path = args.output_dir / episode_filename(episode["episode_index"])
        save_episode(
            episode_path,
            EpisodeArrays(
                points=episode["points"],
                actions=episode["actions"],
                camera_positions=episode["camera_positions"],
                yaw=episode["yaw"],
                point_goal=episode["point_goal"],
                start_position=np.asarray(
                    episode["start_position"], dtype=np.float32
                ),
                goal_position=np.asarray(
                    episode["goal_position"], dtype=np.float32
                ),
            ),
        )

    pointcloud = voxel_downsample(
        collision_points,
        config.pointcloud_voxel_size,
        config.pointcloud_max_points,
        config.seed,
    )
    write_binary_pointcloud(args.output_dir / "pointcloud.ply", pointcloud)

    save_navigation_visualizations(
        args.output_dir, safe, clearance_m, transform, episodes
    )

    manifest = build_trajectory_manifest(
        scene_id=config.scene.scene_id,
        scene_dir=str(scene_dir),
        collision_usd=str(collision_usd),
        seed=config.seed,
        episodes=episodes,
        robot_radius_m=config.safety.robot_radius,
        safety_margin_m=config.safety.safety_margin,
        camera_height_m=config.safety.camera_height,
        camera_clearance_m=config.safety.camera_clearance,
        frame_spacing_m=config.path.frame_spacing,
        requested_path_length_range_m=[
            config.path.min_path_length,
            config.path.max_path_length,
        ],
        endpoint_clearance_m=endpoint_clearance,
        map_info=map_info.to_dict(),
        generation_info=generation_info,
        pointcloud={
            "source_vertex_count": len(collision_points),
            "output_point_count": len(pointcloud),
            "voxel_size_m": config.pointcloud_voxel_size,
            "bounds_min": pointcloud.min(axis=0).astype(float).tolist(),
            "bounds_max": pointcloud.max(axis=0).astype(float).tolist(),
            "color": [160, 160, 160],
        },
    )
    manifest_to_json(manifest, args.output_dir / "trajectory_manifest.json")

    print(
        f"Generated {len(episodes)} episodes for {config.scene.scene_id}: "
        f"{sum(ep['frame_count'] for ep in episodes)} frames"
    )
    for episode in episodes:
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
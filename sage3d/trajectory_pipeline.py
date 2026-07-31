"""Trajectory pipeline: orchestration separated from atomic writes.

``generate(config)`` runs all Isaac-side computation (collision extraction,
navigation map, episode generation, pointcloud downsample, manifest build)
and returns a :class:`TrajectoryResult` with no side effects on the final
output directory.

``write_trajectory_artifacts(staging, result)`` writes every artifact into an
internally allocated sibling staging directory.  The caller then publishes
atomically via :func:`sage3d.publication.atomic_publish_directory`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from sage3d.artifacts import resolve_generation_assets
from sage3d.collision import (
    apply_camera_clearance,
    extract_collision_geometry,
)
from sage3d.config import GenerationConfig
from sage3d.episode_arrays import EpisodeArrays, save_episode
from sage3d.episode_generation import generate_episodes
from sage3d.io_ply import write_binary_pointcloud
from sage3d.naming import episode_filename
from sage3d.navigation_map import MapInfo, connected_components, load_navigation_map
from sage3d.pointcloud import voxel_downsample
from sage3d.schemas import build_trajectory_manifest, manifest_to_json
from sage3d.viz import save_navigation_visualizations


@dataclass(frozen=True)
class TrajectoryResult:
    """Everything ``generate`` produces — no writes yet."""

    episodes: list[dict]
    generation_info: dict[str, Any]
    pointcloud: np.ndarray
    manifest: dict[str, Any]
    map_info: MapInfo
    scene_dir: Path
    collision_usd: Path
    safe: np.ndarray
    clearance_m: np.ndarray
    transform: Any


def generate(config: GenerationConfig) -> TrajectoryResult:
    """Run all generation orchestration and return results without writing."""
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

    pointcloud = voxel_downsample(
        collision_points,
        config.pointcloud_voxel_size,
        config.pointcloud_max_points,
        config.seed,
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

    return TrajectoryResult(
        episodes=episodes,
        generation_info=generation_info,
        pointcloud=pointcloud,
        manifest=manifest,
        map_info=map_info,
        scene_dir=scene_dir,
        collision_usd=collision_usd,
        safe=safe,
        clearance_m=clearance_m,
        transform=transform,
    )


def write_trajectory_artifacts(staging: Path, result: TrajectoryResult) -> None:
    """Write all artifacts (episodes, pointcloud, viz, manifest) into *staging*."""
    for episode in result.episodes:
        episode_path = staging / episode_filename(episode["episode_index"])
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

    write_binary_pointcloud(staging / "pointcloud.ply", result.pointcloud)

    save_navigation_visualizations(
        staging,
        result.safe,
        result.clearance_m,
        result.transform,
        result.episodes,
    )

    manifest_to_json(result.manifest, staging / "trajectory_manifest.json")
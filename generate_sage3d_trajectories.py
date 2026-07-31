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
    collision_distances,
    extract_collision_geometry,
)
from sage3d.cli._args import add_scene_args
from sage3d.episode_arrays import EpisodeArrays, save_episode
from sage3d.geometry import MapTransform, path_length, pixels_to_world, wrap_angle
from sage3d.io_ply import write_binary_pointcloud
from sage3d.naming import episode_filename
from sage3d.navigation_map import connected_components, load_navigation_map
from sage3d.path_postprocess import (
    points_are_safe,
    resample_path,
    smooth_path,
)
from sage3d.pathfinding import astar
from sage3d.pointcloud import voxel_downsample
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


def build_episode_arrays(
    points: np.ndarray, camera_height: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    delta = np.gradient(points, axis=0)
    yaw = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    goal = points[-1]

    actions = np.repeat(np.eye(4, dtype=np.float32)[None, ...], len(points), axis=0)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    actions[:, 0, 0] = cos_yaw
    actions[:, 0, 1] = -sin_yaw
    actions[:, 1, 0] = sin_yaw
    actions[:, 1, 1] = cos_yaw
    actions[:, 0, 3] = points[:, 0]
    actions[:, 1, 3] = points[:, 1]

    camera_positions = np.column_stack(
        (points[:, 0], points[:, 1], np.full(len(points), camera_height))
    ).astype(np.float32)
    goal_delta = goal[None, :] - points
    goal_distance = np.linalg.norm(goal_delta, axis=1)
    goal_bearing = wrap_angle(
        np.arctan2(goal_delta[:, 1], goal_delta[:, 0]) - yaw
    )
    goal_bearing[goal_distance < 1e-6] = 0.0
    point_goal = np.column_stack((goal_distance, goal_bearing)).astype(np.float32)
    return actions, camera_positions, yaw.astype(np.float32), point_goal


def generate_episodes(
    safe: np.ndarray,
    clearance_m: np.ndarray,
    component_labels: np.ndarray,
    transform: MapTransform,
    episode_count: int,
    seed: int,
    min_path_length: float,
    max_path_length: float,
    frame_spacing: float,
    endpoint_clearance: float,
    max_attempts: int,
    camera_height: float,
    collision_mesh: trimesh.Trimesh,
    camera_clearance: float,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    candidate_mask = safe & (clearance_m >= endpoint_clearance)
    component_ids, component_sizes = np.unique(
        component_labels[candidate_mask], return_counts=True
    )
    usable = [
        (int(label), int(size))
        for label, size in zip(component_ids, component_sizes)
        if label != 0 and size >= 2
    ]
    if not usable:
        raise RuntimeError("No connected component has valid endpoint candidates")

    component_weights = np.asarray([size for _, size in usable], dtype=np.float64)
    component_weights /= component_weights.sum()
    component_cells = {
        label: np.argwhere(candidate_mask & (component_labels == label))
        for label, _ in usable
    }

    episodes = []
    rejection_counts: dict[str, int] = {}
    used_endpoints: list[np.ndarray] = []

    def reject(reason: str) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    for attempt in range(1, max_attempts + 1):
        if len(episodes) >= episode_count:
            break
        component_index = int(rng.choice(len(usable), p=component_weights))
        component_id = usable[component_index][0]
        cells = component_cells[component_id]
        selected = rng.choice(len(cells), size=2, replace=False)
        start_pixel = tuple(int(value) for value in cells[selected[0]])
        goal_pixel = tuple(int(value) for value in cells[selected[1]])
        start_xy = np.asarray(transform.pixel_to_world(*start_pixel))
        goal_xy = np.asarray(transform.pixel_to_world(*goal_pixel))

        if float(np.linalg.norm(goal_xy - start_xy)) < min_path_length * 0.55:
            reject("euclidean_too_short")
            continue
        if used_endpoints and min(
            float(np.linalg.norm(start_xy - endpoint))
            + float(np.linalg.norm(goal_xy - other))
            for endpoint, other in zip(
                used_endpoints[0::2], used_endpoints[1::2]
            )
        ) < 1.0:
            reject("duplicate_endpoint_pair")
            continue

        pixel_path = astar(
            safe,
            clearance_m,
            start_pixel,
            goal_pixel,
            transform.scale,
        )
        if pixel_path is None:
            reject("astar_failed")
            continue
        raw_world = pixels_to_world(pixel_path, transform)
        raw_length = path_length(raw_world)
        if raw_length < min_path_length:
            reject("geodesic_too_short")
            continue
        if raw_length > max_path_length:
            reject("geodesic_too_long")
            continue

        smoothed, smoothing_method = smooth_path(raw_world, safe, transform)
        sampled = resample_path(smoothed, frame_spacing)
        if not points_are_safe(sampled, safe, transform):
            reject("resampled_path_not_safe")
            continue

        actions, camera_positions, yaw, point_goal = build_episode_arrays(
            sampled, camera_height
        )
        camera_distances = collision_distances(
            collision_mesh, camera_positions.astype(np.float64)
        )
        minimum_camera_clearance = float(camera_distances.min())
        if minimum_camera_clearance < camera_clearance:
            reject("camera_collision_clearance")
            continue
        episode_index = len(episodes)
        episodes.append(
            {
                "episode_index": episode_index,
                "component_id": component_id,
                "start_pixel": list(start_pixel),
                "goal_pixel": list(goal_pixel),
                "start_position": [float(sampled[0, 0]), float(sampled[0, 1]), 0.0],
                "goal_position": [float(sampled[-1, 0]), float(sampled[-1, 1]), 0.0],
                "raw_path_length_m": raw_length,
                "path_length_m": path_length(sampled),
                "frame_count": len(sampled),
                "minimum_clearance_m": min(
                    float(clearance_m[transform.world_to_pixel(x, y)])
                    for x, y in sampled
                ),
                "minimum_camera_clearance_m": minimum_camera_clearance,
                "smoothing_method": smoothing_method,
                "points": sampled.astype(np.float32),
                "actions": actions,
                "camera_positions": camera_positions,
                "yaw": yaw,
                "point_goal": point_goal,
            }
        )
        used_endpoints.extend((start_xy, goal_xy))

    if len(episodes) != episode_count:
        raise RuntimeError(
            f"Generated only {len(episodes)}/{episode_count} episodes after "
            f"{max_attempts} attempts; rejections={rejection_counts}"
        )
    return episodes, {
        "attempts": attempt,
        "rejection_counts": rejection_counts,
        "usable_component_count": len(usable),
    }


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    assets = resolve_generation_assets(
        args.scene,
        args.sage_root,
        interiorgs_root=args.interiorgs_root,
        collision_usd=args.collision_usd,
    )
    scene_dir = assets.scene_dir
    collision_usd = assets.collision_usd

    collision_points, collision_faces = extract_collision_geometry(collision_usd)
    collision_mesh = trimesh.Trimesh(
        vertices=collision_points,
        faces=collision_faces,
        process=False,
    )
    safe, clearance_m, transform, map_info = load_navigation_map(
        scene_dir, args.robot_radius, args.safety_margin
    )
    camera_clearance = (
        args.robot_radius
        if args.camera_clearance is None
        else args.camera_clearance
    )
    if camera_clearance <= 0:
        raise ValueError("--camera-clearance must be positive")
    safe, camera_clearance_info = apply_camera_clearance(
        safe=safe,
        mesh=collision_mesh,
        transform=transform,
        camera_height=args.camera_height,
        camera_clearance=camera_clearance,
    )
    component_labels, components = connected_components(safe, transform.scale)
    map_info["components"] = components
    map_info["camera_collision_filter"] = camera_clearance_info
    map_info["safe_free_area_m2"] = float(
        safe.sum() * transform.scale**2
    )
    endpoint_clearance = (
        args.robot_radius
        + args.safety_margin
        + args.endpoint_extra_clearance
    )
    episodes, generation_info = generate_episodes(
        safe=safe,
        clearance_m=clearance_m,
        component_labels=component_labels,
        transform=transform,
        episode_count=args.episodes,
        seed=args.seed,
        min_path_length=args.min_path_length,
        max_path_length=args.max_path_length,
        frame_spacing=args.frame_spacing,
        endpoint_clearance=endpoint_clearance,
        max_attempts=args.max_attempts,
        camera_height=args.camera_height,
        collision_mesh=collision_mesh,
        camera_clearance=camera_clearance,
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
        args.pointcloud_voxel_size,
        args.pointcloud_max_points,
        args.seed,
    )
    write_binary_pointcloud(args.output_dir / "pointcloud.ply", pointcloud)

    save_navigation_visualizations(
        args.output_dir, safe, clearance_m, transform, episodes
    )
    from sage3d.schemas import build_trajectory_manifest, manifest_to_json

    manifest = build_trajectory_manifest(
        scene_id=args.scene,
        scene_dir=str(scene_dir),
        collision_usd=str(collision_usd),
        seed=args.seed,
        episodes=episodes,
        robot_radius_m=args.robot_radius,
        safety_margin_m=args.safety_margin,
        camera_height_m=args.camera_height,
        camera_clearance_m=camera_clearance,
        frame_spacing_m=args.frame_spacing,
        requested_path_length_range_m=[
            args.min_path_length,
            args.max_path_length,
        ],
        endpoint_clearance_m=endpoint_clearance,
        map_info=map_info,
        generation_info=generation_info,
        pointcloud={
            "source_vertex_count": len(collision_points),
            "output_point_count": len(pointcloud),
            "voxel_size_m": args.pointcloud_voxel_size,
            "bounds_min": pointcloud.min(axis=0).astype(float).tolist(),
            "bounds_max": pointcloud.max(axis=0).astype(float).tolist(),
            "color": [160, 160, 160],
        },
    )
    manifest_to_json(manifest, args.output_dir / "trajectory_manifest.json")

    print(
        f"Generated {len(episodes)} episodes for {args.scene}: "
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

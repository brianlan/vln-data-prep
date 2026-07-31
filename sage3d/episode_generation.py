"""Episode generation (exact move from generate_sage3d_trajectories.py).

Isaac-lane: imports numpy, trimesh, and sage3d leaf modules that require
scipy/pxr/cv2. Phase 3b moves ``generate_episodes`` and ``build_episode_arrays``
intact so the generation script can later be decomposed (Phase 3c).
"""

from __future__ import annotations

import trimesh
import numpy as np

from sage3d.collision import collision_distances
from sage3d.geometry import MapTransform, path_length, pixels_to_world, wrap_angle
from sage3d.path_postprocess import (
    points_are_safe,
    resample_path,
    smooth_path,
)
from sage3d.pathfinding import astar


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
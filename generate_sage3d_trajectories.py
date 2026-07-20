#!/usr/bin/env python3
"""Generate deterministic, collision-aware PointGoal trajectories for SAGE3D."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from pxr import Gf, Usd, UsdGeom
from scipy.interpolate import splprep, splev
import trimesh


NEIGHBORS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


@dataclass(frozen=True)
class MapTransform:
    height: int
    width: int
    scale: float
    lower_x: float
    lower_y: float

    def pixel_to_world(self, row: int, col: int) -> tuple[float, float]:
        x = self.lower_x + (col + 0.5) * self.scale
        # Raw InteriorGS occupancy maps use row 0 at the lower world-Y bound.
        # SAGE3D's semantic-map export flips the raw occupancy image for
        # visualization, but that flip must not be applied while planning
        # directly on occupancy.png.
        y = self.lower_y + (row + 0.5) * self.scale
        return x, y

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        col = int(round((x - self.lower_x) / self.scale - 0.5))
        row = int(round((y - self.lower_y) / self.scale - 0.5))
        return row, col


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="Numeric SAGE3D scene ID")
    parser.add_argument(
        "--interiorgs-root",
        type=Path,
        default=Path("/ssd5/datasets/SAGE3D/InteriorGS"),
    )
    parser.add_argument(
        "--collision-usd",
        type=Path,
        help="Defaults to the standard SAGE3D collision-mesh location",
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


def resolve_scene_dir(root: Path, scene: str) -> Path:
    matches = sorted(root.glob(f"*_{scene}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one InteriorGS directory matching '*_{scene}' "
            f"under {root}, found {len(matches)}"
        )
    return matches[0]


def load_navigation_map(
    scene_dir: Path, robot_radius: float, safety_margin: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MapTransform, dict]:
    occupancy_path = scene_dir / "occupancy.png"
    occupancy_meta_path = scene_dir / "occupancy.json"
    structure_path = scene_dir / "structure.json"
    for path in (occupancy_path, occupancy_meta_path, structure_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    occupancy = np.asarray(Image.open(occupancy_path).convert("L"))
    with occupancy_meta_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    with structure_path.open("r", encoding="utf-8") as file:
        structure = json.load(file)

    height, width = occupancy.shape
    transform = MapTransform(
        height=height,
        width=width,
        scale=float(metadata["scale"]),
        lower_x=float(metadata["lower"][0]),
        lower_y=float(metadata["lower"][1]),
    )

    room_mask = np.zeros((height, width), dtype=np.uint8)
    valid_rooms = 0
    for room in structure.get("rooms", []):
        profile = room.get("profile", [])
        if len(profile) < 3:
            continue
        pixels = []
        for x, y in profile:
            row, col = transform.world_to_pixel(float(x), float(y))
            pixels.append((col, row))
        cv2.fillPoly(room_mask, [np.asarray(pixels, dtype=np.int32)], 1)
        valid_rooms += 1
    if valid_rooms == 0:
        raise RuntimeError(f"No valid room polygons in {structure_path}")

    # Unknown and exterior pixels are deliberately blocked. Some InteriorGS
    # occupancy PNGs use white for both interior free space and canvas background.
    raw_free = (occupancy == 255) & (room_mask > 0)
    clearance_m = (
        cv2.distanceTransform(
            raw_free.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        * transform.scale
    )
    safe = raw_free & (clearance_m >= robot_radius + safety_margin)

    component_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(safe.astype(np.uint8), connectivity=4)
    )
    components = []
    for label in range(1, component_count):
        cells = int(component_stats[label, cv2.CC_STAT_AREA])
        components.append(
            {
                "label": label,
                "cells": cells,
                "area_m2": cells * transform.scale**2,
            }
        )

    map_info = {
        "shape": [height, width],
        "scale_m_per_pixel": transform.scale,
        "robot_radius_m": robot_radius,
        "safety_margin_m": safety_margin,
        "required_path_clearance_m": robot_radius + safety_margin,
        "room_count": valid_rooms,
        "raw_free_area_m2": float(raw_free.sum() * transform.scale**2),
        "safe_free_area_m2": float(safe.sum() * transform.scale**2),
        "components": components,
        "occupancy_values": {
            str(int(value)): int(count)
            for value, count in zip(*np.unique(occupancy, return_counts=True))
        },
    }
    return safe, clearance_m, component_labels, transform, map_info


def connected_components(safe: np.ndarray, scale: float) -> tuple[np.ndarray, list[dict]]:
    component_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(safe.astype(np.uint8), connectivity=4)
    )
    components = []
    for label in range(1, component_count):
        cells = int(component_stats[label, cv2.CC_STAT_AREA])
        components.append(
            {
                "label": label,
                "cells": cells,
                "area_m2": cells * scale**2,
            }
        )
    return component_labels, components


def collision_distances(
    mesh: trimesh.Trimesh, query_points: np.ndarray, batch_size: int = 2048
) -> np.ndarray:
    distances = np.empty(len(query_points), dtype=np.float64)
    for start in range(0, len(query_points), batch_size):
        stop = min(start + batch_size, len(query_points))
        _, batch_distances, _ = trimesh.proximity.closest_point(
            mesh, query_points[start:stop]
        )
        distances[start:stop] = batch_distances
    return distances


def apply_camera_clearance(
    safe: np.ndarray,
    mesh: trimesh.Trimesh,
    transform: MapTransform,
    camera_height: float,
    camera_clearance: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rows, cols = np.where(safe)
    query_points = np.asarray(
        [
            (*transform.pixel_to_world(int(row), int(col)), camera_height)
            for row, col in zip(rows, cols)
        ],
        dtype=np.float64,
    )
    distances = collision_distances(mesh, query_points)
    distance_map = np.zeros(safe.shape, dtype=np.float32)
    distance_map[rows, cols] = distances.astype(np.float32)
    camera_safe = safe & (distance_map >= camera_clearance)
    removed = int(safe.sum() - camera_safe.sum())
    return (
        camera_safe,
        distance_map,
        {
            "camera_height_m": camera_height,
            "required_camera_clearance_m": camera_clearance,
            "queried_2d_safe_cells": int(len(query_points)),
            "removed_cells": removed,
            "remaining_cells": int(camera_safe.sum()),
            "remaining_area_m2": float(camera_safe.sum() * transform.scale**2),
        },
    )


def astar(
    safe: np.ndarray,
    clearance_m: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    scale: float,
) -> list[tuple[int, int]] | None:
    height, width = safe.shape
    g_score = np.full((height, width), np.inf, dtype=np.float32)
    parent_row = np.full((height, width), -1, dtype=np.int32)
    parent_col = np.full((height, width), -1, dtype=np.int32)
    closed = np.zeros((height, width), dtype=bool)

    def heuristic(row: int, col: int) -> float:
        return math.hypot(row - goal[0], col - goal[1]) * scale

    g_score[start] = 0.0
    queue: list[tuple[float, float, int, int]] = [
        (heuristic(*start), 0.0, start[0], start[1])
    ]

    while queue:
        _, current_g, row, col = heapq.heappop(queue)
        if closed[row, col]:
            continue
        closed[row, col] = True
        if (row, col) == goal:
            path = []
            cursor = goal
            while cursor != (-1, -1):
                path.append(cursor)
                parent = (
                    int(parent_row[cursor]),
                    int(parent_col[cursor]),
                )
                cursor = parent
            return path[::-1]

        for d_row, d_col, step_factor in NEIGHBORS:
            n_row, n_col = row + d_row, col + d_col
            if not (0 <= n_row < height and 0 <= n_col < width):
                continue
            if not safe[n_row, n_col] or closed[n_row, n_col]:
                continue
            if d_row and d_col:
                # Prevent diagonal corner cutting.
                if not safe[row + d_row, col] or not safe[row, col + d_col]:
                    continue
            clearance = max(float(clearance_m[n_row, n_col]), 0.05)
            clearance_multiplier = 1.0 + 0.12 / clearance
            tentative = (
                current_g + step_factor * scale * clearance_multiplier
            )
            if tentative >= float(g_score[n_row, n_col]):
                continue
            g_score[n_row, n_col] = tentative
            parent_row[n_row, n_col] = row
            parent_col[n_row, n_col] = col
            heapq.heappush(
                queue,
                (
                    tentative + heuristic(n_row, n_col),
                    tentative,
                    n_row,
                    n_col,
                ),
            )
    return None


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def pixels_to_world(
    pixels: Iterable[tuple[int, int]], transform: MapTransform
) -> np.ndarray:
    return np.asarray(
        [transform.pixel_to_world(row, col) for row, col in pixels],
        dtype=np.float64,
    )


def sample_segment(start: np.ndarray, end: np.ndarray, step: float) -> np.ndarray:
    length = float(np.linalg.norm(end - start))
    count = max(2, int(math.ceil(length / step)) + 1)
    return np.linspace(start, end, count)


def points_are_safe(
    points: np.ndarray,
    safe: np.ndarray,
    transform: MapTransform,
    check_step: float | None = None,
) -> bool:
    if len(points) == 0:
        return False
    check_step = check_step or transform.scale * 0.5
    samples = []
    for index in range(len(points) - 1):
        samples.append(sample_segment(points[index], points[index + 1], check_step))
    if samples:
        test_points = np.concatenate(samples, axis=0)
    else:
        test_points = points
    for x, y in test_points:
        row, col = transform.world_to_pixel(float(x), float(y))
        if not (
            0 <= row < transform.height
            and 0 <= col < transform.width
            and safe[row, col]
        ):
            return False
    return True


def line_is_safe(
    start: np.ndarray,
    end: np.ndarray,
    safe: np.ndarray,
    transform: MapTransform,
) -> bool:
    return points_are_safe(
        np.asarray([start, end]), safe, transform, transform.scale * 0.5
    )


def simplify_by_visibility(
    points: np.ndarray, safe: np.ndarray, transform: MapTransform
) -> np.ndarray:
    if len(points) <= 2:
        return points
    simplified = [points[0]]
    current = 0
    while current < len(points) - 1:
        candidate = len(points) - 1
        while candidate > current + 1:
            if line_is_safe(points[current], points[candidate], safe, transform):
                break
            candidate -= 1
        simplified.append(points[candidate])
        current = candidate
    return np.asarray(simplified)


def smooth_path(
    points: np.ndarray, safe: np.ndarray, transform: MapTransform
) -> tuple[np.ndarray, str]:
    simplified = simplify_by_visibility(points, safe, transform)
    if len(simplified) < 4:
        return simplified, "line_of_sight"

    distances = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(simplified, axis=0), axis=1)))
    )
    if distances[-1] <= 0:
        return simplified, "line_of_sight"
    parameter = distances / distances[-1]
    sample_count = max(
        len(simplified) * 8,
        int(math.ceil(distances[-1] / (transform.scale * 0.4))) + 1,
    )
    sample_parameter = np.linspace(0.0, 1.0, sample_count)

    for smoothing_per_point in (0.002, 0.0005, 0.0):
        try:
            spline, _ = splprep(
                [simplified[:, 0], simplified[:, 1]],
                u=parameter,
                s=smoothing_per_point * len(simplified),
                k=min(3, len(simplified) - 1),
            )
            x_values, y_values = splev(sample_parameter, spline)
            candidate = np.column_stack((x_values, y_values))
            candidate[0] = points[0]
            candidate[-1] = points[-1]
            if points_are_safe(candidate, safe, transform):
                return candidate, f"cubic_spline_s={smoothing_per_point}"
        except ValueError:
            continue

    # Cubic splines can overshoot at tight obstacle corners. Round each corner
    # independently with a quadratic Bezier curve and retain a sharp corner only
    # where the rounded candidate would violate the clearance mask.
    rounded = [simplified[0]]
    rounded_corner_count = 0
    for index in range(1, len(simplified) - 1):
        previous, corner, following = simplified[index - 1 : index + 2]
        incoming = corner - previous
        outgoing = following - corner
        incoming_length = float(np.linalg.norm(incoming))
        outgoing_length = float(np.linalg.norm(outgoing))
        if incoming_length < 1e-6 or outgoing_length < 1e-6:
            rounded.append(corner)
            continue
        cut = min(0.25, incoming_length * 0.3, outgoing_length * 0.3)
        entry = corner - incoming / incoming_length * cut
        exit_point = corner + outgoing / outgoing_length * cut
        parameter = np.linspace(0.0, 1.0, 9)
        curve = (
            (1.0 - parameter)[:, None] ** 2 * entry
            + 2.0
            * (1.0 - parameter)[:, None]
            * parameter[:, None]
            * corner
            + parameter[:, None] ** 2 * exit_point
        )
        candidate = np.vstack((rounded[-1], curve))
        if points_are_safe(candidate, safe, transform):
            rounded.extend(curve)
            rounded_corner_count += 1
        else:
            rounded.append(corner)
    rounded.append(simplified[-1])
    rounded_array = np.asarray(rounded)
    if rounded_corner_count and points_are_safe(
        rounded_array, safe, transform
    ):
        return (
            rounded_array,
            f"clearance_checked_bezier_corners_{rounded_corner_count}",
        )
    return simplified, "line_of_sight_fallback"


def resample_path(points: np.ndarray, spacing: float) -> np.ndarray:
    if len(points) < 2:
        return points.copy()
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = cumulative[-1]
    if total <= 0:
        return points[:1].copy()
    sample_distances = np.arange(0.0, total, spacing)
    if not np.isclose(sample_distances[-1], total):
        sample_distances = np.append(sample_distances, total)
    x_values = np.interp(sample_distances, cumulative, points[:, 0])
    y_values = np.interp(sample_distances, cumulative, points[:, 1])
    return np.column_stack((x_values, y_values))


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


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


def save_navigation_visualizations(
    output_dir: Path,
    safe: np.ndarray,
    clearance_m: np.ndarray,
    transform: MapTransform,
    episodes: list[dict],
) -> None:
    safe_image = np.zeros((*safe.shape, 3), dtype=np.uint8)
    normalized_clearance = np.clip(clearance_m / max(clearance_m.max(), 1e-6), 0, 1)
    safe_image[..., 0] = (normalized_clearance * 120).astype(np.uint8)
    safe_image[..., 1] = np.where(safe, 180, 0).astype(np.uint8)
    safe_image[..., 2] = np.where(safe, 80, 0).astype(np.uint8)
    Image.fromarray(safe_image).save(output_dir / "navigation_map.png")

    overlay = safe_image.copy()
    colors = (
        (255, 80, 80),
        (80, 180, 255),
        (255, 210, 70),
        (180, 80, 255),
        (80, 255, 160),
        (255, 130, 30),
    )
    for episode in episodes:
        pixels = [
            transform.world_to_pixel(float(x), float(y))
            for x, y in episode["points"]
        ]
        polyline = np.asarray([(col, row) for row, col in pixels], dtype=np.int32)
        color = colors[episode["episode_index"] % len(colors)]
        cv2.polylines(overlay, [polyline], False, color, 2, cv2.LINE_AA)
        cv2.circle(overlay, tuple(polyline[0]), 3, (255, 255, 255), -1)
        cv2.circle(overlay, tuple(polyline[-1]), 3, color, -1)
    Image.fromarray(overlay).save(output_dir / "trajectories_overlay.png")


def extract_collision_geometry(
    collision_usd: Path,
) -> tuple[np.ndarray, np.ndarray]:
    stage = Usd.Stage.Open(str(collision_usd))
    if stage is None:
        raise RuntimeError(f"Could not open collision USD: {collision_usd}")
    transform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    chunks = []
    face_chunks = []
    vertex_offset = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        points_value = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not points_value:
            continue
        counts = np.asarray(
            UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get(), dtype=np.int64
        )
        indices = np.asarray(
            UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get(), dtype=np.int64
        )
        if not len(counts) or not np.all(counts == 3):
            raise RuntimeError(
                f"Collision mesh {prim.GetPath()} is not fully triangulated"
            )
        points = np.asarray(points_value, dtype=np.float64)
        matrix = np.asarray(
            transform_cache.GetLocalToWorldTransform(prim), dtype=np.float64
        )
        homogeneous = np.column_stack((points, np.ones(len(points))))
        world_points = (homogeneous @ matrix)[:, :3]
        chunks.append(world_points)
        face_chunks.append(indices.reshape(-1, 3) + vertex_offset)
        vertex_offset += len(world_points)
    if not chunks:
        raise RuntimeError(f"No mesh vertices found in {collision_usd}")
    return np.concatenate(chunks, axis=0), np.concatenate(face_chunks, axis=0)


def voxel_downsample(
    points: np.ndarray, voxel_size: float, max_points: int, seed: int
) -> np.ndarray:
    voxel = np.floor(points / voxel_size).astype(np.int64)
    _, indices = np.unique(voxel, axis=0, return_index=True)
    sampled = points[np.sort(indices)]
    if len(sampled) > max_points:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(sampled), max_points, replace=False))
        sampled = sampled[selected]
    return sampled.astype(np.float32)


def write_binary_pointcloud(path: Path, points: np.ndarray) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment SAGE3D collision mesh voxel point cloud\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    record = struct.Struct("<fffBBB")
    with path.open("wb") as file:
        file.write(header)
        for x, y, z in points:
            file.write(record.pack(float(x), float(y), float(z), 160, 160, 160))


def serializable_episode(episode: dict) -> dict:
    return {
        key: value
        for key, value in episode.items()
        if key not in {"points", "actions", "camera_positions", "yaw", "point_goal"}
    }


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scene_dir = resolve_scene_dir(args.interiorgs_root, args.scene)
    collision_usd = args.collision_usd or (
        Path("/ssd5/datasets/SAGE3D/Collision_Mesh/Collision_Mesh")
        / args.scene
        / f"{args.scene}_collision.usd"
    )
    if not collision_usd.is_file():
        raise FileNotFoundError(collision_usd)

    collision_points, collision_faces = extract_collision_geometry(collision_usd)
    collision_mesh = trimesh.Trimesh(
        vertices=collision_points,
        faces=collision_faces,
        process=False,
    )
    safe, clearance_m, component_labels, transform, map_info = load_navigation_map(
        scene_dir, args.robot_radius, args.safety_margin
    )
    camera_clearance = (
        args.robot_radius
        if args.camera_clearance is None
        else args.camera_clearance
    )
    if camera_clearance <= 0:
        raise ValueError("--camera-clearance must be positive")
    safe, _camera_distance_m, camera_clearance_info = apply_camera_clearance(
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
        episode_path = (
            args.output_dir
            / f"episode_{episode['episode_index']:06d}.npz"
        )
        np.savez_compressed(
            episode_path,
            points=episode["points"],
            actions=episode["actions"],
            camera_positions=episode["camera_positions"],
            yaw=episode["yaw"],
            point_goal=episode["point_goal"],
            start_position=np.asarray(episode["start_position"], dtype=np.float32),
            goal_position=np.asarray(episode["goal_position"], dtype=np.float32),
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
    manifest = {
        "scene_id": args.scene,
        "scene_dir": str(scene_dir),
        "collision_usd": str(collision_usd),
        "seed": args.seed,
        "episode_count": len(episodes),
        "robot_radius_m": args.robot_radius,
        "safety_margin_m": args.safety_margin,
        "camera_height_m": args.camera_height,
        "camera_clearance_m": camera_clearance,
        "frame_spacing_m": args.frame_spacing,
        "requested_path_length_range_m": [
            args.min_path_length,
            args.max_path_length,
        ],
        "endpoint_clearance_m": endpoint_clearance,
        "map": map_info,
        "generation": generation_info,
        "pointcloud": {
            "source_vertex_count": len(collision_points),
            "output_point_count": len(pointcloud),
            "voxel_size_m": args.pointcloud_voxel_size,
            "bounds_min": pointcloud.min(axis=0).astype(float).tolist(),
            "bounds_max": pointcloud.max(axis=0).astype(float).tolist(),
            "color": [160, 160, 160],
        },
        "episodes": [serializable_episode(episode) for episode in episodes],
    }
    with (args.output_dir / "trajectory_manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=2)

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

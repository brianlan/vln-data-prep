"""Typed artifact schemas and calibration authority (numpy only).

Package-safe: stdlib + numpy. No Isaac, cv2, scipy, trimesh, or PIL.

Phase 2a introduces typed manifest/render summary models so generate and render
write through a single schema implementation, and package derives calibration
from the authoritative render summary. The JSON key order matches the legacy
dict construction exactly, so ``json.dumps(model, indent=2)`` is byte-identical
to the legacy ``json.dump(dict, indent=2)``.

TypedDicts document the required keys; :func:`to_json` serializes with
``json.dumps(indent=2)`` preserving insertion order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict


# --- trajectory manifest episode record --------------------------------------


class TrajectoryEpisodeRecord(TypedDict):
    """Manifest episode record (the serializable subset of an episode dict).

    Distinct from the LeRobot-style ``meta/episodes.jsonl`` record (Phase 5).
    Keys are ordered to match ``generate_sage3d_trajectories.py::serializable_episode``.
    """

    episode_index: int
    component_id: int
    start_pixel: list[int]
    goal_pixel: list[int]
    start_position: list[float]
    goal_position: list[float]
    raw_path_length_m: float
    path_length_m: float
    frame_count: int
    minimum_clearance_m: float
    minimum_camera_clearance_m: float
    smoothing_method: str


# --- trajectory manifest -----------------------------------------------------


class TrajectoryManifest(TypedDict):
    """Full trajectory manifest, ordered to match the legacy construction."""

    scene_id: str
    scene_dir: str
    collision_usd: str
    seed: int
    episode_count: int
    robot_radius_m: float
    safety_margin_m: float
    camera_height_m: float
    camera_clearance_m: float
    frame_spacing_m: float
    requested_path_length_range_m: list[float]
    endpoint_clearance_m: float
    map: dict[str, Any]
    generation: dict[str, Any]
    pointcloud: dict[str, Any]
    episodes: list[TrajectoryEpisodeRecord]


def build_trajectory_manifest(
    *,
    scene_id: str,
    scene_dir: str,
    collision_usd: str,
    seed: int,
    episodes: list[dict],
    robot_radius_m: float,
    safety_margin_m: float,
    camera_height_m: float,
    camera_clearance_m: float,
    frame_spacing_m: float,
    requested_path_length_range_m: list[float],
    endpoint_clearance_m: float,
    map_info: dict[str, Any],
    generation_info: dict[str, Any],
    pointcloud: dict[str, Any],
) -> TrajectoryManifest:
    """Build a manifest dict with the exact legacy key order."""
    return TrajectoryManifest(
        scene_id=scene_id,
        scene_dir=scene_dir,
        collision_usd=collision_usd,
        seed=seed,
        episode_count=len(episodes),
        robot_radius_m=robot_radius_m,
        safety_margin_m=safety_margin_m,
        camera_height_m=camera_height_m,
        camera_clearance_m=camera_clearance_m,
        frame_spacing_m=frame_spacing_m,
        requested_path_length_range_m=requested_path_length_range_m,
        endpoint_clearance_m=endpoint_clearance_m,
        map=map_info,
        generation=generation_info,
        pointcloud=pointcloud,
        episodes=[serializable_episode(ep) for ep in episodes],
    )


def serializable_episode(episode: dict) -> TrajectoryEpisodeRecord:
    """Return the serializable subset of an episode dict, preserving key order."""
    return TrajectoryEpisodeRecord(
        episode_index=episode["episode_index"],
        component_id=episode["component_id"],
        start_pixel=episode["start_pixel"],
        goal_pixel=episode["goal_pixel"],
        start_position=episode["start_position"],
        goal_position=episode["goal_position"],
        raw_path_length_m=episode["raw_path_length_m"],
        path_length_m=episode["path_length_m"],
        frame_count=episode["frame_count"],
        minimum_clearance_m=episode["minimum_clearance_m"],
        minimum_camera_clearance_m=episode["minimum_camera_clearance_m"],
        smoothing_method=episode["smoothing_method"],
    )


def manifest_to_json(manifest: TrajectoryManifest, path: Path) -> None:
    """Write the manifest as JSON with ``indent=2``, matching legacy output."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


# --- render summary (mode-aware) ---------------------------------------------


class RenderSummaryEpisode(TypedDict):
    """Depth-mode episode summary record."""

    episode_index: int
    frame_count: int
    finite_depth_fraction_mean: float
    finite_depth_fraction_min: float
    finite_depth_min_m: float
    finite_depth_max_m: float


class RenderSummary(TypedDict):
    """Mode-aware render summary.

    RGB mode writes ``episodes=[]``; depth mode appends per-episode depth
    statistics. Keys are ordered to match ``render_fisheye_sage3d.py``.
    """

    scene_id: str
    camera_model: str
    resolution: list[int]
    horizontal_fov_deg: float
    vertical_fov_deg: float
    focal_length_pixels: float
    principal_point: list[float]
    fisheye_coefficients: list[float]
    forward_mask_radius_pixels: float
    camera_pitch_deg: float
    depth_type: str
    max_depth_m: float
    min_depth_m: float
    depth_scale: float
    render_mode: str
    episodes: list[RenderSummaryEpisode]
    total_frames: int


def build_render_summary(
    *,
    scene_id: str,
    width: int,
    height: int,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
    focal_length_pixels: float,
    principal_point: list[float],
    fisheye_coefficients: list[float],
    forward_mask_radius_pixels: float,
    max_depth_m: float,
    min_depth_m: float,
    depth_scale: float,
    render_mode: str,
    episodes: list[RenderSummaryEpisode],
    total_frames: int,
) -> RenderSummary:
    """Build a render summary dict with the exact legacy key order."""
    return RenderSummary(
        scene_id=scene_id,
        camera_model="opencv_fisheye",
        resolution=[width, height],
        horizontal_fov_deg=horizontal_fov_deg,
        vertical_fov_deg=vertical_fov_deg,
        focal_length_pixels=focal_length_pixels,
        principal_point=principal_point,
        fisheye_coefficients=fisheye_coefficients,
        forward_mask_radius_pixels=forward_mask_radius_pixels,
        camera_pitch_deg=0.0,
        depth_type="distance_to_camera",
        max_depth_m=max_depth_m,
        min_depth_m=min_depth_m,
        depth_scale=depth_scale,
        render_mode=render_mode,
        episodes=episodes,
        total_frames=total_frames,
    )


def render_summary_to_json(summary: RenderSummary, path: Path) -> None:
    """Write the render summary as JSON with ``indent=2``, matching legacy output."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
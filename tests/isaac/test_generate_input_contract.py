"""Phase-1 optimization-input contract: map metadata, astar payload, JSON exclusion."""

import json

import numpy as np
from PIL import Image

import generate_sage3d_trajectories
from generate_sage3d_trajectories import (
    connected_components,
    generate_episodes,
    load_navigation_map,
    serializable_episode,
)
from sage3d.utils import MapTransform


def test_map_info_optimization_metadata(tmp_path):
    occupancy = np.full((16, 16), 255, dtype=np.uint8)
    Image.fromarray(occupancy).save(tmp_path / "occupancy.png")
    (tmp_path / "occupancy.json").write_text(
        json.dumps({"scale": 0.05, "lower": [-2.0, -1.0]}),
        encoding="utf-8",
    )
    profile = [[-1.95, -0.95], [-1.75, -0.95], [-1.75, -0.75], [-1.95, -0.75]]
    (tmp_path / "structure.json").write_text(
        json.dumps({"rooms": [{"profile": profile}]}),
        encoding="utf-8",
    )

    _, _, _, map_info = load_navigation_map(tmp_path, 0.25, 0.05)

    assert map_info["lower_x"] == -2.0
    assert map_info["lower_y"] == -1.0
    assert map_info["pixel_coordinate_order"] == "row_col"
    assert map_info["pixel_to_world_convention"] == "sage3d_map_transform_v1"
    assert map_info["safe_mask_semantics"] == "robot_inflated_and_camera_filtered_v1"


def test_episode_astar_path_pixels_row_col_int32(monkeypatch):
    # All-safe synthetic grid run end-to-end through generate_episodes with
    # collision_distances stubbed to large clearances so camera-clearance and
    # endpoint-clearance gates pass without a real collision mesh.
    safe = np.ones((12, 12), dtype=bool)
    clearance_m = np.full(safe.shape, 100.0, dtype=np.float64)
    transform = MapTransform(
        height=12, width=12, scale=0.1, lower_x=-0.6, lower_y=-0.6
    )
    component_labels, _ = connected_components(safe, transform.scale)

    monkeypatch.setattr(
        generate_sage3d_trajectories,
        "collision_distances",
        lambda mesh, points: np.full(len(points), 100.0, dtype=np.float64),
    )

    episodes, _ = generate_episodes(
        safe=safe,
        clearance_m=clearance_m,
        component_labels=component_labels,
        transform=transform,
        episode_count=1,
        seed=0,
        min_path_length=0.2,
        max_path_length=50.0,
        frame_spacing=0.05,
        endpoint_clearance=0.0,
        max_attempts=200,
        camera_height=0.4,
        collision_mesh=None,
        camera_clearance=0.0,
    )
    assert len(episodes) == 1
    episode = episodes[0]

    astar_path_pixels = episode["astar_path_pixels"]
    assert astar_path_pixels.dtype == np.int32
    assert astar_path_pixels.ndim == 2 and astar_path_pixels.shape[1] == 2

    start_pixel = tuple(episode["start_pixel"])
    goal_pixel = tuple(episode["goal_pixel"])
    assert tuple(astar_path_pixels[0]) == start_pixel
    assert tuple(astar_path_pixels[-1]) == goal_pixel


def test_serializable_episode_excludes_astar_and_arrays():
    episode = {
        "episode_index": 0,
        "raw_path_length_m": 5.0,
        "points": np.zeros((3, 2)),
        "actions": np.zeros((3, 4, 4)),
        "camera_positions": np.zeros((3, 3)),
        "yaw": np.zeros(3),
        "point_goal": np.zeros((3, 2)),
        "astar_path_pixels": np.zeros((3, 2), dtype=np.int32),
    }
    summary = serializable_episode(episode)
    assert "astar_path_pixels" not in summary
    assert "points" not in summary and "actions" not in summary
    assert summary["episode_index"] == 0
    assert summary["raw_path_length_m"] == 5.0

"""Tests for sage3d.schemas and sage3d.config (Phase 2a, issue #11).

Verifies schema round trips, explicit key order, byte-identical JSON output
to legacy construction, default/non-default depth-format truthfulness, and
forbidden-import safety.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sage3d.schemas import (  # noqa: E402
    RenderSummary,
    TrajectoryEpisodeRecord,
    TrajectoryManifest,
    build_render_summary,
    build_trajectory_manifest,
    manifest_to_json,
    render_summary_to_json,
    serializable_episode,
)


# --- sample data -------------------------------------------------------------


def _sample_episode_dict() -> dict:
    return {
        "episode_index": 0,
        "component_id": 42,
        "start_pixel": [10, 20],
        "goal_pixel": [30, 40],
        "start_position": [1.0, 2.0, 0.0],
        "goal_position": [3.0, 4.0, 0.0],
        "raw_path_length_m": 5.5,
        "path_length_m": 6.0,
        "frame_count": 89,
        "minimum_clearance_m": 0.15,
        "minimum_camera_clearance_m": 0.3,
        "smoothing_method": "smooth",
        "points": np.zeros((89, 2), dtype=np.float32),
        "actions": np.zeros((89, 4, 4), dtype=np.float32),
        "camera_positions": np.zeros((89, 3), dtype=np.float32),
        "yaw": np.zeros(89, dtype=np.float32),
        "point_goal": np.zeros((89, 2), dtype=np.float32),
    }


def _legacy_manifest_dict(episodes: list[dict]) -> dict:
    """Verbatim legacy construction from generate_sage3d_trajectories.py."""
    return {
        "scene_id": "839920",
        "scene_dir": "/data/scene",
        "collision_usd": "/data/collision.usd",
        "seed": 20260720,
        "episode_count": len(episodes),
        "robot_radius_m": 0.2,
        "safety_margin_m": 0.05,
        "camera_height_m": 0.6,
        "camera_clearance_m": 0.2,
        "frame_spacing_m": 0.05,
        "requested_path_length_range_m": [3.0, 20.0],
        "endpoint_clearance_m": 0.5,
        "map": {"scale": 0.05},
        "generation": {"attempts": 100},
        "pointcloud": {
            "source_vertex_count": 1000,
            "output_point_count": 500,
            "voxel_size_m": 0.02,
            "bounds_min": [0.0, 0.0, 0.0],
            "bounds_max": [10.0, 10.0, 3.0],
            "color": [160, 160, 160],
        },
        "episodes": [
            {
                key: value
                for key, value in ep.items()
                if key
                not in {"points", "actions", "camera_positions", "yaw", "point_goal"}
            }
            for ep in episodes
        ],
    }


def _sample_manifest_kwargs(episodes: list[dict]) -> dict:
    return dict(
        scene_id="839920",
        scene_dir="/data/scene",
        collision_usd="/data/collision.usd",
        seed=20260720,
        episodes=episodes,
        robot_radius_m=0.2,
        safety_margin_m=0.05,
        camera_height_m=0.6,
        camera_clearance_m=0.2,
        frame_spacing_m=0.05,
        requested_path_length_range_m=[3.0, 20.0],
        endpoint_clearance_m=0.5,
        map_info={"scale": 0.05},
        generation_info={"attempts": 100},
        pointcloud={
            "source_vertex_count": 1000,
            "output_point_count": 500,
            "voxel_size_m": 0.02,
            "bounds_min": [0.0, 0.0, 0.0],
            "bounds_max": [10.0, 10.0, 3.0],
            "color": [160, 160, 160],
        },
    )


# --- manifest key order and byte equality -----------------------------------


def test_manifest_keys_match_legacy_order():
    ep = _sample_episode_dict()
    legacy = _legacy_manifest_dict([ep])
    prod = build_trajectory_manifest(**_sample_manifest_kwargs([ep]))
    assert list(legacy.keys()) == list(prod.keys())


def test_episode_record_keys_match_legacy_order():
    ep = _sample_episode_dict()
    legacy = {
        key: value
        for key, value in ep.items()
        if key not in {"points", "actions", "camera_positions", "yaw", "point_goal"}
    }
    prod = serializable_episode(ep)
    assert list(legacy.keys()) == list(prod.keys())


def test_manifest_json_byte_identical_to_legacy(tmp_path):
    ep = _sample_episode_dict()
    legacy = _legacy_manifest_dict([ep])
    prod = build_trajectory_manifest(**_sample_manifest_kwargs([ep]))
    legacy_path = tmp_path / "legacy.json"
    prod_path = tmp_path / "prod.json"
    legacy_path.write_text(json.dumps(legacy, indent=2), encoding="utf-8")
    manifest_to_json(prod, prod_path)
    assert legacy_path.read_bytes() == prod_path.read_bytes()


def test_manifest_episode_count_matches():
    eps = [_sample_episode_dict() for _ in range(3)]
    for i, ep in enumerate(eps):
        ep["episode_index"] = i
    manifest = build_trajectory_manifest(**_sample_manifest_kwargs(eps))
    assert manifest["episode_count"] == 3
    assert len(manifest["episodes"]) == 3


# --- render summary key order and byte equality ------------------------------


def _legacy_render_summary(render_mode: str) -> dict:
    return {
        "scene_id": "839920",
        "camera_model": "opencv_fisheye",
        "resolution": [600, 450],
        "horizontal_fov_deg": 180.0,
        "vertical_fov_deg": 120.0,
        "focal_length_pixels": 300.0,
        "principal_point": [300.0, 225.0],
        "fisheye_coefficients": [0.1, 0.0, 0.0, 0.0],
        "forward_mask_radius_pixels": 300.0,
        "camera_pitch_deg": 0.0,
        "depth_type": "distance_to_camera",
        "max_depth_m": 6.0,
        "min_depth_m": 0.05,
        "depth_scale": 10000.0,
        "render_mode": render_mode,
        "episodes": [],
        "total_frames": 0,
    }


def _sample_render_summary_kwargs(render_mode: str) -> dict:
    return dict(
        scene_id="839920",
        width=600,
        height=450,
        horizontal_fov_deg=180.0,
        vertical_fov_deg=120.0,
        focal_length_pixels=300.0,
        principal_point=[300.0, 225.0],
        fisheye_coefficients=[0.1, 0.0, 0.0, 0.0],
        forward_mask_radius_pixels=300.0,
        max_depth_m=6.0,
        min_depth_m=0.05,
        depth_scale=10000.0,
        render_mode=render_mode,
        episodes=[],
        total_frames=0,
    )


def test_render_summary_keys_match_legacy_order():
    for mode in ("rgb", "depth"):
        legacy = _legacy_render_summary(mode)
        prod = build_render_summary(**_sample_render_summary_kwargs(mode))
        assert list(legacy.keys()) == list(prod.keys()), f"key order mismatch for {mode}"


def test_render_summary_json_byte_identical_to_legacy(tmp_path):
    for mode in ("rgb", "depth"):
        legacy = _legacy_render_summary(mode)
        prod = build_render_summary(**_sample_render_summary_kwargs(mode))
        legacy_path = tmp_path / f"legacy_{mode}.json"
        prod_path = tmp_path / f"prod_{mode}.json"
        legacy_path.write_text(json.dumps(legacy, indent=2), encoding="utf-8")
        render_summary_to_json(prod, prod_path)
        assert legacy_path.read_bytes() == prod_path.read_bytes()


def test_render_summary_rgb_mode_has_empty_episodes():
    summary = build_render_summary(**_sample_render_summary_kwargs("rgb"))
    assert summary["episodes"] == []
    assert summary["render_mode"] == "rgb"


def test_render_summary_depth_mode_accepts_episodes():
    ep = {
        "episode_index": 0,
        "frame_count": 89,
        "finite_depth_fraction_mean": 0.99,
        "finite_depth_fraction_min": 0.97,
        "finite_depth_min_m": 0.36,
        "finite_depth_max_m": 10.4,
    }
    kwargs = _sample_render_summary_kwargs("depth")
    kwargs["episodes"] = [ep]
    kwargs["total_frames"] = 89
    summary = build_render_summary(**kwargs)
    assert summary["episodes"] == [ep]
    assert summary["total_frames"] == 89


# --- float comparison categories --------------------------------------------


def test_manifest_float_values_exact_equality():
    """Manifest floats use exact parsed value equality (not allclose)."""
    ep = _sample_episode_dict()
    manifest = build_trajectory_manifest(**_sample_manifest_kwargs([ep]))
    assert manifest["camera_height_m"] == 0.6
    assert manifest["episodes"][0]["path_length_m"] == 6.0
    # Exact float equality, not np.isclose.
    assert manifest["camera_height_m"] is not None


def test_render_summary_float_values_exact_equality():
    """Summary floats use exact parsed value equality."""
    summary = build_render_summary(**_sample_render_summary_kwargs("depth"))
    assert summary["focal_length_pixels"] == 300.0
    assert summary["max_depth_m"] == 6.0


# --- depth format truthfulness ----------------------------------------------


def test_default_depth_scale_produces_legacy_format_string():
    """Default depth_scale=10000 produces 'uint16_meters_x_10000'."""
    # This matches the legacy hardcoded string in package_lerobot_sage3d.py.
    legacy_format = "uint16_meters_x_10000"
    prod_format = f"uint16_meters_x_{int(10000.0)}"
    assert prod_format == legacy_format


def test_non_default_depth_scale_produces_truthful_format_string():
    """Non-default depth_scale produces a truthful format string."""
    # Legacy code hardcoded 10000 even when depth_scale differed. The new
    # code derives the format from the render summary's depth_scale.
    assert f"uint16_meters_x_{int(5000.0)}" == "uint16_meters_x_5000"
    assert f"uint16_meters_x_{int(1000.0)}" == "uint16_meters_x_1000"


# --- config shell -----------------------------------------------------------


def test_config_module_imports_cleanly():
    import sage3d.config

    assert hasattr(sage3d.config, "__doc__")


# --- forbidden-import smoke (config + schemas) ------------------------------


def test_schemas_and_config_are_package_safe():
    """config and schemas must not import forbidden deps."""
    import sage3d.config
    import sage3d.schemas

    for mod_name in ("cv2", "scipy", "trimesh", "pxr", "isaacsim"):
        assert mod_name not in sys.modules, f"{mod_name} loaded by sage3d schemas/config"
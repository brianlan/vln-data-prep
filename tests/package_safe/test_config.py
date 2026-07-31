"""Config validation tests for sage3d.config (Phase 3b, issue #15).

Boundary matrix: every invalid config must raise ValueError with a clear
message; valid defaults must construct without error. Package-safe:
stdlib + pathlib + dataclasses only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sage3d.config import (  # noqa: E402
    GenerationConfig,
    PathConfig,
    SceneConfig,
    SafetyConfig,
)


# --- SceneConfig -------------------------------------------------------------


def test_scene_config_valid():
    sc = SceneConfig(scene_id="839920", sage_root=Path("/data"))
    assert sc.scene_id == "839920"


def test_scene_config_empty_scene_id():
    with pytest.raises(ValueError, match="scene_id"):
        SceneConfig(scene_id="", sage_root=Path("/data"))


# --- SafetyConfig ------------------------------------------------------------


def test_safety_config_valid_defaults():
    sc = SafetyConfig(
        robot_radius=0.25,
        safety_margin=0.05,
        camera_height=0.6,
        camera_clearance=0.25,
        endpoint_extra_clearance=0.10,
    )
    assert sc.robot_radius == 0.25


def test_safety_config_zero_robot_radius():
    with pytest.raises(ValueError, match="robot_radius"):
        SafetyConfig(
            robot_radius=0,
            safety_margin=0.05,
            camera_height=0.6,
            camera_clearance=0.25,
            endpoint_extra_clearance=0.10,
        )


def test_safety_config_negative_safety_margin():
    with pytest.raises(ValueError, match="safety_margin"):
        SafetyConfig(
            robot_radius=0.25,
            safety_margin=-0.01,
            camera_height=0.6,
            camera_clearance=0.25,
            endpoint_extra_clearance=0.10,
        )


def test_safety_config_zero_camera_height():
    with pytest.raises(ValueError, match="camera_height"):
        SafetyConfig(
            robot_radius=0.25,
            safety_margin=0.05,
            camera_height=0,
            camera_clearance=0.25,
            endpoint_extra_clearance=0.10,
        )


def test_safety_config_zero_camera_clearance():
    with pytest.raises(ValueError, match="camera_clearance"):
        SafetyConfig(
            robot_radius=0.25,
            safety_margin=0.05,
            camera_height=0.6,
            camera_clearance=0,
            endpoint_extra_clearance=0.10,
        )


def test_safety_config_negative_endpoint_extra_clearance():
    with pytest.raises(ValueError, match="endpoint_extra_clearance"):
        SafetyConfig(
            robot_radius=0.25,
            safety_margin=0.05,
            camera_height=0.6,
            camera_clearance=0.25,
            endpoint_extra_clearance=-0.01,
        )


# --- PathConfig ---------------------------------------------------------------


def test_path_config_valid_defaults():
    pc = PathConfig(
        min_path_length=3.0,
        max_path_length=15.0,
        frame_spacing=0.05,
        max_attempts=3000,
    )
    assert pc.min_path_length == 3.0


def test_path_config_zero_min_path_length():
    with pytest.raises(ValueError, match="min_path_length"):
        PathConfig(
            min_path_length=0,
            max_path_length=15.0,
            frame_spacing=0.05,
            max_attempts=3000,
        )


def test_path_config_zero_max_path_length():
    with pytest.raises(ValueError, match="max_path_length"):
        PathConfig(
            min_path_length=3.0,
            max_path_length=0,
            frame_spacing=0.05,
            max_attempts=3000,
        )


def test_path_config_min_exceeds_max():
    with pytest.raises(ValueError, match="min_path_length must not exceed"):
        PathConfig(
            min_path_length=20.0,
            max_path_length=10.0,
            frame_spacing=0.05,
            max_attempts=3000,
        )


def test_path_config_zero_frame_spacing():
    with pytest.raises(ValueError, match="frame_spacing"):
        PathConfig(
            min_path_length=3.0,
            max_path_length=15.0,
            frame_spacing=0,
            max_attempts=3000,
        )


def test_path_config_zero_max_attempts():
    with pytest.raises(ValueError, match="max_attempts"):
        PathConfig(
            min_path_length=3.0,
            max_path_length=15.0,
            frame_spacing=0.05,
            max_attempts=0,
        )


# --- GenerationConfig --------------------------------------------------------


def _valid_sub_configs():
    return dict(
        scene=SceneConfig(scene_id="839920", sage_root=Path("/data")),
        safety=SafetyConfig(
            robot_radius=0.25,
            safety_margin=0.05,
            camera_height=0.6,
            camera_clearance=0.25,
            endpoint_extra_clearance=0.10,
        ),
        path=PathConfig(
            min_path_length=3.0,
            max_path_length=15.0,
            frame_spacing=0.05,
            max_attempts=3000,
        ),
    )


def test_generation_config_valid():
    gc = GenerationConfig(
        episodes=5,
        seed=20260720,
        pointcloud_voxel_size=0.05,
        pointcloud_max_points=100_000,
        **_valid_sub_configs(),
    )
    assert gc.episodes == 5


def test_generation_config_zero_episodes():
    with pytest.raises(ValueError, match="episodes"):
        GenerationConfig(
            episodes=0,
            seed=20260720,
            pointcloud_voxel_size=0.05,
            pointcloud_max_points=100_000,
            **_valid_sub_configs(),
        )


def test_generation_config_zero_voxel_size():
    with pytest.raises(ValueError, match="pointcloud_voxel_size"):
        GenerationConfig(
            episodes=5,
            seed=20260720,
            pointcloud_voxel_size=0,
            pointcloud_max_points=100_000,
            **_valid_sub_configs(),
        )


def test_generation_config_zero_max_points():
    with pytest.raises(ValueError, match="pointcloud_max_points"):
        GenerationConfig(
            episodes=5,
            seed=20260720,
            pointcloud_voxel_size=0.05,
            pointcloud_max_points=0,
            **_valid_sub_configs(),
        )


# --- frozen dataclass ---------------------------------------------------------


def test_configs_are_frozen():
    sc = SceneConfig(scene_id="839920", sage_root=Path("/data"))
    with pytest.raises((AttributeError, Exception)):
        sc.scene_id = "other"  # type: ignore[misc]
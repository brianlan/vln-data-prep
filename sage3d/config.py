"""Package-safe generation configuration (stdlib + pathlib only).

Phase 3b adds SceneConfig/SafetyConfig/PathConfig/GenerationConfig with
range validation that preserves the legacy defaults. The config objects are
constructed after CLI parsing so invalid values fail clearly before any
SAGE3D asset is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneConfig:
    """Scene asset-resolution configuration."""

    scene_id: str
    sage_root: Path
    interiorgs_root: Path | None = None
    collision_usd: Path | None = None

    def __post_init__(self) -> None:
        if not self.scene_id:
            raise ValueError("scene_id must be a non-empty string")


@dataclass(frozen=True)
class SafetyConfig:
    """Robot safety and clearance configuration."""

    robot_radius: float
    safety_margin: float
    camera_height: float
    camera_clearance: float
    endpoint_extra_clearance: float

    def __post_init__(self) -> None:
        if self.robot_radius <= 0:
            raise ValueError("robot_radius must be positive")
        if self.safety_margin < 0:
            raise ValueError("safety_margin must be non-negative")
        if self.camera_height <= 0:
            raise ValueError("camera_height must be positive")
        if self.camera_clearance <= 0:
            raise ValueError("camera_clearance must be positive")
        if self.endpoint_extra_clearance < 0:
            raise ValueError("endpoint_extra_clearance must be non-negative")


@dataclass(frozen=True)
class PathConfig:
    """Path planning and sampling configuration."""

    min_path_length: float
    max_path_length: float
    frame_spacing: float
    max_attempts: int

    def __post_init__(self) -> None:
        if self.min_path_length <= 0:
            raise ValueError("min_path_length must be positive")
        if self.max_path_length <= 0:
            raise ValueError("max_path_length must be positive")
        if self.min_path_length > self.max_path_length:
            raise ValueError("min_path_length must not exceed max_path_length")
        if self.frame_spacing <= 0:
            raise ValueError("frame_spacing must be positive")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")


@dataclass(frozen=True)
class GenerationConfig:
    """Top-level generation configuration combining all sub-configs."""

    episodes: int
    seed: int
    pointcloud_voxel_size: float
    pointcloud_max_points: int
    scene: SceneConfig
    safety: SafetyConfig
    path: PathConfig

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if self.pointcloud_voxel_size <= 0:
            raise ValueError("pointcloud_voxel_size must be positive")
        if self.pointcloud_max_points <= 0:
            raise ValueError("pointcloud_max_points must be positive")
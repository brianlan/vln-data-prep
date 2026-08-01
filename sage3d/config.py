"""Package-safe generation configuration (stdlib + pathlib only).

Phase 3b adds SceneConfig/SafetyConfig/PathConfig/GenerationConfig with
range validation that preserves the legacy defaults. The config objects are
constructed after CLI parsing so invalid values fail clearly before any
SAGE3D asset is loaded.

Phase 4 adds RenderConfig (render width/height, depth range, startup/settle
steps, fisheye coefficients, render mode) with stdlib-only basic range
validation. ``RenderConfig.mode`` is a plain ``str``/``Literal`` — it is
package-safe and does not import ``RenderMode`` from ``render_runtime.py``.

Phase 5a adds PackageConfig (package dataset construction) with positive-FPS,
path/output, and optional legacy compatibility-assertion validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


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


@dataclass(frozen=True)
class RenderConfig:
    """Render configuration (package-safe, stdlib-only validation).

    ``mode`` is a plain string (``"rgb"`` or ``"depth"``), not a
    ``RenderMode`` enum imported from Isaac code.
    """

    mode: Literal["rgb", "depth"]
    width: int
    height: int
    horizontal_fov_deg: float
    fisheye_coefficients: tuple[float, float, float, float]
    max_depth_m: float
    min_depth_m: float
    depth_scale: float
    settle_steps: int
    startup_steps: int

    def __post_init__(self) -> None:
        if self.mode not in ("rgb", "depth"):
            raise ValueError(f"mode must be 'rgb' or 'depth', got {self.mode!r}")
        if self.width <= 0:
            raise ValueError("width must be positive")
        if self.height <= 0:
            raise ValueError("height must be positive")
        if self.horizontal_fov_deg <= 0:
            raise ValueError("horizontal_fov_deg must be positive")
        if len(self.fisheye_coefficients) != 4:
            raise ValueError("fisheye_coefficients must have 4 elements")
        if not all(math.isfinite(c) for c in self.fisheye_coefficients):
            raise ValueError("fisheye_coefficients must all be finite")
        if not (math.isfinite(self.max_depth_m) and self.max_depth_m > 0):
            raise ValueError("max_depth_m must be finite positive")
        if not (math.isfinite(self.min_depth_m) and self.min_depth_m >= 0):
            raise ValueError("min_depth_m must be finite non-negative")
        if not (self.min_depth_m < self.max_depth_m):
            raise ValueError("min_depth_m must be < max_depth_m")
        if not (math.isfinite(self.depth_scale) and self.depth_scale > 0):
            raise ValueError("depth_scale must be finite positive")
        if self.settle_steps < 0:
            raise ValueError("settle_steps must be non-negative")
        if self.startup_steps < 0:
            raise ValueError("startup_steps must be non-negative")


@dataclass(frozen=True)
class PackageConfig:
    """Package dataset construction configuration (package-safe).

    Phase 5a adds the config consumed by the pure package builders in
    ``sage3d.lerobot_dataset``. FPS must be positive; the trajectory/render
    source dirs and the output dir are required paths; the optional width /
    height / FOV / coefficients / camera-height fields mirror the legacy CLI
    camera assertions and are validated as finite positive values when given.
    """

    fps: int
    trajectory_dir: Path
    rendered_dir: Path
    output_dir: Path
    scene_id: str
    width: int | None = None
    height: int | None = None
    horizontal_fov_deg: float | None = None
    fisheye_coefficients: tuple[float, float, float, float] | None = None
    camera_height: float | None = None

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if not self.scene_id:
            raise ValueError("scene_id must be a non-empty string")
        if self.width is not None and self.width <= 0:
            raise ValueError("width must be positive when provided")
        if self.height is not None and self.height <= 0:
            raise ValueError("height must be positive when provided")
        if self.horizontal_fov_deg is not None and not (
            math.isfinite(self.horizontal_fov_deg)
            and self.horizontal_fov_deg > 0
        ):
            raise ValueError(
                "horizontal_fov_deg must be finite and positive when provided"
            )
        if self.fisheye_coefficients is not None and (
            len(self.fisheye_coefficients) != 4
            or not all(math.isfinite(c) for c in self.fisheye_coefficients)
        ):
            raise ValueError(
                "fisheye_coefficients must be four finite values when provided"
            )
        if self.camera_height is not None and not math.isfinite(self.camera_height):
            raise ValueError("camera_height must be finite when provided")
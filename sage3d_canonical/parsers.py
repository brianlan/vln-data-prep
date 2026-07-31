"""Artifact parsers for the SAGE3D canonical tooling.

These parsers read the SAGE3D artifact contract (trajectory manifests,
NPZ episodes, binary PLY, render summaries, packaged LeRobot-style datasets)
using only package-safe dependencies (``numpy``, ``pyarrow``, ``PIL``, stdlib).
They deliberately import **no** target production module (``sage3d.*``); they
exist to characterize the legacy contract before the Phase 0b baseline capture.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def parse_trajectory_manifest(trajectory_dir: Path) -> dict:
    """Load and validate a legacy ``trajectory_manifest.json``."""
    path = trajectory_dir / "trajectory_manifest.json"
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    required = (
        "scene_id",
        "scene_dir",
        "collision_usd",
        "seed",
        "episode_count",
        "robot_radius_m",
        "safety_margin_m",
        "camera_height_m",
        "frame_spacing_m",
        "pointcloud",
        "episodes",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"trajectory_manifest missing keys: {missing}")
    if manifest["episode_count"] != len(manifest["episodes"]):
        raise ValueError(
            "trajectory_manifest episode_count does not match episodes length"
        )
    return manifest


def parse_episode_npz(trajectory_dir: Path, episode_index: int) -> dict:
    """Load a legacy ``episode_*.npz`` and validate its key set/shapes."""
    path = trajectory_dir / f"episode_{episode_index:06d}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path)
    keys = set(data.files)
    expected = {
        "points",
        "actions",
        "camera_positions",
        "yaw",
        "point_goal",
        "start_position",
        "goal_position",
    }
    if keys != expected:
        raise ValueError(f"episode {episode_index} npz keys {keys} != {expected}")
    arrays = {key: data[key] for key in expected}
    frame_count = arrays["actions"].shape[0]
    for key in ("points", "camera_positions", "yaw", "point_goal"):
        if arrays[key].shape[0] != frame_count:
            raise ValueError(
                f"episode {episode_index} {key} length {arrays[key].shape[0]} "
                f"!= actions length {frame_count}"
            )
    if arrays["actions"].shape[1:] != (4, 4):
        raise ValueError(
            f"episode {episode_index} actions shape {arrays['actions'].shape}"
        )
    if arrays["point_goal"].shape[1] != 2:
        raise ValueError(
            f"episode {episode_index} point_goal shape {arrays['point_goal'].shape}"
        )
    return arrays


def parse_binary_ply(path: Path) -> dict:
    """Parse the legacy binary-little-endian PLY written by the generator."""
    with path.open("rb") as f:
        header_bytes = b""
        while True:
            line = f.readline()
            header_bytes += line
            if line.strip() == b"end_header":
                break
        header = header_bytes.decode("ascii")
        if "binary_little_endian" not in header:
            raise ValueError(f"unexpected PLY format in {path}")
        vertex_count = _ply_vertex_count(header)
        record = struct.Struct("<fffBBB")
        points = np.empty((vertex_count, 3), dtype=np.float32)
        colors = np.empty((vertex_count, 3), dtype=np.uint8)
        for i in range(vertex_count):
            x, y, z, r, g, b = record.unpack(f.read(record.size))
            points[i] = (x, y, z)
            colors[i] = (r, g, b)
    return {"points": points, "colors": colors, "vertex_count": vertex_count}


def _ply_vertex_count(header: str) -> int:
    for line in header.splitlines():
        if line.startswith("element vertex"):
            return int(line.split()[-1])
    raise ValueError("PLY header has no element vertex count")


def parse_render_summary(rendered_dir: Path, name: str = "render_summary.json") -> dict:
    """Load and validate a legacy render summary JSON."""
    path = rendered_dir / name
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    required = (
        "scene_id",
        "camera_model",
        "resolution",
        "focal_length_pixels",
        "fisheye_coefficients",
        "forward_mask_radius_pixels",
        "render_mode",
        "total_frames",
        "max_depth_m",
        "min_depth_m",
        "depth_scale",
    )
    missing = [key for key in required if key not in summary]
    if missing:
        raise ValueError(f"{name} missing keys: {missing}")
    return summary


def parse_packaged_dataset(dataset_dir: Path) -> dict:
    """Read a legacy packaged LeRobot-style dataset tree structure."""
    info_path = dataset_dir / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    data_dir = dataset_dir / "data" / "chunk-000"
    parquet_files = sorted(data_dir.glob("episode_*.parquet"))
    rgb_dir = dataset_dir / "videos" / "chunk-000" / "observation.images.rgb"
    depth_dir = dataset_dir / "videos" / "chunk-000" / "observation.images.depth"
    rgb_files = sorted(rgb_dir.glob("*.jpg"))
    depth_files = sorted(depth_dir.glob("*.png"))
    meta_files = {p.name for p in (dataset_dir / "meta").iterdir()}
    return {
        "info": info,
        "parquet_files": parquet_files,
        "rgb_files": rgb_files,
        "depth_files": depth_files,
        "meta_files": meta_files,
    }
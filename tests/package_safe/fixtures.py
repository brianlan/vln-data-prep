"""Synthetic SAGE3D artifact fixtures for Phase 0a characterization.

These builders create minimal on-disk trajectory/render/package trees that
match the *legacy* artifact contract written by the current monolithic scripts,
without invoking any target production module. They let the Phase 0a artifact
parsers and package-success fixture operate on deterministic synthetic data.

The shapes and field names mirror the current output of:

- ``generate_sage3d_trajectories.py`` (NPZ + PLY + manifest + viz PNGs)
- ``render_fisheye_sage3d.py`` (RGB JPEGs, depth PNGs, render summaries)
- ``package_lerobot_sage3d.py`` (LeRobot-style parquet/meta/video tree)

Only the package-safe dependencies (``numpy``, ``PIL``, ``pyarrow``) are used.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

# fisheye_camera.py is a package-safe shared helper (stdlib math only), not a
# target production module, so it is safe to import here.
from fisheye_camera import opencv_fisheye_parameters

NPZ_KEYS = (
    "points",
    "actions",
    "camera_positions",
    "yaw",
    "point_goal",
    "start_position",
    "goal_position",
)
DEFAULT_WIDTH = 600
DEFAULT_HEIGHT = 450
DEFAULT_FOV_DEG = 180.0
DEFAULT_COEFFICIENTS = (0.1, 0.0, 0.0, 0.0)


def _episode_arrays(frame_count: int, camera_height: float = 0.6):
    """Build minimal, internally-consistent episode arrays for ``frame_count``."""
    rng = np.random.default_rng(seed=hash((frame_count, camera_height)) & 0xFFFF)
    points = np.cumsum(
        rng.standard_normal((frame_count, 2)).astype(np.float32) * 0.1,
        axis=0,
    )
    delta = np.gradient(points, axis=0)
    yaw = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0])).astype(np.float32)
    actions = np.repeat(
        np.eye(4, dtype=np.float32)[None, ...], frame_count, axis=0
    )
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    actions[:, 0, 0] = cos_yaw
    actions[:, 0, 1] = -sin_yaw
    actions[:, 1, 0] = sin_yaw
    actions[:, 1, 1] = cos_yaw
    actions[:, 0, 3] = points[:, 0]
    actions[:, 1, 3] = points[:, 1]
    camera_positions = np.column_stack(
        (points[:, 0], points[:, 1], np.full(frame_count, camera_height))
    ).astype(np.float32)
    goal = points[-1]
    goal_delta = goal[None, :] - points
    goal_distance = np.linalg.norm(goal_delta, axis=1).astype(np.float32)
    goal_bearing = (np.arctan2(goal_delta[:, 1], goal_delta[:, 0]) - yaw).astype(
        np.float32
    )
    point_goal = np.column_stack((goal_distance, goal_bearing)).astype(np.float32)
    start_pos_3d = np.asarray(
        [points[0, 0], points[0, 1], 0.0], dtype=np.float32
    )
    goal_pos_3d = np.asarray(
        [points[-1, 0], points[-1, 1], 0.0], dtype=np.float32
    )
    return {
        "points": points,
        "actions": actions,
        "camera_positions": camera_positions,
        "yaw": yaw,
        "point_goal": point_goal,
        "start_position": start_pos_3d,
        "goal_position": goal_pos_3d,
    }


def write_binary_pointcloud(path: Path, points: np.ndarray) -> None:
    """Write a binary-little-endian PLY matching the legacy writer format."""
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


def build_trajectory_dir(
    output_dir: Path,
    *,
    episode_frame_counts: tuple[int, ...],
    scene_id: str = "839920",
    seed: int = 20260720,
    camera_height: float = 0.6,
    robot_radius: float = 0.2,
    safety_margin: float = 0.1,
    frame_spacing: float = 0.1,
) -> dict:
    """Write a synthetic trajectory artifact tree matching the legacy contract."""
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_meta = []
    for index, frame_count in enumerate(episode_frame_counts):
        arrays = _episode_arrays(frame_count, camera_height)
        np.savez_compressed(output_dir / f"episode_{index:06d}.npz", **arrays)
        episodes_meta.append(
            {
                "episode_index": index,
                "component_id": 1,
                "start_pixel": [0, 0],
                "goal_pixel": [1, 1],
                "start_position": arrays["start_position"].tolist(),
                "goal_position": arrays["goal_position"].tolist(),
                "raw_path_length_m": float(frame_count * frame_spacing),
                "path_length_m": float(frame_count * frame_spacing),
                "frame_count": frame_count,
                "minimum_clearance_m": 0.05,
                "minimum_camera_clearance_m": 0.5,
                "smoothing_method": "simplify_by_visibility",
            }
        )
    pointcloud = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    write_binary_pointcloud(output_dir / "pointcloud.ply", pointcloud)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(output_dir / "navigation_map.png")
    Image.new("RGB", (8, 8), (40, 50, 60)).save(
        output_dir / "trajectories_overlay.png"
    )
    manifest = {
        "scene_id": scene_id,
        "scene_dir": f"/synthetic/scene/{scene_id}",
        "collision_usd": f"/synthetic/collision/{scene_id}.usd",
        "seed": seed,
        "episode_count": len(episode_frame_counts),
        "robot_radius_m": robot_radius,
        "safety_margin_m": safety_margin,
        "camera_height_m": camera_height,
        "camera_clearance_m": robot_radius,
        "frame_spacing_m": frame_spacing,
        "requested_path_length_range_m": [1.0, 10.0],
        "endpoint_clearance_m": 0.2,
        "map": {"shape": [8, 8], "resolution_m_per_px": 0.1},
        "generation": {"attempts": len(episode_frame_counts), "rejection_counts": {}},
        "pointcloud": {
            "source_vertex_count": len(pointcloud),
            "output_point_count": len(pointcloud),
            "voxel_size_m": 0.05,
            "bounds_min": pointcloud.min(axis=0).astype(float).tolist(),
            "bounds_max": pointcloud.max(axis=0).astype(float).tolist(),
            "color": [160, 160, 160],
        },
        "episodes": episodes_meta,
    }
    with (output_dir / "trajectory_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _render_summary(
    *,
    mode: str,
    scene_id: str,
    width: int,
    height: int,
    fov_deg: float,
    coefficients: tuple[float, ...],
    episodes: list[dict],
    max_depth_m: float,
    min_depth_m: float,
    depth_scale: float,
    total_frames: int,
) -> dict:
    calibration = opencv_fisheye_parameters(width, height, fov_deg, coefficients)
    summary = {
        "scene_id": scene_id,
        "camera_model": "opencv_fisheye",
        "resolution": [width, height],
        "horizontal_fov_deg": calibration["horizontal_fov_deg"],
        "vertical_fov_deg": calibration["vertical_fov_deg"],
        "focal_length_pixels": calibration["fx"],
        "principal_point": [calibration["cx"], calibration["cy"]],
        "fisheye_coefficients": calibration["fisheye_coefficients"],
        "forward_mask_radius_pixels": calibration["forward_mask_radius_pixels"],
        "camera_pitch_deg": 0.0,
        "depth_type": "distance_to_camera",
        "max_depth_m": max_depth_m,
        "min_depth_m": min_depth_m,
        "depth_scale": depth_scale,
        "render_mode": mode,
        "episodes": episodes,
        "total_frames": total_frames,
    }
    return summary


def build_rendered_dir(
    output_dir: Path,
    *,
    trajectory_manifest: dict,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fov_deg: float = DEFAULT_FOV_DEG,
    coefficients: tuple[float, ...] = DEFAULT_COEFFICIENTS,
    max_depth_m: float = 6.0,
    min_depth_m: float = 0.05,
    depth_scale: float = 10000.0,
    depth_fill: str = "gradient",
) -> dict:
    """Write a synthetic rendered artifact tree matching the legacy contract.

    ``depth_fill`` controls inside-mask depth values:

    - ``"gradient"`` (default): linear gradient from ``min_depth_m`` to
      ``max_depth_m * 0.5`` across the mask region; outside-mask is sentinel.
    - ``"sentinel"``: all pixels are sentinel (for all-sentinel mutation tests).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = output_dir / "observation.images.rgb"
    depth_dir = output_dir / "observation.images.depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    calibration = opencv_fisheye_parameters(width, height, fov_deg, coefficients)
    yy, xx = np.ogrid[:height, :width]
    circular_mask = (
        (xx - calibration["cx"]) ** 2 + (yy - calibration["cy"]) ** 2
        <= calibration["forward_mask_radius_pixels"] ** 2
    )
    sentinel_val = int(np.rint(np.float32(max_depth_m) * np.float32(depth_scale)))
    total_frames = 0
    depth_episodes = []
    for episode in trajectory_manifest["episodes"]:
        frame_count = episode["frame_count"]
        for frame_index in range(frame_count):
            stem = f"episode_{episode['episode_index']:06d}_{frame_index:03d}"
            # RGB with per-pixel variation so mutations like flips are detectable.
            rng = np.random.default_rng(episode["episode_index"] * 1000 + frame_index)
            arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            Image.fromarray(arr).save(rgb_dir / f"{stem}.jpg", quality=95)
            depth = np.full((height, width), sentinel_val, dtype=np.uint16)
            if depth_fill == "gradient" and circular_mask.any():
                inside = np.where(circular_mask)
                # Gradient from min_depth to max_depth*0.5 across the mask.
                lo = int(np.rint(np.float32(min_depth_m) * np.float32(depth_scale)))
                hi = int(np.rint(np.float32(max_depth_m * 0.5) * np.float32(depth_scale)))
                n = len(inside[0])
                vals = np.linspace(lo, hi, n, dtype=np.float64)
                np.random.seed(episode["episode_index"] * 100 + frame_index)
                np.random.shuffle(vals)
                depth[inside] = vals.astype(np.uint16)
            Image.fromarray(depth).save(depth_dir / f"{stem}.png")
        total_frames += frame_count
        depth_episodes.append(
            {
                "episode_index": episode["episode_index"],
                "frame_count": frame_count,
                "finite_depth_fraction_mean": 0.9,
                "finite_depth_fraction_min": 0.8,
                "finite_depth_min_m": min_depth_m,
                "finite_depth_max_m": max_depth_m * 0.5,
            }
        )
    rgb_summary = _render_summary(
        mode="rgb",
        scene_id=trajectory_manifest["scene_id"],
        width=width,
        height=height,
        fov_deg=fov_deg,
        coefficients=coefficients,
        episodes=[],
        max_depth_m=max_depth_m,
        min_depth_m=min_depth_m,
        depth_scale=depth_scale,
        total_frames=total_frames,
    )
    depth_summary = _render_summary(
        mode="depth",
        scene_id=trajectory_manifest["scene_id"],
        width=width,
        height=height,
        fov_deg=fov_deg,
        coefficients=coefficients,
        episodes=depth_episodes,
        max_depth_m=max_depth_m,
        min_depth_m=min_depth_m,
        depth_scale=depth_scale,
        total_frames=total_frames,
    )
    for name, payload in (
        ("rgb_render_summary.json", rgb_summary),
        ("depth_render_summary.json", depth_summary),
        ("render_summary.json", depth_summary),
    ):
        with (output_dir / name).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return {"rgb": rgb_summary, "depth": depth_summary}


def build_packaged_dataset(
    output_dir: Path,
    *,
    trajectory_dir: Path,
    rendered_dir: Path,
    scene_id: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fov_deg: float = DEFAULT_FOV_DEG,
    coefficients: tuple[float, ...] = DEFAULT_COEFFICIENTS,
    camera_height: float = 0.6,
    fps: int = 30,
) -> dict:
    """Write a synthetic LeRobot-style packaged dataset tree."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = trajectory_dir / "trajectory_manifest.json"
    render_summary_path = rendered_dir / "render_summary.json"
    rgb_summary_path = rendered_dir / "rgb_render_summary.json"
    depth_summary_path = rendered_dir / "depth_render_summary.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    with render_summary_path.open("r", encoding="utf-8") as f:
        render_summary = json.load(f)
    with rgb_summary_path.open("r", encoding="utf-8") as f:
        rgb_summary = json.load(f)

    calibration = opencv_fisheye_parameters(width, height, fov_deg, coefficients)
    intrinsic = np.asarray(
        [
            [calibration["fx"], 0.0, calibration["cx"]],
            [0.0, calibration["fy"], calibration["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[2, 3] = camera_height
    intrinsic_flat = intrinsic.reshape(-1).tolist()
    extrinsic_flat = extrinsic.reshape(-1).tolist()
    distortion = calibration["fisheye_coefficients"]

    data_dir = output_dir / "data" / "chunk-000"
    meta_dir = output_dir / "meta"
    rgb_output_dir = output_dir / "videos" / "chunk-000" / "observation.images.rgb"
    depth_output_dir = output_dir / "videos" / "chunk-000" / "observation.images.depth"
    for directory in (data_dir, meta_dir, rgb_output_dir, depth_output_dir):
        directory.mkdir(parents=True, exist_ok=True)

    import shutil

    episode_records = []
    total_frames = 0
    for episode in manifest["episodes"]:
        trajectory = np.load(trajectory_dir / f"episode_{episode['episode_index']:06d}.npz")
        actions = trajectory["actions"].astype(np.float32)
        point_goal = trajectory["point_goal"].astype(np.float32)
        frame_count = len(actions)
        table = pa.table(
            {
                "index": pa.array(range(frame_count), type=pa.int64()),
                "observation.camera_intrinsic": pa.array(
                    [intrinsic_flat] * frame_count, type=pa.list_(pa.float32())
                ),
                "observation.camera_extrinsic": pa.array(
                    [extrinsic_flat] * frame_count, type=pa.list_(pa.float32())
                ),
                "observation.camera_distortion": pa.array(
                    [distortion] * frame_count, type=pa.list_(pa.float32())
                ),
                "observation.point_goal": pa.array(
                    point_goal.tolist(), type=pa.list_(pa.float32())
                ),
                "action": pa.array(
                    actions.reshape(frame_count, 16).tolist(),
                    type=pa.list_(pa.float32()),
                ),
            }
        )
        pq.write_table(table, data_dir / f"episode_{episode['episode_index']:06d}.parquet")
        for frame_index in range(frame_count):
            stem = f"episode_{episode['episode_index']:06d}_{frame_index:03d}"
            shutil.copy2(
                rendered_dir / "observation.images.rgb" / f"{stem}.jpg",
                rgb_output_dir / f"{stem}.jpg",
            )
            shutil.copy2(
                rendered_dir / "observation.images.depth" / f"{stem}.png",
                depth_output_dir / f"{stem}.png",
            )
        episode_records.append(
            {
                "episode_index": episode["episode_index"],
                "task_index": 0,
                "task_type": "point_goal_navigation",
                "coordinate_frame": "world_z_up_x_forward",
                "point_goal_representation": ["distance_m", "relative_bearing_rad"],
                "start_position": episode["start_position"],
                "goal_position": episode["goal_position"],
                "path_length_m": episode["path_length_m"],
                "minimum_clearance_m": episode["minimum_clearance_m"],
                "frame_count": frame_count,
                "frame_indexes": [0, frame_count - 1],
                "seed": manifest["seed"],
            }
        )
        total_frames += frame_count

    shutil.copy2(trajectory_dir / "pointcloud.ply", meta_dir / "pointcloud.ply")
    shutil.copy2(manifest_path, meta_dir / "trajectory_manifest.json")
    shutil.copy2(render_summary_path, meta_dir / "render_summary.json")
    shutil.copy2(rgb_summary_path, meta_dir / rgb_summary_path.name)
    shutil.copy2(depth_summary_path, meta_dir / depth_summary_path.name)

    info = {
        "codebase_version": "v2.1",
        "robot_type": "sage3d_pointgoal_fisheye",
        "scene_id": scene_id,
        "total_episodes": len(manifest["episodes"]),
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": len(manifest["episodes"]),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.camera_intrinsic": {"dtype": "float32", "shape": [3, 3]},
            "observation.camera_extrinsic": {"dtype": "float32", "shape": [4, 4]},
            "observation.camera_distortion": {
                "dtype": "float32",
                "shape": [4],
                "names": ["k1", "k2", "k3", "k4"],
            },
            "observation.point_goal": {
                "dtype": "float32",
                "shape": [2],
                "names": ["distance_m", "relative_bearing_rad"],
            },
            "action": {"dtype": "float32", "shape": [4, 4]},
        },
        "action_semantics": "robot-base-to-world pose; +X forward, +Z up, translation z=0",
        "camera_extrinsic_semantics": "camera-to-robot-base pose; identity rotation and +Z camera height",
        "camera_height_m": camera_height,
        "camera_model": "opencv_fisheye",
        "camera_fov_deg": fov_deg,
        "camera_horizontal_fov_deg": fov_deg,
        "camera_vertical_fov_deg": calibration["vertical_fov_deg"],
        "camera_fisheye_coefficients": distortion,
        "camera_pitch_deg": 0.0,
        "camera_forward_mask_radius_pixels": calibration["forward_mask_radius_pixels"],
        "image_width": width,
        "image_height": height,
        "depth_type": "distance_to_camera",
        "depth_format": "uint16_meters_x_10000",
        "depth_clip_m": render_summary["max_depth_m"],
        "depth_min_m": render_summary["min_depth_m"],
        "trajectory_seed": manifest["seed"],
        "robot_radius_m": manifest["robot_radius_m"],
        "frame_spacing_m": manifest["frame_spacing_m"],
    }
    with (meta_dir / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    _write_jsonl(meta_dir / "episodes.jsonl", episode_records)
    _write_jsonl(
        meta_dir / "tasks.jsonl",
        [
            {
                "task_index": 0,
                "task": {
                    "type": "point_goal_navigation",
                    "goal_input": ["distance_m", "relative_bearing_rad"],
                },
            }
        ],
    )
    return info


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
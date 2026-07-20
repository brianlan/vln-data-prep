#!/usr/bin/env python3
"""Render SAGE3D PointGoal trajectories with native Isaac Sim fisheye RGB/depth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--usdz", type=Path, required=True)
    parser.add_argument("--collision-usd", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("rgb", "depth"),
        required=True,
        help=(
            "Render exactly one modality. Invoke the script twice so NuRec "
            "appearance and collision depth use independent fresh stages."
        ),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--fov-deg", type=float, default=195.0)
    parser.add_argument("--max-depth-m", type=float, default=6.0)
    parser.add_argument("--min-depth-m", type=float, default=0.05)
    parser.add_argument("--depth-scale", type=float, default=10000.0)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=10,
        help="Render updates after each pose; 10 avoids one-pose annotator latency",
    )
    parser.add_argument("--startup-steps", type=int, default=40)
    return parser.parse_args()


ARGS = parse_args()

from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": True,
        "renderer": "RaytracedLighting",
        "width": ARGS.width,
        "height": ARGS.height,
    }
)

import numpy as np
from PIL import Image
import omni.usd
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
from pxr import UsdGeom


def render_steps(world: World, count: int) -> None:
    for _ in range(count):
        world.step(render=True)


def camera_quaternion(yaw: float) -> np.ndarray:
    # Isaac's "world" camera axes are +X forward and +Z up. Quaternion order
    # for Camera.set_world_pose is scalar-first: [w, x, y, z].
    return np.asarray(
        [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
        dtype=np.float32,
    )


def validate_inputs() -> list[Path]:
    for path in (ARGS.usdz, ARGS.collision_usd, ARGS.trajectory_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    trajectory_files = sorted(ARGS.trajectory_dir.glob("episode_*.npz"))
    if not trajectory_files:
        raise RuntimeError(
            f"No episode_*.npz files found in {ARGS.trajectory_dir}"
        )
    return trajectory_files


def main() -> None:
    trajectory_files = validate_inputs()
    rgb_dir = ARGS.output_dir / "observation.images.rgb"
    depth_dir = ARGS.output_dir / "observation.images.depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world_prim)

    if ARGS.mode == "rgb":
        gauss = stage.OverridePrim("/World/gauss")
        gauss.GetReferences().AddReference(f"{ARGS.usdz}[gauss.usda]")
    else:
        collision = UsdGeom.Xform.Define(
            stage, "/World/scene_collision"
        ).GetPrim()
        collision.GetPayloads().AddPayload(str(ARGS.collision_usd))

    world = World(stage_units_in_meters=1.0)
    world.reset()
    render_steps(world, ARGS.startup_steps)

    camera = Camera(
        prim_path="/World/PointGoalFisheyeCamera",
        frequency=30,
        resolution=(ARGS.width, ARGS.height),
    )
    camera.initialize()
    camera.set_clipping_range(0.05, max(20.0, ARGS.max_depth_m * 2.0))

    focal_pixels = ARGS.width / math.radians(ARGS.fov_deg)
    camera.set_opencv_fisheye_properties(
        cx=ARGS.width / 2.0,
        cy=ARGS.height / 2.0,
        fx=focal_pixels,
        fy=focal_pixels,
        fisheye=[0.0, 0.0, 0.0, 0.0],
    )
    if ARGS.mode == "depth":
        camera.add_distance_to_camera_to_frame()
    render_steps(world, ARGS.startup_steps)

    yy, xx = np.ogrid[: ARGS.height, : ARGS.width]
    radius = min(ARGS.width, ARGS.height) / 2.0
    circular_mask = (
        (xx - ARGS.width / 2.0) ** 2 + (yy - ARGS.height / 2.0) ** 2
        <= radius**2
    )

    summary = {
        "scene_id": ARGS.scene,
        "camera_model": "opencv_fisheye_equidistant",
        "resolution": [ARGS.width, ARGS.height],
        "fov_deg": ARGS.fov_deg,
        "focal_length_pixels": focal_pixels,
        "principal_point": [ARGS.width / 2.0, ARGS.height / 2.0],
        "fisheye_coefficients": [0.0, 0.0, 0.0, 0.0],
        "depth_type": "distance_to_camera",
        "max_depth_m": ARGS.max_depth_m,
        "min_depth_m": ARGS.min_depth_m,
        "depth_scale": ARGS.depth_scale,
        "render_mode": ARGS.mode,
        "episodes": [],
    }

    trajectories = []
    for episode_index, trajectory_file in enumerate(trajectory_files):
        trajectory = np.load(trajectory_file)
        camera_positions = trajectory["camera_positions"].copy()
        yaw = trajectory["yaw"].copy()
        if len(camera_positions) != len(yaw):
            raise RuntimeError(
                f"Pose/yaw count mismatch in {trajectory_file}: "
                f"{len(camera_positions)} vs {len(yaw)}"
            )

        trajectories.append((camera_positions, yaw))

        if ARGS.mode == "rgb":
            # NuRec appearance is rendered in a dedicated process. Loading a
            # collision payload into this stage can poison subsequent NuRec
            # buffers even after USD visibility changes.
            for frame_index, (position, heading) in enumerate(
                zip(camera_positions, yaw)
            ):
                camera.set_world_pose(
                    position=position,
                    orientation=camera_quaternion(float(heading)),
                    camera_axes="world",
                )

                render_steps(
                    world,
                    (
                        ARGS.startup_steps
                        if frame_index == 0
                        else ARGS.settle_steps
                    ),
                )
                rgba = camera.get_rgba()
                if rgba is None or np.asarray(rgba).size == 0:
                    raise RuntimeError(
                        f"Empty RGB frame at episode={episode_index}, "
                        f"frame={frame_index}"
                    )
                rgb = np.asarray(rgba)[..., :3].astype(np.uint8).copy()
                rgb[~circular_mask] = 0
                inside_pixels = rgb[circular_mask]
                if float(inside_pixels.std()) < 1.0:
                    raise RuntimeError(
                        f"Near-uniform RGB frame at episode={episode_index}, "
                        f"frame={frame_index}; NuRec renderer may have failed"
                    )

                stem = f"episode_{episode_index:06d}_{frame_index:03d}"
                Image.fromarray(rgb).save(rgb_dir / f"{stem}.jpg", quality=95)
                if frame_index % 25 == 0 or frame_index == len(yaw) - 1:
                    print(
                        f"[render-rgb] episode {episode_index:06d}: "
                        f"{frame_index + 1}/{len(yaw)} frames"
                    )

    total_frames = 0
    if ARGS.mode == "depth":
        for episode_index, (camera_positions, yaw) in enumerate(trajectories):
            episode_finite_depth = []
            episode_depth_min = []
            episode_depth_max = []
            for frame_index, (position, heading) in enumerate(
                zip(camera_positions, yaw)
            ):
                camera.set_world_pose(
                    position=position,
                    orientation=camera_quaternion(float(heading)),
                    camera_axes="world",
                )
                render_steps(
                    world,
                    (
                        ARGS.startup_steps
                        if frame_index == 0
                        else ARGS.settle_steps
                    ),
                )
                frame = camera.get_current_frame(clone=True)
                depth = frame.get("distance_to_camera") if frame else None
                if depth is None:
                    raise RuntimeError(
                        f"No distance_to_camera depth at episode={episode_index}, "
                        f"frame={frame_index}; keys={list(frame) if frame else []}"
                    )
                depth = np.asarray(depth, dtype=np.float32).squeeze()
                if depth.shape != (ARGS.height, ARGS.width):
                    raise RuntimeError(
                        f"Unexpected depth shape {depth.shape}; expected "
                        f"{(ARGS.height, ARGS.width)}"
                    )
                finite = np.isfinite(depth) & (depth >= ARGS.min_depth_m)
                valid_inside = finite & circular_mask
                if not valid_inside.any():
                    raise RuntimeError(
                        f"No finite collision depth at episode={episode_index}, "
                        f"frame={frame_index}"
                    )
                finite_fraction = float(
                    valid_inside.sum() / circular_mask.sum()
                )
                episode_finite_depth.append(finite_fraction)
                episode_depth_min.append(float(depth[valid_inside].min()))
                episode_depth_max.append(float(depth[valid_inside].max()))

                depth = np.nan_to_num(
                    depth,
                    nan=ARGS.max_depth_m,
                    posinf=ARGS.max_depth_m,
                    neginf=ARGS.max_depth_m,
                )
                depth[~finite] = ARGS.max_depth_m
                depth[~circular_mask] = ARGS.max_depth_m
                depth = np.clip(depth, 0.0, ARGS.max_depth_m)
                depth_u16 = np.rint(
                    depth * ARGS.depth_scale
                ).astype(np.uint16)
                stem = f"episode_{episode_index:06d}_{frame_index:03d}"
                Image.fromarray(depth_u16).save(depth_dir / f"{stem}.png")

                total_frames += 1
                if frame_index % 25 == 0 or frame_index == len(yaw) - 1:
                    print(
                        f"[render-depth] episode {episode_index:06d}: "
                        f"{frame_index + 1}/{len(yaw)} frames"
                    )

            summary["episodes"].append(
                {
                    "episode_index": episode_index,
                    "frame_count": len(yaw),
                    "finite_depth_fraction_mean": float(
                        np.mean(episode_finite_depth)
                    ),
                    "finite_depth_fraction_min": float(
                        np.min(episode_finite_depth)
                    ),
                    "finite_depth_min_m": (
                        float(min(episode_depth_min))
                        if episode_depth_min
                        else None
                    ),
                    "finite_depth_max_m": (
                        float(max(episode_depth_max))
                        if episode_depth_max
                        else None
                    ),
                }
            )
    else:
        total_frames = sum(len(yaw) for _, yaw in trajectories)

    summary["total_frames"] = total_frames
    summary_path = ARGS.output_dir / f"{ARGS.mode}_render_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    if ARGS.mode == "depth":
        # The packager's canonical render summary describes the metric depth
        # pass; the modality-specific copy makes the two-process split explicit.
        with (ARGS.output_dir / "render_summary.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(summary, file, indent=2)
    print(
        f"[render-{ARGS.mode}] Completed {total_frames} frames: "
        f"{ARGS.output_dir}"
    )


try:
    main()
finally:
    simulation_app.close()

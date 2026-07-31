#!/usr/bin/env python3
"""Render SAGE3D PointGoal trajectories with native Isaac Sim fisheye RGB/depth."""

from __future__ import annotations

import argparse
from pathlib import Path

from sage3d.cli._args import add_scene_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_scene_args(parser)
    parser.add_argument(
        "--usdz",
        type=Path,
        default=None,
        help="Override USDZ; defaults to <sage-root>/InteriorGS_usdz/<scene>.usdz",
    )
    parser.add_argument(
        "--collision-usd",
        type=Path,
        default=None,
        help="Override collision USD; defaults to <sage-root>/Collision_Mesh/...",
    )
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
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--height", type=int, default=450)
    parser.add_argument("--horizontal-fov-deg", type=float, default=180.0)
    parser.add_argument(
        "--fisheye-coefficients",
        type=float,
        nargs=4,
        metavar=("K1", "K2", "K3", "K4"),
        default=(0.1, 0.0, 0.0, 0.0),
    )
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

from sage3d.camera import CameraCalibration
from sage3d.frames import yaw_to_quaternion
from sage3d.naming import frame_stem
from sage3d.render_processing import (
    RawDepthSummaryAccumulator,
    build_forward_mask,
    encode_depth,
    mask_rgb,
)


def render_steps(world: World, count: int) -> None:
    for _ in range(count):
        world.step(render=True)


def validate_inputs() -> list[Path]:
    from sage3d.artifacts import resolve_render_assets

    assets = resolve_render_assets(
        ARGS.scene,
        ARGS.sage_root,
        usdz=ARGS.usdz,
        collision_usd=ARGS.collision_usd,
    )
    # Store resolved paths for use in main().
    ARGS.usdz = assets.usdz
    ARGS.collision_usd = assets.collision_usd
    if not ARGS.trajectory_dir.exists():
        raise FileNotFoundError(ARGS.trajectory_dir)
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

    calibration = CameraCalibration(
        ARGS.width,
        ARGS.height,
        ARGS.horizontal_fov_deg,
        ARGS.fisheye_coefficients,
    )
    camera.set_opencv_fisheye_properties(
        cx=calibration.cx,
        cy=calibration.cy,
        fx=calibration.fx,
        fy=calibration.fy,
        fisheye=calibration.fisheye_coefficients,
    )
    actual_calibration = camera.get_opencv_fisheye_properties()
    expected_calibration = [
        calibration.cx,
        calibration.cy,
        calibration.fx,
        calibration.fy,
        *calibration.fisheye_coefficients,
    ]
    actual_calibration_flat = [
        *actual_calibration[:4],
        *actual_calibration[4],
    ]
    if not np.allclose(
        actual_calibration_flat,
        expected_calibration,
        rtol=1e-6,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Isaac Sim fisheye calibration does not match requested values: "
            f"actual={actual_calibration_flat}, expected={expected_calibration}"
        )
    if ARGS.mode == "depth":
        camera.add_distance_to_camera_to_frame()
    render_steps(world, ARGS.startup_steps)

    circular_mask = build_forward_mask(
        ARGS.width,
        ARGS.height,
        calibration.cx,
        calibration.cy,
        calibration.forward_mask_radius_pixels,
    )

    from sage3d.schemas import build_render_summary, render_summary_to_json

    summary = build_render_summary(
        scene_id=ARGS.scene,
        width=ARGS.width,
        height=ARGS.height,
        horizontal_fov_deg=calibration.horizontal_fov_deg,
        vertical_fov_deg=calibration.vertical_fov_deg,
        focal_length_pixels=calibration.fx,
        principal_point=[calibration.cx, calibration.cy],
        fisheye_coefficients=calibration.fisheye_coefficients,
        forward_mask_radius_pixels=calibration.forward_mask_radius_pixels,
        max_depth_m=ARGS.max_depth_m,
        min_depth_m=ARGS.min_depth_m,
        depth_scale=ARGS.depth_scale,
        render_mode=ARGS.mode,
        episodes=[],
        total_frames=0,
    )

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
                    orientation=yaw_to_quaternion(float(heading)),
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
                rgb = np.asarray(rgba)[..., :3].astype(np.uint8)
                rgb = mask_rgb(rgb, circular_mask)
                inside_pixels = rgb[circular_mask]
                if float(inside_pixels.std()) < 1.0:
                    raise RuntimeError(
                        f"Near-uniform RGB frame at episode={episode_index}, "
                        f"frame={frame_index}; NuRec renderer may have failed"
                    )

                stem = frame_stem(episode_index, frame_index)
                Image.fromarray(rgb).save(rgb_dir / f"{stem}.jpg", quality=95)
                if frame_index % 25 == 0 or frame_index == len(yaw) - 1:
                    print(
                        f"[render-rgb] episode {episode_index:06d}: "
                        f"{frame_index + 1}/{len(yaw)} frames"
                    )

    total_frames = 0
    if ARGS.mode == "depth":
        for episode_index, (camera_positions, yaw) in enumerate(trajectories):
            accumulator = RawDepthSummaryAccumulator(
                circular_mask, ARGS.min_depth_m
            )
            for frame_index, (position, heading) in enumerate(
                zip(camera_positions, yaw)
            ):
                camera.set_world_pose(
                    position=position,
                    orientation=yaw_to_quaternion(float(heading)),
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
                try:
                    accumulator.add(depth)
                except ValueError:
                    raise RuntimeError(
                        f"No finite collision depth at episode={episode_index}, "
                        f"frame={frame_index}"
                    ) from None

                depth_u16 = encode_depth(
                    depth,
                    circular_mask,
                    ARGS.min_depth_m,
                    ARGS.max_depth_m,
                    ARGS.depth_scale,
                )
                stem = frame_stem(episode_index, frame_index)
                Image.fromarray(depth_u16).save(depth_dir / f"{stem}.png")

                total_frames += 1
                if frame_index % 25 == 0 or frame_index == len(yaw) - 1:
                    print(
                        f"[render-depth] episode {episode_index:06d}: "
                        f"{frame_index + 1}/{len(yaw)} frames"
                    )

            depth_summary = accumulator.finish()
            summary["episodes"].append(
                {
                    "episode_index": episode_index,
                    "frame_count": len(yaw),
                    "finite_depth_fraction_mean": depth_summary[
                        "finite_depth_fraction_mean"
                    ],
                    "finite_depth_fraction_min": depth_summary[
                        "finite_depth_fraction_min"
                    ],
                    "finite_depth_min_m": depth_summary["finite_depth_min_m"],
                    "finite_depth_max_m": depth_summary["finite_depth_max_m"],
                }
            )
    else:
        total_frames = sum(len(yaw) for _, yaw in trajectories)

    summary["total_frames"] = total_frames
    summary_path = ARGS.output_dir / f"{ARGS.mode}_render_summary.json"
    render_summary_to_json(summary, summary_path)
    if ARGS.mode == "depth":
        # The packager's canonical render summary describes the metric depth
        # pass; the modality-specific copy makes the two-process split explicit.
        render_summary_to_json(
            summary, ARGS.output_dir / "render_summary.json"
        )
    print(
        f"[render-{ARGS.mode}] Completed {total_frames} frames: "
        f"{ARGS.output_dir}"
    )


try:
    main()
finally:
    simulation_app.close()

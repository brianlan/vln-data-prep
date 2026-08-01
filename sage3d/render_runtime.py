"""Render runtime: RenderMode strategies and unified episode loop (Phase 4).

Imported **after** ``SimulationApp`` construction (see ``render_bootstrap``).
Owns the mode-specific stage construction, camera configuration/readback,
per-frame capture, and the per-episode warmup/capture loop.

The top-level :func:`render` function bootstraps the app, builds the stage,
configures the camera with calibration readback, runs all episodes, and writes
the render summary JSON into ``staging_root``.
"""

from __future__ import annotations

from pathlib import Path

from sage3d.config import RenderConfig
from sage3d.render_bootstrap import bootstrap_render


def _render_steps(world: object, count: int) -> None:
    for _ in range(count):
        world.step(render=True)


# --- RenderMode strategies --------------------------------------------------


class RenderMode:
    """Strategy for mode-specific stage/camera/capture behavior."""

    def build_stage(
        self, stage: object, usdz: Path, collision_usd: Path
    ) -> None:
        raise NotImplementedError

    def configure_camera(self, camera: object) -> None:
        """Attach mode-specific annotators after camera init."""

    def begin_episode(self, config: RenderConfig, circular_mask: object) -> object:
        """Create per-episode state (e.g. depth accumulator). Returns None if unused."""
        return None

    def capture(
        self,
        *,
        camera: object,
        config: RenderConfig,
        circular_mask: object,
        episode_index: int,
        frame_index: int,
        n_frames: int,
        output_dir: Path,
        episode_state: object,
    ) -> int:
        """Capture, validate, and save one frame. Returns frame-count contribution."""
        raise NotImplementedError

    def finish_episode(
        self, episode_state: object
    ) -> dict | None:
        """Return per-episode summary dict, or None if no summary."""
        return None


class RGBMode(RenderMode):
    """RGB (NuRec appearance) render strategy."""

    def build_stage(self, stage: object, usdz: Path, collision_usd: Path) -> None:
        gauss = stage.OverridePrim("/World/gauss")
        gauss.GetReferences().AddReference(f"{usdz}[gauss.usda]")

    def capture(
        self,
        *,
        camera,
        config,
        circular_mask,
        episode_index,
        frame_index,
        n_frames,
        output_dir,
        episode_state,
    ) -> None:
        import numpy as np
        from PIL import Image

        from sage3d.naming import frame_stem
        from sage3d.render_processing import mask_rgb

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
        Image.fromarray(rgb).save(output_dir / f"{stem}.jpg", quality=95)
        if frame_index % 25 == 0 or frame_index == n_frames - 1:
            print(
                f"[render-rgb] episode {episode_index:06d}: "
                f"{frame_index + 1}/{n_frames} frames"
            )


class DepthMode(RenderMode):
    """Depth (collision mesh) render strategy."""

    def build_stage(self, stage: object, usdz: Path, collision_usd: Path) -> None:
        from pxr import UsdGeom

        collision = UsdGeom.Xform.Define(
            stage, "/World/scene_collision"
        ).GetPrim()
        collision.GetPayloads().AddPayload(str(collision_usd))

    def configure_camera(self, camera: object) -> None:
        camera.add_distance_to_camera_to_frame()

    def begin_episode(self, config: RenderConfig, circular_mask: object) -> object:
        from sage3d.render_processing import RawDepthSummaryAccumulator

        return RawDepthSummaryAccumulator(circular_mask, config.min_depth_m)

    def capture(
        self,
        *,
        camera,
        config,
        circular_mask,
        episode_index,
        frame_index,
        n_frames,
        output_dir,
        episode_state,
    ) -> None:
        import numpy as np
        from PIL import Image

        from sage3d.naming import frame_stem
        from sage3d.render_processing import encode_depth

        frame = camera.get_current_frame(clone=True)
        depth = frame.get("distance_to_camera") if frame else None
        if depth is None:
            raise RuntimeError(
                f"No distance_to_camera depth at episode={episode_index}, "
                f"frame={frame_index}; keys={list(frame) if frame else []}"
            )
        depth = np.asarray(depth, dtype=np.float32).squeeze()
        if depth.shape != (config.height, config.width):
            raise RuntimeError(
                f"Unexpected depth shape {depth.shape}; expected "
                f"{(config.height, config.width)}"
            )
        try:
            episode_state.add(depth)
        except ValueError:
            raise RuntimeError(
                f"No finite collision depth at episode={episode_index}, "
                f"frame={frame_index}"
            ) from None
        depth_u16 = encode_depth(
            depth,
            circular_mask,
            config.min_depth_m,
            config.max_depth_m,
            config.depth_scale,
        )
        stem = frame_stem(episode_index, frame_index)
        Image.fromarray(depth_u16).save(output_dir / f"{stem}.png")
        if frame_index % 25 == 0 or frame_index == n_frames - 1:
            print(
                f"[render-depth] episode {episode_index:06d}: "
                f"{frame_index + 1}/{n_frames} frames"
            )

    def finish_episode(self, episode_state: object) -> dict | None:
        summary = episode_state.finish()
        return {
            "finite_depth_fraction_mean": summary["finite_depth_fraction_mean"],
            "finite_depth_fraction_min": summary["finite_depth_fraction_min"],
            "finite_depth_min_m": summary["finite_depth_min_m"],
            "finite_depth_max_m": summary["finite_depth_max_m"],
        }


def render_episode(
    *,
    mode: RenderMode,
    camera: object,
    world: object,
    config: RenderConfig,
    circular_mask: object,
    camera_positions: object,
    yaw: object,
    episode_index: int,
    output_dir: Path,
) -> object:
    """Run one episode: per-frame pose → warmup → capture.

    First frame of **every episode** uses startup steps; later frames use
    settle steps. Returns the mode-specific episode state (e.g. depth
    accumulator), or ``None`` for RGB.
    """
    from sage3d.frames import yaw_to_quaternion

    n_frames = len(yaw)
    if len(camera_positions) != len(yaw):
        raise RuntimeError(
            f"Pose/yaw count mismatch in episode {episode_index}: "
            f"{len(camera_positions)} vs {len(yaw)}"
        )

    episode_state = mode.begin_episode(config, circular_mask)
    for frame_index, (position, heading) in enumerate(
        zip(camera_positions, yaw)
    ):
        camera.set_world_pose(
            position=position,
            orientation=yaw_to_quaternion(float(heading)),
            camera_axes="world",
        )
        _render_steps(
            world,
            config.startup_steps if frame_index == 0 else config.settle_steps,
        )
        mode.capture(
            camera=camera,
            config=config,
            circular_mask=circular_mask,
            episode_index=episode_index,
            frame_index=frame_index,
            n_frames=n_frames,
            output_dir=output_dir,
            episode_state=episode_state,
        )
    return episode_state


_MODES = {"rgb": RGBMode, "depth": DepthMode}


def render(
    config: RenderConfig,
    staging_root: Path,
    *,
    scene_id: str,
    usdz: Path,
    collision_usd: Path,
    trajectory_dir: Path,
) -> None:
    """Render all episodes for one mode into ``staging_root``.

    Bootstraps ``SimulationApp``, builds the mode-specific stage, configures
    the camera with calibration readback, runs the per-episode warmup/capture
    loop, and writes the render summary JSON.
    """
    import numpy as np

    from sage3d.camera import CameraCalibration
    from sage3d.render_processing import build_forward_mask
    from sage3d.schemas import build_render_summary, render_summary_to_json

    if not trajectory_dir.exists():
        raise FileNotFoundError(trajectory_dir)
    trajectory_files = sorted(trajectory_dir.glob("episode_*.npz"))
    if not trajectory_files:
        raise RuntimeError(
            f"No episode_*.npz files found in {trajectory_dir}"
        )

    mode = _MODES[config.mode]()

    rgb_dir = staging_root / "observation.images.rgb"
    depth_dir = staging_root / "observation.images.depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    with bootstrap_render(config, staging_root) as proxy:
        runtime = proxy.runtime

        context = runtime.omni_usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        world_prim = runtime.UsdGeom.Xform.Define(stage, "/World").GetPrim()
        stage.SetDefaultPrim(world_prim)

        mode.build_stage(stage, usdz, collision_usd)

        world = runtime.World(stage_units_in_meters=1.0)
        world.reset()
        _render_steps(world, config.startup_steps)

        camera = runtime.Camera(
            prim_path="/World/PointGoalFisheyeCamera",
            frequency=30,
            resolution=(config.width, config.height),
        )
        camera.initialize()
        camera.set_clipping_range(0.05, max(20.0, config.max_depth_m * 2.0))

        calibration = CameraCalibration(
            config.width,
            config.height,
            config.horizontal_fov_deg,
            config.fisheye_coefficients,
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
        mode.configure_camera(camera)
        _render_steps(world, config.startup_steps)

        circular_mask = build_forward_mask(
            config.width,
            config.height,
            calibration.cx,
            calibration.cy,
            calibration.forward_mask_radius_pixels,
        )

        summary = build_render_summary(
            scene_id=scene_id,
            width=config.width,
            height=config.height,
            horizontal_fov_deg=calibration.horizontal_fov_deg,
            vertical_fov_deg=calibration.vertical_fov_deg,
            focal_length_pixels=calibration.fx,
            principal_point=[calibration.cx, calibration.cy],
            fisheye_coefficients=list(calibration.fisheye_coefficients),
            forward_mask_radius_pixels=calibration.forward_mask_radius_pixels,
            max_depth_m=config.max_depth_m,
            min_depth_m=config.min_depth_m,
            depth_scale=config.depth_scale,
            render_mode=config.mode,
            episodes=[],
            total_frames=0,
        )

        trajectories = []
        for trajectory_file in trajectory_files:
            trajectory = np.load(trajectory_file)
            camera_positions = trajectory["camera_positions"].copy()
            yaw = trajectory["yaw"].copy()
            trajectories.append((camera_positions, yaw))

        total_frames = 0
        for episode_index, (camera_positions, yaw) in enumerate(trajectories):
            output_dir = depth_dir if config.mode == "depth" else rgb_dir
            episode_state = render_episode(
                mode=mode,
                camera=camera,
                world=world,
                config=config,
                circular_mask=circular_mask,
                camera_positions=camera_positions,
                yaw=yaw,
                episode_index=episode_index,
                output_dir=output_dir,
            )
            total_frames += len(yaw)
            ep_summary = mode.finish_episode(episode_state)
            if ep_summary is not None:
                ep_summary["episode_index"] = episode_index
                ep_summary["frame_count"] = len(yaw)
                summary["episodes"].append(ep_summary)

        summary["total_frames"] = total_frames
        summary_path = staging_root / f"{config.mode}_render_summary.json"
        render_summary_to_json(summary, summary_path)
        if config.mode == "depth":
            render_summary_to_json(
                summary, staging_root / "render_summary.json"
            )
        print(
            f"[render-{config.mode}] Completed {total_frames} frames: "
            f"{staging_root}"
        )
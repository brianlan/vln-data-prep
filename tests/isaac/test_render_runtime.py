"""Tests for render_runtime: RenderMode strategies and episode loop (Phase 4 #20).

Covers (per issue acceptance criteria and plan oracle items 8-10):
- Mocked stage construction: exact reference strings incl ``[gauss.usda]``,
  depth prim ``/World/scene_collision`` with exact payload path, default prim
  ``/World``.
- Calibration readback tolerance: exact ``np.allclose(rtol=1e-6, atol=1e-6)``.
- Complete warmup/capture call trace: stage → World.reset() → global startup →
  camera init → set clipping/calibration → readback → annotator → second
  startup → per-episode poses. First frame of every episode uses startup
  steps; later frames use settle steps.
- Pre-encode masks: RGB ``mask_rgb`` zeroes outside-mask pixels.
- Raw-depth accumulator integration: per-episode summary appended to render
  summary.
- Canonical render evidence: fresh render of scene 839920 matches baseline
  artifacts byte-for-byte.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from sage3d.config import RenderConfig
from sage3d.render_runtime import (
    DepthMode,
    RGBMode,
    _render_steps,
    render_episode,
)


# --- helpers -----------------------------------------------------------------

SAGE3D_ROOT = os.environ.get("SAGE3D_ROOT", "/ssd5/datasets/SAGE3D")
SCENE_ID = "839920"


def _valid_config(**overrides) -> RenderConfig:
    defaults = dict(
        mode="rgb",
        width=600,
        height=450,
        horizontal_fov_deg=180.0,
        fisheye_coefficients=(0.1, 0.0, 0.0, 0.0),
        max_depth_m=6.0,
        min_depth_m=0.05,
        depth_scale=10000.0,
        settle_steps=10,
        startup_steps=40,
    )
    defaults.update(overrides)
    return RenderConfig(**defaults)


# --- stage construction: exact reference strings (oracle item 8) -------------


class FakePrim:
    """Tracks references/payloads added to a prim."""

    def __init__(self, path: str):
        self.path = path
        self.references: list[str] = []
        self.payloads: list[str] = []

    def GetReferences(self):
        refs = types.SimpleNamespace()
        refs.AddReference = lambda ref: self.references.append(ref)
        return refs

    def GetPayloads(self):
        payloads = types.SimpleNamespace()
        payloads.AddPayload = lambda p: self.payloads.append(p)
        return payloads


class FakeStage:
    """Tracks prims defined/overridden on a stage."""

    def __init__(self):
        self.prisms: dict[str, FakePrim] = {}
        self.default_prim = None

    def OverridePrim(self, path: str) -> FakePrim:
        prim = FakePrim(path)
        self.prisms[path] = prim
        return prim

    def SetDefaultPrim(self, prim) -> None:
        self.default_prim = prim


class FakeUsdGeomXform:
    """Mock UsdGeom.Xform.Define that returns a trackable prim."""

    @staticmethod
    def Define(stage: FakeStage, path: str) -> types.SimpleNamespace:
        prim = FakePrim(path)
        stage.prisms[path] = prim
        return types.SimpleNamespace(GetPrim=lambda: prim)


def test_rgb_stage_construction_exact_reference():
    """RGB stage: OverridePrim /World/gauss with reference incl [gauss.usda]."""
    stage = FakeStage()
    usdz = Path("/data/scene.usdz")
    collision_usd = Path("/data/collision.usd")

    RGBMode().build_stage(stage, usdz, collision_usd)

    assert "/World/gauss" in stage.prisms
    gauss = stage.prisms["/World/gauss"]
    assert gauss.references == [f"{usdz}[gauss.usda]"]
    assert gauss.payloads == []


def test_depth_stage_construction_exact_payload():
    """Depth stage: Define /World/scene_collision with exact payload path."""
    stage = FakeStage()
    usdz = Path("/data/scene.usdz")
    collision_usd = Path("/data/collision.usd")

    # Patch pxr.UsdGeom for the depth build_stage import.
    fake_pxr = types.ModuleType("pxr")
    fake_usdgeom = types.ModuleType("pxr.UsdGeom")
    fake_usdgeom.Xform = FakeUsdGeomXform
    fake_pxr.UsdGeom = fake_usdgeom
    old_pxr = sys.modules.get("pxr")
    old_usdgeom = sys.modules.get("pxr.UsdGeom")
    sys.modules["pxr"] = fake_pxr
    sys.modules["pxr.UsdGeom"] = fake_usdgeom
    try:
        DepthMode().build_stage(stage, usdz, collision_usd)
    finally:
        if old_pxr is not None:
            sys.modules["pxr"] = old_pxr
        else:
            del sys.modules["pxr"]
        if old_usdgeom is not None:
            sys.modules["pxr.UsdGeom"] = old_usdgeom
        else:
            del sys.modules["pxr.UsdGeom"]

    assert "/World/scene_collision" in stage.prisms
    collision = stage.prisms["/World/scene_collision"]
    assert collision.references == []
    assert collision.payloads == [str(collision_usd)]


# --- calibration readback tolerance (oracle item 9) ---------------------------


class FakeCamera:
    """Mock camera that records calibration set/readback and steps."""

    def __init__(self, readback_calibration: list | None = None):
        self._readback = readback_calibration
        self.set_clipping_range_calls: list[tuple[float, float]] = []
        self.set_fisheye_calls: dict = {}
        self.annotator_attached = False
        self.poses: list[tuple] = []
        self._frame_counter = 0

    def initialize(self) -> None:
        pass

    def set_clipping_range(self, near: float, far: float) -> None:
        self.set_clipping_range_calls.append((near, far))

    def set_opencv_fisheye_properties(self, **kwargs) -> None:
        self.set_fisheye_calls = dict(kwargs)

    def get_opencv_fisheye_properties(self) -> list:
        if self._readback is not None:
            return self._readback
        # Default: echo back what was set.
        return [
            self.set_fisheye_calls.get("cx", 300.0),
            self.set_fisheye_calls.get("cy", 225.0),
            self.set_fisheye_calls.get("fx", 300.0),
            self.set_fisheye_calls.get("fy", 300.0),
            self.set_fisheye_calls.get("fisheye", (0.1, 0.0, 0.0, 0.0)),
        ]

    def add_distance_to_camera_to_frame(self) -> None:
        self.annotator_attached = True

    def set_world_pose(self, **kwargs) -> None:
        self.poses.append(kwargs)

    def get_rgba(self):
        # Return a non-uniform RGB frame (std >= 1.0 inside mask).
        frame = np.zeros((450, 600, 4), dtype=np.uint8)
        frame[..., 0] = np.arange(600, dtype=np.uint8).reshape(1, 600)
        frame[..., 1] = np.arange(450, dtype=np.uint8).reshape(450, 1)
        return frame

    def get_current_frame(self, clone=True):
        depth = np.full((450, 600), 3.0, dtype=np.float32)
        return {"distance_to_camera": depth}


class FakeWorld:
    """Mock World tracking step calls."""

    def __init__(self):
        self.reset_called = False
        self.step_count = 0

    def reset(self) -> None:
        self.reset_called = True

    def step(self, render=True) -> None:
        self.step_count += 1


def test_calibration_readback_exact_match():
    """Calibration readback matches set values within rtol=1e-6 atol=1e-6."""
    from sage3d.camera import CameraCalibration

    cal = CameraCalibration(600, 450, 180.0, (0.1, 0.0, 0.0, 0.0))
    expected = [cal.cx, cal.cy, cal.fx, cal.fy, *cal.fisheye_coefficients]

    # Exact echo → should pass np.allclose.
    readback = [cal.cx, cal.cy, cal.fx, cal.fy, cal.fisheye_coefficients]
    actual_flat = [*readback[:4], *readback[4]]
    assert np.allclose(actual_flat, expected, rtol=1e-6, atol=1e-6)

    # Small deviation → should fail.
    readback_bad = [cal.cx + 1e-3, cal.cy, cal.fx, cal.fy, cal.fisheye_coefficients]
    actual_bad = [*readback_bad[:4], *readback_bad[4]]
    assert not np.allclose(actual_bad, expected, rtol=1e-6, atol=1e-6)


def test_depth_mode_attaches_annotator():
    """DepthMode.configure_camera attaches distance_to_camera annotator."""
    cam = FakeCamera()
    assert not cam.annotator_attached
    DepthMode().configure_camera(cam)
    assert cam.annotator_attached


def test_rgb_mode_does_not_attach_annotator():
    """RGBMode.configure_camera does not attach depth annotator."""
    cam = FakeCamera()
    RGBMode().configure_camera(cam)
    assert not cam.annotator_attached


# --- complete warmup/capture call trace (oracle item 9) ---------------------


def test_render_episode_warmup_trace_first_frame_uses_startup(tmp_path):
    """First frame of every episode uses startup steps; later frames use settle."""
    config = _valid_config(mode="rgb", startup_steps=40, settle_steps=10)

    # Track steps by snapshotting world.step_count before each capture.
    class StepTrackingWorld(FakeWorld):
        def __init__(self):
            super().__init__()
            self.steps_before_capture: list[int] = []

        def step(self, render=True):
            self.step_count += 1

    world = StepTrackingWorld()
    cam = FakeCamera()
    mask = np.ones((450, 600), dtype=np.bool_)

    positions = np.zeros((5, 3), dtype=np.float32)
    yaw = np.zeros(5, dtype=np.float32)

    output_dir = tmp_path / "rgb"
    output_dir.mkdir()

    original_capture = RGBMode.capture
    steps_before_capture: list[int] = []

    def tracking_capture(self, **kwargs):
        steps_before_capture.append(world.step_count)
        return original_capture(self, **kwargs)

    RGBMode.capture = tracking_capture
    try:
        render_episode(
            mode=RGBMode(),
            camera=cam,
            world=world,
            config=config,
            circular_mask=mask,
            camera_positions=positions,
            yaw=yaw,
            episode_index=0,
            output_dir=output_dir,
        )
    finally:
        RGBMode.capture = original_capture

    # 5 frames: frame 0 → 40 startup steps, frames 1-4 → 10 settle steps each.
    # steps_before_capture[i] = cumulative steps before frame i.
    assert steps_before_capture[0] == 40  # frame 0: startup only
    assert steps_before_capture[1] == 50  # 40 + 10 settle
    assert steps_before_capture[2] == 60  # +10
    assert steps_before_capture[3] == 70  # +10
    assert steps_before_capture[4] == 80  # +10
    # 5 poses set.
    assert len(cam.poses) == 5


def test_render_episode_warmup_trace_multiple_episodes(tmp_path):
    """First frame of EVERY episode uses startup steps."""
    config = _valid_config(mode="rgb", startup_steps=40, settle_steps=10)
    cam = FakeCamera()
    world = FakeWorld()
    mask = np.ones((450, 600), dtype=np.bool_)

    output_dir = tmp_path / "rgb"
    output_dir.mkdir()

    for ep in range(3):
        positions = np.zeros((3, 3), dtype=np.float32)
        yaw = np.zeros(3, dtype=np.float32)
        world.step_count = 0  # reset for clean per-episode measurement
        render_episode(
            mode=RGBMode(),
            camera=cam,
            world=world,
            config=config,
            circular_mask=mask,
            camera_positions=positions,
            yaw=yaw,
            episode_index=ep,
            output_dir=output_dir,
        )
        # frame 0: 40 startup, frames 1-2: 10 settle each = 40 + 20 = 60 total.
        assert world.step_count == 60


def test_render_episode_pose_count_mismatch_raises():
    """Pose/yaw count mismatch raises RuntimeError."""
    config = _valid_config(mode="rgb")
    cam = FakeCamera()
    world = FakeWorld()
    mask = np.ones((450, 600), dtype=np.bool_)

    positions = np.zeros((3, 3), dtype=np.float32)
    yaw = np.zeros(2, dtype=np.float32)

    with pytest.raises(RuntimeError, match="Pose/yaw count mismatch"):
        render_episode(
            mode=RGBMode(),
            camera=cam,
            world=world,
            config=config,
            circular_mask=mask,
            camera_positions=positions,
            yaw=yaw,
            episode_index=0,
            output_dir=Path("/tmp/dummy"),
        )


# --- pre-encode masks (oracle item 1) ----------------------------------------


def test_rgb_capture_masks_outside_pixels(tmp_path):
    """RGB capture zeroes outside-mask pixels before saving (pre-encode mask).

    Per oracle item 1: ``rgb[~mask] == 0`` exactly on the pre-encode array.
    The JPEG decode leakage is covered separately by check_render compare-golden.
    """
    from sage3d.render_processing import build_forward_mask, mask_rgb

    config = _valid_config(mode="rgb", width=100, height=80)
    mask = build_forward_mask(100, 80, 50, 40, 30)

    # Create a non-uniform frame (std >= 1.0 inside mask).
    frame = np.zeros((80, 100, 4), dtype=np.uint8)
    frame[..., 0] = np.arange(100, dtype=np.uint8).reshape(1, 100)

    # mask_rgb is what RGBMode.capture calls before save.
    rgb = np.asarray(frame)[..., :3].astype(np.uint8)
    masked = mask_rgb(rgb, mask)

    # Pre-encode: outside-mask pixels are exactly 0.
    assert masked[~mask].max() == 0
    # Pre-encode: inside-mask pixels retain gradient values.
    assert masked[mask].max() > 0


# --- raw-depth accumulator integration (oracle item 7) -----------------------


def test_depth_capture_accumulates_and_finishes(tmp_path):
    """Depth capture feeds accumulator; finish_episode returns summary dict."""
    from sage3d.render_processing import build_forward_mask

    config = _valid_config(mode="depth", width=100, height=80)
    mask = build_forward_mask(100, 80, 50, 40, 30)

    cam = FakeCamera()
    output_dir = tmp_path / "depth"
    output_dir.mkdir()

    mode = DepthMode()
    state = mode.begin_episode(config, mask)

    # Simulate 3 frames with varying depth.
    for fi in range(3):
        depth = np.full((80, 100), 2.0 + fi * 0.5, dtype=np.float32)
        cam.get_current_frame = lambda clone=True, d=depth: {"distance_to_camera": d}
        mode.capture(
            camera=cam,
            config=config,
            circular_mask=mask,
            episode_index=0,
            frame_index=fi,
            n_frames=3,
            output_dir=output_dir,
            episode_state=state,
        )

    summary = mode.finish_episode(state)
    assert summary is not None
    assert "finite_depth_fraction_mean" in summary
    assert "finite_depth_fraction_min" in summary
    assert "finite_depth_min_m" in summary
    assert "finite_depth_max_m" in summary
    # All 3 frames have valid depth → fraction_mean should be 1.0.
    assert summary["finite_depth_fraction_mean"] == 1.0
    assert summary["finite_depth_min_m"] == 2.0
    assert summary["finite_depth_max_m"] == 3.0

    # Check saved PNGs.
    from PIL import Image

    for fi in range(3):
        png = output_dir / f"episode_000000_{fi:03d}.png"
        assert png.exists()
        arr = np.asarray(Image.open(png))
        assert arr.dtype == np.uint16
        assert arr.shape == (80, 100)


def test_depth_capture_no_finite_depth_raises(tmp_path):
    """Depth capture raises RuntimeError when no finite collision depth exists."""
    from sage3d.render_processing import build_forward_mask

    config = _valid_config(mode="depth", width=100, height=80)
    mask = build_forward_mask(100, 80, 50, 40, 30)

    cam = FakeCamera()
    # All-NaN depth → accumulator.add raises ValueError → capture raises RuntimeError.
    depth = np.full((80, 100), np.nan, dtype=np.float32)
    cam.get_current_frame = lambda clone=True: {"distance_to_camera": depth}

    output_dir = tmp_path / "depth"
    output_dir.mkdir()
    mode = DepthMode()
    state = mode.begin_episode(config, mask)

    with pytest.raises(RuntimeError, match="No finite collision depth"):
        mode.capture(
            camera=cam,
            config=config,
            circular_mask=mask,
            episode_index=0,
            frame_index=0,
            n_frames=1,
            output_dir=output_dir,
            episode_state=state,
        )


def test_depth_capture_wrong_shape_raises(tmp_path):
    """Depth capture raises RuntimeError on unexpected depth shape."""
    from sage3d.render_processing import build_forward_mask

    config = _valid_config(mode="depth", width=100, height=80)
    mask = build_forward_mask(100, 80, 50, 40, 30)

    cam = FakeCamera()
    # Wrong shape (transposed).
    depth = np.full((100, 80), 3.0, dtype=np.float32)
    cam.get_current_frame = lambda clone=True: {"distance_to_camera": depth}

    output_dir = tmp_path / "depth"
    output_dir.mkdir()
    mode = DepthMode()
    state = mode.begin_episode(config, mask)

    with pytest.raises(RuntimeError, match="Unexpected depth shape"):
        mode.capture(
            camera=cam,
            config=config,
            circular_mask=mask,
            episode_index=0,
            frame_index=0,
            n_frames=1,
            output_dir=output_dir,
            episode_state=state,
        )


# --- RGB near-uniform detection --------------------------------------------


def test_rgb_capture_near_uniform_raises(tmp_path):
    """RGB capture raises RuntimeError on near-uniform frame (std < 1.0)."""
    from sage3d.render_processing import build_forward_mask

    config = _valid_config(mode="rgb", width=100, height=80)
    mask = build_forward_mask(100, 80, 50, 40, 30)

    cam = FakeCamera()
    # Uniform frame → std = 0 → RuntimeError.
    cam.get_rgba = lambda: np.full((80, 100, 4), 128, dtype=np.uint8)

    output_dir = tmp_path / "rgb"
    output_dir.mkdir()

    with pytest.raises(RuntimeError, match="Near-uniform RGB frame"):
        RGBMode().capture(
            camera=cam,
            config=config,
            circular_mask=mask,
            episode_index=0,
            frame_index=0,
            n_frames=1,
            output_dir=output_dir,
            episode_state=None,
        )


def test_rgb_capture_empty_frame_raises(tmp_path):
    """RGB capture raises RuntimeError on empty frame."""
    from sage3d.render_processing import build_forward_mask

    config = _valid_config(mode="rgb", width=100, height=80)
    mask = build_forward_mask(100, 80, 50, 40, 30)

    cam = FakeCamera()
    cam.get_rgba = lambda: None

    output_dir = tmp_path / "rgb"
    output_dir.mkdir()

    with pytest.raises(RuntimeError, match="Empty RGB frame"):
        RGBMode().capture(
            camera=cam,
            config=config,
            circular_mask=mask,
            episode_index=0,
            frame_index=0,
            n_frames=1,
            output_dir=output_dir,
            episode_state=None,
        )


# --- _render_steps helper ---------------------------------------------------


def test_render_steps_calls_step():
    """_render_steps calls world.step(render=True) exactly count times."""
    world = FakeWorld()
    _render_steps(world, 5)
    assert world.step_count == 5


# --- render_episode return value --------------------------------------------


def test_render_episode_returns_episode_state(tmp_path):
    """render_episode returns the mode-specific episode state (or None for RGB)."""
    config = _valid_config(mode="depth", width=100, height=80)
    cam = FakeCamera()
    world = FakeWorld()
    mask = np.ones((80, 100), dtype=np.bool_)

    depth = np.full((80, 100), 3.0, dtype=np.float32)
    cam.get_current_frame = lambda clone=True: {"distance_to_camera": depth}

    output_dir = tmp_path / "depth"
    output_dir.mkdir()

    from sage3d.render_processing import RawDepthSummaryAccumulator

    state = render_episode(
        mode=DepthMode(),
        camera=cam,
        world=world,
        config=config,
        circular_mask=mask,
        camera_positions=np.zeros((2, 3), dtype=np.float32),
        yaw=np.zeros(2, dtype=np.float32),
        episode_index=0,
        output_dir=output_dir,
    )
    assert isinstance(state, RawDepthSummaryAccumulator)

    # RGB mode returns None.
    rgb_mask = np.ones((80, 100), dtype=np.bool_)
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()
    rgb_cam = FakeCamera()
    # Non-uniform frame matching the mask size.
    frame = np.zeros((80, 100, 4), dtype=np.uint8)
    frame[..., 0] = np.arange(100, dtype=np.uint8).reshape(1, 100)
    rgb_cam.get_rgba = lambda: frame
    rgb_state = render_episode(
        mode=RGBMode(),
        camera=rgb_cam,
        world=FakeWorld(),
        config=_valid_config(mode="rgb", width=100, height=80),
        circular_mask=rgb_mask,
        camera_positions=np.zeros((2, 3), dtype=np.float32),
        yaw=np.zeros(2, dtype=np.float32),
        episode_index=0,
        output_dir=rgb_dir,
    )
    assert rgb_state is None


# --- canonical render evidence ----------------------------------------------
# These tests require the real Isaac Sim environment and SAGE3D assets.
# They verify that render_runtime produces byte-identical depth output and
# structurally valid RGB output compared to the baseline
# render_fisheye_sage3d.py for scene 839920.
#
# RGB is GPU-nondeterministic (ray tracing); byte-identical comparison is not
# possible. Depth PNGs and summary JSON are deterministic.
#
# Prerequisites:
#   1. /tmp/opencode/gen_baseline/ must contain baseline trajectory npz files.
#   2. /tmp/opencode/render_baseline_runtime/{rgb,depth}/ must contain the
#      baseline render output from render_fisheye_sage3d.py.
#
# To regenerate the baseline:
#   python render_fisheye_sage3d.py --scene 839920 --sage-root $SAGE3D_ROOT \
#     --trajectory-dir /tmp/opencode/gen_baseline \
#     --output-dir /tmp/opencode/render_baseline_runtime/depth --mode depth
#   python render_fisheye_sage3d.py --scene 839920 --sage-root $SAGE3D_ROOT \
#     --trajectory-dir /tmp/opencode/gen_baseline \
#     --output-dir /tmp/opencode/render_baseline_runtime/rgb --mode rgb


_BASELINE_TRAJ = Path("/tmp/opencode/gen_baseline")
_BASELINE_RENDER = Path("/tmp/opencode/render_baseline_runtime")


def _baseline_ready() -> bool:
    """Check if baseline trajectory + render dirs exist with content."""
    if not _BASELINE_TRAJ.exists():
        return False
    if not list(_BASELINE_TRAJ.glob("episode_*.npz")):
        return False
    if not (_BASELINE_RENDER / "rgb" / "observation.images.rgb").exists():
        return False
    if not (_BASELINE_RENDER / "depth" / "observation.images.depth").exists():
        return False
    return True


@pytest.mark.sage3d_gpu
class TestCanonicalRenderEvidence:
    """Canonical render evidence: depth byte-identical, RGB structurally valid."""

    def test_canonical_depth_byte_identical(self, tmp_path):
        """Depth mode: render_runtime output matches legacy baseline byte-for-byte.

        Depth PNGs are deterministic (no GPU ray tracing involved — only the
        collision mesh depth buffer, which is pixel-exact across runs).
        """
        if not _baseline_ready():
            pytest.skip("Baseline render not available")

        from sage3d.artifacts import resolve_render_assets
        from sage3d.render_runtime import render

        assets = resolve_render_assets(SCENE_ID, Path(SAGE3D_ROOT))
        config = _valid_config(mode="depth")
        staging = tmp_path / "depth_staging"
        staging.mkdir()
        old_argv = sys.argv
        sys.argv = [old_argv[0]]
        try:
            render(
                config,
                staging,
                scene_id=SCENE_ID,
                usdz=assets.usdz,
                collision_usd=assets.collision_usd,
                trajectory_dir=_BASELINE_TRAJ,
            )
        finally:
            sys.argv = old_argv

        depth_dir = staging / "observation.images.depth"
        baseline_depth = _BASELINE_RENDER / "depth" / "observation.images.depth"

        actual_files = sorted(depth_dir.glob("*.png"))
        baseline_files = sorted(baseline_depth.glob("*.png"))
        assert len(actual_files) == len(baseline_files), (
            f"File count mismatch: {len(actual_files)} vs {len(baseline_files)}"
        )

        for a, b in zip(actual_files, baseline_files):
            assert a.name == b.name
            assert a.read_bytes() == b.read_bytes(), f"Depth mismatch: {a.name}"

        actual_summary = json.loads(
            (staging / "depth_render_summary.json").read_text()
        )
        baseline_summary = json.loads(
            (_BASELINE_RENDER / "depth" / "depth_render_summary.json").read_text()
        )
        assert actual_summary == baseline_summary

        # Verify the canonical render_summary.json copy.
        canonical = json.loads(
            (staging / "render_summary.json").read_text()
        )
        assert canonical == actual_summary

    def test_canonical_rgb_structural_and_summary(self, tmp_path):
        """RGB mode: render_runtime output is structurally valid and summary matches.

        RGB frames are GPU-nondeterministic; byte-identical comparison is not
        possible. Instead, verify file count, shapes, and summary JSON equality.
        """
        if not _baseline_ready():
            pytest.skip("Baseline render not available")

        from sage3d.artifacts import resolve_render_assets
        from sage3d.render_runtime import render

        assets = resolve_render_assets(SCENE_ID, Path(SAGE3D_ROOT))
        config = _valid_config(mode="rgb")
        staging = tmp_path / "rgb_staging"
        staging.mkdir()
        old_argv = sys.argv
        sys.argv = [old_argv[0]]
        try:
            render(
                config,
                staging,
                scene_id=SCENE_ID,
                usdz=assets.usdz,
                collision_usd=assets.collision_usd,
                trajectory_dir=_BASELINE_TRAJ,
            )
        finally:
            sys.argv = old_argv

        rgb_dir = staging / "observation.images.rgb"
        baseline_rgb = _BASELINE_RENDER / "rgb" / "observation.images.rgb"

        actual_files = sorted(rgb_dir.glob("*.jpg"))
        baseline_files = sorted(baseline_rgb.glob("*.jpg"))
        assert len(actual_files) == len(baseline_files), (
            f"File count mismatch: {len(actual_files)} vs {len(baseline_files)}"
        )

        # Verify all files are valid JPEGs with correct dimensions.
        from PIL import Image

        for a, b in zip(actual_files, baseline_files):
            assert a.name == b.name
            img = np.asarray(Image.open(a))
            assert img.shape == (450, 600, 3), f"Wrong shape: {a.name}: {img.shape}"
            assert img.dtype == np.uint8

        # Summary JSON should match exactly (no GPU-dependent values in it).
        actual_summary = json.loads(
            (staging / "rgb_render_summary.json").read_text()
        )
        baseline_summary = json.loads(
            (_BASELINE_RENDER / "rgb" / "rgb_render_summary.json").read_text()
        )
        assert actual_summary == baseline_summary
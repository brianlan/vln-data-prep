"""Tests for render bootstrap and SimulationApp lifecycle (Phase 4 issue #19).

Covers:
- RenderConfig validation (invalid mode, width, height, depth range, scale,
  steps, fisheye coefficients).
- Depth sentinel preflight rejects invalid config before app construction.
- Stage absence/symlink checks.
- Import ordering: SimulationApp constructed before Isaac runtime imports.
- App close in ``finally`` on runtime import failure, stage setup failure,
  and normal exit.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from sage3d.config import RenderConfig
from sage3d.render_bootstrap import (
    SimulationAppProxy,
    bootstrap_render,
    preflight_depth_sentinel,
    validate_staging_root,
)


# --- helpers -----------------------------------------------------------------

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


# --- RenderConfig validation -------------------------------------------------

@pytest.mark.parametrize(
    "kwargs,expected_substring",
    [
        ({"mode": "bad"}, "mode must be 'rgb' or 'depth'"),
        ({"width": 0}, "width must be positive"),
        ({"height": -1}, "height must be positive"),
        ({"horizontal_fov_deg": 0}, "horizontal_fov_deg must be positive"),
        ({"fisheye_coefficients": (0.1, 0, 0)}, "fisheye_coefficients must have 4"),
        ({"max_depth_m": 0}, "max_depth_m must be finite positive"),
        ({"max_depth_m": float("inf")}, "max_depth_m must be finite positive"),
        ({"min_depth_m": -0.1}, "min_depth_m must be finite non-negative"),
        ({"min_depth_m": 6.0, "max_depth_m": 6.0}, "min_depth_m must be < max_depth_m"),
        ({"depth_scale": 0}, "depth_scale must be finite positive"),
        ({"depth_scale": float("nan")}, "depth_scale must be finite positive"),
        ({"settle_steps": -1}, "settle_steps must be non-negative"),
        ({"startup_steps": -5}, "startup_steps must be non-negative"),
    ],
)
def test_render_config_rejects_invalid(kwargs, expected_substring):
    with pytest.raises(ValueError, match=expected_substring):
        _valid_config(**kwargs)


def test_render_config_accepts_valid_both_modes():
    for mode in ("rgb", "depth"):
        c = _valid_config(mode=mode)
        assert c.mode == mode


# --- sentinel preflight ------------------------------------------------------

def test_preflight_depth_sentinel_valid():
    """Valid config does not raise."""
    preflight_depth_sentinel(_valid_config(mode="depth"))
    preflight_depth_sentinel(_valid_config(mode="rgb"))


def test_preflight_depth_sentinel_overflow():
    """Scaled sentinel exceeding 65535 fails before app construction."""
    # 10.0 * 10000 = 100000 > 65535
    cfg = _valid_config(max_depth_m=10.0, depth_scale=10000.0)
    with pytest.raises(ValueError, match="exceeds 65535"):
        preflight_depth_sentinel(cfg)


# --- staging root validation -------------------------------------------------


def test_staging_root_absent_rejected(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="expected existing real directory"):
        validate_staging_root(missing)


def test_staging_root_symlink_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="refusing symlinked directory"):
        validate_staging_root(link)


def test_staging_root_file_rejected(tmp_path):
    f = tmp_path / "not_a_dir"
    f.write_text("data")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        validate_staging_root(f)


def test_staging_root_valid_accepted(tmp_path):
    d = tmp_path / "staging"
    d.mkdir()
    validate_staging_root(d)  # should not raise


# --- bootstrap_render with mocked SimulationApp -----------------------------


class FakeSimulationApp:
    """Minimal mock for SimulationApp tracking close() calls."""

    def __init__(self, config=None, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.closed = False

    def close(self):
        self.closed = True


def _inject_fake_isaacsim(monkeypatch):
    """Inject fake ``isaacsim`` + runtime modules into sys.modules."""
    import numpy as np

    fake_isaacsim = types.ModuleType("isaacsim")
    fake_isaacsim.SimulationApp = FakeSimulationApp
    monkeypatch.setitem(sys.modules, "isaacsim", fake_isaacsim)

    fake_core = types.ModuleType("isaacsim.core")
    fake_core_api = types.ModuleType("isaacsim.core.api")
    fake_core_api.World = object  # any placeholder
    fake_core.api = fake_core_api
    monkeypatch.setitem(sys.modules, "isaacsim.core", fake_core)
    monkeypatch.setitem(sys.modules, "isaacsim.core.api", fake_core_api)

    fake_sensors = types.ModuleType("isaacsim.sensors")
    fake_camera_mod = types.ModuleType("isaacsim.sensors.camera")
    fake_camera_mod.Camera = object
    fake_sensors.camera = fake_camera_mod
    monkeypatch.setitem(sys.modules, "isaacsim.sensors", fake_sensors)
    monkeypatch.setitem(sys.modules, "isaacsim.sensors.camera", fake_camera_mod)

    fake_omni = types.ModuleType("omni")
    fake_omni_usd = types.ModuleType("omni.usd")
    fake_omni.usd = fake_omni_usd
    monkeypatch.setitem(sys.modules, "omni", fake_omni)
    monkeypatch.setitem(sys.modules, "omni.usd", fake_omni_usd)

    fake_pxr = types.ModuleType("pxr")
    fake_usdgeom = types.ModuleType("pxr.UsdGeom")
    fake_pxr.UsdGeom = fake_usdgeom
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)
    monkeypatch.setitem(sys.modules, "pxr.UsdGeom", fake_usdgeom)

    # numpy and PIL.Image already importable in the test environment.
    monkeypatch.setitem(sys.modules, "numpy", np)
    try:
        from PIL import Image
        monkeypatch.setitem(sys.modules, "PIL.Image", Image)
    except ImportError:
        pass


def test_bootstrap_import_ordering(monkeypatch, tmp_path):
    """SimulationApp is constructed before Isaac runtime modules are imported."""
    _inject_fake_isaacsim(monkeypatch)
    staging = tmp_path / "staging"
    staging.mkdir()

    config = _valid_config()
    with bootstrap_render(config, staging) as proxy:
        assert isinstance(proxy, SimulationAppProxy)
        assert isinstance(proxy.app, FakeSimulationApp)
        assert proxy.app.config["headless"] is True
        assert proxy.app.config["width"] == 600
        assert proxy.app.config["height"] == 450
        assert proxy.runtime.np is not None
        assert proxy.runtime.World is not None
    # App closed after context exit.
    assert proxy.app.closed


def test_bootstrap_closes_on_runtime_import_failure(monkeypatch, tmp_path):
    """App is closed even if runtime imports fail after app construction."""
    fake_pkg = types.ModuleType("isaacsim")

    constructed: list[FakeSimulationApp] = []

    class TrackingApp(FakeSimulationApp):
        def __init__(self, config=None, **kwargs):
            super().__init__(config, **kwargs)
            constructed.append(self)

    fake_pkg.SimulationApp = TrackingApp
    monkeypatch.setitem(sys.modules, "isaacsim", fake_pkg)

    # Make omni.usd import fail inside the context manager body (after app
    # construction but before yield). omni.usd is not needed for preflight.
    import builtins

    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "omni.usd":
            raise ImportError("injected omni.usd failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    staging = tmp_path / "staging"
    staging.mkdir()
    config = _valid_config()

    with pytest.raises(ImportError, match="injected omni.usd failure"):
        with bootstrap_render(config, staging):
            pass  # never reached

    assert len(constructed) == 1
    assert constructed[0].closed


def test_bootstrap_closes_on_exception_in_body(monkeypatch, tmp_path):
    """App is closed when an exception occurs in the with-body."""
    _inject_fake_isaacsim(monkeypatch)
    staging = tmp_path / "staging"
    staging.mkdir()
    config = _valid_config()

    proxy_ref = []
    with pytest.raises(RuntimeError, match="stage setup failure"):
        with bootstrap_render(config, staging) as proxy:
            proxy_ref.append(proxy)
            raise RuntimeError("stage setup failure")

    assert proxy_ref[0].app.closed


def test_bootstrap_closes_on_normal_exit(monkeypatch, tmp_path):
    """App is closed on normal context exit."""
    _inject_fake_isaacsim(monkeypatch)
    staging = tmp_path / "staging"
    staging.mkdir()
    config = _valid_config()

    with bootstrap_render(config, staging) as proxy:
        pass

    assert proxy.app.closed


def test_bootstrap_rejects_absent_staging(monkeypatch, tmp_path):
    """Bootstrap fails before app construction if staging is absent."""
    _inject_fake_isaacsim(monkeypatch)
    staging = tmp_path / "missing"
    config = _valid_config()

    with pytest.raises(FileNotFoundError, match="expected existing real directory"):
        with bootstrap_render(config, staging):
            pass  # never reached


def test_bootstrap_rejects_symlink_staging(monkeypatch, tmp_path):
    """Bootstrap fails before app construction if staging is a symlink."""
    _inject_fake_isaacsim(monkeypatch)
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    config = _valid_config()

    with pytest.raises(ValueError, match="refusing symlinked directory"):
        with bootstrap_render(config, link):
            pass


def test_bootstrap_preflight_before_app_construction(monkeypatch, tmp_path):
    """Sentinel preflight failure prevents SimulationApp construction."""
    fake_pkg = types.ModuleType("isaacsim")

    constructed = []

    class TrackingApp(FakeSimulationApp):
        def __init__(self, config=None, **kwargs):
            constructed.append(True)
            super().__init__(config, **kwargs)

    fake_pkg.SimulationApp = TrackingApp
    monkeypatch.setitem(sys.modules, "isaacsim", fake_pkg)

    staging = tmp_path / "staging"
    staging.mkdir()
    # 10.0 * 10000 = 100000 > 65535
    config = _valid_config(max_depth_m=10.0, depth_scale=10000.0)

    with pytest.raises(ValueError, match="exceeds 65535"):
        with bootstrap_render(config, staging):
            pass

    assert not constructed, "SimulationApp should not be constructed on preflight failure"
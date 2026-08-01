"""Render bootstrap and SimulationApp lifecycle management (Phase 4).

Isolate render startup/import ordering and guarantee app closure.

The bootstrap performs, in order:

1. Validate the ``RenderConfig`` (stdlib-only range checks already done in
   ``__post_init__``).
2. Require and validate the orchestrator-owned staging root (must exist as a
   real directory, not a symlink).
3. Preflight the depth sentinel for **both** modes before any staging write or
   ``SimulationApp`` construction.
4. Construct ``SimulationApp``.
5. Import Isaac runtime modules *after* app construction.
6. Yield the app and runtime module namespace to the caller.
7. Close the app in a ``finally`` block, including failures during runtime
   imports or stage setup.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

from sage3d.config import RenderConfig
from sage3d.publication import validate_real_directory


def validate_staging_root(staging_root: Path) -> None:
    """Require the orchestrator-owned staging root to be a real directory."""
    staging_root = Path(staging_root)
    if not os.path.lexists(staging_root):
        raise FileNotFoundError(f"staging root does not exist: {staging_root}")
    if os.path.islink(staging_root):
        raise ValueError(f"refusing symlinked staging root: {staging_root}")
    if not os.path.isdir(staging_root):
        raise NotADirectoryError(f"staging root is not a directory: {staging_root}")


def preflight_depth_sentinel(config: RenderConfig) -> None:
    """Preflight the depth sentinel for both modes before app construction.

    Calls ``render_processing.encoded_depth_sentinel`` which validates
    finite-positive ``max_depth_m``/``depth_scale`` and checks the scaled
    sentinel against 65535 before any ``SimulationApp`` is constructed.
    """
    import numpy as np  # local import: sentinel helper needs numpy

    from sage3d.render_processing import encoded_depth_sentinel

    # ponytail: preflight both modes; RGB also needs the sentinel because the
    # render summary records it and the same encode path is reused.
    encoded_depth_sentinel(config.max_depth_m, config.depth_scale)


@dataclass
class RenderRuntime:
    """Isaac runtime modules imported after SimulationApp construction."""

    np: object
    Image: object
    omni_usd: object
    World: object
    Camera: object
    UsdGeom: object


@contextmanager
def bootstrap_render(
    config: RenderConfig,
    staging_root: Path,
) -> "Iterator[SimulationAppProxy]":
    """Bootstrap render: validate → preflight → construct app → import runtime.

    Yields a proxy with ``app`` and ``runtime`` attributes.  Closes the app
    in a ``finally`` block regardless of success or failure (including
    failures during runtime imports or stage setup).
    """
    validate_staging_root(staging_root)
    preflight_depth_sentinel(config)

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "renderer": "RaytracedLighting",
            "width": config.width,
            "height": config.height,
        }
    )

    try:
        import numpy as np
        from PIL import Image
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.sensors.camera import Camera
        from pxr import UsdGeom

        runtime = RenderRuntime(
            np=np,
            Image=Image,
            omni_usd=omni.usd,
            World=World,
            Camera=Camera,
            UsdGeom=UsdGeom,
        )
        yield SimulationAppProxy(app=app, runtime=runtime)
    finally:
        app.close()


@dataclass
class SimulationAppProxy:
    """Proxy carrying the SimulationApp and imported runtime modules."""

    app: object
    runtime: RenderRuntime
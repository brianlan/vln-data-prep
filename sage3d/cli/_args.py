"""Shared argparse helpers for fisheye camera + scene arguments.

Package-safe (stdlib only). The legacy render and package CLIs both declare
the same fisheye arguments; :func:`add_fisheye_args` captures the shared set so
later wiring phases do not drift the CLI surface. :func:`add_scene_args`
captures the shared ``--sage-root`` + ``--scene`` pair introduced in Phase 2c.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_fisheye_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared OpenCV fisheye camera arguments to ``parser``."""
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--height", type=int, default=450)
    parser.add_argument(
        "--horizontal-fov-deg", type=float, default=180.0
    )
    parser.add_argument(
        "--fisheye-coefficients",
        type=float,
        nargs=4,
        metavar=("K1", "K2", "K3", "K4"),
        default=(0.1, 0.0, 0.0, 0.0),
    )


def add_scene_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--sage-root`` and ``--scene`` arguments.

    ``--sage-root`` is the SAGE3D dataset root directory. ``--scene`` is the
    numeric scene ID. Explicitly-provided asset overrides (``--interiorgs-root``,
    ``--usdz``, ``--collision-usd``) take higher priority than the
    ``--sage-root`` derived defaults.
    """
    parser.add_argument(
        "--sage-root",
        type=Path,
        default=Path("/ssd5/datasets/SAGE3D"),
        help="SAGE3D dataset root (default: /ssd5/datasets/SAGE3D)",
    )
    parser.add_argument(
        "--scene",
        required=True,
        help="Numeric SAGE3D scene ID",
    )
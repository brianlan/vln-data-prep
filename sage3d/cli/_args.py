"""Shared argparse helpers for fisheye camera + scene arguments.

Package-safe (stdlib only). The legacy render and package CLIs both declare
the same fisheye arguments; :func:`add_fisheye_args` captures the shared set so
later wiring phases do not drift the CLI surface.
"""

from __future__ import annotations

import argparse


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
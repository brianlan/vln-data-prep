#!/usr/bin/env python3
"""Render SAGE3D PointGoal trajectories with native Isaac Sim fisheye RGB/depth.

.. deprecated::
    Use ``python -m sage3d.cli.create_staging`` + ``python -m sage3d.cli.render
    --staging-root ...`` + ``python -m sage3d.cli.finalize_render`` for
    validated atomic publication. This shim preserves the legacy
    ``--output-dir`` surface as an explicitly **non-atomic** compatibility
    path: it maps that exact path to the new renderer's staging root and never
    invokes the finalizer.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

from sage3d.cli._args import add_fisheye_args, add_scene_args
from sage3d.cli.render import main as render_main
from sage3d.publication import validate_real_directory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    add_fisheye_args(parser)
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
    return parser.parse_args(argv)


def ensure_legacy_output_dir(path: Path) -> None:
    """Compatibility exception: exclusive exact-path creation when absent.

    If ``path`` is absent, create exactly that directory with ``os.mkdir``
    (never ``mkdtemp`` or ``mkdir -p``) and immediately verify it with
    ``lstat``. If ``path`` exists, it must already be a real directory (not a
    symlink); a later legacy modality may consume the existing valid
    other-modality state.
    """
    path = Path(path)
    if not os.path.lexists(path):
        path.mkdir()
    validate_real_directory(path)


def render_argv_from_args(args: argparse.Namespace) -> list[str]:
    """Map the legacy CLI surface to ``sage3d.cli.render`` argv.

    ``--output-dir`` becomes ``--staging-root``; every other argument is
    preserved so the two-process legacy sequence produces the same artifacts
    as before.
    """
    argv = [
        "--scene", args.scene,
        "--sage-root", str(args.sage_root),
        "--trajectory-dir", str(args.trajectory_dir),
        "--staging-root", str(args.output_dir),
        "--mode", args.mode,
        "--width", str(args.width),
        "--height", str(args.height),
        "--horizontal-fov-deg", str(args.horizontal_fov_deg),
        "--fisheye-coefficients", *(str(c) for c in args.fisheye_coefficients),
        "--max-depth-m", str(args.max_depth_m),
        "--min-depth-m", str(args.min_depth_m),
        "--depth-scale", str(args.depth_scale),
        "--settle-steps", str(args.settle_steps),
        "--startup-steps", str(args.startup_steps),
    ]
    if args.usdz is not None:
        argv += ["--usdz", str(args.usdz)]
    if args.collision_usd is not None:
        argv += ["--collision-usd", str(args.collision_usd)]
    return argv


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    warnings.warn(
        "render_fisheye_sage3d.py is deprecated and explicitly non-atomic; "
        "use 'python -m sage3d.cli.create_staging', "
        "'python -m sage3d.cli.render --staging-root', and "
        "'python -m sage3d.cli.finalize_render' for validated atomic "
        "publication. This shim never invokes the finalizer.",
        DeprecationWarning,
        stacklevel=2,
    )
    ensure_legacy_output_dir(args.output_dir)
    render_main(render_argv_from_args(args))


if __name__ == "__main__":
    sys.exit(main())

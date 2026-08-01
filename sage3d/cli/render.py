"""Render CLI: ``python -m sage3d.cli.render`` (Phase 4).

Isaac-lane entry point that renders exactly one modality (``rgb`` or
``depth``) into an orchestrator-allocated staging root. Accepts
``--staging-root`` (NOT ``--output-dir``) and requires the staging root to be
an existing real directory, verified with ``lstat``; the allocator and
finalizer enforce its sibling relationship to the final target.

Import-level package-safe: no Isaac modules are imported at module scope, so
``--help`` works under either interpreter. ``render_runtime.render``
bootstraps ``SimulationApp`` internally and closes it in a ``finally``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sage3d.artifacts import resolve_render_assets
from sage3d.cli._args import add_fisheye_args, add_scene_args
from sage3d.config import RenderConfig
from sage3d.publication import validate_real_directory
from sage3d.render_runtime import render


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one SAGE3D modality (rgb or depth) into a staging root.",
    )
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
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("rgb", "depth"),
        required=True,
        help=(
            "Render exactly one modality. Invoke the CLI twice so NuRec "
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # Fail fast before any Isaac import; render_runtime also enforces this.
    validate_real_directory(args.staging_root)
    assets = resolve_render_assets(
        args.scene,
        args.sage_root,
        usdz=args.usdz,
        collision_usd=args.collision_usd,
    )
    config = RenderConfig(
        mode=args.mode,
        width=args.width,
        height=args.height,
        horizontal_fov_deg=args.horizontal_fov_deg,
        fisheye_coefficients=tuple(args.fisheye_coefficients),
        max_depth_m=args.max_depth_m,
        min_depth_m=args.min_depth_m,
        depth_scale=args.depth_scale,
        settle_steps=args.settle_steps,
        startup_steps=args.startup_steps,
    )
    render(
        config,
        args.staging_root,
        scene_id=args.scene,
        usdz=assets.usdz,
        collision_usd=assets.collision_usd,
        trajectory_dir=args.trajectory_dir,
    )


if __name__ == "__main__":
    main()

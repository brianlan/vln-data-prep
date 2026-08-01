"""Package CLI: ``python -m sage3d.cli.package``.

Package-safe (stdlib + numpy + pyarrow). Thin wrapper around
:func:`sage3d.lerobot_dataset.package` that preserves the legacy
``package_lerobot_sage3d.py`` CLI surface (``--scene``, ``--trajectory-dir``,
``--rendered-dir``, ``--output-dir``, optional legacy camera expected-value
assertions, ``--fps``).

The CLI is non-destructive: the final output must be absent (or cleared by
the caller/shell), the dataset is built into an internally allocated sibling
staging directory, validated, and atomically renamed onto the target. Legacy
camera values are optional expected-value assertions checked by the staged
validator against the canonical depth summary; the manifest
``camera_height_m`` remains authoritative.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sage3d.config import PackageConfig
from sage3d.lerobot_dataset import package as package_dataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build, validate, and atomically publish a SAGE3D PointGoal "
            "dataset in the project-specific LeRobot-style layout."
        ),
    )
    parser.add_argument(
        "--scene",
        required=True,
        help="Numeric SAGE3D scene ID",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        required=True,
        help="Trajectory directory with manifest, episode npz, pointcloud",
    )
    parser.add_argument(
        "--rendered-dir",
        type=Path,
        required=True,
        help="Finalized render directory (rgb/depth frames + summaries)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Final package output directory (must be absent)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional expected image width (asserted against depth summary)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional expected image height (asserted against depth summary)",
    )
    parser.add_argument(
        "--horizontal-fov-deg",
        type=float,
        default=None,
        help="Optional expected horizontal FOV (asserted against depth summary)",
    )
    parser.add_argument(
        "--fisheye-coefficients",
        type=float,
        nargs=4,
        metavar=("K1", "K2", "K3", "K4"),
        default=None,
        help="Optional expected fisheye coefficients (asserted against summary)",
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=None,
        help="Optional expected camera height (asserted against manifest)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second recorded in info.json (default: 30)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = PackageConfig(
        fps=args.fps,
        trajectory_dir=args.trajectory_dir,
        rendered_dir=args.rendered_dir,
        output_dir=args.output_dir,
        scene_id=args.scene,
        width=args.width,
        height=args.height,
        horizontal_fov_deg=args.horizontal_fov_deg,
        fisheye_coefficients=(
            tuple(args.fisheye_coefficients)
            if args.fisheye_coefficients is not None
            else None
        ),
        camera_height=args.camera_height,
    )
    output = package_dataset(config)
    sys.stdout.write(f"{output.resolve()}\n")


if __name__ == "__main__":
    main()

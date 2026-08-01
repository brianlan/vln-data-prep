"""Render finalizer CLI: ``python -m sage3d.cli.finalize_render``.

Package-safe (stdlib + numpy + PIL). Validates a complete two-modality render
staging root against the full trajectory/render contract, then atomically
publishes the entire directory onto the absent final target via
:func:`~sage3d.publication.atomic_publish_directory`.

The finalizer alone accepts both the staging root and the final output path.
Any failure before the rename (incomplete/stale/invalid inventory, contract
violation, existing target, symlink, device mismatch) leaves the final target
absent and the staging root intact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sage3d.contract import validate_pipeline_contract
from sage3d.episode_arrays import load_episode
from sage3d.naming import parse_episode_filename
from sage3d.publication import (
    atomic_publish_directory,
    validate_real_directory,
)

# The exact complete two-modality inventory accepted at the staging root.
# Mirrors the union of the RGB and depth inventories in
# ``render_runtime.preflight_staging`` (which the package-safe finalizer must
# not import).
COMPLETE_INVENTORY = frozenset(
    {
        "observation.images.rgb",
        "rgb_render_summary.json",
        "observation.images.depth",
        "depth_render_summary.json",
        "render_summary.json",
    }
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_episodes(trajectory_dir: Path) -> dict[int, object]:
    episodes = {}
    for tf in sorted(trajectory_dir.glob("episode_*.npz")):
        episodes[parse_episode_filename(tf.name)] = load_episode(tf)
    return episodes


def require_complete_inventory(staging_root: Path) -> None:
    """Require the exact complete two-modality inventory in ``staging_root``.

    Raises ``RuntimeError`` if any expected entry is missing or any
    unrelated/partial entry is present, so finalization never publishes a
    partial or stale stage. ``staging_root`` itself must be a real directory
    by lstat.
    """
    validate_real_directory(staging_root)
    entries = {entry.name for entry in staging_root.iterdir()}
    if entries != COMPLETE_INVENTORY:
        missing = sorted(COMPLETE_INVENTORY - entries)
        extra = sorted(entries - COMPLETE_INVENTORY)
        raise RuntimeError(
            f"staging root does not have the exact complete render inventory: "
            f"missing={missing}, extra={extra} in {staging_root}"
        )


def finalize(
    *,
    scene_id: str,
    trajectory_dir: Path,
    staging_root: Path,
    output_dir: Path,
) -> None:
    """Validate the staged render and atomically publish it to ``output_dir``.

    Any failure (inventory, contract, existing target, symlink, device
    mismatch) leaves ``output_dir`` absent and ``staging_root`` intact.
    """
    require_complete_inventory(staging_root)
    manifest = _load_json(trajectory_dir / "trajectory_manifest.json")
    rgb_summary = _load_json(staging_root / "rgb_render_summary.json")
    canonical_depth_summary = _load_json(staging_root / "render_summary.json")
    depth_alias_summary = _load_json(staging_root / "depth_render_summary.json")
    episodes_by_id = _load_episodes(trajectory_dir)
    validate_pipeline_contract(
        expected_scene_id=scene_id,
        manifest=manifest,
        rgb_summary=rgb_summary,
        canonical_depth_summary=canonical_depth_summary,
        depth_alias_summary=depth_alias_summary,
        episodes_by_id=episodes_by_id,
        trajectory_dir=trajectory_dir,
        rendered_dir=staging_root,
        pointcloud_path=trajectory_dir / "pointcloud.ply",
    )
    atomic_publish_directory(staging_root, output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a complete two-modality render staging root against "
            "the trajectory/render contract and atomically publish it onto "
            "the absent final output directory."
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
        "--staging-root",
        type=Path,
        required=True,
        help="Complete render staging directory to validate and publish",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Final render output directory (must be absent)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    finalize(
        scene_id=args.scene,
        trajectory_dir=args.trajectory_dir,
        staging_root=args.staging_root,
        output_dir=args.output_dir,
    )
    sys.stdout.write(f"{args.output_dir.resolve()}\n")


if __name__ == "__main__":
    main()

"""Render staging allocator CLI: ``python -m sage3d.cli.create_staging``.

Package-safe thin wrapper around
:func:`~sage3d.publication.create_staging_directory` that prints exactly
one absolute allocated path to stdout (diagnostics go to stderr) and
refuses an existing final target via
:func:`~sage3d.publication.assert_target_absent`.

The orchestrator (shell or canonical harness) invokes this once, captures
the returned path, and passes that exact existing directory to the RGB and
depth render processes via ``--staging-root``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sage3d.publication import (
    assert_target_absent,
    create_staging_directory,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Allocate a shared render staging directory as a sibling of a "
            "final target. Prints the absolute staging path to stdout."
        ),
    )
    parser.add_argument(
        "--final-target",
        type=Path,
        required=True,
        help="Final render output directory (must be absent)",
    )
    parser.add_argument(
        "--prefix",
        default=".rendered.",
        help="Staging directory name prefix (default: .rendered.)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    assert_target_absent(args.final_target)
    staging = create_staging_directory(args.final_target, prefix=args.prefix)
    sys.stdout.write(f"{staging.resolve()}\n")


if __name__ == "__main__":
    main()
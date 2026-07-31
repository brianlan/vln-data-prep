"""Episode and frame filename helpers (stdlib only).

The generation, render, and package stages share one filename contract:

- episode npz: ``episode_{episode_index:06d}.npz``
- frame stems: ``episode_{episode_index:06d}_{frame_index:03d}``

Rendering and packaging append ``.jpg``/``.png`` to frame stems; the parse
helpers accept either a bare stem or a full filename with one suffix.
"""

from __future__ import annotations

from pathlib import Path


def episode_filename(episode_index: int) -> str:
    """Return the episode npz filename, e.g. ``episode_000000.npz``."""
    return f"episode_{int(episode_index):06d}.npz"


def parse_episode_filename(name: str) -> int:
    """Return the integer episode index encoded in ``name``.

    Accepts a bare stem (``episode_000000``) or a full filename
    (``episode_000000.npz``). Raises ``ValueError`` on any other shape.
    """
    stem = Path(name).stem if name.endswith(".npz") else name
    prefix, _, suffix = stem.partition("_")
    if prefix != "episode":
        raise ValueError(f"not an episode filename: {name!r}")
    if not suffix.isdigit():
        raise ValueError(f"episode index is not an integer: {name!r}")
    return int(suffix)


def frame_stem(episode_index: int, frame_index: int) -> str:
    """Return the shared frame stem, e.g. ``episode_000000_000``."""
    return f"episode_{int(episode_index):06d}_{int(frame_index):03d}"


def parse_frame_filename(name: str) -> tuple[int, int]:
    """Return ``(episode_index, frame_index)`` encoded in ``name``.

    Accepts a bare stem, a path, or a filename with a single suffix
    (e.g. ``episode_000000_000.jpg``).
    """
    stem = Path(name).stem if "." in Path(name).name else Path(name).name
    parts = stem.split("_")
    if len(parts) != 3 or parts[0] != "episode":
        raise ValueError(f"not a frame filename: {name!r}")
    if not (parts[1].isdigit() and parts[2].isdigit()):
        raise ValueError(f"frame indices are not integers: {name!r}")
    return int(parts[1]), int(parts[2])
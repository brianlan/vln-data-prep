"""Stage-specific SAGE3D asset resolution (stdlib + pathlib only).

Package-safe: no Isaac, cv2, scipy, trimesh, or PIL.

Phase 2c removes hardcoded asset paths from the monolithic scripts. Each stage
(generation, render) needs a different subset of SAGE3D assets:

- **Generation** needs the InteriorGS scene directory (for occupancy/structure
  maps) and the collision-mesh USD. It does **not** need the USDZ.
- **Render** needs the USDZ and the collision-mesh USD. It does **not** need
  the InteriorGS directory.

Both resolvers accept ``--sage-root`` (the SAGE3D dataset root) plus
``--scene`` (the numeric scene ID), and derive the default asset locations
from them. Explicitly-provided overrides (``--interiorgs-root``,
``--collision-usd``, ``--usdz``) take higher priority than the derived
defaults. ``resolve_scene_dir`` preserves the legacy exactly-one-match glob
behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenerationAssets:
    """Resolved assets for the generation stage."""

    scene_dir: Path
    collision_usd: Path


@dataclass(frozen=True)
class RenderAssets:
    """Resolved assets for the render stage."""

    usdz: Path
    collision_usd: Path


def resolve_scene_dir(interiorgs_root: Path, scene: str) -> Path:
    """Return the single InteriorGS directory matching ``*_{scene}``.

    Raises ``FileNotFoundError`` if the root does not exist, or ``RuntimeError``
    if zero or multiple matches are found. Preserves the legacy behavior from
    ``generate_sage3d_trajectories.py``.
    """
    matches = sorted(interiorgs_root.glob(f"*_{scene}"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one InteriorGS directory matching '*_{scene}' "
            f"under {interiorgs_root}, found {len(matches)}"
        )
    return matches[0]


def _default_collision_usd(sage_root: Path, scene: str) -> Path:
    return (
        sage_root / "Collision_Mesh" / "Collision_Mesh"
        / scene / f"{scene}_collision.usd"
    )


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {path}")
    return path


def resolve_generation_assets(
    scene: str,
    sage_root: Path,
    *,
    interiorgs_root: Path | None = None,
    collision_usd: Path | None = None,
) -> GenerationAssets:
    """Resolve generation-stage assets.

    Generation needs the InteriorGS scene directory and the collision-mesh USD.
    It does not need the USDZ.

    Override precedence (highest first):
    1. Explicitly-provided ``interiorgs_root`` / ``collision_usd``.
    2. Derived from ``sage_root``.
    """
    root = interiorgs_root if interiorgs_root is not None else sage_root / "InteriorGS"
    _require_dir(root, "InteriorGS root")
    scene_dir = resolve_scene_dir(root, scene)

    usd = collision_usd if collision_usd is not None else _default_collision_usd(sage_root, scene)
    _require_file(usd, "collision USD")

    return GenerationAssets(scene_dir=scene_dir, collision_usd=usd)


def resolve_render_assets(
    scene: str,
    sage_root: Path,
    *,
    usdz: Path | None = None,
    collision_usd: Path | None = None,
) -> RenderAssets:
    """Resolve render-stage assets.

    Render needs the USDZ and the collision-mesh USD. It does not need the
    InteriorGS directory.

    Override precedence (highest first):
    1. Explicitly-provided ``usdz`` / ``collision_usd``.
    2. Derived from ``sage_root``.
    """
    resolved_usdz = usdz if usdz is not None else sage_root / "InteriorGS_usdz" / f"{scene}.usdz"
    _require_file(resolved_usdz, "USDZ")

    resolved_usd = collision_usd if collision_usd is not None else _default_collision_usd(sage_root, scene)
    _require_file(resolved_usd, "collision USD")

    return RenderAssets(usdz=resolved_usdz, collision_usd=resolved_usd)
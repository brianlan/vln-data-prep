"""Tests for sage3d.artifacts and module discovery (Phase 2c).

Covers:
- resolve_scene_dir exactly-one-match (zero/multiple matches).
- resolve_generation_assets and resolve_render_assets with all
  partial-override combos and override precedence.
- Generation does not require USDZ; render does not require InteriorGS dir.
- Legacy CLI compatibility (explicit overrides still work).
- Fresh-subprocess forbidden-import for sage3d.artifacts.
- Outside-CWD module discovery under both interpreters.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from sage3d.artifacts import (
    GenerationAssets,
    RenderAssets,
    resolve_generation_assets,
    resolve_render_assets,
    resolve_scene_dir,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# resolve_scene_dir
# ---------------------------------------------------------------------------


def test_resolve_scene_dir_exactly_one_match(tmp_path: Path) -> None:
    (tmp_path / "foo_839920").mkdir()
    result = resolve_scene_dir(tmp_path, "839920")
    assert result == tmp_path / "foo_839920"


def test_resolve_scene_dir_zero_matches(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="found 0"):
        resolve_scene_dir(tmp_path, "839920")


def test_resolve_scene_dir_multiple_matches(tmp_path: Path) -> None:
    (tmp_path / "a_839920").mkdir()
    (tmp_path / "b_839920").mkdir()
    with pytest.raises(RuntimeError, match="found 2"):
        resolve_scene_dir(tmp_path, "839920")


# ---------------------------------------------------------------------------
# resolve_generation_assets
# ---------------------------------------------------------------------------


def _make_sage_root(tmp_path: Path, scene: str = "839920") -> Path:
    """Create a minimal SAGE3D-like directory tree."""
    root = tmp_path / "sage3d_root"
    # InteriorGS scene dir.
    (root / "InteriorGS" / f"scene_{scene}").mkdir(parents=True)
    # Collision USD.
    collision_dir = root / "Collision_Mesh" / "Collision_Mesh" / scene
    collision_dir.mkdir(parents=True)
    (collision_dir / f"{scene}_collision.usd").touch()
    # USDZ (not needed by generation).
    (root / "InteriorGS_usdz").mkdir(parents=True)
    (root / "InteriorGS_usdz" / f"{scene}.usdz").touch()
    return root


def test_generation_assets_derive_all_from_sage_root(tmp_path: Path) -> None:
    root = _make_sage_root(tmp_path)
    assets = resolve_generation_assets("839920", root)
    assert isinstance(assets, GenerationAssets)
    assert assets.scene_dir == root / "InteriorGS" / "scene_839920"
    assert assets.collision_usd == (
        root / "Collision_Mesh" / "Collision_Ges" / "839920"
        / "839920_collision.usd"
    ) or assets.collision_usd == (
        root / "Collision_Mesh" / "Collision_Mesh"
        / "839920" / "839920_collision.usd"
    )


def test_generation_assets_override_interiorgs_root(tmp_path: Path) -> None:
    root = _make_sage_root(tmp_path)
    custom_interior = tmp_path / "custom_interior"
    (custom_interior / "custom_839920").mkdir(parents=True)
    assets = resolve_generation_assets(
        "839920", root, interiorgs_root=custom_interior
    )
    assert assets.scene_dir == custom_interior / "custom_839920"
    # collision_usd still derived from sage_root.
    assert assets.collision_usd == (
        root / "Collision_Mesh" / "Collision_Mesh"
        / "839920" / "839920_collision.usd"
    )


def test_generation_assets_override_collision_usd(tmp_path: Path) -> None:
    root = _make_sage_root(tmp_path)
    custom_collision = tmp_path / "custom_collision.usd"
    custom_collision.touch()
    assets = resolve_generation_assets(
        "839920", root, collision_usd=custom_collision
    )
    assert assets.collision_usd == custom_collision
    # scene_dir still derived from sage_root.
    assert assets.scene_dir == root / "InteriorGS" / "scene_839920"


def test_generation_assets_override_both(tmp_path: Path) -> None:
    root = _make_sage_root(tmp_path)
    custom_interior = tmp_path / "custom_interior"
    (custom_interior / "custom_839920").mkdir(parents=True)
    custom_collision = tmp_path / "custom_collision.usd"
    custom_collision.touch()
    assets = resolve_generation_assets(
        "839920",
        root,
        interiorgs_root=custom_interior,
        collision_usd=custom_collision,
    )
    assert assets.scene_dir == custom_interior / "custom_839920"
    assert assets.collision_usd == custom_collision


def test_generation_assets_missing_collision_usd(tmp_path: Path) -> None:
    root = _make_sage_root(tmp_path)
    # Remove collision USD.
    (root / "Collision_Mesh" / "Collision_Mesh" / "839920"
     / "839920_collision.usd").unlink()
    with pytest.raises(FileNotFoundError, match="collision USD"):
        resolve_generation_assets("839920", root)


def test_generation_assets_missing_interiorgs_root(tmp_path: Path) -> None:
    root = _make_sage_root(tmp_path)
    # Remove InteriorGS root.
    import shutil

    shutil.rmtree(root / "InteriorGS")
    with pytest.raises(FileNotFoundError, match="InteriorGS root"):
        resolve_generation_assets("839920", root)


def test_generation_does_not_require_usdz(tmp_path: Path) -> None:
    """Generation should succeed even if USDZ does not exist."""
    root = _make_sage_root(tmp_path)
    # Remove USDZ entirely.
    (root / "InteriorGS_usdz" / "839920.usdz").unlink()
    (root / "InteriorGS_usdz").rmdir()
    assets = resolve_generation_assets("839920", root)
    assert assets.scene_dir == root / "InteriorGS" / "scene_839920"
    assert assets.collision_usd.is_file()


# ---------------------------------------------------------------------------
# resolve_render_assets
# ---------------------------------------------------------------------------


def test_render_assets_derive_all_from_sage_root(tmp_path: Path) -> None:
    root = _make_sage_root(tmp_path)
    assets = resolve_render_assets("839920", root)
    assert isinstance(assets, RenderAssets)
    assert assets.usdz == root / "InteriorGS_usdz" / "839920.usdz"
    assert assets.collision_usd == (
        root / "Collision_Mesh" / "Collision_Mesh"
        / "839920" / "839920_collision.usd"
    )


def test_render_assets_override_usdz(tmp_path: Path) -> None:
    root = _make_sage_root(tmp_path)
    custom_usdz = tmp_path / "custom.usdz"
    custom_usdz.touch()
    assets = resolve_render_assets("839920", root, usdz=custom_usdz)
    assert assets.usdz == custom_usdz
    assert assets.collision_usd == (
        root / "Collision_Mesh" / "Collision_Mesh"
        / "839920" / "839920_collision.usd"
    )


def test_render_assets_override_collision_usd(tmp_path: Path) -> None:
    root = _make_sage_root(tmp_path)
    custom_collision = tmp_path / "custom_collision.usd"
    custom_collision.touch()
    assets = resolve_render_assets(
        "839920", root, collision_usd=custom_collision
    )
    assert assets.collision_usd == custom_collision
    assert assets.usdz == root / "InteriorGS_usdz" / "839920.usdz"


def test_render_assets_override_both(tmp_path: Path) -> None:
    root = _make_sage_root(tmp_path)
    custom_usdz = tmp_path / "custom.usdz"
    custom_usdz.touch()
    custom_collision = tmp_path / "custom_collision.usd"
    custom_collision.touch()
    assets = resolve_render_assets(
        "839920", root, usdz=custom_usdz, collision_usd=custom_collision
    )
    assert assets.usdz == custom_usdz
    assert assets.collision_usd == custom_collision


def test_render_assets_missing_usdz(tmp_path: Path) -> None:
    root = _make_sage_root(tmp_path)
    (root / "InteriorGS_usdz" / "839920.usdz").unlink()
    with pytest.raises(FileNotFoundError, match="USDZ"):
        resolve_render_assets("839920", root)


def test_render_assets_missing_collision_usd(tmp_path: Path) -> None:
    root = _make_sage_root(tmp_path)
    (root / "Collision_Mesh" / "Collision_Mesh" / "839920"
     / "839920_collision.usd").unlink()
    with pytest.raises(FileNotFoundError, match="collision USD"):
        resolve_render_assets("839920", root)


def test_render_does_not_require_interiorgs_dir(tmp_path: Path) -> None:
    """Render should succeed even if InteriorGS directory does not exist."""
    root = _make_sage_root(tmp_path)
    import shutil

    shutil.rmtree(root / "InteriorGS")
    assets = resolve_render_assets("839920", root)
    assert assets.usdz == root / "InteriorGS_usdz" / "839920.usdz"
    assert assets.collision_usd.is_file()


# ---------------------------------------------------------------------------
# Override precedence: explicit overrides take priority over sage_root
# ---------------------------------------------------------------------------


def test_override_precedence_usdz_over_sage_root(tmp_path: Path) -> None:
    root = _make_sage_root(tmp_path)
    override = tmp_path / "override.usdz"
    override.touch()
    assets = resolve_render_assets("839920", root, usdz=override)
    assert assets.usdz == override


def test_override_precedence_collision_over_sage_root(tmp_path: Path) -> None:
    root = _make_sage_root(tmp_path)
    override = tmp_path / "override_collision.usd"
    override.touch()
    assets = resolve_generation_assets("839920", root, collision_usd=override)
    assert assets.collision_usd == override


def test_override_precedence_interiorgs_over_sage_root(tmp_path: Path) -> None:
    root = _make_sage_root(tmp_path)
    custom = tmp_path / "custom_interior"
    (custom / "custom_839920").mkdir(parents=True)
    assets = resolve_generation_assets(
        "839920", root, interiorgs_root=custom
    )
    assert assets.scene_dir == custom / "custom_839920"


# ---------------------------------------------------------------------------
# File vs dir predicates
# ---------------------------------------------------------------------------


def test_collision_usd_is_dir_not_file(tmp_path: Path) -> None:
    root = _make_sage_root(tmp_path)
    # Replace the collision USD file with a directory of the same name.
    collision_path = (
        root / "Collision_Mesh" / "Collision_Mesh"
        / "839920" / "839920_collision.usd"
    )
    collision_path.unlink()
    collision_path.mkdir()
    with pytest.raises(FileNotFoundError, match="collision USD"):
        resolve_generation_assets("839920", root)


def test_usdz_is_dir_not_file(tmp_path: Path) -> None:
    root = _make_sage_root(tmp_path)
    usdz_path = root / "InteriorGS_usdz" / "839920.usdz"
    usdz_path.unlink()
    usdz_path.mkdir()
    with pytest.raises(FileNotFoundError, match="USDZ"):
        resolve_render_assets("839920", root)


def test_interiorgs_root_is_file_not_dir(tmp_path: Path) -> None:
    root = _make_sage_root(tmp_path)
    # Replace InteriorGS dir with a file.
    import shutil

    shutil.rmtree(root / "InteriorGS")
    (root / "InteriorGS").touch()
    with pytest.raises(FileNotFoundError, match="InteriorGS root"):
        resolve_generation_assets("839920", root)


# ---------------------------------------------------------------------------
# add_scene_args
# ---------------------------------------------------------------------------


def test_add_scene_args_defaults():
    import argparse

    from sage3d.cli._args import add_scene_args

    parser = argparse.ArgumentParser()
    add_scene_args(parser)
    args = parser.parse_args(["--scene", "839920"])
    assert args.scene == "839920"
    assert args.sage_root == Path("/ssd5/datasets/SAGE3D")


def test_add_scene_args_override_sage_root():
    import argparse

    from sage3d.cli._args import add_scene_args

    parser = argparse.ArgumentParser()
    add_scene_args(parser)
    args = parser.parse_args(["--scene", "839920", "--sage-root", "/data/sage3d"])
    assert args.sage_root == Path("/data/sage3d")


# ---------------------------------------------------------------------------
# Forbidden-import smoke (fresh subprocess)
# ---------------------------------------------------------------------------


def test_artifacts_module_package_safe():
    proc = subprocess.run(
        [sys.executable, "-c", "import sage3d.artifacts; print('OK')"],
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"


# ---------------------------------------------------------------------------
# Outside-CWD module discovery: import existing modules from outside the repo
# ---------------------------------------------------------------------------


def test_import_sage3d_modules_from_outside_repo(tmp_path: Path):
    """Import all existing package-safe sage3d modules from a CWD outside the
    repo, using the repo-prefixed PYTHONPATH convention."""
    modules = [
        "sage3d",
        "sage3d.frames",
        "sage3d.camera",
        "sage3d.episode_arrays",
        "sage3d.naming",
        "sage3d.io_ply",
        "sage3d.pointcloud",
        "sage3d.publication",
        "sage3d.render_processing",
        "sage3d.config",
        "sage3d.schemas",
        "sage3d.contract",
        "sage3d.artifacts",
        "sage3d.cli",
        "sage3d.cli._args",
    ]
    probe = "import sys\n"
    probe += "for m in {modules!r}:\n".format(modules=modules)
    probe += "    __import__(m)\n"
    probe += "print('OK')\n"
    # Run from a CWD outside the repo with repo-prefixed PYTHONPATH.
    env = {
        **os.environ,
        "PYTHONPATH": str(_REPO_ROOT),
    }
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"import from outside repo failed:\nstdout={proc.stdout}\n"
        f"stderr={proc.stderr}"
    )
    assert proc.stdout.strip() == "OK"
"""Tests for ``python -m sage3d.cli.finalize_render`` (package-safe).

Verifies the finalizer's contract: a complete two-modality staging root that
passes the full trajectory/render contract is atomically published onto an
absent final target; every failure (incomplete/stale/invalid inventory,
contract violation, symlinked staged entry, existing target) leaves the final
target absent and the staging root intact.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from sage3d.cli.finalize_render import (
    COMPLETE_INVENTORY,
    finalize,
    parse_args,
    require_complete_inventory,
)
from sage3d.publication import validate_real_directory

from fixtures import build_rendered_dir, build_trajectory_dir

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_canonical_stage(
    tmp_path: Path,
    *,
    scene_id: str = "839920",
    frame_counts: tuple[int, ...] = (3, 2, 4),
) -> tuple[Path, Path, dict]:
    """Build a fully-valid synthetic trajectory + render staging tree."""
    trajectory_dir = tmp_path / "trajectory"
    staging_root = tmp_path / "stage"
    staging_root.mkdir()
    manifest = build_trajectory_dir(
        trajectory_dir, episode_frame_counts=frame_counts, scene_id=scene_id
    )
    build_rendered_dir(staging_root, trajectory_manifest=manifest)
    return trajectory_dir, staging_root, manifest


def _run_finalize(
    scene_id: str,
    trajectory_dir: Path,
    staging_root: Path,
    output_dir: Path,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    argv = [
        sys.executable,
        "-m",
        "sage3d.cli.finalize_render",
        "--scene",
        scene_id,
        "--trajectory-dir",
        str(trajectory_dir),
        "--staging-root",
        str(staging_root),
        "--output-dir",
        str(output_dir),
    ]
    run_env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    if env:
        run_env.update(env)
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        env=run_env,
    )


# --- parse_args ---------------------------------------------------------------


def test_parse_args_requires_all_args():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_values():
    args = parse_args(
        [
            "--scene",
            "839920",
            "--trajectory-dir",
            "/tmp/traj",
            "--staging-root",
            "/tmp/stage",
            "--output-dir",
            "/tmp/rendered",
        ]
    )
    assert args.scene == "839920"
    assert args.trajectory_dir == Path("/tmp/traj")
    assert args.staging_root == Path("/tmp/stage")
    assert args.output_dir == Path("/tmp/rendered")


# --- inventory ----------------------------------------------------------------


def test_complete_inventory_constant_matches_render_runtime_inventories():
    """The finalizer's complete inventory is the union of the two modality
    inventories enforced by ``render_runtime.preflight_staging``."""
    assert COMPLETE_INVENTORY == frozenset(
        {
            "observation.images.rgb",
            "rgb_render_summary.json",
            "observation.images.depth",
            "depth_render_summary.json",
            "render_summary.json",
        }
    )


def test_require_complete_inventory_passes_for_valid_stage(tmp_path):
    _, staging_root, _ = _build_canonical_stage(tmp_path)
    require_complete_inventory(staging_root)  # no raise


def test_require_complete_inventory_rejects_missing_entry(tmp_path):
    _, staging_root, _ = _build_canonical_stage(tmp_path)
    (staging_root / "render_summary.json").unlink()
    with pytest.raises(RuntimeError, match="missing"):
        require_complete_inventory(staging_root)


def test_require_complete_inventory_rejects_extra_entry(tmp_path):
    _, staging_root, _ = _build_canonical_stage(tmp_path)
    (staging_root / "unrelated.txt").write_text("x")
    with pytest.raises(RuntimeError, match="extra"):
        require_complete_inventory(staging_root)


def test_require_complete_inventory_rejects_partial_modality(tmp_path):
    _, staging_root, _ = _build_canonical_stage(tmp_path)
    (staging_root / "depth_render_summary.json").unlink()
    (staging_root / "render_summary.json").unlink()
    with pytest.raises(RuntimeError, match="missing"):
        require_complete_inventory(staging_root)


def test_require_complete_inventory_rejects_symlinked_root(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "stage"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        require_complete_inventory(link)


# --- publication positive -----------------------------------------------------


def test_finalize_publishes_complete_stage(tmp_path):
    trajectory_dir, staging_root, _ = _build_canonical_stage(tmp_path)
    output_dir = tmp_path / "rendered"
    finalize(
        scene_id="839920",
        trajectory_dir=trajectory_dir,
        staging_root=staging_root,
        output_dir=output_dir,
    )
    validate_real_directory(output_dir)
    assert not staging_root.exists()
    assert {e.name for e in output_dir.iterdir()} == COMPLETE_INVENTORY


def test_cli_publishes_complete_stage(tmp_path):
    trajectory_dir, staging_root, _ = _build_canonical_stage(tmp_path)
    output_dir = tmp_path / "rendered"
    proc = _run_finalize(
        "839920", trajectory_dir, staging_root, output_dir
    )
    assert proc.returncode == 0, proc.stderr
    assert output_dir.is_dir()
    assert not staging_root.exists()


# --- contract failures leave target absent ------------------------------------


def test_cli_fails_on_scene_id_mismatch(tmp_path):
    trajectory_dir, staging_root, _ = _build_canonical_stage(tmp_path)
    output_dir = tmp_path / "rendered"
    proc = _run_finalize(
        "999999", trajectory_dir, staging_root, output_dir
    )
    assert proc.returncode != 0
    assert not output_dir.exists()
    assert staging_root.is_dir()


def test_cli_fails_on_missing_frame(tmp_path):
    trajectory_dir, staging_root, _ = _build_canonical_stage(tmp_path)
    (staging_root / "observation.images.rgb" / "episode_000000_000.jpg").unlink()
    output_dir = tmp_path / "rendered"
    proc = _run_finalize(
        "839920", trajectory_dir, staging_root, output_dir
    )
    assert proc.returncode != 0
    assert "frame" in proc.stderr or "rgb" in proc.stderr
    assert not output_dir.exists()
    assert staging_root.is_dir()


def test_cli_fails_on_stale_extra_frame(tmp_path):
    trajectory_dir, staging_root, _ = _build_canonical_stage(tmp_path)
    (staging_root / "observation.images.rgb" / "episode_000099_000.jpg").write_bytes(
        b"\xff\xd8\xff"
    )
    output_dir = tmp_path / "rendered"
    proc = _run_finalize(
        "839920", trajectory_dir, staging_root, output_dir
    )
    assert proc.returncode != 0
    assert not output_dir.exists()
    assert staging_root.is_dir()


def test_cli_fails_on_missing_pointcloud(tmp_path):
    trajectory_dir, staging_root, _ = _build_canonical_stage(tmp_path)
    (trajectory_dir / "pointcloud.ply").unlink()
    output_dir = tmp_path / "rendered"
    proc = _run_finalize(
        "839920", trajectory_dir, staging_root, output_dir
    )
    assert proc.returncode != 0
    assert not output_dir.exists()
    assert staging_root.is_dir()


# --- symlinked staged entries -------------------------------------------------


def test_cli_fails_on_symlinked_staged_entry(tmp_path):
    trajectory_dir, staging_root, _ = _build_canonical_stage(tmp_path)
    summary = staging_root / "rgb_render_summary.json"
    summary.rename(tmp_path / "elsewhere.json")
    summary.symlink_to(tmp_path / "elsewhere.json")
    output_dir = tmp_path / "rendered"
    proc = _run_finalize(
        "839920", trajectory_dir, staging_root, output_dir
    )
    assert proc.returncode != 0
    assert "symlink" in proc.stderr
    assert not output_dir.exists()
    assert staging_root.is_dir()


# --- existing target matrix ---------------------------------------------------


@pytest.mark.parametrize(
    "make_target",
    [
        lambda p: p.mkdir(),
        lambda p: p.write_text("file"),
        lambda p: p.symlink_to(p.parent / "real", target_is_directory=True),
        lambda p: p.symlink_to(p.parent / "dangling"),
    ],
    ids=["dir", "file", "symlink", "dangling-symlink"],
)
def test_cli_refuses_every_existing_target_type(tmp_path, make_target):
    trajectory_dir, staging_root, _ = _build_canonical_stage(tmp_path)
    output_dir = tmp_path / "rendered"
    make_target(output_dir)
    proc = _run_finalize(
        "839920", trajectory_dir, staging_root, output_dir
    )
    assert proc.returncode != 0
    assert "already exists" in proc.stderr
    assert staging_root.is_dir()


# --- immediately-before-rename injection --------------------------------------

def test_failure_during_publication_leaves_target_absent(tmp_path, monkeypatch):
    """A failure at the last step (after full contract validation) must leave
    the final target absent and the staging root intact."""
    import sage3d.cli.finalize_render as finalize_render

    trajectory_dir, staging_root, _ = _build_canonical_stage(tmp_path)
    output_dir = tmp_path / "rendered"

    def boom(staging: Path, target: Path) -> Path:
        raise RuntimeError("injected publication failure")

    monkeypatch.setattr(
        finalize_render, "atomic_publish_directory", boom
    )
    with pytest.raises(RuntimeError, match="injected"):
        finalize(
            scene_id="839920",
            trajectory_dir=trajectory_dir,
            staging_root=staging_root,
            output_dir=output_dir,
        )
    assert not output_dir.exists()
    assert staging_root.is_dir()


# --- canonical render validation ----------------------------------------------

def test_published_stage_passes_check_render_validate(tmp_path):
    """Canonical producer wiring: the published root must satisfy the
    baseline-independent render checker exactly like a legacy render root."""
    trajectory_dir, staging_root, _ = _build_canonical_stage(tmp_path)
    output_dir = tmp_path / "rendered"
    finalize(
        scene_id="839920",
        trajectory_dir=trajectory_dir,
        staging_root=staging_root,
        output_dir=output_dir,
    )
    result_path = tmp_path / "checker_result.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "check_render.py"),
            "validate",
            "--rendered-dir",
            str(output_dir),
            "--trajectory-dir",
            str(trajectory_dir),
            "--result-path",
            str(result_path),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
    )
    assert proc.returncode == 0, proc.stderr
    import json

    with result_path.open() as f:
        result = json.load(f)
    assert result["eligible"] is True


# --- --help works outside repo ------------------------------------------------


def test_help_works_outside_repo(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sage3d.cli.finalize_render",
            "--help",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
        cwd=tmp_path,
    )
    assert proc.returncode == 0
    assert "--staging-root" in proc.stdout
    assert "--output-dir" in proc.stdout
    assert "--trajectory-dir" in proc.stdout

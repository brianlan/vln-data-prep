"""Tests for the generation CLI shim and module entry point (Phase 3e).

Covers:
- ``python -m sage3d.cli.generate --help`` works from outside the repo.
- Invalid arguments exit with code 2.
- The legacy ``generate_sage3d_trajectories.py`` shim produces identical
  artifacts to the module CLI.
- The trajectory target is absent at CLI entry (no pre-creation).
- ``--help`` from outside CWD via ``PYTHONPATH``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ISAAC_PYTHON = os.environ.get("SAGE3D_ISAAC_PYTHON", "/ssd4/envs/isaac_sim_py311/bin/python")
SAGE3D_ROOT = os.environ.get("SAGE3D_ROOT", "/ssd5/datasets/SAGE3D")
SCENE_ID = "839920"
SEED = "20260720"
EPISODES = "3"


def _run_cli(args: list[str], cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run ``python -m sage3d.cli.generate`` with the given args."""
    full_env = os.environ.copy()
    full_env["PYTHONPATH"] = str(REPO_ROOT)
    if env:
        full_env.update(env)
    return subprocess.run(
        [ISAAC_PYTHON, "-m", "sage3d.cli.generate"] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=full_env,
    )


def _run_legacy(args: list[str], cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    """Run the legacy ``generate_sage3d_trajectories.py`` shim."""
    full_env = os.environ.copy()
    full_env["PYTHONPATH"] = str(REPO_ROOT)
    if env:
        full_env.update(env)
    return subprocess.run(
        [ISAAC_PYTHON, str(REPO_ROOT / "generate_sage3d_trajectories.py")] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=full_env,
    )


# --- --help works from outside the repo --------------------------------------

def test_help_works_from_outside_repo(tmp_path):
    """``python -m sage3d.cli.generate --help`` exits 0 from a different cwd."""
    result = _run_cli(["--help"], cwd=tmp_path)
    assert result.returncode == 0
    assert "usage:" in result.stdout


# --- invalid argument exit code ---------------------------------------------

def test_invalid_argument_exits_2(tmp_path):
    """Unknown argument exits with code 2 (argparse convention)."""
    result = _run_cli(["--bad-arg"], cwd=tmp_path)
    assert result.returncode == 2
    assert "unrecognized" in result.stderr.lower() or "error" in result.stderr.lower()


def test_missing_required_args_exits_2(tmp_path):
    """Missing required --scene and --output-dir exits with code 2."""
    result = _run_cli([], cwd=tmp_path)
    assert result.returncode == 2
    assert "required" in result.stderr.lower()


# --- absent target at CLI entry ---------------------------------------------

def test_trajectory_target_absent_at_entry(tmp_path):
    """The CLI creates the target itself via atomic publication; the shell no
    longer pre-creates ``TRAJECTORY_DIR``.  Verify the target does not need
    to exist before the CLI runs, and the CLI creates it atomically."""
    if not Path(SAGE3D_ROOT).is_dir():
        pytest.skip("SAGE3D_ROOT not available")
    target = tmp_path / "trajectories"
    assert not target.exists()
    result = _run_cli(
        [
            "--scene", SCENE_ID,
            "--output-dir", str(target),
            "--episodes", EPISODES,
            "--seed", SEED,
        ],
        cwd=tmp_path,
        env={"SAGE3D_ROOT": SAGE3D_ROOT},
    )
    assert result.returncode == 0, result.stderr
    assert target.is_dir()
    # All 7 artifacts present.
    files = sorted(f.name for f in target.iterdir() if f.is_file())
    assert len(files) == int(EPISODES) + 4


def test_existing_target_is_refused(tmp_path):
    """If the target already exists, the CLI refuses (atomic publication)."""
    if not Path(SAGE3D_ROOT).is_dir():
        pytest.skip("SAGE3D_ROOT not available")
    target = tmp_path / "trajectories"
    target.mkdir()
    result = _run_cli(
        [
            "--scene", SCENE_ID,
            "--output-dir", str(target),
            "--episodes", EPISODES,
            "--seed", SEED,
        ],
        cwd=tmp_path,
        env={"SAGE3D_ROOT": SAGE3D_ROOT},
    )
    assert result.returncode != 0
    assert "already exists" in result.stderr


# --- legacy shim parity -------------------------------------------------------

def test_legacy_shim_produces_identical_artifacts(tmp_path):
    """The legacy ``generate_sage3d_trajectories.py`` shim produces the same
    artifacts as ``python -m sage3d.cli.generate``."""
    if not Path(SAGE3D_ROOT).is_dir():
        pytest.skip("SAGE3D_ROOT not available")
    target_module = tmp_path / "module_out" / "trajectories"
    target_legacy = tmp_path / "legacy_out" / "trajectories"
    target_module.parent.mkdir()
    target_legacy.parent.mkdir()

    args = [
        "--scene", SCENE_ID,
        "--output-dir", str(target_module),
        "--episodes", EPISODES,
        "--seed", SEED,
    ]
    result_mod = _run_cli(args, cwd=tmp_path, env={"SAGE3D_ROOT": SAGE3D_ROOT})
    assert result_mod.returncode == 0, result_mod.stderr

    args_legacy = [
        "--scene", SCENE_ID,
        "--output-dir", str(target_legacy),
        "--episodes", EPISODES,
        "--seed", SEED,
    ]
    result_legacy = _run_legacy(args_legacy, cwd=tmp_path, env={"SAGE3D_ROOT": SAGE3D_ROOT})
    assert result_legacy.returncode == 0, result_legacy.stderr

    # Compare sha256 of every artifact.
    import hashlib

    def _digests(root: Path) -> dict[str, str]:
        out = {}
        for f in sorted(root.iterdir()):
            if f.is_file():
                h = hashlib.sha256()
                h.update(f.read_bytes())
                out[f.name] = h.hexdigest()
        return out

    assert _digests(target_module) == _digests(target_legacy)
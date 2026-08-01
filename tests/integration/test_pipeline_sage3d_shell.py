"""Full SAGE3D pipeline shell integration test (Phase 6, issue #28).

Runs the real ``run_pipeline_sage3d.sh`` end-to-end from a CWD outside the
repository with the pinned scene, then verifies the packaged dataset passes
``scripts/check_package.py validate``. Requires Isaac Sim, a compatible NVIDIA
GPU, and the external SAGE3D assets; skipped otherwise.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.sage3d_gpu

REPO_ROOT = Path(__file__).resolve().parents[2]
SHELL = REPO_ROOT / "run_pipeline_sage3d.sh"
ISAAC_PYTHON = Path(
    os.environ.get("SAGE3D_ISAAC_PYTHON", "/ssd4/envs/isaac_sim_py311/bin/python")
)
PACKAGE_PYTHON = Path(
    os.environ.get("SAGE3D_PACKAGE_PYTHON", "/ssd4/envs/vln_data_prep_py311/bin/python")
)
SAGE3D_ROOT = Path(os.environ.get("SAGE3D_ROOT", "/ssd5/datasets/SAGE3D"))
SCENE_ID = "839920"


def _assets_ready() -> bool:
    if not ISAAC_PYTHON.exists() or not PACKAGE_PYTHON.exists():
        return False
    if not SAGE3D_ROOT.is_dir():
        return False
    return (SAGE3D_ROOT / "InteriorGS_usdz" / f"{SCENE_ID}.usdz").exists()


def test_full_pipeline_from_outside_repo(tmp_path):
    """Run the pinned full pipeline from an unrelated CWD and validate the
    packaged output with the baseline-independent checker."""
    if not _assets_ready():
        pytest.skip("SAGE3D assets or interpreters not available")

    output_root = tmp_path / "output"
    output_root.mkdir()
    work_root = tmp_path / "work"

    env = os.environ.copy()
    env["SAGE3D_ISAAC_PYTHON"] = str(ISAAC_PYTHON)
    env["SAGE3D_PACKAGE_PYTHON"] = str(PACKAGE_PYTHON)
    env["PYTHONPATH"] = str(REPO_ROOT)

    result = subprocess.run(
        [
            "bash",
            str(SHELL),
            SCENE_ID,
            "--episodes", "2",
            "--output-root", str(output_root),
            "--work-root", str(work_root),
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert result.returncode == 0, (
        f"shell failed (exit {result.returncode}):\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "DONE:" in result.stdout

    dataset_dir = output_root / SCENE_ID
    trajectory_dir = work_root / SCENE_ID / "trajectories"
    rendered_dir = work_root / SCENE_ID / "rendered"
    assert dataset_dir.is_dir()

    check = subprocess.run(
        [
            str(PACKAGE_PYTHON),
            str(REPO_ROOT / "scripts" / "check_package.py"),
            "validate",
            "--dataset-dir", str(dataset_dir),
            "--trajectory-dir", str(trajectory_dir),
            "--rendered-dir", str(rendered_dir),
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, (
        f"check_package failed (exit {check.returncode}):\n"
        f"stdout={check.stdout}\nstderr={check.stderr}"
    )
    assert "[check_package:validate] ELIGIBLE" in check.stdout

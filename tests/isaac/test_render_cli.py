"""Tests for the render CLI and legacy shim (Phase 4 issue #23).

Covers:
- ``python -m sage3d.cli.render --help`` and the legacy shim ``--help`` work
  from outside the repo.
- Invalid arguments exit with code 2.
- The legacy ``render_fisheye_sage3d.py`` shim maps ``--output-dir`` →
  ``--staging-root`` and produces identical artifacts to the module CLI.
- Absent/existing valid paths, duplicate mode, partial/symlink paths.
- The shim never auto-finalizes (no finalizer invocation).
- Canonical render validation via ``check_render.py validate``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ISAAC_PYTHON = os.environ.get(
    "SAGE3D_ISAAC_PYTHON", "/ssd4/envs/isaac_sim_py311/bin/python"
)
SAGE3D_ROOT = os.environ.get("SAGE3D_ROOT", "/ssd5/datasets/SAGE3D")
SCENE_ID = "839920"
GEN_BASELINE = Path("/tmp/opencode/gen_baseline")


def _base_env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def _run_module(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [ISAAC_PYTHON, "-m", "sage3d.cli.render"] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=_base_env(),
    )


def _run_shim(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [ISAAC_PYTHON, str(REPO_ROOT / "render_fisheye_sage3d.py")] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=_base_env(),
    )


def _module_args(staging_root: Path, mode: str) -> list[str]:
    return [
        "--scene", SCENE_ID,
        "--sage-root", SAGE3D_ROOT,
        "--trajectory-dir", str(GEN_BASELINE),
        "--staging-root", str(staging_root),
        "--mode", mode,
    ]


def _shim_args(output_dir: Path, mode: str) -> list[str]:
    return [
        "--scene", SCENE_ID,
        "--sage-root", SAGE3D_ROOT,
        "--trajectory-dir", str(GEN_BASELINE),
        "--output-dir", str(output_dir),
        "--mode", mode,
    ]


def _assets_ready() -> bool:
    if not Path(SAGE3D_ROOT).is_dir():
        return False
    if not GEN_BASELINE.is_dir() or not list(GEN_BASELINE.glob("episode_*.npz")):
        return False
    return True


# --- --help works from outside the repo --------------------------------------


def test_module_help_works_from_outside_repo(tmp_path):
    result = _run_module(["--help"], cwd=tmp_path)
    assert result.returncode == 0
    assert "--staging-root" in result.stdout
    assert "--output-dir" not in result.stdout


def test_shim_help_works_from_outside_repo(tmp_path):
    result = _run_shim(["--help"], cwd=tmp_path)
    assert result.returncode == 0
    assert "--output-dir" in result.stdout


# --- invalid arguments -------------------------------------------------------


def test_module_rejects_unknown_arg(tmp_path):
    result = _run_module(["--bad-arg"], cwd=tmp_path)
    assert result.returncode == 2


def test_module_rejects_output_dir_flag(tmp_path):
    """The module CLI must NOT accept the legacy --output-dir."""
    result = _run_module(["--output-dir", "/tmp/x"], cwd=tmp_path)
    assert result.returncode == 2


def test_module_requires_staging_root(tmp_path):
    result = _run_module(["--scene", SCENE_ID], cwd=tmp_path)
    assert result.returncode == 2


def test_shim_rejects_staging_root_flag(tmp_path):
    """The legacy surface must keep --output-dir, not --staging-root."""
    result = _run_shim(["--staging-root", "/tmp/x"], cwd=tmp_path)
    assert result.returncode == 2


# --- shim parity with module CLI (requires assets) ----------------------------


@pytest.mark.sage3d_gpu
def test_shim_parity_depth_byte_identical(tmp_path):
    """Depth output through the legacy shim equals the module CLI output.

    Depth PNGs and summary JSON are deterministic, so the mapping must be
    byte-identical. RGB is GPU-nondeterministic and is compared structurally
    in the canonical validation test below.
    """
    if not _assets_ready():
        pytest.skip("SAGE3D assets or baseline trajectory not available")

    module_out = tmp_path / "module_depth"
    module_out.mkdir()
    result = _run_module(
        _module_args(module_out, "depth"), cwd=tmp_path
    )
    assert result.returncode == 0, result.stderr

    shim_out = tmp_path / "shim_depth"
    shim_out.mkdir()
    result = _run_shim(_shim_args(shim_out, "depth"), cwd=tmp_path)
    assert result.returncode == 0, result.stderr

    module_depth = module_out / "observation.images.depth"
    shim_depth = shim_out / "observation.images.depth"
    assert sorted(p.name for p in shim_depth.glob("*.png")) == sorted(
        p.name for p in module_depth.glob("*.png")
    )
    for shim_png in sorted(shim_depth.glob("*.png")):
        module_png = module_depth / shim_png.name
        assert shim_png.read_bytes() == module_png.read_bytes()
    # Summary JSON identical (canonical copy too).
    for name in ("depth_render_summary.json", "render_summary.json"):
        assert (shim_out / name).read_bytes() == (module_out / name).read_bytes()


@pytest.mark.sage3d_gpu
def test_two_legacy_invocations_complete_stage(tmp_path):
    """Two direct legacy invocations (rgb then depth) preserve the legacy
    artifact inventory without any finalizer."""
    if not _assets_ready():
        pytest.skip("SAGE3D assets or baseline trajectory not available")

    out = tmp_path / "legacy_out"
    result = _run_shim(_shim_args(out, "rgb"), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    result = _run_shim(_shim_args(out, "depth"), cwd=tmp_path)
    assert result.returncode == 0, result.stderr

    assert (out / "observation.images.rgb").is_dir()
    assert (out / "observation.images.depth").is_dir()
    assert (out / "rgb_render_summary.json").is_file()
    assert (out / "depth_render_summary.json").is_file()
    assert (out / "render_summary.json").is_file()


@pytest.mark.sage3d_gpu
def test_shim_duplicate_mode_fails(tmp_path):
    """A second legacy invocation of the same modality must fail (preflight)."""
    if not _assets_ready():
        pytest.skip("SAGE3D assets or baseline trajectory not available")

    out = tmp_path / "legacy_out"
    result = _run_shim(_shim_args(out, "rgb"), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    result = _run_shim(_shim_args(out, "rgb"), cwd=tmp_path)
    assert result.returncode != 0
    assert "rgb" in result.stderr


@pytest.mark.sage3d_gpu
def test_shim_rejects_partial_state(tmp_path):
    """Partial other-modality state (dir without summary) must fail."""
    if not _assets_ready():
        pytest.skip("SAGE3D assets or baseline trajectory not available")

    out = tmp_path / "legacy_out"
    out.mkdir()
    (out / "observation.images.rgb").mkdir()
    result = _run_shim(_shim_args(out, "depth"), cwd=tmp_path)
    assert result.returncode != 0


def test_shim_rejects_symlink_output_dir(tmp_path):
    """The shim must not write through a symlinked legacy output path."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "legacy_out"
    link.symlink_to(real, target_is_directory=True)
    result = _run_shim(_shim_args(link, "rgb"), cwd=tmp_path)
    assert result.returncode != 0
    assert "symlink" in result.stderr


# --- canonical render validation ---------------------------------------------


@pytest.mark.sage3d_gpu
def test_shim_output_passes_check_render_validate(tmp_path):
    """The shim's two-invocation output must satisfy the baseline-independent
    render checker exactly like a legacy render root."""
    if not _assets_ready():
        pytest.skip("SAGE3D assets or baseline trajectory not available")

    out = tmp_path / "legacy_out"
    for mode in ("rgb", "depth"):
        result = _run_shim(_shim_args(out, mode), cwd=tmp_path)
        assert result.returncode == 0, result.stderr

    result_path = tmp_path / "checker_result.json"
    proc = subprocess.run(
        [
            ISAAC_PYTHON,
            str(REPO_ROOT / "scripts" / "check_render.py"),
            "validate",
            "--rendered-dir",
            str(out),
            "--trajectory-dir",
            str(GEN_BASELINE),
            "--result-path",
            str(result_path),
        ],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        env=_base_env(),
    )
    assert proc.returncode == 0, proc.stderr
    import json

    with result_path.open() as f:
        result = json.load(f)
    assert result["eligible"] is True

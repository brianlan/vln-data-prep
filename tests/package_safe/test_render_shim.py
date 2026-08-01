"""Package-safe tests for the legacy render shim (issue #23).

Covers the shim's pure logic without constructing a ``SimulationApp``:
legacy CLI surface, ``--output-dir`` → ``--staging-root`` mapping, exclusive
exact-path directory creation, lstat validation, the deprecation warning, and
delegation to ``sage3d.cli.render``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import render_fisheye_sage3d as shim
from sage3d.publication import validate_real_directory

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --- legacy CLI surface -------------------------------------------------------


def test_parse_args_keeps_legacy_output_dir():
    args = shim.parse_args(
        [
            "--scene", "839920",
            "--trajectory-dir", "/tmp/traj",
            "--output-dir", "/tmp/out",
            "--mode", "rgb",
        ]
    )
    assert args.output_dir == Path("/tmp/out")
    assert args.mode == "rgb"


def test_parse_args_rejects_staging_root_flag():
    """The legacy surface must not silently accept the new --staging-root."""
    with pytest.raises(SystemExit):
        shim.parse_args(
            [
                "--scene", "839920",
                "--trajectory-dir", "/tmp/traj",
                "--staging-root", "/tmp/out",
                "--mode", "rgb",
            ]
        )


def test_parse_args_requires_legacy_args():
    with pytest.raises(SystemExit):
        shim.parse_args([])


# --- output-dir mapping -------------------------------------------------------


def test_render_argv_maps_output_dir_to_staging_root():
    args = shim.parse_args(
        [
            "--scene", "839920",
            "--sage-root", "/sage",
            "--usdz", "/x/usdz",
            "--collision-usd", "/x/collision.usd",
            "--trajectory-dir", "/traj",
            "--output-dir", "/out",
            "--mode", "depth",
            "--width", "100",
            "--height", "80",
            "--horizontal-fov-deg", "150.0",
            "--fisheye-coefficients", "0.2", "0.1", "0.0", "0.0",
            "--max-depth-m", "8.0",
            "--min-depth-m", "0.1",
            "--depth-scale", "5000.0",
            "--settle-steps", "5",
            "--startup-steps", "20",
        ]
    )
    argv = shim.render_argv_from_args(args)
    assert "--staging-root" in argv
    assert argv[argv.index("--staging-root") + 1] == "/out"
    assert "--output-dir" not in argv
    assert argv[argv.index("--scene") + 1] == "839920"
    assert argv[argv.index("--mode") + 1] == "depth"
    assert argv[argv.index("--usdz") + 1] == "/x/usdz"
    assert argv[argv.index("--fisheye-coefficients") + 1 :][:4] == [
        "0.2", "0.1", "0.0", "0.0",
    ]


def test_render_argv_preserves_default_asset_resolution():
    """Without explicit overrides, the mapped argv omits --usdz/--collision-usd
    so the module CLI resolves them from --sage-root."""
    args = shim.parse_args(
        [
            "--scene", "839920",
            "--trajectory-dir", "/traj",
            "--output-dir", "/out",
            "--mode", "rgb",
        ]
    )
    argv = shim.render_argv_from_args(args)
    assert "--usdz" not in argv
    assert "--collision-usd" not in argv
    assert "--sage-root" in argv


# --- exclusive exact-path creation --------------------------------------------


def test_ensure_creates_absent_path_exactly(tmp_path):
    target = tmp_path / "legacy_out"
    shim.ensure_legacy_output_dir(target)
    validate_real_directory(target)
    assert not target.is_symlink()


def test_ensure_accepts_existing_valid_dir(tmp_path):
    target = tmp_path / "legacy_out"
    target.mkdir()
    (target / "observation.images.rgb").mkdir()
    shim.ensure_legacy_output_dir(target)  # no raise
    validate_real_directory(target)


def test_ensure_rejects_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "legacy_out"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        shim.ensure_legacy_output_dir(link)


def test_ensure_rejects_missing_parent(tmp_path):
    """Exclusive exact-path os.mkdir: the parent must already exist."""
    target = tmp_path / "missing_parent" / "legacy_out"
    with pytest.raises(FileNotFoundError):
        shim.ensure_legacy_output_dir(target)


# --- warning and delegation ---------------------------------------------------


def test_main_emits_actionable_warning_and_delegates(tmp_path, monkeypatch):
    """The shim warns about deprecation/non-atomicity, creates the legacy
    output dir exclusively, maps --output-dir → --staging-root, and delegates
    to sage3d.cli.render without ever invoking the finalizer."""
    called: list[list[str]] = []

    def fake_render_main(argv: list[str]) -> None:
        called.append(argv)

    monkeypatch.setattr(shim, "render_main", fake_render_main)
    target = tmp_path / "legacy_out"

    with pytest.warns(DeprecationWarning, match="non-atomic"):
        shim.main(
            [
                "--scene", "839920",
                "--trajectory-dir", str(tmp_path / "traj"),
                "--output-dir", str(target),
                "--mode", "rgb",
            ]
        )

    validate_real_directory(target)
    assert len(called) == 1
    argv = called[0]
    assert "--staging-root" in argv
    assert argv[argv.index("--staging-root") + 1] == str(target)
    assert "--output-dir" not in argv
    assert "finalize" not in " ".join(argv)

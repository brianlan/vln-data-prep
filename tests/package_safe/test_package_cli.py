"""Tests for ``python -m sage3d.cli.package`` and the legacy rollout shim.

Covers the Phase 5d CLI contract: module CLI parsing and publication, legacy
shim parity (same artifact tree, matching exit codes), --help and outside-CWD
operation, default/non-default metadata truthfulness, checker mode parity, and
the supplemental ``prepare_trajectories.py`` known-consumer smoke.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fixtures import build_rendered_dir, build_trajectory_dir
from sage3d.cli.package import parse_args
from sage3d.publication import validate_real_directory

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = _REPO_ROOT
_LEGACY_SHIM = _REPO_ROOT / "package_lerobot_sage3d.py"
_CHECKER = _REPO_ROOT / "scripts" / "check_package.py"
_PREPARE = _REPO_ROOT / "prepare_trajectories.py"

_DEFAULT_ARGS = [
    "--scene", "839920",
    "--width", "600",
    "--height", "450",
    "--horizontal-fov-deg", "180.0",
    "--fisheye-coefficients", "0.1", "0.0", "0.0", "0.0",
    "--camera-height", "0.6",
]


def _build_sources(tmp_path: Path, frame_counts=(3, 2)):
    traj = tmp_path / "traj"
    manifest = build_trajectory_dir(traj, episode_frame_counts=frame_counts)
    rendered = tmp_path / "rendered"
    build_rendered_dir(rendered, trajectory_manifest=manifest)
    return traj, rendered, manifest


def _run_cli(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    run_env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    return subprocess.run(
        [sys.executable, "-m", "sage3d.cli.package", *args],
        capture_output=True,
        text=True,
        env=run_env,
        cwd=cwd,
    )


def _run_legacy(args: list[str]) -> subprocess.CompletedProcess:
    run_env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    return subprocess.run(
        [sys.executable, str(_LEGACY_SHIM), *args],
        capture_output=True,
        text=True,
        env=run_env,
    )


def _base_args(tmp_path: Path, output_dir: Path) -> list[str]:
    traj, rendered, _ = _build_sources(tmp_path)
    return _DEFAULT_ARGS + [
        "--trajectory-dir", str(traj),
        "--rendered-dir", str(rendered),
        "--output-dir", str(output_dir),
    ]


def _info(output_dir: Path) -> dict:
    with (output_dir / "meta" / "info.json").open("r", encoding="utf-8") as f:
        return json.load(f)


# --- parse_args ---------------------------------------------------------------


def test_parse_args_requires_core_args():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_values(tmp_path):
    args = parse_args(
        _DEFAULT_ARGS
        + [
            "--trajectory-dir", str(tmp_path / "traj"),
            "--rendered-dir", str(tmp_path / "rendered"),
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert args.scene == "839920"
    assert args.trajectory_dir == tmp_path / "traj"
    assert args.rendered_dir == tmp_path / "rendered"
    assert args.output_dir == tmp_path / "out"
    assert args.width == 600
    assert args.height == 450
    assert args.horizontal_fov_deg == 180.0
    assert args.fisheye_coefficients == [0.1, 0.0, 0.0, 0.0]
    assert args.camera_height == 0.6
    assert args.fps == 30


def test_parse_args_defaults_optional_camera_fields(tmp_path):
    args = parse_args(
        [
            "--scene", "839920",
            "--trajectory-dir", str(tmp_path / "traj"),
            "--rendered-dir", str(tmp_path / "rendered"),
            "--output-dir", str(tmp_path / "out"),
        ]
    )
    assert args.width is None
    assert args.height is None
    assert args.horizontal_fov_deg is None
    assert args.fisheye_coefficients is None
    assert args.camera_height is None


# --- positive publication -----------------------------------------------------


def test_cli_publishes_dataset(tmp_path):
    output_dir = tmp_path / "out"
    proc = _run_cli(_base_args(tmp_path, output_dir))
    assert proc.returncode == 0, proc.stderr
    validate_real_directory(output_dir)
    assert (output_dir / "meta" / "info.json").is_file()
    assert (output_dir / "data" / "chunk-000" / "episode_000000.parquet").is_file()
    assert proc.stdout.strip() == str(output_dir.resolve())


def test_cli_publishes_without_optional_camera_args(tmp_path):
    traj, rendered, _ = _build_sources(tmp_path)
    output_dir = tmp_path / "out"
    args = [
        "--scene", "839920",
        "--trajectory-dir", str(traj),
        "--rendered-dir", str(rendered),
        "--output-dir", str(output_dir),
    ]
    proc = _run_cli(args)
    assert proc.returncode == 0, proc.stderr
    validate_real_directory(output_dir)


def test_cli_refuses_existing_output(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    proc = _run_cli(_base_args(tmp_path, output_dir))
    assert proc.returncode != 0
    assert "already exists" in proc.stderr
    assert output_dir.is_dir()


# --- contract failures leave target absent ------------------------------------


def test_cli_fails_on_scene_id_mismatch(tmp_path):
    output_dir = tmp_path / "out"
    traj, rendered, _ = _build_sources(tmp_path)
    args = [
        "--scene", "999999",
        "--trajectory-dir", str(traj),
        "--rendered-dir", str(rendered),
        "--output-dir", str(output_dir),
    ]
    proc = _run_cli(args)
    assert proc.returncode != 0
    assert not output_dir.exists()


def test_cli_fails_on_missing_pointcloud(tmp_path):
    output_dir = tmp_path / "out"
    traj, rendered, _ = _build_sources(tmp_path)
    (traj / "pointcloud.ply").unlink()
    args = [
        "--scene", "839920",
        "--trajectory-dir", str(traj),
        "--rendered-dir", str(rendered),
        "--output-dir", str(output_dir),
    ]
    proc = _run_cli(args)
    assert proc.returncode != 0
    assert not output_dir.exists()


def test_cli_fails_on_camera_value_mismatch(tmp_path):
    output_dir = tmp_path / "out"
    traj, rendered, _ = _build_sources(tmp_path)
    args = [
        "--scene", "839920",
        "--width", "999",
        "--trajectory-dir", str(traj),
        "--rendered-dir", str(rendered),
        "--output-dir", str(output_dir),
    ]
    proc = _run_cli(args)
    assert proc.returncode != 0
    assert not output_dir.exists()


# --- --help and outside-CWD ---------------------------------------------------


def test_help_works_outside_repo(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "sage3d.cli.package", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
        cwd=tmp_path,
    )
    assert proc.returncode == 0
    assert "--trajectory-dir" in proc.stdout
    assert "--rendered-dir" in proc.stdout
    assert "--output-dir" in proc.stdout
    assert "--fisheye-coefficients" in proc.stdout


def test_cli_publishes_from_outside_cwd(tmp_path):
    output_dir = tmp_path / "out"
    proc = _run_cli(_base_args(tmp_path, output_dir), cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    validate_real_directory(output_dir)


# --- legacy shim parity -------------------------------------------------------


def test_legacy_shim_publishes_same_tree(tmp_path):
    """The rollout shim must produce the same artifact tree and exit code as
    the module CLI when run with the same arguments."""
    out_cli = tmp_path / "out_cli"
    proc_cli = _run_cli(_base_args(tmp_path, out_cli))
    assert proc_cli.returncode == 0, proc_cli.stderr

    out_legacy = tmp_path / "out_legacy"
    proc_legacy = _run_legacy(_base_args(tmp_path, out_legacy))
    assert proc_legacy.returncode == 0, proc_legacy.stderr
    validate_real_directory(out_legacy)

    # Same info.json metadata.
    assert _info(out_cli) == _info(out_legacy)
    # Same deterministic tree digests via compare-golden.
    proc = subprocess.run(
        [
            sys.executable, str(_CHECKER), "compare-golden",
            "--dataset-dir", str(out_cli),
            "--trajectory-dir", str(tmp_path / "traj"),
            "--rendered-dir", str(tmp_path / "rendered"),
            "--baseline-dir", str(out_legacy),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
    )
    assert proc.returncode == 0, proc.stderr


def test_legacy_shim_exit_codes_match_on_failure(tmp_path):
    """Both the module CLI and the shim exit non-zero on a contract failure
    and leave the target absent."""
    output_dir = tmp_path / "out"
    traj, rendered, _ = _build_sources(tmp_path)
    (traj / "pointcloud.ply").unlink()
    args = [
        "--scene", "839920",
        "--trajectory-dir", str(traj),
        "--rendered-dir", str(rendered),
        "--output-dir", str(output_dir),
    ]
    proc_cli = _run_cli(args)
    proc_legacy = _run_legacy(args)
    assert proc_cli.returncode != 0
    assert proc_legacy.returncode != 0
    assert not output_dir.exists()


# --- metadata truthfulness ----------------------------------------------------


def test_cli_default_metadata(tmp_path):
    output_dir = tmp_path / "out"
    proc = _run_cli(_base_args(tmp_path, output_dir))
    assert proc.returncode == 0, proc.stderr
    info = _info(output_dir)
    assert info["fps"] == 30
    assert info["scene_id"] == "839920"
    assert info["depth_format"] == "uint16_meters_x_10000"
    assert info["image_width"] == 600
    assert info["image_height"] == 450


def test_cli_non_default_depth_scale_truthful(tmp_path):
    """A non-default depth scale must produce a truthful depth_format string."""
    traj, _, _ = _build_sources(tmp_path)
    rendered = tmp_path / "rendered"
    # Rebuild the render root with a non-default scale.
    import shutil

    shutil.rmtree(rendered)
    build_rendered_dir(
        rendered, trajectory_manifest=build_trajectory_dir(
            tmp_path / "traj2", episode_frame_counts=(3, 2)
        ), depth_scale=5000.0,
    )
    # The manifest used for rendering must match the trajectory dir used for
    # packaging; rebuild the trajectory to keep scene/episode ids aligned.
    traj2 = tmp_path / "traj2"
    output_dir = tmp_path / "out"
    args = [
        "--scene", "839920",
        "--trajectory-dir", str(traj2),
        "--rendered-dir", str(rendered),
        "--output-dir", str(output_dir),
    ]
    proc = _run_cli(args)
    assert proc.returncode == 0, proc.stderr
    info = _info(output_dir)
    assert info["depth_format"] == "uint16_meters_x_5000"


# --- checker mode parity ------------------------------------------------------


def test_checker_validate_passes_on_cli_output(tmp_path):
    output_dir = tmp_path / "out"
    proc = _run_cli(_base_args(tmp_path, output_dir))
    assert proc.returncode == 0, proc.stderr
    proc = subprocess.run(
        [
            sys.executable, str(_CHECKER), "validate",
            "--dataset-dir", str(output_dir),
            "--trajectory-dir", str(tmp_path / "traj"),
            "--rendered-dir", str(tmp_path / "rendered"),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
    )
    assert proc.returncode == 0, proc.stderr
    assert "ELIGIBLE" in proc.stdout


def test_checker_compare_golden_passes_on_rebuilt_baseline(tmp_path):
    """compare-golden against a reference package built on the same sources."""
    output_dir = tmp_path / "out"
    proc = _run_cli(_base_args(tmp_path, output_dir))
    assert proc.returncode == 0, proc.stderr
    baseline = tmp_path / "baseline"
    traj, rendered, _ = _build_sources(tmp_path)
    from fixtures import build_packaged_dataset

    build_packaged_dataset(
        baseline, trajectory_dir=traj, rendered_dir=rendered, scene_id="839920"
    )
    proc = subprocess.run(
        [
            sys.executable, str(_CHECKER), "compare-golden",
            "--dataset-dir", str(output_dir),
            "--trajectory-dir", str(traj),
            "--rendered-dir", str(rendered),
            "--baseline-dir", str(baseline),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
    )
    assert proc.returncode == 0, proc.stderr


# --- known-consumer smoke -----------------------------------------------------


def test_prepare_trajectories_consumer_smoke(tmp_path):
    """Supplemental (non-authoritative) smoke: ``prepare_trajectories.py`` can
    read the shared Parquet path plus ``observation.camera_extrinsic`` and
    ``action`` columns from a CLI-produced package."""
    output_dir = tmp_path / "out"
    proc = _run_cli(_base_args(tmp_path, output_dir))
    assert proc.returncode == 0, proc.stderr
    prep_out = tmp_path / "prep"
    proc = subprocess.run(
        [
            sys.executable, str(_PREPARE),
            "--traj_dir", str(output_dir),
            "--output_dir", str(prep_out),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
    )
    assert proc.returncode == 0, proc.stderr
    npz_files = sorted(prep_out.glob("episode_*.npz"))
    assert len(npz_files) == 2
    data = np.load(npz_files[0])
    assert "extrinsic" in data
    assert "actions" in data
    assert data["actions"].shape[1:] == (4, 4)

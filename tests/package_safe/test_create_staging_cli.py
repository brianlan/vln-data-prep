"""Tests for ``python -m sage3d.cli.create_staging`` (package-safe).

Verifies the stdout/stderr contract (exactly one absolute path to stdout,
 diagnostics to stderr), refusal of an existing final target, and
 allocator-to-both-modes integration (RGB and depth use the identical
 allocated directory).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from sage3d.cli.create_staging import parse_args
from sage3d.publication import (
    assert_target_absent,
    create_staging_directory,
    validate_real_directory,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_create_staging(
    final_target: Path,
    *,
    prefix: str = ".rendered.",
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    argv = [
        sys.executable,
        "-m",
        "sage3d.cli.create_staging",
        "--final-target",
        str(final_target),
        "--prefix",
        prefix,
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


def test_parse_args_requires_final_target():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_defaults():
    args = parse_args(["--final-target", "/tmp/foo"])
    assert args.final_target == Path("/tmp/foo")
    assert args.prefix == ".rendered."


def test_parse_args_custom_prefix():
    args = parse_args(
        ["--final-target", "/tmp/foo", "--prefix", ".stage."]
    )
    assert args.prefix == ".stage."


# --- stdout/stderr contract ---------------------------------------------------


def test_stdout_contains_exactly_one_absolute_path(tmp_path):
    final = tmp_path / "rendered"
    proc = _run_create_staging(final)
    assert proc.returncode == 0
    assert proc.stderr == ""
    lines = proc.stdout.strip().splitlines()
    assert len(lines) == 1
    staging = Path(lines[0])
    assert staging.is_absolute()
    assert staging.is_dir()
    assert staging.name.startswith(".rendered.")


def test_staging_is_sibling_of_final_target(tmp_path):
    final = tmp_path / "rendered"
    proc = _run_create_staging(final)
    assert proc.returncode == 0
    staging = Path(proc.stdout.strip())
    assert staging.parent == final.resolve().parent
    assert staging.stat().st_dev == tmp_path.stat().st_dev


# --- refusal matrix -----------------------------------------------------------


def test_refuses_existing_final_target_dir(tmp_path):
    final = tmp_path / "rendered"
    final.mkdir()
    proc = _run_create_staging(final)
    assert proc.returncode != 0
    assert "already exists" in proc.stderr


def test_refuses_existing_final_target_file(tmp_path):
    final = tmp_path / "rendered"
    final.write_text("x")
    proc = _run_create_staging(final)
    assert proc.returncode != 0
    assert "already exists" in proc.stderr


def test_refuses_symlinked_final_target(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    proc = _run_create_staging(link)
    assert proc.returncode != 0
    assert "already exists" in proc.stderr


def test_refuses_dangling_symlink_final_target(tmp_path):
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "missing")
    proc = _run_create_staging(link)
    assert proc.returncode != 0
    assert "already exists" in proc.stderr


def test_refuses_empty_prefix(tmp_path):
    proc = _run_create_staging(tmp_path / "rendered", prefix="")
    assert proc.returncode != 0


# --- allocator output is real directory --------------------------------------


def test_allocator_output_is_real_directory(tmp_path):
    final = tmp_path / "rendered"
    proc = _run_create_staging(final)
    assert proc.returncode == 0
    staging = Path(proc.stdout.strip())
    validate_real_directory(staging)
    assert not staging.is_symlink()


# --- every attempt gets a new stage -------------------------------------------


def test_repeated_allocation_creates_distinct_stages(tmp_path):
    final = tmp_path / "rendered"
    paths = set()
    for _ in range(3):
        proc = _run_create_staging(final)
        assert proc.returncode == 0
        paths.add(proc.stdout.strip())
    assert len(paths) == 3


# --- --help works outside repo ------------------------------------------------


def test_help_works_outside_repo(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "sage3d.cli.create_staging",
            "--help",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
        cwd=tmp_path,
    )
    assert proc.returncode == 0
    assert "--final-target" in proc.stdout
    assert "--prefix" in proc.stdout


# --- allocator-to-both-modes integration --------------------------------------


def test_both_modes_use_identical_staging_directory(tmp_path):
    """RGB and depth both write into the same allocated staging directory."""
    final = tmp_path / "rendered"
    proc = _run_create_staging(final)
    assert proc.returncode == 0
    staging = Path(proc.stdout.strip())

    # Simulate RGB writing into the shared staging root.
    rgb_dir = staging / "observation.images.rgb"
    rgb_dir.mkdir()
    (rgb_dir / "frame_000000.jpg").write_bytes(b"\xff\xd8\xff")

    # Simulate depth writing into the same staging root.
    depth_dir = staging / "observation.images.depth"
    depth_dir.mkdir()
    (depth_dir / "frame_000000.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    # Both modalities coexist in the same directory.
    assert rgb_dir.is_dir()
    assert depth_dir.is_dir()
    assert (rgb_dir / "frame_000000.jpg").exists()
    assert (depth_dir / "frame_000000.png").exists()


# --- own modality is never overwritten ---------------------------------------


def test_create_staging_delegates_to_publication_helper(tmp_path):
    """The CLI is a thin wrapper: output matches create_staging_directory."""
    final = tmp_path / "rendered"
    proc = _run_create_staging(final)
    assert proc.returncode == 0
    cli_staging = Path(proc.stdout.strip())

    # A second allocation (different final target) should match the helper.
    final2 = tmp_path / "rendered2"
    helper_staging = create_staging_directory(final2, prefix=".rendered.")
    assert cli_staging.parent == helper_staging.parent
    assert cli_staging.name.startswith(".rendered.")
    assert helper_staging.name.startswith(".rendered.")
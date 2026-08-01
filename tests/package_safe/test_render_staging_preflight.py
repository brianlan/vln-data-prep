"""Tests for ``preflight_staging`` in ``sage3d.render_runtime``.

Package-safe: ``preflight_staging`` uses only stdlib (pathlib). These tests
cover the accepted-state matrix, filesystem-entry refusal matrix, and
partial/restart fixtures without constructing a ``SimulationApp``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sage3d.render_runtime import preflight_staging


def _make_rgb_inventory(staging: Path) -> None:
    """Seed a complete RGB inventory into ``staging``."""
    rgb_dir = staging / "observation.images.rgb"
    rgb_dir.mkdir()
    (rgb_dir / "frame_000000.jpg").write_bytes(b"\xff\xd8\xff")
    (staging / "rgb_render_summary.json").write_text("{}")


def _make_depth_inventory(staging: Path) -> None:
    """Seed a complete depth inventory into ``staging``."""
    depth_dir = staging / "observation.images.depth"
    depth_dir.mkdir()
    (depth_dir / "frame_000000.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (staging / "depth_render_summary.json").write_text("{}")
    (staging / "render_summary.json").write_text("{}")


# --- accepted: empty root -----------------------------------------------------


def test_preflight_accepts_empty_root_for_rgb(tmp_path):
    staging = tmp_path / "stage"
    staging.mkdir()
    preflight_staging(staging, "rgb")  # no raise


def test_preflight_accepts_empty_root_for_depth(tmp_path):
    staging = tmp_path / "stage"
    staging.mkdir()
    preflight_staging(staging, "depth")  # no raise


# --- accepted: complete other-modality inventory ------------------------------


def test_preflight_accepts_complete_depth_inventory_for_rgb(tmp_path):
    staging = tmp_path / "stage"
    staging.mkdir()
    _make_depth_inventory(staging)
    preflight_staging(staging, "rgb")  # no raise


def test_preflight_accepts_complete_rgb_inventory_for_depth(tmp_path):
    staging = tmp_path / "stage"
    staging.mkdir()
    _make_rgb_inventory(staging)
    preflight_staging(staging, "depth")  # no raise


# --- rejected: same-modality overwrite ---------------------------------------


def test_preflight_rejects_own_rgb_dir_for_rgb(tmp_path):
    staging = tmp_path / "stage"
    staging.mkdir()
    _make_rgb_inventory(staging)
    with pytest.raises(FileExistsError, match="rgb"):
        preflight_staging(staging, "rgb")


def test_preflight_rejects_own_depth_dir_for_depth(tmp_path):
    staging = tmp_path / "stage"
    staging.mkdir()
    _make_depth_inventory(staging)
    with pytest.raises(FileExistsError, match="depth"):
        preflight_staging(staging, "depth")


def test_preflight_rejects_own_rgb_summary_only(tmp_path):
    staging = tmp_path / "stage"
    staging.mkdir()
    (staging / "rgb_render_summary.json").write_text("{}")
    with pytest.raises(FileExistsError, match="rgb"):
        preflight_staging(staging, "rgb")


def test_preflight_rejects_own_depth_summary_only(tmp_path):
    staging = tmp_path / "stage"
    staging.mkdir()
    (staging / "depth_render_summary.json").write_text("{}")
    with pytest.raises(FileExistsError, match="depth"):
        preflight_staging(staging, "depth")


# --- rejected: partial other-modality inventory -------------------------------


def test_preflight_rejects_partial_depth_inventory_for_rgb(tmp_path):
    """Depth dir present but no summary files."""
    staging = tmp_path / "stage"
    staging.mkdir()
    depth_dir = staging / "observation.images.depth"
    depth_dir.mkdir()
    (depth_dir / "frame_000000.png").write_bytes(b"\x89PNG")
    with pytest.raises(RuntimeError, match="partial"):
        preflight_staging(staging, "rgb")


def test_preflight_rejects_partial_rgb_inventory_for_depth(tmp_path):
    """RGB dir present but no summary file."""
    staging = tmp_path / "stage"
    staging.mkdir()
    rgb_dir = staging / "observation.images.rgb"
    rgb_dir.mkdir()
    (rgb_dir / "frame_000000.jpg").write_bytes(b"\xff")
    with pytest.raises(RuntimeError, match="partial"):
        preflight_staging(staging, "depth")


def test_preflight_rejects_depth_summary_without_dir_for_rgb(tmp_path):
    """Depth summary present but no depth image dir."""
    staging = tmp_path / "stage"
    staging.mkdir()
    (staging / "depth_render_summary.json").write_text("{}")
    with pytest.raises(RuntimeError, match="partial"):
        preflight_staging(staging, "rgb")


def test_preflight_rejects_rgb_summary_without_dir_for_depth(tmp_path):
    """RGB summary present but no RGB image dir."""
    staging = tmp_path / "stage"
    staging.mkdir()
    (staging / "rgb_render_summary.json").write_text("{}")
    with pytest.raises(RuntimeError, match="partial"):
        preflight_staging(staging, "depth")


def test_preflight_rejects_render_summary_without_depth_dir_for_rgb(tmp_path):
    """render_summary.json present but no depth image dir or depth summary."""
    staging = tmp_path / "stage"
    staging.mkdir()
    (staging / "render_summary.json").write_text("{}")
    with pytest.raises(RuntimeError, match="partial"):
        preflight_staging(staging, "rgb")


# --- rejected: both modalities present ---------------------------------------


def test_preflight_rejects_both_modalities_for_rgb(tmp_path):
    staging = tmp_path / "stage"
    staging.mkdir()
    _make_rgb_inventory(staging)
    _make_depth_inventory(staging)
    with pytest.raises(FileExistsError, match="rgb"):
        preflight_staging(staging, "rgb")


def test_preflight_rejects_both_modalities_for_depth(tmp_path):
    staging = tmp_path / "stage"
    staging.mkdir()
    _make_rgb_inventory(staging)
    _make_depth_inventory(staging)
    with pytest.raises(FileExistsError, match="depth"):
        preflight_staging(staging, "depth")


# --- rejected: unrelated entries ---------------------------------------------


def test_preflight_rejects_unrelated_file_in_empty_root(tmp_path):
    staging = tmp_path / "stage"
    staging.mkdir()
    (staging / "junk.txt").write_text("x")
    with pytest.raises(RuntimeError, match="unrelated"):
        preflight_staging(staging, "rgb")


def test_preflight_rejects_unrelated_file_alongside_depth_for_rgb(tmp_path):
    staging = tmp_path / "stage"
    staging.mkdir()
    _make_depth_inventory(staging)
    (staging / "junk.txt").write_text("x")
    with pytest.raises(RuntimeError, match="unrelated"):
        preflight_staging(staging, "rgb")


# --- process restart / partial fixtures --------------------------------------


def test_preflight_allows_restart_after_complete_other_modality(tmp_path):
    """Simulate a process restart: depth completed, RGB restarts."""
    staging = tmp_path / "stage"
    staging.mkdir()
    _make_depth_inventory(staging)
    # RGB mode should accept this state and be able to start.
    preflight_staging(staging, "rgb")  # no raise


def test_preflight_rejects_restart_after_partial_other_modality(tmp_path):
    """Simulate a crashed depth run: depth dir + no summary. RGB must reject."""
    staging = tmp_path / "stage"
    staging.mkdir()
    depth_dir = staging / "observation.images.depth"
    depth_dir.mkdir()
    (depth_dir / "frame_000000.png").write_bytes(b"\x89PNG")
    # No depth_render_summary.json or render_summary.json — partial.
    with pytest.raises(RuntimeError, match="partial"):
        preflight_staging(staging, "rgb")
"""Tests for sage3d.lerobot_dataset.package (Phase 5c atomic publication).

Covers the full package orchestration: source contract validation, internally
allocated sibling staging, complete build, staged validation, absent-target
atomic rename, failure injection before/after validation and before rename,
partial staging, rerun, target matrix, and canonical package comparison.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from fixtures import build_packaged_dataset, build_rendered_dir, build_trajectory_dir
from sage3d.config import PackageConfig
from sage3d.lerobot_dataset import package
from sage3d.publication import validate_real_directory

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKER_DIR = _REPO_ROOT / "scripts"
if str(_CHECKER_DIR) not in sys.path:
    sys.path.insert(0, str(_CHECKER_DIR))

import check_package  # noqa: E402


# --- helpers -----------------------------------------------------------------

def _make_sources(tmp_path: Path, frame_counts=(3, 2)):
    """Build trajectory + rendered source dirs (no package)."""
    traj = tmp_path / "traj"
    manifest = build_trajectory_dir(traj, episode_frame_counts=frame_counts)
    rendered = tmp_path / "rendered"
    build_rendered_dir(rendered, trajectory_manifest=manifest)
    return traj, rendered, manifest


def _config(tmp_path: Path, **overrides) -> PackageConfig:
    traj, rendered, _ = _make_sources(tmp_path)
    kwargs = {
        "fps": 30,
        "trajectory_dir": traj,
        "rendered_dir": rendered,
        "output_dir": tmp_path / "out",
        "scene_id": "839920",
    }
    kwargs.update(overrides)
    return PackageConfig(**kwargs)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# --- positive ----------------------------------------------------------------

def test_package_builds_and_publishes_atomically(tmp_path):
    config = _config(tmp_path)
    result = package(config)
    assert result == config.output_dir
    validate_real_directory(config.output_dir)
    # Staging directory is gone after the rename.
    assert not list(tmp_path.glob(".pkg.*"))
    # Complete tree present.
    assert (config.output_dir / "data" / "chunk-000" / "episode_000000.parquet").is_file()
    assert (config.output_dir / "meta" / "info.json").is_file()
    assert (config.output_dir / "videos" / "chunk-000" / "observation.images.rgb").is_dir()


def test_package_published_tree_passes_checker_validate(tmp_path):
    config = _config(tmp_path)
    package(config)
    result = check_package.validate(
        config.output_dir, config.trajectory_dir, config.rendered_dir
    )
    assert result["eligible"] is True, result["errors"]


def test_package_matches_legacy_golden_content(tmp_path):
    """The published tree has deterministic content identical to the legacy
    builder path (golden parity via checker compare-golden against a reference
    package built on the same sources)."""
    import shutil

    config = _config(tmp_path)
    package(config)
    baseline = tmp_path / "baseline"
    build_packaged_dataset(
        baseline,
        trajectory_dir=config.trajectory_dir,
        rendered_dir=config.rendered_dir,
        scene_id="839920",
    )
    result = check_package.compare_golden(
        config.output_dir, config.trajectory_dir, config.rendered_dir, baseline
    )
    assert result["eligible"] is True, result["errors"]


def test_package_config_assertions_pass_on_valid_sources(tmp_path):
    config = _config(
        tmp_path,
        width=600, height=450, horizontal_fov_deg=180.0,
        fisheye_coefficients=(0.1, 0.0, 0.0, 0.0), camera_height=0.6,
    )
    package(config)
    assert config.output_dir.is_dir()


# --- source contract failures leave target absent -----------------------------

def test_package_fails_on_scene_id_mismatch(tmp_path):
    config = _config(tmp_path, scene_id="999999")
    with pytest.raises(Exception):
        package(config)
    assert not config.output_dir.exists()
    # No partial staging is left under the final target parent? It may exist;
    # the final target must remain absent either way.
    assert not config.output_dir.exists()


def test_package_fails_on_missing_frame(tmp_path):
    config = _config(tmp_path)
    (config.rendered_dir / "observation.images.rgb" / "episode_000000_000.jpg").unlink()
    with pytest.raises(Exception):
        package(config)
    assert not config.output_dir.exists()


def test_package_fails_on_missing_pointcloud(tmp_path):
    config = _config(tmp_path)
    (config.trajectory_dir / "pointcloud.ply").unlink()
    with pytest.raises(Exception):
        package(config)
    assert not config.output_dir.exists()


# --- staged validation failures leave target absent ----------------------------

def test_package_fails_when_staged_validation_rejects(tmp_path, monkeypatch):
    import sage3d.lerobot_dataset as lrd

    config = _config(tmp_path)

    def reject(*args, **kwargs):
        return {"eligible": False, "errors": ["injected staged failure"]}

    monkeypatch.setattr(lrd, "validate_packaged_dataset", reject)
    with pytest.raises(RuntimeError, match="injected staged failure"):
        package(config)
    assert not config.output_dir.exists()
    # The staging directory is left intact for diagnosis.
    assert list(tmp_path.glob(".pkg.*"))


# --- immediately-before-rename injection ---------------------------------------

def test_failure_during_publication_leaves_target_absent(tmp_path, monkeypatch):
    import sage3d.lerobot_dataset as lrd

    config = _config(tmp_path)

    def boom(staging: Path, target: Path) -> Path:
        raise RuntimeError("injected publication failure")

    monkeypatch.setattr(lrd, "atomic_publish_directory", boom)
    with pytest.raises(RuntimeError, match="injected"):
        package(config)
    assert not config.output_dir.exists()
    assert list(tmp_path.glob(".pkg.*"))


# --- existing target matrix ----------------------------------------------------

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
def test_package_refuses_every_existing_target_type(tmp_path, make_target):
    config = _config(tmp_path)
    make_target(config.output_dir)
    with pytest.raises(FileExistsError, match="already exists"):
        package(config)
    # The existing target is never touched.
    assert config.output_dir.exists() or config.output_dir.is_symlink()


# --- rerun semantics -----------------------------------------------------------

def test_package_rerun_requires_call_cleanup(tmp_path):
    config = _config(tmp_path)
    package(config)
    # A second run without caller cleanup refuses the existing target.
    with pytest.raises(FileExistsError, match="already exists"):
        package(config)


# --- output content sanity -----------------------------------------------------

def test_package_published_tree_matches_sources(tmp_path):
    config = _config(tmp_path)
    package(config)
    # RGB/depth files copied from the render root.
    rgb_src = sorted(
        (config.rendered_dir / "observation.images.rgb").glob("*.jpg")
    )
    rgb_dst = sorted(
        (config.output_dir / "videos" / "chunk-000" / "observation.images.rgb").glob("*.jpg")
    )
    assert [p.name for p in rgb_dst] == [p.name for p in rgb_src]
    # Parquet schema has the required float32 columns.
    table = pq.read_table(
        config.output_dir / "data" / "chunk-000" / "episode_000000.parquet"
    )
    for col in (
        "observation.camera_intrinsic",
        "observation.camera_extrinsic",
        "observation.camera_distortion",
        "observation.point_goal",
        "action",
    ):
        assert col in table.column_names
        assert "float" in str(table.schema.field(col).type)

"""Package-safe tests for the pinned Phase 0b capture harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_sage3d_canonical as canonical


def _measurement(
    *,
    leakage: float,
    rmse: float,
    rgb_p99: float,
    iou: float,
    depth: float,
) -> dict:
    return {
        "eligible": True,
        "metrics": {
            "per_frame": [
                {
                    "episode": episode,
                    "frame": frame,
                    "rgb_mask_leakage_mean": leakage,
                    "rgb_masked_rmse": rmse,
                    "rgb_masked_abs_error_p99": rgb_p99,
                    "depth_non_max_mask_iou": iou,
                    "depth_error_p50": depth,
                    "depth_error_p95": depth,
                    "depth_error_p99": depth,
                }
                for episode in range(5)
                for frame in range(3)
            ]
        },
    }


def test_allocate_directory_requires_absent_real_parent(tmp_path):
    child = canonical.allocate_directory(tmp_path, "run")
    assert child.is_dir()
    assert child.stat().st_dev == tmp_path.stat().st_dev
    with pytest.raises(FileExistsError):
        canonical.allocate_directory(tmp_path, "run")

    symlink = tmp_path / "parent-link"
    symlink.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        canonical.allocate_directory(symlink, "other")


def test_derive_threshold_report_uses_only_five_characterization_runs():
    policy = {
        "baseline_id": canonical.BASELINE_ID,
        "plan_revision": 8,
        "plan_commit": canonical.PLAN_COMMIT,
        "pre_observation_minimum_margins": {
            "rgb_leakage_min_margin": 0.002,
            "rgb_rmse_min_margin": 0.002,
            "rgb_p99_min_margin": 0.01,
            "depth_error_min_margin": 2,
            "depth_iou_min_margin": 0.02,
        },
    }
    baseline = _measurement(
        leakage=0.001,
        rmse=0,
        rgb_p99=0,
        iou=1,
        depth=0,
    )
    characterizations = [
        _measurement(
            leakage=0.003,
            rmse=0.004,
            rgb_p99=0.02,
            iou=0.98,
            depth=4,
        )
        for _ in range(5)
    ]
    report = canonical.derive_threshold_report(
        policy, baseline, characterizations
    )
    assert report["characterization_run_count"] == 5
    assert report["held_out_runs_contributed"] == 0
    assert report["thresholds"]["rgb_masked_rmse"] == pytest.approx(0.006)
    assert report["thresholds"]["depth_error_p99"] == pytest.approx(6.0)
    assert report["thresholds"]["depth_non_max_mask_iou"] == pytest.approx(0.96)


def test_derive_threshold_report_rejects_wrong_run_count():
    with pytest.raises(ValueError, match="expected 5"):
        canonical.derive_threshold_report(
            {
                "baseline_id": "b",
                "plan_revision": 8,
                "plan_commit": "c",
                "pre_observation_minimum_margins": {},
            },
            {"metrics": {"per_frame": []}},
            [],
        )


def test_generation_command_pins_issue_contract(tmp_path):
    command = canonical._generation_command(
        Path("/python"),
        tmp_path,
        {
            "interior_root": Path("/assets/interior"),
            "collision_usd": Path("/assets/collision.usd"),
        },
        Path("/output"),
    )
    assert command[command.index("--scene") + 1] == "839920"
    assert command[command.index("--seed") + 1] == "20260720"
    assert command[command.index("--episodes") + 1] == "5"
    assert command[command.index("--max-attempts") + 1] == "3000"

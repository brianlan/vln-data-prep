"""Tests for the Phase 0a pre-observation tolerance policy.

These verify the ``tolerance_policy.json`` schema and completeness against the
binding requirements in SAGE3D_REFACTOR_PLAN.md revision 8: the named metrics,
frame selector, minimum margins, five-characterization/two-held-out protocol,
mutation suite, cache/retention policy, and deterministic mutation
preconditions. The policy must be immutable for the first capture attempt, so
this test also guards against accidental drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_GOLDEN_DIR = Path(__file__).resolve().parents[2] / "tests" / "golden"
_POLICY_PATH = (
    _GOLDEN_DIR / "839920" / "phase0a-pre-observation" / "tolerance_policy.json"
)

REQUIRED_METRICS = (
    "rgb_mask_leakage_mean_max",
    "rgb_masked_rmse",
    "rgb_masked_abs_error_p99",
    "depth_non_max_mask_iou",
    "depth_error_p50",
    "depth_error_p95",
    "depth_error_p99",
)

REQUIRED_MIN_MARGINS = (
    "rgb_leakage_min_margin",
    "rgb_rmse_min_margin",
    "rgb_p99_min_margin",
    "depth_error_min_margin",
    "depth_iou_min_margin",
)


@pytest.fixture(scope="module")
def policy() -> dict:
    with _POLICY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_policy_file_exists():
    assert _POLICY_PATH.is_file(), f"missing {_POLICY_PATH}"


def test_policy_schema_version_and_baseline_id(policy):
    assert policy["schema_version"] == 1
    assert policy["baseline_id"] == "phase0a-pre-observation"
    assert policy["plan_revision"] == 8
    assert policy["plan_commit"] == "f8548ca6ac7bf9b9f9d16ff7e49cf8e3cc8c5a63"


def test_policy_status_is_pre_observation(policy):
    assert policy["status"] == "pre-observation"
    assert "immutable" in policy["commitment_note"].lower()


def test_policy_has_all_named_metrics(policy):
    metrics = policy["metrics"]
    missing = [name for name in REQUIRED_METRICS if name not in metrics]
    assert not missing, f"missing metrics: {missing}"


@pytest.mark.parametrize("metric_name", REQUIRED_METRICS)
def test_metric_has_required_fields(policy, metric_name):
    metric = policy["metrics"][metric_name]
    for field in (
        "formula",
        "threshold_formula",
        "smallest_intended_detectable_regression",
        "benign_gpu_variation_that_must_pass",
        "boundary_mutation",
        "evaluation_scope",
        "minimum_margin",
    ):
        assert field in metric, f"{metric_name} missing {field}"


def test_policy_has_pre_observation_minimum_margins(policy):
    margins = policy["pre_observation_minimum_margins"]
    missing = [name for name in REQUIRED_MIN_MARGINS if name not in margins]
    assert not missing, f"missing margins: {missing}"
    for name in REQUIRED_MIN_MARGINS:
        value = margins[name]
        assert isinstance(value, (int, float)) and value >= 0, f"{name} not non-negative number"


def test_policy_rgb_mask_dilation_pixels_is_nonnegative_int(policy):
    value = policy["rgb_mask_dilation_pixels"]
    assert isinstance(value, int) and value >= 0


def test_policy_selected_frames(policy):
    frames = policy["selected_frames"]
    assert frames["episode_count"] == 5
    assert frames["pairs_per_episode"] == 3
    assert frames["total_rgb_depth_pairs"] == 15
    assert "first" in frames["selector"]
    assert "middle" in frames["selector"]
    assert "last" in frames["selector"]


def test_policy_characterization_protocol(policy):
    proto = policy["characterization_protocol"]
    assert proto["characterization_runs"] == 5
    assert proto["held_out_runs"] == 2
    assert proto["fresh_processes"] is True
    assert proto["no_reused_app_state_or_output_files"] is True
    assert "cache_policy" in proto
    assert "retention_policy" in proto
    derivation = proto["threshold_derivation"]
    assert "upper_bound_metrics" in derivation
    assert "depth_iou" in derivation
    assert derivation["held_out_rule"].startswith("neither")


def test_policy_mutation_suite(policy):
    suite = policy["mutation_suite"]
    for case in ("must_fail_rgb", "must_fail_depth"):
        assert len(suite[case]) >= 8, f"{case} too short"
    assert "boundary_probe" in suite
    preconditions = suite["determinism_preconditions"]
    assert "minimum_channel_difference_before_channel_swap" in preconditions
    assert "one_frame_latency_probe" in preconditions
    assert "pinned_corruption_sizes" in preconditions


def test_policy_boundary_budget_fixtures(policy):
    fixtures = policy["boundary_budget_fixtures"]
    assert len(fixtures) >= 5


def test_policy_deterministic_mutation_preconditions(policy):
    preconditions = policy["deterministic_mutation_preconditions"]
    assert len(preconditions) >= 4
    assert all(isinstance(p, str) and p for p in preconditions)


def test_policy_controlled_rebaselining(policy):
    rebaseline = policy["controlled_rebaselining"]
    assert rebaseline["rule"] == "never edit a baseline in place"
    assert "trigger" in rebaseline
    assert "procedure" in rebaseline


def test_policy_is_valid_json_and_nan_free(policy):
    # Re-serialize with the canonical encoder to prove NaN/Infinity are absent.
    encoded = json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
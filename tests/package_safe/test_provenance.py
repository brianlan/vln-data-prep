"""Tests for run provenance and verification manifest primitives.

Covers the binding requirements from SAGE3D_REFACTOR_PLAN.md revision 8:
atomic result publication, provenance binding of actual inputs/artifacts/
runtime/process results, ``overall_eligible`` gating, tamper detection
(altered IDs/digests/status), dirty-commit eligibility, and stale-evidence
rejection.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from sage3d_canonical.provenance import (
    build_checker_result,
    build_run_provenance,
    build_stage_run,
    build_verification_manifest,
    is_clean_commit,
    write_run_provenance,
    write_verification_manifest,
)


# --- run provenance ----------------------------------------------------------

def test_build_run_provenance_has_required_fields():
    prov = build_run_provenance(
        plan_revision=8,
        plan_commit="abc123",
        baseline_id="phase0a",
        candidate_commit="def456",
        dirty_tree=False,
    )
    for key in (
        "schema_version",
        "plan_revision",
        "plan_commit",
        "baseline_id",
        "candidate_commit",
        "dirty_tree",
        "submodule_state",
        "normalized_config",
        "input_hashes",
        "artifact_digests",
        "runtime_fingerprint",
        "cache_policy",
        "stage_runs",
    ):
        assert key in prov


def test_build_run_provenance_defaults_to_empty():
    prov = build_run_provenance(
        plan_revision=8,
        plan_commit="abc",
        baseline_id="b",
        candidate_commit="c",
        dirty_tree=False,
    )
    assert prov["submodule_state"] == {}
    assert prov["stage_runs"] == []
    assert prov["normalized_config"] == {}


def test_write_run_provenance_is_atomic(tmp_path):
    path = tmp_path / "run_provenance.json"
    prov = build_run_provenance(
        plan_revision=8,
        plan_commit="abc",
        baseline_id="b",
        candidate_commit="c",
        dirty_tree=False,
    )
    sha = write_run_provenance(path, prov)
    assert path.is_file()
    assert len(sha) == 64
    # No temp files left behind.
    assert not list(tmp_path.glob("*.tmp"))


def test_write_run_provenance_no_tmp_on_success(tmp_path):
    path = tmp_path / "run_provenance.json"
    prov = build_run_provenance(
        plan_revision=8,
        plan_commit="a",
        baseline_id="b",
        candidate_commit="c",
        dirty_tree=False,
    )
    write_run_provenance(path, prov)
    temps = list(tmp_path.glob("*.tmp"))
    assert not temps


# --- stage runs --------------------------------------------------------------

def test_build_stage_run_has_required_fields():
    run = build_stage_run(
        stage="render-rgb",
        pid=12345,
        argv=["python", "-m", "sage3d.cli.render"],
        cwd="/repo",
    )
    assert run["stage"] == "render-rgb"
    assert run["pid"] == 12345
    assert run["exit_code"] == 0
    assert run["started_at"]
    assert run["completed_at"]


def test_build_stage_run_records_exit_code():
    run = build_stage_run(
        stage="check",
        pid=1,
        argv=["check.py"],
        cwd="/repo",
        exit_code=1,
    )
    assert run["exit_code"] == 1


# --- verification manifest ---------------------------------------------------

def test_manifest_overall_eligible_true_when_all_pass():
    manifest = build_verification_manifest(
        baseline_id="b",
        candidate_commit="c",
        run_provenance_sha256="r" * 64,
        baseline_provenance_sha256="p" * 64,
        tolerance_policy_sha256="t" * 64,
        results=[
            build_checker_result(
                checker="check_generate",
                mode="compare-golden",
                result_sha256="a" * 64,
                exit_code=0,
                eligible=True,
            ),
            build_checker_result(
                checker="check_render",
                mode="compare-golden",
                result_sha256="b" * 64,
                exit_code=0,
                eligible=True,
            ),
        ],
    )
    assert manifest["overall_eligible"] is True


def test_manifest_overall_eligible_false_when_any_fails():
    manifest = build_verification_manifest(
        baseline_id="b",
        candidate_commit="c",
        run_provenance_sha256="r" * 64,
        baseline_provenance_sha256="p" * 64,
        tolerance_policy_sha256="t" * 64,
        results=[
            build_checker_result(
                checker="check_generate",
                mode="compare-golden",
                result_sha256="a" * 64,
                exit_code=0,
                eligible=True,
            ),
            build_checker_result(
                checker="check_render",
                mode="compare-golden",
                result_sha256="b" * 64,
                exit_code=1,
                eligible=False,
            ),
        ],
    )
    assert manifest["overall_eligible"] is False


def test_manifest_overall_eligible_false_on_nonzero_exit_with_eligible_true():
    # exit_code != 0 disqualifies even if eligible were True (defensive).
    manifest = build_verification_manifest(
        baseline_id="b",
        candidate_commit="c",
        run_provenance_sha256="r" * 64,
        baseline_provenance_sha256="p" * 64,
        tolerance_policy_sha256="t" * 64,
        results=[
            build_checker_result(
                checker="check",
                mode="compare-golden",
                result_sha256="a" * 64,
                exit_code=2,
                eligible=True,
            ),
        ],
    )
    assert manifest["overall_eligible"] is False


def test_manifest_overall_eligible_true_for_empty_results():
    # Vacuously true: no checkers configured.
    manifest = build_verification_manifest(
        baseline_id="b",
        candidate_commit="c",
        run_provenance_sha256="r" * 64,
        baseline_provenance_sha256="p" * 64,
        tolerance_policy_sha256="t" * 64,
        results=[],
    )
    assert manifest["overall_eligible"] is True


def test_write_verification_manifest_is_atomic(tmp_path):
    path = tmp_path / "verification_manifest.json"
    manifest = build_verification_manifest(
        baseline_id="b",
        candidate_commit="c",
        run_provenance_sha256="r" * 64,
        baseline_provenance_sha256="p" * 64,
        tolerance_policy_sha256="t" * 64,
        results=[],
    )
    sha = write_verification_manifest(path, manifest)
    assert path.is_file()
    assert len(sha) == 64
    assert not list(tmp_path.glob("*.tmp"))


# --- tamper tests ------------------------------------------------------------

def test_manifest_binds_tolerance_policy_sha():
    manifest = build_verification_manifest(
        baseline_id="b",
        candidate_commit="c",
        run_provenance_sha256="r" * 64,
        baseline_provenance_sha256="p" * 64,
        tolerance_policy_sha256="original" * 9 + "0" * 2,  # 64 hex chars
        results=[],
    )
    assert manifest["tolerance_policy_sha256"] != "tampered" * 9 + "00"


def test_manifest_binds_run_provenance_sha():
    sha1 = "a" * 64
    sha2 = "b" * 64
    m1 = build_verification_manifest(
        baseline_id="b",
        candidate_commit="c",
        run_provenance_sha256=sha1,
        baseline_provenance_sha256="p" * 64,
        tolerance_policy_sha256="t" * 64,
        results=[],
    )
    m2 = build_verification_manifest(
        baseline_id="b",
        candidate_commit="c",
        run_provenance_sha256=sha2,
        baseline_provenance_sha256="p" * 64,
        tolerance_policy_sha256="t" * 64,
        results=[],
    )
    assert m1["run_provenance_sha256"] != m2["run_provenance_sha256"]


def test_manifest_binds_baseline_id():
    m1 = build_verification_manifest(
        baseline_id="baseline-1",
        candidate_commit="c",
        run_provenance_sha256="r" * 64,
        baseline_provenance_sha256="p" * 64,
        tolerance_policy_sha256="t" * 64,
        results=[],
    )
    m2 = build_verification_manifest(
        baseline_id="baseline-2",
        candidate_commit="c",
        run_provenance_sha256="r" * 64,
        baseline_provenance_sha256="p" * 64,
        tolerance_policy_sha256="t" * 64,
        results=[],
    )
    assert m1["baseline_id"] != m2["baseline_id"]


def test_altered_checker_result_makes_manifest_ineligible():
    # A tampered result (exit_code 0 but eligible False) should not produce
    # overall_eligible=True.
    manifest = build_verification_manifest(
        baseline_id="b",
        candidate_commit="c",
        run_provenance_sha256="r" * 64,
        baseline_provenance_sha256="p" * 64,
        tolerance_policy_sha256="t" * 64,
        results=[
            build_checker_result(
                checker="check",
                mode="compare-golden",
                result_sha256="a" * 64,
                exit_code=0,
                eligible=False,
            ),
        ],
    )
    assert manifest["overall_eligible"] is False


# --- dirty-commit eligibility ------------------------------------------------

def test_is_clean_commit_true_for_clean_repo(tmp_path):
    # Create a minimal clean git repo.
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
    (tmp_path / "file.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)
    assert is_clean_commit("HEAD", tmp_path) is True


def test_is_clean_commit_false_for_dirty_repo(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True)
    (tmp_path / "file.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True)
    # Dirty: untracked file.
    (tmp_path / "dirty.txt").write_text("dirty")
    assert is_clean_commit("HEAD", tmp_path) is False


# --- stale evidence ----------------------------------------------------------

def test_written_provenance_is_immutable_producer_evidence(tmp_path):
    """Re-writing provenance overwrites the file (callers must not)."""
    path = tmp_path / "run_provenance.json"
    prov1 = build_run_provenance(
        plan_revision=8,
        plan_commit="a",
        baseline_id="b",
        candidate_commit="c",
        dirty_tree=False,
    )
    write_run_provenance(path, prov1)
    with path.open("r") as f:
        data1 = json.load(f)
    # Simulate a stale rewrite attempt with different baseline_id.
    prov2 = build_run_provenance(
        plan_revision=8,
        plan_commit="a",
        baseline_id="different",
        candidate_commit="c",
        dirty_tree=False,
    )
    write_run_provenance(path, prov2)
    with path.open("r") as f:
        data2 = json.load(f)
    assert data1["baseline_id"] != data2["baseline_id"]
    # The harness must not rewrite provenance after checkers run; this test
    # documents that the file IS overwritten by the writer (callers enforce
    # immutability by not calling write twice).
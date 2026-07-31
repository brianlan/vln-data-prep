"""Validate the review-sized Phase 0b evidence committed to the repository."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


_EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / "golden"
    / "839920"
    / "phase0a-pre-observation"
    / "phase0b"
)


def _read_json(name: str) -> dict:
    return json.loads((_EVIDENCE / name).read_text(encoding="utf-8"))


def _sha256(name: str) -> str:
    return hashlib.sha256((_EVIDENCE / name).read_bytes()).hexdigest()


def test_phase0b_committed_evidence_is_complete():
    report = _read_json("capture_report.json")
    thresholds = _read_json("threshold_report.json")
    mutations = _read_json("mutation_report.json")
    manifest = _read_json("verification_manifest.json")
    baseline_provenance = _read_json("baseline_run_provenance.json")

    assert report["machine_gate"] == "passed"
    assert report["formal_gate"] == "pending-independent-approval"
    assert len(report["characterization"]) == 5
    assert len(report["held_outs"]) == 2
    assert all(run["render_eligible"] for run in report["held_outs"])
    assert all(run["package_eligible"] for run in report["held_outs"])

    assert thresholds["held_out_runs_contributed"] == 0
    assert thresholds["status"] == "pending-independent-approval"
    assert mutations["all_expected_outcomes_observed"] is True
    assert len(mutations["cases"]) == 25
    assert all(case["detected_as_expected"] for case in mutations["cases"])
    assert manifest["overall_eligible"] is True
    assert len(manifest["results"]) == 9
    assert all(result["eligible"] for result in manifest["results"])

    assert baseline_provenance["candidate_commit"] == report["candidate_commit"]
    assert baseline_provenance["dirty_tree"] is False
    assert len(baseline_provenance["stage_runs"]) == 4
    assert all(stage["exit_code"] == 0 for stage in baseline_provenance["stage_runs"])

    assert report["threshold_report_sha256"] == _sha256("threshold_report.json")
    assert report["mutation_report_sha256"] == _sha256("mutation_report.json")
    assert report["verification_manifest_sha256"] == _sha256(
        "verification_manifest.json"
    )

    rgb = list((_EVIDENCE / "selected_frames" / "rgb").glob("*.jpg"))
    depth = list((_EVIDENCE / "selected_frames" / "depth").glob("*.png"))
    assert len(rgb) == 15
    assert len(depth) == 15
    assert {path.stem for path in rgb} == {path.stem for path in depth}

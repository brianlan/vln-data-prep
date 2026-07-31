"""Validate the external Phase 0b capture as a binding GPU gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.sage3d_gpu


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_phase0b_capture_machine_gate():
    capture_value = os.environ.get("SAGE3D_PHASE0B_CAPTURE")
    if not capture_value:
        pytest.fail("SAGE3D_PHASE0B_CAPTURE is required for the canonical GPU lane")
    capture = Path(capture_value)
    evidence = capture / "evidence"
    report_path = evidence / "capture_report.json"
    manifest_path = evidence / "verification_manifest.json"
    threshold_path = evidence / "threshold_report.json"
    mutation_path = capture / "mutations" / "mutation_report.json"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
    mutations = json.loads(mutation_path.read_text(encoding="utf-8"))

    assert report["machine_gate"] == "passed"
    assert report["formal_gate"] == "pending-independent-approval"
    assert len(report["characterization"]) == 5
    assert len(report["held_outs"]) == 2
    assert all(run["render_eligible"] for run in report["held_outs"])
    assert all(run["package_eligible"] for run in report["held_outs"])
    assert thresholds["held_out_runs_contributed"] == 0
    assert thresholds["status"] == "pending-independent-approval"
    assert mutations["all_expected_outcomes_observed"] is True
    assert manifest["overall_eligible"] is True
    assert manifest["results"]

    assert report["threshold_report_sha256"] == _sha256(threshold_path)
    assert report["mutation_report_sha256"] == _sha256(mutation_path)
    assert report["verification_manifest_sha256"] == _sha256(manifest_path)

    render_stage_records = [
        stage
        for run in report["characterization"] + report["held_outs"]
        for stage in run["stage_runs"]
        if "render-" in stage["stage"]
    ]
    assert len(render_stage_records) == 14
    assert len({stage["pid"] for stage in render_stage_records}) == 14
    assert all(stage["exit_code"] == 0 for stage in render_stage_records)

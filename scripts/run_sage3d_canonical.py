#!/usr/bin/env python3
"""Canonical SAGE3D Phase 0b evidence orchestration harness.

Writes ``run_provenance.json`` and ``verification_manifest.json`` sidecars
under ``--evidence-dir``, binding checker results, run provenance, baseline
provenance, and the tolerance policy via SHA-256. Failed/incomplete
orchestration writes a separate non-binding status file and never a successful
verification manifest.

Phase 0b evidence primitives (issue #2); the checker scripts (issues #3-#5)
are invoked when present. When no checkers are configured, the harness writes
provenance and a manifest with an empty results list (overall_eligible=True
vacuously for the empty case, per the binding rule).

Package-safe: stdlib + ``hashlib`` + the canonical test helpers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the canonical helpers importable when run from the repo checkout.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tests" / "package_safe"))

from canonical.provenance import (  # noqa: E402
    build_checker_result,
    build_run_provenance,
    build_stage_run,
    build_verification_manifest,
    is_clean_commit,
    write_run_provenance,
    write_verification_manifest,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Canonical SAGE3D evidence harness")
    p.add_argument("--evidence-dir", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--baseline-id", required=True)
    p.add_argument("--plan-commit", required=True)
    p.add_argument("--candidate-commit", required=True)
    p.add_argument("--plan-revision", type=int, default=8)
    p.add_argument(
        "--baseline-provenance",
        type=Path,
        required=True,
        help="Path to baseline run_provenance.json",
    )
    p.add_argument(
        "--tolerance-policy",
        type=Path,
        required=True,
        help="Path to tolerance_policy.json",
    )
    p.add_argument(
        "--normalized-config",
        type=Path,
        default=None,
        help="Optional path to normalized config JSON",
    )
    p.add_argument(
        "--input-hashes",
        type=Path,
        default=None,
        help="Optional path to input-asset hashes JSON",
    )
    p.add_argument(
        "--artifact-digests",
        type=Path,
        default=None,
        help="Optional path to artifact digests JSON",
    )
    p.add_argument(
        "--checkers",
        nargs="*",
        default=None,
        help="Checker commands to run (each as a shell string); omit for empty",
    )
    return p.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    args = parse_args()
    evidence_dir = args.evidence_dir
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Check clean tree.
    dirty = not is_clean_commit(args.candidate_commit, args.repo_root)

    # Load optional inputs.
    normalized_config = (
        _load_json(args.normalized_config) if args.normalized_config else {}
    )
    input_hashes = (
        _load_json(args.input_hashes) if args.input_hashes else {}
    )
    artifact_digests = (
        _load_json(args.artifact_digests) if args.artifact_digests else {}
    )

    # Runtime fingerprint (diagnostic; equality-required fields set by callers).
    runtime_fingerprint = {
        "python_version": sys.version,
        "captured_at": _utc_now_iso(),
    }

    # Run checkers if configured.
    stage_runs: list[dict] = []
    checker_results: list[dict] = []
    overall_ok = True
    if args.checkers:
        for cmd_str in args.checkers:
            started = _utc_now_iso()
            proc = subprocess.run(
                cmd_str,
                shell=True,
                cwd=args.repo_root,
                capture_output=True,
                text=True,
            )
            completed = _utc_now_iso()
            stage_runs.append(
                build_stage_run(
                    stage=cmd_str.split()[0] if cmd_str else "checker",
                    pid=proc.pid,
                    argv=cmd_str.split(),
                    cwd=str(args.repo_root),
                    started_at=started,
                    completed_at=completed,
                    exit_code=proc.returncode,
                )
            )
            if proc.returncode != 0:
                overall_ok = False
            # Attempt to parse checker JSON result; if unavailable, build a
            # minimal result from the exit code.
            result_sha = _sha256_bytes(proc.stdout.encode("utf-8"))
            checker_results.append(
                build_checker_result(
                    checker=cmd_str.split()[0] if cmd_str else "checker",
                    mode="compare-golden",
                    result_sha256=result_sha,
                    exit_code=proc.returncode,
                    eligible=proc.returncode == 0,
                )
            )

    # Build and write run_provenance.json.
    provenance = build_run_provenance(
        plan_revision=args.plan_revision,
        plan_commit=args.plan_commit,
        baseline_id=args.baseline_id,
        candidate_commit=args.candidate_commit,
        dirty_tree=dirty,
        normalized_config=normalized_config,
        input_hashes=input_hashes,
        artifact_digests=artifact_digests,
        runtime_fingerprint=runtime_fingerprint,
        cache_policy={"policy": "recorded"},
        stage_runs=stage_runs,
    )
    provenance_path = evidence_dir / "run_provenance.json"
    provenance_sha = write_run_provenance(provenance_path, provenance)

    if not overall_ok:
        # Failed/incomplete: write non-binding status, no verification manifest.
        status_path = evidence_dir / "orchestration_status.json"
        with status_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"eligible": False, "completed_at": _utc_now_iso()},
                f,
                indent=2,
            )
        print(f"[canonical] orchestration incomplete: {status_path}")
        return 1

    # Compute binding hashes.
    baseline_prov_sha = _sha256_file(args.baseline_provenance)
    policy_sha = _sha256_file(args.tolerance_policy)

    # Build and write verification_manifest.json.
    manifest = build_verification_manifest(
        baseline_id=args.baseline_id,
        candidate_commit=args.candidate_commit,
        run_provenance_sha256=provenance_sha,
        baseline_provenance_sha256=baseline_prov_sha,
        tolerance_policy_sha256=policy_sha,
        results=checker_results,
    )
    manifest_path = evidence_dir / "verification_manifest.json"
    manifest_sha = write_verification_manifest(manifest_path, manifest)
    print(
        f"[canonical] wrote {manifest_path} "
        f"(sha256={manifest_sha[:12]}, overall_eligible={manifest['overall_eligible']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
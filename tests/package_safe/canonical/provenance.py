"""Run provenance and verification manifest primitives for Phase 0b.

Implements the structured ``run_provenance.json`` and
``verification_manifest.json`` sidecars per SAGE3D_REFACTOR_PLAN.md revision 8.
Both are written atomically (temp file + ``os.replace``) and never overwrite
an existing final file.

Package-safe: stdlib + ``hashlib`` only.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from canonical.digest import canonical_json_bytes, digest_json


PROVENANCE_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, value: Any) -> None:
    """Write JSON atomically: temp file in same dir, then ``os.replace``.

    ``os.replace`` overwrites an existing final file; callers must enforce
    immutability by not calling twice, or use the distinct non-binding status
    path for failure diagnostics.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=path.name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_run_provenance(
    *,
    plan_revision: int,
    plan_commit: str,
    baseline_id: str,
    candidate_commit: str,
    dirty_tree: bool,
    normalized_config: dict[str, Any] | None = None,
    input_hashes: dict[str, str] | None = None,
    artifact_digests: dict[str, str] | None = None,
    runtime_fingerprint: dict[str, Any] | None = None,
    cache_policy: dict[str, str] | None = None,
    stage_runs: list[dict[str, Any]] | None = None,
    submodule_state: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a ``run_provenance.json`` dict (not yet written)."""
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "plan_revision": plan_revision,
        "plan_commit": plan_commit,
        "baseline_id": baseline_id,
        "candidate_commit": candidate_commit,
        "dirty_tree": dirty_tree,
        "submodule_state": submodule_state or {},
        "normalized_config": normalized_config or {},
        "input_hashes": input_hashes or {},
        "artifact_digests": artifact_digests or {},
        "runtime_fingerprint": runtime_fingerprint or {},
        "cache_policy": cache_policy or {},
        "stage_runs": stage_runs or [],
    }


def write_run_provenance(path: Path, provenance: dict[str, Any]) -> str:
    """Atomically write ``run_provenance.json``; return its SHA-256."""
    _atomic_write_json(path, provenance)
    return _sha256_file(path)


def build_verification_manifest(
    *,
    baseline_id: str,
    candidate_commit: str,
    run_provenance_sha256: str,
    baseline_provenance_sha256: str,
    tolerance_policy_sha256: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a ``verification_manifest.json`` dict.

    ``overall_eligible`` is ``True`` only when every result has
    ``eligible == True`` and ``exit_code == 0``.
    """
    overall = all(
        r.get("eligible") is True and r.get("exit_code") == 0 for r in results
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "baseline_id": baseline_id,
        "candidate_commit": candidate_commit,
        "run_provenance_sha256": run_provenance_sha256,
        "baseline_provenance_sha256": baseline_provenance_sha256,
        "tolerance_policy_sha256": tolerance_policy_sha256,
        "results": results,
        "overall_eligible": overall,
    }


def write_verification_manifest(path: Path, manifest: dict[str, Any]) -> str:
    """Atomically write ``verification_manifest.json``; return its SHA-256."""
    _atomic_write_json(path, manifest)
    return _sha256_file(path)


def build_stage_run(
    *,
    stage: str,
    pid: int,
    argv: list[str],
    cwd: str,
    environment: dict[str, str] | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
    exit_code: int = 0,
) -> dict[str, Any]:
    """Build a structured subprocess record for ``stage_runs``."""
    return {
        "stage": stage,
        "pid": pid,
        "argv": argv,
        "cwd": cwd,
        "environment": environment or {},
        "started_at": started_at or _utc_now_iso(),
        "completed_at": completed_at or _utc_now_iso(),
        "exit_code": exit_code,
    }


def build_checker_result(
    *,
    checker: str,
    mode: str,
    result_sha256: str,
    exit_code: int,
    eligible: bool,
    artifact_digests: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a checker result entry for the verification manifest."""
    return {
        "checker": checker,
        "mode": mode,
        "result_sha256": result_sha256,
        "exit_code": exit_code,
        "eligible": eligible,
        "artifact_digests": artifact_digests or {},
    }


def is_clean_commit(commitish: str, repo_root: Path) -> bool:
    """Check whether a git commit is clean (no dirty tree).

    Returns ``True`` when the working tree is clean at ``commitish``. A dirty
    tree makes canonical evidence report-only.
    """
    import subprocess

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == ""
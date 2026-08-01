#!/usr/bin/env python3
"""Read-only package-Python checker for packaged SAGE3D datasets.

Two modes per the plan (Checker execution contract, revision 8):

- ``validate``: baseline-independent oracle against current trajectory/render
  inputs. Delegates to the production validator
  ``sage3d.lerobot_dataset.validate_packaged_dataset`` (inventory, schema/
  order/counts, copied files, calibration/extrinsics, depth metadata) and adds
  canonical artifact digests for evidence. Takes ``--dataset-dir
  --trajectory-dir --rendered-dir``.
- ``compare-golden``: runs validate, then deterministic Arrow/JSON content
  comparison against a baseline. Never compares nondeterministic media to
  package goldens and never invokes the render checker.

Package-safe: stdlib + numpy + pyarrow. No forbidden imports. Must not import
``sage3d.cli.package`` or call publication code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make the repo root importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sage3d_canonical.digest import (  # noqa: E402
    digest_arrays,
    digest_directory,
    digest_file,
    digest_json,
)
from sage3d_canonical.parsers import parse_trajectory_manifest  # noqa: E402
from sage3d_canonical.provenance import _atomic_write_json  # noqa: E402
from sage3d.lerobot_dataset import (  # noqa: E402
    REQUIRED_META_FILES,
    _read_jsonl,
    validate_packaged_dataset,
)


def validate(dataset_dir: Path, trajectory_dir: Path, rendered_dir: Path) -> dict[str, Any]:
    """Baseline-independent validation of a packaged dataset.

    Delegates to the production validator and adds canonical artifact digests
    for evidence. Result shape is preserved for checker callers.
    """
    result = validate_packaged_dataset(dataset_dir, trajectory_dir, rendered_dir)
    result["artifact_digests"] = {
        "packaged_root": digest_directory("packaged_root", dataset_dir),
        "trajectory_root": digest_directory("trajectory", trajectory_dir),
        "rendered_root": digest_directory("rendered_root", rendered_dir),
    }
    return result


def compare_golden(
    dataset_dir: Path,
    trajectory_dir: Path,
    rendered_dir: Path,
    baseline_dir: Path,
    *,
    baseline_trajectory_dir: Path | None = None,
    baseline_rendered_dir: Path | None = None,
    run_provenance: Path | None = None,
    baseline_provenance: Path | None = None,
) -> dict[str, Any]:
    """Run validate, then deterministic comparison against a baseline."""
    # Validate candidate.
    val = validate(dataset_dir, trajectory_dir, rendered_dir)
    if not val["eligible"]:
        return {
            "eligible": False,
            "errors": ["validate failed"] + val["errors"],
            "warnings": [],
            "artifact_digests": {},
        }

    # Validate baseline.
    base_val = validate(
        baseline_dir,
        baseline_trajectory_dir or trajectory_dir,
        baseline_rendered_dir or rendered_dir,
    )
    if not base_val["eligible"]:
        return {
            "eligible": False,
            "errors": ["baseline validate failed"] + base_val["errors"],
            "warnings": [],
            "artifact_digests": {},
        }

    errors: list[str] = []
    artifact_digests: dict[str, str] = {}

    # Compare info.json (canonical JSON).
    try:
        with (dataset_dir / "meta" / "info.json").open() as f:
            cand_info = json.load(f)
        with (baseline_dir / "meta" / "info.json").open() as f:
            base_info = json.load(f)
        cand_info_digest = digest_json("packaged_root", cand_info)
        base_info_digest = digest_json("packaged_root", base_info)
        artifact_digests["info"] = cand_info_digest
        if cand_info_digest != base_info_digest:
            errors.append("info.json canonical JSON mismatch")
    except Exception as e:
        errors.append(f"info.json comparison failed: {e}")

    # Compare episodes.jsonl (order-sensitive).
    try:
        cand_episodes = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
        base_episodes = _read_jsonl(baseline_dir / "meta" / "episodes.jsonl")
        if len(cand_episodes) != len(base_episodes):
            errors.append("episodes.jsonl length mismatch")
        else:
            for i, (c, b) in enumerate(zip(cand_episodes, base_episodes)):
                if c != b:
                    errors.append(f"episodes.jsonl[{i}] content mismatch")
                    break
        artifact_digests["episodes_jsonl"] = digest_json("packaged_root", cand_episodes)
    except Exception as e:
        errors.append(f"episodes.jsonl comparison failed: {e}")

    # Compare tasks.jsonl.
    try:
        cand_tasks = _read_jsonl(dataset_dir / "meta" / "tasks.jsonl")
        base_tasks = _read_jsonl(baseline_dir / "meta" / "tasks.jsonl")
        if cand_tasks != base_tasks:
            errors.append("tasks.jsonl content mismatch")
        artifact_digests["tasks_jsonl"] = digest_json("packaged_root", cand_tasks)
    except Exception as e:
        errors.append(f"tasks.jsonl comparison failed: {e}")

    # Compare Parquet content (Arrow schema + values).
    import pyarrow.parquet as pq

    try:
        manifest = parse_trajectory_manifest(trajectory_dir)
        for ep in manifest["episodes"]:
            idx = ep["episode_index"]
            cand_pq = dataset_dir / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"
            base_pq = baseline_dir / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"
            cand_table = pq.read_table(cand_pq)
            base_table = pq.read_table(base_pq)
            if cand_table.schema != base_table.schema:
                errors.append(f"episode_{idx:06d} parquet schema mismatch")
                continue
            if cand_table.num_rows != base_table.num_rows:
                errors.append(f"episode_{idx:06d} parquet row count mismatch")
                continue
            # Compare column-by-column as numpy arrays.
            for col_name in cand_table.column_names:
                cand_col = cand_table[col_name].to_pylist()
                base_col = base_table[col_name].to_pylist()
                if cand_col != base_col:
                    errors.append(f"episode_{idx:06d} column {col_name} values mismatch")
                    break
            # Compute digest of the parquet arrays for evidence.
            cand_arrays = {name: np.array(cand_table[name].to_pylist(), dtype=np.float32)
                          for name in cand_table.column_names if name != "index"}
            if cand_arrays:
                ep_digest = digest_arrays("packaged_root", f"episode_{idx:06d}", cand_arrays)
                artifact_digests[f"episode_{idx:06d}"] = ep_digest
    except Exception as e:
        errors.append(f"parquet comparison failed: {e}")

    # Compare copied meta files (exact bytes: PLY, manifests, summaries).
    for name in REQUIRED_META_FILES - {"info.json", "episodes.jsonl", "tasks.jsonl"}:
        cand_path = dataset_dir / "meta" / name
        base_path = baseline_dir / "meta" / name
        if cand_path.is_file() and base_path.is_file():
            cand_digest = digest_file("packaged_root", cand_path)
            base_digest = digest_file("packaged_root", base_path)
            if name == "pointcloud.ply":
                artifact_digests["pointcloud_ply"] = cand_digest
            elif name == "trajectory_manifest.json":
                artifact_digests["manifest"] = cand_digest
            if cand_digest != base_digest:
                errors.append(f"meta/{name} bytes mismatch")

    # Provenance binding.
    if run_provenance and baseline_provenance:
        try:
            with run_provenance.open() as f:
                rp = json.load(f)
            with baseline_provenance.open() as f:
                bp = json.load(f)
            for field in ("plan_revision", "baseline_id"):
                if rp.get(field) != bp.get(field):
                    errors.append(f"provenance {field} mismatch: {rp.get(field)} != {bp.get(field)}")
        except Exception as e:
            errors.append(f"provenance load failed: {e}")

    eligible = len(errors) == 0
    artifact_digests.update(
        {
            "packaged_root": digest_directory("packaged_root", dataset_dir),
            "trajectory_root": digest_directory("trajectory", trajectory_dir),
            "rendered_root": digest_directory("rendered_root", rendered_dir),
        }
    )
    return {
        "eligible": eligible,
        "errors": errors,
        "warnings": [],
        "artifact_digests": artifact_digests,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAGE3D package artifact checker")
    sub = p.add_subparsers(dest="mode", required=True)

    pv = sub.add_parser("validate", help="Baseline-independent validation")
    pv.add_argument("--dataset-dir", type=Path, required=True)
    pv.add_argument("--trajectory-dir", type=Path, required=True)
    pv.add_argument("--rendered-dir", type=Path, required=True)
    pv.add_argument("--result-path", type=Path, default=None)

    pg = sub.add_parser("compare-golden", help="Compare against a golden baseline")
    pg.add_argument("--dataset-dir", type=Path, required=True)
    pg.add_argument("--trajectory-dir", type=Path, required=True)
    pg.add_argument("--rendered-dir", type=Path, required=True)
    pg.add_argument("--baseline-dir", type=Path, required=True)
    pg.add_argument("--baseline-trajectory-dir", type=Path, default=None)
    pg.add_argument("--baseline-rendered-dir", type=Path, default=None)
    pg.add_argument("--baseline-provenance", type=Path, default=None)
    pg.add_argument("--run-provenance", type=Path, default=None)
    pg.add_argument("--result-path", type=Path, default=None)

    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.mode == "validate":
        result = validate(args.dataset_dir, args.trajectory_dir, args.rendered_dir)
    elif args.mode == "compare-golden":
        result = compare_golden(
            args.dataset_dir,
            args.trajectory_dir,
            args.rendered_dir,
            args.baseline_dir,
            baseline_trajectory_dir=args.baseline_trajectory_dir,
            baseline_rendered_dir=args.baseline_rendered_dir,
            run_provenance=args.run_provenance,
            baseline_provenance=args.baseline_provenance,
        )
    else:
        print(f"unknown mode: {args.mode}", file=sys.stderr)
        return 2

    result["checker"] = "check_package"
    result["mode"] = args.mode

    if args.result_path:
        _atomic_write_json(args.result_path, result)

    status = "ELIGIBLE" if result["eligible"] else "INELIGIBLE"
    print(f"[check_package:{args.mode}] {status}")
    for err in result.get("errors", []):
        print(f"  ERROR: {err}")

    return 0 if result["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

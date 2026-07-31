#!/usr/bin/env python3
"""Read-only package-Python checker for packaged SAGE3D datasets.

Two modes per the plan (Checker execution contract, revision 8):

- ``validate``: baseline-independent oracle against current trajectory/render
  inputs. Checks inventory, schema/order/counts, copied files, calibration/
  extrinsics, and depth metadata. Takes ``--dataset-dir --trajectory-dir
  --rendered-dir``.
- ``compare-golden``: runs validate, then deterministic Arrow/JSON content
  comparison against a baseline. Never compares nondeterministic media to
  package goldens and never invokes the render checker.

Package-safe: stdlib + numpy + PIL + pyarrow. No forbidden imports.
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

from sage3d_canonical.parsers import (  # noqa: E402
    parse_packaged_dataset,
    parse_render_summary,
    parse_trajectory_manifest,
)
from sage3d_canonical.digest import (  # noqa: E402
    digest_arrays,
    digest_directory,
    digest_file,
    digest_json,
)
from sage3d_canonical.provenance import _atomic_write_json, _sha256_file  # noqa: E402

# Parquet columns that must be present with float32 list type.
PARQUET_REQUIRED_COLUMNS = (
    "index",
    "observation.camera_intrinsic",
    "observation.camera_extrinsic",
    "observation.camera_distortion",
    "observation.point_goal",
    "action",
)

# Meta files that must exist in a packaged dataset.
REQUIRED_META_FILES = {
    "info.json",
    "episodes.jsonl",
    "tasks.jsonl",
    "trajectory_manifest.json",
    "render_summary.json",
    "rgb_render_summary.json",
    "depth_render_summary.json",
    "pointcloud.ply",
}


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate(dataset_dir: Path, trajectory_dir: Path, rendered_dir: Path) -> dict[str, Any]:
    """Baseline-independent validation of a packaged dataset."""
    errors: list[str] = []

    # Load packaged dataset structure.
    try:
        pkg = parse_packaged_dataset(dataset_dir)
    except Exception as e:
        return {"eligible": False, "errors": [f"packaged dataset: {e}"], "warnings": []}

    info = pkg["info"]

    # Load trajectory manifest and render summaries for cross-artifact checks.
    try:
        manifest = parse_trajectory_manifest(trajectory_dir)
    except Exception as e:
        return {"eligible": False, "errors": [f"trajectory manifest: {e}"], "warnings": []}

    try:
        rgb_summary = parse_render_summary(rendered_dir, "rgb_render_summary.json")
    except Exception as e:
        errors.append(f"rgb_render_summary: {e}")
        rgb_summary = None

    try:
        depth_summary = parse_render_summary(rendered_dir, "depth_render_summary.json")
    except Exception as e:
        errors.append(f"depth_render_summary: {e}")
        depth_summary = None

    scene_id = manifest.get("scene_id")

    # --- Meta files inventory ---
    meta_files = pkg["meta_files"]
    for name in REQUIRED_META_FILES:
        if name not in meta_files:
            errors.append(f"missing meta file: {name}")

    # --- Scene ID consistency ---
    if info.get("scene_id") != scene_id:
        errors.append(f"info scene_id {info.get('scene_id')} != manifest {scene_id}")
    if depth_summary is not None and depth_summary.get("scene_id") != scene_id:
        errors.append(f"depth summary scene_id {depth_summary.get('scene_id')} != manifest {scene_id}")
    if rgb_summary is not None and rgb_summary.get("scene_id") != scene_id:
        errors.append(f"rgb summary scene_id {rgb_summary.get('scene_id')} != manifest {scene_id}")

    # --- Episode count consistency ---
    expected_episodes = manifest["episode_count"]
    if info.get("total_episodes") != expected_episodes:
        errors.append(f"info total_episodes {info.get('total_episodes')} != manifest {expected_episodes}")

    # --- Parquet file count ---
    parquet_files = pkg["parquet_files"]
    if len(parquet_files) != expected_episodes:
        errors.append(f"parquet count {len(parquet_files)} != manifest episodes {expected_episodes}")

    # --- RGB/depth file counts ---
    expected_frames = sum(ep["frame_count"] for ep in manifest["episodes"])
    if len(pkg["rgb_files"]) != expected_frames:
        errors.append(f"RGB file count {len(pkg['rgb_files'])} != manifest total_frames {expected_frames}")
    if len(pkg["depth_files"]) != expected_frames:
        errors.append(f"depth file count {len(pkg['depth_files'])} != manifest total_frames {expected_frames}")

    # --- info.json total_frames ---
    if info.get("total_frames") != expected_frames:
        errors.append(f"info total_frames {info.get('total_frames')} != manifest {expected_frames}")

    # --- Parquet schema and values ---
    import pyarrow.parquet as pq

    for ep in manifest["episodes"]:
        idx = ep["episode_index"]
        parquet_path = dataset_dir / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"
        if not parquet_path.is_file():
            errors.append(f"missing parquet: episode_{idx:06d}.parquet")
            continue
        try:
            table = pq.read_table(parquet_path)
            schema = table.schema

            # Check required columns exist.
            col_names = set(schema.names)
            for col in PARQUET_REQUIRED_COLUMNS:
                if col not in col_names:
                    errors.append(f"episode_{idx:06d} parquet missing column: {col}")

            # Check row count matches manifest frame_count.
            row_count = table.num_rows
            if row_count != ep["frame_count"]:
                errors.append(
                    f"episode_{idx:06d} parquet rows {row_count} != manifest frame_count {ep['frame_count']}"
                )

            # Check float32 type for camera/extrinsic/distortion/point_goal/action.
            for col_name in PARQUET_REQUIRED_COLUMNS[1:]:
                if col_name in col_names:
                    field = schema.field(col_name)
                    type_str = str(field.type)
                    # pyarrow reports list<float> as "list<element: float>"
                    if "float" not in type_str:
                        errors.append(
                            f"episode_{idx:06d} column {col_name} type {field.type} != list<float32>"
                        )

            # Check index column is int64.
            if "index" in col_names:
                field = schema.field("index")
                if str(field.type) != "int64":
                    errors.append(f"episode_{idx:06d} index type {field.type} != int64")

        except Exception as e:
            errors.append(f"episode_{idx:06d} parquet read failed: {e}")

    # --- Calibration/extrinsics: float32 camera_intrinsic/extrinsic ---
    if parquet_files:
        try:
            first_table = pq.read_table(parquet_files[0])
            if "observation.camera_intrinsic" in first_table.column_names:
                intrinsic = np.array(first_table["observation.camera_intrinsic"][0].as_py(), dtype=np.float32)
                if intrinsic.shape != (9,):
                    errors.append(f"camera_intrinsic flat shape {intrinsic.shape} != (9,)")
            if "observation.camera_extrinsic" in first_table.column_names:
                extrinsic = np.array(first_table["observation.camera_extrinsic"][0].as_py(), dtype=np.float32)
                if extrinsic.shape != (16,):
                    errors.append(f"camera_extrinsic flat shape {extrinsic.shape} != (16,)")
                # Extrinsic [2,3] should match camera_height_m.
                ext_mat = extrinsic.reshape(4, 4)
                if abs(float(ext_mat[2, 3]) - float(info.get("camera_height_m", 0.0))) > 1e-6:
                    errors.append(
                        f"extrinsic z={float(ext_mat[2, 3])} != info camera_height_m {info.get('camera_height_m')}"
                    )
        except Exception as e:
            errors.append(f"calibration check failed: {e}")

    # --- Depth metadata authority ---
    if depth_summary is not None:
        if info.get("depth_clip_m") != depth_summary.get("max_depth_m"):
            errors.append(
                f"info depth_clip_m {info.get('depth_clip_m')} != depth summary max_depth_m {depth_summary.get('max_depth_m')}"
            )
        if info.get("depth_min_m") != depth_summary.get("min_depth_m"):
            errors.append(
                f"info depth_min_m {info.get('depth_min_m')} != depth summary min_depth_m {depth_summary.get('min_depth_m')}"
            )

    # --- Copied files: PLY and manifest checksums match source ---
    pkg_ply = dataset_dir / "meta" / "pointcloud.ply"
    src_ply = trajectory_dir / "pointcloud.ply"
    if pkg_ply.is_file() and src_ply.is_file():
        if _sha256_file(pkg_ply) != _sha256_file(src_ply):
            errors.append("meta/pointcloud.ply checksum != trajectory pointcloud.ply")

    pkg_manifest = dataset_dir / "meta" / "trajectory_manifest.json"
    src_manifest = trajectory_dir / "trajectory_manifest.json"
    if pkg_manifest.is_file() and src_manifest.is_file():
        if _sha256_file(pkg_manifest) != _sha256_file(src_manifest):
            errors.append("meta/trajectory_manifest.json checksum != trajectory manifest")

    # --- Copied render summaries match source ---
    for summary_name in ("render_summary.json", "rgb_render_summary.json", "depth_render_summary.json"):
        pkg_summary = dataset_dir / "meta" / summary_name
        src_summary = rendered_dir / summary_name
        if pkg_summary.is_file() and src_summary.is_file():
            if _sha256_file(pkg_summary) != _sha256_file(src_summary):
                errors.append(f"meta/{summary_name} checksum != rendered/{summary_name}")

    # --- Copied RGB/depth files match source ---
    for rgb_file in pkg["rgb_files"][:3]:  # Check first few for efficiency.
        src = rendered_dir / "observation.images.rgb" / rgb_file.name
        if src.is_file():
            if _sha256_file(rgb_file) != _sha256_file(src):
                errors.append(f"RGB file {rgb_file.name} checksum != rendered source")
                break

    for depth_file in pkg["depth_files"][:3]:
        src = rendered_dir / "observation.images.depth" / depth_file.name
        if src.is_file():
            if _sha256_file(depth_file) != _sha256_file(src):
                errors.append(f"depth file {depth_file.name} checksum != rendered source")
                break

    # --- Episodes JSONL order matches manifest ---
    try:
        episodes_jsonl = _read_jsonl(dataset_dir / "meta" / "episodes.jsonl")
        if len(episodes_jsonl) != expected_episodes:
            errors.append(f"episodes.jsonl count {len(episodes_jsonl)} != manifest {expected_episodes}")
        for i, (ep_rec, ep_manifest) in enumerate(zip(episodes_jsonl, manifest["episodes"])):
            if ep_rec.get("episode_index") != ep_manifest["episode_index"]:
                errors.append(f"episodes.jsonl[{i}] episode_index mismatch")
                break
            if ep_rec.get("frame_count") != ep_manifest["frame_count"]:
                errors.append(f"episodes.jsonl[{i}] frame_count mismatch")
                break
    except Exception as e:
        errors.append(f"episodes.jsonl read failed: {e}")

    eligible = len(errors) == 0
    return {
        "eligible": eligible,
        "errors": errors,
        "warnings": [],
        "scene_id": scene_id,
        "episode_count": expected_episodes,
        "artifact_digests": {
            "packaged_root": digest_directory("packaged_root", dataset_dir),
            "trajectory_root": digest_directory("trajectory", trajectory_dir),
            "rendered_root": digest_directory("rendered_root", rendered_dir),
        },
    }


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

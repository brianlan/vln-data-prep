#!/usr/bin/env python3
"""Read-only package-Python checker for generated trajectory artifacts.

Two modes per the plan (Checker execution contract, revision 8):

- ``validate``: baseline-independent schema/inventory/cross-artifact consistency.
- ``compare-golden``: runs validate, then exact-array, manifest (canonical JSON
  with normalized ``scene_dir``/``collision_usd``), exact PLY bytes, and decoded
  visualization-image equality against a baseline trajectory root.

Package-safe: stdlib + numpy + PIL. No forbidden imports.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Make the canonical helpers importable when run from the repo checkout.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "tests" / "package_safe") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tests" / "package_safe"))

from artifact_parsers import (  # noqa: E402
    parse_binary_ply,
    parse_episode_npz,
    parse_trajectory_manifest,
)
from canonical.digest import (  # noqa: E402
    digest_arrays,
    digest_file,
    digest_json,
)

NPZ_KEYS = (
    "points",
    "actions",
    "camera_positions",
    "yaw",
    "point_goal",
    "start_position",
    "goal_position",
)
VIZ_FILES = ("navigation_map.png", "trajectories_overlay.png")
# Equality-required manifest path fields that are normalized for comparison.
NORMALIZED_PATH_FIELDS = ("scene_dir", "collision_usd")


def _atomic_write_json(path: Path, value: Any) -> None:
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


def _normalize_path_field(value: str) -> str:
    """Normalize a manifest path field: POSIX separators, strip trailing /."""
    return value.replace("\\", "/").rstrip("/")


def _normalize_manifest(manifest: dict) -> dict:
    """Return a copy of manifest with the two named path fields normalized."""
    out = dict(manifest)
    for field in NORMALIZED_PATH_FIELDS:
        if field in out and isinstance(out[field], str):
            out[field] = _normalize_path_field(out[field])
    return out


def _load_npz_arrays(trajectory_dir: Path, episode_index: int) -> dict[str, np.ndarray]:
    path = trajectory_dir / f"episode_{episode_index:06d}.npz"
    data = np.load(path)
    return {key: data[key] for key in data.files}


def _load_viz_pixels(trajectory_dir: Path) -> dict[str, bytes]:
    """Load decoded PNG bytes for each viz file."""
    from PIL import Image
    import io

    result = {}
    for name in VIZ_FILES:
        path = trajectory_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing viz file: {path}")
        img = Image.open(path)
        result[name] = np.array(img)
    return result


def validate(trajectory_dir: Path) -> dict[str, Any]:
    """Run baseline-independent validation. Returns result dict."""
    errors: list[str] = []

    # Manifest schema.
    try:
        manifest = parse_trajectory_manifest(trajectory_dir)
    except Exception as e:
        return {"eligible": False, "errors": [str(e)], "warnings": []}

    episode_count = manifest["episode_count"]
    episodes = manifest["episodes"]

    # Cross-artifact: NPZ files exist and have correct schema.
    for ep in episodes:
        idx = ep["episode_index"]
        try:
            arrays = _load_npz_arrays(trajectory_dir, idx)
            keys = set(arrays.keys())
            if keys != set(NPZ_KEYS):
                errors.append(f"episode {idx} NPZ keys {keys} != {set(NPZ_KEYS)}")
                continue
            # Cross-artifact: NPZ frame_count matches manifest frame_count.
            npz_frames = arrays["actions"].shape[0]
            if npz_frames != ep["frame_count"]:
                errors.append(
                    f"episode {idx} NPZ frame_count {npz_frames} != manifest {ep['frame_count']}"
                )
        except Exception as e:
            errors.append(f"episode {idx} NPZ load failed: {e}")

    # PLY exists and is parseable.
    ply_path = trajectory_dir / "pointcloud.ply"
    try:
        parse_binary_ply(ply_path)
    except Exception as e:
        errors.append(f"PLY parse failed: {e}")

    # Viz files exist.
    for name in VIZ_FILES:
        if not (trajectory_dir / name).is_file():
            errors.append(f"missing viz file: {name}")

    eligible = len(errors) == 0
    return {
        "eligible": eligible,
        "errors": errors,
        "warnings": [],
        "episode_count": episode_count,
        "scene_id": manifest.get("scene_id"),
    }


def compare_golden(
    trajectory_dir: Path,
    baseline_dir: Path,
    *,
    run_provenance: Path | None = None,
    baseline_provenance: Path | None = None,
) -> dict[str, Any]:
    """Run validate then exact comparison against baseline. Returns result dict."""
    # Validate candidate first.
    val = validate(trajectory_dir)
    if not val["eligible"]:
        return {
            "eligible": False,
            "errors": ["validate failed"] + val["errors"],
            "warnings": [],
            "artifact_digests": {},
        }

    # Validate baseline.
    base_val = validate(baseline_dir)
    if not base_val["eligible"]:
        return {
            "eligible": False,
            "errors": ["baseline validate failed"] + base_val["errors"],
            "warnings": [],
            "artifact_digests": {},
        }

    errors: list[str] = []
    artifact_digests: dict[str, str] = {}

    # Load manifests and compare canonical JSON with normalized path fields.
    with (trajectory_dir / "trajectory_manifest.json").open() as f:
        cand_manifest = json.load(f)
    with (baseline_dir / "trajectory_manifest.json").open() as f:
        base_manifest = json.load(f)

    cand_norm = _normalize_manifest(cand_manifest)
    base_norm = _normalize_manifest(base_manifest)
    cand_manifest_digest = digest_json("trajectory", cand_norm)
    base_manifest_digest = digest_json("trajectory", base_norm)
    artifact_digests["manifest"] = cand_manifest_digest
    if cand_manifest_digest != base_manifest_digest:
        errors.append("manifest canonical JSON mismatch (after path normalization)")

    # Compare exact NPZ arrays per episode.
    for ep in cand_manifest["episodes"]:
        idx = ep["episode_index"]
        cand_arrays = _load_npz_arrays(trajectory_dir, idx)
        base_arrays = _load_npz_arrays(baseline_dir, idx)
        # Check key sets match.
        if set(cand_arrays.keys()) != set(base_arrays.keys()):
            errors.append(f"episode {idx} NPZ key set mismatch")
            continue
        cand_digest = digest_arrays("trajectory", f"episode_{idx:06d}", cand_arrays)
        base_digest = digest_arrays("trajectory", f"episode_{idx:06d}", base_arrays)
        artifact_digests[f"episode_{idx:06d}"] = cand_digest
        if cand_digest != base_digest:
            errors.append(f"episode {idx} array digest mismatch")

    # Compare exact PLY bytes.
    cand_ply_digest = digest_file("trajectory", trajectory_dir / "pointcloud.ply")
    base_ply_digest = digest_file("trajectory", baseline_dir / "pointcloud.ply")
    artifact_digests["pointcloud_ply"] = cand_ply_digest
    if cand_ply_digest != base_ply_digest:
        errors.append("PLY exact bytes mismatch")

    # Compare decoded viz images.
    try:
        cand_viz = _load_viz_pixels(trajectory_dir)
        base_viz = _load_viz_pixels(baseline_dir)
        for name in VIZ_FILES:
            if np.array_equal(cand_viz[name], base_viz[name]):
                continue
            errors.append(f"viz {name} decoded pixel mismatch")
    except Exception as e:
        errors.append(f"viz comparison failed: {e}")

    # Verify equality-required provenance fields if provided.
    if run_provenance and baseline_provenance:
        try:
            with run_provenance.open() as f:
                rp = json.load(f)
            with baseline_provenance.open() as f:
                bp = json.load(f)
            eq_fields = ("plan_revision", "baseline_id")
            for field in eq_fields:
                if rp.get(field) != bp.get(field):
                    errors.append(f"provenance {field} mismatch: {rp.get(field)} != {bp.get(field)}")
        except Exception as e:
            errors.append(f"provenance load failed: {e}")

    eligible = len(errors) == 0
    return {
        "eligible": eligible,
        "errors": errors,
        "warnings": [],
        "artifact_digests": artifact_digests,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAGE3D generation artifact checker")
    sub = p.add_subparsers(dest="mode", required=True)

    pv = sub.add_parser("validate", help="Baseline-independent validation")
    pv.add_argument("--trajectory-dir", type=Path, required=True)
    pv.add_argument("--result-path", type=Path, default=None)

    pg = sub.add_parser("compare-golden", help="Compare against a golden baseline")
    pg.add_argument("--trajectory-dir", type=Path, required=True)
    pg.add_argument("--baseline-dir", type=Path, required=True)
    pg.add_argument("--baseline-provenance", type=Path, default=None)
    pg.add_argument("--run-provenance", type=Path, default=None)
    pg.add_argument("--result-path", type=Path, default=None)

    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.mode == "validate":
        result = validate(args.trajectory_dir)
    elif args.mode == "compare-golden":
        result = compare_golden(
            args.trajectory_dir,
            args.baseline_dir,
            run_provenance=args.run_provenance,
            baseline_provenance=args.baseline_provenance,
        )
    else:
        print(f"unknown mode: {args.mode}", file=sys.stderr)
        return 2

    result["checker"] = "check_generate"
    result["mode"] = args.mode

    # Write atomic JSON result if requested.
    if args.result_path:
        _atomic_write_json(args.result_path, result)

    # Human-readable summary.
    status = "ELIGIBLE" if result["eligible"] else "INELIGIBLE"
    print(f"[check_generate:{args.mode}] {status}")
    for err in result.get("errors", []):
        print(f"  ERROR: {err}")

    return 0 if result["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
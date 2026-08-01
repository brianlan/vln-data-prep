#!/usr/bin/env python3
"""Execute the pinned SAGE3D Phase 0b canonical capture protocol."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sage3d_canonical.digest import digest_directory  # noqa: E402
from sage3d_canonical.provenance import (  # noqa: E402
    _atomic_write_json,
    _sha256_file,
    build_checker_result,
    build_run_provenance,
    build_stage_run,
    build_verification_manifest,
    is_clean_commit,
    write_run_provenance,
    write_verification_manifest,
)
from sage3d.publication import create_named_directory as _create_named_directory  # noqa: E402


SCENE = "839920"
SEED = 20260720
EPISODES = 5
PLAN_REVISION = 8
PLAN_COMMIT = "f8548ca6ac7bf9b9f9d16ff7e49cf8e3cc8c5a63"
BASELINE_ID = "phase0a-pre-observation"
CHARACTERIZATION_RUNS = 5
HELD_OUT_RUNS = 2

CONFIG = {
    "scene": SCENE,
    "seed": SEED,
    "episodes": EPISODES,
    "max_attempts": 3000,
    "robot_radius_m": 0.25,
    "safety_margin_m": 0.05,
    "camera_height_m": 0.6,
    "min_path_length_m": 3.0,
    "max_path_length_m": 15.0,
    "frame_spacing_m": 0.05,
    "width": 600,
    "height": 450,
    "horizontal_fov_deg": 180.0,
    "fisheye_coefficients": [0.1, 0.0, 0.0, 0.0],
    "max_depth_m": 6.0,
    "min_depth_m": 0.05,
    "depth_scale": 10000.0,
    "settle_steps": 10,
    "startup_steps": 40,
}

FRAME_METRIC_TO_THRESHOLD = {
    "rgb_mask_leakage_mean": "rgb_mask_leakage_mean_max",
    "rgb_masked_rmse": "rgb_masked_rmse",
    "rgb_masked_abs_error_p99": "rgb_masked_abs_error_p99",
    "depth_non_max_mask_iou": "depth_non_max_mask_iou",
    "depth_error_p50": "depth_error_p50",
    "depth_error_p95": "depth_error_p95",
    "depth_error_p99": "depth_error_p99",
}


class StageFailure(RuntimeError):
    def __init__(self, stage: str, record: dict[str, Any]) -> None:
        super().__init__(f"{stage} failed with exit code {record['exit_code']}")
        self.stage = stage
        self.record = record


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_status(path: Path, value: dict[str, Any]) -> None:
    """Write replaceable non-binding status beside immutable evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def allocate_directory(parent: Path, name: str) -> Path:
    """Create one absent, non-symlinked child on the parent's filesystem.

    Delegates to :func:`sage3d.publication.create_named_directory` so the
    canonical harness shares the single production allocation/safety formula.
    From Phase 1 onward an independent canonical-harness allocation formula is
    forbidden.
    """
    return _create_named_directory(parent, name)


def derive_threshold_report(
    policy: dict[str, Any],
    baseline_measurement: dict[str, Any],
    characterization_measurements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Derive binding thresholds from the committed five-run protocol."""
    if len(characterization_measurements) != CHARACTERIZATION_RUNS:
        raise ValueError(
            f"expected {CHARACTERIZATION_RUNS} characterization results, "
            f"got {len(characterization_measurements)}"
        )

    observations: dict[str, list[float]] = {
        threshold_name: [] for threshold_name in FRAME_METRIC_TO_THRESHOLD.values()
    }
    baseline_frames = baseline_measurement["metrics"]["per_frame"]
    observations["rgb_mask_leakage_mean_max"].extend(
        float(frame["rgb_mask_leakage_mean"]) for frame in baseline_frames
    )

    for result in characterization_measurements:
        if not result.get("eligible"):
            raise ValueError("ineligible characterization measurement")
        for frame in result["metrics"]["per_frame"]:
            for frame_name, threshold_name in FRAME_METRIC_TO_THRESHOLD.items():
                if frame_name in frame:
                    observations[threshold_name].append(float(frame[frame_name]))

    margins = policy["pre_observation_minimum_margins"]
    margin_names = {
        "rgb_mask_leakage_mean_max": "rgb_leakage_min_margin",
        "rgb_masked_rmse": "rgb_rmse_min_margin",
        "rgb_masked_abs_error_p99": "rgb_p99_min_margin",
        "depth_error_p50": "depth_error_min_margin",
        "depth_error_p95": "depth_error_min_margin",
        "depth_error_p99": "depth_error_min_margin",
    }
    thresholds: dict[str, float] = {}
    derivations: dict[str, dict[str, Any]] = {}
    for metric, values in observations.items():
        if not values:
            raise ValueError(f"no observations for {metric}")
        if metric == "depth_non_max_mask_iou":
            observed = min(values)
            margin = max(
                0.25 * (1.0 - observed),
                float(margins["depth_iou_min_margin"]),
            )
            threshold = max(0.0, observed - margin)
            formula = "max(0, min_observed - max(0.25 * (1 - min_observed), depth_iou_min_margin))"
        else:
            observed = max(values)
            margin_name = margin_names[metric]
            margin = max(0.25 * observed, float(margins[margin_name]))
            threshold = observed + margin
            formula = f"max_observed + max(0.25 * max_observed, {margin_name})"
        thresholds[metric] = threshold
        derivations[metric] = {
            "formula": formula,
            "observation_count": len(values),
            "observed_extreme": observed,
            "applied_margin": margin,
            "threshold": threshold,
            "observations": values,
        }

    return {
        "schema_version": 1,
        "baseline_id": policy["baseline_id"],
        "plan_revision": policy["plan_revision"],
        "plan_commit": policy["plan_commit"],
        "status": "pending-independent-approval",
        "source_policy_sha256": None,
        "characterization_run_count": len(characterization_measurements),
        "held_out_runs_contributed": 0,
        "thresholds": thresholds,
        "derivations": derivations,
        "approval": {
            "status": "pending",
            "requirement": "independent approval or an explicit owner-approved exception",
        },
    }


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _command_output(argv: list[str], cwd: Path) -> str:
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    return (result.stdout + result.stderr).strip()


def _run_stage(
    stage: str,
    argv: list[str],
    repo_root: Path,
    logs_dir: Path,
    *,
    environment: dict[str, str],
    accepted_exit_codes: tuple[int, ...] = (0,),
) -> dict[str, Any]:
    log_path = logs_dir / f"{stage}.log"
    started = _utc_now()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            argv,
            cwd=repo_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        exit_code = process.wait()
    record = build_stage_run(
        stage=stage,
        pid=process.pid,
        argv=argv,
        cwd=str(repo_root),
        environment={
            key: environment[key]
            for key in (
                "SAGE3D_ISAAC_PYTHON",
                "SAGE3D_PACKAGE_PYTHON",
                "PYTHONPATH",
            )
            if key in environment
        },
        started_at=started,
        completed_at=_utc_now(),
        exit_code=exit_code,
    )
    record["log"] = str(log_path)
    if exit_code not in accepted_exit_codes:
        raise StageFailure(stage, record)
    return record


def _resolve_scene_dir(interior_root: Path) -> Path:
    direct = interior_root / SCENE
    if direct.is_dir():
        return direct
    matches = sorted(interior_root.glob(f"*_{SCENE}"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one InteriorGS scene for {SCENE}, got {matches}")
    return matches[0]


def _asset_paths(sage3d_root: Path) -> dict[str, Path]:
    paths = {
        "interior_root": sage3d_root / "InteriorGS",
        "usdz": sage3d_root / "InteriorGS_usdz" / f"{SCENE}.usdz",
        "collision_usd": (
            sage3d_root
            / "Collision_Mesh"
            / "Collision_Mesh"
            / SCENE
            / f"{SCENE}_collision.usd"
        ),
    }
    paths["scene_dir"] = _resolve_scene_dir(paths["interior_root"])
    for name, path in paths.items():
        if name == "interior_root":
            if not path.is_dir():
                raise FileNotFoundError(path)
        elif name == "scene_dir":
            if not path.is_dir():
                raise FileNotFoundError(path)
        elif not path.is_file():
            raise FileNotFoundError(path)
    return paths


def _runtime_fingerprint(
    repo_root: Path,
    isaac_python: Path,
    package_python: Path,
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "captured_at": _utc_now(),
        "gpu": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            repo_root,
        ),
        "generation": {
            "python": str(isaac_python),
            "version": _command_output([str(isaac_python), "--version"], repo_root),
            "asset_hashes": input_hashes,
        },
        "rendering": {
            "python": str(isaac_python),
            "version": _command_output([str(isaac_python), "--version"], repo_root),
            "gpu_bound": True,
            "asset_hashes": {
                key: input_hashes[key] for key in ("usdz", "collision_usd")
            },
        },
        "packaging": {
            "python": str(package_python),
            "version": _command_output([str(package_python), "--version"], repo_root),
            "pyarrow": _command_output(
                [
                    str(package_python),
                    "-c",
                    "import pyarrow; print(pyarrow.__version__)",
                ],
                repo_root,
            ),
        },
    }


def _base_environment(
    repo_root: Path,
    isaac_python: Path,
    package_python: Path,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "SAGE3D_ISAAC_PYTHON": str(isaac_python),
            "SAGE3D_PACKAGE_PYTHON": str(package_python),
            "PYTHONPATH": str(repo_root),
        }
    )
    return environment


def _generation_command(
    isaac_python: Path,
    repo_root: Path,
    assets: dict[str, Path],
    output_dir: Path,
) -> list[str]:
    return [
        str(isaac_python),
        str(repo_root / "generate_sage3d_trajectories.py"),
        "--scene",
        SCENE,
        "--interiorgs-root",
        str(assets["interior_root"]),
        "--collision-usd",
        str(assets["collision_usd"]),
        "--output-dir",
        str(output_dir),
        "--episodes",
        str(EPISODES),
        "--seed",
        str(SEED),
        "--robot-radius",
        str(CONFIG["robot_radius_m"]),
        "--safety-margin",
        str(CONFIG["safety_margin_m"]),
        "--camera-height",
        str(CONFIG["camera_height_m"]),
        "--min-path-length",
        str(CONFIG["min_path_length_m"]),
        "--max-path-length",
        str(CONFIG["max_path_length_m"]),
        "--frame-spacing",
        str(CONFIG["frame_spacing_m"]),
        "--max-attempts",
        str(CONFIG["max_attempts"]),
    ]


def _render_command(
    mode: str,
    isaac_python: Path,
    repo_root: Path,
    assets: dict[str, Path],
    trajectory_dir: Path,
    output_dir: Path,
) -> list[str]:
    return [
        str(isaac_python),
        str(repo_root / "render_fisheye_sage3d.py"),
        "--mode",
        mode,
        "--scene",
        SCENE,
        "--usdz",
        str(assets["usdz"]),
        "--collision-usd",
        str(assets["collision_usd"]),
        "--trajectory-dir",
        str(trajectory_dir),
        "--output-dir",
        str(output_dir),
        "--width",
        str(CONFIG["width"]),
        "--height",
        str(CONFIG["height"]),
        "--horizontal-fov-deg",
        str(CONFIG["horizontal_fov_deg"]),
        "--fisheye-coefficients",
        *(str(value) for value in CONFIG["fisheye_coefficients"]),
        "--max-depth-m",
        str(CONFIG["max_depth_m"]),
        "--min-depth-m",
        str(CONFIG["min_depth_m"]),
        "--depth-scale",
        str(CONFIG["depth_scale"]),
        "--settle-steps",
        str(CONFIG["settle_steps"]),
        "--startup-steps",
        str(CONFIG["startup_steps"]),
    ]


def _package_command(
    package_python: Path,
    repo_root: Path,
    trajectory_dir: Path,
    rendered_dir: Path,
    output_dir: Path,
) -> list[str]:
    return [
        str(package_python),
        str(repo_root / "package_lerobot_sage3d.py"),
        "--scene",
        SCENE,
        "--trajectory-dir",
        str(trajectory_dir),
        "--rendered-dir",
        str(rendered_dir),
        "--output-dir",
        str(output_dir),
        "--width",
        str(CONFIG["width"]),
        "--height",
        str(CONFIG["height"]),
        "--horizontal-fov-deg",
        str(CONFIG["horizontal_fov_deg"]),
        "--fisheye-coefficients",
        *(str(value) for value in CONFIG["fisheye_coefficients"]),
        "--camera-height",
        str(CONFIG["camera_height_m"]),
    ]


def _artifact_digests(
    trajectory_dir: Path,
    rendered_dir: Path | None = None,
    dataset_dir: Path | None = None,
) -> dict[str, str]:
    digests = {"trajectory": digest_directory("trajectory", trajectory_dir)}
    if rendered_dir is not None:
        digests["rendered_root"] = digest_directory("rendered_root", rendered_dir)
    if dataset_dir is not None:
        digests["packaged_root"] = digest_directory("packaged_root", dataset_dir)
    return digests


def _write_provenance(
    path: Path,
    repo_root: Path,
    commit: str,
    input_hashes: dict[str, str],
    runtime_fingerprint: dict[str, Any],
    cache_policy: dict[str, str],
    stage_runs: list[dict[str, Any]],
    artifact_digests: dict[str, str],
) -> None:
    provenance = build_run_provenance(
        plan_revision=PLAN_REVISION,
        plan_commit=PLAN_COMMIT,
        baseline_id=BASELINE_ID,
        candidate_commit=commit,
        dirty_tree=not is_clean_commit(commit, repo_root),
        normalized_config=CONFIG,
        input_hashes=input_hashes,
        artifact_digests=artifact_digests,
        runtime_fingerprint=runtime_fingerprint,
        cache_policy=cache_policy,
        stage_runs=stage_runs,
        submodule_state={"status": _git(repo_root, "submodule", "status")},
    )
    write_run_provenance(path, provenance)


def _run_checker(
    name: str,
    argv: list[str],
    repo_root: Path,
    logs_dir: Path,
    environment: dict[str, str],
    result_path: Path,
    *,
    expected_eligible: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _run_stage(
        name,
        argv,
        repo_root,
        logs_dir,
        environment=environment,
        accepted_exit_codes=(0, 1),
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if bool(result.get("eligible")) is not expected_eligible:
        raise RuntimeError(
            f"{name} eligible={result.get('eligible')}, expected {expected_eligible}"
        )
    if record["exit_code"] != (0 if expected_eligible else 1):
        raise RuntimeError(f"{name} exit/result disagreement")
    return record, result


def _checker_result(record: dict[str, Any], result_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return build_checker_result(
        checker=result["checker"],
        mode=result["mode"],
        result_sha256=_sha256_file(result_path),
        exit_code=record["exit_code"],
        eligible=bool(result["eligible"]),
        artifact_digests=result.get("artifact_digests", {}),
    )


def _create_render_stage(
    final_target: Path,
    repo_root: Path,
    package_python: Path,
    environment: dict[str, str],
) -> Path:
    """Allocate a shared render staging directory via the production CLI.

    Invokes ``python -m sage3d.cli.create_staging --final-target <path>``
    under the package-safe interpreter and returns the absolute staging path
    printed to stdout. The staging directory is a real sibling of
    ``final_target`` allocated by
    :func:`~sage3d.publication.create_staging_directory`.
    """
    argv = [
        str(package_python),
        "-m",
        "sage3d.cli.create_staging",
        "--final-target",
        str(final_target),
    ]
    proc = subprocess.run(
        argv,
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"create_staging failed (exit {proc.returncode}):\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )
    staging = Path(proc.stdout.strip())
    if not staging.is_absolute() or not staging.is_dir():
        raise RuntimeError(
            f"create_staging returned invalid path: {staging!r}"
        )
    return staging


def _render_pair(
    name: str,
    parent: Path,
    trajectory_dir: Path,
    repo_root: Path,
    assets: dict[str, Path],
    isaac_python: Path,
    environment: dict[str, str],
    *,
    package_python: Path,
    package: bool = True,
) -> dict[str, Any]:
    run_dir = allocate_directory(parent, name)
    logs_dir = allocate_directory(run_dir, "logs")
    final_target = run_dir / "rendered"
    rendered_dir = _create_render_stage(
        final_target, repo_root, package_python, environment
    )
    records = [
        _run_stage(
            f"{name}-render-rgb",
            _render_command(
                "rgb", isaac_python, repo_root, assets, trajectory_dir, rendered_dir
            ),
            repo_root,
            logs_dir,
            environment=environment,
        ),
        _run_stage(
            f"{name}-render-depth",
            _render_command(
                "depth", isaac_python, repo_root, assets, trajectory_dir, rendered_dir
            ),
            repo_root,
            logs_dir,
            environment=environment,
        ),
    ]
    dataset_dir = None
    if package:
        dataset_dir = allocate_directory(run_dir, "dataset")
        records.append(
            _run_stage(
                f"{name}-package",
                _package_command(
                    package_python,
                    repo_root,
                    trajectory_dir,
                    rendered_dir,
                    dataset_dir,
                ),
                repo_root,
                logs_dir,
                environment=environment,
            )
        )
    return {
        "name": name,
        "run_dir": run_dir,
        "logs_dir": logs_dir,
        "rendered_dir": rendered_dir,
        "dataset_dir": dataset_dir,
        "stage_runs": records,
    }


def _break_hardlink(path: Path, baseline_path: Path) -> None:
    path.unlink()
    shutil.copy2(baseline_path, path)


def _save_rgb(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8)).save(path, quality=95)


def _save_depth(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint16)).save(path)


def _circular_mask(summary: dict[str, Any]) -> np.ndarray:
    width, height = summary["resolution"]
    cx, cy = summary["principal_point"]
    radius = summary["forward_mask_radius_pixels"]
    yy, xx = np.ogrid[:height, :width]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def _mutation_cases(
    baseline_rendered: Path,
    trajectory_dir: Path,
    threshold_report: dict[str, Any],
) -> list[tuple[str, str, bool, Callable[[np.ndarray, np.ndarray], np.ndarray]]]:
    del baseline_rendered, trajectory_dir
    sentinel = int(
        np.rint(
            np.float32(CONFIG["max_depth_m"]) * np.float32(CONFIG["depth_scale"])
        )
    )
    thresholds = threshold_report["thresholds"]
    one_code = 1.0 / 255.0
    one_code_pass = (
        one_code <= thresholds["rgb_masked_rmse"]
        and one_code <= thresholds["rgb_masked_abs_error_p99"]
    )

    def rgb_shift(amount: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        return lambda array, mask: np.where(
            mask[..., None],
            np.clip(array.astype(np.int16) + amount, 0, 255),
            array,
        ).astype(np.uint8)

    def rgb_outside(value: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        def mutate(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
            result = array.copy()
            result[~mask] = value
            return result
        return mutate

    def rgb_block(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
        del mask
        result = array.copy()
        result[100:170, 100:170] = 255 - result[100:170, 100:170]
        return result

    def rgb_near_uniform(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = np.zeros_like(array)
        result[mask] = 127
        return result

    def depth_all(value: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        return lambda array, mask: np.full_like(array, value)

    def depth_inside(
        operation: Callable[[np.ndarray], np.ndarray],
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        def mutate(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
            result = array.copy()
            valid = mask & (result != sentinel)
            result[valid] = operation(result[valid])
            return result
        return mutate

    def depth_outside(value: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        def mutate(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
            result = array.copy()
            result[~mask] = value
            return result
        return mutate

    def erode_nonmax(pixels: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        def mutate(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
            result = array.copy()
            inner = mask.copy()
            for _ in range(pixels):
                inner = (
                    inner
                    & np.roll(inner, 1, 0)
                    & np.roll(inner, -1, 0)
                    & np.roll(inner, 1, 1)
                    & np.roll(inner, -1, 1)
                )
            result[mask & ~inner] = sentinel
            return result
        return mutate

    def expand_nonmax(pixels: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        def mutate(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
            result = array.copy()
            original = mask & (result != sentinel)
            expanded = original.copy()
            for _ in range(pixels):
                expanded = mask & (
                    expanded
                    | np.roll(expanded, 1, 0)
                    | np.roll(expanded, -1, 0)
                    | np.roll(expanded, 1, 1)
                    | np.roll(expanded, -1, 1)
                )
            result[expanded & ~original] = sentinel - 1000
            return result
        return mutate

    return [
        ("rgb-all-black", "rgb", False, lambda a, m: np.zeros_like(a)),
        ("rgb-near-uniform", "rgb", False, rgb_near_uniform),
        ("rgb-channel-swap", "rgb", False, lambda a, m: a[..., [2, 1, 0]]),
        ("rgb-horizontal-flip", "rgb", False, lambda a, m: a[:, ::-1, :]),
        ("rgb-vertical-flip", "rgb", False, lambda a, m: a[::-1, :, :]),
        ("rgb-changed-forward-mask", "rgb", False, rgb_outside(255)),
        ("rgb-block-corruption", "rgb", False, rgb_block),
        ("rgb-multi-code-shift", "rgb", False, rgb_shift(5)),
        ("depth-all-sentinel", "depth", False, depth_all(sentinel)),
        (
            "depth-wrong-scale",
            "depth",
            False,
            depth_inside(lambda values: values // 2),
        ),
        ("depth-missing-sentinel", "depth", False, depth_outside(0)),
        ("depth-horizontal-flip", "depth", False, lambda a, m: a[:, ::-1]),
        ("depth-vertical-flip", "depth", False, lambda a, m: a[::-1, :]),
        ("depth-changed-outside-mask", "depth", False, depth_outside(sentinel - 1)),
        ("depth-eroded-nonmax-mask", "depth", False, erode_nonmax(12)),
        ("depth-expanded-nonmax-mask", "depth", False, expand_nonmax(12)),
        (
            "depth-constant-offset",
            "depth",
            False,
            depth_inside(
                lambda values: np.clip(
                    values.astype(np.int32) + 50, 0, 65535
                ).astype(np.uint16)
            ),
        ),
        ("rgb-one-code-boundary", "rgb", one_code_pass, rgb_shift(1)),
        ("rgb-natural-jpeg-edge-leakage", "rgb", True, lambda a, m: a),
        ("rgb-widespread-one-code-leakage", "rgb", False, rgb_outside(1)),
        ("rgb-high-intensity-exterior-leakage", "rgb", False, rgb_outside(255)),
        ("depth-small-boundary-perturbation", "depth", True, erode_nonmax(1)),
        ("depth-meaningful-mask-area-loss", "depth", False, erode_nonmax(20)),
    ]


def _pick_offset_source(
    baseline_rendered: Path,
    target_stem: str,
    modality: str,
) -> Path:
    suffix = ".jpg" if modality == "rgb" else ".png"
    directory = baseline_rendered / f"observation.images.{modality}"
    target = np.asarray(Image.open(directory / f"{target_stem}{suffix}"), dtype=np.float64)
    candidates = sorted(directory.glob(f"episode_000000_*{suffix}"))
    return max(
        (path for path in candidates if path.stem != target_stem),
        key=lambda path: float(
            np.mean(
                np.abs(np.asarray(Image.open(path), dtype=np.float64) - target)
            )
        ),
    )


def _run_mutation_suite(
    attempt_dir: Path,
    baseline_rendered: Path,
    trajectory_dir: Path,
    baseline_provenance: Path,
    candidate_provenance: Path,
    policy_path: Path,
    threshold_report_path: Path,
    package_python: Path,
    repo_root: Path,
    environment: dict[str, str],
) -> tuple[Path, dict[str, Any]]:
    mutations_dir = allocate_directory(attempt_dir, "mutations")
    logs_dir = allocate_directory(mutations_dir, "logs")
    summary = json.loads(
        (baseline_rendered / "depth_render_summary.json").read_text(encoding="utf-8")
    )
    mask = _circular_mask(summary)
    manifest = json.loads(
        (trajectory_dir / "trajectory_manifest.json").read_text(encoding="utf-8")
    )
    target_frame = manifest["episodes"][0]["frame_count"] // 2
    target_stem = f"episode_000000_{target_frame:03d}"
    threshold_report = json.loads(threshold_report_path.read_text(encoding="utf-8"))
    cases = _mutation_cases(baseline_rendered, trajectory_dir, threshold_report)
    cases.extend(
        [
            ("rgb-one-frame-offset", "rgb", False, lambda a, m: a),
            ("depth-one-frame-offset", "depth", False, lambda a, m: a),
        ]
    )
    results = []
    for case_name, modality, expected_eligible, mutator in cases:
        case_target_stem = (
            "episode_000003_000"
            if case_name == "depth-expanded-nonmax-mask"
            else target_stem
        )
        case_dir = mutations_dir / case_name
        if os.path.lexists(case_dir):
            raise FileExistsError(case_dir)
        shutil.copytree(baseline_rendered, case_dir, copy_function=os.link)
        suffix = ".jpg" if modality == "rgb" else ".png"
        target = (
            case_dir
            / f"observation.images.{modality}"
            / f"{case_target_stem}{suffix}"
        )
        baseline_target = (
            baseline_rendered
            / f"observation.images.{modality}"
            / f"{case_target_stem}{suffix}"
        )
        _break_hardlink(target, baseline_target)
        if case_name.endswith("one-frame-offset"):
            source = _pick_offset_source(
                baseline_rendered, case_target_stem, modality
            )
            shutil.copy2(source, target)
            source_frame = source.stem
        elif case_name == "rgb-natural-jpeg-edge-leakage":
            source_frame = None
        else:
            array = np.asarray(Image.open(target)).copy()
            mutated = mutator(array, mask)
            (_save_rgb if modality == "rgb" else _save_depth)(target, mutated)
            source_frame = None

        result_path = case_dir / "checker_result.json"
        checker_argv = [
            str(package_python),
            str(repo_root / "scripts" / "check_render.py"),
            "compare-golden",
            "--rendered-dir",
            str(case_dir),
            "--trajectory-dir",
            str(trajectory_dir),
            "--baseline-dir",
            str(baseline_rendered),
            "--baseline-provenance",
            str(baseline_provenance),
            "--run-provenance",
            str(candidate_provenance),
            "--tolerance-policy",
            str(policy_path),
            "--threshold-report",
            str(threshold_report_path),
            "--result-path",
            str(result_path),
        ]
        record, result = _run_checker(
            f"mutation-{case_name}",
            checker_argv,
            repo_root,
            logs_dir,
            environment,
            result_path,
            expected_eligible=expected_eligible,
        )
        results.append(
            {
                "name": case_name,
                "modality": modality,
                "target_frame": case_target_stem,
                "source_frame": source_frame,
                "expected_eligible": expected_eligible,
                "observed_eligible": result["eligible"],
                "detected_as_expected": bool(result["eligible"]) is expected_eligible,
                "checker_result_sha256": _sha256_file(result_path),
                "exit_code": record["exit_code"],
                "errors": result.get("errors", []),
            }
        )

    rgb_target = np.asarray(
        Image.open(
            baseline_rendered
            / "observation.images.rgb"
            / f"{target_stem}.jpg"
        ),
        dtype=np.float64,
    )
    max_channel_difference = float(np.max(np.ptp(rgb_target, axis=2)))
    channel_difference_minimum = 10.0
    report = {
        "schema_version": 1,
        "baseline_id": BASELINE_ID,
        "target_frame": target_stem,
        "channel_swap_precondition": {
            "required_minimum": channel_difference_minimum,
            "observed_max_channel_difference": max_channel_difference,
            "satisfied": max_channel_difference >= channel_difference_minimum,
        },
        "cases": results,
        "all_expected_outcomes_observed": (
            max_channel_difference >= channel_difference_minimum
            and all(result["detected_as_expected"] for result in results)
        ),
    }
    report_path = mutations_dir / "mutation_report.json"
    _atomic_write_json(report_path, report)
    if not report["all_expected_outcomes_observed"]:
        raise RuntimeError("mutation suite did not match committed expectations")
    return report_path, report


def _copy_selected_frames(
    baseline_rendered: Path,
    trajectory_dir: Path,
    destination: Path,
) -> None:
    manifest = json.loads(
        (trajectory_dir / "trajectory_manifest.json").read_text(encoding="utf-8")
    )
    rgb_dir = allocate_directory(destination, "rgb")
    depth_dir = allocate_directory(destination, "depth")
    for episode in manifest["episodes"]:
        frame_count = episode["frame_count"]
        indices = sorted({0, frame_count // 2, frame_count - 1})
        for frame_index in indices:
            stem = f"episode_{episode['episode_index']:06d}_{frame_index:03d}"
            shutil.copy2(
                baseline_rendered / "observation.images.rgb" / f"{stem}.jpg",
                rgb_dir / f"{stem}.jpg",
            )
            shutil.copy2(
                baseline_rendered / "observation.images.depth" / f"{stem}.png",
                depth_dir / f"{stem}.png",
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--sage3d-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--isaac-python", type=Path, default=None)
    parser.add_argument("--package-python", type=Path, default=None)
    parser.add_argument("--cache-policy", required=True)
    parser.add_argument("--cache-limitation", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    sage3d_root = args.sage3d_root.resolve()
    capture_root = args.capture_root.resolve()
    isaac_python = args.isaac_python or Path(
        os.environ["SAGE3D_ISAAC_PYTHON"]
    )
    package_python = args.package_python or Path(
        os.environ["SAGE3D_PACKAGE_PYTHON"]
    )
    for executable in (isaac_python, package_python):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(executable)
    if capture_root.is_symlink() or not capture_root.is_dir():
        raise ValueError(f"capture root must be a real existing directory: {capture_root}")
    forbidden = {Path("/").resolve(), Path.home().resolve(), repo_root, sage3d_root}
    if capture_root in forbidden or capture_root.is_relative_to(sage3d_root):
        raise ValueError(f"unsafe capture root: {capture_root}")

    policy_path = (
        repo_root
        / "tests"
        / "golden"
        / SCENE
        / BASELINE_ID
        / "tolerance_policy.json"
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if (
        policy["baseline_id"] != BASELINE_ID
        or policy["plan_commit"] != PLAN_COMMIT
        or policy["status"] != "pre-observation"
    ):
        raise RuntimeError("unexpected pre-observation tolerance policy")

    commit = _git(repo_root, "rev-parse", "HEAD")
    if not is_clean_commit(commit, repo_root):
        raise RuntimeError("canonical capture requires a clean, resolvable HEAD")
    assets = _asset_paths(sage3d_root)
    input_hashes = {
        "usdz": _sha256_file(assets["usdz"]),
        "collision_usd": _sha256_file(assets["collision_usd"]),
        "scene_dir": digest_directory("evidence", assets["scene_dir"]),
    }
    runtime = _runtime_fingerprint(
        repo_root, isaac_python, package_python, input_hashes
    )
    environment = _base_environment(repo_root, isaac_python, package_python)
    cache_policy = {
        "policy": args.cache_policy,
        "limitation": args.cache_limitation,
    }

    baseline_root = capture_root / BASELINE_ID
    baseline_root.mkdir(exist_ok=True)
    attempt_dir = allocate_directory(baseline_root, args.attempt_id)
    status_path = attempt_dir / "orchestration_status.json"
    capture_started_at = _utc_now()
    _write_status(
        status_path,
        {
            "status": "running",
            "started_at": capture_started_at,
            "candidate_commit": commit,
        },
    )

    try:
        logs_dir = allocate_directory(attempt_dir, "logs")
        runs_dir = allocate_directory(attempt_dir, "runs")
        evidence_dir = allocate_directory(attempt_dir, "evidence")

        # Designated baseline: generate, two fresh render processes, package.
        baseline_dir = allocate_directory(runs_dir, "baseline")
        baseline_logs = allocate_directory(baseline_dir, "logs")
        baseline_trajectory = allocate_directory(baseline_dir, "trajectories")
        baseline_records = [
            _run_stage(
                "baseline-generate",
                _generation_command(
                    isaac_python, repo_root, assets, baseline_trajectory
                ),
                repo_root,
                baseline_logs,
                environment=environment,
            )
        ]
        baseline_rendered = allocate_directory(baseline_dir, "rendered")
        for mode in ("rgb", "depth"):
            baseline_records.append(
                _run_stage(
                    f"baseline-render-{mode}",
                    _render_command(
                        mode,
                        isaac_python,
                        repo_root,
                        assets,
                        baseline_trajectory,
                        baseline_rendered,
                    ),
                    repo_root,
                    baseline_logs,
                    environment=environment,
                )
            )
        baseline_dataset = allocate_directory(baseline_dir, "dataset")
        baseline_records.append(
            _run_stage(
                "baseline-package",
                _package_command(
                    package_python,
                    repo_root,
                    baseline_trajectory,
                    baseline_rendered,
                    baseline_dataset,
                ),
                repo_root,
                baseline_logs,
                environment=environment,
            )
        )
        baseline_provenance = baseline_dir / "run_provenance.json"
        _write_provenance(
            baseline_provenance,
            repo_root,
            commit,
            input_hashes,
            runtime,
            cache_policy,
            baseline_records,
            _artifact_digests(
                baseline_trajectory, baseline_rendered, baseline_dataset
            ),
        )

        baseline_validation_results = []
        validation_specs = [
            (
                "baseline-validate-generate",
                [
                    str(package_python),
                    str(repo_root / "scripts" / "check_generate.py"),
                    "validate",
                    "--trajectory-dir",
                    str(baseline_trajectory),
                ],
            ),
            (
                "baseline-validate-render",
                [
                    str(package_python),
                    str(repo_root / "scripts" / "check_render.py"),
                    "validate",
                    "--rendered-dir",
                    str(baseline_rendered),
                    "--trajectory-dir",
                    str(baseline_trajectory),
                ],
            ),
            (
                "baseline-validate-package",
                [
                    str(package_python),
                    str(repo_root / "scripts" / "check_package.py"),
                    "validate",
                    "--dataset-dir",
                    str(baseline_dataset),
                    "--trajectory-dir",
                    str(baseline_trajectory),
                    "--rendered-dir",
                    str(baseline_rendered),
                ],
            ),
        ]
        for validation_name, command in validation_specs:
            result_path = evidence_dir / f"{validation_name}.json"
            record, _ = _run_checker(
                validation_name,
                [*command, "--result-path", str(result_path)],
                repo_root,
                logs_dir,
                environment,
                result_path,
            )
            baseline_validation_results.append(
                _checker_result(record, result_path)
            )

        # Fresh deterministic generation comparison.
        deterministic_dir = allocate_directory(runs_dir, "deterministic-generation")
        deterministic_logs = allocate_directory(deterministic_dir, "logs")
        deterministic_trajectory = allocate_directory(
            deterministic_dir, "trajectories"
        )
        deterministic_records = [
            _run_stage(
                "deterministic-generate",
                _generation_command(
                    isaac_python, repo_root, assets, deterministic_trajectory
                ),
                repo_root,
                deterministic_logs,
                environment=environment,
            )
        ]

        # Baseline absolute leakage and report-only all-frame distributions.
        baseline_measure_path = evidence_dir / "baseline_measurement.json"
        _, baseline_measure = _run_checker(
            "baseline-measure",
            [
                str(package_python),
                str(repo_root / "scripts" / "check_render.py"),
                "measure-golden",
                "--rendered-dir",
                str(baseline_rendered),
                "--trajectory-dir",
                str(baseline_trajectory),
                "--baseline-dir",
                str(baseline_rendered),
                "--baseline-provenance",
                str(baseline_provenance),
                "--run-provenance",
                str(baseline_provenance),
                "--tolerance-policy",
                str(policy_path),
                "--all-frames",
                "--result-path",
                str(baseline_measure_path),
            ],
            repo_root,
            logs_dir,
            environment,
            baseline_measure_path,
        )

        characterization = []
        characterization_measurements = []
        for index in range(1, CHARACTERIZATION_RUNS + 1):
            run = _render_pair(
                f"characterization-{index:02d}",
                runs_dir,
                baseline_trajectory,
                repo_root,
                assets,
                isaac_python,
                environment,
                package_python=package_python,
                package=False,
            )
            provenance_path = run["run_dir"] / "run_provenance.json"
            _write_provenance(
                provenance_path,
                repo_root,
                commit,
                input_hashes,
                runtime,
                cache_policy,
                run["stage_runs"],
                _artifact_digests(
                    baseline_trajectory, run["rendered_dir"]
                ),
            )
            result_path = evidence_dir / f"characterization-{index:02d}.json"
            measure_record, measurement = _run_checker(
                f"characterization-{index:02d}-measure",
                [
                    str(package_python),
                    str(repo_root / "scripts" / "check_render.py"),
                    "measure-golden",
                    "--rendered-dir",
                    str(run["rendered_dir"]),
                    "--trajectory-dir",
                    str(baseline_trajectory),
                    "--baseline-dir",
                    str(baseline_rendered),
                    "--baseline-provenance",
                    str(baseline_provenance),
                    "--run-provenance",
                    str(provenance_path),
                    "--tolerance-policy",
                    str(policy_path),
                    "--all-frames",
                    "--result-path",
                    str(result_path),
                ],
                repo_root,
                run["logs_dir"],
                environment,
                result_path,
            )
            characterization_measurements.append(measurement)
            characterization.append(
                {
                    "name": run["name"],
                    "run_dir": str(run["run_dir"]),
                    "run_provenance_sha256": _sha256_file(provenance_path),
                    "measurement_sha256": _sha256_file(result_path),
                    "stage_runs": run["stage_runs"],
                    "measurement_stage": measure_record,
                }
            )

        threshold_report = derive_threshold_report(
            policy, baseline_measure, characterization_measurements
        )
        threshold_report["source_policy_sha256"] = _sha256_file(policy_path)
        threshold_report["baseline_measurement_sha256"] = _sha256_file(
            baseline_measure_path
        )
        threshold_report["characterization_measurement_sha256"] = [
            item["measurement_sha256"] for item in characterization
        ]
        threshold_report_path = evidence_dir / "threshold_report.json"
        _atomic_write_json(threshold_report_path, threshold_report)

        # Held-outs never contribute to thresholds.
        held_outs = []
        held_out_checker_results = []
        for index in range(1, HELD_OUT_RUNS + 1):
            run = _render_pair(
                f"held-out-{index:02d}",
                runs_dir,
                baseline_trajectory,
                repo_root,
                assets,
                isaac_python,
                environment,
                package_python=package_python,
            )
            provenance_path = run["run_dir"] / "run_provenance.json"
            combined_records = deterministic_records + run["stage_runs"]
            _write_provenance(
                provenance_path,
                repo_root,
                commit,
                input_hashes,
                runtime,
                cache_policy,
                combined_records,
                _artifact_digests(
                    deterministic_trajectory,
                    run["rendered_dir"],
                    run["dataset_dir"],
                ),
            )

            render_result_path = evidence_dir / f"held-out-{index:02d}-render.json"
            render_record, render_result = _run_checker(
                f"held-out-{index:02d}-check-render",
                [
                    str(package_python),
                    str(repo_root / "scripts" / "check_render.py"),
                    "compare-golden",
                    "--rendered-dir",
                    str(run["rendered_dir"]),
                    "--trajectory-dir",
                    str(baseline_trajectory),
                    "--baseline-dir",
                    str(baseline_rendered),
                    "--baseline-provenance",
                    str(baseline_provenance),
                    "--run-provenance",
                    str(provenance_path),
                    "--tolerance-policy",
                    str(policy_path),
                    "--threshold-report",
                    str(threshold_report_path),
                    "--all-frames",
                    "--result-path",
                    str(render_result_path),
                ],
                repo_root,
                run["logs_dir"],
                environment,
                render_result_path,
            )
            package_result_path = evidence_dir / f"held-out-{index:02d}-package.json"
            package_record, package_result = _run_checker(
                f"held-out-{index:02d}-check-package",
                [
                    str(package_python),
                    str(repo_root / "scripts" / "check_package.py"),
                    "compare-golden",
                    "--dataset-dir",
                    str(run["dataset_dir"]),
                    "--trajectory-dir",
                    str(baseline_trajectory),
                    "--rendered-dir",
                    str(run["rendered_dir"]),
                    "--baseline-dir",
                    str(baseline_dataset),
                    "--baseline-trajectory-dir",
                    str(baseline_trajectory),
                    "--baseline-rendered-dir",
                    str(baseline_rendered),
                    "--baseline-provenance",
                    str(baseline_provenance),
                    "--run-provenance",
                    str(provenance_path),
                    "--result-path",
                    str(package_result_path),
                ],
                repo_root,
                run["logs_dir"],
                environment,
                package_result_path,
            )
            held_out_checker_results.extend(
                [
                    _checker_result(render_record, render_result_path),
                    _checker_result(package_record, package_result_path),
                ]
            )
            held_outs.append(
                {
                    "name": run["name"],
                    "run_dir": str(run["run_dir"]),
                    "run_provenance": str(provenance_path),
                    "run_provenance_sha256": _sha256_file(provenance_path),
                    "render_result_sha256": _sha256_file(render_result_path),
                    "package_result_sha256": _sha256_file(package_result_path),
                    "render_eligible": render_result["eligible"],
                    "package_eligible": package_result["eligible"],
                    "stage_runs": combined_records,
                }
            )

        final_provenance = Path(held_outs[-1]["run_provenance"])
        generate_result_path = evidence_dir / "deterministic-generate.json"
        generate_record, generate_result = _run_checker(
            "check-deterministic-generate",
            [
                str(package_python),
                str(repo_root / "scripts" / "check_generate.py"),
                "compare-golden",
                "--trajectory-dir",
                str(deterministic_trajectory),
                "--baseline-dir",
                str(baseline_trajectory),
                "--baseline-provenance",
                str(baseline_provenance),
                "--run-provenance",
                str(final_provenance),
                "--result-path",
                str(generate_result_path),
            ],
            repo_root,
            deterministic_logs,
            environment,
            generate_result_path,
        )

        mutation_report_path, mutation_report = _run_mutation_suite(
            attempt_dir,
            baseline_rendered,
            baseline_trajectory,
            baseline_provenance,
            final_provenance,
            policy_path,
            threshold_report_path,
            package_python,
            repo_root,
            environment,
        )
        mutation_checker_result = build_checker_result(
            checker="check_render_mutations",
            mode="mutation-suite",
            result_sha256=_sha256_file(mutation_report_path),
            exit_code=0,
            eligible=mutation_report["all_expected_outcomes_observed"],
        )

        selected_frames_dir = allocate_directory(evidence_dir, "selected_frames")
        _copy_selected_frames(
            baseline_rendered, baseline_trajectory, selected_frames_dir
        )

        manifest_results = [
            *baseline_validation_results,
            _checker_result(generate_record, generate_result_path),
            *held_out_checker_results,
            mutation_checker_result,
        ]
        verification_manifest = build_verification_manifest(
            baseline_id=BASELINE_ID,
            candidate_commit=commit,
            run_provenance_sha256=_sha256_file(final_provenance),
            baseline_provenance_sha256=_sha256_file(baseline_provenance),
            tolerance_policy_sha256=_sha256_file(policy_path),
            results=manifest_results,
        )
        verification_manifest_path = evidence_dir / "verification_manifest.json"
        write_verification_manifest(
            verification_manifest_path, verification_manifest
        )
        if not verification_manifest["overall_eligible"]:
            raise RuntimeError("final verification manifest is ineligible")

        baseline_manifest = json.loads(
            (
                baseline_trajectory / "trajectory_manifest.json"
            ).read_text(encoding="utf-8")
        )
        completed_at = _utc_now()
        duration_seconds = (
            datetime.fromisoformat(completed_at)
            - datetime.fromisoformat(capture_started_at)
        ).total_seconds()
        artifact_size_bytes = sum(
            path.stat().st_size
            for path in attempt_dir.rglob("*")
            if path.is_file()
        )
        generation_attempts = baseline_manifest["generation"]["attempts"]
        rejection_count = sum(
            baseline_manifest["generation"]["rejection_counts"].values()
        )
        capture_report = {
            "schema_version": 1,
            "baseline_id": BASELINE_ID,
            "scene": SCENE,
            "candidate_commit": commit,
            "plan_revision": PLAN_REVISION,
            "plan_commit": PLAN_COMMIT,
            "started_at": capture_started_at,
            "completed_at": completed_at,
            "capture_root": str(attempt_dir),
            "cache_policy": cache_policy,
            "input_hashes": input_hashes,
            "runtime_fingerprint": runtime,
            "baseline": {
                "run_dir": str(baseline_dir),
                "provenance_sha256": _sha256_file(baseline_provenance),
                "generation_attempts": generation_attempts,
                "generation_rejection_count": rejection_count,
                "generation_rejection_rate": (
                    rejection_count / generation_attempts
                ),
                "stage_runs": baseline_records,
            },
            "deterministic_generation": {
                "run_dir": str(deterministic_dir),
                "eligible": generate_result["eligible"],
                "result_sha256": _sha256_file(generate_result_path),
                "stage_runs": deterministic_records,
            },
            "characterization": characterization,
            "held_outs": held_outs,
            "threshold_report_sha256": _sha256_file(threshold_report_path),
            "mutation_report_sha256": _sha256_file(mutation_report_path),
            "verification_manifest_sha256": _sha256_file(
                verification_manifest_path
            ),
            "measured_capture": {
                "duration_seconds": duration_seconds,
                "artifact_size_bytes_before_capture_report": artifact_size_bytes,
                "failed_run_count": 0,
            },
            "delivery_reestimate": {
                "status": "pending-owner-acceptance",
                "phase_1_estimate_days": 1.5,
                "risk_after_machine_gate": "low-to-medium",
                "basis": (
                    "Machine gate passed with the measured runtime, artifact "
                    "size, zero failed runs, and recorded generation retry rate."
                ),
            },
            "diagnostic_retention": (
                "The complete attempt, including logs and mutation artifacts, "
                "is retained until the engineering owner explicitly removes it."
            ),
            "machine_gate": "passed",
            "formal_gate": "pending-independent-approval",
            "approval_required": (
                "Engineering owner approval of thresholds and revised "
                "schedule/risk estimate."
            ),
        }
        capture_report_path = evidence_dir / "capture_report.json"
        _atomic_write_json(capture_report_path, capture_report)
        _write_status(
            status_path,
            {
                "status": "complete",
                "completed_at": _utc_now(),
                "machine_gate": "passed",
                "formal_gate": "pending-independent-approval",
                "capture_report": str(capture_report_path),
            },
        )
        print(f"[phase0b] machine gate passed: {capture_report_path}")
        print("[phase0b] formal gate awaits independent approval")
        return 0
    except Exception as error:
        _write_status(
            status_path,
            {
                "status": "failed",
                "completed_at": _utc_now(),
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())

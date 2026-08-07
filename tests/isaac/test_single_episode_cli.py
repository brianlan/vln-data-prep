"""Tests for the post-6.4 candidate CLI adapter."""

import argparse
import json

import numpy as np
import pytest
from PIL import Image

import optimize_sage3d_trajectories
from optimize_sage3d_trajectories import _evaluate_spline
from sage3d.utils import MapTransform

LIMITS = {
    "v_max": 0.5, "a_max": 1.0, "j_max": 2.0,
    "yaw_rate_max": 1.0, "yaw_accel_max": 2.0, "yaw_jerk_max": 3.0,
}
OBJECTIVE = {
    "w_ref": 1.0, "w_jerk_xy": 1.0, "w_jerk_yaw": 1.0,
    "w_yaw_rate": 1.0, "w_time": 1.0,
    "reference_distance_scale_m": 0.5, "jerk_xy_scale": 1.0,
    "jerk_yaw_scale": 1.0, "yaw_rate_scale": 1.0, "time_scale_s": 10.0,
}
TRUST = {"trust_xy_resolution_cells": 2.0, "trust_xy_max_m": 1.0, "trust_yaw_rad": 0.5}
SOLVER = {
    "ftol": 1e-9, "episode_timeout_s": 60.0, "constraint_tolerance": 1e-4,
    "clearance_scale_m": 0.1, "maxiter": 200, "final_objective_tolerance": 1e-6,
}
INITIALIZATION = {
    "target_control_spacing_m": 0.5,
    "min_control_points": 8,
    "max_control_points": 64,
    "lambda_init": 1.0,
    "gamma": 1.2,
}
NPZ_FIELDS = {
    "time_s", "pose_world", "yaw_unwrapped_rad", "velocity_world_mps",
    "yaw_rate_radps", "acceleration_world_mps2", "yaw_acceleration_radps2",
    "jerk_world_mps3", "yaw_jerk_radps3",
}


def _write_scene(
    scene_root,
    drop_key=None,
    episode_count=1,
):
    height = width = 16
    scale = 0.05
    lower_x = lower_y = -0.4
    transform = MapTransform(
        height=height, width=width, scale=scale, lower_x=lower_x, lower_y=lower_y
    )
    scene_dir = scene_root / "scene001"
    map_dir = scene_dir / "map"
    traj_dir = scene_dir / "trajectories"
    map_dir.mkdir(parents=True)
    traj_dir.mkdir(parents=True)
    np.save(map_dir / "esdf.npy", np.full((height, width), 0.2, dtype=np.float64))
    Image.fromarray(np.full((height, width), 50, dtype=np.uint8)).save(
        map_dir / "esdf.png"
    )
    # Non-symmetric row/col pairs so a row/col swap in the adapter would fail.
    pixels = np.array(
        [[3, 7], [4, 6], [5, 5], [6, 4], [7, 3], [8, 2]], dtype=np.int32
    )
    points = np.asarray(
        [transform.pixel_to_world(int(r), int(c)) for r, c in pixels],
        dtype=np.float64,
    )
    yaw = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    arrays = {
        "points": points.astype(np.float32),
        "yaw": yaw,
        "astar_path_pixels": pixels,
    }
    if drop_key:
        del arrays[drop_key]
    for episode_index in range(episode_count):
        np.savez_compressed(
            traj_dir / f"episode_{episode_index:06d}.npz", **arrays
        )
    manifest = {
        "scene_id": "scene001",
        "episode_count": episode_count,
        "map": {
            "shape": [height, width],
            "scale_m_per_pixel": scale,
            "lower_x": lower_x,
            "lower_y": lower_y,
            "required_path_clearance_m": 0.1,
        },
        "episodes": [
            {"episode_index": episode_index}
            for episode_index in range(episode_count)
        ],
    }
    (traj_dir / "trajectory_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return transform, pixels, points, yaw


def _write_config(tmp_path):
    config = {
        "limits": LIMITS,
        "objective": OBJECTIVE,
        "trust": TRUST,
        "solver": SOLVER,
        "yaw_tangent_weight": 0.5,
        "initialization": INITIALIZATION,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _fake_result(success=True, status="success"):
    control_points = np.zeros((8, 3))
    control_points[:, 0] = np.linspace(-0.35, 0.35, 8)
    candidate = _evaluate_spline(control_points, 2.0)
    initial_position_world = candidate["position_world"].copy()
    initial_position_world[:, 1] += 0.15
    return {
        "success": success,
        "status": status,
        "objective_initial": {"total": 2.0},
        "objective_continuous": {"total": 1.5},
        "objective": {"total": 1.0},
        "constraint_diagnostics": {
            "initial_feasible": True, "finiteness_ok": True, "margins_ok": True,
            "monotonic_ok": True, "final_margins_min": 0.1,
            "final_margin_groups_min": {"velocity_xy": 0.1},
            "t_output_within_policy": True,
        },
        "solver_metadata": {
            "solver": "SLSQP", "result_success": True, "nit": 3,
            "nfev": 12, "elapsed_s": 0.01,
        },
        "initial_position_world": initial_position_world,
        "control_points": control_points,
        "T_continuous": 1.9,
        "T_output": 2.0,
        "candidate": candidate,
    }


def _stub_optimize(captured, result):
    def fake(astar_path_xy, start_pose, goal_pose, limits, **kwargs):
        captured.update(
            astar_path_xy=astar_path_xy, start_pose=start_pose,
            goal_pose=goal_pose, limits=limits, **kwargs,
        )
        return result

    return fake


def _args(
    scene_root, config_path, output_dir, visualize=False, episode_index=0
):
    output_dir.mkdir(parents=True, exist_ok=True)
    return argparse.Namespace(
        scene_root=scene_root, scene_id="scene001", episode_index=episode_index,
        config=config_path, output_dir=output_dir,
        visualize_optimized_trajectories=visualize,
    )


def test_success_path(tmp_path, monkeypatch):
    scene_root = tmp_path / "scenes"
    transform, pixels, points, yaw = _write_scene(scene_root)
    (scene_root / "scene001" / "map" / "esdf.png").unlink()
    config_path = _write_config(tmp_path)
    output_dir = tmp_path / "out"
    captured = {}
    monkeypatch.setattr(
        optimize_sage3d_trajectories, "optimize_trajectory",
        _stub_optimize(captured, _fake_result()),
    )
    optimize_sage3d_trajectories.main(_args(scene_root, config_path, output_dir))

    expected_xy = np.asarray(
        [transform.pixel_to_world(int(r), int(c)) for r, c in pixels], dtype=float
    )
    assert np.allclose(captured["astar_path_xy"], expected_xy)
    assert captured["start_pose"] == pytest.approx(
        (float(points[0, 0]), float(points[0, 1]), float(yaw[0]))
    )
    assert captured["goal_pose"] == pytest.approx(
        (float(points[-1, 0]), float(points[-1, 1]), float(yaw[-1]))
    )
    assert captured["limits"] == LIMITS
    assert all(captured[key] == value for key, value in INITIALIZATION.items())

    npz_path = output_dir / "episode_000000.npz"
    meta_path = output_dir / "candidate_metadata.json"
    assert sorted(p.name for p in output_dir.iterdir()) == [
        "candidate_metadata.json", "episode_000000.npz",
    ]
    with np.load(npz_path) as data:
        assert set(data.files) == NPZ_FIELDS
        for name in data.files:
            assert data[name].dtype == np.float64
        assert data["time_s"].shape == (21,)
        assert data["pose_world"].shape == (21, 3)
        assert data["velocity_world_mps"].shape == (21, 2)
        assert data["yaw_rate_radps"].shape == (21,)
        assert data["time_s"][0] == 0.0 and data["time_s"][-1] == 2.0
        # pose_world's yaw column is the wrapped version of yaw_unwrapped.
        assert np.allclose(
            data["pose_world"][:, 2],
            (data["yaw_unwrapped_rad"] + np.pi) % (2 * np.pi) - np.pi,
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["scene_id"] == "scene001"
    assert meta["effective_config"]["initialization"] == INITIALIZATION
    assert meta["summary"] == {"requested": 1, "succeeded": 1, "failed": 0}
    assert len(meta["episodes"]) == 1
    episode = meta["episodes"][0]
    assert episode["inputs"]["episode_npz"].endswith("episode_000000.npz")
    assert episode["success"] is True and episode["status"] == "success"
    assert episode["T_continuous"] == 1.9 and episode["T_output"] == 2.0
    assert episode["objectives"]["initial"]["total"] == 2.0
    assert episode["objectives"]["continuous"]["total"] == 1.5
    assert episode["objectives"]["output"]["total"] == 1.0
    assert episode["npz_filename"] == "episode_000000.npz"
    assert episode["validated"] is False and episode["executable"] is False


def test_visualization_overlay(tmp_path, monkeypatch):
    scene_root = tmp_path / "scenes"
    _write_scene(scene_root)
    config_path = _write_config(tmp_path)
    output_dir = tmp_path / "out"
    monkeypatch.setattr(
        optimize_sage3d_trajectories, "optimize_trajectory",
        _stub_optimize({}, _fake_result()),
    )

    optimize_sage3d_trajectories.main(
        _args(scene_root, config_path, output_dir, visualize=True)
    )

    vis_dir = output_dir / "vis"
    assert [path.name for path in vis_dir.iterdir()] == [
        "episode_000000_overlay.png"
    ]
    image = np.asarray(Image.open(vis_dir / "episode_000000_overlay.png"))
    assert np.any(np.all(image == (180, 80, 255), axis=-1))
    assert np.any(np.all(image == (255, 0, 0), axis=-1))
    assert np.any(np.all(image == (255, 255, 0), axis=-1))


@pytest.mark.parametrize("key", ["points", "yaw", "astar_path_pixels"])
def test_missing_episode_key_rejected(tmp_path, key):
    scene_root = tmp_path / "scenes"
    _write_scene(scene_root, drop_key=key)
    config_path = _write_config(tmp_path)
    output_dir = tmp_path / "out"
    with pytest.raises(KeyError, match=key):
        optimize_sage3d_trajectories.main(_args(scene_root, config_path, output_dir))


def test_existing_output_is_overwritten_without_cleaning_directory(
    tmp_path, monkeypatch
):
    scene_root = tmp_path / "scenes"
    _write_scene(scene_root)
    config_path = _write_config(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    np.savez_compressed(output_dir / "episode_000000.npz", stale=np.array([1]))
    (output_dir / "candidate_metadata.json").write_text(
        '{"old": true}', encoding="utf-8"
    )
    (output_dir / "unrelated.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        optimize_sage3d_trajectories,
        "optimize_trajectory",
        _stub_optimize({}, _fake_result()),
    )

    optimize_sage3d_trajectories.main(
        _args(scene_root, config_path, output_dir)
    )

    with np.load(output_dir / "episode_000000.npz") as data:
        assert set(data.files) == NPZ_FIELDS
    metadata = json.loads(
        (output_dir / "candidate_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["summary"]["succeeded"] == 1
    assert (output_dir / "unrelated.txt").read_text(encoding="utf-8") == "keep"


def test_input_trajectory_directory_cannot_be_used_as_output(tmp_path):
    scene_root = tmp_path / "scenes"
    _write_scene(scene_root)
    config_path = _write_config(tmp_path)
    input_dir = scene_root / "scene001" / "trajectories"

    with pytest.raises(SystemExit, match="must differ"):
        optimize_sage3d_trajectories.main(
            _args(scene_root, config_path, input_dir)
        )


def test_optimizer_failure_is_recorded_without_candidate(tmp_path, monkeypatch):
    scene_root = tmp_path / "scenes"
    _write_scene(scene_root)
    config_path = _write_config(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    np.savez_compressed(output_dir / "episode_000000.npz", stale=np.array([1]))
    result = _fake_result(success=False, status="solver_failed")
    result["constraint_diagnostics"]["final_margins_min"] = -np.inf
    monkeypatch.setattr(
        optimize_sage3d_trajectories, "optimize_trajectory",
        _stub_optimize({}, result),
    )
    with pytest.raises(SystemExit, match="1/1"):
        optimize_sage3d_trajectories.main(_args(scene_root, config_path, output_dir))
    with np.load(output_dir / "episode_000000.npz") as data:
        assert set(data.files) == {"stale"}
    metadata = json.loads(
        (output_dir / "candidate_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["summary"] == {"requested": 1, "succeeded": 0, "failed": 1}
    assert metadata["episodes"][0]["status"] == "solver_failed"
    assert metadata["episodes"][0]["npz_filename"] is None
    assert metadata["episodes"][0]["constraint_diagnostics"][
        "final_margins_min"
    ] is None


def test_omitted_episode_index_processes_manifest_and_continues_failures(
    tmp_path, monkeypatch
):
    scene_root = tmp_path / "scenes"
    _write_scene(scene_root, episode_count=3)
    config_path = _write_config(tmp_path)
    output_dir = tmp_path / "out"
    results = iter(
        [
            _fake_result(),
            _fake_result(success=False, status="solver_failed"),
            _fake_result(),
        ]
    )
    calls = []

    def fake_optimize(*args, **kwargs):
        calls.append(args[0])
        return next(results)

    monkeypatch.setattr(
        optimize_sage3d_trajectories, "optimize_trajectory", fake_optimize
    )

    with pytest.raises(SystemExit, match="1/3"):
        optimize_sage3d_trajectories.main(
            _args(scene_root, config_path, output_dir, episode_index=None)
        )

    assert len(calls) == 3
    assert (output_dir / "episode_000000.npz").is_file()
    assert not (output_dir / "episode_000001.npz").exists()
    assert (output_dir / "episode_000002.npz").is_file()
    metadata = json.loads(
        (output_dir / "candidate_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["summary"] == {"requested": 3, "succeeded": 2, "failed": 1}
    assert [episode["episode_index"] for episode in metadata["episodes"]] == [
        0, 1, 2,
    ]
    assert [episode["success"] for episode in metadata["episodes"]] == [
        True, False, True,
    ]

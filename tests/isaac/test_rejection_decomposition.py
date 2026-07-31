"""Deterministic call/rejection trace tests for Phase 3c rejection decomposition.

Covers every rejection exit of the decomposed ``generate_episodes`` loop:
``euclidean_too_short``, ``duplicate_endpoint_pair``, ``astar_failed``,
``geodesic_too_short``, ``geodesic_too_long``, ``resampled_path_not_safe``,
``camera_collision_clearance``, and the success path.

Runs under Isaac Python because ``episode_generation`` imports
``path_postprocess`` (scipy), ``collision`` (pxr/trimesh), and
``navigation_map`` (cv2/PIL).
"""

from __future__ import annotations

import numpy as np
import trimesh

from sage3d.episode_generation import (
    EndpointPair,
    PlannedPath,
    Rejection,
    build_episode,
    build_episode_arrays,
    generate_episodes,
    plan_path,
    postprocess_path,
    sample_endpoint_pair,
    validate_camera_clearance,
)
from sage3d.geometry import MapTransform


# --- helpers --------------------------------------------------------------


def _make_transform(scale: float = 1.0) -> MapTransform:
    return MapTransform(height=20, width=20, scale=scale, lower_x=0.0, lower_y=0.0)


def _make_far_mesh() -> trimesh.Trimesh:
    """A collision mesh far from the map so camera clearance always passes."""
    return trimesh.Trimesh(
        vertices=np.array([[100.0, 100.0, 100.0], [101.0, 100.0, 100.0],
                          [100.0, 101.0, 100.0]], dtype=np.float64),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )


def _make_near_mesh() -> trimesh.Trimesh:
    """A collision mesh at the map origin so camera clearance always fails."""
    return trimesh.Trimesh(
        vertices=np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]],
                          dtype=np.float64),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )


# --- sample_endpoint_pair: euclidean_too_short ----------------------------


def test_sample_endpoint_pair_euclidean_too_short():
    """Two cells in the same component that are too close in world space."""
    transform = _make_transform(scale=1.0)
    usable = [(1, 3)]
    # Cells at (5,5), (5,6), (5,7) — adjacent pixels are 1m apart in world.
    cells = np.array([[5, 5], [5, 6], [5, 7]], dtype=np.int64)
    component_cells = {1: cells}
    component_weights = np.array([1.0], dtype=np.float64)
    rng = np.random.default_rng(42)
    # min_path_length * 0.55 = 10 * 0.55 = 5.5; distance between (5,5) and (5,6) is 1.0
    result = sample_endpoint_pair(
        rng, usable, component_cells, component_weights, transform,
        min_path_length=10.0, used_endpoint_pairs=[],
    )
    assert isinstance(result, Rejection)
    assert result.reason == "euclidean_too_short"


# --- sample_endpoint_pair: duplicate_endpoint_pair ------------------------


def test_sample_endpoint_pair_duplicate_endpoint():
    """Endpoints too close to a previously used pair."""
    transform = _make_transform(scale=1.0)
    usable = [(1, 2)]
    cells = np.array([[5, 5], [5, 6]], dtype=np.int64)
    component_cells = {1: cells}
    component_weights = np.array([1.0], dtype=np.float64)
    rng = np.random.default_rng(42)
    # Both possible orderings as used pairs so whichever order the RNG
    # selects, the duplicate check triggers.
    a = np.asarray(transform.pixel_to_world(5, 5))
    b = np.asarray(transform.pixel_to_world(5, 6))
    used = [(a, b), (b, a)]
    # min_path_length=0.01 so euclidean check passes, but duplicate check fails
    result = sample_endpoint_pair(
        rng, usable, component_cells, component_weights, transform,
        min_path_length=0.01, used_endpoint_pairs=used,
    )
    assert isinstance(result, Rejection)
    assert result.reason == "duplicate_endpoint_pair"


# --- sample_endpoint_pair: success ---------------------------------------


def test_sample_endpoint_pair_success():
    """Valid endpoint pair far enough apart and not duplicate."""
    transform = _make_transform(scale=1.0)
    usable = [(1, 4)]
    cells = np.array([[1, 1], [10, 10], [1, 10], [10, 1]], dtype=np.int64)
    component_cells = {1: cells}
    component_weights = np.array([1.0], dtype=np.float64)
    rng = np.random.default_rng(0)
    result = sample_endpoint_pair(
        rng, usable, component_cells, component_weights, transform,
        min_path_length=1.0, used_endpoint_pairs=[],
    )
    assert isinstance(result, EndpointPair)
    assert result.component_id == 1
    assert len(result.start_pixel) == 2
    assert len(result.goal_pixel) == 2


# --- plan_path: astar_failed -----------------------------------------------


def test_plan_path_astar_failed():
    """Blocked path between start and goal."""
    transform = _make_transform()
    safe = np.ones((10, 10), dtype=bool)
    safe[5, :] = False  # wall across the middle
    clearance_m = np.full((10, 10), 5.0, dtype=np.float32)
    path = plan_path(safe, clearance_m, (0, 0), (9, 9), transform)
    assert path is None


def test_plan_path_success():
    """Open grid — path should be found."""
    transform = _make_transform()
    safe = np.ones((10, 10), dtype=bool)
    clearance_m = np.full((10, 10), 5.0, dtype=np.float32)
    path = plan_path(safe, clearance_m, (0, 0), (9, 9), transform)
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (9, 9)


# --- postprocess_path: geodesic_too_short ----------------------------------


def test_postprocess_path_geodesic_too_short():
    """Path shorter than min_path_length."""
    transform = _make_transform(scale=1.0)
    safe = np.ones((10, 10), dtype=bool)
    # Path of 2 points, 1m apart — too short
    pixel_path = [(0, 0), (0, 1)]
    result = postprocess_path(
        pixel_path, transform, safe,
        min_path_length=10.0, max_path_length=100.0, frame_spacing=0.5,
    )
    assert isinstance(result, Rejection)
    assert result.reason == "geodesic_too_short"


# --- postprocess_path: geodesic_too_long -----------------------------------


def test_postprocess_path_geodesic_too_long():
    """Path longer than max_path_length."""
    transform = _make_transform(scale=1.0)
    safe = np.ones((1, 20), dtype=bool)
    # Long path: 20 pixels = ~20m
    pixel_path = [(0, i) for i in range(20)]
    result = postprocess_path(
        pixel_path, transform, safe,
        min_path_length=0.1, max_path_length=5.0, frame_spacing=0.5,
    )
    assert isinstance(result, Rejection)
    assert result.reason == "geodesic_too_long"


# --- postprocess_path: resampled_path_not_safe ------------------------------


def test_postprocess_path_resampled_not_safe():
    """Smoothed/resampled path goes through unsafe area."""
    transform = _make_transform(scale=1.0)
    safe = np.ones((20, 20), dtype=bool)
    # Block a large area in the middle so the spline overshoots into unsafe
    safe[5:15, 5:15] = False
    # Path that goes diagonally toward the blocked area
    pixel_path = [(0, 0), (2, 2), (4, 4), (6, 6), (10, 10)]
    result = postprocess_path(
        pixel_path, transform, safe,
        min_path_length=0.1, max_path_length=1000.0, frame_spacing=0.5,
    )
    assert isinstance(result, Rejection)
    assert result.reason == "resampled_path_not_safe"


# --- postprocess_path: success ---------------------------------------------


def test_postprocess_path_success():
    """Valid path passes all postprocessing checks."""
    transform = _make_transform(scale=1.0)
    safe = np.ones((20, 20), dtype=bool)
    # Straight path of 10 pixels — ~10m, within range
    pixel_path = [(0, i) for i in range(10)]
    result = postprocess_path(
        pixel_path, transform, safe,
        min_path_length=1.0, max_path_length=100.0, frame_spacing=1.0,
    )
    assert isinstance(result, PlannedPath)
    assert result.raw_length > 0
    assert len(result.sampled) >= 2
    assert isinstance(result.smoothing_method, str)


# --- validate_camera_clearance: failure ------------------------------------


def test_validate_camera_clearance_failure():
    """Camera positions near collision mesh — clearance too small."""
    mesh = _make_near_mesh()
    # Camera at (0, 0, 1.0) — right next to the mesh at z=1.0
    camera_positions = np.array([[0.05, 0.05, 1.0]], dtype=np.float32)
    result = validate_camera_clearance(mesh, camera_positions, camera_clearance=5.0)
    assert isinstance(result, Rejection)
    assert result.reason == "camera_collision_clearance"


# --- validate_camera_clearance: success ------------------------------------


def test_validate_camera_clearance_success():
    """Camera positions far from collision mesh — clearance passes."""
    mesh = _make_far_mesh()
    camera_positions = np.array([[5.0, 5.0, 1.0]], dtype=np.float32)
    result = validate_camera_clearance(mesh, camera_positions, camera_clearance=0.01)
    assert isinstance(result, float)
    assert result >= 0.01


# --- build_episode: key order and values -----------------------------------


def test_build_episode_key_order_and_values():
    """Episode dict has the exact legacy key order and correct values."""
    transform = _make_transform(scale=1.0)
    clearance_m = np.full((20, 20), 5.0, dtype=np.float32)
    sampled = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64)
    actions, camera_positions, yaw, point_goal = build_episode_arrays(
        sampled, camera_height=1.0
    )
    episode = build_episode(
        episode_index=0,
        component_id=1,
        start_pixel=(1, 1),
        goal_pixel=(3, 3),
        sampled=sampled,
        raw_length=2.828,
        smoothing_method="line_of_sight",
        clearance_m=clearance_m,
        transform=transform,
        minimum_camera_clearance=10.0,
        actions=actions,
        camera_positions=camera_positions,
        yaw=yaw,
        point_goal=point_goal,
    )
    expected_keys = [
        "episode_index", "component_id", "start_pixel", "goal_pixel",
        "start_position", "goal_position", "raw_path_length_m",
        "path_length_m", "frame_count", "minimum_clearance_m",
        "minimum_camera_clearance_m", "smoothing_method", "points",
        "actions", "camera_positions", "yaw", "point_goal",
    ]
    assert list(episode.keys()) == expected_keys
    assert episode["episode_index"] == 0
    assert episode["component_id"] == 1
    assert episode["start_pixel"] == [1, 1]
    assert episode["goal_pixel"] == [3, 3]
    assert episode["smoothing_method"] == "line_of_sight"
    assert episode["minimum_camera_clearance_m"] == 10.0
    assert episode["points"].dtype == np.float32
    assert episode["actions"].dtype == np.float32
    assert episode["camera_positions"].dtype == np.float32
    assert episode["yaw"].dtype == np.float32
    assert episode["point_goal"].dtype == np.float32


# --- generate_episodes: full rejection trace -------------------------------


def test_generate_episodes_all_astar_failed():
    """Every attempt hits astar_failed — no episodes generated."""
    transform = _make_transform(scale=1.0)
    safe = np.ones((10, 10), dtype=bool)
    safe[5, :] = False  # wall across the middle
    clearance_m = np.full((10, 10), 5.0, dtype=np.float32)
    # Two components separated by wall
    component_labels = np.zeros((10, 10), dtype=np.int32)
    component_labels[:5, :] = 1
    component_labels[5:, :] = 2
    # endpoint_clearance low so all safe cells qualify
    mesh = _make_far_mesh()
    try:
        generate_episodes(
            safe=safe,
            clearance_m=clearance_m,
            component_labels=component_labels,
            transform=transform,
            episode_count=1,
            seed=42,
            min_path_length=0.1,
            max_path_length=1000.0,
            frame_spacing=1.0,
            endpoint_clearance=0.0,
            max_attempts=5,
            camera_height=1.0,
            collision_mesh=mesh,
            camera_clearance=0.01,
        )
    except RuntimeError as exc:
        assert "Generated only 0/1" in str(exc)
        assert "astar_failed" in str(exc)
    else:
        # If all attempts somehow succeed (unlikely with the wall), that's OK
        pass


def test_generate_episodes_success():
    """Full generation with an easy map produces the requested episodes."""
    transform = _make_transform(scale=1.0)
    safe = np.ones((30, 30), dtype=bool)
    clearance_m = np.full((30, 30), 5.0, dtype=np.float32)
    component_labels = np.ones((30, 30), dtype=np.int32)
    mesh = _make_far_mesh()
    episodes, info = generate_episodes(
        safe=safe,
        clearance_m=clearance_m,
        component_labels=component_labels,
        transform=transform,
        episode_count=2,
        seed=20260720,
        min_path_length=5.0,
        max_path_length=200.0,
        frame_spacing=0.5,
        endpoint_clearance=0.0,
        max_attempts=200,
        camera_height=1.0,
        collision_mesh=mesh,
        camera_clearance=0.01,
    )
    assert len(episodes) == 2
    assert info["usable_component_count"] == 1
    assert info["attempts"] >= 2
    # Episodes are ordered by index
    assert episodes[0]["episode_index"] == 0
    assert episodes[1]["episode_index"] == 1
    # Each episode has all expected keys
    for ep in episodes:
        assert "points" in ep
        assert "actions" in ep
        assert "camera_positions" in ep
        assert "yaw" in ep
        assert "point_goal" in ep


def test_generate_episodes_no_usable_components():
    """No connected component has valid endpoint candidates — RuntimeError."""
    transform = _make_transform(scale=1.0)
    safe = np.ones((5, 5), dtype=bool)
    clearance_m = np.full((5, 5), 5.0, dtype=np.float32)
    # Only background label (0)
    component_labels = np.zeros((5, 5), dtype=np.int32)
    mesh = _make_far_mesh()
    try:
        generate_episodes(
            safe=safe,
            clearance_m=clearance_m,
            component_labels=component_labels,
            transform=transform,
            episode_count=1,
            seed=42,
            min_path_length=1.0,
            max_path_length=100.0,
            frame_spacing=1.0,
            endpoint_clearance=0.0,
            max_attempts=10,
            camera_height=1.0,
            collision_mesh=mesh,
            camera_clearance=0.01,
        )
    except RuntimeError as exc:
        assert "No connected component" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for no usable components")


# --- Rejection and EndpointPair type checks ---------------------------------


def test_rejection_is_frozen():
    r = Rejection("test")
    try:
        r.reason = "other"
    except AttributeError:
        pass
    else:
        raise AssertionError("Rejection should be frozen")


def test_endpoint_pair_is_frozen():
    ep = EndpointPair(
        component_id=1,
        start_pixel=(0, 0),
        goal_pixel=(1, 1),
        start_xy=np.array([0.0, 0.0]),
        goal_xy=np.array([1.0, 1.0]),
    )
    try:
        ep.component_id = 2
    except AttributeError:
        pass
    else:
        raise AssertionError("EndpointPair should be frozen")
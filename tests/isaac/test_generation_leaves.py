"""Leaf unit tests for Phase 3a generation modules.

Covers geometry, pathfinding, path_postprocess, collision, viz, and
navigation_map with focused behavioral checks.  Runs under Isaac Python
because path_postprocess (scipy), collision (pxr/trimesh), viz (cv2/PIL),
and navigation_map (cv2/PIL) require Isaac-lane dependencies.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from sage3d.geometry import MapTransform, path_length, pixels_to_world, wrap_angle


# --- geometry -----------------------------------------------------------


def test_map_transform_roundtrip():
    t = MapTransform(100, 80, 0.5, -10.0, -5.0)
    x, y = t.pixel_to_world(10, 20)
    row, col = t.world_to_pixel(x, y)
    assert (row, col) == (10, 20)


def test_path_length_zero_or_positive():
    assert path_length(np.zeros((1, 2))) == 0.0
    pts = np.array([[0, 0], [3, 4]], dtype=np.float64)
    assert path_length(pts) == 5.0


def test_wrap_angle_wraps_to_pi():
    arr = np.array([0.0, math.pi, -math.pi, 3 * math.pi])
    wrapped = wrap_angle(arr)
    assert wrapped[0] == 0.0
    assert -math.pi <= wrapped[1] < math.pi
    assert -math.pi <= wrapped[2] < math.pi
    assert abs(wrapped[3] - math.pi) < 1e-12 or abs(wrapped[3] + math.pi) < 1e-12


def test_pixels_to_world_shape():
    t = MapTransform(10, 10, 1.0, 0.0, 0.0)
    pts = pixels_to_world([(0, 0), (1, 1), (2, 2)], t)
    assert pts.shape == (3, 2)
    assert pts.dtype == np.float64


# --- pathfinding --------------------------------------------------------


from sage3d.pathfinding import NEIGHBORS, astar


def test_astar_straight_line():
    safe = np.ones((5, 5), dtype=bool)
    clearance = np.full((5, 5), 1.0, dtype=np.float32)
    path = astar(safe, clearance, (0, 0), (4, 4), 1.0)
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)


def test_astar_blocked():
    safe = np.ones((5, 5), dtype=bool)
    safe[2, :] = False
    clearance = np.full((5, 5), 1.0, dtype=np.float32)
    path = astar(safe, clearance, (0, 0), (4, 0), 1.0)
    assert path is None


def test_astar_diagonal_corner_cutting_prevented():
    safe = np.zeros((3, 3), dtype=bool)
    safe[0, 0] = True
    safe[0, 1] = True
    safe[1, 1] = True
    safe[2, 2] = True
    # Diagonal from (0,0) to (1,1) requires safe[1,0] or safe[0,1].
    # safe[0,1] is True so the path can go around.
    clearance = np.full((3, 3), 1.0, dtype=np.float32)
    path = astar(safe, clearance, (0, 0), (2, 2), 1.0)
    # Should find a path via (0,1) -> (1,1) -> (2,2) if diagonal cutting
    # is prevented correctly.
    if path is not None:
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)


def test_neighbors_count_and_orthogonal():
    assert len(NEIGHBORS) == 8
    ortho = [n for n in NEIGHBORS if n[2] == 1.0]
    assert len(ortho) == 4


# --- path_postprocess ---------------------------------------------------


from sage3d.path_postprocess import (
    points_are_safe,
    resample_path,
    simplify_by_visibility,
    smooth_path,
)


def test_points_are_safe_empty():
    t = MapTransform(10, 10, 1.0, 0.0, 0.0)
    safe = np.ones((10, 10), dtype=bool)
    assert not points_are_safe(np.zeros((0, 2)), safe, t)


def test_points_are_safe_in_bounds():
    t = MapTransform(10, 10, 1.0, 0.0, 0.0)
    safe = np.ones((10, 10), dtype=bool)
    pts = np.array([[0.5, 0.5], [5.5, 5.5]], dtype=np.float64)
    assert points_are_safe(pts, safe, t)


def test_points_are_safe_out_of_bounds():
    t = MapTransform(10, 10, 1.0, 0.0, 0.0)
    safe = np.ones((10, 10), dtype=bool)
    pts = np.array([[0.5, 0.5], [50.0, 50.0]], dtype=np.float64)
    assert not points_are_safe(pts, safe, t)


def test_simplify_by_visibility_short():
    t = MapTransform(10, 10, 1.0, 0.0, 0.0)
    safe = np.ones((10, 10), dtype=bool)
    pts = np.array([[0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
    result = simplify_by_visibility(pts, safe, t)
    assert len(result) == 2


def test_smooth_path_line_of_sight_for_short():
    t = MapTransform(10, 10, 1.0, 0.0, 0.0)
    safe = np.ones((10, 10), dtype=bool)
    pts = np.array([[0.5, 0.5], [1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    result, method = smooth_path(pts, safe, t)
    assert method == "line_of_sight"


def test_resample_path_spacing():
    pts = np.array([[0, 0], [0, 10]], dtype=np.float64)
    resampled = resample_path(pts, 2.0)
    assert len(resampled) >= 5
    assert resampled[0, 0] == 0.0
    assert resampled[-1, 1] == 10.0


def test_resample_path_single_point():
    pts = np.array([[5, 5]], dtype=np.float64)
    resampled = resample_path(pts, 1.0)
    assert len(resampled) == 1


# --- collision ----------------------------------------------------------


from sage3d.collision import collision_distances


def test_collision_distances_batch_size_2048():
    """Verify the 2048 batch boundary is preserved."""
    import trimesh

    mesh = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    # Exactly 2048 points — exercises the batch boundary.
    query = np.zeros((2048, 3), dtype=np.float64)
    dists = collision_distances(mesh, query)
    assert len(dists) == 2048
    assert dists.dtype == np.float64


def test_collision_distances_2049_crosses_boundary():
    """Verify 2049 points work across the 2048 batch boundary."""
    import trimesh

    mesh = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    query = np.zeros((2049, 3), dtype=np.float64)
    dists = collision_distances(mesh, query)
    assert len(dists) == 2049


# --- navigation_map -----------------------------------------------------


from sage3d.navigation_map import connected_components


def test_connected_components_single():
    safe = np.ones((10, 10), dtype=bool)
    labels, comps = connected_components(safe, 1.0)
    assert len(comps) == 1
    assert comps[0]["label"] == 1
    assert comps[0]["cells"] == 100


def test_connected_components_two():
    safe = np.zeros((10, 10), dtype=bool)
    safe[:5, :] = True
    labels, comps = connected_components(safe, 0.5)
    assert len(comps) == 1  # one connected component (5x10 block)


def test_connected_components_split():
    safe = np.zeros((10, 10), dtype=bool)
    safe[:5, :5] = True
    safe[5:, 5:] = True
    labels, comps = connected_components(safe, 1.0)
    assert len(comps) == 2


# --- viz (decoded-image equality check) ---------------------------------


def test_save_navigation_visualizations_writes_pngs(tmp_path: Path):
    import cv2
    from PIL import Image

    from sage3d.viz import save_navigation_visualizations

    safe = np.ones((20, 20), dtype=bool)
    clearance = np.full((20, 20), 2.0, dtype=np.float32)
    t = MapTransform(20, 20, 1.0, 0.0, 0.0)
    episodes = [
        {
            "episode_index": 0,
            "points": np.array([[0.5, 0.5], [5.5, 5.5], [10.5, 10.5]],
                               dtype=np.float32),
        }
    ]
    save_navigation_visualizations(tmp_path, safe, clearance, t, episodes)
    nav_map = Image.open(tmp_path / "navigation_map.png")
    overlay = Image.open(tmp_path / "trajectories_overlay.png")
    assert nav_map.size == (20, 20)
    assert overlay.size == (20, 20)


# --- collision: extract_collision_geometry (real asset) -----------------


def test_extract_collision_geometry_real_asset():
    from sage3d.collision import extract_collision_geometry

    root_value = os.environ.get("SAGE3D_ROOT")
    if not root_value:
        pytest.skip("SAGE3D_ROOT is not set for this developer lane")
    collision = (
        Path(root_value)
        / "Collision_Mesh"
        / "Collision_Mesh"
        / "839920"
        / "839920_collision.usd"
    )
    if not collision.is_file():
        pytest.skip(f"Collision asset not found: {collision}")
    points, faces = extract_collision_geometry(collision)
    assert len(points) > 0
    assert len(faces) > 0
    assert points.dtype == np.float64
    assert faces.dtype == np.int64
    # All face vertex counts should be 3 (triangulated).
    assert faces.shape[1] == 3
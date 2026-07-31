"""Collision geometry extraction and distance queries.

Isaac-lane (imports pxr, trimesh).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from pxr import Usd, UsdGeom

from sage3d.geometry import MapTransform


def extract_collision_geometry(
    collision_usd: Path,
) -> tuple[np.ndarray, np.ndarray]:
    stage = Usd.Stage.Open(str(collision_usd))
    if stage is None:
        raise RuntimeError(f"Could not open collision USD: {collision_usd}")
    transform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    chunks = []
    face_chunks = []
    vertex_offset = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        points_value = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not points_value:
            continue
        counts = np.asarray(
            UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get(), dtype=np.int64
        )
        indices = np.asarray(
            UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get(), dtype=np.int64
        )
        if not len(counts) or not np.all(counts == 3):
            raise RuntimeError(
                f"Collision mesh {prim.GetPath()} is not fully triangulated"
            )
        points = np.asarray(points_value, dtype=np.float64)
        matrix = np.asarray(
            transform_cache.GetLocalToWorldTransform(prim), dtype=np.float64
        )
        homogeneous = np.column_stack((points, np.ones(len(points))))
        world_points = (homogeneous @ matrix)[:, :3]
        chunks.append(world_points)
        face_chunks.append(indices.reshape(-1, 3) + vertex_offset)
        vertex_offset += len(world_points)
    if not chunks:
        raise RuntimeError(f"No mesh vertices found in {collision_usd}")
    return np.concatenate(chunks, axis=0), np.concatenate(face_chunks, axis=0)


def collision_distances(
    mesh: trimesh.Trimesh, query_points: np.ndarray
) -> np.ndarray:
    distances = np.empty(len(query_points), dtype=np.float64)
    for start in range(0, len(query_points), 2048):
        stop = min(start + 2048, len(query_points))
        _, batch_distances, _ = trimesh.proximity.closest_point(
            mesh, query_points[start:stop]
        )
        distances[start:stop] = batch_distances
    return distances


def apply_camera_clearance(
    safe: np.ndarray,
    mesh: trimesh.Trimesh,
    transform: MapTransform,
    camera_height: float,
    camera_clearance: float,
) -> tuple[np.ndarray, dict]:
    rows, cols = np.where(safe)
    query_points = np.asarray(
        [
            (*transform.pixel_to_world(int(row), int(col)), camera_height)
            for row, col in zip(rows, cols)
        ],
        dtype=np.float64,
    )
    distances = collision_distances(mesh, query_points)
    distance_map = np.zeros(safe.shape, dtype=np.float32)
    distance_map[rows, cols] = distances.astype(np.float32)
    camera_safe = safe & (distance_map >= camera_clearance)
    removed = int(safe.sum() - camera_safe.sum())
    return (
        camera_safe,
        {
            "camera_height_m": camera_height,
            "required_camera_clearance_m": camera_clearance,
            "queried_2d_safe_cells": int(len(query_points)),
            "removed_cells": removed,
            "remaining_cells": int(camera_safe.sum()),
            "remaining_area_m2": float(camera_safe.sum() * transform.scale**2),
        },
    )
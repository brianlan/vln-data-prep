"""Navigation map loading and connected-component analysis.

Isaac-lane (imports cv2, PIL).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sage3d.geometry import MapTransform


def load_navigation_map(
    scene_dir: Path, robot_radius: float, safety_margin: float
) -> tuple[np.ndarray, np.ndarray, MapTransform, dict]:
    occupancy_path = scene_dir / "occupancy.png"
    occupancy_meta_path = scene_dir / "occupancy.json"
    structure_path = scene_dir / "structure.json"
    for path in (occupancy_path, occupancy_meta_path, structure_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    occupancy = np.fliplr(np.asarray(Image.open(occupancy_path).convert("L")))
    with occupancy_meta_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    with structure_path.open("r", encoding="utf-8") as file:
        structure = json.load(file)

    height, width = occupancy.shape
    transform = MapTransform(
        height=height,
        width=width,
        scale=float(metadata["scale"]),
        lower_x=float(metadata["lower"][0]),
        lower_y=float(metadata["lower"][1]),
    )

    room_mask = np.zeros((height, width), dtype=np.uint8)
    valid_rooms = 0
    for room in structure.get("rooms", []):
        profile = room.get("profile", [])
        if len(profile) < 3:
            continue
        pixels = []
        for x, y in profile:
            row, col = transform.world_to_pixel(float(x), float(y))
            pixels.append((col, row))
        cv2.fillPoly(room_mask, [np.asarray(pixels, dtype=np.int32)], 1)
        valid_rooms += 1
    if valid_rooms == 0:
        raise RuntimeError(f"No valid room polygons in {structure_path}")

    # Unknown and exterior pixels are deliberately blocked. Some InteriorGS
    # occupancy PNGs use white for both interior free space and canvas background.
    raw_free = (occupancy == 255) & (room_mask > 0)
    clearance_m = (
        cv2.distanceTransform(
            raw_free.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        * transform.scale
    )
    safe = raw_free & (clearance_m >= robot_radius + safety_margin)

    map_info = {
        "shape": [height, width],
        "scale_m_per_pixel": transform.scale,
        "robot_radius_m": robot_radius,
        "safety_margin_m": safety_margin,
        "required_path_clearance_m": robot_radius + safety_margin,
        "room_count": valid_rooms,
        "raw_free_area_m2": float(raw_free.sum() * transform.scale**2),
        "safe_free_area_m2": float(safe.sum() * transform.scale**2),
        "occupancy_values": {
            str(int(value)): int(count)
            for value, count in zip(*np.unique(occupancy, return_counts=True))
        },
    }
    return safe, clearance_m, transform, map_info


def connected_components(safe: np.ndarray, scale: float) -> tuple[np.ndarray, list[dict]]:
    component_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(safe.astype(np.uint8), connectivity=4)
    )
    components = []
    for label in range(1, component_count):
        cells = int(component_stats[label, cv2.CC_STAT_AREA])
        components.append(
            {
                "label": label,
                "cells": cells,
                "area_m2": cells * scale**2,
            }
        )
    return component_labels, components
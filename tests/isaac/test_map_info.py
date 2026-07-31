"""MapInfo dataclass tests for sage3d.navigation_map (Phase 3b, issue #15).

Verifies that MapInfo.to_dict() preserves the exact serialized field values
+ key order of the legacy dict construction (including the post-clearance
mutations that were previously done in main()).

Isaac-lane: imports cv2, PIL (via sage3d.navigation_map).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sage3d.navigation_map import MapInfo  # noqa: E402


# --- legacy dict key order (from the pre-3b generate_sage3d_trajectories.py) --
# After load_navigation_map + camera_clearance + connected_components, the
# final dict had these keys in this insertion order:
#   shape, scale_m_per_pixel, robot_radius_m, safety_margin_m,
#   required_path_clearance_m, room_count, raw_free_area_m2,
#   safe_free_area_m2, occupancy_values, components, camera_collision_filter
_LEGACY_KEYS = [
    "shape",
    "scale_m_per_pixel",
    "robot_radius_m",
    "safety_margin_m",
    "required_path_clearance_m",
    "room_count",
    "raw_free_area_m2",
    "safe_free_area_m2",
    "occupancy_values",
    "components",
    "camera_collision_filter",
]


def _sample_map_info() -> MapInfo:
    return MapInfo(
        shape=[100, 80],
        scale_m_per_pixel=0.05,
        robot_radius_m=0.25,
        safety_margin_m=0.05,
        required_path_clearance_m=0.30,
        room_count=3,
        raw_free_area_m2=50.0,
        safe_free_area_m2=40.0,
        occupancy_values={"255": 5000, "0": 3000},
        components=[{"label": 1, "cells": 100, "area_m2": 0.25}],
        camera_collision_filter={
            "camera_height_m": 0.6,
            "required_camera_clearance_m": 0.25,
        },
    )


def test_map_info_to_dict_keys_match_legacy_order():
    info = _sample_map_info()
    d = info.to_dict()
    assert list(d.keys()) == _LEGACY_KEYS


def test_map_info_to_dict_values_match():
    info = _sample_map_info()
    d = info.to_dict()
    assert d["shape"] == [100, 80]
    assert d["scale_m_per_pixel"] == 0.05
    assert d["robot_radius_m"] == 0.25
    assert d["safety_margin_m"] == 0.05
    assert d["required_path_clearance_m"] == 0.30
    assert d["room_count"] == 3
    assert d["raw_free_area_m2"] == 50.0
    assert d["safe_free_area_m2"] == 40.0
    assert d["occupancy_values"] == {"255": 5000, "0": 3000}
    assert d["components"] == [{"label": 1, "cells": 100, "area_m2": 0.25}]
    assert d["camera_collision_filter"] == {
        "camera_height_m": 0.6,
        "required_camera_clearance_m": 0.25,
    }


def test_map_info_default_components_and_filter():
    """MapInfo with defaults for components and camera_collision_filter."""
    info = MapInfo(
        shape=[10, 10],
        scale_m_per_pixel=1.0,
        robot_radius_m=0.2,
        safety_margin_m=0.0,
        required_path_clearance_m=0.2,
        room_count=1,
        raw_free_area_m2=10.0,
        safe_free_area_m2=8.0,
        occupancy_values={"255": 80},
    )
    d = info.to_dict()
    assert d["components"] == []
    assert d["camera_collision_filter"] == {}
    assert list(d.keys()) == _LEGACY_KEYS


def test_map_info_to_dict_is_json_serializable():
    import json

    info = _sample_map_info()
    d = info.to_dict()
    # Must not raise.
    json.dumps(d)


def test_map_info_is_frozen():
    info = _sample_map_info()
    with pytest.raises((AttributeError, Exception)):
        info.shape = [200, 200]  # type: ignore[misc]
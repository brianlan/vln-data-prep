"""Isaac-lane prerequisites for the pinned Phase 0b capture."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pxr import Usd


def test_scene_839920_assets_open_under_isaac_python():
    root_value = os.environ.get("SAGE3D_ROOT")
    if not root_value:
        pytest.skip("SAGE3D_ROOT is not set for this developer lane")
    root = Path(root_value)
    usdz = root / "InteriorGS_usdz" / "839920.usdz"
    collision = (
        root
        / "Collision_Mesh"
        / "Collision_Mesh"
        / "839920"
        / "839920_collision.usd"
    )
    assert usdz.is_file()
    assert collision.is_file()
    assert Usd.Stage.Open(str(collision)) is not None

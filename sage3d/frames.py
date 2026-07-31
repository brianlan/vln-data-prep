"""Frame transforms and camera extrinsics (numpy only).

Package-safe: depends only on numpy. ``camera_extrinsic`` and
``yaw_to_quaternion`` preserve the exact formulas used by
``render_fisheye_sage3d.py`` and ``package_lerobot_sage3d.py`` so that later
wiring phases can delegate here without behavior drift.
"""

from __future__ import annotations

import numpy as np

# Coordinate frame advertised by packaged episode records. Matches the
# ``coordinate_frame`` field written by ``package_lerobot_sage3d.py``.
COORDINATE_FRAME = "world_z_up_x_forward"


def yaw_to_quaternion(yaw: float) -> np.ndarray:
    """Return Isaac scalar-first quaternion ``[w, x, y, z]`` for ``yaw``.

    Isaac's world camera axes are +X forward and +Z up; the rotation is purely
    about Z, so this matches ``render_fisheye_sage3d.py::camera_quaternion``.
    """
    return np.asarray(
        [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
        dtype=np.float32,
    )


def camera_extrinsic(camera_height: float) -> np.ndarray:
    """Return the camera-to-robot-base extrinsic for the given camera height.

    Identity rotation with ``[2, 3] == camera_height``; matches
    ``package_lerobot_sage3d.py::camera_extrinsic`` exactly.
    """
    transform = np.eye(4, dtype=np.float32)
    transform[2, 3] = camera_height
    return transform


def yaw_to_rotation2d(yaw: float) -> np.ndarray:
    """Return the 2x2 world-to-robot-base rotation matrix for ``yaw``.

    Matches the per-frame action block written by
    ``generate_sage3d_trajectories.py::build_episode_arrays``:
    ``[[cos, -sin], [sin, cos]]``.
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    return np.asarray(
        [[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]],
        dtype=np.float32,
    )
"""SAGE3D episode NPZ schema (numpy only).

``EpisodeArrays`` is the single load/save authority for the per-episode
``episode_*.npz`` files written by generation and consumed by render and
package. It preserves the exact keys, dtypes, and ``np.savez_compressed`` call
used by ``generate_sage3d_trajectories.py`` so later wiring phases do not drift
the artifact contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Authoritative key set written to every episode npz.
EPISODE_KEYS = (
    "points",
    "actions",
    "camera_positions",
    "yaw",
    "point_goal",
    "start_position",
    "goal_position",
)


class EpisodeArrays:
    """Typed view of one episode's NPZ arrays."""

    __slots__ = EPISODE_KEYS

    def __init__(
        self,
        *,
        points: np.ndarray,
        actions: np.ndarray,
        camera_positions: np.ndarray,
        yaw: np.ndarray,
        point_goal: np.ndarray,
        start_position: np.ndarray,
        goal_position: np.ndarray,
    ) -> None:
        self.points = np.asarray(points, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)
        self.camera_positions = np.asarray(camera_positions, dtype=np.float32)
        self.yaw = np.asarray(yaw, dtype=np.float32)
        self.point_goal = np.asarray(point_goal, dtype=np.float32)
        self.start_position = np.asarray(start_position, dtype=np.float32)
        self.goal_position = np.asarray(goal_position, dtype=np.float32)

    def to_dict(self) -> dict[str, np.ndarray]:
        return {key: getattr(self, key) for key in EPISODE_KEYS}


def save_episode(path: Path, episode: EpisodeArrays) -> None:
    """Write one episode npz with the exact legacy key set + compression."""
    np.savez_compressed(path, **episode.to_dict())


def load_episode(path: Path) -> EpisodeArrays:
    """Load and validate one episode npz."""
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in EPISODE_KEYS if key not in data]
        if missing:
            raise KeyError(
                f"{path} is missing episode keys: {missing}"
            )
        arrays = {key: data[key] for key in EPISODE_KEYS}
    return EpisodeArrays(**arrays)
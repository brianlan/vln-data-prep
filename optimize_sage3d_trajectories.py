import argparse
import json
from pathlib import Path

import numpy as np
from box import Box
from PIL import Image
from loguru import logger
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--scene-id", type=str, required=True)
    return parser.parse_args()


def main(args):
    scene_dir = args.scene_root / args.scene_id
    safe_mask = load_safe_mask(scene_dir / "map" / "safe_mask.png")
    esdf = load_esdf(scene_dir / "map" / "esdf.npy")
    episode_manifest: Box = load_episode_manifest(
        scene_dir / "trajectories" / "trajectory_manifest.json"
    )
    for eid in tqdm(range(episode_manifest.episode_count)):
        episode = np.load(scene_dir / "trajectories" / f"episode_{eid:06}.npz")
        init_traj = get_init_traj_from_episode(episode)
        traj = optimize_trajectory(init_traj, safe_mask, esdf)


def get_init_traj_from_episode(episode):
    """
    get trajectory position from episode['points'] and get yaw from episode['actions']
    return a (N, 3) np.ndarray with each row (x, y, yaw)
    """
    points = episode["points"]  # (N, 2)
    yaw = episode["yaw"][:, None]  # (N, 1)
    return np.hstack([points, yaw])


def load_safe_mask(path: Path) -> np.ndarray:
    """Load safe mask PNG as a boolean array (True = navigable)."""
    return np.array(Image.open(path)) > 0


def load_esdf(path: Path) -> np.ndarray:
    """Load Euclidean signed distance field (.npy)."""
    return np.load(path)


def load_episode_manifest(path: Path) -> Box:
    with open(path, "r") as f:
        return Box(json.load(f))


def optimize_trajectory(trajectory: np.ndarray, safe_mask: np.ndarray, esdf: np.ndarray):
    pass


if __name__ == "__main__":
    main(parse_args())

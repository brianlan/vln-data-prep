"""Navigation visualization output.

Isaac-lane (imports cv2, PIL).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from sage3d.geometry import MapTransform


def save_navigation_visualizations(
    output_dir: Path,
    safe: np.ndarray,
    clearance_m: np.ndarray,
    transform: MapTransform,
    episodes: list[dict],
) -> None:
    safe_image = np.zeros((*safe.shape, 3), dtype=np.uint8)
    normalized_clearance = np.clip(clearance_m / max(clearance_m.max(), 1e-6), 0, 1)
    safe_image[..., 0] = (normalized_clearance * 120).astype(np.uint8)
    safe_image[..., 1] = np.where(safe, 180, 0).astype(np.uint8)
    safe_image[..., 2] = np.where(safe, 80, 0).astype(np.uint8)
    Image.fromarray(safe_image).save(output_dir / "navigation_map.png")

    overlay = safe_image.copy()
    colors = (
        (255, 80, 80),
        (80, 180, 255),
        (255, 210, 70),
        (180, 80, 255),
        (80, 255, 160),
        (255, 130, 30),
    )
    for episode in episodes:
        pixels = [
            transform.world_to_pixel(float(x), float(y))
            for x, y in episode["points"]
        ]
        polyline = np.asarray([(col, row) for row, col in pixels], dtype=np.int32)
        color = colors[episode["episode_index"] % len(colors)]
        cv2.polylines(overlay, [polyline], False, color, 2, cv2.LINE_AA)
        cv2.circle(overlay, tuple(polyline[0]), 3, (255, 255, 255), -1)
        cv2.circle(overlay, tuple(polyline[-1]), 3, color, -1)
    Image.fromarray(overlay).save(output_dir / "trajectories_overlay.png")
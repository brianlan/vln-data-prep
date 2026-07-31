"""Path post-processing: safety checks, simplification, smoothing, resampling.

Isaac-lane (imports scipy).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import splprep, splev

from sage3d.geometry import MapTransform


def points_are_safe(
    points: np.ndarray, safe: np.ndarray, transform: MapTransform
) -> bool:
    if len(points) == 0:
        return False
    step = transform.scale * 0.5
    samples = []
    for index in range(len(points) - 1):
        start, end = points[index], points[index + 1]
        length = float(np.linalg.norm(end - start))
        count = max(2, int(math.ceil(length / step)) + 1)
        samples.append(np.linspace(start, end, count))
    if samples:
        test_points = np.concatenate(samples, axis=0)
    else:
        test_points = points
    for x, y in test_points:
        row, col = transform.world_to_pixel(float(x), float(y))
        if not (
            0 <= row < transform.height
            and 0 <= col < transform.width
            and safe[row, col]
        ):
            return False
    return True


def simplify_by_visibility(
    points: np.ndarray, safe: np.ndarray, transform: MapTransform
) -> np.ndarray:
    if len(points) <= 2:
        return points
    simplified = [points[0]]
    current = 0
    while current < len(points) - 1:
        candidate = len(points) - 1
        while candidate > current + 1:
            if points_are_safe(
                np.asarray([points[current], points[candidate]]),
                safe,
                transform,
            ):
                break
            candidate -= 1
        simplified.append(points[candidate])
        current = candidate
    return np.asarray(simplified)


def smooth_path(
    points: np.ndarray, safe: np.ndarray, transform: MapTransform
) -> tuple[np.ndarray, str]:
    simplified = simplify_by_visibility(points, safe, transform)
    if len(simplified) < 4:
        return simplified, "line_of_sight"

    distances = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(simplified, axis=0), axis=1)))
    )
    if distances[-1] <= 0:
        return simplified, "line_of_sight"
    parameter = distances / distances[-1]
    sample_count = max(
        len(simplified) * 8,
        int(math.ceil(distances[-1] / (transform.scale * 0.4))) + 1,
    )
    sample_parameter = np.linspace(0.0, 1.0, sample_count)

    for smoothing_per_point in (0.002, 0.0005, 0.0):
        try:
            spline, _ = splprep(
                [simplified[:, 0], simplified[:, 1]],
                u=parameter,
                s=smoothing_per_point * len(simplified),
                k=min(3, len(simplified) - 1),
            )
            x_values, y_values = splev(sample_parameter, spline)
            candidate = np.column_stack((x_values, y_values))
            candidate[0] = points[0]
            candidate[-1] = points[-1]
            if points_are_safe(candidate, safe, transform):
                return candidate, f"cubic_spline_s={smoothing_per_point}"
        except ValueError:
            continue

    # Cubic splines can overshoot at tight obstacle corners. Round each corner
    # independently with a quadratic Bezier curve and retain a sharp corner only
    # where the rounded candidate would violate the clearance mask.
    rounded = [simplified[0]]
    rounded_corner_count = 0
    for index in range(1, len(simplified) - 1):
        previous, corner, following = simplified[index - 1 : index + 2]
        incoming = corner - previous
        outgoing = following - corner
        incoming_length = float(np.linalg.norm(incoming))
        outgoing_length = float(np.linalg.norm(outgoing))
        if incoming_length < 1e-6 or outgoing_length < 1e-6:
            rounded.append(corner)
            continue
        cut = min(0.25, incoming_length * 0.3, outgoing_length * 0.3)
        entry = corner - incoming / incoming_length * cut
        exit_point = corner + outgoing / outgoing_length * cut
        parameter = np.linspace(0.0, 1.0, 9)
        curve = (
            (1.0 - parameter)[:, None] ** 2 * entry
            + 2.0
            * (1.0 - parameter)[:, None]
            * parameter[:, None]
            * corner
            + parameter[:, None] ** 2 * exit_point
        )
        candidate = np.vstack((rounded[-1], curve))
        if points_are_safe(candidate, safe, transform):
            rounded.extend(curve)
            rounded_corner_count += 1
        else:
            rounded.append(corner)
    rounded.append(simplified[-1])
    rounded_array = np.asarray(rounded)
    if rounded_corner_count and points_are_safe(
        rounded_array, safe, transform
    ):
        return (
            rounded_array,
            f"clearance_checked_bezier_corners_{rounded_corner_count}",
        )
    return simplified, "line_of_sight_fallback"


def resample_path(points: np.ndarray, spacing: float) -> np.ndarray:
    if len(points) < 2:
        return points.copy()
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = cumulative[-1]
    if total <= 0:
        return points[:1].copy()
    sample_distances = np.arange(0.0, total, spacing)
    if not np.isclose(sample_distances[-1], total):
        sample_distances = np.append(sample_distances, total)
    x_values = np.interp(sample_distances, cumulative, points[:, 0])
    y_values = np.interp(sample_distances, cumulative, points[:, 1])
    return np.column_stack((x_values, y_values))
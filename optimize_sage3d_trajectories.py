import argparse
import json
from pathlib import Path

import numpy as np
from box import Box
from PIL import Image
from scipy.interpolate import BSpline
from tqdm import tqdm


# --------------------------------------------------------------------------
# Work package 6.2: quintic open-uniform clamped B-spline math kernel.
# Pure functions. No A*, no collision, no NLP. T selection and control-point
# initialization belong to work package 6.3.
# --------------------------------------------------------------------------

SPLINE_DEGREE = 5
CONTROL_DT = 0.1
_DT_TOL = 1e-9
_ENDPOINT_ATOL = 1e-12

# 3-point Gauss-Legendre nodes/weights on [-1, 1] (reference values).
_GL3_NODES = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
_GL3_WEIGHTS = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])


def clamped_knots(n_ctrl: int) -> np.ndarray:
    """Return a quintic open-uniform clamped knot vector over [0, 1]."""
    if n_ctrl < SPLINE_DEGREE + 1:
        raise ValueError(
            f"need at least {SPLINE_DEGREE + 1} control points, got {n_ctrl}"
        )
    interior_count = n_ctrl - SPLINE_DEGREE - 1
    interior = (
        np.arange(1, interior_count + 1) / (interior_count + 1)
        if interior_count > 0
        else np.zeros(0)
    )
    return np.concatenate(
        [
            np.zeros(SPLINE_DEGREE + 1),
            interior,
            np.ones(SPLINE_DEGREE + 1),
        ]
    )


def build_clamped_spline(control_points: np.ndarray) -> BSpline:
    """Build a clamped quintic B-spline over u in [0, 1]."""
    control_points = np.asarray(control_points, dtype=float)
    knots = clamped_knots(control_points.shape[0])
    return BSpline(knots, control_points, SPLINE_DEGREE, extrapolate=False)


def derivative_control_points(
    knots: np.ndarray, control_points: np.ndarray, order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit r-th derivative control points of a B-spline (de Boor).

    At each step the current spline has knot vector t, coefficients c, degree
    p; the derivative has knots t[1:-1], degree p-1, and
        Q_i = p * (c_{i+1} - c_i) / (t_{i+p+1} - t_{i+1}).
    Returns (derivative_knots, derivative_control_points). The control points
    have shape (n_ctrl - order, dim).
    """
    if order < 0 or order > SPLINE_DEGREE:
        raise ValueError(
            f"order must be in [0, {SPLINE_DEGREE}], got {order}"
        )
    t = np.asarray(knots, dtype=float)
    c = np.asarray(control_points, dtype=float)
    p = SPLINE_DEGREE
    for _ in range(order):
        denom = t[p + 1 : p + 1 + c.shape[0] - 1] - t[1 : c.shape[0]]
        c = (p / denom[:, None]) * (c[1:] - c[:-1])
        t = t[1:-1]
        p -= 1
    return t, c


def eval_derivatives(control_points: np.ndarray, T: float, u: np.ndarray) -> dict:
    """Evaluate q(u) and its 1/2/3 parametric and time derivatives at u in [0,1].

    Real-time scaling: q_t^(k) = (1/T^k) * q_u^(k). T must be positive finite.
    """
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError(f"T must be positive and finite, got {T}")
    u = np.asarray(u, dtype=float)
    spline = build_clamped_spline(control_points)
    d1 = spline.derivative(1)
    d2 = spline.derivative(2)
    d3 = spline.derivative(3)
    pos = spline(u)
    return {
        "position": pos,
        "velocity": d1(u) / T,
        "acceleration": d2(u) / T**2,
        "jerk": d3(u) / T**3,
    }


def jerk_integral_sq(control_points: np.ndarray, T: float) -> float:
    """Squared-norm jerk integral over [0, T].

        int_0^T ||q_t^(3)(t)||^2 dt = (1/T^5) * int_0^1 ||q_u^(3)(u)||^2 du.

    The third parametric derivative of a quintic is quadratic, so its squared
    norm is a degree-4 polynomial in u. 3-point Gauss-Legendre quadrature per
    nonzero knot span is therefore exact up to floating-point error.
    """
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError(f"T must be positive and finite, got {T}")
    spline = build_clamped_spline(control_points)
    d3 = spline.derivative(3)
    knots = spline.t
    total = 0.0
    for a, b in zip(knots[:-1], knots[1:]):
        if b <= a:
            continue
        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        xs = mid + half * _GL3_NODES
        vals = d3(xs)
        total += half * np.sum(_GL3_WEIGHTS * np.sum(vals**2, axis=1))
    return float(total / T**5)


def yaw_unwrap(yaw: np.ndarray) -> np.ndarray:
    """Unwrap a 1D yaw sequence (radians) to remove 2*pi discontinuities."""
    return np.unwrap(np.asarray(yaw, dtype=float))


def yaw_wrap(yaw: np.ndarray) -> np.ndarray:
    """Wrap yaw (radians) to [-pi, pi)."""
    return (np.asarray(yaw, dtype=float) + np.pi) % (2 * np.pi) - np.pi


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
        # WP 6.2: optimize_trajectory requires total_time from WP 6.3.
        traj = optimize_trajectory(init_traj, safe_mask, esdf, total_time=8)


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


def optimize_trajectory(
    trajectory: np.ndarray,
    safe_mask: np.ndarray,
    esdf: np.ndarray,
    *,
    total_time: float,
) -> dict:
    """Evaluate explicit quintic control points on the fixed 0.1 s grid.

    Control-point construction, time selection, collision checks, and nonlinear
    optimization belong to later work packages. `safe_mask` and `esdf` are
    reserved for that integration.
    """
    T = float(total_time)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError(f"total_time must be positive and finite, got {T}")
    n_steps = int(round(T / CONTROL_DT))
    if abs(T - n_steps * CONTROL_DT) > _DT_TOL:
        raise ValueError(
            f"total_time must be aligned to dt={CONTROL_DT} s within "
            f"tolerance {_DT_TOL}, got T={T}"
        )
    if n_steps < 1:
        raise ValueError(
            f"total_time must be at least one control step ({CONTROL_DT} s), "
            f"got T={T}"
        )
    # Canonicalize T to the nearest integer number of control steps.
    T = n_steps * CONTROL_DT

    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError(
            f"trajectory must be (N, 3), got shape {trajectory.shape}"
        )
    if trajectory.shape[0] < SPLINE_DEGREE + 1:
        raise ValueError(
            f"need at least {SPLINE_DEGREE + 1} control points, "
            f"got {trajectory.shape[0]}"
        )
    if not np.all(np.isfinite(trajectory)):
        raise ValueError("trajectory must contain only finite values")

    # Unwrap yaw before endpoint validation so coterminal representations
    # (e.g. -pi vs +pi) compare on the same continuous branch.
    yaw_u = yaw_unwrap(trajectory[:, 2])
    ctrl = np.hstack([trajectory[:, :2], yaw_u[:, None]])

    # Endpoint zero-velocity control-point relationships (x, y, unwrapped yaw).
    # Use rtol=0 with an explicit absolute tolerance instead of np.allclose's
    # default relative tolerance.
    if not np.allclose(ctrl[1], ctrl[0], rtol=0.0, atol=_ENDPOINT_ATOL):
        raise ValueError(
            "endpoint zero-velocity requires P1 == P0 for all (x, y, yaw)"
        )
    if not np.allclose(ctrl[-2], ctrl[-1], rtol=0.0, atol=_ENDPOINT_ATOL):
        raise ValueError(
            "endpoint zero-velocity requires P[-2] == P[-1] for all (x, y, yaw)"
        )

    # Grid constructed from integer indices times CONTROL_DT; uniform within
    # floating-point precision.
    t = np.arange(n_steps + 1, dtype=float) * CONTROL_DT
    u = t / T

    ev = eval_derivatives(ctrl, T, u)

    yaw_unwrapped = ev["position"][:, 2]
    return {
        "time": t,
        "position_world": ev["position"][:, :2],
        "yaw_unwrapped": yaw_unwrapped,
        "yaw_wrapped": yaw_wrap(yaw_unwrapped),
        "velocity_world": ev["velocity"][:, :2],
        "acceleration_world": ev["acceleration"][:, :2],
        "jerk_world": ev["jerk"][:, :2],
        "yaw_rate": ev["velocity"][:, 2],
        "yaw_acceleration": ev["acceleration"][:, 2],
        "yaw_jerk": ev["jerk"][:, 2],
        "total_time": T,
    }


if __name__ == "__main__":
    main(parse_args())

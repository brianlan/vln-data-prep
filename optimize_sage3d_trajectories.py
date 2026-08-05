import argparse
import json
from pathlib import Path

import numpy as np
from box import Box
from PIL import Image
from loguru import logger
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


def _validate_control_points(
    control_points: np.ndarray, degree: int = SPLINE_DEGREE
) -> np.ndarray:
    """Validate and coerce a control-point array for the math kernel.

    Requires a finite 2D array with at least `degree + 1` rows and a nonzero
    component dimension. Returns the float-cast array.
    """
    c = np.asarray(control_points, dtype=float)
    if c.ndim != 2:
        raise ValueError(f"control_points must be 2D, got {c.ndim}D")
    if c.shape[1] == 0:
        raise ValueError("control_points must have a nonzero component dimension")
    if c.shape[0] < degree + 1:
        raise ValueError(
            f"need at least degree+1={degree + 1} control points, got {c.shape[0]}"
        )
    if not np.all(np.isfinite(c)):
        raise ValueError("control_points must contain only finite values")
    return c


def clamped_knots(n_ctrl: int, degree: int = SPLINE_DEGREE) -> np.ndarray:
    """Open-uniform clamped knot vector over [0, 1] for n_ctrl control points.

    Returns (n_ctrl + degree + 1,) knots with degree+1 zeros at the start,
    degree+1 ones at the end, and uniformly spaced interior knots.
    Raises ValueError when n_ctrl < degree + 1.
    """
    if n_ctrl < degree + 1:
        raise ValueError(
            f"need at least degree+1={degree + 1} control points, got {n_ctrl}"
        )
    interior_count = (n_ctrl - 1) - degree
    interior = (
        np.arange(1, interior_count + 1) / (interior_count + 1)
        if interior_count > 0
        else np.zeros(0)
    )
    return np.concatenate(
        [np.zeros(degree + 1), interior, np.ones(degree + 1)]
    )


def build_clamped_spline(
    control_points: np.ndarray, degree: int = SPLINE_DEGREE
) -> BSpline:
    """Build a clamped quintic B-spline over u in [0, 1].

    control_points: (n_ctrl, dim) array. Returns scipy.interpolate.BSpline.
    """
    control_points = _validate_control_points(control_points, degree)
    knots = clamped_knots(control_points.shape[0], degree)
    return BSpline(knots, control_points, degree, extrapolate=False)


def derivative_control_points(
    knots: np.ndarray, control_points: np.ndarray, degree: int, order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit r-th derivative control points of a B-spline (de Boor).

    At each step the current spline has knot vector t, coefficients c, degree
    p; the derivative has knots t[1:-1], degree p-1, and
        Q_i = p * (c_{i+1} - c_i) / (t_{i+p+1} - t_{i+1}).
    Returns (derivative_knots, derivative_control_points). The control points
    have shape (n_ctrl - order, dim). order must be in [0, degree].
    """
    if order < 0 or order > degree:
        raise ValueError(f"order must be in [0, {degree}], got {order}")
    t = np.asarray(knots, dtype=float)
    c = _validate_control_points(control_points, degree)
    # Knot vector must be long enough for the recurrence: at each step the
    # denominator slices t[p+1:p+1+n] and t[1:n] with n = c.shape[0].
    if t.ndim != 1:
        raise ValueError(f"knots must be 1D, got {t.ndim}D")
    if t.shape[0] < degree + 1 + c.shape[0]:
        raise ValueError(
            f"knots length {t.shape[0]} too short for {c.shape[0]} control "
            f"points at degree {degree}"
        )
    p = degree
    for _ in range(order):
        denom = t[p + 1 : p + 1 + c.shape[0] - 1] - t[1 : c.shape[0]]
        c = (p / denom[:, None]) * (c[1:] - c[:-1])
        t = t[1:-1]
        p -= 1
    return t, c


def eval_derivatives(
    control_points: np.ndarray, T: float, u: np.ndarray, degree: int = SPLINE_DEGREE
) -> dict:
    """Evaluate q(u) and its 1/2/3 parametric and time derivatives at u in [0,1].

    Real-time scaling: q_t^(k) = (1/T^k) * q_u^(k). T must be positive finite.
    `degree` must be at least 3 to support the third derivative. `u` values must
    be finite and within [0, 1].
    """
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError(f"T must be positive and finite, got {T}")
    if degree < 3:
        raise ValueError(f"degree must be >= 3 for three derivatives, got {degree}")
    u = np.asarray(u, dtype=float)
    if u.size == 0 or not np.all(np.isfinite(u)):
        raise ValueError("u must contain finite values")
    if np.min(u) < 0.0 or np.max(u) > 1.0:
        raise ValueError("u must lie within [0, 1]")
    spline = build_clamped_spline(control_points, degree)
    d1 = spline.derivative(1)
    d2 = spline.derivative(2)
    d3 = spline.derivative(3)
    pos = spline(u)
    return {
        "position": pos,
        "velocity": d1(u) / T,
        "acceleration": d2(u) / T**2,
        "jerk": d3(u) / T**3,
        "u_position": pos,
        "u_velocity": d1(u),
        "u_acceleration": d2(u),
        "u_jerk": d3(u),
        "T": T,
        "u": u,
    }


def jerk_integral_sq(
    control_points: np.ndarray, T: float, degree: int = SPLINE_DEGREE
) -> float:
    """Squared-norm jerk integral over [0, T].

        int_0^T ||q_t^(3)(t)||^2 dt = (1/T^5) * int_0^1 ||q_u^(3)(u)||^2 du.

    The third parametric derivative of a quintic is quadratic, so its squared
    norm is a degree-4 polynomial in u. 3-point Gauss-Legendre quadrature per
    nonzero knot span is therefore exact up to floating-point error. This
    exactness guarantee only holds for degree == SPLINE_DEGREE; other degrees
    are rejected to avoid silently returning inaccurate results.
    """
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError(f"T must be positive and finite, got {T}")
    if degree != SPLINE_DEGREE:
        raise ValueError(
            f"jerk_integral_sq only supports degree={SPLINE_DEGREE}, got {degree}"
        )
    spline = build_clamped_spline(control_points, degree)
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
    """Unwrap a 1D yaw sequence (radians) to remove 2*pi discontinuities.

    Rejects non-1D or nonfinite input.
    """
    yaw = np.asarray(yaw, dtype=float)
    if yaw.ndim != 1:
        raise ValueError(f"yaw must be 1D, got {yaw.ndim}D")
    if yaw.size and not np.all(np.isfinite(yaw)):
        raise ValueError("yaw must contain only finite values")
    if yaw.size == 0:
        return yaw.copy()
    out = np.empty_like(yaw)
    out[0] = yaw[0]
    for i in range(1, yaw.size):
        delta = ((yaw[i] - yaw[i - 1] + np.pi) % (2 * np.pi)) - np.pi
        out[i] = out[i - 1] + delta
    return out


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
    total_time: float | None = None,
) -> dict:
    """Apply the quintic clamped B-spline math kernel (work package 6.2).

    Phase-6.2 evaluation contract:
      - `trajectory` is an explicit (N, 3) array of de Boor control points with
        columns (x, y, yaw). It is NOT an optimized trajectory; A* initialization,
        collision constraints, and nonlinear optimization belong to work package
        6.3 and are deliberately not performed here.
      - `total_time` (T, seconds) must be supplied by the caller. Work package
        6.3 is responsible for constructing control points and selecting T. When
        `total_time` is None this function raises NotImplementedError.
      - `safe_mask` and `esdf` are collision-validation inputs reserved for
        later work packages; they are intentionally unused in this phase.

    Validation:
      - `trajectory` must be finite, 2D with shape (N, 3) and N >= 6.
      - `total_time` must be positive, finite, and aligned to the fixed control
        period dt=0.1 s within tolerance, and at least one step. It is
        canonicalized to n_steps * dt and the canonical value is returned.
      - Yaw is unwrapped before validating the endpoint zero-velocity
        relationships. P1 == P0 and P[-2] == P[-1] must hold for all of
        (x, y, unwrapped yaw); they are NOT silently enforced.

    Returns a dict (subset of the planned `TimedTrajectory`) evaluated on the
    fixed dt=0.1 s grid:
      {
        "time", "position_world", "yaw_unwrapped", "yaw_wrapped",
        "velocity_world", "acceleration_world", "jerk_world",
        "yaw_rate", "yaw_acceleration", "yaw_jerk", "total_time",
      }
    """
    if total_time is None:
        raise NotImplementedError(
            "optimize_trajectory phase 6.2 requires total_time; work package 6.3 "
            "must construct control points and select T."
        )

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

    trajectory = _validate_control_points(trajectory)
    # Phase entry requires exactly 3 columns (x, y, yaw).
    if trajectory.shape[1] != 3:
        raise ValueError(
            f"trajectory must be (N, 3), got shape {trajectory.shape}"
        )

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

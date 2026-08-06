import argparse
import json
from pathlib import Path

import numpy as np
import osqp
from box import Box
from PIL import Image
from scipy import sparse
from scipy.interpolate import BSpline


# --------------------------------------------------------------------------
# Work package 6.2: quintic open-uniform clamped B-spline math kernel.
# Work package 6.3: A* reference curve, control-point initialization, and
# initial time selection.
# --------------------------------------------------------------------------

SPLINE_DEGREE = 5
CONTROL_DT = 0.1
_DT_TOL = 1e-9

# 6.3 initialization defaults (plan section 6.3): single location for the
# control-count formula and init knobs. yaw_tangent_weight is intentionally
# not defaulted here (the plan does not define a value); callers must pass it.
TARGET_CONTROL_SPACING_M = 0.5
MIN_CONTROL_POINTS = 8
MAX_CONTROL_POINTS = 64
LAMBDA_INIT = 1.0
GAMMA_INIT = 1.2

_LIMIT_KEYS = (
    "v_max",
    "a_max",
    "j_max",
    "yaw_rate_max",
    "yaw_accel_max",
    "yaw_jerk_max",
)

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
        raise ValueError(f"order must be in [0, {SPLINE_DEGREE}], got {order}")
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


# --------------------------------------------------------------------------
# Work package 6.3: A* reference curve, control points, initial time.
# --------------------------------------------------------------------------


def cumulative_arc_length(path_xy: np.ndarray) -> np.ndarray:
    """Cumulative Euclidean arc length s of a world XY polyline, s[0] = 0."""
    path_xy = np.asarray(path_xy, dtype=float)
    segments = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segments)])


def compute_n_control_points(
    path_length_m: float,
    *,
    target_control_spacing_m: float = TARGET_CONTROL_SPACING_M,
    min_control_points: int = MIN_CONTROL_POINTS,
    max_control_points: int = MAX_CONTROL_POINTS,
) -> int:
    """N_ctrl = clip(ceil(L / spacing) + SPLINE_DEGREE, min, max) (plan 6.3)."""
    if not np.isfinite(path_length_m) or path_length_m <= 0.0:
        raise ValueError(f"path_length_m must be positive and finite, got {path_length_m}")
    n = int(np.ceil(path_length_m / target_control_spacing_m)) + SPLINE_DEGREE
    return int(np.clip(n, min_control_points, max_control_points))


def resample_by_arc_length(
    path_xy: np.ndarray, arc_lengths: np.ndarray, n_points: int
) -> np.ndarray:
    """Resample an XY polyline at n_points evenly spaced arc-length values.

    With the shared cumulative arc-length parameter, per-axis `np.interp` is
    the exact piecewise-linear interpolation of the polyline.
    """
    targets = np.linspace(0.0, float(arc_lengths[-1]), n_points)
    out = np.empty((n_points, 2), dtype=float)
    out[:, 0] = np.interp(targets, arc_lengths, path_xy[:, 0])
    out[:, 1] = np.interp(targets, arc_lengths, path_xy[:, 1])
    return out


def lift_to_reference_branch(yaw: float, reference_yaw: float) -> float:
    """Lift yaw by an integer number of 2*pi turns onto the branch nearest
    `reference_yaw` (plan 6.3 step 2)."""
    return yaw + np.round((reference_yaw - yaw) / (2.0 * np.pi)) * 2.0 * np.pi


def _init_xy_control_points(
    targets: np.ndarray,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    lambda_init: float,
    spacing: float,
) -> np.ndarray:
    """Constrained QP init: target fit + lambda_init * second-difference smoothing.

    Solves the 6.3 initialization QP with OSQP, then pins P0=P1=start and
    P[-2]=P[-1]=goal exactly. All residuals are normalized by `spacing` (the
    plan's target_control_spacing_m), keeping lambda_init dimensionless.
    """
    n = targets.shape[0]
    second_diff = np.zeros((n - 2, n))
    for i in range(n - 2):
        second_diff[i, i] = 1.0
        second_diff[i, i + 1] = -2.0
        second_diff[i, i + 2] = 1.0
    block = np.eye(n) + lambda_init * (second_diff.T @ second_diff)
    P = 2.0 / spacing**2 * sparse.bmat([[block, None], [None, block]], format="csc")
    q = -(2.0 / spacing**2) * np.concatenate([targets[:, 0], targets[:, 1]])

    # Equality constraints: x0=x1=x_start, x[-2]=x[-1]=x_goal, same for y.
    A = np.zeros((8, 2 * n))
    bounds = np.empty(8)
    for i, (var, value) in enumerate(
        [
            (0, start_xy[0]),
            (1, start_xy[0]),
            (n - 2, goal_xy[0]),
            (n - 1, goal_xy[0]),
            (n, start_xy[1]),
            (n + 1, start_xy[1]),
            (2 * n - 2, goal_xy[1]),
            (2 * n - 1, goal_xy[1]),
        ]
    ):
        A[i, var] = 1.0
        bounds[i] = value

    solver = osqp.OSQP()
    solver.setup(P=P, q=q, A=sparse.csc_matrix(A), l=bounds, u=bounds, verbose=False)
    result = solver.solve()
    if result.info.status not in ("solved", "solved inaccurate"):
        raise RuntimeError(f"OSQP control-point init failed: {result.info.status}")
    xy = result.x.reshape(2, n).T
    xy[0] = start_xy
    xy[1] = start_xy
    xy[-2] = goal_xy
    xy[-1] = goal_xy
    return xy


def _init_yaw_control_points(
    n_ctrl: int,
    start_yaw: float,
    goal_yaw: float,
    tangent_weight: float,
    reference_path_xy: np.ndarray,
    reference_yaw: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Yaw init: reference-relative branch lift + constrained smoothing QP.

    The full unwrapped reference yaw is resampled at the yaw control-point
    locations by normalized reference arc length (plan 6.3). The supplied
    start and goal yaw are lifted by integer 2*pi turns onto the branches
    nearest the unwrapped reference endpoints. Free interior yaw variables
    solve
        min w*sum_i (theta_i - tilde_i)^2 + sum_i (theta_{i+1} - 2 theta_i
        + theta_{i-1})^2
    with theta0=theta1=start_yaw and theta[-2]=theta[-1]=goal_yaw exactly.
    """
    ref_arc = cumulative_arc_length(reference_path_xy)
    ref_yaw_u = yaw_unwrap(reference_yaw)
    target = np.interp(
        np.linspace(0.0, 1.0, n_ctrl), ref_arc / ref_arc[-1], ref_yaw_u
    )
    start_u = lift_to_reference_branch(start_yaw, ref_yaw_u[0])
    goal_u = lift_to_reference_branch(goal_yaw, ref_yaw_u[-1])

    theta = np.empty(n_ctrl)
    theta[0] = theta[1] = start_u
    theta[-2] = theta[-1] = goal_u

    free = np.arange(2, n_ctrl - 2)
    fixed = np.array([0, 1, n_ctrl - 2, n_ctrl - 1])
    second_diff = np.zeros((n_ctrl - 2, n_ctrl))
    for i in range(n_ctrl - 2):
        second_diff[i, i] = 1.0
        second_diff[i, i + 1] = -2.0
        second_diff[i, i + 2] = 1.0
    dfree, dfixed = second_diff[:, free], second_diff[:, fixed]
    lhs = tangent_weight * np.eye(free.size) + dfree.T @ dfree
    rhs = tangent_weight * target[free] - dfree.T @ (dfixed @ theta[fixed])
    theta[free] = np.linalg.solve(lhs, rhs)
    return theta, start_u, goal_u


def _validate_limits(limits: dict) -> dict:
    """Coerce and validate the six physical derivative limits from the caller."""
    values = np.array([limits[key] for key in _LIMIT_KEYS], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"limits must be positive and finite for {_LIMIT_KEYS}")
    return {key: float(limits[key]) for key in _LIMIT_KEYS}


def time_policy_bounds(path_length_m: float) -> tuple[float, float]:
    """dt-aligned [T_min, T_max] from the 6.1 1.0/3.0 s/m policy on A* length."""
    # Floating-point ratios near an integer boundary (e.g. 3*0.3/0.1) would
    # otherwise floor/ceil to the wrong control step; nudge by _DT_TOL.
    t_min = np.ceil((path_length_m - _DT_TOL) / CONTROL_DT) * CONTROL_DT
    t_max = np.floor((3.0 * path_length_m + _DT_TOL) / CONTROL_DT) * CONTROL_DT
    return float(t_min), float(t_max)


def estimate_time_components(
    control_points: np.ndarray, limits: dict
) -> dict:
    """Minimum per-order times from derivative control points and real limits.

    Real-time powers: velocity scales 1/T, acceleration 1/T^2, jerk 1/T^3. The
    translational channel uses the Euclidean norm over the xy columns, the yaw
    channel the absolute value over the yaw column; each order reports the
    worst of the two.
    """
    knots = clamped_knots(control_points.shape[0])
    d1, d2, d3 = (
        derivative_control_points(knots, control_points, order)[1]
        for order in (1, 2, 3)
    )
    v_xy = float(np.max(np.linalg.norm(d1[:, :2], axis=1)))
    a_xy = float(np.max(np.linalg.norm(d2[:, :2], axis=1)))
    j_xy = float(np.max(np.linalg.norm(d3[:, :2], axis=1)))
    w1 = float(np.max(np.abs(d1[:, 2])))
    w2 = float(np.max(np.abs(d2[:, 2])))
    w3 = float(np.max(np.abs(d3[:, 2])))
    return {
        "t_v": max(v_xy / limits["v_max"], w1 / limits["yaw_rate_max"]),
        "t_a": max(
            np.sqrt(a_xy / limits["a_max"]), np.sqrt(w2 / limits["yaw_accel_max"])
        ),
        "t_j": max(
            np.cbrt(j_xy / limits["j_max"]), np.cbrt(w3 / limits["yaw_jerk_max"])
        ),
    }


def compute_initial_time(
    control_points: np.ndarray,
    path_length_m: float,
    limits: dict,
    *,
    gamma: float = GAMMA_INIT,
) -> dict:
    """T_init = dt-aligned clip(gamma*max(T_v, T_a, T_j, T_min), T_min, T_max)."""
    components = estimate_time_components(control_points, limits)
    t_min, t_max = time_policy_bounds(path_length_m)
    if t_max < t_min:
        raise ValueError(
            f"degenerate policy time range for path length {path_length_m} m: "
            f"T_min={t_min} > T_max={t_max}"
        )
    candidate = gamma * max(
        components["t_v"], components["t_a"], components["t_j"], t_min
    )
    exceeds = candidate > t_max
    clipped = float(np.clip(candidate, t_min, t_max))
    # Output time lives on the fixed control grid: round up to a control step.
    t_init = int(np.ceil(clipped / CONTROL_DT - 1e-9)) * CONTROL_DT
    return {
        "t_min": t_min,
        "t_max": t_max,
        "t_v": float(components["t_v"]),
        "t_a": float(components["t_a"]),
        "t_j": float(components["t_j"]),
        "candidate": float(candidate),
        "t_init": float(t_init),
        "initial_time_exceeds_policy_max": exceeds,
    }


def _evaluate_spline(control_points: np.ndarray, total_time: float) -> dict:
    """Evaluate control points on the fixed CONTROL_DT grid."""
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
            f"total_time must be at least one control step ({CONTROL_DT} s), got T={T}"
        )
    T = n_steps * CONTROL_DT

    control_points = np.asarray(control_points, dtype=float)
    if control_points.ndim != 2 or control_points.shape[1] != 3:
        raise ValueError(f"control_points must be (N, 3), got shape {control_points.shape}")
    if control_points.shape[0] < SPLINE_DEGREE + 1:
        raise ValueError(
            f"need at least {SPLINE_DEGREE + 1} control points, "
            f"got {control_points.shape[0]}"
        )
    if not np.all(np.isfinite(control_points)):
        raise ValueError("control_points must contain only finite values")

    t = np.arange(n_steps + 1, dtype=float) * CONTROL_DT
    u = t / T
    ev = eval_derivatives(control_points, T, u)
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


def optimize_trajectory(
    astar_path_xy: np.ndarray,
    start_pose,
    goal_pose,
    limits: dict,
    *,
    reference_path_xy: np.ndarray,
    reference_yaw: np.ndarray,
    yaw_tangent_weight: float,
    target_control_spacing_m: float = TARGET_CONTROL_SPACING_M,
    min_control_points: int = MIN_CONTROL_POINTS,
    max_control_points: int = MAX_CONTROL_POINTS,
    lambda_init: float = LAMBDA_INIT,
    gamma: float = GAMMA_INIT,
) -> dict:
    """6.3 initialization + 6.2 evaluation on the fixed CONTROL_DT grid.

    `astar_path_xy` is the raw A* world polyline (not the smoothed episode
    reference path); it drives path length, N_ctrl, and the XY control-point
    QP. `reference_path_xy`/`reference_yaw` are the smoothed episode points
    and their wrapped yaw; the full unwrapped reference yaw is resampled at
    the yaw control-point locations by normalized reference arc length and
    used as the soft yaw reference (plan 6.3). `start_pose`/`goal_pose` are
    (x, y, yaw). The stationary-to-stationary contract P0=P1=start,
    P[-2]=P[-1]=goal holds for all channels by construction. Physical
    derivative limits are required from the caller; no versioned config
    exists yet. `yaw_tangent_weight` is the nonnegative relative penalty of
    the yaw reference against unit-weight second-difference smoothing; it is
    not a [0, 1] blend coefficient and may exceed 1. It has no documented
    default, so it is a required keyword argument.
    """
    astar_path_xy = np.asarray(astar_path_xy, dtype=float)
    if astar_path_xy.ndim != 2 or astar_path_xy.shape[1] != 2:
        raise ValueError(
            f"astar_path_xy must be an [M, 2] polyline, got shape {astar_path_xy.shape}"
        )
    if astar_path_xy.shape[0] < 2:
        raise ValueError("astar_path_xy must contain at least two points")
    if not np.all(np.isfinite(astar_path_xy)):
        raise ValueError("astar_path_xy must contain only finite values")
    if np.any(np.all(np.diff(astar_path_xy, axis=0) == 0.0, axis=1)):
        raise ValueError(
            "astar_path_xy must not contain consecutive duplicate points "
            "(arc-length interpolation would be ambiguous)"
        )

    reference_path_xy = np.asarray(reference_path_xy, dtype=float)
    reference_yaw = np.asarray(reference_yaw, dtype=float)
    if (
        reference_path_xy.ndim != 2
        or reference_path_xy.shape[1] != 2
        or reference_path_xy.shape[0] < 2
        or reference_yaw.ndim != 1
        or reference_yaw.shape[0] != reference_path_xy.shape[0]
    ):
        raise ValueError(
            "reference_path_xy must be an [M, 2] polyline and reference_yaw an "
            "[M] yaw sequence over the same smoothed reference points, got "
            f"shape {reference_path_xy.shape} and {reference_yaw.shape}"
        )
    if (
        not np.all(np.isfinite(reference_path_xy))
        or not np.all(np.isfinite(reference_yaw))
        or np.any(np.all(np.diff(reference_path_xy, axis=0) == 0.0, axis=1))
    ):
        raise ValueError(
            "reference_path_xy and reference_yaw must contain only finite "
            "values, and reference_path_xy no consecutive duplicate points"
        )
    if not np.isfinite(yaw_tangent_weight) or yaw_tangent_weight < 0.0:
        raise ValueError(
            f"yaw_tangent_weight must be nonnegative and finite, got "
            f"{yaw_tangent_weight}"
        )

    start = np.asarray(start_pose, dtype=float)
    goal = np.asarray(goal_pose, dtype=float)
    if (
        start.shape != (3,)
        or goal.shape != (3,)
        or not np.all(np.isfinite(start))
        or not np.all(np.isfinite(goal))
    ):
        raise ValueError("start_pose and goal_pose must be finite (x, y, yaw)")
    limits = _validate_limits(limits)

    arc_lengths = cumulative_arc_length(astar_path_xy)
    path_length_m = float(arc_lengths[-1])
    n_ctrl = compute_n_control_points(
        path_length_m,
        target_control_spacing_m=target_control_spacing_m,
        min_control_points=min_control_points,
        max_control_points=max_control_points,
    )
    targets = resample_by_arc_length(astar_path_xy, arc_lengths, n_ctrl)
    xy = _init_xy_control_points(
        targets, start[:2], goal[:2], lambda_init, target_control_spacing_m
    )
    theta, start_yaw_u, goal_yaw_u = _init_yaw_control_points(
        n_ctrl, start[2], goal[2], yaw_tangent_weight, reference_path_xy,
        reference_yaw,
    )
    control_points = np.column_stack([xy, theta])

    init_time = compute_initial_time(
        control_points, path_length_m, limits, gamma=gamma
    )
    output = _evaluate_spline(control_points, init_time["t_init"])
    output["control_points"] = control_points
    output["initialization"] = {
        "n_control_points": n_ctrl,
        "path_length_m": path_length_m,
        "start_yaw_unwrapped": float(start_yaw_u),
        "goal_yaw_unwrapped": float(goal_yaw_u),
        "yaw_tangent_weight": float(yaw_tangent_weight),
        "time": {
            key: init_time[key]
            for key in ("t_min", "t_max", "t_v", "t_a", "t_j", "candidate", "t_init")
        },
        "initial_time_exceeds_policy_max": bool(
            init_time["initial_time_exceeds_policy_max"]
        ),
    }
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--scene-id", type=str, required=True)
    return parser.parse_args()


def main(args):
    # WP 6.3 boundary: the generator already stores `astar_path_pixels` in
    # each episode NPZ and MapTransform fields in the manifest; batch loading
    # and output orchestration belong to later work packages and remain
    # unimplemented.
    raise NotImplementedError(
        "batch loading/output orchestration for optimize_trajectory is "
        "deferred to later work packages"
    )


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


if __name__ == "__main__":
    main(parse_args())

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import osqp
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.optimize import minimize

from sage3d.utils import MapTransform


# --------------------------------------------------------------------------
# Work package 6.2: quintic open-uniform clamped B-spline math kernel.
# Work package 6.3: A* reference curve, control-point initialization, and
# initial time selection.
# Work package 6.4: minimal SLSQP joint optimizer over internal control
# points and total time.
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
    second_diff = np.diff(np.eye(n), n=2, axis=0)
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
    second_diff = np.diff(np.eye(n_ctrl), n=2, axis=0)
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


def initialize_trajectory(
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


# --------------------------------------------------------------------------
# Work package 6.4: minimal joint optimizer (plan section 6.4).
# --------------------------------------------------------------------------

# 5-point Gauss-Lobatto nodes on [-1, 1] (span endpoints included). Used only
# as fixed collocation points for derivative/collision constraints, so their
# quadrature weights are not needed.
_LOBATTO5_NODES = np.array(
    [-1.0, -np.sqrt(3.0 / 7.0), 0.0, np.sqrt(3.0 / 7.0), 1.0]
)


def canonical_output_time(t_continuous: float) -> float:
    """T_output = ceil((T_continuous - tol) / CONTROL_DT) * CONTROL_DT.

    The tolerance is applied in seconds (before the division by dt), so a
    T_continuous within tol below a grid point snaps to that grid point.
    """
    if not np.isfinite(t_continuous) or t_continuous <= 0.0:
        raise ValueError(
            f"t_continuous must be positive and finite, got {t_continuous}"
        )
    return int(np.ceil((t_continuous - _DT_TOL) / CONTROL_DT)) * CONTROL_DT


def continuous_world_to_pixel(transform, x, y) -> tuple[np.ndarray, np.ndarray]:
    """Continuous [row, col] from world [x, y] (m), consistent with
    MapTransform's X reversal (columns run opposite world +X)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    col = transform.width - 0.5 - (x - transform.lower_x) / transform.scale
    row = (y - transform.lower_y) / transform.scale - 0.5
    return row, col


def bilinear_clearance(
    clearance_m: np.ndarray, row: np.ndarray, col: np.ndarray
) -> np.ndarray:
    """Bilinear clearance at continuous (row, col); any of the four neighbor
    cells outside the array makes the candidate infeasible (-inf)."""
    r0 = np.floor(row).astype(int)
    c0 = np.floor(col).astype(int)
    r1, c1 = r0 + 1, c0 + 1
    height, width = clearance_m.shape
    inside = (r0 >= 0) & (r1 < height) & (c0 >= 0) & (c1 < width)
    out = np.full(np.shape(row), -np.inf)
    if not np.any(inside):
        return out
    tr = row[inside] - r0[inside]
    tc = col[inside] - c0[inside]
    out[inside] = (
        (1.0 - tr) * (1.0 - tc) * clearance_m[r0[inside], c0[inside]]
        + (1.0 - tr) * tc * clearance_m[r0[inside], c1[inside]]
        + tr * (1.0 - tc) * clearance_m[r1[inside], c0[inside]]
        + tr * tc * clearance_m[r1[inside], c1[inside]]
    )
    return out


def _span_nodes(knots: np.ndarray, nodes: np.ndarray, weights=None):
    """Map [-1, 1] nodes onto every nonzero knot span.

    Returns (u, w): concatenated node locations in [0, 1] and per-span
    half-width-scaled quadrature weights (None when `weights` is None).
    """
    us = []
    ws = [] if weights is not None else None
    for a, b in zip(knots[:-1], knots[1:]):
        if b <= a:
            continue
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        us.append(mid + half * nodes)
        if ws is not None:
            ws.append(half * weights)
    u = np.concatenate(us)
    return u, (np.concatenate(ws) if ws is not None else None)


def _basis_matrices(knots: np.ndarray, n_ctrl: int, u: np.ndarray):
    """(B0, B1, B2, B3): basis matrices (len(u), n_ctrl) for orders 0..3 at u."""
    spline = BSpline(knots, np.eye(n_ctrl), SPLINE_DEGREE)
    return (
        spline(u),
        spline.derivative(1)(u),
        spline.derivative(2)(u),
        spline.derivative(3)(u),
    )


def _validate_configs(objective_config, trust_config, solver_config) -> None:
    """6.4 config dicts are caller-supplied; the plan defines no defaults."""
    weight_names = ("w_ref", "w_jerk_xy", "w_jerk_yaw", "w_yaw_rate", "w_time")
    scale_names = (
        "reference_distance_scale_m",
        "jerk_xy_scale",
        "jerk_yaw_scale",
        "yaw_rate_scale",
        "time_scale_s",
    )
    for key in (*weight_names, *scale_names):
        if key not in objective_config:
            raise ValueError(f"objective_config must define {key}")
    weights = np.array([objective_config[k] for k in weight_names], dtype=float)
    scales = np.array([objective_config[k] for k in scale_names], dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("objective weights must be nonnegative and finite")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("objective scales must be positive and finite")

    trust_names = ("trust_xy_resolution_cells", "trust_xy_max_m", "trust_yaw_rad")
    for key in trust_names:
        if key not in trust_config:
            raise ValueError(f"trust_config must define {key}")
    trust = np.array([trust_config[k] for k in trust_names], dtype=float)
    if not np.all(np.isfinite(trust)) or np.any(trust <= 0.0):
        raise ValueError("trust_config radii must be positive and finite")

    solver_positive = ("ftol", "episode_timeout_s", "constraint_tolerance",
                       "clearance_scale_m")
    for key in (*solver_positive, "maxiter", "final_objective_tolerance"):
        if key not in solver_config:
            raise ValueError(f"solver_config must define {key}")
    pos = np.array([solver_config[k] for k in solver_positive], dtype=float)
    if not np.all(np.isfinite(pos)) or np.any(pos <= 0.0):
        raise ValueError(
            "solver_config ftol/episode_timeout_s/constraint_tolerance/"
            "clearance_scale_m must be positive and finite"
        )
    if (
        not isinstance(solver_config["maxiter"], int)
        or solver_config["maxiter"] < 1
    ):
        raise ValueError(
            f"solver_config maxiter must be a positive integer, "
            f"got {solver_config['maxiter']}"
        )
    fot = solver_config["final_objective_tolerance"]
    if not np.isfinite(fot) or fot < 0.0:
        raise ValueError(
            "solver_config final_objective_tolerance must be nonnegative "
            "and finite"
        )


def _build_nlp(
    control_points0,
    t_min,
    t_max,
    limits,
    reference_xy,
    clearance_m,
    map_transform,
    required_clearance_m,
    objective_config,
    trust_config,
    solver_config,
) -> dict:
    """Shared objective/constraint evaluator for the 6.4 NLP.

    Variables x = [z_xy (2 * n_free), z_yaw (n_free), tau] with
        P_xy = P0_xy + r_xy * z_xy,
        theta = theta0 + trust_yaw_rad * z_yaw,
        T = t_min + tau * (t_max - t_min),
    r_xy = min(trust_xy_resolution_cells * map scale, trust_xy_max_m).
    Endpoint controls (0, 1, n-2, n-1) stay fixed at the initialization.
    Objective quadrature: fixed 3-point Gauss-Legendre per nonzero span.
    Derivative/collision constraints: 5-point Gauss-Lobatto per span.
    The same `evaluate` backs the objective, the constraints, and the audit.
    """
    n_ctrl = control_points0.shape[0]
    knots = clamped_knots(n_ctrl)
    free = np.arange(2, n_ctrl - 2)
    fixed = np.array([0, 1, n_ctrl - 2, n_ctrl - 1])
    p0_free = control_points0[free]
    p0_fixed = control_points0[fixed]
    n_free = free.size

    u_gl, w_gl = _span_nodes(knots, _GL3_NODES, _GL3_WEIGHTS)
    u_lb, _ = _span_nodes(knots, _LOBATTO5_NODES)
    b_gl = _basis_matrices(knots, n_ctrl, u_gl)
    b_lb = _basis_matrices(knots, n_ctrl, u_lb)

    def split(b):
        return b[:, free], b[:, fixed]

    b0_gl_f, b0_gl_x = split(b_gl[0])
    b1_gl_f, b1_gl_x = split(b_gl[1])
    b3_gl_f, b3_gl_x = split(b_gl[3])
    b0_lb_f, b0_lb_x = split(b_lb[0])
    b1_lb_f, b1_lb_x = split(b_lb[1])
    b2_lb_f, b2_lb_x = split(b_lb[2])
    b3_lb_f, b3_lb_x = split(b_lb[3])
    off0_gl = b0_gl_x @ p0_fixed
    off1_gl = b1_gl_x @ p0_fixed
    off3_gl = b3_gl_x @ p0_fixed
    off0_lb = b0_lb_x @ p0_fixed
    off1_lb = b1_lb_x @ p0_fixed
    off2_lb = b2_lb_x @ p0_fixed
    off3_lb = b3_lb_x @ p0_fixed

    # Reference curve resampled at the GL nodes by normalized arc length, so
    # J_ref integrates over normalized path progress (plan 6.4).
    ref_arc = cumulative_arc_length(reference_xy)
    ref_norm = ref_arc / ref_arc[-1]
    ref_gl = np.column_stack(
        [
            np.interp(u_gl, ref_norm, reference_xy[:, 0]),
            np.interp(u_gl, ref_norm, reference_xy[:, 1]),
        ]
    )

    r_xy = min(
        trust_config["trust_xy_resolution_cells"] * map_transform.scale,
        trust_config["trust_xy_max_m"],
    )
    yaw_rad = trust_config["trust_yaw_rad"]
    t_span = t_max - t_min
    oc = objective_config
    sc = solver_config
    weights = [oc["w_ref"], oc["w_jerk_xy"], oc["w_jerk_yaw"],
               oc["w_yaw_rate"], oc["w_time"]]
    s_ref, s_jxy, s_jyaw, s_wr, s_time = (
        oc["reference_distance_scale_m"],
        oc["jerk_xy_scale"],
        oc["jerk_yaw_scale"],
        oc["yaw_rate_scale"],
        oc["time_scale_s"],
    )
    c_scale = sc["clearance_scale_m"]
    required = required_clearance_m

    def unpack(x):
        x = np.asarray(x, dtype=float)
        z_xy = x[: 2 * n_free].reshape(n_free, 2)
        z_yaw = x[2 * n_free : 3 * n_free]
        tau = float(x[-1])
        T = t_min + tau * t_span
        p_free = p0_free + np.column_stack([r_xy * z_xy, yaw_rad * z_yaw])
        return z_xy, z_yaw, tau, T, p_free

    def pack(p_free, T):
        z_xy = (p_free[:, :2] - p0_free[:, :2]) / r_xy
        z_yaw = (p_free[:, 2] - p0_free[:, 2]) / yaw_rad
        tau = (T - t_min) / t_span if t_span > 0.0 else 0.0
        return np.concatenate([z_xy.ravel(), z_yaw, [tau]])

    def evaluate(x):
        z_xy, z_yaw, tau, T, p_free = unpack(x)
        v0_gl = b0_gl_f @ p_free + off0_gl
        v1_gl = b1_gl_f @ p_free + off1_gl
        v3_gl = b3_gl_f @ p_free + off3_gl
        ref_raw = float(w_gl @ np.sum((v0_gl[:, :2] - ref_gl) ** 2, axis=1))
        jerk_xy_raw = float(w_gl @ np.sum(v3_gl[:, :2] ** 2, axis=1)) / T**5
        jerk_yaw_raw = float(w_gl @ (v3_gl[:, 2] ** 2)) / T**5
        yaw_rate_raw = float(w_gl @ (v1_gl[:, 2] ** 2)) / T**2
        raw = {
            "ref": ref_raw,
            "jerk_xy": jerk_xy_raw,
            "jerk_yaw": jerk_yaw_raw,
            "yaw_rate": yaw_rate_raw,
            "time": float(T),
        }
        normalized = {
            "ref": ref_raw / s_ref**2,
            "jerk_xy": jerk_xy_raw / (s_jxy**2 * s_time),
            "jerk_yaw": jerk_yaw_raw / (s_jyaw**2 * s_time),
            "yaw_rate": yaw_rate_raw / s_wr**2,
            "time": float(T) / s_time,
        }
        weighted = {
            key: weights[i] * normalized[key]
            for i, key in enumerate(normalized)
        }
        objective = {
            "raw": raw,
            "normalized": normalized,
            "weighted": weighted,
            "total": float(sum(weighted.values())),
        }

        v0_lb = b0_lb_f @ p_free + off0_lb
        v1_lb = b1_lb_f @ p_free + off1_lb
        v2_lb = b2_lb_f @ p_free + off2_lb
        v3_lb = b3_lb_f @ p_free + off3_lb
        row, col = continuous_world_to_pixel(
            map_transform, v0_lb[:, 0], v0_lb[:, 1]
        )
        groups = {
            "velocity_xy": 1.0
            - np.sum(v1_lb[:, :2] ** 2, axis=1) / T**2 / limits["v_max"] ** 2,
            "accel_xy": 1.0
            - np.sum(v2_lb[:, :2] ** 2, axis=1) / T**4 / limits["a_max"] ** 2,
            "jerk_xy": 1.0
            - np.sum(v3_lb[:, :2] ** 2, axis=1) / T**6 / limits["j_max"] ** 2,
            "yaw_rate": 1.0
            - v1_lb[:, 2] ** 2 / T**2 / limits["yaw_rate_max"] ** 2,
            "yaw_accel": 1.0
            - v2_lb[:, 2] ** 2 / T**4 / limits["yaw_accel_max"] ** 2,
            "yaw_jerk": 1.0
            - v3_lb[:, 2] ** 2 / T**6 / limits["yaw_jerk_max"] ** 2,
            "clearance": (
                bilinear_clearance(clearance_m, row, col) - required
            )
            / c_scale,
            "trust_xy_disk": 1.0 - np.sum(z_xy**2, axis=1),
            "yaw_bound_low": 1.0 + z_yaw,
            "yaw_bound_high": 1.0 - z_yaw,
            "tau_low": np.array([tau]),
            "tau_high": np.array([1.0 - tau]),
        }
        margins = np.concatenate(list(groups.values()))
        p_full = control_points0.copy()
        p_full[free] = p_free
        return {
            "x": np.asarray(x, dtype=float),
            "T_continuous": float(T),
            "control_points": p_full,
            "objective": objective,
            "margins": margins,
            "margin_groups": groups,
        }

    bounds = (
        [(None, None)] * (2 * n_free)
        + [(-1.0, 1.0)] * n_free
        + [(0.0, 1.0)]
    )
    return {
        "n_free": n_free,
        "r_xy": float(r_xy),
        "yaw_rad": float(yaw_rad),
        "pack": pack,
        "unpack": unpack,
        "evaluate": evaluate,
        "fun": lambda x: evaluate(x)["objective"]["total"],
        "ineq": lambda x: evaluate(x)["margins"],
        "bounds": bounds,
    }


def _audit_candidate(init_eval, final_eval, solver_config) -> dict:
    """Independent acceptance (never `result.success` alone): finiteness of
    the objective, decision variables and control points, every recomputed
    margin (derivative, clearance, trust disk, yaw bounds, tau bounds) within
    tolerance, and no objective increase beyond tolerance when the
    initialization is constraint-feasible (plan 6.4 success criteria)."""
    ctol = solver_config["constraint_tolerance"]
    final_margins = np.asarray(final_eval["margins"], dtype=float)
    init_margins = np.asarray(init_eval["margins"], dtype=float)
    init_feasible = bool(
        np.all(np.isfinite(init_margins)) and init_margins.min() >= -ctol
    )
    finite = bool(
        np.all(np.isfinite(final_margins))
        and np.all(np.isfinite(final_eval["x"]))
        and np.all(np.isfinite(final_eval["control_points"]))
        and np.isfinite(final_eval["objective"]["total"])
        and all(
            np.isfinite(v)
            for v in final_eval["objective"]["normalized"].values()
        )
    )
    margins_ok = bool(finite and final_margins.min() >= -ctol)
    monotonic_ok = not (
        init_feasible
        and final_eval["objective"]["total"]
        > init_eval["objective"]["total"]
        + solver_config["final_objective_tolerance"]
    )
    return {
        "initial_feasible": init_feasible,
        "finiteness_ok": finite,
        "margins_ok": margins_ok,
        "monotonic_ok": monotonic_ok,
        "final_margins_min": float(np.min(final_margins)) if finite else -np.inf,
        "success": bool(margins_ok and monotonic_ok),
    }


def optimize_trajectory(
    astar_path_xy,
    start_pose,
    goal_pose,
    limits,
    *,
    reference_path_xy,
    reference_yaw,
    yaw_tangent_weight,
    clearance_m,
    map_transform,
    required_clearance_m,
    objective_config,
    trust_config,
    solver_config,
    target_control_spacing_m=TARGET_CONTROL_SPACING_M,
    min_control_points=MIN_CONTROL_POINTS,
    max_control_points=MAX_CONTROL_POINTS,
    lambda_init=LAMBDA_INIT,
    gamma=GAMMA_INIT,
) -> dict:
    """6.4 minimal joint optimizer (plan section 6.4).

    Runs the 6.3 initialization first (see initialize_trajectory), then
    minimizes the weighted normalized objective on fixed 3-point
    Gauss-Legendre nodes per knot span with SciPy SLSQP over normalized
    internal XY/yaw deltas and continuous time tau in [0, 1], subject to
    normalized squared-margin derivative limits, bilinear map clearance,
    per-internal-point Euclidean XY disk trust region, per-point yaw bound,
    and the policy time range mapped to tau. Endpoint controls stay pinned to
    the initialization. All physical scales, weights, trust radii and solver
    knobs come from the caller's explicit config dicts.

    The result is a solver candidate for first-stage validation (plan 6.5).
    Per plan 6.1/7, T_output is the authoritative duration: the reported
    objective, constraints, monotonic comparison against the aligned initial
    trajectory and the saved fixed-dt candidate are all re-evaluated on
    [0, T_output]; the continuous NLP objective stays visible separately as
    solver diagnostics. The candidate is rejected if canonicalization pushes
    T_output beyond T_max. Success additionally requires the independent
    audit (finiteness, recomputed margins including yaw/tau bounds, no
    objective increase when the initialization is feasible), and the
    candidate is not claimed to be executable or independently validated.
    """
    limits = _validate_limits(limits)
    _validate_configs(objective_config, trust_config, solver_config)
    clearance_m = np.asarray(clearance_m, dtype=float)
    if (
        clearance_m.ndim != 2
        or not np.all(np.isfinite(clearance_m))
        or clearance_m.shape != (map_transform.height, map_transform.width)
    ):
        raise ValueError(
            "clearance_m must be a finite 2D array matching MapTransform "
            f"shape {map_transform.height}x{map_transform.width}, got "
            f"{clearance_m.shape}"
        )
    if not np.isfinite(required_clearance_m) or required_clearance_m < 0.0:
        raise ValueError(
            f"required_clearance_m must be nonnegative and finite, got "
            f"{required_clearance_m}"
        )

    init = initialize_trajectory(
        astar_path_xy,
        start_pose,
        goal_pose,
        limits,
        reference_path_xy=reference_path_xy,
        reference_yaw=reference_yaw,
        yaw_tangent_weight=yaw_tangent_weight,
        target_control_spacing_m=target_control_spacing_m,
        min_control_points=min_control_points,
        max_control_points=max_control_points,
        lambda_init=lambda_init,
        gamma=gamma,
    )
    ctrl0 = init["control_points"]
    t_init = init["initialization"]["time"]["t_init"]
    t_min = init["initialization"]["time"]["t_min"]
    t_max = init["initialization"]["time"]["t_max"]
    nlp = _build_nlp(
        ctrl0,
        t_min,
        t_max,
        limits,
        reference_path_xy,
        clearance_m,
        map_transform,
        required_clearance_m,
        objective_config,
        trust_config,
        solver_config,
    )
    x0 = nlp["pack"](ctrl0[2:-2], t_init)
    init_eval = nlp["evaluate"](x0)

    solve_start = time.monotonic()
    timeout_s = solver_config["episode_timeout_s"]
    last_x = {"x": x0}

    def timeout_callback(xk):
        last_x["x"] = np.asarray(xk, dtype=float)
        if time.monotonic() - solve_start > timeout_s:
            raise TimeoutError(
                f"SLSQP exceeded episode_timeout_s={timeout_s}"
            )

    status = "solver_failed"
    res = None
    try:
        res = minimize(
            nlp["fun"],
            x0,
            method="SLSQP",
            bounds=nlp["bounds"],
            constraints=[{"type": "ineq", "fun": nlp["ineq"]}],
            options={
                "ftol": solver_config["ftol"],
                "maxiter": solver_config["maxiter"],
                "disp": False,
            },
            callback=timeout_callback,
        )
        status = "success" if bool(res.success) else "solver_failed"
    except TimeoutError:
        status = "timeout"
    elapsed_s = time.monotonic() - solve_start

    final_x = np.asarray(res.x, dtype=float) if res is not None else last_x["x"]
    continuous_eval = nlp["evaluate"](final_x)
    t_continuous = continuous_eval["T_continuous"]
    if np.isfinite(t_continuous):
        t_output = canonical_output_time(t_continuous)
        # Plan 6.1/7: T_output is the authoritative duration; objective and
        # constraints are re-evaluated on the T_output re-parameterization.
        aligned_eval = nlp["evaluate"](
            nlp["pack"](continuous_eval["control_points"][2:-2], t_output)
        )
    else:
        t_output = float("nan")
        aligned_eval = continuous_eval
    audit = _audit_candidate(init_eval, aligned_eval, solver_config)
    t_output_in_policy = np.isfinite(t_output) and t_output <= t_max + _DT_TOL
    if status == "success" and not (audit["success"] and t_output_in_policy):
        status = "audit_failed"
    success = bool(status == "success")

    candidate = (
        _evaluate_spline(continuous_eval["control_points"], t_output)
        if np.isfinite(t_output)
        else None
    )
    return {
        "success": success,
        "status": status,
        "objective_initial": init_eval["objective"],
        "objective": aligned_eval["objective"],
        "objective_continuous": continuous_eval["objective"],
        "constraint_diagnostics": {
            "initial_feasible": audit["initial_feasible"],
            "finiteness_ok": audit["finiteness_ok"],
            "margins_ok": audit["margins_ok"],
            "monotonic_ok": audit["monotonic_ok"],
            "final_margins_min": audit["final_margins_min"],
            "final_margin_groups_min": {
                name: float(vals.min())
                for name, vals in aligned_eval["margin_groups"].items()
            },
            "t_output_within_policy": bool(t_output_in_policy),
        },
        "solver_metadata": {
            "solver": "SLSQP",
            "ftol": float(solver_config["ftol"]),
            "maxiter": int(solver_config["maxiter"]),
            "episode_timeout_s": float(solver_config["episode_timeout_s"]),
            "result_success": bool(res.success) if res is not None else None,
            "message": (
                str(res.message)
                if res is not None
                else ("timeout" if status == "timeout" else "not run")
            ),
            "nit": int(res.nit) if res is not None else None,
            "nfev": int(res.nfev) if res is not None else None,
            "elapsed_s": float(elapsed_s),
        },
        "control_points": continuous_eval["control_points"],
        "T_continuous": float(t_continuous),
        "T_output": float(t_output),
        "candidate": candidate,
    }


# --------------------------------------------------------------------------
# Post-6.4 single-episode candidate adapter: minimal CLI and candidate
# output. Batch iteration, resume, independent validation (plan 6.5),
# mecanum diagnostics and production publication are deliberately out of
# scope (plan sections 8.3 and 9).
# --------------------------------------------------------------------------

OPTIMIZATION_INPUT_SCHEMA_VERSION = "vln_data_prep.trajectory_optimization_input.v1"
OPTIMIZATION_CANDIDATE_SCHEMA_VERSION = (
    "vln_data_prep.trajectory_optimization_candidate.v1"
)

_INIT_CONFIG_KEYS = (
    "target_control_spacing_m",
    "min_control_points",
    "max_control_points",
    "lambda_init",
    "gamma",
)


def _load_scene_inputs(scene_root, scene_id, episode_index):
    """Validate and load the single-episode optimization inputs (plan 8.2).

    Any contract violation exits nonzero before any output is staged.
    """
    scene_dir = Path(scene_root) / scene_id
    manifest_path = scene_dir / "trajectories" / "trajectory_manifest.json"
    episode_path = scene_dir / "trajectories" / f"episode_{episode_index:06d}.npz"
    esdf_path = scene_dir / "map" / "esdf.npy"
    if not manifest_path.is_file():
        raise SystemExit(f"trajectory manifest not found: {manifest_path}")
    if not episode_path.is_file():
        raise SystemExit(f"episode NPZ not found: {episode_path}")
    if not esdf_path.is_file():
        raise SystemExit(f"esdf not found: {esdf_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("optimization_input_schema_version")
        != OPTIMIZATION_INPUT_SCHEMA_VERSION
    ):
        raise SystemExit(
            "unsupported optimization_input_schema_version: "
            f"{manifest.get('optimization_input_schema_version')}"
        )
    if manifest.get("scene_id") != scene_id:
        raise SystemExit(
            f"manifest scene_id {manifest.get('scene_id')!r} does not match "
            f"--scene-id {scene_id!r}"
        )
    if not any(
        ep.get("episode_index") == episode_index
        for ep in manifest.get("episodes", [])
    ):
        raise SystemExit(
            f"manifest has no episode record with episode_index {episode_index}"
        )

    map_info = manifest.get("map", {})
    try:
        height, width = map_info["shape"]
        transform = MapTransform(
            height=height,
            width=width,
            scale=float(map_info["scale_m_per_pixel"]),
            lower_x=float(map_info["lower_x"]),
            lower_y=float(map_info["lower_y"]),
        )
        required_clearance_m = float(map_info["required_path_clearance_m"])
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(
            "manifest map must define shape, scale_m_per_pixel, lower_x, "
            "lower_y and required_path_clearance_m"
        ) from error
    if map_info.get("pixel_coordinate_order") != "row_col":
        raise SystemExit(
            "unsupported pixel_coordinate_order: "
            f"{map_info.get('pixel_coordinate_order')}"
        )
    if map_info.get("pixel_to_world_convention") != "sage3d_map_transform_v1":
        raise SystemExit(
            "unsupported pixel_to_world_convention: "
            f"{map_info.get('pixel_to_world_convention')}"
        )

    clearance_m = np.load(esdf_path)
    if clearance_m.shape != (height, width):
        raise SystemExit(
            f"esdf shape {clearance_m.shape} does not match map shape "
            f"{(height, width)}"
        )

    with np.load(episode_path) as data:
        missing = [
            key
            for key in ("points", "yaw", "astar_path_pixels")
            if key not in data.files
        ]
        if missing:
            raise SystemExit(
                f"episode NPZ {episode_path} missing required key(s): "
                f"{', '.join(missing)}"
            )
        points = np.asarray(data["points"], dtype=float)
        yaw = np.asarray(data["yaw"], dtype=float)
        astar_path_pixels = np.asarray(data["astar_path_pixels"])
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise SystemExit("episode points must be an [N, 2] polyline")
    if yaw.ndim != 1 or yaw.shape[0] != points.shape[0]:
        raise SystemExit("episode yaw must be a length-N sequence")
    if (
        astar_path_pixels.ndim != 2
        or astar_path_pixels.shape[1] != 2
        or not np.issubdtype(astar_path_pixels.dtype, np.integer)
    ):
        raise SystemExit(
            "episode astar_path_pixels must be an [M, 2] integer array"
        )
    astar_path_xy = np.asarray(
        [
            transform.pixel_to_world(int(row), int(col))
            for row, col in astar_path_pixels
        ],
        dtype=float,
    )
    return {
        "manifest_path": manifest_path,
        "episode_path": episode_path,
        "esdf_path": esdf_path,
        "points": points,
        "yaw": yaw,
        "astar_path_xy": astar_path_xy,
        "clearance_m": clearance_m,
        "transform": transform,
        "required_clearance_m": required_clearance_m,
    }


def _candidate_npz_arrays(candidate):
    """Fixed first-stage NPZ fields (plan 8.3 table), all float64."""
    yaw_wrapped = np.asarray(candidate["yaw_wrapped"], dtype=np.float64)
    return {
        "time_s": np.asarray(candidate["time"], dtype=np.float64),
        "pose_world": np.column_stack(
            [
                np.asarray(candidate["position_world"], dtype=np.float64),
                yaw_wrapped,
            ]
        ),
        "yaw_unwrapped_rad": np.asarray(
            candidate["yaw_unwrapped"], dtype=np.float64
        ),
        "velocity_world_mps": np.asarray(
            candidate["velocity_world"], dtype=np.float64
        ),
        "yaw_rate_radps": np.asarray(candidate["yaw_rate"], dtype=np.float64),
        "acceleration_world_mps2": np.asarray(
            candidate["acceleration_world"], dtype=np.float64
        ),
        "yaw_acceleration_radps2": np.asarray(
            candidate["yaw_acceleration"], dtype=np.float64
        ),
        "jerk_world_mps3": np.asarray(candidate["jerk_world"], dtype=np.float64),
        "yaw_jerk_radps3": np.asarray(candidate["yaw_jerk"], dtype=np.float64),
    }


def _publish_output(output_dir, episode_index, arrays, metadata):
    """Write into a unique sibling staging directory, then atomically rename.

    tempfile.mkdtemp guarantees the staging directory is unique and owned by
    this invocation; on failure only that directory is removed, so the final
    output directory is never partial and unrelated directories are left
    untouched.
    """
    npz_name = f"episode_{episode_index:06d}.npz"
    staging = Path(tempfile.mkdtemp(dir=output_dir.parent))
    try:
        np.savez_compressed(staging / npz_name, **arrays)
        (staging / "candidate_metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        if output_dir.exists():
            raise SystemExit(f"output-dir already exists: {output_dir}")
        staging.rename(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-root", type=Path, required=True,
        help="parent directory containing scene ID directories",
    )
    parser.add_argument("--scene-id", type=str, required=True)
    parser.add_argument(
        "--episode-index", type=int, required=True,
        help="nonnegative episode index within the scene",
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="JSON with top-level limits/objective/trust/solver/"
        "yaw_tangent_weight",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="must not already exist",
    )
    return parser.parse_args()


def main(args):
    """Single-episode CLI adapter: validate inputs, optimize, publish."""
    if args.episode_index < 0:
        raise SystemExit("--episode-index must be nonnegative")
    if args.output_dir.exists():
        raise SystemExit(f"output-dir already exists: {args.output_dir}")

    inputs = _load_scene_inputs(args.scene_root, args.scene_id, args.episode_index)
    with args.config.open("r", encoding="utf-8") as file:
        config = json.load(file)
    for key in (
        "limits",
        "objective",
        "trust",
        "solver",
        "yaw_tangent_weight",
        "initialization",
    ):
        if key not in config:
            raise SystemExit(f"config must define top-level {key}")

    init_cfg = config["initialization"]
    if not isinstance(init_cfg, dict):
        raise SystemExit("config initialization must be an object")
    missing = set(_INIT_CONFIG_KEYS) - set(init_cfg)
    unknown = set(init_cfg) - set(_INIT_CONFIG_KEYS)
    if missing or unknown:
        raise SystemExit(
            "config initialization keys must be exactly "
            f"{', '.join(_INIT_CONFIG_KEYS)}"
        )

    points = inputs["points"]
    yaw = inputs["yaw"]
    # Boundary poses come from the smoothed episode points/yaw sequences, not
    # from start_position/goal_position third components (plan 8.2).
    start_pose = (float(points[0, 0]), float(points[0, 1]), float(yaw[0]))
    goal_pose = (float(points[-1, 0]), float(points[-1, 1]), float(yaw[-1]))

    result = optimize_trajectory(
        inputs["astar_path_xy"],
        start_pose,
        goal_pose,
        config["limits"],
        reference_path_xy=points,
        reference_yaw=yaw,
        yaw_tangent_weight=config["yaw_tangent_weight"],
        clearance_m=inputs["clearance_m"],
        map_transform=inputs["transform"],
        required_clearance_m=inputs["required_clearance_m"],
        objective_config=config["objective"],
        trust_config=config["trust"],
        solver_config=config["solver"],
        **init_cfg,
    )
    if not result["success"]:
        raise SystemExit(
            f"optimization failed for episode {args.episode_index} with "
            f"status {result['status']}"
        )

    npz_name = f"episode_{args.episode_index:06d}.npz"
    metadata = {
        "schema_version": OPTIMIZATION_CANDIDATE_SCHEMA_VERSION,
        "scene_id": args.scene_id,
        "episode_index": args.episode_index,
        "inputs": {
            "trajectory_manifest": str(inputs["manifest_path"]),
            "episode_npz": str(inputs["episode_path"]),
            "esdf_npy": str(inputs["esdf_path"]),
        },
        "effective_config": config,
        "success": result["success"],
        "status": result["status"],
        "T_continuous": float(result["T_continuous"]),
        "T_output": float(result["T_output"]),
        "objectives": {
            "initial": result["objective_initial"],
            "continuous": result["objective_continuous"],
            "output": result["objective"],
        },
        "constraint_diagnostics": result["constraint_diagnostics"],
        "solver_metadata": result["solver_metadata"],
        "npz_filename": npz_name,
        "validated": False,
        "executable": False,
    }
    _publish_output(
        args.output_dir,
        args.episode_index,
        _candidate_npz_arrays(result["candidate"]),
        metadata,
    )
if __name__ == "__main__":
    main(parse_args())

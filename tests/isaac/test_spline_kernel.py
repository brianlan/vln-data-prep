"""Tests for the work package 6.2/6.3 B-spline kernel and initialization,
and work package 6.4 minimal SLSQP joint optimizer."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import BSpline

from optimize_sage3d_trajectories import (
    CONTROL_DT,
    SPLINE_DEGREE,
    _LOBATTO5_NODES,
    _build_nlp,
    _evaluate_spline,
    _span_nodes,
    bilinear_clearance,
    build_clamped_spline,
    canonical_output_time,
    clamped_knots,
    compute_initial_time,
    compute_n_control_points,
    continuous_world_to_pixel,
    cumulative_arc_length,
    derivative_control_points,
    estimate_time_components,
    eval_derivatives,
    initialize_trajectory,
    jerk_integral_sq,
    lift_to_reference_branch,
    optimize_trajectory,
    resample_by_arc_length,
    time_policy_bounds,
    yaw_unwrap,
    yaw_wrap,
)
from sage3d.utils import MapTransform

# Large limits so t_min dominates the initial-time selection for the
# straight synthetic paths used here; derivative limits are exercised
# separately in test_estimate_time_components_scaling and
# test_time_policy_bounds_and_exceeds_flag.
LIMITS = {
    "v_max": 1000.0,
    "a_max": 1000.0,
    "j_max": 1000.0,
    "yaw_rate_max": 1000.0,
    "yaw_accel_max": 1000.0,
    "yaw_jerk_max": 1000.0,
}

TINY_LIMITS = {
    "v_max": 1.0,
    "a_max": 1.0,
    "j_max": 1.0,
    "yaw_rate_max": 1.0,
    "yaw_accel_max": 1.0,
    "yaw_jerk_max": 1.0,
}

# Explicit yaw_tangent_weight for initialize_trajectory calls: the plan defines
# no default, so tests supply one. The straight synthetic paths keep the
# tangent at zero, so the weight does not affect the asserted results.
YAW_WEIGHT = 0.5

# Work package 6.4 shared fixtures: 60 x 40 cells at 0.5 m give world x in
# [0, 30], y in [0, 20]; paths stay fully interior so every bilinear clearance
# query keeps all four neighbors in-bounds. All 6.4 config values are supplied
# by the caller (the plan defines no defaults).
MAP_TRANSFORM = MapTransform(height=40, width=60, scale=0.5,
                              lower_x=0.0, lower_y=0.0)
CLEARANCE_FULL = np.full((40, 60), 2.0)
CLEARANCE_ZERO = np.zeros((40, 60))
OBJECTIVE_CONFIG = {
    "w_ref": 1.0, "w_jerk_xy": 1.0, "w_jerk_yaw": 1.0, "w_yaw_rate": 1.0,
    "w_time": 1.0,
    "reference_distance_scale_m": 1.0, "jerk_xy_scale": 10.0,
    "jerk_yaw_scale": 10.0, "yaw_rate_scale": 1.0, "time_scale_s": 1.0,
}
TRUST_CONFIG = {"trust_xy_resolution_cells": 2.0, "trust_xy_max_m": 1.0,
                "trust_yaw_rad": 0.5}
SOLVER_CONFIG = {
    "ftol": 1e-8, "maxiter": 1000, "episode_timeout_s": 30.0,
    "constraint_tolerance": 1e-6, "final_objective_tolerance": 1e-6,
    "clearance_scale_m": 1.0,
}


def straight_init(path_length=1.5):
    path = straight_path(path_length, y=2.0, x0=2.0)
    return initialize_trajectory(
        path, (2.0, 2.0, 0.0), (2.0 + path_length, 2.0, 0.0), LIMITS,
        reference_path_xy=path, reference_yaw=np.zeros(len(path)),
        yaw_tangent_weight=YAW_WEIGHT,
    )


def straight_nlp(path_length=1.5, limits=LIMITS, clearance=CLEARANCE_FULL,
                 trust=TRUST_CONFIG):
    init = straight_init(path_length)
    t = init["initialization"]["time"]
    return _build_nlp(
        init["control_points"], t["t_min"], t["t_max"], limits,
        straight_path(path_length, y=2.0, x0=2.0), clearance, MAP_TRANSFORM,
        0.3, OBJECTIVE_CONFIG, trust, SOLVER_CONFIG,
    )


@pytest.fixture
def ctrl():
    points = np.random.default_rng(20260720).standard_normal((8, 3))
    points[1] = points[0]
    points[-2] = points[-1]
    return points


def straight_path(length=5.0, n=26, y=0.0, x0=0.0):
    return np.column_stack(
        [np.linspace(x0, x0 + length, n), np.full(n, y)]
    )


def reference_from_path(path, yaw=None):
    """Smoothed-episode stand-in: the path itself with continuous tangent yaw."""
    if yaw is None:
        yaw = np.unwrap(np.arctan2(np.gradient(path[:, 1]), np.gradient(path[:, 0])))
    return path, np.asarray(yaw, dtype=float)


def test_clamped_knots():
    knots = clamped_knots(8)
    expected = np.array([0] * 6 + [1 / 3, 2 / 3] + [1] * 6)
    assert np.allclose(knots, expected)
    assert clamped_knots(6).shape == (12,)
    with pytest.raises(ValueError):
        clamped_knots(5)


def test_basis_nonnegative_and_partition_of_unity():
    spline = BSpline(clamped_knots(8), np.eye(8), SPLINE_DEGREE)
    values = spline(np.linspace(0.0, 1.0, 101))
    assert np.all(values >= -1e-12)
    assert np.allclose(values.sum(axis=1), 1.0)


def test_clamped_spline_interpolates_endpoints(ctrl):
    spline = build_clamped_spline(ctrl)
    assert np.allclose(spline(0.0), ctrl[0])
    assert np.allclose(spline(1.0), ctrl[-1])


def test_repeated_endpoint_control_points_give_zero_velocity(ctrl):
    spline = build_clamped_spline(ctrl)
    assert np.allclose(spline.derivative(1)([0.0, 1.0]), 0.0)
    evaluated = eval_derivatives(ctrl, T=2.0, u=np.array([0.0, 1.0]))
    assert np.allclose(evaluated["velocity"], 0.0)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_analytic_derivatives_match_finite_difference(ctrl, order):
    spline = build_clamped_spline(ctrl)
    u = np.array([0.35])
    h = {1: 1e-6, 2: 1e-4, 3: 1e-3}[order]
    if order == 1:
        finite_difference = (spline(u + h) - spline(u - h)) / (2 * h)
    elif order == 2:
        finite_difference = (
            spline(u + h) - 2 * spline(u) + spline(u - h)
        ) / h**2
    else:
        finite_difference = (
            spline(u + 2 * h)
            - 2 * spline(u + h)
            + 2 * spline(u - h)
            - spline(u - 2 * h)
        ) / (2 * h**3)

    analytic = spline.derivative(order)(u)
    assert np.allclose(analytic, finite_difference, rtol=1e-4, atol=1e-4)
    key = {1: "velocity", 2: "acceleration", 3: "jerk"}[order]
    evaluated = eval_derivatives(ctrl, T=2.5, u=u)
    assert np.allclose(evaluated[key], analytic / 2.5**order)


def test_derivative_control_points_match_scipy(ctrl):
    spline = build_clamped_spline(ctrl)
    u = np.linspace(0.05, 0.95, 37)
    for order in (1, 2, 3):
        knots, points = derivative_control_points(spline.t, ctrl, order)
        explicit = BSpline(knots, points, SPLINE_DEGREE - order)
        assert np.allclose(explicit(u), spline.derivative(order)(u))


def test_time_scaling(ctrl):
    u = np.array([0.3, 0.7])
    first = eval_derivatives(ctrl, T=1.0, u=u)
    second = eval_derivatives(ctrl, T=2.0, u=u)
    assert np.allclose(second["position"], first["position"])
    assert np.allclose(second["velocity"], first["velocity"] / 2)
    assert np.allclose(second["acceleration"], first["acceleration"] / 4)
    assert np.allclose(second["jerk"], first["jerk"] / 8)


def test_jerk_integral(ctrl):
    derivative = build_clamped_spline(ctrl).derivative(3)

    def integrand(u):
        return float(np.sum(derivative(u) ** 2))

    reference, _ = quad(integrand, 0.0, 1.0, epsabs=1e-12, epsrel=1e-12)
    assert np.isclose(jerk_integral_sq(ctrl, 1.0), reference, rtol=1e-10)
    assert np.isclose(
        jerk_integral_sq(ctrl, 2.0),
        jerk_integral_sq(ctrl, 1.0) / 32,
    )
    assert np.isclose(
        jerk_integral_sq(ctrl, 1.0),
        jerk_integral_sq(ctrl[:, :2], 1.0)
        + jerk_integral_sq(ctrl[:, 2:], 1.0),
    )


def test_yaw_wrap_and_unwrap():
    yaw = np.deg2rad([170.0, -170.0])
    unwrapped = yaw_unwrap(yaw)
    assert np.allclose(np.rad2deg(unwrapped), [170.0, 190.0])
    assert np.allclose(np.rad2deg(yaw_wrap(unwrapped)), [170.0, -170.0])
    assert yaw_wrap(np.array([np.pi]))[0] == -np.pi


# --------------------------------------------------------------------------
# Work package 6.3: initialization and integration tests.
# --------------------------------------------------------------------------


def test_compute_n_control_points():
    assert compute_n_control_points(2.0) == 9  # ceil(2/0.5) + 5
    assert compute_n_control_points(0.1) == 8  # clamped to min
    assert compute_n_control_points(100.0) == 64  # clamped to max
    with pytest.raises(ValueError):
        compute_n_control_points(0.0)
    with pytest.raises(ValueError):
        compute_n_control_points(np.nan)


def test_cumulative_arc_length():
    path = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 4.0]])
    assert np.allclose(cumulative_arc_length(path), [0.0, 5.0, 8.0])


def test_resample_by_arc_length_preserves_endpoints():
    path = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]])
    s = cumulative_arc_length(path)
    resampled = resample_by_arc_length(path, s, 5)
    assert np.allclose(resampled[0], path[0])
    assert np.allclose(resampled[-1], path[-1])


def test_lift_to_reference_branch():
    assert np.isclose(lift_to_reference_branch(7 * np.pi / 4, 0.0), -np.pi / 4)
    assert np.isclose(lift_to_reference_branch(np.pi, 0.0), np.pi)
    assert np.isclose(
        np.rad2deg(lift_to_reference_branch(np.deg2rad(-170.0), np.deg2rad(190.0))),
        190.0,
    )


def test_initialize_trajectory_output():
    ref_path, ref_yaw = reference_from_path(straight_path())
    output = initialize_trajectory(
        straight_path(), (0.0, 0.0, 0.0), (5.0, 0.0, 0.0), LIMITS,
        reference_path_xy=ref_path, reference_yaw=ref_yaw,
        yaw_tangent_weight=YAW_WEIGHT,
    )
    init = output["initialization"]
    assert init["path_length_m"] == pytest.approx(5.0)
    assert init["n_control_points"] == 15  # ceil(5/0.5) + 5
    assert init["time"]["t_min"] == pytest.approx(5.0)
    assert init["time"]["t_max"] == pytest.approx(15.0)
    assert init["time"]["t_init"] == output["total_time"]
    assert init["time"]["t_init"] == pytest.approx(6.0)  # gamma * t_min
    assert not init["initial_time_exceeds_policy_max"]
    assert init["goal_yaw_unwrapped"] == 0.0
    assert init["start_yaw_unwrapped"] == 0.0
    assert init["yaw_tangent_weight"] == YAW_WEIGHT

    ctrl = output["control_points"]
    assert ctrl.shape == (15, 3)
    assert np.array_equal(ctrl[0], [0.0, 0.0, 0.0])
    assert np.array_equal(ctrl[1], [0.0, 0.0, 0.0])
    assert np.array_equal(ctrl[-2], [5.0, 0.0, 0.0])
    assert np.array_equal(ctrl[-1], [5.0, 0.0, 0.0])

    assert output["time"].shape == (round(output["total_time"] / CONTROL_DT) + 1,)
    assert np.allclose(output["velocity_world"][[0, -1]], 0.0)
    assert np.allclose(output["yaw_rate"][[0, -1]], 0.0)
    assert np.allclose(output["position_world"][0], [0.0, 0.0])
    assert np.allclose(output["position_world"][-1], [5.0, 0.0])
    assert np.isclose(output["yaw_unwrapped"][0], 0.0)
    assert np.isclose(output["yaw_unwrapped"][-1], init["goal_yaw_unwrapped"])
    assert np.all(output["yaw_wrapped"] >= -np.pi)
    assert np.all(output["yaw_wrapped"] < np.pi)


def test_initialize_trajectory_yaw_lifted_to_reference_endpoint():
    # 7*pi/4 wrapped equals -pi/4; with an all-zero reference it lifts to
    # -pi/4 (the branch nearest the reference endpoint, not shortest from 0).
    ref_path, ref_yaw = reference_from_path(straight_path())
    output = initialize_trajectory(
        straight_path(), (0.0, 0.0, 0.0), (5.0, 0.0, 7 * np.pi / 4), LIMITS,
        reference_path_xy=ref_path, reference_yaw=ref_yaw,
        yaw_tangent_weight=YAW_WEIGHT,
    )
    goal_yaw = output["initialization"]["goal_yaw_unwrapped"]
    assert np.isclose(goal_yaw, -np.pi / 4)
    ctrl = output["control_points"]
    assert np.isclose(ctrl[0, 2], 0.0)
    assert np.isclose(ctrl[1, 2], 0.0)
    assert np.isclose(ctrl[-2, 2], goal_yaw)
    assert np.isclose(ctrl[-1, 2], goal_yaw)
    assert np.allclose(output["yaw_rate"][[0, -1]], 0.0)


def test_initialize_trajectory_init_knobs_on_bent_path():
    # L-shaped path so the QP and tangent refs are non-trivial. lambda_init
    # smooths the XY second differences; yaw_tangent_weight pulls interior yaw
    # toward the reference yaw. Boundaries stay pinned regardless of both.
    path = np.array([[0, 0], [2, 0], [4, 0], [6, 0], [6, 2], [6, 4], [6, 6]])
    pose0, pose1 = (0.0, 0.0, 0.0), (6.0, 6.0, 0.0)
    ref_path, ref_yaw = reference_from_path(path)

    xy0 = initialize_trajectory(path, pose0, pose1, LIMITS,
                              reference_path_xy=ref_path, reference_yaw=ref_yaw,
                              yaw_tangent_weight=0.0, lambda_init=0.0)
    xy1 = initialize_trajectory(path, pose0, pose1, LIMITS,
                              reference_path_xy=ref_path, reference_yaw=ref_yaw,
                              yaw_tangent_weight=0.0, lambda_init=1.0)

    def xy_second_diff_norm(ctrl):
        return float(np.linalg.norm(np.diff(ctrl[:, :2], axis=0, n=2)))

    assert xy_second_diff_norm(xy1["control_points"]) < xy_second_diff_norm(
        xy0["control_points"]
    )

    yaw0 = initialize_trajectory(path, pose0, pose1, LIMITS,
                               reference_path_xy=ref_path, reference_yaw=ref_yaw,
                               yaw_tangent_weight=0.0, lambda_init=1.0)
    yaw1 = initialize_trajectory(path, pose0, pose1, LIMITS,
                               reference_path_xy=ref_path, reference_yaw=ref_yaw,
                               yaw_tangent_weight=1.0, lambda_init=1.0)
    c0, c1 = yaw0["control_points"], yaw1["control_points"]
    assert not np.allclose(c0[2:-2, 2], c1[2:-2, 2])
    assert np.allclose(c0[0, :], c1[0, :]) and np.allclose(c0[-1, :], c1[-1, :])
    assert np.allclose(c0[1, :], c1[1, :]) and np.allclose(c0[-2, :], c1[-2, :])


def test_yaw_follows_reference_continuous_branch():
    # Reference yaw wraps 170 deg -> -170 deg, which unwraps to the continuous
    # 170 deg -> 190 deg branch. The optimizer must follow that branch instead
    # of rotating -340 deg around the short way.
    path = straight_path()
    ref_yaw = yaw_wrap(np.deg2rad(np.linspace(170.0, 190.0, path.shape[0])))
    output = initialize_trajectory(
        path, (0.0, 0.0, np.deg2rad(170.0)), (5.0, 0.0, np.deg2rad(-170.0)),
        LIMITS, reference_path_xy=path, reference_yaw=ref_yaw,
        yaw_tangent_weight=YAW_WEIGHT,
    )
    goal_u = output["initialization"]["goal_yaw_unwrapped"]
    assert np.isclose(np.rad2deg(goal_u), 190.0)
    yaw_u = output["yaw_unwrapped"]
    assert np.isclose(np.rad2deg(yaw_u[0]), 170.0)
    assert np.isclose(np.rad2deg(yaw_u[-1]), 190.0)
    # A smooth ~20 deg sweep on the reference branch, never a 340 deg wrap.
    assert np.all(np.abs(np.diff(yaw_u)) < np.deg2rad(10.0))
    ctrl = output["control_points"]
    assert np.isclose(np.rad2deg(ctrl[0, 2]), 170.0)
    assert np.isclose(np.rad2deg(ctrl[-1, 2]), 190.0)
    assert np.allclose(output["yaw_rate"][[0, -1]], 0.0)


def test_yaw_tangent_weight_above_one_is_penalty_not_blend():
    # L path with tangent reference yaw 0 -> pi/2, goal already on the
    # reference branch. A >1 blend coefficient would extrapolate interior yaw
    # past the [0, pi/2] hull; the penalty weight must keep it inside.
    path = np.array([[0, 0], [2, 0], [4, 0], [6, 0], [6, 2], [6, 4], [6, 6]])
    ref_path, ref_yaw = reference_from_path(path)
    output = initialize_trajectory(
        path, (0.0, 0.0, 0.0), (6.0, 6.0, np.pi / 2), LIMITS,
        reference_path_xy=ref_path, reference_yaw=ref_yaw,
        yaw_tangent_weight=5.0,
    )
    interior = output["control_points"][2:-2, 2]
    # Inside the [0, pi/2] hull up to the clamped-spline overshoot (~0.01 rad);
    # a >1 blend coefficient would extrapolate an order of magnitude further.
    assert np.all(interior >= -1e-2)
    assert np.all(interior <= np.pi / 2 + 1e-2)


def test_u_shaped_reference_no_first_leg_pre_rotation():
    # U path: up the left leg (heading pi/2), across, down the right leg
    # (heading -pi/2). Reference yaw unwraps continuously pi/2 -> 0 -> -pi/2;
    # the first-leg yaw must stay near pi/2 with no 2*pi jumps.
    path = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [3, 2], [3, 1], [3, 0]],
        dtype=float,
    )
    ref_path, ref_yaw = reference_from_path(path)
    output = initialize_trajectory(
        path, (0.0, 0.0, np.pi / 2), (3.0, 0.0, -np.pi / 2), LIMITS,
        reference_path_xy=ref_path, reference_yaw=ref_yaw,
        yaw_tangent_weight=2.0,
    )
    goal_u = output["initialization"]["goal_yaw_unwrapped"]
    assert np.isclose(goal_u, -np.pi / 2)
    yaw_u = output["yaw_unwrapped"]
    # Single continuous sweep pi/2 -> -pi/2, no 2*pi jumps.
    assert np.all(np.abs(np.diff(yaw_u)) < np.pi / 2)
    # Midway up the first leg (first eighth of the trajectory): yaw must not
    # have pre-rotated more than 45 deg away from the first-leg heading.
    assert yaw_u[len(yaw_u) // 8] > np.pi / 4
    assert np.isclose(yaw_u[0], np.pi / 2)
    assert np.isclose(yaw_u[-1], -np.pi / 2)
    assert np.allclose(output["yaw_rate"][[0, -1]], 0.0)


def test_estimate_time_components_constant_control_points():
    ctrl_constant = np.tile([1.0, 2.0, 0.5], (8, 1))
    comps = estimate_time_components(ctrl_constant, LIMITS)
    assert comps["t_v"] == 0.0
    assert comps["t_a"] == 0.0
    assert comps["t_j"] == 0.0


def test_estimate_time_components_scaling():
    ctrl_ramp = np.column_stack(
        [np.linspace(0.0, 5.0, 15), np.zeros(15), np.linspace(0.0, 1.0, 15)]
    )
    base = estimate_time_components(ctrl_ramp, TINY_LIMITS)
    assert base["t_v"] > 0.0
    assert base["t_a"] > 0.0
    assert base["t_j"] > 0.0
    k = 4.0
    scaled = estimate_time_components(ctrl_ramp * k, TINY_LIMITS)
    assert np.isclose(scaled["t_v"], base["t_v"] * k, rtol=1e-6)
    assert np.isclose(scaled["t_a"], base["t_a"] * np.sqrt(k), rtol=1e-6)
    assert np.isclose(scaled["t_j"], base["t_j"] * np.cbrt(k), rtol=1e-6)


def test_time_policy_bounds_dt_aligned():
    # Floating-point ratios (3*0.3/0.1, 3*0.7/0.1) must not floor to the wrong
    # control step: expected 0.9 and 2.1, not 0.8 and 2.0.
    assert time_policy_bounds(0.3) == pytest.approx((0.3, 0.9))
    assert time_policy_bounds(0.7) == pytest.approx((0.7, 2.1))


def test_time_policy_bounds_and_exceeds_flag():
    ctrl_ramp = np.column_stack(
        [np.linspace(0.0, 5.0, 15), np.zeros(15), np.linspace(0.0, 1.0, 15)]
    )
    # Large limits: t_min dominates, candidate = gamma * t_min = 6.0.
    small = compute_initial_time(ctrl_ramp, 5.0, LIMITS)
    assert small["t_min"] == pytest.approx(5.0)
    assert small["t_max"] == pytest.approx(15.0)
    assert not small["initial_time_exceeds_policy_max"]
    assert small["candidate"] == pytest.approx(1.2 * 5.0)
    assert small["t_init"] == pytest.approx(6.0)

    # Tiny limits: derivative times dominate and exceed t_max.
    large = compute_initial_time(ctrl_ramp, 5.0, TINY_LIMITS)
    assert large["initial_time_exceeds_policy_max"]
    assert large["t_init"] == pytest.approx(15.0)


def test_initialize_trajectory_rejects_invalid_inputs():
    ref_path, ref_yaw = reference_from_path(straight_path())

    def call(**overrides):
        kwargs = {
            "reference_path_xy": ref_path,
            "reference_yaw": ref_yaw,
            "yaw_tangent_weight": YAW_WEIGHT,
        }
        kwargs.update(overrides)
        return initialize_trajectory(
            straight_path(), (0, 0, 0), (5, 0, 0), LIMITS, **kwargs
        )

    with pytest.raises(ValueError):
        initialize_trajectory(straight_path()[:, :1], (0, 0, 0), (5, 0, 0), LIMITS,
                            reference_path_xy=ref_path, reference_yaw=ref_yaw,
                            yaw_tangent_weight=YAW_WEIGHT)
    with pytest.raises(ValueError):
        initialize_trajectory(straight_path()[:1], (0, 0, 0), (5, 0, 0), LIMITS,
                            reference_path_xy=ref_path, reference_yaw=ref_yaw,
                            yaw_tangent_weight=YAW_WEIGHT)
    bad = straight_path()
    bad[5, 0] = np.nan
    with pytest.raises(ValueError):
        initialize_trajectory(bad, (0, 0, 0), (5, 0, 0), LIMITS,
                            reference_path_xy=ref_path, reference_yaw=ref_yaw,
                            yaw_tangent_weight=YAW_WEIGHT)
    with pytest.raises(ValueError):
        initialize_trajectory(
            straight_path(0.0, n=2), (0, 0, 0), (0, 0, 0), LIMITS,
            reference_path_xy=ref_path, reference_yaw=ref_yaw,
            yaw_tangent_weight=YAW_WEIGHT,
        )
    with pytest.raises(ValueError):
        initialize_trajectory(straight_path(), (0, 0), (5, 0, 0), LIMITS,
                            reference_path_xy=ref_path, reference_yaw=ref_yaw,
                            yaw_tangent_weight=YAW_WEIGHT)
    bad_limits = dict(LIMITS)
    bad_limits["v_max"] = 0.0
    with pytest.raises(ValueError):
        initialize_trajectory(straight_path(), (0, 0, 0), (5, 0, 0), bad_limits,
                            reference_path_xy=ref_path, reference_yaw=ref_yaw,
                            yaw_tangent_weight=YAW_WEIGHT)
    dup = straight_path()
    dup[5] = dup[4]
    with pytest.raises(ValueError):
        initialize_trajectory(dup, (0, 0, 0), (5, 0, 0), LIMITS,
                            reference_path_xy=ref_path, reference_yaw=ref_yaw,
                            yaw_tangent_weight=YAW_WEIGHT)

    with pytest.raises(ValueError):
        call(reference_path_xy=straight_path()[:, :1])
    with pytest.raises(ValueError):
        call(reference_yaw=np.zeros(10))
    with pytest.raises(ValueError):
        call(reference_yaw=np.full(26, np.nan))
    bad_ref = ref_path.copy()
    bad_ref[3] = bad_ref[2]
    with pytest.raises(ValueError):
        call(reference_path_xy=bad_ref)
    with pytest.raises(ValueError):
        call(yaw_tangent_weight=-1.0)
    with pytest.raises(ValueError):
        call(yaw_tangent_weight=np.nan)


# --------------------------------------------------------------------------
# Work package 6.2: spline evaluation contract (driven through _evaluate_spline).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("total_time", [0.0, np.nan, 0.15, 1e-12])
def test_evaluate_spline_rejects_invalid_time(ctrl, total_time):
    with pytest.raises(ValueError):
        _evaluate_spline(ctrl, total_time)


def test_evaluate_spline_rejects_invalid_control_points(ctrl):
    invalid = [
        np.zeros(8),
        np.zeros((8, 2)),
        np.zeros((5, 3)),
        ctrl.copy(),
    ]
    invalid[-1][3, 0] = np.nan
    for points in invalid:
        with pytest.raises(ValueError):
            _evaluate_spline(points, 1.0)


@pytest.mark.parametrize("total_time", [0.1 * 3, 45.0])
def test_evaluate_spline_fixed_period(ctrl, total_time):
    output = _evaluate_spline(ctrl, total_time)
    expected_steps = round(total_time / CONTROL_DT)
    assert output["time"].shape == (expected_steps + 1,)
    assert output["total_time"] == expected_steps * CONTROL_DT
    assert np.allclose(np.diff(output["time"]), CONTROL_DT, atol=1e-12)


# --------------------------------------------------------------------------
# Work package 6.4: minimal SLSQP joint optimizer.
# --------------------------------------------------------------------------


def test_64_pack_unpack_and_fixed_endpoint_controls():
    init = straight_init()
    ctrl0 = init["control_points"]
    t = init["initialization"]["time"]
    nlp = straight_nlp()
    x0 = nlp["pack"](ctrl0[2:-2], t["t_init"])
    z_xy, z_yaw, tau, T, p_free = nlp["unpack"](x0)
    assert T == pytest.approx(t["t_init"])
    assert np.allclose(z_xy, 0.0)
    assert np.allclose(z_yaw, 0.0)
    # Endpoint controls (0, 1, n-2, n-1) stay pinned exactly.
    p_full = nlp["evaluate"](x0)["control_points"]
    for i in (0, 1, -2, -1):
        assert np.array_equal(p_full[i], ctrl0[i])
    assert np.allclose(p_full[2:-2], p_free)
    # Deltas are normalized: xy by r_xy, yaw by trust_yaw_rad.
    moved = p_free.copy()
    moved[0, 0] += 0.5 * nlp["r_xy"]
    moved[0, 2] += 0.25 * nlp["yaw_rad"]
    z2_xy, z2_yaw, _, T2, _ = nlp["unpack"](nlp["pack"](moved, t["t_init"]))
    assert np.isclose(z2_xy[0, 0], 0.5)
    assert np.isclose(z2_yaw[0], 0.25)
    assert T2 == pytest.approx(t["t_init"])


def test_64_objective_time_scaling_and_sampling_independence():
    # Path length 5 -> policy [T_min, T_max] = [5, 15]; T=6 and T=12 are both
    # reachable by tau in [0, 1]. Internal yaw is perturbed inside the trust
    # region so the yaw terms are nonzero (an all-zero yaw would make the
    # yaw-scaling assertions vacuous).
    init = straight_init(5.0)
    p_free = init["control_points"][2:-2].copy()
    p_free[:, 2] = TRUST_CONFIG["trust_yaw_rad"] * np.array(
        [0.6, 0.4, -0.4, -0.6, 0.6, 0.4, -0.4, -0.6, 0.6, 0.4, -0.4]
    )
    nlp = straight_nlp(5.0)
    r1 = nlp["evaluate"](nlp["pack"](p_free, 6.0))["objective"]["raw"]
    r2 = nlp["evaluate"](nlp["pack"](p_free, 12.0))["objective"]["raw"]
    assert r1["jerk_yaw"] > 0.0 and r1["yaw_rate"] > 0.0
    assert r2["jerk_xy"] == pytest.approx(r1["jerk_xy"] / 32.0, rel=1e-9)
    assert r2["jerk_yaw"] == pytest.approx(r1["jerk_yaw"] / 32.0, rel=1e-9)
    assert r2["yaw_rate"] == pytest.approx(r1["yaw_rate"] / 4.0, rel=1e-9)
    assert r2["ref"] == pytest.approx(r1["ref"], rel=1e-9)
    assert r2["time"] == pytest.approx(2.0 * r1["time"])

    # Sampling independence: the raw terms are fixed-quadrature integrals; an
    # independent dense-grid trapezoid over the same spans must agree.
    p_full = init["control_points"].copy()
    p_full[2:-2] = p_free
    spline = BSpline(clamped_knots(p_full.shape[0]), p_full, SPLINE_DEGREE)
    d1, d3 = spline.derivative(1), spline.derivative(3)
    ref_path = straight_path(5.0, y=2.0, x0=2.0)
    arc_n = cumulative_arc_length(ref_path)
    arc_n = arc_n / arc_n[-1]
    u = np.linspace(0.0, 1.0, 4001)
    pos, v1, v3 = spline(u), d1(u), d3(u)
    dense_ref = np.trapz(
        (pos[:, 0] - np.interp(u, arc_n, ref_path[:, 0])) ** 2
        + (pos[:, 1] - np.interp(u, arc_n, ref_path[:, 1])) ** 2,
        u,
    )
    # Jerk integrands are quartic per span (GL3 exact); the degree-8 yaw-rate
    # and degree-10 ref integrands carry the fixed quadrature's truncation
    # (~1.5% on this path).
    assert r1["jerk_xy"] == pytest.approx(
        np.trapz(np.sum(v3[:, :2] ** 2, axis=1), u) / 6.0**5, rel=1e-3
    )
    assert r1["jerk_yaw"] == pytest.approx(
        np.trapz(v3[:, 2] ** 2, u) / 6.0**5, rel=1e-3
    )
    assert r1["yaw_rate"] == pytest.approx(
        np.trapz(v1[:, 2] ** 2, u) / 6.0**2, rel=2e-2
    )
    assert r1["ref"] == pytest.approx(dense_ref, rel=2e-2)


def test_64_derivative_margin_sign():
    init = straight_init()
    ctrl0 = init["control_points"]
    t = init["initialization"]["time"]
    x0 = straight_nlp()["pack"](ctrl0[2:-2], t["t_init"])
    ok = straight_nlp()["evaluate"](x0)["margin_groups"]["velocity_xy"]
    # v_max = 0.5 violates the ~0.9 m/s straight-line init at T=1.8 (the shared
    # TINY_LIMITS use v_max = 1.0 and would stay feasible here).
    bad = straight_nlp(limits=dict(LIMITS, v_max=0.5))["evaluate"](x0)[
        "margin_groups"
    ]["velocity_xy"]
    assert ok.min() > 0.0
    # Endpoint Lobatto nodes sit at u=0/u=1 where the clamped spline has zero
    # velocity (margin 1.0); the interior nodes must violate under tiny limits.
    assert bad.min() < 0.0
    assert bad[1:-1].max() < 0.0
    # Margin is exactly 1 - ||v||^2 / v_max^2 at the 5-point Gauss-Lobatto
    # nodes (span endpoints included) of each nonzero knot span.
    knots = clamped_knots(ctrl0.shape[0])
    u_lb, _ = _span_nodes(knots, _LOBATTO5_NODES)
    velocity = eval_derivatives(ctrl0, t["t_init"], u_lb)["velocity"]
    expected = 1.0 - np.sum(velocity[:, :2] ** 2, axis=1) / LIMITS["v_max"] ** 2
    assert np.allclose(ok, expected, rtol=1e-12)


def test_64_map_transform_reversal_bilinear_and_out_of_bounds():
    rng = np.random.default_rng(7)
    for row, col in zip(
        rng.integers(1, MAP_TRANSFORM.height - 2, 10),
        rng.integers(1, MAP_TRANSFORM.width - 2, 10),
    ):
        x, y = MAP_TRANSFORM.pixel_to_world(int(row), int(col))
        row_c, col_c = continuous_world_to_pixel(MAP_TRANSFORM, x, y)
        assert row_c == pytest.approx(float(row))
        assert col_c == pytest.approx(float(col))
    # X reversal: world +X maps to -col; the origin maps to col = width - 0.5.
    row_c, col_c = continuous_world_to_pixel(MAP_TRANSFORM, MAP_TRANSFORM.lower_x,
                                             MAP_TRANSFORM.lower_y)
    assert col_c == pytest.approx(MAP_TRANSFORM.width - 0.5)
    assert row_c == pytest.approx(-0.5)

    ones = np.ones((10, 10)) * 3.0
    row = np.array([2.25, 5.5])
    col = np.array([3.75, 7.25])
    assert np.allclose(bilinear_clearance(ones, row, col), 3.0)
    r, c = np.meshgrid(np.arange(10.0), np.arange(10.0), indexing="ij")
    ramp = 2.0 * r + 3.0 * c
    value = bilinear_clearance(ramp, np.array([4.5]), np.array([5.5]))[0]
    expected = (ramp[4, 5] + ramp[4, 6] + ramp[5, 5] + ramp[5, 6]) / 4.0
    assert value == pytest.approx(expected)
    # Any out-of-array neighbor makes the candidate infeasible.
    assert bilinear_clearance(ramp, np.array([-0.5]), np.array([5.0]))[0] == -np.inf
    assert bilinear_clearance(ramp, np.array([9.5]), np.array([5.0]))[0] == -np.inf
    assert bilinear_clearance(ramp, np.array([4.5]), np.array([9.6]))[0] == -np.inf


def test_64_separate_xy_disk_and_yaw_trust_regions():
    nlp = straight_nlp()
    assert nlp["r_xy"] == pytest.approx(1.0)  # min(2 cells * 0.5 m, 1.0 m)
    assert straight_nlp(trust=dict(TRUST_CONFIG, trust_xy_max_m=0.6))[
        "r_xy"
    ] == pytest.approx(0.6)
    n_free = nlp["n_free"]
    bounds = nlp["bounds"]
    # XY deltas have no box bounds (a Euclidean disk constraint governs),
    # yaw deltas are boxed in [-1, 1], time tau in [0, 1].
    assert all(lo is None for lo, hi in bounds[: 2 * n_free])
    assert all(bounds[i] == (-1.0, 1.0) for i in range(2 * n_free, 3 * n_free))
    assert bounds[-1] == (0.0, 1.0)
    # Recomputed bound margins: disk is 1 - ||z_xy||^2, yaw/tau bounds separate.
    init = straight_init()
    t = init["initialization"]["time"]
    x0 = nlp["pack"](init["control_points"][2:-2], t["t_init"])
    groups = nlp["evaluate"](x0)["margin_groups"]
    assert np.allclose(groups["trust_xy_disk"], 1.0)
    assert np.allclose(groups["yaw_bound_low"], 1.0)
    assert np.allclose(groups["yaw_bound_high"], 1.0)
    tau0 = (t["t_init"] - t["t_min"]) / (t["t_max"] - t["t_min"])
    assert groups["tau_low"][0] == pytest.approx(tau0)
    assert groups["tau_high"][0] == pytest.approx(1.0 - tau0)
    # A yaw decision outside [-1, 1] must show a negative recomputed margin.
    x_out = x0.copy()
    x_out[2 * n_free] = 1.5
    assert nlp["evaluate"](x_out)["margin_groups"]["yaw_bound_high"].min() < 0.0
    # Canonicalization past T_max (tau > 1) must fail the tau bound margin.
    over = nlp["evaluate"](
        nlp["pack"](init["control_points"][2:-2], t["t_max"] + CONTROL_DT)
    )
    assert over["margin_groups"]["tau_high"].min() < 0.0


def test_64_slsqp_feasible_straight_end_to_end():
    path = straight_path(1.5, y=2.0, x0=2.0)
    result = optimize_trajectory(
        path, (2.0, 2.0, 0.0), (3.5, 2.0, 0.0), LIMITS,
        reference_path_xy=path, reference_yaw=np.zeros(len(path)),
        yaw_tangent_weight=YAW_WEIGHT, clearance_m=CLEARANCE_FULL,
        map_transform=MAP_TRANSFORM, required_clearance_m=0.3,
        objective_config=OBJECTIVE_CONFIG, trust_config=TRUST_CONFIG,
        solver_config=SOLVER_CONFIG,
    )
    assert result["success"]
    assert result["status"] == "success"
    assert result["solver_metadata"]["result_success"]
    ctrl = result["control_points"]
    assert np.allclose(ctrl[0], [2.0, 2.0, 0.0])
    assert np.allclose(ctrl[1], [2.0, 2.0, 0.0])
    assert np.allclose(ctrl[-2], [3.5, 2.0, 0.0])
    assert np.allclose(ctrl[-1], [3.5, 2.0, 0.0])
    # T_output is authoritative: canonical, within policy, and the candidate
    # and reported objective/constraints are re-evaluated on it.
    assert result["T_output"] == pytest.approx(
        canonical_output_time(result["T_continuous"])
    )
    assert result["constraint_diagnostics"]["t_output_within_policy"]
    assert result["candidate"]["total_time"] == result["T_output"]
    assert result["candidate"]["time"].shape[0] == round(
        result["T_output"] / CONTROL_DT
    ) + 1
    assert np.all(np.isfinite(result["candidate"]["position_world"]))
    assert (
        result["constraint_diagnostics"]["final_margins_min"]
        >= -SOLVER_CONFIG["constraint_tolerance"]
    )
    # Audit flags are machine-diagnosable and must all pass here.
    assert all(
        result["constraint_diagnostics"][key]
        for key in ("finiteness_ok", "margins_ok", "monotonic_ok")
    )
    # Initialization is feasible here, so a higher aligned final objective
    # beyond tolerance would have to fail the candidate.
    assert (
        result["objective"]["total"]
        <= result["objective_initial"]["total"]
        + SOLVER_CONFIG["final_objective_tolerance"]
    )
    # Trust regions respected: per-point Euclidean disk and yaw bound.
    init_ctrl = straight_init()["control_points"]
    radius = min(TRUST_CONFIG["trust_xy_resolution_cells"] * MAP_TRANSFORM.scale,
                 TRUST_CONFIG["trust_xy_max_m"])
    assert np.max(
        np.linalg.norm(ctrl[2:-2, :2] - init_ctrl[2:-2, :2], axis=1)
    ) <= radius + 1e-9
    assert np.max(np.abs(ctrl[2:-2, 2] - init_ctrl[2:-2, 2])) <= TRUST_CONFIG[
        "trust_yaw_rad"
    ] + 1e-9


def test_64_infeasible_clearance_or_time_fails():
    path = straight_path(1.5, y=2.0, x0=2.0)
    kwargs = dict(
        reference_path_xy=path, reference_yaw=np.zeros(len(path)),
        yaw_tangent_weight=YAW_WEIGHT, map_transform=MAP_TRANSFORM,
        required_clearance_m=0.3, objective_config=OBJECTIVE_CONFIG,
        trust_config=TRUST_CONFIG, solver_config=SOLVER_CONFIG,
    )
    bad_clearance = optimize_trajectory(
        path, (2.0, 2.0, 0.0), (3.5, 2.0, 0.0), LIMITS,
        clearance_m=CLEARANCE_ZERO, **kwargs,
    )
    assert not bad_clearance["success"]
    # v_max = 0.2 is below the ~0.33 m/s fastest allowed re-parameterization
    # (1.5 m in T_max = 4.5 s), so the derivative constraints are infeasible
    # at every tau in [0, 1].
    bad_time = optimize_trajectory(
        path, (2.0, 2.0, 0.0), (3.5, 2.0, 0.0), dict(LIMITS, v_max=0.2),
        clearance_m=CLEARANCE_FULL, **kwargs,
    )
    assert not bad_time["success"]


def test_64_audit_rejects_solver_faults(monkeypatch):
    from types import SimpleNamespace

    import optimize_sage3d_trajectories as mod

    path = straight_path(1.5, y=2.0, x0=2.0)
    kwargs = dict(
        reference_path_xy=path, reference_yaw=np.zeros(len(path)),
        yaw_tangent_weight=YAW_WEIGHT, clearance_m=CLEARANCE_FULL,
        map_transform=MAP_TRANSFORM, required_clearance_m=0.3,
        objective_config=OBJECTIVE_CONFIG, trust_config=TRUST_CONFIG,
        solver_config=SOLVER_CONFIG,
    )

    def fake_minimize(result_x, success):
        def minimize(fun, x0, **kw):
            return SimpleNamespace(success=success, x=result_x, message="fake",
                                   nit=1, nfev=1)
        return minimize

    nlp = straight_nlp()
    x0 = nlp["pack"](straight_init()["control_points"][2:-2],
                     straight_init()["initialization"]["time"]["t_init"])
    # result_success=True with an out-of-yaw-bound decision: audit must reject.
    x_yaw = x0.copy()
    x_yaw[2 * nlp["n_free"]] = 1.5
    monkeypatch.setattr(mod, "minimize", fake_minimize(x_yaw, True))
    result = optimize_trajectory(
        path, (2.0, 2.0, 0.0), (3.5, 2.0, 0.0), LIMITS, **kwargs,
    )
    assert result["status"] == "audit_failed" and not result["success"]
    assert result["constraint_diagnostics"]["margins_ok"] is False
    # result_success=True with tau above T_max: canonical T_output > T_max
    # rejection must also fail the candidate.
    x_tau = x0.copy()
    x_tau[-1] = 2.0
    monkeypatch.setattr(mod, "minimize", fake_minimize(x_tau, True))
    result = optimize_trajectory(
        path, (2.0, 2.0, 0.0), (3.5, 2.0, 0.0), LIMITS, **kwargs,
    )
    assert result["status"] == "audit_failed" and not result["success"]
    assert result["constraint_diagnostics"]["t_output_within_policy"] is False
    # result_success=False with an otherwise valid x: never upgraded to success.
    monkeypatch.setattr(mod, "minimize", fake_minimize(x0, False))
    result = optimize_trajectory(
        path, (2.0, 2.0, 0.0), (3.5, 2.0, 0.0), LIMITS, **kwargs,
    )
    assert result["status"] == "solver_failed" and not result["success"]


def test_64_canonical_output_time_rounding():
    assert canonical_output_time(6.0) == pytest.approx(6.0)  # exact grid
    # Within tolerance below the next grid point: snaps down to the grid.
    assert canonical_output_time(6.0 + 0.5e-9) == pytest.approx(6.0)
    # Genuinely above the grid: rounds up to the next step.
    assert canonical_output_time(6.0 + 1e-6) == pytest.approx(6.1)
    assert canonical_output_time(6.37) == pytest.approx(6.4)
    for value in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            canonical_output_time(value)

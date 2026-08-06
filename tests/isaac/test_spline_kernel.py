"""Tests for the work package 6.2/6.3 B-spline kernel and initialization."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import BSpline

from optimize_sage3d_trajectories import (
    CONTROL_DT,
    SPLINE_DEGREE,
    _evaluate_spline,
    build_clamped_spline,
    clamped_knots,
    compute_initial_time,
    compute_n_control_points,
    cumulative_arc_length,
    derivative_control_points,
    estimate_time_components,
    eval_derivatives,
    jerk_integral_sq,
    lift_to_reference_branch,
    optimize_trajectory,
    resample_by_arc_length,
    time_policy_bounds,
    yaw_unwrap,
    yaw_wrap,
)

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

# Explicit yaw_tangent_weight for optimize_trajectory calls: the plan defines
# no default, so tests supply one. The straight synthetic paths keep the
# tangent at zero, so the weight does not affect the asserted results.
YAW_WEIGHT = 0.5


@pytest.fixture
def ctrl():
    points = np.random.default_rng(20260720).standard_normal((8, 3))
    points[1] = points[0]
    points[-2] = points[-1]
    return points


def straight_path(length=5.0, n=26):
    return np.column_stack([np.linspace(0.0, length, n), np.zeros(n)])


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


def test_optimize_trajectory_output():
    ref_path, ref_yaw = reference_from_path(straight_path())
    output = optimize_trajectory(
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


def test_optimize_trajectory_yaw_lifted_to_reference_endpoint():
    # 7*pi/4 wrapped equals -pi/4; with an all-zero reference it lifts to
    # -pi/4 (the branch nearest the reference endpoint, not shortest from 0).
    ref_path, ref_yaw = reference_from_path(straight_path())
    output = optimize_trajectory(
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


def test_optimize_trajectory_init_knobs_on_bent_path():
    # L-shaped path so the QP and tangent refs are non-trivial. lambda_init
    # smooths the XY second differences; yaw_tangent_weight pulls interior yaw
    # toward the reference yaw. Boundaries stay pinned regardless of both.
    path = np.array([[0, 0], [2, 0], [4, 0], [6, 0], [6, 2], [6, 4], [6, 6]])
    pose0, pose1 = (0.0, 0.0, 0.0), (6.0, 6.0, 0.0)
    ref_path, ref_yaw = reference_from_path(path)

    xy0 = optimize_trajectory(path, pose0, pose1, LIMITS,
                              reference_path_xy=ref_path, reference_yaw=ref_yaw,
                              yaw_tangent_weight=0.0, lambda_init=0.0)
    xy1 = optimize_trajectory(path, pose0, pose1, LIMITS,
                              reference_path_xy=ref_path, reference_yaw=ref_yaw,
                              yaw_tangent_weight=0.0, lambda_init=1.0)

    def xy_second_diff_norm(ctrl):
        return float(np.linalg.norm(np.diff(ctrl[:, :2], axis=0, n=2)))

    assert xy_second_diff_norm(xy1["control_points"]) < xy_second_diff_norm(
        xy0["control_points"]
    )

    yaw0 = optimize_trajectory(path, pose0, pose1, LIMITS,
                               reference_path_xy=ref_path, reference_yaw=ref_yaw,
                               yaw_tangent_weight=0.0, lambda_init=1.0)
    yaw1 = optimize_trajectory(path, pose0, pose1, LIMITS,
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
    output = optimize_trajectory(
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
    output = optimize_trajectory(
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
    output = optimize_trajectory(
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


def test_optimize_trajectory_rejects_invalid_inputs():
    ref_path, ref_yaw = reference_from_path(straight_path())

    def call(**overrides):
        kwargs = {
            "reference_path_xy": ref_path,
            "reference_yaw": ref_yaw,
            "yaw_tangent_weight": YAW_WEIGHT,
        }
        kwargs.update(overrides)
        return optimize_trajectory(
            straight_path(), (0, 0, 0), (5, 0, 0), LIMITS, **kwargs
        )

    with pytest.raises(ValueError):
        optimize_trajectory(straight_path()[:, :1], (0, 0, 0), (5, 0, 0), LIMITS,
                            reference_path_xy=ref_path, reference_yaw=ref_yaw,
                            yaw_tangent_weight=YAW_WEIGHT)
    with pytest.raises(ValueError):
        optimize_trajectory(straight_path()[:1], (0, 0, 0), (5, 0, 0), LIMITS,
                            reference_path_xy=ref_path, reference_yaw=ref_yaw,
                            yaw_tangent_weight=YAW_WEIGHT)
    bad = straight_path()
    bad[5, 0] = np.nan
    with pytest.raises(ValueError):
        optimize_trajectory(bad, (0, 0, 0), (5, 0, 0), LIMITS,
                            reference_path_xy=ref_path, reference_yaw=ref_yaw,
                            yaw_tangent_weight=YAW_WEIGHT)
    with pytest.raises(ValueError):
        optimize_trajectory(
            straight_path(0.0, n=2), (0, 0, 0), (0, 0, 0), LIMITS,
            reference_path_xy=ref_path, reference_yaw=ref_yaw,
            yaw_tangent_weight=YAW_WEIGHT,
        )
    with pytest.raises(ValueError):
        optimize_trajectory(straight_path(), (0, 0), (5, 0, 0), LIMITS,
                            reference_path_xy=ref_path, reference_yaw=ref_yaw,
                            yaw_tangent_weight=YAW_WEIGHT)
    bad_limits = dict(LIMITS)
    bad_limits["v_max"] = 0.0
    with pytest.raises(ValueError):
        optimize_trajectory(straight_path(), (0, 0, 0), (5, 0, 0), bad_limits,
                            reference_path_xy=ref_path, reference_yaw=ref_yaw,
                            yaw_tangent_weight=YAW_WEIGHT)
    dup = straight_path()
    dup[5] = dup[4]
    with pytest.raises(ValueError):
        optimize_trajectory(dup, (0, 0, 0), (5, 0, 0), LIMITS,
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

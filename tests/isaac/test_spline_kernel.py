"""Tests for the work package 6.2 quintic B-spline kernel."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import BSpline

from optimize_sage3d_trajectories import (
    CONTROL_DT,
    SPLINE_DEGREE,
    build_clamped_spline,
    clamped_knots,
    derivative_control_points,
    eval_derivatives,
    jerk_integral_sq,
    optimize_trajectory,
    yaw_unwrap,
    yaw_wrap,
)


@pytest.fixture
def ctrl():
    points = np.random.default_rng(20260720).standard_normal((8, 3))
    points[1] = points[0]
    points[-2] = points[-1]
    return points


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


def test_optimize_trajectory_output(ctrl):
    output = optimize_trajectory(ctrl, None, None, total_time=2.0)
    assert set(output) == {
        "time",
        "position_world",
        "yaw_unwrapped",
        "yaw_wrapped",
        "velocity_world",
        "acceleration_world",
        "jerk_world",
        "yaw_rate",
        "yaw_acceleration",
        "yaw_jerk",
        "total_time",
    }
    assert output["time"].shape == (21,)
    assert output["position_world"].shape == (21, 2)
    assert np.allclose(output["velocity_world"][[0, -1]], 0.0)
    assert np.allclose(output["yaw_rate"][[0, -1]], 0.0)
    assert np.all(output["yaw_wrapped"] >= -np.pi)
    assert np.all(output["yaw_wrapped"] < np.pi)


@pytest.mark.parametrize("total_time", [0.0, np.nan, 0.15, 1e-12])
def test_optimize_trajectory_rejects_invalid_time(ctrl, total_time):
    with pytest.raises(ValueError):
        optimize_trajectory(ctrl, None, None, total_time=total_time)


def test_optimize_trajectory_rejects_invalid_control_points(ctrl):
    invalid = [
        np.zeros(8),
        np.zeros((8, 2)),
        np.zeros((5, 3)),
        ctrl.copy(),
    ]
    invalid[-1][3, 0] = np.nan
    for points in invalid:
        with pytest.raises(ValueError):
            optimize_trajectory(points, None, None, total_time=1.0)


def test_optimize_trajectory_rejects_nonzero_endpoint_velocity(ctrl):
    for index in (1, -2):
        invalid = ctrl.copy()
        invalid[index, 0] += 0.5
        with pytest.raises(ValueError):
            optimize_trajectory(invalid, None, None, total_time=1.0)


@pytest.mark.parametrize("total_time", [0.1 * 3, 45.0])
def test_optimize_trajectory_fixed_period(ctrl, total_time):
    output = optimize_trajectory(ctrl, None, None, total_time=total_time)
    expected_steps = round(total_time / CONTROL_DT)
    assert output["time"].shape == (expected_steps + 1,)
    assert output["total_time"] == expected_steps * CONTROL_DT
    assert np.allclose(np.diff(output["time"]), CONTROL_DT, atol=1e-12)


def test_optimize_trajectory_accepts_coterminal_endpoint_yaw(ctrl):
    ctrl[0, 2] = -np.pi
    ctrl[1, 2] = np.pi
    output = optimize_trajectory(ctrl, None, None, total_time=1.0)
    assert np.isclose(output["yaw_rate"][0], 0.0)

"""Focused tests for work package 6.2: quintic clamped B-spline math kernel.

Every verification item in section 6.2 of
docs/trajectory_optimization_implementation_plan.md is covered, plus the
review-feedback validation requirements:

  - clamped knot construction
  - basis nonnegativity and partition of unity
  - exact endpoint interpolation
  - zero-endpoint translational velocity and yaw-rate control-point relationship
  - analytic 1/2/3 derivatives vs finite differences
  - T scaling 1/2, 1/4, 1/8
  - jerk squared integral: 3-point Gauss-Legendre per span vs independent
    high-accuracy reference, and T^-5 scaling (1/32 for T -> 2T)
  - yaw unwrap (170 -> -170 becomes 170 -> 190 internally) and output wrap
  - validation: invalid T, invalid shapes / nonfinite control points,
    derivative orders outside [0, degree], phase entry refusing missing
    total_time, nonzero endpoint derivatives
"""

import numpy as np
import pytest
from scipy.interpolate import BSpline
from scipy.integrate import quad

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


def _valid_ctrl(rng, n: int = 8) -> np.ndarray:
    """Control points that satisfy the endpoint zero-velocity relationships."""
    c = rng.standard_normal((n, 3))
    c[1] = c[0]
    c[-2] = c[-1]
    return c


@pytest.fixture
def rng():
    return np.random.default_rng(20260720)


@pytest.fixture
def ctrl(rng):
    return _valid_ctrl(rng)


# -- knot construction -----------------------------------------------------


def test_clamped_knots_endpoints_and_interior():
    knots = clamped_knots(8, SPLINE_DEGREE)
    assert knots[0] == 0.0
    assert knots[-1] == 1.0
    assert np.all(knots[: SPLINE_DEGREE + 1] == 0.0)
    assert np.all(knots[-(SPLINE_DEGREE + 1) :] == 1.0)
    assert knots.shape[0] == 8 + SPLINE_DEGREE + 1
    interior = knots[SPLINE_DEGREE + 1 : -SPLINE_DEGREE - 1]
    if interior.size:
        assert np.all(interior > 0.0)
        assert np.all(interior < 1.0)
        diffs = np.diff(interior)
        assert np.allclose(diffs, diffs[0])


def test_clamped_knots_minimum_control_points():
    knots = clamped_knots(SPLINE_DEGREE + 1, SPLINE_DEGREE)
    assert knots.shape[0] == 2 * (SPLINE_DEGREE + 1)
    assert np.all(knots[: SPLINE_DEGREE + 1] == 0.0)
    assert np.all(knots[-SPLINE_DEGREE - 1 :] == 1.0)


def test_clamped_knots_too_few_raises():
    with pytest.raises(ValueError):
        clamped_knots(SPLINE_DEGREE, SPLINE_DEGREE)


# -- basis nonnegativity and partition of unity ----------------------------


def test_basis_nonnegativity_and_partition_of_unity():
    knots = clamped_knots(8, SPLINE_DEGREE)
    n_ctrl = 8
    spline = BSpline(knots, np.eye(n_ctrl), SPLINE_DEGREE, extrapolate=False)
    u = np.linspace(0.0, 1.0, 501)
    vals = spline(u)
    assert np.all(vals >= -1e-12)
    sums = vals.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-10)


# -- exact endpoint interpolation ------------------------------------------


def test_endpoint_interpolation(ctrl):
    spline = build_clamped_spline(ctrl, SPLINE_DEGREE)
    assert np.allclose(spline(0.0), ctrl[0], atol=1e-12)
    assert np.allclose(spline(1.0), ctrl[-1], atol=1e-12)


# -- zero endpoint velocity: control-point relationship --------------------


def test_zero_endpoint_velocity_control_point_relation():
    rng = np.random.default_rng(1)
    ctrl = rng.standard_normal((8, 3))
    ctrl[1] = ctrl[0]
    ev = eval_derivatives(ctrl, T=1.0, u=np.array([0.0]))
    assert np.allclose(ev["u_velocity"][0], 0.0, atol=1e-12)
    assert np.allclose(ev["velocity"][0], 0.0, atol=1e-12)


def test_zero_endpoint_velocity_at_end():
    rng = np.random.default_rng(2)
    ctrl = rng.standard_normal((8, 3))
    ctrl[-2] = ctrl[-1]
    ev = eval_derivatives(ctrl, T=1.0, u=np.array([1.0]))
    assert np.allclose(ev["u_velocity"][0], 0.0, atol=1e-12)


# -- analytic derivatives vs finite differences ----------------------------


@pytest.mark.parametrize("order,T", [(1, 1.0), (2, 1.0), (3, 1.0), (1, 2.5), (3, 2.5)])
def test_analytic_vs_finite_diff(ctrl, order, T):
    spline = build_clamped_spline(ctrl, SPLINE_DEGREE)
    d = spline.derivative(order)
    u0 = np.array([0.35])
    h = {1: 1e-6, 2: 1e-4, 3: 1e-3}[order]
    if order == 1:
        fd = (spline(u0 + h) - spline(u0 - h)) / (2 * h)
    elif order == 2:
        fd = (spline(u0 + h) - 2 * spline(u0) + spline(u0 - h)) / (h * h)
    else:
        fd = (
            spline(u0 + 2 * h)
            - 2 * spline(u0 + h)
            + 2 * spline(u0 - h)
            - spline(u0 - 2 * h)
        ) / (2 * h**3)
    analytic = d(u0)
    assert np.allclose(analytic, fd, rtol=1e-4, atol=1e-4)
    ev = eval_derivatives(ctrl, T=T, u=u0)
    key = {1: "velocity", 2: "acceleration", 3: "jerk"}[order]
    assert np.allclose(ev[key][0], analytic[0] / T**order, atol=1e-12)


# -- explicit derivative control points ------------------------------------


def test_derivative_control_points_match_scipy(ctrl):
    spline = build_clamped_spline(ctrl, SPLINE_DEGREE)
    for order in (0, 1, 2, 3):
        d_knots, c_explicit = derivative_control_points(
            spline.t, ctrl, SPLINE_DEGREE, order
        )
        scipy_spline = spline.derivative(order)
        assert np.allclose(d_knots, scipy_spline.t)
        d_spline = BSpline(d_knots, c_explicit, SPLINE_DEGREE - order, extrapolate=False)
        u = np.linspace(0.05, 0.95, 37)
        assert np.allclose(d_spline(u), scipy_spline(u), atol=1e-10)


def test_derivative_control_points_order_out_of_range(ctrl):
    spline = build_clamped_spline(ctrl, SPLINE_DEGREE)
    for bad in (-1, SPLINE_DEGREE + 1, SPLINE_DEGREE + 2):
        with pytest.raises(ValueError):
            derivative_control_points(spline.t, ctrl, SPLINE_DEGREE, bad)


# -- T scaling 1/2, 1/4, 1/8 -----------------------------------------------


def test_time_scaling_derivatives(ctrl):
    u = np.array([0.3, 0.7])
    ev1 = eval_derivatives(ctrl, T=1.0, u=u)
    ev2 = eval_derivatives(ctrl, T=2.0, u=u)
    assert np.allclose(ev1["position"], ev2["position"], atol=1e-12)
    assert np.allclose(ev2["velocity"], ev1["velocity"] / 2.0, atol=1e-12)
    assert np.allclose(ev2["acceleration"], ev1["acceleration"] / 4.0, atol=1e-12)
    assert np.allclose(ev2["jerk"], ev1["jerk"] / 8.0, atol=1e-12)


# -- jerk integral: 3-pt Gauss-Legendre per span vs independent reference ---


def _jerk_reference(ctrl: np.ndarray, T: float) -> float:
    """Independent high-accuracy reference using scipy.quad per dimension."""
    spline = build_clamped_spline(ctrl, SPLINE_DEGREE)
    d3 = spline.derivative(3)

    def integrand(u):
        return float(np.sum(d3(np.array([u]))[0] ** 2))

    integral_u, _ = quad(integrand, 0.0, 1.0, limit=200, epsabs=1e-12, epsrel=1e-12)
    return integral_u / T**5


def test_jerk_integral_matches_high_accuracy_reference(ctrl):
    T = 3.7
    assert np.isclose(jerk_integral_sq(ctrl, T), _jerk_reference(ctrl, T), rtol=1e-10)


def test_jerk_integral_t_scaling(ctrl):
    J1 = jerk_integral_sq(ctrl, T=1.0)
    J2 = jerk_integral_sq(ctrl, T=2.0)
    assert np.isclose(J2, J1 / 32.0, rtol=1e-12)


def test_jerk_integral_invalid_T(ctrl):
    for bad in (0.0, -1.0, np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError):
            jerk_integral_sq(ctrl, T=bad)


# -- yaw unwrap: 170 -> -170 becomes 170 -> 190 internally -----------------


def test_yaw_unwrap_170_to_minus170():
    yaw = np.array([np.deg2rad(170.0), np.deg2rad(-170.0)])
    unwrapped = yaw_unwrap(yaw)
    assert np.allclose(np.rad2deg(unwrapped), [170.0, 190.0])


def test_yaw_unwrap_adjacent_steps_absolute():
    # All inputs within [-180, 180). A forward step crossing the wrap boundary
    # must unwrap to a small positive step, not a large negative one. Checks
    # absolute adjacent differences and the absolute unwrapped values.
    deg = np.array([170.0, -170.0, 150.0, 170.0, -170.0, -150.0])
    yaw = np.deg2rad(deg)
    unwrapped = yaw_unwrap(yaw)
    diffs_deg = np.rad2deg(np.diff(unwrapped))
    expected_diffs = np.array([20.0, -40.0, 20.0, 20.0, 20.0])
    assert np.allclose(diffs_deg, expected_diffs, atol=1e-9)
    expected_unwrapped_deg = np.array([170.0, 190.0, 150.0, 170.0, 190.0, 210.0])
    assert np.allclose(np.rad2deg(unwrapped), expected_unwrapped_deg, atol=1e-9)
    # All adjacent unwrapped steps must have magnitude strictly below pi.
    assert np.all(np.abs(np.diff(unwrapped)) < np.pi)


def test_yaw_wrap_interval_is_closed_open():
    # [-pi, pi): +pi wraps to -pi, -pi stays -pi.
    wrapped = yaw_wrap(np.array([np.pi, -np.pi, np.pi - 1e-12, -np.pi + 1e-12]))
    assert np.isclose(wrapped[0], -np.pi)
    assert np.isclose(wrapped[1], -np.pi)
    assert np.isclose(wrapped[2], np.pi - 1e-12)
    assert np.isclose(wrapped[3], -np.pi + 1e-12)


def test_yaw_wrap_output_of_unwrap_190():
    unwrapped = np.array([np.deg2rad(190.0)])
    wrapped = yaw_wrap(unwrapped)
    assert np.allclose(np.rad2deg(wrapped), [-170.0])


# -- optimize_trajectory phase entry ---------------------------------------


def test_optimize_trajectory_requires_total_time(ctrl):
    with pytest.raises(NotImplementedError):
        optimize_trajectory(ctrl, safe_mask=None, esdf=None)


def test_optimize_trajectory_returns_expected_keys(rng):
    ctrl = _valid_ctrl(rng, n=8)
    out = optimize_trajectory(ctrl, safe_mask=None, esdf=None, total_time=2.0)
    for k in (
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
    ):
        assert k in out
    assert out["position_world"].shape[1] == 2
    n_steps = int(round(2.0 / CONTROL_DT))
    assert out["time"].shape[0] == n_steps + 1
    assert out["position_world"].shape[0] == n_steps + 1
    assert out["total_time"] == 2.0


def test_optimize_trajectory_yaw_wrapped_in_principal_interval(ctrl):
    out = optimize_trajectory(ctrl, safe_mask=None, esdf=None, total_time=1.0)
    yw = out["yaw_wrapped"]
    assert np.all(yw >= -np.pi - 1e-12)
    assert np.all(yw < np.pi + 1e-12)


def test_optimize_trajectory_endpoint_zero_velocity(ctrl):
    out = optimize_trajectory(ctrl, safe_mask=None, esdf=None, total_time=1.0)
    assert np.allclose(out["velocity_world"][0], 0.0, atol=1e-12)
    assert np.allclose(out["velocity_world"][-1], 0.0, atol=1e-12)
    assert np.allclose(out["yaw_rate"][0], 0.0, atol=1e-12)
    assert np.allclose(out["yaw_rate"][-1], 0.0, atol=1e-12)


# -- optimize_trajectory validation: invalid T -----------------------------


@pytest.mark.parametrize(
    "T",
    [0.0, -1.0, np.nan, np.inf, -np.inf, 0.05, 0.15, 1.0001, 0.9999],
)
def test_optimize_trajectory_invalid_T(ctrl, T):
    with pytest.raises(ValueError):
        optimize_trajectory(ctrl, safe_mask=None, esdf=None, total_time=T)


# -- optimize_trajectory validation: invalid shapes ------------------------


def test_optimize_trajectory_invalid_shape_1d(rng):
    bad = rng.standard_normal((10,))
    with pytest.raises(ValueError):
        optimize_trajectory(bad, safe_mask=None, esdf=None, total_time=1.0)


def test_optimize_trajectory_invalid_shape_wrong_columns(rng):
    bad = rng.standard_normal((10, 2))
    with pytest.raises(ValueError):
        optimize_trajectory(bad, safe_mask=None, esdf=None, total_time=1.0)


def test_optimize_trajectory_too_few_control_points(rng):
    bad = rng.standard_normal((5, 3))
    with pytest.raises(ValueError):
        optimize_trajectory(bad, safe_mask=None, esdf=None, total_time=1.0)


def test_optimize_trajectory_nonfinite_control_points(rng):
    bad = _valid_ctrl(rng, n=8)
    bad[3, 0] = np.nan
    with pytest.raises(ValueError):
        optimize_trajectory(bad, safe_mask=None, esdf=None, total_time=1.0)


def test_optimize_trajectory_nonfinite_inf(rng):
    bad = _valid_ctrl(rng, n=8)
    bad[4, 1] = np.inf
    with pytest.raises(ValueError):
        optimize_trajectory(bad, safe_mask=None, esdf=None, total_time=1.0)


# -- optimize_trajectory validation: nonzero endpoint derivatives -----------


def test_optimize_trajectory_rejects_nonzero_start_velocity(rng):
    bad = rng.standard_normal((8, 3))
    # P1 != P0
    with pytest.raises(ValueError):
        optimize_trajectory(bad, safe_mask=None, esdf=None, total_time=1.0)


def test_optimize_trajectory_rejects_nonzero_end_velocity(rng):
    bad = rng.standard_normal((8, 3))
    bad[1] = bad[0]
    # P[-2] != P[-1]
    with pytest.raises(ValueError):
        optimize_trajectory(bad, safe_mask=None, esdf=None, total_time=1.0)


def test_optimize_trajectory_rejects_nonzero_yaw_rate_endpoint(rng):
    bad = rng.standard_normal((8, 3))
    bad[1] = bad[0]
    bad[-2] = bad[-1]
    # Violate only the yaw component of the start relationship.
    bad[1, 2] = bad[0, 2] + 0.5
    with pytest.raises(ValueError):
        optimize_trajectory(bad, safe_mask=None, esdf=None, total_time=1.0)


# -- fixed-period grid canonicalization -----------------------------------


def test_optimize_trajectory_canonicalizes_float_repr_T():
    # 0.30000000000000004 is the classic float repr of 3 * 0.1; it must be
    # canonicalized to exactly n_steps * CONTROL_DT and the grid must have
    # uniform CONTROL_DT spacing within strict floating tolerance.
    T_input = 0.1 * 3  # 0.30000000000000004
    assert T_input != 0.3  # confirm the float repr issue is real
    out = optimize_trajectory(
        _valid_ctrl(np.random.default_rng(11), n=8),
        safe_mask=None,
        esdf=None,
        total_time=T_input,
    )
    assert out["total_time"] == 3 * CONTROL_DT
    diffs = np.diff(out["time"])
    # Strict floating tolerance: 1e-15 is ~1 ULP at the 0.1 scale, well above
    # the float64 accumulation noise of arange * dt for small step counts.
    assert np.allclose(diffs, CONTROL_DT, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("T_input", [1.0, 2.0, 5.0, 10.0, 0.7])
def test_optimize_trajectory_uniform_grid_spacing(ctrl, T_input):
    out = optimize_trajectory(ctrl, safe_mask=None, esdf=None, total_time=T_input)
    diffs = np.diff(out["time"])
    # Strict floating tolerance scaled to float64 accumulation noise (~1 ULP
    # per step at the 0.1 scale). 1e-12 covers grids up to ~10000 steps.
    assert np.allclose(diffs, CONTROL_DT, rtol=0.0, atol=1e-12)
    assert out["total_time"] == int(round(T_input / CONTROL_DT)) * CONTROL_DT


# -- coterminal yaw endpoints ----------------------------------------------


def test_optimize_trajectory_accepts_coterminal_yaw_endpoints():
    # P0 yaw = -pi, P1 yaw = +pi: coterminal. After unwrap they are on the same
    # branch and the endpoint zero-rate relationship must hold.
    rng = np.random.default_rng(21)
    ctrl = rng.standard_normal((8, 3))
    ctrl[1] = ctrl[0]
    ctrl[-2] = ctrl[-1]
    ctrl[0, 2] = -np.pi
    ctrl[1, 2] = np.pi  # coterminal with -pi
    out = optimize_trajectory(ctrl, safe_mask=None, esdf=None, total_time=1.0)
    assert np.allclose(out["yaw_rate"][0], 0.0, atol=1e-12)
    assert np.allclose(out["yaw_rate"][-1], 0.0, atol=1e-12)


def test_optimize_trajectory_rejects_noncoterminal_yaw_endpoints():
    # P0 yaw = 0, P1 yaw = 0.5: not coterminal, not equal after unwrap.
    rng = np.random.default_rng(22)
    bad = rng.standard_normal((8, 3))
    bad[1] = bad[0]
    bad[-2] = bad[-1]
    bad[0, 2] = 0.0
    bad[1, 2] = 0.5
    with pytest.raises(ValueError):
        optimize_trajectory(bad, safe_mask=None, esdf=None, total_time=1.0)


def test_eval_derivatives_accepts_boundary_u(ctrl):
    # u=0 and u=1 are valid boundary values.
    ev = eval_derivatives(ctrl, T=1.0, u=np.array([0.0, 1.0]))
    assert ev["position"].shape == (2, 3)


# -- combined jerk integral == translation + yaw ---------------------------


def test_jerk_integral_combined_equals_translation_plus_yaw(rng):
    ctrl = _valid_ctrl(rng, n=10)
    T = 2.0
    combined = jerk_integral_sq(ctrl, T)
    translation = jerk_integral_sq(ctrl[:, :2], T)
    yaw = jerk_integral_sq(ctrl[:, 2:3], T)
    assert np.isclose(combined, translation + yaw, rtol=1e-12, atol=0.0)


# -- tiny-positive-T regression --------------------------------------------


def test_optimize_trajectory_rejects_tiny_positive_T(ctrl):
    # T values that round to n_steps=0 must not become canonical T=0.
    with pytest.raises(ValueError):
        optimize_trajectory(
            ctrl, safe_mask=None, esdf=None, total_time=1e-12
        )

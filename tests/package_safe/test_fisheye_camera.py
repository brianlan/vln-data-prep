"""Unit tests for ``fisheye_camera.py``.

These cover the calibration edge cases called out by the Phase 0a scope:
invalid coefficient arity, non-finite coefficients, invalid width/FOV, the
fisheye polynomial round-trip, forward-mask radius, vertical-FOV derivation,
and the bisection inversion bounds. ``fisheye_camera.py`` is package-safe
(stdlib ``math`` only) so these tests run under ``$SAGE3D_PACKAGE_PYTHON``.
"""

from __future__ import annotations

import math

import pytest

from fisheye_camera import (
    angle_for_image_radius,
    distort_theta,
    focal_length_for_horizontal_fov,
    opencv_fisheye_parameters,
    validate_fisheye_coefficients,
)


# --- validate_fisheye_coefficients -------------------------------------------

def test_validate_coefficients_accepts_four_finite():
    assert validate_fisheye_coefficients([0.1, 0.0, 0.0, 0.0]) == (0.1, 0.0, 0.0, 0.0)


def test_validate_coefficients_rejects_wrong_arity():
    for bad in ([], [0.1], [0.1, 0.0], [0.1, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0, 0.0]):
        with pytest.raises(ValueError, match="exactly four"):
            validate_fisheye_coefficients(bad)


@pytest.mark.parametrize("bad_value", [math.inf, -math.inf, math.nan])
def test_validate_coefficients_rejects_non_finite(bad_value):
    with pytest.raises(ValueError, match="finite"):
        validate_fisheye_coefficients([0.1, 0.0, 0.0, bad_value])


def test_validate_coefficients_coerces_ints_and_strings():
    # float() coercion: ints and numeric strings are accepted.
    assert validate_fisheye_coefficients((0, 0, 0, 0)) == (0.0, 0.0, 0.0, 0.0)
    assert validate_fisheye_coefficients(("0.1", "0", "0", "0")) == (
        0.1,
        0.0,
        0.0,
        0.0,
    )


# --- distort_theta -----------------------------------------------------------

def test_distort_theta_identity_with_zero_coefficients():
    assert distort_theta(0.5, [0.0, 0.0, 0.0, 0.0]) == 0.5
    assert distort_theta(0.0, [0.1, 0.0, 0.0, 0.0]) == 0.0


def test_distort_theta_monotonic_for_typical_coefficients():
    k = [0.1, 0.0, 0.0, 0.0]
    prev = -1.0
    for theta in [0.0, 0.1, 0.2, 0.5, 1.0, 1.5]:
        value = distort_theta(theta, k)
        assert value > prev
        prev = value


def test_distort_theta_matches_manual_polynomial():
    theta = 0.7
    k1, k2, k3, k4 = 0.1, 0.02, 0.0, 0.0
    expected = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
    assert math.isclose(distort_theta(theta, [k1, k2, k3, k4]), expected)


# --- focal_length_for_horizontal_fov -----------------------------------------

def test_focal_length_positive_for_valid_fov():
    focal = focal_length_for_horizontal_fov(600, 180.0, [0.1, 0.0, 0.0, 0.0])
    assert focal > 0.0


def test_focal_length_rejects_nonpositive_width():
    for width in (0, -1):
        with pytest.raises(ValueError, match="width must be positive"):
            focal_length_for_horizontal_fov(width, 180.0, [0.1, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("bad_fov", [0.0, -1.0, 360.001])
def test_focal_length_rejects_invalid_fov(bad_fov):
    with pytest.raises(ValueError, match="Horizontal FOV"):
        focal_length_for_horizontal_fov(600, bad_fov, [0.1, 0.0, 0.0, 0.0])


def test_focal_length_rejects_nonpositive_distorted_half_angle():
    # A coefficient set that drives the distorted half-angle non-positive at
    # the requested FOV edge is rejected.
    with pytest.raises(ValueError, match="non-positive"):
        focal_length_for_horizontal_fov(600, 180.0, [-1.0, 0.0, 0.0, 0.0])


# --- angle_for_image_radius --------------------------------------------------

def test_angle_for_image_radius_round_trips():
    k = [0.1, 0.0, 0.0, 0.0]
    focal = focal_length_for_horizontal_fov(600, 180.0, k)
    # The distorted half-angle should invert back to the original half-angle.
    half_angle = math.radians(180.0) / 2.0
    radius = focal * distort_theta(half_angle, k)
    recovered = angle_for_image_radius(radius, focal, k)
    assert math.isclose(recovered, half_angle, abs_tol=1e-9)


def test_angle_for_image_radius_rejects_negative_radius():
    with pytest.raises(ValueError, match="Radius must be non-negative"):
        angle_for_image_radius(-1.0, 300.0, [0.1, 0.0, 0.0, 0.0])


def test_angle_for_image_radius_rejects_nonpositive_focal():
    with pytest.raises(ValueError, match="focal length positive"):
        angle_for_image_radius(10.0, 0.0, [0.1, 0.0, 0.0, 0.0])


def test_angle_for_image_radius_rejects_out_of_range():
    # A radius larger than the model can reach at max_angle is rejected.
    k = [0.1, 0.0, 0.0, 0.0]
    focal = focal_length_for_horizontal_fov(600, 180.0, k)
    huge_radius = focal * distort_theta(math.pi, k) + 1.0
    with pytest.raises(ValueError, match="outside the fisheye model"):
        angle_for_image_radius(huge_radius, focal, k)


def test_angle_for_image_radius_zero_radius_returns_zero():
    # Bisection converges to within the 80-iteration tolerance of zero.
    assert math.isclose(
        angle_for_image_radius(0.0, 300.0, [0.1, 0.0, 0.0, 0.0]),
        0.0,
        abs_tol=1e-12,
    )


# --- opencv_fisheye_parameters -----------------------------------------------

def test_opencv_fisheye_parameters_structure_and_values():
    params = opencv_fisheye_parameters(600, 450, 180.0, [0.1, 0.0, 0.0, 0.0])
    assert params["cx"] == 300.0
    assert params["cy"] == 225.0
    assert params["fx"] == params["fy"]
    assert params["horizontal_fov_deg"] == 180.0
    assert params["vertical_fov_deg"] > 0.0
    assert params["forward_mask_radius_pixels"] > 0.0
    assert params["fisheye_coefficients"] == [0.1, 0.0, 0.0, 0.0]


def test_opencv_fisheye_parameters_rejects_nonpositive_height():
    for height in (0, -1):
        with pytest.raises(ValueError, match="height must be positive"):
            opencv_fisheye_parameters(600, height, 180.0, [0.1, 0.0, 0.0, 0.0])


def test_opencv_fisheye_parameters_vertical_fov_larger_for_tall_image():
    # A taller image has a larger image radius, so the inverted half-angle (and
    # thus vertical FOV) grows rather than shrinking.
    k = [0.1, 0.0, 0.0, 0.0]
    short = opencv_fisheye_parameters(600, 300, 180.0, k)["vertical_fov_deg"]
    tall = opencv_fisheye_parameters(600, 1200, 180.0, k)["vertical_fov_deg"]
    assert tall > short


def test_opencv_fisheye_parameters_focal_matches_focal_length_helper():
    k = [0.1, 0.0, 0.0, 0.0]
    params = opencv_fisheye_parameters(600, 450, 180.0, k)
    direct = focal_length_for_horizontal_fov(600, 180.0, k)
    assert math.isclose(params["fx"], direct)


def test_opencv_fisheye_parameters_invalid_coefficients_propagate():
    with pytest.raises(ValueError, match="exactly four"):
        opencv_fisheye_parameters(600, 450, 180.0, [0.1, 0.0, 0.0])
import math
import sys
from pathlib import Path

import pytest


RMPFLOW_DIR = (
    Path(__file__).parents[3]
    / "isaacpjt"
    / "M0609"
    / "rmpflow"
)
sys.path.insert(0, str(RMPFLOW_DIR))

from fine_alignment_policy import (  # noqa: E402
    BUSBAR_HOLE_LOCAL_BOTTOM_CENTERS_M,
    BUSBAR_HOLE_LOCAL_CENTERS_M,
    BoundedPlanarCommandLead,
    FineAlignmentStepGate,
    busbar_bore_unavailable_gate,
    extract_local_z_world_axis,
    planar_command_response,
    planar_yaw_delta,
    project_world_axis_to_z,
    solve_busbar_bore_alignment,
    solve_busbar_hole_alignment,
    transformed_usd_points_bounds,
)


def _usd_planar_transform(
    *,
    scale=1.0,
    yaw_rad=0.0,
    translation=(0.0, 0.0, 0.0),
):
    """Return a USD/Gf-style row-vector scale/rotate/translate matrix."""
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return (
        (scale * cosine, scale * sine, 0.0, 0.0),
        (-scale * sine, scale * cosine, 0.0, 0.0),
        (0.0, 0.0, scale, 0.0),
        (translation[0], translation[1], translation[2], 1.0),
    )


def _usd_y_tilt_transform(
    *,
    tilt_rad,
    translation=(0.0, 0.0, 0.0),
):
    """Return a row-vector affine whose local Z tilts toward world +X."""
    cosine = math.cos(tilt_rad)
    sine = math.sin(tilt_rad)
    return (
        (cosine, 0.0, -sine, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (sine, 0.0, cosine, 0.0),
        (translation[0], translation[1], translation[2], 1.0),
    )


def test_local_z_world_axis_projects_to_requested_height_with_full_affine():
    tilt_rad = math.radians(2.0)
    transform = _usd_y_tilt_transform(
        tilt_rad=tilt_rad,
        translation=(1.0, -0.5, 0.13),
    )

    axis = extract_local_z_world_axis(transform)
    projected = project_world_axis_to_z(axis, 0.15)

    assert axis.origin_m == pytest.approx((1.0, -0.5, 0.13))
    assert axis.tilt_rad == pytest.approx(tilt_rad)
    assert projected == pytest.approx((
        1.0 + 0.02 * math.tan(tilt_rad),
        -0.5,
        0.15,
    ))


@pytest.mark.parametrize(
    "transform",
    [
        _usd_y_tilt_transform(tilt_rad=math.radians(3.01)),
        _usd_y_tilt_transform(tilt_rad=math.radians(89.99999999)),
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, float("nan"), 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    ],
)
def test_local_z_world_axis_rejects_excessive_or_invalid_axes(transform):
    with pytest.raises(ValueError):
        extract_local_z_world_axis(transform)


def test_busbar_bore_alignment_pairs_axes_and_checks_top_and_bottom_planes():
    scale = 0.75
    translation = (1.1, -0.4, 0.15)
    mesh_transform = _usd_planar_transform(
        scale=scale,
        translation=translation,
    )
    top_holes = tuple(
        (
            translation[0] + scale * local_x,
            translation[1] + scale * local_y,
            translation[2] + scale * local_z,
        )
        for local_x, local_y, local_z in BUSBAR_HOLE_LOCAL_CENTERS_M
    )
    tilt_rad = math.radians(2.0)
    bolt_origin_z = 0.13
    bolt_transforms = []
    for hole in top_holes:
        origin_x = hole[0] - (
            hole[2] - bolt_origin_z
        ) * math.tan(tilt_rad)
        bolt_transforms.append(_usd_y_tilt_transform(
            tilt_rad=tilt_rad,
            translation=(origin_x, hole[1], bolt_origin_z),
        ))

    observation = solve_busbar_bore_alignment(
        mesh_transform,
        tuple(reversed(bolt_transforms)),
    )

    assert observation.alignment.swapped
    assert observation.paired_axis_indices == (1, 0)
    assert observation.top_endpoint_residuals_m == pytest.approx(
        (0.0, 0.0),
        abs=1.0e-12,
    )
    expected_bottom_residual = (
        scale
        * abs(
            BUSBAR_HOLE_LOCAL_CENTERS_M[0][2]
            - BUSBAR_HOLE_LOCAL_BOTTOM_CENTERS_M[0][2]
        )
        * math.tan(tilt_rad)
    )
    assert observation.bottom_endpoint_residuals_m == pytest.approx(
        (expected_bottom_residual, expected_bottom_residual),
    )
    assert observation.max_bore_endpoint_residual_m < 0.00075


def test_authored_point_bounds_transform_real_points_and_reject_invalid_data():
    transform = _usd_y_tilt_transform(
        tilt_rad=math.radians(2.0),
        translation=(1.0, 2.0, 3.0),
    )
    local_points = (
        (-0.2, -0.1, -0.003),
        (0.2, 0.1, 0.003),
    )

    world_min, world_max = transformed_usd_points_bounds(
        transform,
        local_points,
    )
    transformed = []
    for x_m, y_m, z_m in local_points:
        transformed.append((
            1.0 + math.cos(math.radians(2.0)) * x_m
            + math.sin(math.radians(2.0)) * z_m,
            2.0 + y_m,
            3.0 - math.sin(math.radians(2.0)) * x_m
            + math.cos(math.radians(2.0)) * z_m,
        ))
    assert world_min == pytest.approx(tuple(
        min(point[index] for point in transformed)
        for index in range(3)
    ))
    assert world_max == pytest.approx(tuple(
        max(point[index] for point in transformed)
        for index in range(3)
    ))

    with pytest.raises(ValueError):
        transformed_usd_points_bounds(transform, ())
    with pytest.raises(ValueError):
        transformed_usd_points_bounds(
            transform,
            ((0.0, float("nan"), 0.0),),
        )
    with pytest.raises(ValueError):
        transformed_usd_points_bounds(
            transform,
            ((10 ** 10000, 0.0, 0.0),),
        )


def test_busbar_bore_unavailable_gate_requires_stable_consecutive_samples():
    kwargs = {
        "alignment_tolerance_m": 0.00075,
        "required_steps": 3,
    }

    count, unavailable = busbar_bore_unavailable_gate(
        (0.0002, 0.0003),
        (0.0008, 0.0009),
        0,
        observation_stable=False,
        **kwargs,
    )
    assert (count, unavailable) == (0, False)

    for expected_count in (1, 2):
        count, unavailable = busbar_bore_unavailable_gate(
            (0.0002, 0.0003),
            (0.0008, 0.0009),
            count,
            observation_stable=True,
            **kwargs,
        )
        assert (count, unavailable) == (expected_count, False)

    count, unavailable = busbar_bore_unavailable_gate(
        (0.0002, 0.0003),
        (0.0008, 0.0009),
        count,
        observation_stable=True,
        **kwargs,
    )
    assert (count, unavailable) == (3, True)

    # A correctable top plane or a bottom plane inside tolerance resets the
    # liveness evidence instead of inheriting a stale failure count.
    assert busbar_bore_unavailable_gate(
        (0.0008, 0.0003),
        (0.0009, 0.0009),
        count,
        observation_stable=True,
        **kwargs,
    ) == (0, False)
    assert busbar_bore_unavailable_gate(
        (0.0002, 0.0003),
        (0.0007, 0.0007),
        count,
        observation_stable=True,
        **kwargs,
    ) == (0, False)

    with pytest.raises(ValueError):
        busbar_bore_unavailable_gate(
            (0.0, float("nan")),
            (0.0, 0.0),
            0,
            observation_stable=True,
            **kwargs,
        )


def test_busbar_hole_alignment_selects_direct_pair_and_midpoint_correction():
    transform = _usd_planar_transform(translation=(1.0, -2.0, 0.4))
    correction = (0.012, -0.008)
    local_a, local_b = BUSBAR_HOLE_LOCAL_CENTERS_M
    bolts = (
        (
            1.0 + local_a[0] + correction[0],
            -2.0 + local_a[1] + correction[1],
        ),
        (
            1.0 + local_b[0] + correction[0],
            -2.0 + local_b[1] + correction[1],
        ),
    )

    result = solve_busbar_hole_alignment(transform, bolts)

    assert result.pairing == "direct"
    assert not result.swapped
    assert result.hole_midpoint_xy_m == pytest.approx((1.0, -2.0))
    assert result.bolt_midpoint_xy_m == pytest.approx(
        (1.0 + correction[0], -2.0 + correction[1])
    )
    assert result.desired_rigid_xy_correction_m == pytest.approx(
        correction
    )
    assert result.midpoint_residual_m == pytest.approx(
        math.hypot(*correction)
    )
    assert result.yaw_residual_rad == pytest.approx(0.0)
    assert result.max_endpoint_residual_m == pytest.approx(0.0, abs=1e-12)


def test_busbar_hole_alignment_selects_swapped_unordered_bolt_axes():
    yaw = math.radians(23.0)
    translation = (-0.3, 0.8, 0.2)
    transform = _usd_planar_transform(
        yaw_rad=yaw,
        translation=(-0.3, 0.8, 0.2),
    )
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    holes = tuple(
        (
            translation[0] + cosine * local_x - sine * local_y,
            translation[1] + sine * local_x + cosine * local_y,
            translation[2] + local_z,
        )
        for local_x, local_y, local_z
        in BUSBAR_HOLE_LOCAL_CENTERS_M
    )

    result = solve_busbar_hole_alignment(
        transform,
        (
            (holes[1][0], holes[1][1]),
            (holes[0][0], holes[0][1]),
        ),
    )

    assert result.pairing == "swapped"
    assert result.swapped
    assert result.paired_bolt_axes_xy_m[0] == pytest.approx(holes[0][:2])
    assert result.paired_bolt_axes_xy_m[1] == pytest.approx(holes[1][:2])
    assert result.desired_rigid_xy_correction_m == pytest.approx((0.0, 0.0))
    assert result.yaw_residual_rad == pytest.approx(0.0)
    assert result.max_endpoint_residual_m == pytest.approx(0.0, abs=1e-12)


def test_busbar_hole_alignment_applies_full_scale_rotation_and_span_residual():
    scale = 0.75
    yaw = math.radians(-90.0)
    translation = (1.2, -0.9, 0.25)
    transform = _usd_planar_transform(
        scale=scale,
        yaw_rad=yaw,
        translation=translation,
    )
    expected_holes = []
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    for local_x, local_y, local_z in BUSBAR_HOLE_LOCAL_CENTERS_M:
        expected_holes.append((
            translation[0] + scale * (
                cosine * local_x - sine * local_y
            ),
            translation[1] + scale * (
                sine * local_x + cosine * local_y
            ),
            translation[2] + scale * local_z,
        ))

    exact = solve_busbar_hole_alignment(
        transform,
        tuple((point[0], point[1]) for point in expected_holes),
    )

    for actual, expected in zip(
        exact.hole_world_points_m,
        expected_holes,
    ):
        assert actual == pytest.approx(expected)
    transformed_span = math.dist(
        exact.hole_world_points_m[0][:2],
        exact.hole_world_points_m[1][:2],
    )
    local_span = math.dist(
        BUSBAR_HOLE_LOCAL_CENTERS_M[0][:2],
        BUSBAR_HOLE_LOCAL_CENTERS_M[1][:2],
    )
    assert transformed_span == pytest.approx(scale * local_span)
    assert exact.max_endpoint_residual_m == pytest.approx(0.0, abs=1e-12)

    unscaled_bolts = []
    for local_x, local_y, _ in BUSBAR_HOLE_LOCAL_CENTERS_M:
        unscaled_bolts.append((
            translation[0] + cosine * local_x - sine * local_y,
            translation[1] + sine * local_x + cosine * local_y,
        ))
    scale_mismatch = solve_busbar_hole_alignment(
        transform,
        unscaled_bolts,
    )
    expected_endpoint_residual = (
        (1.0 - scale)
        * math.hypot(
            BUSBAR_HOLE_LOCAL_CENTERS_M[0][0],
            BUSBAR_HOLE_LOCAL_CENTERS_M[0][1],
        )
    )
    assert scale_mismatch.max_endpoint_residual_m == pytest.approx(
        expected_endpoint_residual
    )


def test_step_ack_requires_six_stable_pose_observations():
    gate = FineAlignmentStepGate()
    gate.begin(123)

    for expected_count in range(1, 6):
        status = gate.update(
            position_error_m=0.004,
            orientation_error_rad=math.radians(2.0),
            motion_m=0.00009,
            orientation_motion_rad=math.radians(0.004),
            position_response_m=0.001,
            orientation_response_rad=math.radians(0.1),
        )
        assert status.stamp_ns == 123
        assert status.settle_count == expected_count
        assert not status.settled

    status = gate.update(
        position_error_m=0.004,
        orientation_error_rad=math.radians(2.0),
        motion_m=0.00009,
        orientation_motion_rad=math.radians(0.004),
        position_response_m=0.001,
        orientation_response_rad=math.radians(0.1),
    )
    assert status.stamp_ns == 123
    assert status.settled
    assert gate.active_stamp_ns is None


@pytest.mark.parametrize(
    (
        "position_error_m",
        "orientation_error_rad",
        "motion_m",
        "orientation_motion_rad",
    ),
    [
        (0.00501, 0.0, 0.0, 0.0),
        (0.0, math.radians(3.01), 0.0, 0.0),
        (0.0, 0.0, 0.000101, 0.0),
        (0.0, 0.0, 0.0, math.radians(0.0051)),
    ],
)
def test_any_out_of_tolerance_observation_restarts_settle_count(
    position_error_m,
    orientation_error_rad,
    motion_m,
    orientation_motion_rad,
):
    gate = FineAlignmentStepGate(settle_steps=2)
    gate.begin(456)
    assert gate.update(
        position_error_m=0.0,
        orientation_error_rad=0.0,
        motion_m=0.0,
        orientation_motion_rad=0.0,
        position_response_m=0.0,
        orientation_response_rad=0.0,
    ).settle_count == 1

    status = gate.update(
        position_error_m=position_error_m,
        orientation_error_rad=orientation_error_rad,
        motion_m=motion_m,
        orientation_motion_rad=orientation_motion_rad,
        position_response_m=0.0,
        orientation_response_rad=0.0,
    )
    assert status.settle_count == 0
    assert not status.settled


def test_gate_rejects_overlapping_or_invalid_steps():
    gate = FineAlignmentStepGate()
    with pytest.raises(ValueError):
        gate.begin(0)

    gate.begin(1)
    with pytest.raises(RuntimeError):
        gate.begin(2)
    gate.reset()
    assert not gate.update(
        position_error_m=0.0,
        orientation_error_rad=0.0,
        motion_m=0.0,
        orientation_motion_rad=0.0,
        position_response_m=0.0,
        orientation_response_rad=0.0,
    ).settled


def test_step_cannot_ack_until_commanded_response_is_observed():
    gate = FineAlignmentStepGate(settle_steps=2)
    gate.begin(
        10,
        required_position_response_m=0.00002,
        required_orientation_response_rad=math.radians(0.001),
    )

    for _ in range(5):
        status = gate.update(
            position_error_m=0.0,
            orientation_error_rad=0.0,
            motion_m=0.0,
            orientation_motion_rad=0.0,
            position_response_m=0.000019,
            orientation_response_rad=math.radians(0.0009),
        )
        assert status.settle_count == 0
        assert not status.settled

    assert gate.update(
        position_error_m=0.0,
        orientation_error_rad=0.0,
        motion_m=0.0,
        orientation_motion_rad=0.0,
        position_response_m=0.00002,
        orientation_response_rad=math.radians(0.001),
    ).settle_count == 1
    assert gate.update(
        position_error_m=0.0,
        orientation_error_rad=0.0,
        motion_m=0.0,
        orientation_motion_rad=0.0,
        position_response_m=0.00002,
        orientation_response_rad=math.radians(0.001),
    ).settled


def test_subquantized_position_step_acks_after_strict_stationary_settle():
    gate = FineAlignmentStepGate(
        position_response_noise_margin_m=0.0001,
        orientation_response_noise_margin_rad=math.radians(0.005),
        settle_steps=2,
    )
    gate.begin(
        20,
        required_position_response_m=0.00016,
        required_orientation_response_rad=0.0,
    )

    # 0.2 mm * 80% - 0.1 mm margin = 0.06 mm.  That is below the
    # 0.1 mm per-tick settle quantum, so asking for a measured movement would
    # be a false proof.  The non-zero correction still needs both stable
    # in-tolerance observations before the executor acknowledges it.
    assert gate.required_position_response_m == pytest.approx(0.0)
    for expected_count in (1, 2):
        status = gate.update(
            position_error_m=0.003,
            orientation_error_rad=math.radians(1.0),
            motion_m=0.0,
            orientation_motion_rad=0.0,
            position_response_m=0.0,
            orientation_response_rad=0.0,
        )
        assert status.settle_count == expected_count
    assert status.settled


def test_subquantized_step_uses_bounded_controller_jitter_profile_only():
    gate = FineAlignmentStepGate(
        motion_tolerance_m=0.0001,
        orientation_motion_tolerance_rad=math.radians(0.005),
        unobservable_motion_tolerance_m=0.0002,
        unobservable_orientation_motion_tolerance_rad=math.radians(0.1),
        position_response_noise_margin_m=0.0001,
        orientation_response_noise_margin_rad=math.radians(0.005),
        settle_steps=2,
    )
    # 0.2 mm at 80% becomes a 0.06 mm proof after the 0.1 mm margin,
    # below the 0.1 mm strict stationarity resolution.  It is a valid
    # sub-quantized/no-op confirmation, never an observable-motion ACK.
    gate.begin(
        22,
        required_position_response_m=0.00016,
        required_orientation_response_rad=math.radians(0.004),
    )
    assert gate.required_position_response_m == 0.0
    assert gate.required_orientation_response_rad == 0.0
    assert gate.using_unobservable_motion_tolerance

    for expected_count in (1, 2):
        status = gate.update(
            position_error_m=0.0034,
            orientation_error_rad=math.radians(1.0),
            motion_m=0.00015,
            orientation_motion_rad=math.radians(0.05),
            position_response_m=0.0002,
            orientation_response_rad=0.0,
        )
        assert status.settle_count == expected_count
    assert status.settled


def test_observable_step_keeps_the_strict_jitter_profile():
    gate = FineAlignmentStepGate(
        motion_tolerance_m=0.0001,
        orientation_motion_tolerance_rad=math.radians(0.005),
        unobservable_motion_tolerance_m=0.0002,
        unobservable_orientation_motion_tolerance_rad=math.radians(0.1),
        position_response_noise_margin_m=0.0001,
        settle_steps=2,
    )
    # The effective 0.101 mm response remains observable, so the looser
    # no-op profile must not be selected.
    gate.begin(23, required_position_response_m=0.000201)
    assert gate.required_position_response_m == pytest.approx(0.000101)
    assert not gate.using_unobservable_motion_tolerance

    status = gate.update(
        position_error_m=0.003,
        orientation_error_rad=0.0,
        motion_m=0.00015,
        orientation_motion_rad=math.radians(0.05),
        position_response_m=0.000101,
        orientation_response_rad=0.0,
    )
    assert status.settle_count == 0
    assert not status.settled


def test_response_above_settle_quantization_remains_nonzero_validated():
    gate = FineAlignmentStepGate(
        position_response_noise_margin_m=0.0001,
        settle_steps=2,
    )
    gate.begin(
        21,
        required_position_response_m=0.000201,
    )

    # The effective 0.101 mm response is just above the 0.1 mm stationary
    # quantization, so a same-frame/no-motion ACK remains forbidden.
    assert gate.required_position_response_m == pytest.approx(0.000101)
    for _ in range(3):
        status = gate.update(
            position_error_m=0.003,
            orientation_error_rad=0.0,
            motion_m=0.0,
            orientation_motion_rad=0.0,
            position_response_m=0.0,
            orientation_response_rad=0.0,
        )
        assert status.settle_count == 0
        assert not status.settled

    assert gate.update(
        position_error_m=0.003,
        orientation_error_rad=0.0,
        motion_m=0.0,
        orientation_motion_rad=0.0,
        position_response_m=0.000101,
        orientation_response_rad=0.0,
    ).settle_count == 1
    assert gate.update(
        position_error_m=0.003,
        orientation_error_rad=0.0,
        motion_m=0.0,
        orientation_motion_rad=0.0,
        position_response_m=0.000101,
        orientation_response_rad=0.0,
    ).settled


def test_larger_step_cannot_ack_same_frame_or_without_meaningful_response():
    gate = FineAlignmentStepGate(
        position_response_noise_margin_m=0.0001,
        orientation_response_noise_margin_rad=math.radians(0.005),
        settle_steps=2,
    )
    gate.begin(
        30,
        required_position_response_m=0.0008,
        required_orientation_response_rad=0.0,
    )

    for response in (0.0, 0.000699):
        status = gate.update(
            position_error_m=0.003,
            orientation_error_rad=0.0,
            motion_m=0.0,
            orientation_motion_rad=0.0,
            position_response_m=response,
            orientation_response_rad=0.0,
        )
        assert status.settle_count == 0
        assert not status.settled

    assert gate.update(
        position_error_m=0.003,
        orientation_error_rad=0.0,
        motion_m=0.0,
        orientation_motion_rad=0.0,
        position_response_m=0.0007,
        orientation_response_rad=0.0,
    ).settle_count == 1
    assert gate.update(
        position_error_m=0.003,
        orientation_error_rad=0.0,
        motion_m=0.0,
        orientation_motion_rad=0.0,
        position_response_m=0.0007,
        orientation_response_rad=0.0,
    ).settled


def test_controller_lead_grows_to_a_one_millimetre_norm_bound():
    lead = BoundedPlanarCommandLead(
        growth_step_m=0.00005,
        max_lead_m=0.001,
    )
    lead.begin(0.0002, 0.0)

    for step in range(1, 21):
        status = lead.advance(
            observed_response_m=0.0,
            required_response_m=0.00006,
        )
        assert status.magnitude_m == pytest.approx(step * 0.00005)
        assert status.grew
        assert status.saturated == (step == 20)

    saturated = lead.advance(
        observed_response_m=0.0,
        required_response_m=0.00006,
    )
    assert saturated.magnitude_m == pytest.approx(0.001)
    assert not saturated.grew
    assert saturated.saturated


def test_controller_lead_uses_only_delta_direction_and_logical_copy():
    lead = BoundedPlanarCommandLead()
    lead.begin(-0.0003, 0.0004)
    status = lead.advance(
        observed_response_m=0.0,
        required_response_m=0.0001,
    )

    assert status.x_m == pytest.approx(-0.00003)
    assert status.y_m == pytest.approx(0.00004)
    assert status.magnitude_m == pytest.approx(0.00005)
    logical = (1.0, 2.0)
    controller = lead.command_xy(*logical)
    assert logical == (1.0, 2.0)
    assert controller == pytest.approx(
        (1.0 - 0.00003, 2.0 + 0.00004)
    )


def test_controller_lead_resumes_if_response_drops_before_settle():
    lead = BoundedPlanarCommandLead()
    lead.begin(0.0002, 0.0)
    for _ in range(3):
        lead.advance(
            observed_response_m=0.00001,
            required_response_m=0.00006,
        )

    met = lead.advance(
        observed_response_m=0.00006,
        required_response_m=0.00006,
    )
    assert met.magnitude_m == pytest.approx(0.00015)
    assert not met.grew

    regressed = lead.advance(
        observed_response_m=0.0,
        required_response_m=0.00006,
    )
    assert regressed.magnitude_m == pytest.approx(0.00020)
    assert regressed.grew


def test_controller_lead_commit_preserves_effective_target_and_resets():
    lead = BoundedPlanarCommandLead()
    lead.begin(0.0002, 0.0)
    for _ in range(3):
        lead.advance(
            observed_response_m=0.0,
            required_response_m=0.00006,
        )

    logical = (1.0, 2.0)
    effective_before_commit = lead.command_xy(*logical)
    committed = lead.commit_xy(*logical)

    assert committed == pytest.approx(effective_before_commit)
    assert lead.status().magnitude_m == 0.0
    assert lead.command_xy(*committed) == pytest.approx(
        effective_before_commit
    )


def test_zero_planar_step_or_zero_requirement_never_invents_lead():
    lead = BoundedPlanarCommandLead()
    lead.begin(0.0, 0.0)
    assert lead.advance(
        observed_response_m=0.0,
        required_response_m=0.0001,
    ).magnitude_m == 0.0

    lead.reset()
    lead.begin(0.0002, 0.0)
    assert lead.advance(
        observed_response_m=0.0,
        required_response_m=0.0,
    ).magnitude_m == 0.0


def test_saturated_controller_lead_cannot_fake_a_step_ack():
    lead = BoundedPlanarCommandLead(
        growth_step_m=0.001,
        max_lead_m=0.001,
    )
    lead.begin(0.0002, 0.0)
    assert lead.advance(
        observed_response_m=0.0,
        required_response_m=0.00016,
    ).saturated

    gate = FineAlignmentStepGate(settle_steps=1)
    gate.begin(40, required_position_response_m=0.00016)
    blocked = gate.update(
        position_error_m=0.0,
        orientation_error_rad=0.0,
        motion_m=0.0,
        orientation_motion_rad=0.0,
        position_response_m=0.000159,
        orientation_response_rad=0.0,
    )
    assert not blocked.settled
    assert blocked.settle_count == 0

    real_response = gate.update(
        position_error_m=0.0,
        orientation_error_rad=0.0,
        motion_m=0.0,
        orientation_motion_rad=0.0,
        position_response_m=0.00016,
        orientation_response_rad=0.0,
    )
    assert real_response.settled


@pytest.mark.parametrize(
    ("growth_step_m", "max_lead_m"),
    [
        (0.0, 0.001),
        (0.002, 0.001),
        (math.nan, 0.001),
        (0.00005, math.inf),
    ],
)
def test_controller_lead_rejects_invalid_configuration(
    growth_step_m,
    max_lead_m,
):
    with pytest.raises(ValueError):
        BoundedPlanarCommandLead(
            growth_step_m=growth_step_m,
            max_lead_m=max_lead_m,
        )


def test_controller_lead_rejects_nonfinite_or_negative_observations():
    lead = BoundedPlanarCommandLead()
    with pytest.raises(ValueError, match="finite"):
        lead.begin(math.nan, 0.0)
    lead.begin(0.0002, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        lead.advance(
            observed_response_m=-0.1,
            required_response_m=0.1,
        )
    with pytest.raises(ValueError, match="finite"):
        lead.advance(
            observed_response_m=0.0,
            required_response_m=math.inf,
        )
    with pytest.raises(ValueError, match="finite"):
        lead.command_xy(math.nan, 0.0)


def test_planar_response_uses_command_projection_not_total_target_error():
    assert planar_command_response(
        start_x=0.0,
        start_y=0.0,
        current_x=0.0001,
        current_y=0.001,
        command_x=0.0002,
        command_y=0.0,
    ) == pytest.approx(0.0001)
    assert planar_command_response(
        start_x=0.0,
        start_y=0.0,
        current_x=-0.0001,
        current_y=0.0,
        command_x=0.0002,
        command_y=0.0,
    ) == 0.0

    with pytest.raises(ValueError, match="finite"):
        planar_command_response(
            start_x=0.0,
            start_y=0.0,
            current_x=math.nan,
            current_y=0.0,
            command_x=0.0002,
            command_y=0.0,
        )


def test_planar_yaw_delta_is_normalized_and_bounded():
    yaw = math.radians(0.25)
    assert planar_yaw_delta(
        quaternion_x=0.0,
        quaternion_y=0.0,
        quaternion_z=2.0 * math.sin(yaw / 2.0),
        quaternion_w=2.0 * math.cos(yaw / 2.0),
    ) == pytest.approx(yaw)

    with pytest.raises(ValueError):
        planar_yaw_delta(
            quaternion_x=0.0,
            quaternion_y=0.0,
            quaternion_z=math.sin(math.radians(1.1) / 2.0),
            quaternion_w=math.cos(math.radians(1.1) / 2.0),
        )
    with pytest.raises(ValueError):
        planar_yaw_delta(
            quaternion_x=0.1,
            quaternion_y=0.0,
            quaternion_z=0.0,
            quaternion_w=1.0,
        )

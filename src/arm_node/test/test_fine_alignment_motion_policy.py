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
    BoundedPlanarCommandLead,
    FineAlignmentStepGate,
    planar_command_response,
    planar_yaw_delta,
)


def test_step_ack_requires_twelve_stable_pose_observations():
    gate = FineAlignmentStepGate()
    gate.begin(123)

    for expected_count in range(1, 12):
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


def test_submillimeter_step_uses_noise_margin_but_still_requires_response():
    gate = FineAlignmentStepGate(
        position_response_noise_margin_m=0.0001,
        orientation_response_noise_margin_rad=math.radians(0.005),
        settle_steps=2,
    )
    gate.begin(
        20,
        required_position_response_m=0.00016,
        required_orientation_response_rad=math.radians(0.008),
    )

    for _ in range(5):
        status = gate.update(
            position_error_m=0.003,
            orientation_error_rad=math.radians(1.0),
            motion_m=0.0,
            orientation_motion_rad=0.0,
            position_response_m=0.0,
            orientation_response_rad=0.0,
        )
        assert status.settle_count == 0
        assert not status.settled

    # 0.2 mm * 80% - 0.1 mm margin = 0.06 mm; the observed 0.1 mm
    # command-direction response is enough once the EE is stationary.
    for expected_count in (1, 2):
        status = gate.update(
            position_error_m=0.003,
            orientation_error_rad=math.radians(1.0),
            motion_m=0.0,
            orientation_motion_rad=0.0,
            position_response_m=0.0001,
            orientation_response_rad=math.radians(0.004),
        )
        assert status.settle_count == expected_count
    assert status.settled


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

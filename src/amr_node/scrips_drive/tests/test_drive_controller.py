"""Unit tests for the pure physical AMR drive controller."""

import math
import sys
from dataclasses import fields
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from drive_controller import (  # noqa: E402
    ControllerConfig,
    Goal2D,
    GoToPoseController,
    PoseState,
    normalize_angle,
    quaternion_to_yaw,
    yaw_to_quaternion,
)


def _fast_config(**overrides):
    values = {
        'linear_acceleration': 100.0,
        'angular_acceleration': 100.0,
    }
    values.update(overrides)
    return ControllerConfig(**values)


def test_approved_physical_defaults():
    config = ControllerConfig()

    assert config.wheel_radius == pytest.approx(0.14)
    assert config.track_width == pytest.approx(0.4132)
    assert config.max_linear == pytest.approx(0.35)
    assert config.max_angular == pytest.approx(0.75)
    assert config.position_tolerance == pytest.approx(0.02)
    assert config.yaw_tolerance == pytest.approx(0.03)
    assert config.settle_steps == 12
    assert config.max_wheel_radps == pytest.approx(10.0)


def test_controller_has_no_timeout_or_workspace_policy():
    names = {field.name for field in fields(ControllerConfig)}

    assert names.isdisjoint({
        'goal_timeout_sec',
        'stuck_timeout_sec',
        'lease_timeout_sec',
        'workspace_radius',
    })


def test_angle_and_quaternion_round_trip():
    expected = math.radians(-123.0)

    assert normalize_angle(3.0 * math.pi) == pytest.approx(-math.pi)
    assert quaternion_to_yaw(*yaw_to_quaternion(expected)) == pytest.approx(
        expected,
    )


def test_zero_quaternion_is_rejected():
    with pytest.raises(ValueError, match='quaternion'):
        quaternion_to_yaw(0.0, 0.0, 0.0, 0.0)


def test_goal_ahead_selects_forward():
    controller = GoToPoseController(_fast_config())
    controller.set_goal(
        Goal2D(1.0, 0.0, 0.0),
        current_pose=PoseState(0.0, 0.0, 0.0),
    )

    command = controller.compute(PoseState(0.0, 0.0, 0.0), 0.1)

    assert controller.travel_direction == 'forward'
    assert command.phase == 'drive'
    assert command.linear > 0.0


def test_goal_behind_selects_reverse_without_a_half_turn():
    controller = GoToPoseController(_fast_config())
    controller.set_goal(
        Goal2D(-1.0, 0.0, 0.0),
        current_pose=PoseState(0.0, 0.0, 0.0),
    )

    command = controller.compute(PoseState(0.0, 0.0, 0.0), 0.1)

    assert controller.travel_direction == 'reverse'
    assert command.phase == 'drive'
    assert command.linear < 0.0
    assert command.angular == pytest.approx(0.0)


def test_direction_is_selected_on_first_compute_when_pose_is_deferred():
    controller = GoToPoseController(_fast_config())
    controller.set_goal(Goal2D(-1.0, 0.0, 0.0))

    command = controller.compute(PoseState(0.0, 0.0, 0.0), 0.1)

    assert command.travel_direction == 'reverse'


def test_selected_direction_is_held_for_the_whole_goal():
    controller = GoToPoseController(_fast_config())
    controller.set_goal(
        Goal2D(-1.0, 0.0, 0.0),
        current_pose=PoseState(0.0, 0.0, 0.0),
    )
    controller.compute(PoseState(0.0, 0.0, 0.0), 0.1)

    command = controller.compute(
        PoseState(-0.2, 0.0, math.pi),
        0.1,
    )

    assert controller.travel_direction == 'reverse'
    assert command.travel_direction == 'reverse'


def test_reverse_can_be_disabled_per_goal():
    controller = GoToPoseController(_fast_config())
    controller.set_goal(
        Goal2D(-1.0, 0.0, 0.0, allow_reverse=False),
        current_pose=PoseState(0.0, 0.0, 0.0),
    )

    command = controller.compute(PoseState(0.0, 0.0, 0.0), 0.1)

    assert command.travel_direction == 'forward'
    assert command.phase == 'turn_to_path'


def test_chassis_velocity_is_bounded():
    controller = GoToPoseController(_fast_config())
    controller.set_goal(
        Goal2D(10.0, 0.0, 0.0),
        current_pose=PoseState(0.0, 0.0, 0.0),
    )
    drive = controller.compute(PoseState(0.0, 0.0, 0.0), 0.1)

    turn_controller = GoToPoseController(_fast_config())
    turn_controller.set_goal(
        Goal2D(0.0, 10.0, math.pi / 2.0),
        current_pose=PoseState(0.0, 0.0, 0.0),
    )
    turn = turn_controller.compute(PoseState(0.0, 0.0, 0.0), 0.1)

    assert abs(drive.linear) <= 0.35
    assert abs(drive.angular) <= 0.75
    assert abs(turn.angular) <= 0.75


def test_wheel_conversion_and_joint_scaling():
    controller = GoToPoseController()
    straight = controller.to_wheel_speeds(0.28, 0.0)
    turn = controller.to_wheel_speeds(0.0, 0.75)
    saturated = controller.to_wheel_speeds(2.0, 3.0)

    assert straight == pytest.approx((2.0, 2.0))
    assert turn[0] < 0.0 < turn[1]
    assert max(abs(value) for value in saturated) == pytest.approx(10.0)


def test_position_and_yaw_tolerances_gate_settling():
    controller = GoToPoseController(_fast_config())
    controller.set_goal(
        Goal2D(0.0, 0.0, 0.0),
        current_pose=PoseState(0.0, 0.0, 0.0),
    )

    position = controller.compute(PoseState(0.0201, 0.0, 0.0), 0.1)
    yaw = controller.compute(PoseState(0.0, 0.0, 0.0301), 0.1)
    inside = controller.compute(PoseState(0.02, 0.0, 0.03), 0.1)

    assert position.phase in {'drive', 'turn_to_path'}
    assert yaw.phase == 'final_align'
    assert inside.phase == 'settling'


def test_arrival_requires_twelve_consecutive_stationary_ticks():
    controller = GoToPoseController(_fast_config())
    controller.set_goal(
        Goal2D(0.0, 0.0, 0.0),
        current_pose=PoseState(0.0, 0.0, 0.0),
    )
    stationary = PoseState(0.0, 0.0, 0.0)

    for _ in range(11):
        command = controller.compute(stationary, 1.0 / 60.0)
        assert not command.arrived

    command = controller.compute(stationary, 1.0 / 60.0)
    assert command.arrived
    assert command.phase == 'arrived'


def test_motion_resets_arrival_stability_counter():
    controller = GoToPoseController(_fast_config())
    controller.set_goal(
        Goal2D(0.0, 0.0, 0.0),
        current_pose=PoseState(0.0, 0.0, 0.0),
    )
    stationary = PoseState(0.0, 0.0, 0.0)

    for _ in range(11):
        controller.compute(stationary, 1.0 / 60.0)
    moving = controller.compute(
        PoseState(0.0, 0.0, 0.0, linear_speed=-0.03),
        1.0 / 60.0,
    )
    after_reset = controller.compute(stationary, 1.0 / 60.0)

    assert moving.phase == 'settling'
    assert not moving.arrived
    assert not after_reset.arrived


def test_nonfinite_goal_is_rejected():
    controller = GoToPoseController()

    with pytest.raises(ValueError, match='NaN'):
        controller.set_goal(Goal2D(float('nan'), 0.0, 0.0))

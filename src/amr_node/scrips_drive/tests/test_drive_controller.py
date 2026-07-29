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
    assert config.position_capture_tolerance == pytest.approx(0.015)
    assert config.position_release_tolerance == pytest.approx(0.025)
    assert config.yaw_tolerance == pytest.approx(0.03)
    assert config.yaw_capture_tolerance == pytest.approx(0.02)
    assert config.turn_in_place_threshold == pytest.approx(
        math.radians(5.0),
    )
    assert config.turn_in_place_release_threshold == pytest.approx(
        math.radians(3.0),
    )
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


def test_unreached_goal_remains_active_through_many_controller_steps():
    """A fixed distant pose must never synthesize a timeout/stuck failure."""
    controller = GoToPoseController(_fast_config())
    goal = Goal2D(100.0, 0.0, 0.0)
    fixed_pose = PoseState(0.0, 0.0, 0.0)
    controller.set_goal(goal, current_pose=fixed_pose)

    for _ in range(100_000):
        command = controller.compute(fixed_pose, 1.0 / 60.0)
        assert command.phase == 'drive'
        assert not command.arrived
        assert controller.goal == goal

    assert command.linear > 0.0
    assert command.distance_error == pytest.approx(100.0)


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


def test_near_goal_capture_is_identical_with_deferred_initial_pose():
    goal = Goal2D(0.0, 0.0, 0.0)
    pose = PoseState(0.018, 0.0, 0.025)

    immediate = GoToPoseController(_fast_config())
    immediate.set_goal(goal, current_pose=pose)
    immediate_command = immediate.compute(pose, 0.1)

    deferred = GoToPoseController(_fast_config())
    deferred.set_goal(goal)
    deferred_command = deferred.compute(pose, 0.1)

    assert immediate_command == deferred_command
    assert deferred_command.phase == 'settling'
    assert deferred_command.linear == pytest.approx(0.0)
    assert deferred_command.angular == pytest.approx(0.0)


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


def test_station5_turn_to_path_hysteresis_survives_boundary_jitter():
    """The measured 5-degree station-5 boundary must not reset acceleration."""
    controller = GoToPoseController(ControllerConfig())
    goal = Goal2D(0.6667, -1.1964, -1.5707)
    start = PoseState(-0.9586, 1.9078, -math.pi / 2.0)
    controller.set_goal(goal, current_pose=start)
    bearing = math.atan2(goal.y - 1.858, goal.x - (-0.940))

    def pose_with_heading_error(error_degrees):
        return PoseState(
            -0.940,
            1.858,
            normalize_angle(bearing - math.radians(error_degrees)),
        )

    entering = controller.compute(
        pose_with_heading_error(5.05),
        1.0 / 60.0,
    )
    entry_jitter = controller.compute(
        pose_with_heading_error(4.95),
        1.0 / 60.0,
    )
    above_release = controller.compute(
        pose_with_heading_error(3.10),
        1.0 / 60.0,
    )
    released = controller.compute(
        pose_with_heading_error(2.90),
        1.0 / 60.0,
    )
    drive_jitter = controller.compute(
        pose_with_heading_error(4.95),
        1.0 / 60.0,
    )

    assert [
        entering.phase,
        entry_jitter.phase,
        above_release.phase,
        released.phase,
        drive_jitter.phase,
    ] == [
        'turn_to_path',
        'turn_to_path',
        'turn_to_path',
        'drive',
        'drive',
    ]
    assert all(
        command.travel_direction == 'forward'
        for command in (
            entering,
            entry_jitter,
            above_release,
            released,
            drive_jitter,
        )
    )
    assert released.linear == pytest.approx(0.35 / 60.0)
    assert drive_jitter.linear == pytest.approx(2.0 * 0.35 / 60.0)


def test_turn_to_path_latch_resets_on_stop_and_replacement_goal():
    controller = GoToPoseController(_fast_config())
    goal = Goal2D(1.0, 0.0, 0.0)

    def pose_with_heading_error(error_degrees):
        return PoseState(
            0.0,
            0.0,
            -math.radians(error_degrees),
        )

    controller.set_goal(goal, current_pose=pose_with_heading_error(5.1))
    assert controller.compute(
        pose_with_heading_error(5.1),
        0.1,
    ).phase == 'turn_to_path'

    controller.stop()
    after_stop = controller.compute(
        pose_with_heading_error(4.9),
        0.1,
    )
    assert after_stop.phase == 'drive'

    controller.compute(pose_with_heading_error(5.1), 0.1)
    controller.set_goal(goal, current_pose=pose_with_heading_error(4.9))
    after_replacement = controller.compute(
        pose_with_heading_error(4.9),
        0.1,
    )
    assert after_replacement.phase == 'drive'
    assert controller.travel_direction == 'forward'


def test_turn_to_path_release_threshold_must_be_below_entry_threshold():
    with pytest.raises(ValueError, match='turn thresholds'):
        GoToPoseController(ControllerConfig(
            turn_in_place_threshold=math.radians(5.0),
            turn_in_place_release_threshold=math.radians(5.0),
        ))


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
    position_controller = GoToPoseController(_fast_config())
    position_controller.set_goal(
        Goal2D(0.0, 0.0, 0.0),
        current_pose=PoseState(0.0, 0.0, 0.0),
    )
    position = position_controller.compute(
        PoseState(0.0201, 0.0, 0.0),
        0.1,
    )

    yaw_controller = GoToPoseController(_fast_config())
    yaw_controller.set_goal(
        Goal2D(0.0, 0.0, 0.0),
        current_pose=PoseState(0.0, 0.0, 0.0),
    )
    yaw = yaw_controller.compute(PoseState(0.0, 0.0, 0.0301), 0.1)

    inside_controller = GoToPoseController(_fast_config())
    inside_pose = PoseState(0.02, 0.0, 0.03)
    inside_controller.set_goal(
        Goal2D(0.0, 0.0, 0.0),
        current_pose=inside_pose,
    )
    inside = inside_controller.compute(inside_pose, 0.1)

    assert position.phase in {'drive', 'turn_to_path'}
    assert yaw.phase == 'final_align'
    assert inside.phase == 'settling'


def test_final_yaw_reacquires_inner_boundary_before_settling():
    controller = GoToPoseController(_fast_config())
    controller.set_goal(
        Goal2D(0.0, 0.0, 0.0),
        current_pose=PoseState(0.0, 0.0, 0.0),
    )

    outside = controller.compute(PoseState(0.0, 0.0, 0.0301), 0.1)
    barely_inside = controller.compute(
        PoseState(0.0, 0.0, 0.0299),
        0.1,
    )
    captured = controller.compute(PoseState(0.0, 0.0, 0.0199), 0.1)
    drifted = controller.compute(PoseState(0.0, 0.0, 0.0299), 0.1)

    assert outside.phase == 'final_align'
    assert barely_inside.phase == 'final_align'
    assert captured.phase == 'settling'
    assert drifted.phase == 'settling'


def test_yaw_capture_margin_converges_with_small_boundary_drift():
    controller = GoToPoseController(ControllerConfig())
    goal = Goal2D(0.0, 0.0, 0.0)
    yaw = -0.0301
    controller.set_goal(
        goal,
        current_pose=PoseState(0.0199, 0.0, yaw),
    )

    command = None
    dt = 1.0 / 60.0
    drift_per_tick = -0.0001
    for _ in range(240):
        command = controller.compute(
            PoseState(
                0.0199,
                0.0,
                yaw,
                angular_speed=abs(drift_per_tick / dt),
            ),
            dt,
        )
        yaw = normalize_angle(
            yaw + command.angular * dt + drift_per_tick
        )
        if command.arrived:
            break

    assert command is not None
    assert command.arrived
    assert command.phase == 'arrived'
    assert command.distance_error <= controller.config.position_tolerance
    assert abs(command.yaw_error) <= controller.config.yaw_tolerance


def test_active_position_trim_reacquires_inner_boundary_before_settling():
    controller = GoToPoseController(ControllerConfig())
    goal = Goal2D(0.0, 0.0, 0.0)
    controller.set_goal(
        goal,
        current_pose=PoseState(-1.0, 0.0, 0.0),
    )

    # Capture the goal position, then reproduce the measured outward slip
    # beyond 20mm that activates final position trim.
    controller.compute(PoseState(-0.014, 0.0, 0.0), 1.0 / 60.0)
    x = -0.0201
    command = None
    dt = 1.0 / 60.0
    drift_per_tick = -0.0001
    phases = []
    for _ in range(240):
        command = controller.compute(
            PoseState(
                x,
                0.0,
                0.0,
                linear_speed=abs(drift_per_tick / dt),
            ),
            dt,
        )
        phases.append(command.phase)
        x += command.linear * dt + drift_per_tick
        if command.arrived:
            break

    assert command is not None
    assert command.arrived
    assert 'position_trim' in phases
    first_settling = phases.index('settling')
    assert all(
        phase not in {'settling', 'arrived'}
        for phase in phases[:first_settling]
    )
    assert command.distance_error <= controller.config.position_tolerance
    assert abs(command.yaw_error) <= controller.config.yaw_tolerance


def test_reverse_position_trim_prevents_boundary_phase_limit_cycle():
    controller = GoToPoseController(_fast_config())
    goal = Goal2D(
        0.5867,
        1.9078,
        -math.pi / 2.0,
    )
    start = PoseState(0.665, -0.019, -math.pi / 2.0)
    controller.set_goal(goal, current_pose=start)
    controller.compute(start, 0.1)

    # Capture inside 15mm, then inject the position drift measured during
    # the final yaw correction.
    entry = PoseState(0.5855, 1.8948, math.radians(-94.4))
    first = controller.compute(entry, 0.1)
    slipped = controller.compute(
        PoseState(0.5840, 1.8859, math.radians(-91.5)),
        0.1,
    )

    assert controller.travel_direction == 'reverse'
    assert first.phase == 'final_align'
    assert first.angular > 0.0
    assert slipped.phase == 'position_trim'
    assert slipped.linear < 0.0
    assert slipped.angular > 0.0
    assert slipped.angular < controller.config.min_angular
    assert not slipped.arrived

    # The measured limit cycle used to switch to turn_to_path here. Continue
    # feeding a converging reverse correction and require the controller to
    # retain the final-yaw trim phase until the strict 20mm boundary is met.
    outside = controller.compute(
        PoseState(0.5842, 1.88795, math.radians(-90.9)),
        0.1,
    )
    inside = controller.compute(
        PoseState(0.5845, 1.8882, math.radians(-90.5)),
        0.1,
    )

    assert outside.distance_error > 0.02
    assert outside.phase == 'position_trim'
    assert outside.linear < 0.0
    assert inside.distance_error < 0.02
    assert inside.distance_error > controller.config.position_capture_tolerance
    assert inside.phase == 'position_trim'
    assert not inside.arrived

    captured = controller.compute(
        PoseState(0.5845, 1.8930, math.radians(-90.5)),
        0.1,
    )
    assert captured.distance_error < controller.config.position_capture_tolerance
    assert captured.phase == 'settling'


def test_position_trim_falls_back_when_residual_is_not_reachable():
    controller = GoToPoseController(_fast_config())
    goal = Goal2D(0.0, 0.0, 0.0)
    start = PoseState(-1.0, 0.0, 0.0)
    controller.set_goal(goal, current_pose=start)

    controller.compute(
        PoseState(-0.010, -0.005, math.radians(-3.0)),
        0.1,
    )
    lateral_residual = controller.compute(
        PoseState(-0.010, -0.021, 0.0),
        0.1,
    )

    assert lateral_residual.distance_error > 0.02
    assert lateral_residual.phase in {'drive', 'turn_to_path'}
    assert lateral_residual.phase != 'position_trim'
    assert not lateral_residual.arrived


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


def test_24mm_position_drift_resets_arrival_stability_counter():
    controller = GoToPoseController(_fast_config())
    goal = Goal2D(0.0, 0.0, 0.0)
    stationary = PoseState(0.0, 0.0, 0.0)
    controller.set_goal(goal, current_pose=stationary)

    for _ in range(11):
        command = controller.compute(stationary, 1.0 / 60.0)
        assert not command.arrived

    drifted = controller.compute(
        PoseState(0.024, 0.0, 0.0),
        1.0 / 60.0,
    )
    assert drifted.distance_error == pytest.approx(0.024)
    assert not drifted.arrived

    # Returning inside the strict boundary starts a new 12-tick window; the
    # eleven pre-drift observations must not carry over.
    for _ in range(11):
        reacquired = controller.compute(stationary, 1.0 / 60.0)
        assert not reacquired.arrived
    reacquired = controller.compute(stationary, 1.0 / 60.0)
    assert reacquired.arrived
    assert reacquired.distance_error <= controller.config.position_tolerance


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

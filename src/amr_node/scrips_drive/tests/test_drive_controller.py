import math
import sys
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
from vision_goal import (  # noqa: E402
    DetectionCandidate,
    DetectionWindow,
    VisionUnavailable,
    compute_busbar_pick_targets,
    compute_busbar_scan_xy,
    compute_directional_standoff_goal,
    compute_standoff_goal,
    select_highest_confidence,
    select_nearest_busbar_path,
    select_nearest_valid_pair,
    select_nearest_to_reference,
)


def test_normalize_angle_wraps_both_directions():
    assert math.isclose(normalize_angle(3.0 * math.pi), -math.pi)
    assert math.isclose(normalize_angle(-3.0 * math.pi), -math.pi)
    assert math.isclose(normalize_angle(0.25), 0.25)


def test_yaw_quaternion_round_trip():
    expected = math.radians(-123.0)
    quaternion = yaw_to_quaternion(expected)
    actual = quaternion_to_yaw(*quaternion)
    assert math.isclose(actual, expected, abs_tol=1.0e-12)


def test_invalid_zero_quaternion_is_rejected():
    try:
        quaternion_to_yaw(0.0, 0.0, 0.0, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero quaternion must be rejected")


def test_straight_goal_produces_equal_wheel_speeds():
    config = ControllerConfig(linear_acceleration=100.0)
    controller = GoToPoseController(config)
    controller.set_goal(Goal2D(1.0, 0.0, 0.0))

    command = controller.compute(PoseState(0.0, 0.0, 0.0), 0.1)
    left, right = controller.to_wheel_speeds(
        command.linear, command.angular
    )

    assert command.phase == "drive"
    assert command.linear > 0.0
    assert math.isclose(command.angular, 0.0, abs_tol=1.0e-12)
    assert math.isclose(left, right, abs_tol=1.0e-12)


def test_large_heading_error_turns_in_place():
    config = ControllerConfig(angular_acceleration=100.0)
    controller = GoToPoseController(config)
    controller.set_goal(Goal2D(0.0, 1.0, 0.0))

    command = controller.compute(PoseState(0.0, 0.0, 0.0), 0.1)
    left, right = controller.to_wheel_speeds(
        command.linear, command.angular
    )

    assert command.phase == "turn_to_path"
    assert command.linear == 0.0
    assert command.angular > 0.0
    assert left < 0.0 < right


def test_turn_to_path_uses_minimum_angular_speed_near_threshold():
    config = ControllerConfig(
        angular_acceleration=100.0,
        turn_in_place_threshold=math.radians(5.0),
        min_angular=0.30,
    )
    controller = GoToPoseController(config)
    heading = math.radians(6.8)
    controller.set_goal(
        Goal2D(
            math.cos(heading),
            math.sin(heading),
            heading,
        )
    )

    command = controller.compute(PoseState(0.0, 0.0, 0.0), 0.1)

    assert command.phase == "turn_to_path"
    assert command.linear == 0.0
    assert command.angular == pytest.approx(config.min_angular)


def test_visible_heading_error_is_removed_before_forward_drive():
    config = ControllerConfig(angular_acceleration=100.0)
    controller = GoToPoseController(config)
    heading = math.radians(10.0)
    controller.set_goal(
        Goal2D(math.cos(heading), math.sin(heading), heading)
    )

    command = controller.compute(PoseState(0.0, 0.0, 0.0), 0.1)

    assert command.phase == "turn_to_path"
    assert command.travel_direction == "forward"
    assert command.linear == 0.0
    assert command.angular > 0.0


def test_scan_docking_aligns_goal_yaw_before_chasing_position():
    config = ControllerConfig(angular_acceleration=100.0)
    controller = GoToPoseController(config)
    controller.set_goal(
        Goal2D(
            -0.2,
            0.0,
            0.0,
            allow_reverse=True,
            align_yaw_before_drive=True,
        )
    )

    command = controller.compute(
        PoseState(0.0, 0.0, -math.pi / 2.0),
        0.1,
    )
    assert command.phase == "initial_align"
    assert command.linear == 0.0
    assert command.angular > 0.0

    command = controller.compute(PoseState(0.0, 0.0, 0.0), 0.1)
    assert command.phase == "drive"
    assert command.travel_direction == "reverse"
    assert command.linear < 0.0


def test_reverse_docking_uses_goal_yaw_inside_final_approach():
    """수 mm 횡오차가 남아도 목표 직전에서 bearing 회전 교착에 빠지지 않는다."""
    config = ControllerConfig(
        linear_acceleration=100.0,
        angular_acceleration=100.0,
        position_tolerance=0.04,
        final_approach_distance=0.08,
    )
    controller = GoToPoseController(config)
    controller.set_goal(
        Goal2D(
            0.9056,
            2.2893,
            0.0,
            allow_reverse=True,
            align_yaw_before_drive=True,
        )
    )

    command = controller.compute(
        # 최신 실패 로그의 정지 pose: 거리 0.04019m, reverse bearing 약 8도.
        PoseState(0.9454, 2.2949, 0.0),
        0.1,
    )

    assert command.phase == "drive"
    assert command.travel_direction == "reverse"
    assert command.linear < 0.0
    assert math.isclose(command.heading_error, 0.0, abs_tol=1.0e-12)


def test_busbar_scan_search_starts_at_selected_target_and_stays_bounded():
    target = (0.0056, 2.2893)
    first = compute_busbar_scan_xy(target, math.pi, 0, 0.05)
    assert first == pytest.approx(target)

    attempts = [
        compute_busbar_scan_xy(target, math.pi, index, 0.05)
        for index in range(6)
    ]
    assert attempts[:3] == pytest.approx(attempts[3:])
    assert all(
        math.hypot(x - target[0], y - target[1])
        <= 0.05 + 1.0e-12
        for x, y in attempts
    )
    # AMR 반대편으로 더 멀어지는 접근-normal 탐색은 하지 않는다.
    assert all(x == pytest.approx(target[0]) for x, _ in attempts)


def test_busbar_pick_lift_is_vertical_without_world_y_offset():
    approach, pick, lift = compute_busbar_pick_targets(
        target_xy=(-0.0339, 2.3049),
        pick_z=0.455,
        approach_z=0.600,
    )
    assert approach == pytest.approx((-0.0339, 2.3049, 0.600))
    assert pick == pytest.approx((-0.0339, 2.3049, 0.455))
    assert lift == pytest.approx((-0.0339, 2.3049, 0.600))


def test_selected_vision_busbar_resolves_nearest_live_usd_prim():
    path = select_nearest_busbar_path(
        target_xy=(0.0081, 2.2907),
        candidates=(
            ("/World/Z_busbar3/Mesh", 0.0305, 1.7777),
            ("/World/Z_busbar3_01/Mesh", 0.0312, 2.7912),
            ("/World/Z_busbar3_02/Mesh", 0.0324, 2.2812),
        ),
        max_distance_m=0.30,
    )
    assert path == "/World/Z_busbar3_02/Mesh"


def test_busbar_prim_resolution_rejects_unrelated_object():
    with pytest.raises(VisionUnavailable, match="최근접 거리"):
        select_nearest_busbar_path(
            target_xy=(10.0, 10.0),
            candidates=(("/World/Z_busbar3/Mesh", 0.0, 0.0),),
            max_distance_m=0.30,
        )


@pytest.mark.parametrize("attempt", [-1, 1.5, True])
def test_busbar_scan_search_rejects_invalid_attempt(attempt):
    with pytest.raises(ValueError):
        compute_busbar_scan_xy((0.0, 0.0), 0.0, attempt, 0.05)


def test_reverse_permission_alone_does_not_auto_select_reverse():
    config = ControllerConfig(
        linear_acceleration=100.0,
        angular_acceleration=100.0,
        allow_reverse=True,
    )
    controller = GoToPoseController(config)
    controller.set_goal(Goal2D(-1.0, 0.0, math.pi))

    command = controller.compute(PoseState(0.0, 0.0, 0.0), 0.1)
    left, right = controller.to_wheel_speeds(
        command.linear,
        command.angular,
    )

    assert command.phase == "turn_to_path"
    assert command.travel_direction == "forward"
    assert command.linear == 0.0
    assert abs(command.angular) > 0.0
    assert left * right < 0.0


def test_forward_only_policy_keeps_turn_in_place_for_goal_behind():
    config = ControllerConfig(
        angular_acceleration=100.0,
        allow_reverse=False,
    )
    controller = GoToPoseController(config)
    controller.set_goal(Goal2D(-1.0, 0.0, math.pi))

    command = controller.compute(PoseState(0.0, 0.0, 0.0), 0.1)

    assert command.phase == "turn_to_path"
    assert command.travel_direction == "forward"
    assert command.linear == 0.0
    assert abs(command.angular) > 0.0


def test_goal_can_forbid_explicit_reverse():
    config = ControllerConfig(
        angular_acceleration=100.0,
        allow_reverse=True,
    )
    controller = GoToPoseController(config)
    controller.set_goal(
        Goal2D(
            -1.0,
            0.0,
            math.pi,
            allow_reverse=False,
        )
    )

    command = controller.compute(PoseState(0.0, 0.0, 0.0), 0.1)

    assert command.phase == "turn_to_path"
    assert command.travel_direction == "forward"
    assert command.linear == 0.0


def test_explicit_reverse_choice_is_latched_for_entire_goal():
    config = ControllerConfig(
        linear_acceleration=100.0,
        angular_acceleration=100.0,
        allow_reverse=True,
    )
    controller = GoToPoseController(config)
    controller.set_goal(
        Goal2D(
            -1.0,
            0.0,
            0.0,
            allow_reverse=True,
            align_yaw_before_drive=True,
        )
    )

    first = controller.compute(PoseState(0.0, 0.0, 0.0), 0.1)
    changed_pose = controller.compute(
        PoseState(0.0, 0.0, math.pi),
        0.1,
    )

    assert first.travel_direction == "reverse"
    assert changed_pose.travel_direction == "reverse"
    assert changed_pose.phase == "turn_to_path"


def test_position_arrival_requires_final_yaw_alignment():
    config = ControllerConfig(
        angular_acceleration=100.0,
        settle_steps=2,
    )
    controller = GoToPoseController(config)
    controller.set_goal(Goal2D(0.0, 0.0, math.pi / 2.0))

    command = controller.compute(PoseState(0.0, 0.0, 0.0), 0.1)
    assert command.phase == "final_align"
    assert command.angular > 0.0
    assert not command.arrived

    aligned = PoseState(0.0, 0.0, math.pi / 2.0)
    first = controller.compute(aligned, 0.1)
    second = controller.compute(aligned, 0.1)
    assert first.phase == "settling"
    assert not first.arrived
    assert second.phase == "arrived"
    assert second.arrived


def test_final_alignment_uses_minimum_angular_speed_to_overcome_friction():
    config = ControllerConfig(
        yaw_tolerance=math.radians(3.0),
        min_angular=0.30,
        angular_acceleration=100.0,
    )
    controller = GoToPoseController(config)
    controller.set_goal(Goal2D(0.0, 0.0, 0.0))

    command = controller.compute(
        PoseState(0.0, 0.0, math.radians(-5.0)),
        0.1,
    )

    assert command.phase == "final_align"
    assert math.isclose(command.angular, 0.30)


def test_position_only_goal_does_not_rotate_at_intermediate_waypoint():
    config = ControllerConfig(settle_steps=1)
    controller = GoToPoseController(config)
    controller.set_goal(
        Goal2D(
            0.0,
            0.0,
            math.pi,
            yaw_required=False,
        )
    )

    command = controller.compute(PoseState(0.0, 0.0, 0.0), 0.1)

    assert command.phase == "arrived"
    assert command.arrived
    assert command.angular == 0.0


def test_arrival_waits_until_root_motion_has_settled():
    config = ControllerConfig(settle_steps=2)
    controller = GoToPoseController(config)
    controller.set_goal(Goal2D(0.0, 0.0, 0.0))

    moving = PoseState(
        0.0,
        0.0,
        0.0,
        linear_speed=0.03,
        angular_speed=0.06,
    )
    assert not controller.compute(moving, 0.1).arrived
    assert not controller.compute(moving, 0.1).arrived

    stopped = PoseState(0.0, 0.0, 0.0)
    assert not controller.compute(stopped, 0.1).arrived
    assert controller.compute(stopped, 0.1).arrived


def test_wheel_speed_limit_scales_both_sides():
    config = ControllerConfig(
        wheel_radius=0.1,
        track_width=0.5,
        max_wheel_radps=2.0,
    )
    controller = GoToPoseController(config)
    left, right = controller.to_wheel_speeds(1.0, 1.0)

    assert max(abs(left), abs(right)) <= 2.0
    assert math.isclose(left / right, 0.75 / 1.25)


def test_new_goal_resets_acceleration_history():
    config = ControllerConfig(linear_acceleration=1.0)
    controller = GoToPoseController(config)
    pose = PoseState(0.0, 0.0, 0.0)
    controller.set_goal(Goal2D(1.0, 0.0, 0.0))
    first = controller.compute(pose, 0.1)
    second = controller.compute(pose, 0.1)
    assert second.linear > first.linear

    controller.set_goal(Goal2D(2.0, 0.0, 0.0))
    reset = controller.compute(pose, 0.1)
    assert math.isclose(reset.linear, first.linear)


def test_turn_only_phases_drop_residual_forward_velocity_immediately():
    config = ControllerConfig(
        linear_acceleration=0.1,
        angular_acceleration=100.0,
    )
    controller = GoToPoseController(config)
    controller.set_goal(Goal2D(1.0, 0.0, math.pi / 2.0))

    driving = controller.compute(PoseState(0.0, 0.0, 0.0), 1.0)
    assert driving.phase == "drive"
    assert driving.linear > 0.0

    turn_to_path = controller.compute(
        PoseState(0.0, 0.0, math.pi / 2.0),
        0.01,
    )
    assert turn_to_path.phase == "turn_to_path"
    assert turn_to_path.linear == 0.0

    final_align = controller.compute(
        PoseState(1.0, 0.0, 0.0),
        0.01,
    )
    assert final_align.phase == "final_align"
    assert final_align.linear == 0.0


def _gate(expected_candidates=1):
    return DetectionWindow(
        expected_candidates=expected_candidates,
        min_score=0.6,
        min_samples=3,
        max_samples=5,
        max_age_sec=1.0,
        max_std_m=0.01,
    )


def _candidate(x, y, score=0.9, z=0.2):
    return DetectionCandidate(x=x, y=y, z=z, score=score)


def test_highest_confidence_candidate_is_selected():
    selected = select_highest_confidence(
        [
            _candidate(1.0, 2.0, score=0.72),
            _candidate(3.0, 4.0, score=0.91),
            _candidate(5.0, 6.0, score=0.84),
        ]
    )

    assert len(selected) == 1
    assert selected[0] == _candidate(3.0, 4.0, score=0.91)


def test_active_busbar_keeps_nearest_when_confidence_order_flips():
    selected = select_nearest_to_reference(
        [
            _candidate(0.031, 2.807, score=0.81),
            _candidate(0.431, 2.807, score=0.95),
        ],
        (0.032, 2.807),
    )

    assert selected == (_candidate(0.031, 2.807, score=0.81),)


def test_tied_highest_confidence_is_rejected():
    with pytest.raises(VisionUnavailable, match="같은 후보"):
        select_highest_confidence(
            [
                _candidate(1.0, 2.0, score=0.91),
                _candidate(3.0, 4.0, score=0.91),
            ]
        )


def test_nearest_valid_bolt_pair_is_selected_from_multiple_batteries():
    candidates = [
        _candidate(2.00, 0.00, score=0.91),
        _candidate(2.42, 0.00, score=0.89),
        _candidate(0.80, 0.10, score=0.80),
        _candidate(1.22, 0.10, score=0.81),
    ]

    selected = select_nearest_valid_pair(
        candidates,
        (0.0, 0.0),
        expected_spacing_m=0.42,
        spacing_tolerance_m=0.02,
    )

    assert selected == (candidates[2], candidates[3])


def test_bolt_pair_selection_rejects_cross_battery_spacing():
    with pytest.raises(VisionUnavailable, match="유효한 볼트쌍"):
        select_nearest_valid_pair(
            [
                _candidate(0.0, 0.0),
                _candidate(0.1, 0.0),
                _candidate(1.0, 0.0),
            ],
            (0.0, 0.0),
            expected_spacing_m=0.42,
            spacing_tolerance_m=0.02,
        )


def test_bolt_pair_tracking_uses_previous_midpoint_reference():
    candidates = [
        _candidate(0.80, 0.00, score=0.80),
        _candidate(1.22, 0.00, score=0.81),
        _candidate(2.00, 0.00, score=0.99),
        _candidate(2.42, 0.00, score=0.99),
    ]

    selected = select_nearest_valid_pair(
        candidates,
        (2.20, 0.0),
        expected_spacing_m=0.42,
        spacing_tolerance_m=0.02,
    )

    assert selected == (candidates[2], candidates[3])


def test_nonfinite_candidate_confidence_is_rejected_before_selection():
    with pytest.raises(VisionUnavailable, match="NaN/inf"):
        select_highest_confidence(
            [
                _candidate(1.0, 2.0, score=0.91),
                _candidate(3.0, 4.0, score=float("nan")),
            ]
        )


def test_stable_single_detection_is_averaged():
    gate = _gate()
    assert gate.add_frame([_candidate(1.00, 2.00)], 10.0, 10_000)
    assert gate.add_frame([_candidate(1.01, 2.00)], 10.1, 10_100)
    assert gate.add_frame([_candidate(0.99, 2.00)], 10.2, 10_200)

    snapshot = gate.snapshot(10.3)

    assert snapshot.sample_count == 3
    assert math.isclose(snapshot.x, 1.0)
    assert math.isclose(snapshot.y, 2.0)
    assert snapshot.min_score == 0.9


def test_bolt_pair_uses_only_geometric_midpoint():
    gate = _gate(expected_candidates=2)
    for stamp in (20.0, 20.1, 20.2):
        assert gate.add_frame(
            [
                _candidate(1.0, 0.2),
                _candidate(1.4, 0.6),
            ],
            stamp,
            int(stamp * 1000),
        )

    snapshot = gate.snapshot(20.3)

    assert math.isclose(snapshot.x, 1.2)
    assert math.isclose(snapshot.y, 0.4)


def test_missing_or_extra_candidates_clear_the_window():
    gate = _gate()
    gate.add_frame([_candidate(1.0, 2.0)], 30.0, 30_000)
    gate.add_frame([_candidate(1.0, 2.0)], 30.1, 30_100)
    assert not gate.add_frame([], 30.2, 30_200)
    assert gate.sample_count == 0
    with pytest.raises(VisionUnavailable):
        gate.snapshot(30.3)

    assert not gate.add_frame(
        [_candidate(1.0, 2.0), _candidate(1.1, 2.1)],
        30.4,
        30_400,
    )
    assert gate.sample_count == 0


def test_transient_rejection_preserves_valid_pair_samples():
    gate = _gate(expected_candidates=2)
    pair = [
        _candidate(1.0, 0.2),
        _candidate(1.331, 0.2),
    ]
    assert gate.add_frame(pair, 30.0, 30_000)
    assert gate.add_frame(pair, 30.1, 30_100)

    gate.note_transient_rejection("볼트 후보 수 1개")

    assert gate.sample_count == 2
    assert gate.add_frame(pair, 30.3, 30_300)
    snapshot = gate.snapshot(30.4)
    assert snapshot.sample_count == 3
    assert math.isclose(snapshot.x, 1.1655)


def test_low_confidence_has_no_coordinate_fallback():
    gate = _gate()
    assert not gate.add_frame(
        [_candidate(1.0, 2.0, score=0.59)],
        40.0,
        40_000,
    )
    with pytest.raises(VisionUnavailable):
        gate.snapshot(40.1)


def test_stale_detection_is_rejected():
    gate = _gate()
    for stamp in (50.0, 50.1, 50.2):
        gate.add_frame(
            [_candidate(1.0, 2.0)],
            stamp,
            int(stamp * 1000),
        )
    with pytest.raises(VisionUnavailable, match="오래"):
        gate.snapshot(51.3)


def test_active_snapshot_can_use_explicit_longer_dropout_limit():
    gate = _gate()
    for stamp in (50.0, 50.1, 50.2):
        gate.add_frame(
            [_candidate(1.0, 2.0)],
            stamp,
            int(stamp * 1000),
        )

    snapshot = gate.snapshot(53.7, max_age_sec=6.0)

    assert math.isclose(snapshot.age_sec, 3.5)


def test_unstable_detection_is_rejected():
    gate = _gate()
    gate.add_frame([_candidate(1.00, 2.0)], 60.0, 60_000)
    gate.add_frame([_candidate(1.04, 2.0)], 60.1, 60_100)
    gate.add_frame([_candidate(0.96, 2.0)], 60.2, 60_200)
    with pytest.raises(VisionUnavailable, match="표준편차"):
        gate.snapshot(60.3)


def test_standoff_goal_uses_only_current_and_detected_world_points():
    goal = compute_standoff_goal(
        current_xy=(0.0, 0.0),
        target_xy=(2.0, 0.0),
        standoff_m=0.5,
    )
    assert math.isclose(goal.x, 1.5)
    assert math.isclose(goal.y, 0.0)
    assert math.isclose(goal.yaw, 0.0)


def test_target_inside_standoff_is_rejected_instead_of_fallback():
    with pytest.raises(VisionUnavailable):
        compute_standoff_goal(
            current_xy=(0.0, 0.0),
            target_xy=(0.2, 0.0),
            standoff_m=0.5,
        )


def test_directional_standoff_uses_detection_and_workcell_normal():
    goal = compute_directional_standoff_goal(
        target_xy=(0.03, 2.79),
        approach_yaw_rad=math.pi,
        standoff_m=0.55,
    )
    assert math.isclose(goal.x, 0.58)
    assert math.isclose(goal.y, 2.79)
    assert math.isclose(abs(goal.yaw), math.pi)


def test_directional_standoff_can_face_scan_arm_toward_target():
    goal = compute_directional_standoff_goal(
        target_xy=(0.03, 2.79),
        approach_yaw_rad=math.pi,
        standoff_m=0.55,
        docking_yaw_offset_rad=math.pi,
    )
    assert math.isclose(goal.x, 0.58)
    assert math.isclose(goal.y, 2.79)
    assert math.isclose(goal.yaw, 0.0, abs_tol=1.0e-12)


def test_directional_standoff_rejects_nonfinite_heading():
    with pytest.raises(VisionUnavailable, match="NaN/inf"):
        compute_directional_standoff_goal(
            target_xy=(0.03, 2.79),
            approach_yaw_rad=float("nan"),
            standoff_m=0.55,
        )


def test_live_battery_start_pose_accepts_separate_standoff():
    current_xy = (0.4755832552909851, -0.02661629021167755)
    battery_midpoint_xy = (1.2209, -0.2012)

    with pytest.raises(VisionUnavailable, match="standoff"):
        compute_standoff_goal(
            current_xy=current_xy,
            target_xy=battery_midpoint_xy,
            standoff_m=0.90,
        )

    goal = compute_standoff_goal(
        current_xy=current_xy,
        target_xy=battery_midpoint_xy,
        standoff_m=0.75,
    )
    assert math.hypot(
        battery_midpoint_xy[0] - goal.x,
        battery_midpoint_xy[1] - goal.y,
    ) == pytest.approx(0.75)


def test_duplicate_source_stamp_cannot_fake_consecutive_frames():
    gate = _gate()
    assert gate.add_frame([_candidate(1.0, 2.0)], 70.0, 70_000)
    assert not gate.add_frame(
        [_candidate(1.0, 2.0)],
        70.1,
        70_000,
    )
    assert gate.sample_count == 0
    with pytest.raises(VisionUnavailable):
        gate.snapshot(70.2)


def test_debug_snapshot_can_disable_only_the_age_gate():
    gate = _gate()
    assert gate.add_frame([_candidate(1.0, 2.0)], 10.0, 10_000)
    assert gate.add_frame([_candidate(1.0, 2.0)], 10.1, 10_001)
    assert gate.add_frame([_candidate(1.0, 2.0)], 10.2, 10_002)
    with pytest.raises(VisionUnavailable, match="오래됐습니다"):
        gate.snapshot(100.0)
    snapshot = gate.snapshot(100.0, enforce_age=False)
    assert math.isclose(snapshot.x, 1.0)
    assert math.isclose(snapshot.y, 2.0)


def test_unstable_bolt_pair_distance_is_rejected():
    gate = _gate(expected_candidates=2)
    gate.add_frame(
        [_candidate(1.0, 0.0), _candidate(1.2, 0.0)],
        80.0,
        80_000,
    )
    gate.add_frame(
        [_candidate(0.9, 0.0), _candidate(1.3, 0.0)],
        80.1,
        80_100,
    )
    gate.add_frame(
        [_candidate(1.05, 0.0), _candidate(1.15, 0.0)],
        80.2,
        80_200,
    )
    with pytest.raises(VisionUnavailable, match="볼트쌍 간격"):
        gate.snapshot(80.3)

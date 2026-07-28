from pathlib import Path


EXECUTOR_SOURCE = (
    Path(__file__).parents[3]
    / "isaacpjt"
    / "M0609"
    / "execute_isaac.py"
).read_text(encoding="utf-8")


def _source_between(start_marker, end_marker):
    start = EXECUTOR_SOURCE.index(start_marker)
    end = EXECUTOR_SOURCE.index(end_marker, start)
    return EXECUTOR_SOURCE[start:end]


def test_absolute_targets_and_fine_deltas_are_stored_separately():
    callback = _source_between(
        "def _on_target_pose(self, msg: PoseStamped):",
        "def _on_amr_goal_pose",
    )

    assert "FINE_ALIGNMENT_DELTA_FRAME" in callback
    assert ".startswith(" in callback
    assert "self.latest_fine_alignment_delta = msg" in callback
    assert "self.latest_target_pose = msg" in callback


def test_fine_delta_is_integrated_only_behind_a_settle_gate():
    phase = _source_between(
        'elif phase == "FINE_ALIGNMENT":',
        "# [8단계] 버스바 수직 하강 체결",
    )

    consume_at = phase.index(
        "isaac_node.latest_fine_alignment_delta"
    )
    inactive_gate_at = phase.index(
        "fine_step_gate.active_stamp_ns is not None"
    )
    begin_at = phase.index("fine_step_gate.begin(")
    integrate_at = phase.index("target_fine_pos[0] += offset.x")
    update_at = phase.index("fine_step_gate.update(")
    ack_at = phase.index('"FINE_ALIGNMENT_STEP_SETTLED:"')

    assert consume_at < inactive_gate_at < integrate_at < begin_at
    assert begin_at < update_at < ack_at
    assert "FINE_STEP_SETTLE_STEPS = 12" in EXECUTOR_SOURCE
    assert "position_error_m=position_error_m" in phase
    assert "orientation_error_rad=orientation_error_rad" in phase
    assert "position_error_m = math.dist(" in phase
    assert "orientation_error_rad = _quat_angular_error(" in phase
    assert "motion_m=math.dist(" in phase
    assert "orientation_motion_rad=(" in phase
    assert "fine_step_previous_ee_orientation" in phase
    assert "required_position_response_m=(" in phase
    assert "FINE_STEP_MIN_RESPONSE_RATIO = 0.80" in EXECUTOR_SOURCE
    assert (
        "FINE_STEP_POSITION_RESPONSE_NOISE_MARGIN_M = 0.0001"
        in EXECUTOR_SOURCE
    )
    assert (
        "FINE_STEP_ORIENTATION_RESPONSE_NOISE_MARGIN_RAD = "
        "math.radians(0.005)"
        in EXECUTOR_SOURCE
    )
    assert "FINE_STEP_MIN_RESPONSE_RATIO" in phase
    assert "position_response_m=position_response_m" in phase
    assert "orientation_response_rad=(" in phase
    assert "planar_command_response(" in phase
    assert "start_x=fine_step_start_ee_pos[0]" in phase
    assert "command_x=fine_step_command_delta_xy[0]" in phase
    assert (
        "fine_step_start_orientation_error_rad\n"
        "                                - orientation_error_rad"
        in phase
    )
    assert "and not fine_step_started_this_tick" in phase
    assert "FINE_STEP_ORIENTATION_MOTION_TOL_RAD" in EXECUTOR_SOURCE


def test_fine_delta_frame_stamp_bounds_and_yaw_are_validated():
    phase = _source_between(
        'elif phase == "FINE_ALIGNMENT":',
        "# [8단계] 버스바 수직 하강 체결",
    )

    assert "pose_stamp_ns(correction_msg)" in phase
    assert "expected_frame_id = (" in phase
    assert "stale fine-alignment generation" in phase
    assert "abs(offset.x) > 0.0025" in phase
    assert "abs(offset.y) > 0.0025" in phase
    assert "planar_yaw_delta(" in phase
    assert "target_fine_yaw_rad += yaw_delta" in phase
    assert "FAILURE:INVALID_FINE_ALIGNMENT_DELTA" in phase


def test_each_fine_task_sends_a_new_generation_to_error_fix():
    setup = _source_between(
        'elif task == "FINE_ALIGNMENT":',
        'elif task == "ASSEMBLE_BUSBAR":',
    )

    increment_at = setup.index("fine_alignment_generation += 1")
    command_at = setup.index('"START_ERRORFIX_CORRECTION:"')
    clear_at = setup.index(
        "isaac_node.latest_fine_alignment_delta = None"
    )
    publish_at = setup.index(
        "isaac_node.pub_errorfix_command.publish(start_cmd)"
    )
    expected_at = setup.index(
        "isaac_node.expected_fine_alignment_generation = ("
    )
    assert increment_at < command_at < expected_at < clear_at < publish_at


def test_alignment_success_requires_the_active_expected_generation():
    callback = _source_between(
        "def _on_task_command(self, msg: String):",
        "def euler_to_quaternion_wxyz",
    )

    assert 'msg.data == "ALIGNMENT_SUCCESS"' in callback
    assert 'msg.data.startswith("ALIGNMENT_SUCCESS:")' in callback
    assert "generation = int(generation_text)" in callback
    assert "generation 없는 legacy ALIGNMENT_SUCCESS 무시" in callback
    assert "expected_fine_alignment_generation" in callback
    assert "generation != expected_generation" in callback
    assert "stale FINE_ALIGNMENT 완료 신호 무시" in callback
    assert callback.index("generation != expected_generation") < (
        callback.index("self.alignment_success = True")
    )
    assert callback.count("return") >= 5


def test_fine_yaw_is_preserved_during_busbar_insert_and_retract():
    assembly = _source_between(
        'elif phase == "BUSBAR_DESCEND_TO_BOLT":',
        "# ════════════════════════════════════════════════════════════════",
    )

    assert assembly.count("target_fine_yaw_rad") >= 2
    assert assembly.count("euler_to_quaternion_wxyz(") >= 2


def test_cancel_and_playback_restart_clear_pending_fine_step():
    restart = _source_between(
        "if playing and not was_playing:",
        "# battery4_main의 너트 AMR glue 추종",
    )
    cancel = _source_between(
        'if task == "CANCEL_ARM_TASK":',
        "# 안전장치: 팔 Task는 AMR이 도착해",
    )

    for source in (restart, cancel):
        assert "fine_step_gate.reset()" in source
        assert "fine_controller_lead.reset()" in source
        assert "fine_step_previous_ee_pos = None" in source
        assert "fine_step_previous_ee_orientation = None" in source
        assert "fine_step_start_ee_pos = None" in source
        assert "fine_step_command_delta_xy = None" in source
        assert "fine_step_start_orientation_error_rad = None" in source
        assert "latest_fine_alignment_delta = None" in source
        assert "expected_fine_alignment_generation = None" in source


def test_fine_failure_and_success_clear_expected_completion_generation():
    phase = _source_between(
        'elif phase == "FINE_ALIGNMENT":',
        "# [8단계] 버스바 수직 하강 체결",
    )

    invalid = phase[
        phase.index("FAILURE:INVALID_FINE_ALIGNMENT_DELTA"):
        phase.index("else:", phase.index(
            "FAILURE:INVALID_FINE_ALIGNMENT_DELTA"))
    ]
    success = phase[
        phase.index("FINE_ALIGNMENT_COMPLETE"):
        phase.index("# [8단계]")
        if "# [8단계]" in phase
        else len(phase)
    ]
    assert "expected_fine_alignment_generation" in invalid
    assert "expected_fine_alignment_generation = None" in success


def test_fine_controller_lead_changes_only_the_bounded_xy_command():
    phase = _source_between(
        'elif phase == "FINE_ALIGNMENT":',
        "# [8단계] 버스바 수직 하강 체결",
    )
    assembly_setup = _source_between(
        'elif task == "ASSEMBLE_BUSBAR":',
        'elif task in ("SCAN_NUT1", "SCAN_NUT2"):',
    )

    assert "FINE_CONTROLLER_LEAD_GROWTH_M = 0.00005" in (
        EXECUTOR_SOURCE
    )
    assert "FINE_CONTROLLER_MAX_LEAD_M = 0.001" in EXECUTOR_SOURCE
    assert "fine_controller_lead = BoundedPlanarCommandLead(" in (
        EXECUTOR_SOURCE
    )
    assert "fine_controller_lead.reset()" in phase
    assert "fine_controller_lead.begin(" in phase
    assert (
        "required_position_response_m = (\n"
        "                            "
        "fine_step_gate.required_position_response_m"
    ) in phase
    assert "fine_controller_lead.advance(" in phase
    assert "observed_response_m=position_response_m" in phase
    assert "fine_controller_target_pos = target_fine_pos.copy()" in phase
    assert "fine_controller_target_pos[0] = controller_x" in phase
    assert "fine_controller_target_pos[1] = controller_y" in phase
    assert (
        "target_end_effector_position=(\n"
        "                            fine_controller_target_pos"
    ) in phase
    assert "position_error_m = math.dist(" in phase
    assert "cur_ee_pos,\n                            target_fine_pos" in phase
    assert "target_fine_pos[0] += lead" not in phase
    assert "target_fine_pos[1] += lead" not in phase
    assert "target_fine_yaw_rad += yaw_delta" in phase
    assert "logical target/Z/yaw unchanged until settle" in phase
    assert "controller XY lead가 안전 " in phase
    assert "실제 80% 응답 전에는 ACK하지 " in phase
    assert "[FINE_ALIGNMENT RESPONSE] actual=" in phase
    assert "position_response_m * 1000" in phase
    assert "required_position_response_m * 1000" in phase
    assert "lead_status.magnitude_m * 1000" in phase
    assert "saturated={lead_status.saturated}" in phase
    assert "step_count % 60 == 0" in phase

    # Strictly settled controller XY becomes the canonical target before ACK;
    # assembly then consumes that canonical target without adding lead twice.
    assert "fine_controller_lead.commit_xy(" in phase
    assert "target_fine_pos[0]" in assembly_setup
    assert "target_fine_pos[1]" in assembly_setup
    assert "fine_controller_target_pos" not in assembly_setup


def test_fine_settle_commits_effective_xy_before_ack_without_gate_relaxation():
    phase = _source_between(
        'elif phase == "FINE_ALIGNMENT":',
        "# [8단계] 버스바 수직 하강 체결",
    )
    settled = phase[
        phase.index("if settle_status.settled:"):
        phase.index("if (", phase.index("if settle_status.settled:"))
    ]

    commit_at = settled.index("fine_controller_lead.commit_xy(")
    target_x_at = settled.index("target_fine_pos[0] = committed_x")
    target_y_at = settled.index("target_fine_pos[1] = committed_y")
    publish_at = settled.index(
        "isaac_node.pub_errorfix_command.publish("
    )
    assert commit_at < target_x_at < target_y_at < publish_at
    assert "fine_controller_lead.reset()" not in settled
    assert "position_response_m" in settled
    assert "required_position_response_m" in settled
    assert "committed lead=" in settled
    assert "fine_step_gate.update(" in phase
    assert "FINE_STEP_MIN_RESPONSE_RATIO = 0.80" in EXECUTOR_SOURCE
    assert "FINE_STEP_SETTLE_STEPS = 12" in EXECUTOR_SOURCE
    assert "FINE_STEP_POSITION_TOL_M = 0.005" in EXECUTOR_SOURCE
    assert (
        "FINE_STEP_POSITION_RESPONSE_NOISE_MARGIN_M = 0.0001"
        in EXECUTOR_SOURCE
    )


def test_every_fine_step_reset_clears_motion_and_progress_state():
    reset_marker = "fine_step_gate.reset()"
    cursor = 0
    reset_count = 0
    required_clears = (
        "fine_controller_lead.reset()",
        "fine_step_previous_ee_pos = None",
        "fine_step_previous_ee_orientation = None",
        "fine_step_start_ee_pos = None",
        "fine_step_command_delta_xy = None",
        "fine_step_start_orientation_error_rad = None",
    )

    while True:
        reset_at = EXECUTOR_SOURCE.find(reset_marker, cursor)
        if reset_at < 0:
            break
        reset_count += 1
        reset_window = EXECUTOR_SOURCE[
            reset_at:reset_at + 700
        ]
        for clear in required_clears:
            assert clear in reset_window
        cursor = reset_at + len(reset_marker)

    assert reset_count >= 6

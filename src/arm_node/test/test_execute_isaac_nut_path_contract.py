import re
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


def test_nut_scan_and_retract_wait_for_actual_pose_to_settle():
    scan_lift = _source_between(
        'elif phase == "NUT_SCAN_LIFT":',
        'elif phase == "NUT_SCAN_APPROACH":',
    )
    retract_rotate = _source_between(
        'elif phase == "NUT_RETRACT_ROTATE":',
        'elif phase == "RETURN_HOME_JOINTS":',
    )

    assert re.search(
        r"^NUT_POSE_POSITION_TOL_M\s*=\s*0\.02\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )
    assert re.search(
        r"^NUT_POSE_ORIENTATION_TOL_RAD\s*="
        r"\s*math\.radians\(3\.0\)\s*$",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )
    assert re.search(
        r"^NUT_POSE_SETTLE_STEPS\s*=\s*12\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )

    for phase_source in (scan_lift, retract_rotate):
        assert "robot.end_effector.get_world_pose()" in phase_source
        assert "_quat_angular_error(" in phase_source
        assert "consecutive_pose_settle(" in phase_source
        assert "NUT_POSE_POSITION_TOL_M" in phase_source
        assert "NUT_POSE_ORIENTATION_TOL_RAD" in phase_source
        assert "NUT_POSE_SETTLE_STEPS" in phase_source
        assert "if pose_settled:" in phase_source
        assert "step_count > MAX_STUCK_STEPS" not in phase_source

    assert scan_lift.index("if pose_settled:") < scan_lift.index(
        'phase = "NUT_SCAN_APPROACH"'
    )
    assert retract_rotate.index("if pose_settled:") < (
        retract_rotate.index('publish_status("SUCCESS")')
    )


def test_nut_pose_settle_counter_is_reset_at_all_phase_boundaries():
    scan_setup = _source_between(
        'elif task in ("SCAN_NUT1", "SCAN_NUT2"):',
        'elif task in ("PICK_NUT1", "PICK_NUT2"):',
    )
    unwind = _source_between(
        'elif phase == "NUT_RETRACT_UNWIND":',
        'elif phase == "NUT_RETRACT_ROTATE":',
    )
    restart = _source_between(
        "if playing and not was_playing:",
        "# battery4_main의 너트 AMR glue 추종",
    )
    cancel = _source_between(
        'if task == "CANCEL_ARM_TASK":',
        "# 안전장치: 팔 Task는 AMR이 도착해",
    )
    conflict = _source_between(
        "# ArmNode는 ExecuteArmTask를 직렬화하지만 /task_command 자체는",
        'if task == "SCAN_BATTERY":',
    )

    for source in (scan_setup, unwind, restart, cancel, conflict):
        assert "nut_pose_settle_count = 0" in source


def test_six_nuts_map_to_exactly_their_own_tray_pegs():
    specs = _source_between(
        "NUT_EXACT_PEG_FILTER_SPECS = (",
        "# 그리퍼 파라미터",
    )
    configure = _source_between(
        "def _configure_nut_exact_peg_filter(",
        "def _all_nut_collision_filter_failures(stage):",
    )

    expected_peg_paths = {
        1: "peg_5",
        2: "peg_4",
        3: "peg_3",
        4: "peg_1",
        5: "peg_0",
        6: "peg_2",
    }
    assert specs.count('"NUT') == 6
    for nut_index, peg_name in expected_peg_paths.items():
        assert f'"NUT{nut_index}"' in specs
        assert (
            f'"/World/Nova_Carter/chassis_link/carter_tray/{peg_name}"'
            in EXECUTOR_SOURCE
        )

    clear_index = configure.index("legacy_relation.SetTargets([])")
    apply_index = configure.index(
        "UsdPhysics.FilteredPairsAPI.Apply(body_prim)"
    )
    exact_index = configure.index(
        "relation.SetTargets([Sdf.Path(target_paths[0])])"
    )
    assert clear_index < apply_index < exact_index
    assert "AddTarget(" not in configure
    assert "M0609" not in configure
    assert "left_inner_finger" not in configure
    assert "right_inner_finger" not in configure


def test_six_exact_filters_are_fail_closed_and_preflight_visible():
    validate_filter = _source_between(
        "def _nut_collision_filter_failures(",
        "def _configure_nut_exact_peg_filter(",
    )
    configure = _source_between(
        "def _configure_nut_exact_peg_filter(",
        "def _all_nut_collision_filter_failures(stage):",
    )
    all_filters = _source_between(
        "def _all_nut_collision_filter_failures(stage):",
        "def create_runtime_stage_overlay(",
    )
    preflight = _source_between(
        "def validate_stage_contract(stage):",
        "def run_wheel_smoke_test(",
    )

    assert "{label} 실제 rigid body prim 누락" in validate_filter
    assert "{label} 간섭 peg prim 누락" in validate_filter
    assert "actual_targets != expected_targets" in validate_filter
    assert "legacy_targets" in validate_filter
    assert "예상하지 않은 {label} filteredPairs" in validate_filter
    assert "if missing:" in configure
    assert "{label} exact collision-filter prim 누락" in configure
    assert "failures = _nut_collision_filter_failures(" in configure
    assert (
        "for label, root_path, body_path, peg_path "
        "in NUT_EXACT_PEG_FILTER_SPECS:"
    ) in all_filters
    assert "_nut_collision_filter_failures(" in all_filters
    assert "_configure_nut_exact_peg_filter(" in all_filters
    assert (
        "failures.extend(_all_nut_collision_filter_failures(stage))"
        in preflight
    )
    assert "six exact nut/own-peg collision pairs" in preflight


def test_six_exact_filters_are_authored_into_runtime_overlay_before_open():
    overlay = _source_between(
        "def create_runtime_stage_overlay(",
        "class Execute_Isaac_Busar(",
    )
    main = _source_between(
        "def main():",
        "\n\nif __name__ ==",
    )

    configure_index = overlay.index(
        "configure_exact_nut_peg_filters(stage)"
    )
    save_index = overlay.index("layer.Save()")
    assert configure_index < save_index
    assert main.index("create_runtime_stage_overlay(") < main.index(
        "ctx.open_stage("
    )
    assert main.index("ctx.open_stage(") < main.index(
        "world = World("
    )


def test_nut_assembly_builds_separate_horizontal_and_vertical_targets():
    task_setup = _source_between(
        'elif task in ("ASSEMBLE_NUT1", "ASSEMBLE_NUT2"):',
        "# 3. FSM 제어 루프",
    )

    assert "bolt_travel_pos = np.array([" in task_setup
    assert "NUT_APPROACH_Z," in task_setup
    assert "bolt_approach_pos = np.array([" in task_setup
    assert "BOLT_APPROACH_Z," in task_setup
    assert 'phase = "MOVE_TO_BOLT_NUT"' in task_setup


def test_nut_assembly_moves_xy_before_descending_to_bolt_approach():
    horizontal = _source_between(
        'elif phase == "MOVE_TO_BOLT_NUT":',
        'elif phase == "MOVE_TO_BOLT_NUT_VERTICAL":',
    )
    vertical = _source_between(
        'elif phase == "MOVE_TO_BOLT_NUT_VERTICAL":',
        'elif phase == "NUT_DESCEND_TO_BOLT":',
    )

    assert "bolt_travel_command, nut_xy_lead = lead_xy_target(" in horizontal
    assert "target_end_effector_position=bolt_travel_command" in horizontal
    assert "math.dist(cur_pos, tuple(bolt_travel_pos))" in horizontal
    assert 'publish_progress("MOVE_TO_BOLT", 20.0)' in horizontal
    assert 'phase = "MOVE_TO_BOLT_NUT_VERTICAL"' in horizontal

    assert "bolt_approach_command, nut_xy_lead = lead_xy_target(" in vertical
    assert "target_end_effector_position=bolt_approach_command" in vertical
    assert "math.dist(cur_pos, tuple(bolt_approach_pos))" in vertical
    assert 'publish_progress("MOVE_TO_BOLT", 30.0)' in vertical
    assert 'phase = "NUT_DESCEND_TO_BOLT"' in vertical

    for phase_source in (horizontal, vertical):
        assert "current_err < PICK_TOLERANCE_STRICT" in phase_source
        assert "step_count > MAX_STUCK_STEPS" in phase_source

    assert re.search(
        r"^MAX_STUCK_STEPS\s*=\s*1_?000_?000\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )


def test_nut_motion_completion_is_measured_against_unmodified_goal():
    horizontal = _source_between(
        'elif phase == "MOVE_TO_BOLT_NUT":',
        'elif phase == "MOVE_TO_BOLT_NUT_VERTICAL":',
    )
    vertical = _source_between(
        'elif phase == "MOVE_TO_BOLT_NUT_VERTICAL":',
        'elif phase == "NUT_DESCEND_TO_BOLT":',
    )

    assert "math.dist(cur_pos, tuple(bolt_travel_command))" not in horizontal
    assert "math.dist(cur_pos, tuple(bolt_approach_command))" not in vertical


def test_peg_clear_uses_actual_80mm_displacement_for_15_ticks():
    clear_phase = _source_between(
        'elif phase == "NUT_LIFT_CLEAR_PEG":',
        'elif phase == "NUT_LIFT":',
    )

    assert "lead_xy_target(" in clear_phase
    assert "lead_z_target(" in clear_phase
    assert "target_end_effector_position=nut_peg_clear_command" in clear_phase
    assert (
        "cur_pos[2] - nut_peg_clear_start_pos[2]"
        in clear_phase
    )
    assert (
        "nut_now_pos[2] - nut_peg_clear_start_nut_z"
        in clear_phase
    )
    assert "actual_lift >= NUT_PEG_CLEARANCE_Z" in clear_phase
    assert "actual_nut_lift >= NUT_PEG_CLEARANCE_Z" in clear_phase
    assert "xy_err < PICK_TOLERANCE_STRICT" in clear_phase
    assert "nut_peg_clear_hold += 1" in clear_phase
    assert "nut_peg_clear_hold >= NUT_PEG_CLEAR_HOLD_STEPS" in clear_phase
    assert "NUT_PEG_CLEAR_TOLERANCE" not in clear_phase
    assert (
        "actual_nut_lift + NUT_PEG_CLEAR_COMMAND_MARGIN_M"
        not in clear_phase
    )


def test_peg_clear_margin_changes_only_the_command_target():
    grasp_phase = _source_between(
        'elif phase == "NUT_GRASP":',
        'elif phase == "NUT_LIFT_CLEAR_PEG":',
    )
    clear_phase = _source_between(
        'elif phase == "NUT_LIFT_CLEAR_PEG":',
        'elif phase == "NUT_LIFT":',
    )

    assert (
        "command_clear_z = (\n"
        "                        clear_z + "
        "NUT_PEG_CLEAR_COMMAND_MARGIN_M"
    ) in grasp_phase
    assert "elif command_clear_z > NUT_APPROACH_Z:" in grasp_phase
    assert (
        "nut_peg_clear_tracking_pos[2] += (\n"
        "                        NUT_PEG_CLEAR_COMMAND_MARGIN_M"
    ) in clear_phase
    assert (
        "lead_z_target(\n"
        "                        nut_peg_clear_tracking_pos,"
    ) in clear_phase
    assert re.search(
        r"^NUT_PEG_CLEAR_COMMAND_MARGIN_M\s*=\s*0\.002\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )
    assert re.search(
        r"^NUT_EE_COUPLING_TOLERANCE_M\s*=\s*0\.01\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )


def test_nut_vertical_lift_target_is_rate_limited_without_weakening_goal():
    clear_phase = _source_between(
        'elif phase == "NUT_LIFT_CLEAR_PEG":',
        'elif phase == "NUT_LIFT":',
    )
    lift_phase = _source_between(
        'elif phase == "NUT_LIFT":',
        'elif phase == "MOVE_TO_BOLT_NUT":',
    )

    for phase_source in (clear_phase, lift_phase):
        assert "rate_limited_z_target(" in phase_source
        assert "max_step=NUT_LIFT_COMMAND_MAX_STEP_M" in phase_source

    assert (
        "nut_peg_clear_command[2] = nut_lift_command_z"
        in clear_phase
    )
    assert "nut_lift_command[2] = nut_lift_command_z" in lift_phase
    assert "math.dist(cur_pos, tuple(nut_approach_pos))" in lift_phase
    assert "math.dist(cur_pos, tuple(nut_lift_command))" not in lift_phase
    assert re.search(
        r"^NUT_LIFT_COMMAND_MAX_STEP_M\s*=\s*0\.0005\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )
    assert re.search(
        r"^NUT_PEG_CLEARANCE_Z\s*=\s*0\.08\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )
    assert re.search(
        r"^NUT_PEG_CLEAR_HOLD_STEPS\s*=\s*15\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )
    assert re.search(
        r"^REGRASP_LIFT_HEIGHT\s*=\s*0\.06\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )


def test_pick_nut_keeps_tray_glue_until_fingers_are_fully_closed():
    pick_dispatch = _source_between(
        'elif task in ("PICK_NUT1", "PICK_NUT2"):',
        'elif task in ("ASSEMBLE_NUT1", "ASSEMBLE_NUT2"):',
    )
    grasp_phase = _source_between(
        'elif phase == "NUT_GRASP":',
        'elif phase == "NUT_LIFT_CLEAR_PEG":',
    )
    descend_phase = _source_between(
        'elif phase == "NUT_DESCEND":',
        'elif phase == "NUT_GRASP":',
    )

    assert "enable_physics_recursively(" not in pick_dispatch
    assert "nut_released[nut_array_index] = True" not in pick_dispatch
    assert "enable_collision_recursively(" in descend_phase
    assert "enable_physics_recursively(" not in descend_phase
    close_gate = grasp_phase.index(
        "if grasp_timer >= GRIP_CLOSE_RAMP_STEPS:"
    )
    release_index = grasp_phase.index(
        "enable_physics_recursively("
    )
    released_flag_index = grasp_phase.index(
        "nut_released[nut_pick_active_array_index] = True"
    )
    assert close_gate < release_index < released_flag_index
    assert "nut_release_timer >= GRIP_SETTLE_STEPS" in grasp_phase


def test_release_settles_before_transport_reference_is_latched():
    grasp_phase = _source_between(
        'elif phase == "NUT_GRASP":',
        'elif phase == "NUT_LIFT_CLEAR_PEG":',
    )

    release_limit = grasp_phase.index(
        "> NUT_EE_COUPLING_TOLERANCE_M"
    )
    motion_gate = grasp_phase.index(
        "settle_motion\n                                "
        "<= NUT_EE_SETTLE_MOTION_TOLERANCE_M"
    )
    stable_ticks = grasp_phase.index(
        "nut_release_timer >= GRIP_SETTLE_STEPS"
    )
    transport_latch = grasp_phase.index(
        "# release 순간의 preload 기준이 아니라"
    )
    peg_clear = grasp_phase.index(
        'phase = "NUT_LIFT_CLEAR_PEG"'
    )

    assert (
        release_limit
        < motion_gate
        < stable_ticks
        < transport_latch
        < peg_clear
    )
    assert "nut_release_timer = 0" in grasp_phase[motion_gate:stable_ticks]
    assert "nut_ee_reference_offset = (" in grasp_phase[
        transport_latch:peg_clear
    ]
    assert re.search(
        r"^NUT_EE_COUPLING_TOLERANCE_M\s*=\s*0\.01\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )


def test_cancel_and_task_conflict_restore_unfinished_nut_pick():
    cancel_dispatch = _source_between(
        'if task == "CANCEL_ARM_TASK":',
        "# 안전장치: 팔 Task는 AMR이 도착해",
    )
    conflict_dispatch = _source_between(
        "# ArmNode는 ExecuteArmTask를 직렬화하지만 /task_command 자체는",
        'if task == "SCAN_BATTERY":',
    )
    restore_helper = _source_between(
        "def restore_active_nut_to_tray_glue(reason):",
        "def publish_status(status_str: str):",
    )

    assert "restore_active_nut_to_tray_glue(" in cancel_dispatch
    assert "restore_active_nut_to_tray_glue(" in conflict_dispatch
    physics_off = restore_helper.index("disable_physics_recursively(")
    pose_restore = restore_helper.index("nut_xf.set_world_pose(")
    glue_owner = restore_helper.index(
        "nut_released[array_index] = False"
    )
    assert physics_off < pose_restore < glue_owner
    assert "nut_local_offsets[array_index]" in restore_helper


def test_peg_clear_and_lift_require_ee_relative_nut_coupling():
    clear_phase = _source_between(
        'elif phase == "NUT_LIFT_CLEAR_PEG":',
        'elif phase == "NUT_LIFT":',
    )
    lift_phase = _source_between(
        'elif phase == "NUT_LIFT":',
        'elif phase == "MOVE_TO_BOLT_NUT":',
    )

    coupling_index = clear_phase.index("nut_ee_coupling_error(")
    drift_gate_index = clear_phase.index(
        "coupling_error\n                        "
        "> NUT_EE_COUPLING_TOLERANCE_M"
    )
    hold_index = clear_phase.index("nut_peg_clear_hold += 1")
    assert coupling_index < drift_gate_index < hold_index
    assert "FAILURE:NUT_GRIP_COUPLING_LOST" in clear_phase

    lift_coupling_index = lift_phase.index(
        "nut_ee_coupling_error("
    )
    lift_drift_gate_index = lift_phase.index(
        "coupling_error\n                        "
        "> NUT_EE_COUPLING_TOLERANCE_M"
    )
    success_index = lift_phase.index(
        'publish_status("SUCCESS")'
    )
    assert (
        lift_coupling_index
        < lift_drift_gate_index
        < success_index
    )
    assert "restore_active_nut_to_tray_glue(" in lift_phase


def test_playback_restart_aborts_every_active_phase():
    restart = _source_between(
        "if playing and not was_playing:",
        "# battery4_main의 너트 AMR glue 추종",
    )

    assert "is_playback_restart = playback_started_once" in restart
    assert 'phase if phase not in {"IDLE", "DONE"} else None' in restart
    restart_gate = restart.index("if is_playback_restart:")
    interrupted_gate = restart.index("if interrupted_phase is not None:")
    for transient_clear in (
        'phase = "IDLE"',
        "isaac_node.requested_task = None",
        "isaac_node.latest_target_pose = None",
        "isaac_node.alignment_success = False",
        "pub_wrist_perception_reset.publish(Empty())",
        "pub_busbar_perception_reset.publish(Empty())",
        "pub_bolt_perception_reset.publish(Empty())",
        "RESET_BOLT_DETECTION",
    ):
        clear_index = restart.index(transient_clear)
        assert restart_gate < clear_index < interrupted_gate
    assert 'FAILURE:PLAYBACK_RESTART_DURING_' in restart
    assert 'publish_status("FAILURE:PLAYBACK_RESTART")' in restart


def test_busbar_wrist_scan_uses_selected_mesh_not_relative_ee_offset():
    scan_setup = _source_between(
        'elif task == "CONTINUE_BUSBAR_WRIST_SCAN":',
        'elif task == "PICK_BUSBAR":',
    )

    assert "asset = busbar_assets.get(current_station)" in scan_setup
    assert 'mesh_center, _ = asset["xform"].get_world_pose()' in scan_setup
    assert "BUSBAR_SCAN_POS = busbar_wrist_scan_target(" in scan_setup
    assert "cur_pos[0] - 0.5" not in scan_setup
    assert "STATION_BUSBAR_SCAN_XY" not in scan_setup


def test_scan_battery_moves_bolt_camera_to_current_station_bolt_2():
    scan_setup = _source_between(
        'if task == "SCAN_BATTERY":',
        'elif task == "RETURN_HOME":',
    )

    assert "bolt_camera_pose_for_target(" in scan_setup
    assert "bolt_paths[1]" in scan_setup
    assert "bolt_camera_pose_for_pair(" not in scan_setup


def test_busbar_tasks_fail_fast_when_amr_is_not_at_busbar_stop():
    scan_setup = _source_between(
        'elif task == "SCAN_BUSBAR":',
        'elif task == "CONTINUE_BUSBAR_WRIST_SCAN":',
    )
    pick_setup = _source_between(
        'elif task == "PICK_BUSBAR":',
        'elif task == "MOVE_BATTERY_CENTER":',
    )

    for task_setup in (scan_setup, pick_setup):
        assert (
            "busbar_station_pose_error(robot, current_station)"
            in task_setup
        )
        assert "last_arrived_busbar_station != current_station" in task_setup
        assert "busbar_amr_position_error > AMR_POS_TOL" in task_setup
        assert "busbar_amr_yaw_error > AMR_YAW_TOL" in task_setup
        assert "BUSBAR_AMR_POSITION_MAX_M" not in task_setup
        assert "BUSBAR_AMR_YAW_MAX_RAD" not in task_setup
        assert "FAILURE:AMR_NOT_AT_BUSBAR_STATION" in task_setup
        assert "hold_current_arm_pose()" in task_setup


def test_busbar_motion_reanchors_rmpflow_to_actual_amr_pose():
    continue_setup = _source_between(
        'elif task == "CONTINUE_BUSBAR_WRIST_SCAN":',
        'elif task == "PICK_BUSBAR":',
    )
    pick_setup = _source_between(
        'elif task == "PICK_BUSBAR":',
        'elif task == "MOVE_BATTERY_CENTER":',
    )
    restart_setup = _source_between(
        "if playing and not was_playing:",
        "# battery4_main의 너트 AMR glue 추종",
    )
    hold_setup = _source_between(
        "def hold_current_arm_pose():",
        "\n    sync_rmpflow_base_pose()\n\n    print",
    )

    for task_setup in (continue_setup, pick_setup):
        reset_index = task_setup.index("arm_controller.reset()")
        sync_index = task_setup.index("sync_rmpflow_base_pose()")
        assert reset_index < sync_index

    reset_index = hold_setup.index("arm_controller.reset()")
    sync_index = hold_setup.index("sync_rmpflow_base_pose()")
    assert reset_index < sync_index

    hold_index = restart_setup.index("hold_current_arm_pose()")
    interrupted_index = restart_setup.index(
        "if interrupted_phase is not None:"
    )
    assert hold_index < interrupted_index


def test_active_arm_phase_rejects_conflicting_task_and_holds_pose():
    dispatch = _source_between(
        "# ArmNode는 ExecuteArmTask를 직렬화하지만 /task_command 자체는",
        'if task == "SCAN_BATTERY":',
    )

    assert 'task == "CONTINUE_BUSBAR_WRIST_SCAN"' in dispatch
    assert 'phase == "WAIT_BUSBAR_CAMERA_LATCH"' in dispatch
    assert 'phase not in {"IDLE", "DONE"}' in dispatch
    hold_index = dispatch.index("hold_current_arm_pose()")
    idle_index = dispatch.index('phase = "IDLE"')
    failure_index = dispatch.index(
        'publish_status("FAILURE:ARM_TASK_CONFLICT")'
    )
    assert hold_index < idle_index < failure_index

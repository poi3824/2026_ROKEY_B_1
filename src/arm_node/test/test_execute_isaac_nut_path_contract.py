import ast
import math
import re
from types import SimpleNamespace
from pathlib import Path

import numpy as np


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


def _load_nut_contact_helpers():
    helper_names = {
        "nut_torque_contact_geometry",
        "retained_nut_regrasp_target",
    }
    tree = ast.parse(EXECUTOR_SOURCE)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in helper_names
    ]
    namespace = {
        "np": np,
        "NUT_BODY_HEIGHT_M": 0.00847725,
        "NUT_THREAD_ENTRY_CLEARANCE_M": 0.0002,
        "NUT_EE_Z_COUPLING_TOLERANCE_M": 0.001,
        "NUT_TORQUE_TCP_Z_TOL_M": 0.001,
        "NUT_SEATING_SURFACE_TOL_M": 0.0003,
    }
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            filename="<execute_isaac_nut_contact_helpers>",
            mode="exec",
        ),
        namespace,
    )
    return (
        namespace["nut_torque_contact_geometry"],
        namespace["retained_nut_regrasp_target"],
    )


def _load_screw_axis_gate():
    tree = ast.parse(EXECUTOR_SOURCE)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "axis_gated_screw_theta"
    ]
    assert len(selected) == 1
    namespace = {"math": math, "np": np}
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            filename="<execute_isaac_screw_axis_gate>",
            mode="exec",
        ),
        namespace,
    )
    return namespace["axis_gated_screw_theta"]


def _load_bolt_axis_geometry():
    tree = ast.parse(EXECUTOR_SOURCE)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "bolt_axis_geometry_from_transform"
    ]
    assert len(selected) == 1

    def project(axis, reference_z):
        distance = (reference_z - axis.origin_m[2]) / axis.direction[2]
        return (
            axis.origin_m[0] + distance * axis.direction[0],
            axis.origin_m[1] + distance * axis.direction[1],
            reference_z,
        )

    namespace = {
        "np": np,
        "math": math,
        "NUT_BOLT_AXIS_TOL_RAD": math.radians(3.0),
        "BOLT_THREAD_BASE_FROM_BODY_M": -0.01950137,
        "BOLT_THREAD_TIP_FROM_BODY_M": 0.02850200,
        "extract_local_z_world_axis": (
            lambda transform, **_kwargs: transform
        ),
        "project_world_axis_to_z": project,
    }
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            filename="<execute_isaac_bolt_axis_geometry>",
            mode="exec",
        ),
        namespace,
    )
    return namespace["bolt_axis_geometry_from_transform"]


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


def test_scan_nut_reanchors_rmpflow_before_capturing_lift_target():
    scan_setup = _source_between(
        'elif task in ("SCAN_NUT1", "SCAN_NUT2"):',
        'elif task in ("PICK_NUT1", "PICK_NUT2"):',
    )

    reset_index = scan_setup.index("arm_controller.reset()")
    sync_index = scan_setup.index("sync_rmpflow_base_pose()")
    current_pose_index = scan_setup.index(
        "cur_pos = world_xf("
    )
    phase_index = scan_setup.index('phase = "NUT_SCAN_LIFT"')
    assert reset_index < sync_index < current_pose_index < phase_index


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


def test_threaded_nut_and_bolt_collision_sdf_is_uniform_and_preflight_visible():
    constants = _source_between(
        "THREADED_FASTENER_SDF_RESOLUTION = 256",
        "# 각 nut의 SDF hole",
    )
    configure = _source_between(
        "def configure_threaded_fastener_sdf_collisions(stage):",
        "def set_station_battery_kinematic(",
    )
    validate = _source_between(
        "def _threaded_fastener_sdf_failures(stage):",
        "def configure_threaded_fastener_sdf_collisions(stage):",
    )
    overlay = _source_between(
        "def create_runtime_stage_overlay(",
        "class Execute_Isaac_Busar(",
    )
    preflight = _source_between(
        "def validate_stage_contract(stage):",
        "def run_wheel_smoke_test(",
    )

    assert "THREADED_FASTENER_SDF_RESOLUTION = 256" in constants
    assert "STATION_BOLT_BODY_PATHS.values()" in constants
    assert "NUT1_POLYSHAPE_PATH" in constants
    assert "NUT2_POLYSHAPE_PATH" in constants
    assert "*EXTRA_NUT_POLYSHAPE_PATHS" in constants

    assert "len(paths) != 12" in validate
    assert "len(set(paths)) != 12" in validate
    assert "UsdPhysics.MeshCollisionAPI(prim)" in validate
    assert "PhysxSchema.PhysxSDFMeshCollisionAPI(prim)" in validate
    assert "THREADED_FASTENER_COLLISION_APPROXIMATION" in validate
    assert "THREADED_FASTENER_SDF_RESOLUTION" in validate

    assert "UsdPhysics.MeshCollisionAPI.Apply(prim)" in configure
    assert "CreateApproximationAttr(" in configure
    assert "THREADED_FASTENER_COLLISION_APPROXIMATION" in configure
    assert "PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)" in configure
    assert "CreateSdfResolutionAttr(" in configure
    assert "_threaded_fastener_sdf_failures(stage)" in configure

    assert "configure_threaded_fastener_sdf_collisions(stage)" in overlay
    assert "_threaded_fastener_sdf_failures(stage)" in preflight


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
    assert "bolt_approach_pos = bolt_travel_pos.copy()" in task_setup
    assert "bolt_approach_pos[2] = BOLT_APPROACH_Z" in task_setup
    assert 'phase = "MOVE_TO_BOLT_NUT"' in task_setup


def test_bolt_axis_geometry_projects_a_tilted_live_thread_not_its_origin():
    geometry = _load_bolt_axis_geometry()
    axis = SimpleNamespace(
        origin_m=(1.0, -0.5, 0.2),
        direction=(0.02, -0.01, math.sqrt(1.0 - 0.02**2 - 0.01**2)),
        tilt_rad=math.radians(1.28),
    )

    target, thread_base, thread_tip, returned_axis = geometry(axis, 0.5)

    expected_scale = (0.5 - axis.origin_m[2]) / axis.direction[2]
    np.testing.assert_allclose(
        target,
        np.array([
            axis.origin_m[0] + expected_scale * axis.direction[0],
            axis.origin_m[1] + expected_scale * axis.direction[1],
            0.5,
        ]),
    )
    np.testing.assert_allclose(
        thread_base,
        np.array(axis.origin_m)
        + (-0.01950137) * np.array(axis.direction),
    )
    np.testing.assert_allclose(
        thread_tip,
        np.array(axis.origin_m)
        + 0.02850200 * np.array(axis.direction),
    )
    assert returned_axis is axis

    try:
        geometry(axis, float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite bolt-axis reference must fail")


def test_nut_assembly_uses_full_affine_axis_for_every_thread_phase():
    task_setup = _source_between(
        'elif task in ("ASSEMBLE_NUT1", "ASSEMBLE_NUT2"):',
        "# 3. FSM 제어 루프",
    )
    vertical = _source_between(
        'elif phase == "MOVE_TO_BOLT_NUT_VERTICAL":',
        'elif phase == "NUT_DESCEND_TO_BOLT":',
    )
    descend = _source_between(
        'elif phase == "NUT_DESCEND_TO_BOLT":',
        'elif phase == "NUT_SCREW":',
    )
    rotate = _source_between(
        'if screw_sub == "rotate":',
        'elif screw_sub == "release":',
    )
    regrasp = _source_between(
        'elif screw_sub == "regrasp":',
        'elif phase == "NUT_SEATING_VERIFY":',
    )

    helper = _source_between(
        "def bolt_axis_geometry_from_transform(",
        "def live_bolt_axis_geometry(",
    )
    assert "extract_local_z_world_axis(" in helper
    assert "project_world_axis_to_z(" in helper
    assert "BOLT_THREAD_BASE_FROM_BODY_M" in helper
    assert "BOLT_THREAD_TIP_FROM_BODY_M" in helper
    assert "NUT_BOLT_AXIS_TOL_RAD" in helper

    for phase_source in (task_setup, vertical, descend, rotate, regrasp):
        assert "live_bolt_axis_geometry(" in phase_source
    assert "_nut_thread_overlap_from_limits(" in rotate
    assert "project_world_axis_to_z(" in descend
    assert "project_world_axis_to_z(" in rotate


def test_nut_thread_entry_and_seating_depth_follow_live_geometry():
    task_setup = _source_between(
        'elif task in ("ASSEMBLE_NUT1", "ASSEMBLE_NUT2"):',
        "# 3. FSM 제어 루프",
    )

    assert "bolt_thread_tip_pos" in task_setup
    assert "live_bolt_axis_geometry(" in task_setup
    assert "thread_entry_ee_z(" in task_setup
    assert "NUT_THREAD_ENTRY_CLEARANCE_M" in task_setup
    assert "observe_latched_busbar_seating(" in task_setup
    assert "required_seating_depth_to_surface(" in task_setup
    assert "nut_required_seating_depth_m" in task_setup
    assert re.search(
        r"nut_raw_required_seating_depth_m\s*"
        r">\s*NUT_COMMANDED_DEPTH_M",
        task_setup,
    )
    assert "FAILURE:NUT_THREAD_ENTRY_GEOMETRY_INVALID" in task_setup
    assert "0.3697" not in task_setup
    assert re.search(
        r"^NUT_THREAD_ENTRY_CLEARANCE_M\s*=\s*0\.0002\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )
    assert re.search(
        r"^REGRASP_CYCLES\s*=\s*5\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )


def test_nut_assembly_refuses_to_hide_busbar_misalignment():
    task_setup = _source_between(
        'elif task in ("ASSEMBLE_NUT1", "ASSEMBLE_NUT2"):',
        "# 3. FSM 제어 루프",
    )

    assert "observe_latched_busbar_seating(" in task_setup
    assert "max(busbar_endpoint_residuals)" in task_setup
    assert "BUSBAR_HOLE_ALIGNMENT_TOL_M" in task_setup
    assert "busbar_min[2]" in task_setup
    assert "support_top_z" in task_setup
    assert "BUSBAR_SEAT_Z_TOL_M" in task_setup
    assert "busbar가 실제 두 bolt축/지지면에 " in task_setup
    assert '"안착하지 않았습니다 "' in task_setup


def test_nut_assembly_uses_live_collider_axis_without_coordinate_fallback():
    constants = _source_between(
        "STATION_BOLT_PRIM_PATHS = {",
        "# 너트 Prim 경로",
    )
    task_setup = _source_between(
        'elif task in ("ASSEMBLE_NUT1", "ASSEMBLE_NUT2"):',
        "# 3. FSM 제어 루프",
    )
    preflight = _source_between(
        "def validate_stage_contract(stage):",
        "def run_wheel_smoke_test(",
    )

    assert 'f"{bolt_path}/geo/PolyShape"' in constants
    assert "STATION_BOLT_WORLD_POS" not in EXECUTOR_SOURCE
    assert "STATION_BOLT_BODY_PATHS.get(current_station)" in task_setup
    assert "bolt_body_paths[nut_slot - 1]" in task_setup
    assert "live_bolt_axis_geometry(" in task_setup
    assert "bolt_body_path" in task_setup
    assert "STATION_BOLT_BODY_PATHS[station]" in preflight
    assert "CollisionAPI" in preflight


def test_pick_latches_physical_nut_offset_for_matching_assembly_only():
    lift_phase = _source_between(
        'elif phase == "NUT_LIFT":',
        'elif phase == "MOVE_TO_BOLT_NUT":',
    )
    task_setup = _source_between(
        'elif task in ("ASSEMBLE_NUT1", "ASSEMBLE_NUT2"):',
        "# 3. FSM 제어 루프",
    )

    success_index = lift_phase.index(
        'publish_status("SUCCESS")'
    )
    latch_index = lift_phase.index(
        "nut_transport_offset = nut_from_ee_offset("
    )
    index_latch = lift_phase.index(
        "nut_transport_index = nut_index"
    )
    assert latch_index < index_latch < success_index
    assert "nut_transport_index != nut_index" in task_setup
    assert "nut_transport_offset is None" in task_setup
    assert re.search(
        r"bolt_target_pos\[0\]\s*-\s*nut_transport_offset\[0\]",
        task_setup,
    )
    assert re.search(
        r"bolt_target_pos\[1\]\s*-\s*nut_transport_offset\[1\]",
        task_setup,
    )


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

    assert "compensated_ee_target_for_nut_axis(" in vertical
    assert "nut_now_pos" in vertical
    assert "bolt_approach_command, nut_xy_lead = lead_xy_target(" in vertical
    assert "bolt_approach_command" in vertical
    assert "consecutive_planar_alignment(" in vertical
    assert "NUT_BOLT_XY_ALIGNMENT_TOL_M" in vertical
    assert "NUT_BOLT_ALIGNMENT_SETTLE_STEPS" in vertical
    assert 'publish_progress("MOVE_TO_BOLT", 30.0)' in vertical
    assert 'phase = "NUT_DESCEND_TO_BOLT"' in vertical

    assert "current_err < PICK_TOLERANCE_STRICT" in horizontal
    assert "step_count > MAX_STUCK_STEPS" in horizontal
    assert "step_count > MAX_STUCK_STEPS" not in vertical

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
    assert "nut_now_pos" in vertical
    assert "bolt_target_pos" in vertical
    assert "consecutive_planar_alignment(" in vertical


def test_vertical_alignment_rejects_lost_nut_before_sending_action():
    vertical = _source_between(
        'elif phase == "MOVE_TO_BOLT_NUT_VERTICAL":',
        'elif phase == "NUT_DESCEND_TO_BOLT":',
    )

    coupling_gate = vertical.index(
        "> NUT_EE_COUPLING_TOLERANCE_M"
    )
    z_coupling_gate = vertical.index(
        "> NUT_EE_Z_COUPLING_TOLERANCE_M"
    )
    action = vertical.index("robot.apply_action(actions)")
    assert coupling_gate < action
    assert z_coupling_gate < action
    assert "restore_transported_nut_to_tray_glue(" in vertical[
        coupling_gate:action
    ]


def test_nut_descent_is_blocked_outside_actual_one_mm_axis_gate():
    descend = _source_between(
        'elif phase == "NUT_DESCEND_TO_BOLT":',
        'elif phase == "NUT_SCREW":',
    )

    valid_index = descend.index(
        "insertion_alignment_valid = ("
    )
    lower_index = descend.index(
        "descend_target_z = max("
    )
    command_index = descend.index(
        "compensated_ee_target_for_nut_axis("
    )
    lead_index = descend.index(
        "step_target_pos, nut_xy_lead = lead_xy_target("
    )
    action_index = descend.index(
        "target_end_effector_position=step_target_pos"
    )
    screw_index = descend.index('phase = "NUT_SCREW"')
    assert "NUT_BOLT_XY_INSERT_MAX_M" in descend
    assert "NUT_BOLT_XY_ALIGNMENT_TOL_M" in descend
    assert "nut_axis_error" in descend
    assert "axis_tilt" in descend
    assert (
        valid_index
        < lower_index
        < command_index
        < lead_index
        < action_index
        < screw_index
    )
    assert "logical_step_target_pos" in descend
    assert "nut_xy_lead," in descend[lead_index:action_index]
    assert "or descend_target_z <= bolt_touch_pos[2]" not in descend
    assert re.search(
        r"^NUT_TOUCH_Z_TOL_M\s*=\s*0\.001\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )
    assert re.search(
        r"^NUT_TOUCH_SETTLE_STEPS\s*=\s*12\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )
    assert (
        "descend_target_z - bolt_touch_pos[2]"
        in descend
    )
    assert "touch_z_error <= NUT_TOUCH_Z_TOL_M" in descend
    assert "nut_touch_settle_count += 1" in descend
    assert (
        "nut_touch_settle_count\n"
        "                        >= NUT_TOUCH_SETTLE_STEPS"
        in descend
    )
    assert "PICK_TOLERANCE_LOOSE_VAL" not in descend


def test_screw_uses_actual_pitch_joint6_and_never_treats_stall_as_success():
    screw = _source_between(
        'elif phase == "NUT_SCREW":',
        'elif phase == "NUT_SEATING_VERIFY":',
    )

    assert re.search(
        r"^NUT_PITCH_M\s*=\s*0\.001714406\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )
    assert "joint_6_effort_index = robot.get_dof_index(\"joint_6\")" in (
        EXECUTOR_SOURCE
    )
    assert "joint_efforts[joint_6_effort_index]" in screw
    assert "joint_efforts[-1]" not in screw
    assert "stuck_counter >= STUCK_STEP_LIMIT" not in screw
    assert "is_seated_by_torque" not in screw
    assert "physical_seating_candidate = (" in screw
    assert "actual_nut_progress" in screw
    assert "screw_pass_target_z(" in screw
    assert "screw_pass_start_ee_pos[2]" in screw
    assert "regrasp_extra" not in screw
    assert "torque_contact_settled and not final_pass" in screw
    assert "FAILURE:NUT_SCREW_JAM" in screw
    assert "and final_pass" in screw
    assert "nut_required_seating_depth_m" in screw
    assert "required_final_pass_rotation_rad" in screw
    assert "screw_torque_settle_count == 0" in screw
    assert 'phase = "NUT_SEATING_VERIFY"' in screw
    assert 'phase = "NUT_RETRACT_LIFT"' not in screw


def test_screw_axis_gate_freezes_immediately_and_resumes_after_twelve_ticks():
    gate = _load_screw_axis_gate()
    theta_deg = 100.0
    aligned_steps = 12

    theta_deg, aligned_steps, rotation_enabled = gate(
        theta_deg,
        aligned_steps,
        0.000501,
        math.radians(1.0),
        xy_tolerance_m=0.0005,
        tilt_tolerance_rad=math.radians(3.0),
        required_aligned_steps=12,
        theta_step_deg=2.0,
        maximum_theta_deg=350.0,
    )
    assert theta_deg == 100.0
    assert aligned_steps == 0
    assert not rotation_enabled

    for expected_count in range(1, 12):
        theta_deg, aligned_steps, rotation_enabled = gate(
            theta_deg,
            aligned_steps,
            0.0005,
            math.radians(3.0),
            xy_tolerance_m=0.0005,
            tilt_tolerance_rad=math.radians(3.0),
            required_aligned_steps=12,
            theta_step_deg=2.0,
            maximum_theta_deg=350.0,
        )
        assert theta_deg == 100.0
        assert aligned_steps == expected_count
        assert not rotation_enabled

    theta_deg, aligned_steps, rotation_enabled = gate(
        theta_deg,
        aligned_steps,
        0.0005,
        math.radians(3.0),
        xy_tolerance_m=0.0005,
        tilt_tolerance_rad=math.radians(3.0),
        required_aligned_steps=12,
        theta_step_deg=2.0,
        maximum_theta_deg=350.0,
    )
    assert theta_deg == 102.0
    assert aligned_steps == 12
    assert rotation_enabled

    theta_deg, aligned_steps, rotation_enabled = gate(
        theta_deg,
        aligned_steps,
        0.0001,
        math.radians(3.01),
        xy_tolerance_m=0.0005,
        tilt_tolerance_rad=math.radians(3.0),
        required_aligned_steps=12,
        theta_step_deg=2.0,
        maximum_theta_deg=350.0,
    )
    assert theta_deg == 102.0
    assert aligned_steps == 0
    assert not rotation_enabled


def test_screw_rotate_uses_live_axis_gate_and_narrow_rate_limited_xy_lead():
    rotate = _source_between(
        'if screw_sub == "rotate":',
        'elif screw_sub == "release":',
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
    descend = _source_between(
        'elif phase == "NUT_DESCEND_TO_BOLT":',
        'elif phase == "NUT_SCREW":',
    )
    regrasp = _source_between(
        'elif screw_sub == "regrasp":',
        'elif screw_sub == "prove_coupling":',
    )
    probe = _source_between(
        'elif screw_sub == "prove_coupling":',
        'elif phase == "NUT_SEATING_VERIFY":',
    )

    assert re.search(
        r"^NUT_SCREW_XY_LEAD_MAX_M\s*=\s*0\.002\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )
    assert re.search(
        r"^NUT_SCREW_XY_LEAD_STEP_M\s*=\s*0\.0002\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )
    assert re.search(
        r"^NUT_BOLT_ALIGNMENT_SETTLE_STEPS\s*=\s*12\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )

    live_bolt_index = rotate.index("live_bolt_axis_geometry(")
    axis_error_index = rotate.index("nut_axis_error = float(")
    gate_index = rotate.index("axis_gated_screw_theta(")
    pitch_index = rotate.index("screw_pass_target_z(")
    lead_index = rotate.index(
        "target_pos, screw_xy_lead = lead_xy_target("
    )
    action_index = rotate.index(
        "target_end_effector_position=target_pos"
    )
    assert (
        live_bolt_index
        < axis_error_index
        < gate_index
        < pitch_index
        < lead_index
        < action_index
    )
    assert "nut_now_pos[:2]" in rotate[
        axis_error_index:gate_index
    ]
    assert "live_bolt_target_pos[:2]" in rotate[
        axis_error_index:gate_index
    ]
    assert "NUT_BOLT_XY_ALIGNMENT_TOL_M" in rotate[
        gate_index:pitch_index
    ]
    assert "NUT_BOLT_AXIS_TOL_RAD" in rotate[
        gate_index:pitch_index
    ]
    assert "NUT_BOLT_ALIGNMENT_SETTLE_STEPS" in rotate[
        gate_index:pitch_index
    ]
    assert "max_lead=NUT_SCREW_XY_LEAD_MAX_M" in rotate[
        lead_index:action_index
    ]
    assert "max_lead_step=NUT_SCREW_XY_LEAD_STEP_M" in rotate[
        lead_index:action_index
    ]
    assert "screw_pass_theta +" not in rotate
    for reset_source in (restart, cancel, conflict, descend, regrasp, probe):
        assert "screw_axis_alignment_count = 0" in reset_source
        assert "screw_xy_lead = np.zeros(2, dtype=float)" in reset_source


def test_screw_requires_full_physical_lead_rotation_and_final_torque():
    constants = _source_between(
        "# 너트 체결(Screwing) 파라미터",
        "TORQUE_THRESHOLD",
    )
    screw = _source_between(
        'elif phase == "NUT_SCREW":',
        'elif phase == "NUT_SEATING_VERIFY":',
    )

    assert "planned_screw_depth(" in constants
    assert "1 + REGRASP_CYCLES" in constants
    assert re.search(
        r"^REGRASP_CYCLES\s*=\s*5\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )
    assert "signed_yaw_delta_wxyz(" in screw
    assert "screw_pass_accumulated_nut_yaw_rad" in screw
    assert "pass_coupling_orientation_error" in screw
    assert "ee_target_orientation_error" in screw
    assert "pass_actual_nut_progress" in screw
    assert "NUT_SCREW_PASS_MIN_AXIAL_PROGRESS_M" in screw
    assert "NUT_SCREW_PASS_VERIFY_STEPS" in screw
    candidate = screw[
        screw.index("physical_seating_candidate = ("):
        screw.index('phase = "NUT_SEATING_VERIFY"')
    ]
    assert "pass_follow_verified" in candidate
    assert "torque_contact_settled" in candidate
    assert "nut_required_seating_depth_m" in screw
    assert "required_final_pass_lead_m" in screw


def test_screw_separates_global_seating_and_per_pass_axial_tolerances():
    constants = _source_between(
        "NUT_SCREW_ORIENTATION_TOL_RAD",
        "# 초기 자세",
    )
    screw = _source_between(
        'elif phase == "NUT_SCREW":',
        'elif phase == "NUT_SEATING_VERIFY":',
    )

    assert re.search(
        r"^NUT_SEATING_AXIAL_TOL_M\s*=\s*0\.0002\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )
    assert re.search(
        r"^NUT_SCREW_PASS_AXIAL_PROGRESS_TOL_M\s*=\s*0\.0001\b",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )
    assert "NUT_SCREW_AXIAL_PROGRESS_TOL_M" not in EXECUTOR_SOURCE
    assert "required_cumulative_pass_progress_m" in screw
    assert "(screw_pass_idx + 1)" in screw
    assert "actual_nut_progress" in screw
    assert ">= required_cumulative_pass_progress_m" in screw
    assert "NUT_SCREW_PASS_AXIAL_PROGRESS_TOL_M" in constants


def test_raw_seating_depth_is_preserved_and_remaining_pass_capacity_is_gated():
    task_setup = _source_between(
        'elif task in ("ASSEMBLE_NUT1", "ASSEMBLE_NUT2"):',
        "# 3. FSM 제어 루프",
    )
    descend = _source_between(
        'elif phase == "NUT_DESCEND_TO_BOLT":',
        'elif phase == "NUT_SCREW":',
    )
    screw = _source_between(
        'elif phase == "NUT_SCREW":',
        'elif phase == "NUT_SEATING_VERIFY":',
    )
    reset_helper = _source_between(
        "def restore_transported_nut_to_tray_glue(reason):",
        "def publish_status(status_str: str):",
    )

    assert "nut_raw_required_seating_depth_m" in task_setup
    assert "nut_raw_required_seating_depth_m" in descend
    assert "nut_raw_required_seating_depth_m" in screw
    assert (
        "required_final_pass_lead_m = max(" in screw
        and "nut_raw_required_seating_depth_m" in screw
    )
    assert "remaining_raw_seating_depth_m" in screw
    assert "remaining_command_capacity_m" in screw
    assert "NUT_SCREW_REMAINING_TRAVEL_INSUFFICIENT" in screw
    assert "nut_raw_required_seating_depth_m = None" in reset_helper


def test_final_torque_and_following_wait_independently_under_one_bound():
    screw = _source_between(
        'elif phase == "NUT_SCREW":',
        'elif phase == "NUT_SEATING_VERIFY":',
    )

    assert "NUT_FINAL_TORQUE_BEFORE_SEAT" not in screw
    assert "Torque and following are independent physical" in screw
    assert "torque_contact_settled\n                                or pass_done" in screw
    assert "screw_final_verify_count" in screw
    assert "NUT_SEATING_VERIFY_MAX_STEPS" in screw
    success_gate = screw.index("physical_seating_candidate")
    success_transition = screw.index('phase = "NUT_SEATING_VERIFY"')
    bounded_failure = screw.index("NUT_SEATING_NOT_VERIFIED")
    assert success_gate < success_transition < bounded_failure


def test_regrasp_has_no_time_shortcut_and_requires_nut_retention_and_ee_settle():
    screw = _source_between(
        'elif phase == "NUT_SCREW":',
        'elif phase == "NUT_SEATING_VERIFY":',
    )

    assert "screw_release_step > 40" not in screw
    assert "nut_regrasp_retention_errors(" in screw
    assert "NUT_REGRASP_NUT_Z_RETENTION_TOL_M" in screw
    assert "NUT_REGRASP_NUT_YAW_RETENTION_TOL_RAD" in screw
    assert "FAILURE:NUT_REGRASP_RETENTION_LOST" in screw
    assert screw.count("update_screw_regrasp_pose_settle(") >= 5
    for substate in (
        '"release"',
        '"lift_up"',
        '"unwind"',
        '"descend_down"',
        '"regrasp"',
        '"prove_coupling"',
    ):
        assert f"screw_sub == {substate}" in screw
    assert "rf >= 1.0 and release_pose_settled" in screw
    assert "if lift_pose_settled:" in screw
    assert "if unwind_done and unwind_pose_settled:" in screw
    assert "if descend_pose_settled:" in screw
    assert "rf >= 1.0 and regrasp_pose_settled" in screw
    assert "retained_nut_regrasp_target(" in screw
    assert "screw_sub = \"prove_coupling\"" in screw
    assert "NUT_REGRASP_COUPLING_PROBE_DEG" in screw
    assert "screw_pass_accumulated_nut_yaw_rad" in screw
    assert "NUT_REGRASP_COUPLING_PROBE_STEPS" in screw
    assert "NUT_REGRASP_COUPLING_PROBE_MAX_STEPS" in screw
    assert "NUT_REGRASP_COUPLING_NOT_VERIFIED" in screw
    probe = screw[
        screw.index('elif screw_sub == "prove_coupling":'):
    ]
    assert "screw_regrasp_anchor_nut_quat" in probe
    assert (
        "probe_local_orientation_error\n"
        "                        <= "
        "NUT_REGRASP_COUPLING_PROBE_TOL_RAD"
    ) in probe
    assert (
        "screw_pass_end_pos + np.array([0.0, 0.0, "
        "REGRASP_Z_OFFSET])"
    ) not in screw


def test_tcp_force_and_success_use_live_busbar_support_holes_and_surface():
    screw = _source_between(
        'elif phase == "NUT_SCREW":',
        'elif phase == "NUT_SEATING_VERIFY":',
    )
    release = _source_between(
        'elif phase == "NUT_SEATING_VERIFY":',
        'elif phase == "NUT_RETRACT_LIFT":',
    )
    retract = _source_between(
        'elif phase == "NUT_RETRACT_VERIFY":',
        'elif phase == "NUT_RETRACT_UNWIND":',
    )

    assert "TCP_FORCE_CHECK_Z" not in EXECUTOR_SOURCE
    assert "nut_torque_contact_geometry(" in screw
    assert 'live_seating["busbar_top_z"]' in screw
    assert '"surface_contact"' in screw
    assert '"entry_progress_m"' in screw
    assert "live_nut_busbar_seating_errors(" in screw
    assert 'live_seating["endpoint_max_m"]' in screw
    assert '"busbar_bottom_support_error_m"' in screw
    assert '"nut_bottom_to_surface_m"' in screw
    assert "NUT_SEATING_SURFACE_TOL_M" in screw
    for verify_phase in (release, retract):
        assert "live_nut_busbar_seating_errors(" in verify_phase
        assert "BUSBAR_HOLE_ALIGNMENT_TOL_M" in verify_phase
        assert "BUSBAR_SEAT_Z_TOL_M" in verify_phase
        assert "NUT_SEATING_SURFACE_TOL_M" in verify_phase


def test_dynamic_torque_contact_uses_each_pass_geometry_not_absolute_tcp_z():
    torque_geometry, _ = _load_nut_contact_helpers()
    nut_half_height = 0.5 * 0.00847725
    nut_from_ee_z = -0.231

    # Representative live top heights for stations 3, 4 and 5.  All valid
    # surface TCP heights are above the removed 0.378m absolute gate.
    for busbar_top_z in (0.1503, 0.1506, 0.150856):
        nut_center_z = busbar_top_z + nut_half_height
        current_ee_z = nut_center_z - nut_from_ee_z
        geometry = torque_geometry(
            current_ee_z,
            nut_center_z,
            0.400,
            0.400 + nut_from_ee_z,
            busbar_top_z + 0.009,
            busbar_top_z,
        )
        assert current_ee_z > 0.378
        assert geometry["surface_contact"]
        assert math.isclose(
            geometry["surface_tcp_z"],
            current_ee_z,
            abs_tol=1.0e-12,
        )

        # Moving the TCP without the nut breaks the pass latch, even though
        # the physical nut still touches the busbar surface.
        uncoupled = torque_geometry(
            current_ee_z + 0.002,
            nut_center_z,
            0.400,
            0.400 + nut_from_ee_z,
            busbar_top_z + 0.009,
            busbar_top_z,
        )
        assert not uncoupled["surface_contact"]

        # Moving nut and TCP together off the surface preserves coupling but
        # must still fail the independent live surface test.
        off_surface = torque_geometry(
            current_ee_z + 0.002,
            nut_center_z + 0.002,
            0.400,
            0.400 + nut_from_ee_z,
            busbar_top_z + 0.009,
            busbar_top_z,
        )
        assert not off_surface["surface_contact"]

    with np.testing.assert_raises(ValueError):
        torque_geometry(
            float("nan"),
            0.155,
            0.400,
            0.169,
            0.160,
            0.151,
        )


def test_regrasp_target_applies_five_mm_once_across_all_six_passes():
    _, regrasp_target = _load_nut_contact_helpers()
    touch_ee = np.array([1.0, -0.5, 0.400])
    touch_nut = np.array([1.001, -0.498, 0.170])
    initial_offset = touch_nut - touch_ee
    fixed_grip_offset = initial_offset.copy()
    fixed_grip_offset[2] -= 0.005

    anchor_positions = [
        touch_nut - np.array([0.0, 0.0, pass_index * 0.0016])
        for pass_index in range(1, 6)
    ]
    targets = [
        regrasp_target(anchor, fixed_grip_offset)
        for anchor in anchor_positions
    ]
    for anchor, target in zip(anchor_positions, targets):
        np.testing.assert_allclose(
            anchor - target,
            fixed_grip_offset,
            atol=1.0e-12,
        )

    assert math.isclose(
        fixed_grip_offset[2],
        initial_offset[2] - 0.005,
        abs_tol=1.0e-12,
    )
    assert not math.isclose(
        fixed_grip_offset[2],
        initial_offset[2] - 5 * 0.005,
        abs_tol=1.0e-12,
    )
    with np.testing.assert_raises(ValueError):
        regrasp_target(
            np.array([0.0, 0.0, float("inf")]),
            fixed_grip_offset,
        )


def test_arm_is_the_only_perception_empty_reset_owner():
    arm_source = (
        Path(__file__).parents[1]
        / "arm_node"
        / "arm_node.py"
    ).read_text(encoding="utf-8")
    scan_battery = _source_between(
        'if task == "SCAN_BATTERY":',
        'elif task == "RETURN_HOME":',
    )
    scan_busbar = _source_between(
        'elif task == "SCAN_BUSBAR":',
        'elif task == "CONTINUE_BUSBAR_WRIST_SCAN":',
    )

    for reset_topic in (
        "/wrist/perception/reset_cache",
        "/busbar_cam/perception/reset_cache",
        "/bolt_cam/perception/reset_cache",
    ):
        assert reset_topic not in EXECUTOR_SOURCE
        assert reset_topic in arm_source
    assert "perception_reset.publish(Empty())" not in scan_battery
    assert "perception_reset.publish(Empty())" not in scan_busbar
    assert "RESET_BOLT_DETECTION" in scan_battery
    assert "self.pub_bolt_reset" in arm_source
    assert "self.pub_busbar_reset" in arm_source
    assert "self.pub_wrist_reset" in arm_source


def test_success_requires_release_stability_and_retract_separation():
    release_verify = _source_between(
        'elif phase == "NUT_SEATING_VERIFY":',
        'elif phase == "NUT_RETRACT_LIFT":',
    )
    retract = _source_between(
        'elif phase == "NUT_RETRACT_LIFT":',
        'elif phase == "NUT_RETRACT_UNWIND":',
    )

    assert "GRIPPER_OPEN" in release_verify
    assert "NUT_SEATING_STABLE_MOTION_TOL_M" in release_verify
    assert "NUT_BOLT_XY_ALIGNMENT_TOL_M" in release_verify
    assert "nut_required_seating_depth_m" in release_verify
    assert "nut_seating_verify_count +=" in release_verify
    assert "NUT_SEATING_VERIFY_STEPS" in release_verify
    assert "NUT_SEATING_VERIFY_MAX_STEPS" in release_verify

    assert 'phase = "NUT_RETRACT_VERIFY"' in retract
    assert 'elif phase == "NUT_RETRACT_VERIFY":' in retract
    assert "nut_seating_release_ee_pos" in retract
    assert "nut_seating_release_separation" in retract
    assert "nut_seating_release_nut_pos" in retract
    assert "ee_retract_lift" in retract
    assert "separation_increase" in retract
    assert "release_nut_motion" in retract
    assert "NUT_RETRACT_NUT_MOTION_TOL_M" in retract
    assert "nut_seating_verify_count += 1" in retract
    assert "NUT_SEATING_VERIFY_STEPS" in retract
    assert "NUT_NOT_RETAINED_ON_BOLT" in retract
    assert "nut_transport_offset = None" in retract
    assert 'phase = "NUT_RETRACT_UNWIND"' in retract


def test_carried_nut_blocks_every_nonmatching_arm_task():
    dispatch = _source_between(
        "# PICK 성공 뒤 실제 nut가 gripper에 남아 있는 동안은",
        'if task == "SCAN_BATTERY":',
    )

    assert "expected_carried_nut_task" in dispatch
    assert "nut_transport_index is not None" in dispatch
    assert "task != expected_carried_nut_task" in dispatch
    assert "hold_current_arm_pose()" in dispatch
    assert "FAILURE:NUT_TRANSPORT_TASK_MISMATCH" in dispatch
    assert 'task = ""' in dispatch


def test_cancel_and_post_release_failure_restore_transported_nut():
    helper = _source_between(
        "def restore_transported_nut_to_tray_glue(reason):",
        "def publish_status(status_str: str):",
    )
    cancel = _source_between(
        'if task == "CANCEL_ARM_TASK":',
        "# 안전장치: 팔 Task는 AMR이 도착해",
    )
    release_verify = _source_between(
        'elif phase == "NUT_SEATING_VERIFY":',
        'elif phase == "NUT_RETRACT_LIFT":',
    )
    retract_verify = _source_between(
        'elif phase == "NUT_RETRACT_VERIFY":',
        'elif phase == "NUT_RETRACT_UNWIND":',
    )
    assembly_setup = _source_between(
        'elif task in ("ASSEMBLE_NUT1", "ASSEMBLE_NUT2"):',
        "# 3. FSM 제어 루프",
    )
    timeout = _source_between(
        "# Phase 타임아웃 예외 처리",
        "step_count += 1",
    )

    assert "disable_physics_recursively(stage, nut_root)" in helper
    assert "nut_xf.set_world_pose(" in helper
    assert "nut_released[array_index] = False" in helper
    assert "nut_transport_offset = None" in helper
    assert "nut_transport_index = None" in helper
    assert "restore_transported_nut_to_tray_glue(" in cancel
    assert "restore_transported_nut_to_tray_glue(" in release_verify
    assert "restore_transported_nut_to_tray_glue(" in retract_verify
    assert assembly_setup.count(
        "restore_transported_nut_to_tray_glue("
    ) >= 5
    assert "restore_active_nut_to_tray_glue(" in timeout
    assert "restore_transported_nut_to_tray_glue(" in timeout


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
        "RESET_BOLT_DETECTION",
    ):
        clear_index = restart.index(transient_clear)
        assert restart_gate < clear_index < interrupted_gate
    assert "perception_reset.publish(Empty())" not in restart
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


def test_preflight_checks_every_bolt_camera_pose_and_station5_bolt1_xy():
    preflight = _source_between(
        "def validate_stage_contract(stage):",
        "def run_wheel_smoke_test(",
    )

    assert (
        "_world_pose_for_path(\n                stage,\n"
        "                BOLT_CAMERA_PATH,"
        in preflight
    )
    assert (
        "for station, (root_path, mesh_path) "
        "in STATION_BUSBAR_PATHS.items()"
        in preflight
    )
    assert "bolt2_path = bolt_paths[1]" in preflight
    assert "bolt_camera_pose_for_target(" in preflight
    assert "bolt_camera_position[:2] - bolt2_position[:2]" in preflight
    assert "bolt_camera_position[2] - initial_position[2]" in preflight
    assert "_quat_angular_error(" in preflight
    assert "station_bolt_camera_poses[station]" in preflight
    assert "STATION_BOLT_BODY_PATHS[station][0]" in preflight
    assert "STATION5_BOLT1_EXPECTED_XY" in preflight
    assert "STATION5_BOLT1_PREFLIGHT_TOL_M" in preflight
    assert "station 5 bolt 1 collider XY 오차" in preflight
    assert "bolt2_xy=" in preflight

    assert re.search(
        r"^STATION5_BOLT1_EXPECTED_XY\s*=\s*"
        r"np\.array\(\[1\.0552,\s*-0\.7312\],\s*dtype=float\)$",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )
    assert re.search(
        r"^STATION5_BOLT1_PREFLIGHT_TOL_M\s*=\s*0\.001$",
        EXECUTOR_SOURCE,
        flags=re.MULTILINE,
    )


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

    assert '"CONTINUE_BUSBAR_WRIST_SCAN"' in dispatch
    assert '"COMPLETE_BUSBAR_FIXED_SCAN"' in dispatch
    assert "busbar_camera_handshake" in dispatch
    assert 'phase == "WAIT_BUSBAR_CAMERA_LATCH"' in dispatch
    assert 'phase not in {"IDLE", "DONE"}' in dispatch
    hold_index = dispatch.index("hold_current_arm_pose()")
    idle_index = dispatch.index('phase = "IDLE"')
    failure_index = dispatch.index(
        'publish_status("FAILURE:ARM_TASK_CONFLICT")'
    )
    assert hold_index < idle_index < failure_index

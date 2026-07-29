import ast
import math
import re
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


def _load_pure_executor_functions(*names):
    tree = ast.parse(EXECUTOR_SOURCE)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"math": math, "np": np}
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            filename="<execute_isaac_pure_helpers>",
            mode="exec",
        ),
        namespace,
    )
    return tuple(namespace[name] for name in names)


def _load_support_geometry_helpers(points_by_path, transforms_by_path):
    tree = ast.parse(EXECUTOR_SOURCE)
    helper_names = {
        "transformed_mesh_vertex_world_bounds",
        "station_busbar_support_top_z",
    }
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    assert {node.name for node in selected} == helper_names

    class _PointsAttr:
        def __init__(self, points):
            self._points = points

        def Get(self):
            return self._points

    class _Mesh:
        def __init__(self, prim):
            self._prim = prim

        def GetPointsAttr(self):
            return _PointsAttr(self._prim.points)

    class _Prim:
        def __init__(self, path):
            self.path = path
            self.points = points_by_path.get(path)

        def IsValid(self):
            return self.path in points_by_path

        def IsA(self, schema):
            return schema is _Mesh and self.IsValid()

    class _Stage:
        def GetPrimAtPath(self, path):
            return _Prim(path)

    class _UsdGeom:
        Mesh = _Mesh

    def _point_bounds(transform, local_points):
        try:
            points = np.asarray(local_points, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid authored points") from exc
        if (
            points.ndim != 2
            or points.shape[0] == 0
            or points.shape[1] != 3
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("invalid authored points")
        translation = np.asarray(transform, dtype=float)[3, :3]
        world_points = points + translation
        return (
            tuple(np.min(world_points, axis=0)),
            tuple(np.max(world_points, axis=0)),
        )

    namespace = {
        "np": np,
        "UsdGeom": _UsdGeom,
        "STATION_BUSBAR_SUPPORT_PATHS": {
            3: ("/support/a", "/support/b"),
        },
        "BUSBAR_SEAT_Z_TOL_M": 0.001,
        "world_xf": lambda _stage, path: transforms_by_path[path],
        "transformed_usd_points_bounds": _point_bounds,
    }
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            filename="<execute_isaac_support_geometry_helpers>",
            mode="exec",
        ),
        namespace,
    )
    return (
        namespace["transformed_mesh_vertex_world_bounds"],
        namespace["station_busbar_support_top_z"],
        _Stage(),
    )


def _load_live_bolt_midpoint(axis_origins, station_paths):
    tree = ast.parse(EXECUTOR_SOURCE)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "live_station_bolt_midpoint"
    ]
    assert len(selected) == 1

    class _Prim:
        def __init__(self, valid):
            self._valid = valid

        def IsValid(self):
            return self._valid

    class _Stage:
        def GetPrimAtPath(self, path):
            return _Prim(path in axis_origins)

    def _world_pose(_stage, path):
        return (
            np.asarray(axis_origins[path], dtype=float),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )

    def _axis_geometry(_stage, path, reference_z):
        origin = np.asarray(axis_origins[path], dtype=float)
        return (
            np.array([origin[0], origin[1], reference_z]),
            None,
            None,
            None,
        )

    namespace = {
        "math": math,
        "np": np,
        "STATION_BOLT_BODY_PATHS": station_paths,
        "_world_pose_for_path": _world_pose,
        "live_bolt_axis_geometry": _axis_geometry,
    }
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            filename="<execute_isaac_live_bolt_midpoint>",
            mode="exec",
        ),
        namespace,
    )
    return namespace["live_station_bolt_midpoint"], _Stage()


def test_live_station_bolt_midpoint_uses_exactly_two_physical_axes():
    paths = {
        5: ("/station5/bolt1", "/station5/bolt2"),
    }
    axis_origins = {
        "/station5/bolt1": (1.0, -0.7, 0.12),
        "/station5/bolt2": (1.2, -1.1, 0.14),
    }
    midpoint, stage = _load_live_bolt_midpoint(axis_origins, paths)

    np.testing.assert_allclose(
        midpoint(stage, 5),
        np.array([1.1, -0.9, 0.13]),
    )

    invalid_count, invalid_count_stage = _load_live_bolt_midpoint(
        axis_origins,
        {
            5: (
                "/station5/bolt1",
                "/station5/bolt2",
                "/station5/bolt3",
            ),
        },
    )
    with np.testing.assert_raises(RuntimeError):
        invalid_count(invalid_count_stage, 5)

    invalid_finite, invalid_finite_stage = _load_live_bolt_midpoint(
        {
            **axis_origins,
            "/station5/bolt2": (float("nan"), -1.1, 0.14),
        },
        paths,
    )
    with np.testing.assert_raises(RuntimeError):
        invalid_finite(invalid_finite_stage, 5)

    missing_prim, missing_prim_stage = _load_live_bolt_midpoint(
        {
            "/station5/bolt1": axis_origins["/station5/bolt1"],
        },
        paths,
    )
    with np.testing.assert_raises(RuntimeError):
        missing_prim(missing_prim_stage, 5)

    degenerate, degenerate_stage = _load_live_bolt_midpoint(
        {
            "/station5/bolt1": axis_origins["/station5/bolt1"],
            "/station5/bolt2": axis_origins["/station5/bolt1"],
        },
        paths,
    )
    with np.testing.assert_raises(RuntimeError):
        degenerate(degenerate_stage, 5)


def test_move_battery_center_uses_live_bolt_midpoint_not_target_pose():
    setup = _source_between(
        'elif task == "MOVE_BATTERY_CENTER":',
        'elif task == "FINE_ALIGNMENT":',
    )

    clear_index = setup.index(
        "isaac_node.latest_target_pose = None"
    )
    midpoint_index = setup.index(
        "bolt_midpoint = live_station_bolt_midpoint("
    )
    target_index = setup.index(
        "BATTERY_CENTER_POS = np.array("
    )
    assert clear_index < midpoint_index < target_index
    assert "current_station" in setup
    assert "BATTERY_CENTER_Z" in setup
    assert ".latest_target_pose.pose.position" not in setup
    assert "FAILURE:NO_TARGET_POSE" not in setup
    assert "FAILURE:LIVE_BOLT_TARGET_INVALID" in setup
    assert "hold_current_arm_pose()" in setup
    assert "busbar_pose_settle_count = 0" in setup
    assert "busbar_previous_ee_pos = None" in setup
    assert "busbar_previous_ee_orientation = None" in setup
    assert 'phase = "MOVE_BATTERY_CENTER_APPROACH"' in setup


def test_free_space_busbar_transport_uses_shorter_pose_settle_only():
    """The visible approach pause may be shorter; contact proof may not."""

    approach = _source_between(
        'elif phase == "MOVE_BATTERY_CENTER_APPROACH":',
        'elif phase == "FINE_ALIGNMENT":',
    )
    assert "BUSBAR_TRANSPORT_POSE_SETTLE_STEPS = 6" in EXECUTOR_SOURCE
    assert (
        "required_steps=BUSBAR_TRANSPORT_POSE_SETTLE_STEPS"
        in approach
    )
    assert "BUSBAR_HOLE_ALIGNMENT_SETTLE_STEPS = 12" in EXECUTOR_SOURCE
    assert "BUSBAR_SEAT_SETTLE_STEPS = 12" in EXECUTOR_SOURCE


def test_pose_motion_settle_accepts_unmeasured_first_sample_sentinel():
    (settle,) = _load_pure_executor_functions(
        "consecutive_pose_motion_settle",
    )
    kwargs = {
        "position_tolerance_m": 0.01,
        "orientation_tolerance_rad": 0.02,
        "position_motion_tolerance_m": 0.001,
        "orientation_motion_tolerance_rad": 0.002,
        "required_steps": 2,
    }

    first_count, first_settled = settle(
        0.0,
        0.0,
        float("inf"),
        float("inf"),
        0,
        **kwargs,
    )
    second_count, second_settled = settle(
        0.0,
        0.0,
        0.0,
        0.0,
        first_count,
        **kwargs,
    )
    final_count, final_settled = settle(
        0.0,
        0.0,
        0.0,
        0.0,
        second_count,
        **kwargs,
    )

    assert (first_count, first_settled) == (0, False)
    assert (second_count, second_settled) == (1, False)
    assert (final_count, final_settled) == (2, True)


def test_busbar_descent_pauses_for_realign_and_caps_vertical_lead():
    (command,) = _load_pure_executor_functions(
        "busbar_descent_z_command",
    )
    kwargs = {
        "alignment_tolerance_m": 0.00075,
        "maximum_step_m": 0.0002,
        "maximum_lead_m": 0.002,
        "realign_retract_m": 0.002,
    }

    hold_z, realign, lead_limited = command(
        0.000751,
        0.4200,
        0.4190,
        0.4000,
        **kwargs,
    )
    assert realign is True
    assert lead_limited is False
    assert math.isclose(hold_z, 0.4220)

    next_z, realign, lead_limited = command(
        0.000750,
        0.4200,
        0.4200,
        0.4000,
        **kwargs,
    )
    assert realign is False
    assert lead_limited is False
    assert math.isclose(next_z, 0.4198)

    frozen_z, realign, lead_limited = command(
        0.0002,
        0.4200,
        0.4100,
        0.4000,
        **kwargs,
    )
    assert realign is False
    assert lead_limited is True
    assert math.isclose(frozen_z, 0.4100)

    accumulated_z = 0.4200
    monotonic_trace = [accumulated_z]
    for _ in range(20):
        accumulated_z, realign, lead_limited = command(
            0.0002,
            0.4200,
            accumulated_z,
            0.4000,
            **kwargs,
        )
        monotonic_trace.append(accumulated_z)
        assert realign is False
        assert lead_limited is (_ >= 10)
    assert math.isclose(accumulated_z, 0.4180)
    assert math.isclose(0.4200 - accumulated_z, 0.0020)
    assert all(
        next_z <= previous_z
        for previous_z, next_z in zip(
            monotonic_trace,
            monotonic_trace[1:],
        )
    )
    assert all(target_z >= 0.4000 for target_z in monotonic_trace)

    monotonic_z, realign, lead_limited = command(
        0.0002,
        0.420001,
        0.4198,
        0.4000,
        **kwargs,
    )
    assert realign is False
    assert lead_limited is False
    assert monotonic_z < 0.4198

    disturbed_z, realign, lead_limited = command(
        0.0002,
        0.4201,
        0.4180,
        0.4000,
        **kwargs,
    )
    assert realign is False
    assert lead_limited is True
    assert math.isclose(disturbed_z, 0.4180)

    clamped_z, realign, lead_limited = command(
        0.0,
        0.4001,
        0.4001,
        0.4000,
        **kwargs,
    )
    assert realign is False
    assert lead_limited is False
    assert math.isclose(clamped_z, 0.4000)

    with np.testing.assert_raises(ValueError):
        command(
            float("nan"),
            0.42,
            0.42,
            0.40,
            **kwargs,
        )
    with np.testing.assert_raises(ValueError):
        command(
            -0.001,
            0.42,
            0.42,
            0.40,
            **kwargs,
        )
    with np.testing.assert_raises(ValueError):
        command(
            0.0,
            0.42,
            0.42,
            0.40,
            **{**kwargs, "maximum_lead_m": 0.0001},
        )
    with np.testing.assert_raises(ValueError):
        command(
            0.0,
            0.399,
            0.399,
            0.400,
            **kwargs,
        )


def test_busbar_free_air_cruise_allows_only_effective_clearance_margin():
    """Sub-millimetre free-air jitter must not spend a re-align attempt."""

    (command,) = _load_pure_executor_functions(
        "busbar_descent_z_command",
    )
    kwargs = {
        "alignment_tolerance_m": 0.00095,
        "maximum_step_m": 0.0002,
        "maximum_lead_m": 0.002,
        "realign_retract_m": 0.002,
    }

    next_z, realign, lead_limited = command(
        0.00090,
        0.4200,
        0.4200,
        0.4000,
        **kwargs,
    )
    assert realign is False
    assert lead_limited is False
    assert math.isclose(next_z, 0.4198)

    hold_z, realign, lead_limited = command(
        0.000951,
        0.4200,
        0.4200,
        0.4000,
        **kwargs,
    )
    assert realign is True
    assert lead_limited is False
    assert math.isclose(hold_z, 0.4220)


def test_busbar_descent_realign_gate_holds_noise_and_fails_closed():
    (gate,) = _load_pure_executor_functions(
        "busbar_descent_realign_gate",
    )
    kwargs = {
        "alignment_tolerance_m": 0.00075,
        "immediate_realign_m": 0.0015,
        "required_violation_steps": 3,
    }

    count, realign, hold, retract = gate(0.000751, 0, **kwargs)
    assert (count, realign, hold, retract) == (
        1,
        False,
        True,
        False,
    )

    # A recovered sample clears the debounce state and permits descent.
    count, realign, hold, retract = gate(0.000700, count, **kwargs)
    assert (count, realign, hold, retract) == (
        0,
        False,
        False,
        False,
    )

    # The gate classifies persistent moderate displacement separately from a
    # gross residual.  The executor may still turn that confirmed
    # in-contact condition into its bounded escape retract.
    for expected_count in (1, 2):
        count, realign, hold, retract = gate(
            0.000913,
            count,
            **kwargs,
        )
        assert (count, realign, hold, retract) == (
            expected_count,
            False,
            True,
            False,
        )
    count, realign, hold, retract = gate(0.000913, count, **kwargs)
    assert (count, realign, hold, retract) == (
        3,
        True,
        True,
        False,
    )

    # Even the top of the moderate band remains an in-place correction.
    count, realign, hold, retract = gate(0.001499, count, **kwargs)
    assert (count, realign, hold, retract) == (
        3,
        True,
        True,
        False,
    )

    # Gross displacement is never debounced.
    count, realign, hold, retract = gate(0.0015, 0, **kwargs)
    assert (count, realign, hold, retract) == (
        1,
        True,
        True,
        True,
    )

    with np.testing.assert_raises(ValueError):
        gate(float("nan"), 0, **kwargs)
    with np.testing.assert_raises(ValueError):
        gate(
            0.001,
            0,
            **{**kwargs, "immediate_realign_m": 0.00075},
        )


def test_busbar_bore_tip_axial_clearance_separates_free_air_from_contact():
    (clearance,) = _load_pure_executor_functions(
        "busbar_bore_tip_axial_clearance_m",
    )

    class _Axis:
        def __init__(self, origin, direction):
            self.origin_m = origin
            self.direction = direction

    axes = (
        _Axis((1.0, 2.0, 0.126), (0.0, 0.0, 1.0)),
        _Axis((2.0, 3.0, 0.126), (0.0, 0.0, 2.0)),
    )
    # The paired tip is z=0.154502m.  The second direction deliberately has
    # non-unit magnitude so the helper proves it normalizes live affine axes.
    assert math.isclose(
        clearance(
            ((1.0, 2.0, 0.173225), (2.0, 3.0, 0.174502)),
            axes,
            (0, 1),
            thread_tip_from_body_m=0.028502,
        ),
        0.018723,
        abs_tol=1.0e-9,
    )
    assert math.isclose(
        clearance(
            ((1.0, 2.0, 0.153502), (2.0, 3.0, 0.154502)),
            axes,
            (0, 1),
            thread_tip_from_body_m=0.028502,
        ),
        -0.001,
        abs_tol=1.0e-9,
    )

    with np.testing.assert_raises(ValueError):
        clearance(
            ((1.0, 2.0, 0.17),),
            axes,
            (0, 1),
            thread_tip_from_body_m=0.028502,
        )
    with np.testing.assert_raises(ValueError):
        clearance(
            ((1.0, 2.0, 0.17), (2.0, 3.0, 0.17)),
            axes,
            (0, 2),
            thread_tip_from_body_m=0.028502,
        )


def test_busbar_descent_distinguishes_free_bore_realign_from_contact_retract():
    """Loose travel ends before the strict, physically-free guard band."""

    descend = _source_between(
        'elif phase == "BUSBAR_DESCEND_TO_BOLT":',
        'elif phase == "BUSBAR_RELEASE_SETTLE":',
    )
    # The observed 0.96--1.03mm residual occurs while the bore bottom is still
    # about 30mm above the bolt tip.  A 2.00mm free-travel tolerance prevents
    # a needless 12-tick correction epoch there, but it is unavailable inside
    # the 5.5mm pre-contact guard.  The latter changes back to the unchanged
    # strict 0.75mm gate before any loose command can reach a bolt.
    assert "live_busbar_bore_tip_axial_clearance_m(" in descend
    assert "BUSBAR_DESCENT_FREE_BORE_CLEARANCE_M" in descend
    assert "BUSBAR_DESCENT_FREE_CRUISE_TOL_M" in descend
    assert "BUSBAR_DESCENT_PRECONTACT_GUARD_M" in descend
    assert "bore_mode = classify_busbar_descent_bore_mode(" in descend
    assert 'precontact_interval = bore_mode == "precontact"' in descend
    assert 'noncontact_interval = bore_mode != "contact"' in descend
    assert "free_bore_cruise" in descend
    assert "free_bore_realign = True" in descend
    assert "BUSBAR_DESCENT_MAX_FREE_REALIGN_ATTEMPTS" in descend
    assert "free_realign_exact_position" in descend
    assert "free_realign_quat" in descend
    assert "free_realign_translation" in descend
    assert "BUSBAR_DESCENT_FREE_REALIGN_MAX_TRANSLATION_M" in descend
    assert "free_realign_yaw" in descend
    assert "BUSBAR_DESCENT_FREE_REALIGN_MAX_YAW_RAD" in descend
    assert "FAILURE:BUSBAR_DESCENT_FREE_REALIGN_SAFETY_LIMIT" in descend
    assert "FAILURE:BUSBAR_DESCENT_FREE_REALIGN_EXHAUSTED" in descend
    assert "not free_bore_realign" in descend
    assert 'bore_mode == "contact"' in descend
    assert "rigid XY/yaw 재정렬" in descend

    # No free-air/pre-contact sub-branch may construct a physical retract
    # command or reserve the five-attempt contact budget.  The only contact
    # branch starts at/inside the physical 3mm clearance boundary.
    free_branch_start = descend.index("if free_bore_interval:")
    contact_branch_start = descend.index(
        "# At/inside the final 3mm",
        free_branch_start,
    )
    free_branch = descend[free_branch_start:contact_branch_start]
    assert "BUSBAR_DESCENT_REALIGN_RETRACT_M" not in free_branch
    assert "reserve_busbar_descent_retract_attempt(" not in free_branch
    assert "elif precontact_interval:" in free_branch
    assert "alignment_tolerance_m=(\n                            BUSBAR_HOLE_ALIGNMENT_TOL_M" in free_branch
    # A free-bore correction starts from the measured pose, not the prior
    # vertical-insertion target/lead.  The reset avoids replaying an old RMP
    # yaw trajectory into a newly observed rigid correction.  The live exact
    # rigid solve is also copied into target_mid_pos, which is the XY source
    # for the next vertical cadence command.
    assert "A non-contact correction is a new control epoch" in descend
    assert "hold_current_arm_pose()" in descend
    assert "busbar_alignment_controller_lead_xy = np.zeros(" in descend
    assert "busbar_alignment_last_retarget_endpoint_residuals = None" in descend
    assert "free_realign_command_position = (\n                            free_realign_exact_position.copy()" in descend
    assert "free_realign_command_quat = (\n                            free_realign_quat.copy()" in descend
    assert "free_realign_command_position = (\n                            busbar_alignment_target_pos.copy()" not in descend
    assert "free_realign_command_quat = (\n                            busbar_alignment_target_quat.copy()" not in descend
    assert "target_mid_pos[:2] = (\n                            free_realign_command_position[:2]" in descend

    # Once the bore is near/over a thread tip, the former bounded physical
    # escape remains intact, including its reservation before the action.
    assert "BUSBAR_DESCENT_REALIGN_RETRACT_M" in descend
    assert "+ BUSBAR_DESCENT_REALIGN_RETRACT_M" in descend
    assert "reserve_busbar_descent_retract_attempt(" in descend
    assert "FAILURE:BUSBAR_DESCENT_REALIGN_EXHAUSTED" in descend

    # Retraction is an escape only: a release still requires the unchanged
    # 0.75mm endpoint gate and consecutive physical seat confirmation.
    assert (
        "current_endpoint_max\n                    "
        "<= BUSBAR_HOLE_ALIGNMENT_TOL_M"
    ) in descend
    assert "BUSBAR_SEAT_SETTLE_STEPS" in descend


def test_high_clearance_bore_mode_cannot_request_contact_retract():
    """Loose travel is blocked by the strict pre-contact guard."""

    (classify_mode,) = _load_pure_executor_functions(
        "classify_busbar_descent_bore_mode",
    )
    kwargs = {
        "free_bore_clearance_m": 0.003,
        "precontact_guard_m": 0.0055,
        "free_cruise_tolerance_m": 0.00200,
    }
    # Keep the live executor's deliberately relaxed free-air threshold from
    # regressing to the former 0.75/1.25mm values.  This does not loosen the
    # 3mm contact boundary or the final insertion proof.
    assert "BUSBAR_DESCENT_FREE_CRUISE_TOL_M = 0.00200" in EXECUTOR_SOURCE

    # The live trace's roughly 1.6mm loaded-wrist sag is still free travel at
    # 20--30mm tip clearance; it cannot enter contact recovery.
    assert classify_mode(0.020, 0.00103, **kwargs) == "free_cruise"
    assert classify_mode(0.020, 0.00158, **kwargs) == "free_cruise"
    assert classify_mode(0.030, 0.00200, **kwargs) == "free_cruise"
    # A larger free-air residual holds for a rigid correction, not a retract.
    assert classify_mode(0.020, 0.002001, **kwargs) == "free_realign"
    # At/inside the conservative 5.5mm guard the same residual never uses
    # loose travel.  It must pass the strict 0.75mm branch before it can
    # command the final contact-capable 3mm.
    assert classify_mode(0.0055, 0.0, **kwargs) == "precontact"
    assert classify_mode(0.003001, 0.0, **kwargs) == "precontact"
    # The clearance boundary itself is the first contact-capable sample.
    assert classify_mode(0.003, 0.00088, **kwargs) == "contact"

    with np.testing.assert_raises(ValueError):
        classify_mode(0.020, -0.001, **kwargs)
    with np.testing.assert_raises(ValueError):
        classify_mode(
            0.020,
            0.001,
            free_bore_clearance_m=0.003,
            precontact_guard_m=0.003,
            free_cruise_tolerance_m=0.00200,
        )


def test_busbar_descent_handoff_uses_the_proven_actual_orientation():
    """A strict hole proof must never replay its old controller yaw lead."""

    alignment = _source_between(
        'elif phase == "BUSBAR_HOLE_ALIGNMENT":',
        'elif phase == "BUSBAR_DESCEND_TO_BOLT":',
    )

    assert "busbar_alignment_target_pos = cur_pos.copy()" in alignment
    assert "busbar_alignment_target_quat = cur_orientation.copy()" in alignment
    assert "bounded_busbar_descent_yaw_hold_delta(" not in alignment
    assert "BUSBAR_DESCENT_YAW_HOLD_MAX_RAD" not in alignment


def test_busbar_retract_attempt_reservation_blocks_the_sixth_action():
    (reserve,) = _load_pure_executor_functions(
        "reserve_busbar_descent_retract_attempt",
    )

    attempts = 0
    reservations = []
    for _ in range(6):
        attempts, reserved = reserve(attempts, 5)
        reservations.append(reserved)

    assert attempts == 5
    assert reservations == [True, True, True, True, True, False]

    with np.testing.assert_raises(TypeError):
        reserve(0.0, 5)
    with np.testing.assert_raises(ValueError):
        reserve(-1, 5)
    with np.testing.assert_raises(ValueError):
        reserve(0, 0)


def test_busbar_joint_targets_direct_rigid_bodies_with_matching_anchors():
    attach = _source_between(
        "def attach_busbar_to_gripper(",
        "def detach_busbar_from_gripper(",
    )

    assert "gripper_prim.HasAPI(UsdPhysics.RigidBodyAPI)" in attach
    assert "busbar_root_prim.HasAPI(UsdPhysics.RigidBodyAPI)" in attach
    assert "joint.CreateBody0Rel().SetTargets([gripper_link_path])" in attach
    assert "joint.CreateBody1Rel().SetTargets([busbar_root_path])" in attach
    assert "joint.CreateBody1Rel().SetTargets([busbar_mesh_path])" not in attach
    assert "gripper_world_matrix.GetInverse().Transform(" in attach
    assert "busbar_root_world_matrix.GetInverse().Transform(" in attach
    assert "gripper_world_matrix.Transform(" in attach
    assert "busbar_root_world_matrix.Transform(" in attach
    assert "_, local_quat0 = compute_local_offset(" in attach
    assert "_, local_quat1 = compute_local_offset(" in attach
    assert "frame_position_error > 0.001" in attach
    assert "frame_orientation_error > math.radians(1.0)" in attach


def test_busbar_joint_is_configured_disabled_and_removed_after_release():
    attach = _source_between(
        "def attach_busbar_to_gripper(",
        "def detach_busbar_from_gripper(",
    )
    detach = _source_between(
        "def detach_busbar_from_gripper(",
        "def nut_paths_for_index(",
    )

    disable_index = attach.index(
        "joint.CreateJointEnabledAttr(False).Set(False)"
    )
    body_index = attach.index(
        "joint.CreateBody0Rel().SetTargets([gripper_link_path])"
    )
    enable_index = attach.index(
        "joint.GetJointEnabledAttr().Set(True)"
    )
    assert disable_index < body_index < enable_index
    assert "joint.CreateCollisionEnabledAttr(False).Set(False)" in attach
    assert "joint.CreateExcludeFromArticulationAttr(True).Set(True)" in attach
    assert "stale_joint_path" in attach
    assert "stage.RemovePrim(stale_joint_path)" in attach

    assert "busbar_root_path" in detach
    assert "busbar_mesh_path" in detach
    assert "enabled_attr.Set(False)" in detach
    assert "stage.RemovePrim(joint_path)" in detach


def test_preflight_requires_busbar_rigid_body_root_and_collision_mesh():
    preflight = _source_between(
        "def validate_stage_contract(stage):",
        "def run_wheel_smoke_test(",
    )

    assert "root_prim.HasAPI(UsdPhysics.RigidBodyAPI)" in preflight
    assert "mesh_prim.HasAPI(UsdPhysics.CollisionAPI)" in preflight


def test_runtime_overlay_authors_only_exact_insertion_contact_envelopes():
    constants = _source_between(
        "BUSBAR_INSERTION_CONTACT_OFFSET_M =",
        "# 너트 Prim 경로",
    )
    configure = _source_between(
        "def configure_exact_busbar_insertion_contact_envelopes(stage):",
        "def create_runtime_stage_overlay(",
    )
    overlay = _source_between(
        "def create_runtime_stage_overlay(",
        "class Execute_Isaac_Busar(",
    )

    assert "BUSBAR_INSERTION_CONTACT_OFFSET_M = 0.00025" in constants
    assert "BUSBAR_INSERTION_REST_OFFSET_M = 0.0" in constants
    assert "STATION_BUSBAR_PATHS[station][1]" in constants
    assert "*STATION_BOLT_BODY_PATHS[station]" in constants
    assert "*STATION_BUSBAR_SUPPORT_PATHS[station]" in constants
    assert "for station in (3, 4, 5)" in constants

    assert (
        "STATION_BUSBAR_INSERTION_COLLIDER_PATHS.items()"
        in configure
    )
    assert "stage.GetPrimAtPath(collider_path)" in configure
    assert "prim.HasAPI(UsdPhysics.CollisionAPI)" in configure
    assert "PhysxSchema.PhysxCollisionAPI.Apply(prim)" in configure
    assert "CreateContactOffsetAttr(" in configure
    assert "BUSBAR_INSERTION_CONTACT_OFFSET_M" in configure
    assert "CreateRestOffsetAttr(" in configure
    assert "BUSBAR_INSERTION_REST_OFFSET_M" in configure
    assert "Traverse(" not in configure
    assert "PrimRange(" not in configure
    assert "MeshCollisionAPI" not in configure
    assert "CreateApproximationAttr" not in configure
    assert "FilteredPairsAPI" not in configure

    configure_call = (
        "configure_exact_busbar_insertion_contact_envelopes(stage)"
    )
    assert configure_call in overlay
    assert overlay.index(configure_call) < overlay.index("layer.Save()")
    assert "source_stage.Save" not in overlay


def test_preflight_verifies_authored_exact_insertion_envelope_values():
    validator = _source_between(
        "def _busbar_insertion_contact_envelope_failures(stage):",
        "def configure_exact_busbar_insertion_contact_envelopes(stage):",
    )
    preflight = _source_between(
        "def validate_stage_contract(stage):",
        "def run_wheel_smoke_test(",
    )

    assert "len(all_paths) != 15" in validator
    assert "len(set(all_paths)) != 15" in validator
    assert "len(collider_paths) != 5" in validator
    assert "prim.HasAPI(UsdPhysics.CollisionAPI)" in validator
    assert "prim.HasAPI(PhysxSchema.PhysxCollisionAPI)" in validator
    assert "GetContactOffsetAttr()" in validator
    assert "GetRestOffsetAttr()" in validator
    assert "attribute.HasAuthoredValueOpinion()" in validator
    assert "rel_tol=0.0" in validator
    assert "abs_tol=1.0e-9" in validator

    assert (
        "failures.extend(_busbar_insertion_contact_envelope_failures(stage))"
        in preflight
    )
    assert (
        "15 exact insertion contact envelopes (0.00025m/0m)"
        in preflight
    )


def test_active_station_battery_is_anchored_only_after_transport():
    configure = _source_between(
        "def set_station_battery_kinematic(stage, station, enabled):",
        "def create_runtime_stage_overlay(",
    )
    overlay = _source_between(
        "def create_runtime_stage_overlay(",
        "class Execute_Isaac_Busar(",
    )
    fine_setup = _source_between(
        'elif task == "FINE_ALIGNMENT":',
        'elif (\n                task == "ASSEMBLE_BUSBAR"',
    )
    preflight = _source_between(
        "def validate_stage_contract(stage):",
        "def run_wheel_smoke_test(",
    )

    assert "STATION_BATTERY_ROOT_PATHS.get(station)" in configure
    assert "prim.HasAPI(UsdPhysics.RigidBodyAPI)" in configure
    assert "GetRigidBodyEnabledAttr().Set(True)" in configure
    assert "GetKinematicEnabledAttr().Set(bool(enabled))" in configure
    assert "GetVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))" in configure
    assert (
        "GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))"
        in configure
    )
    assert "set_station_battery_kinematic(" not in overlay
    assert "set_station_battery_kinematic(" in fine_setup
    assert "station_busbar_support_top_z(" in fine_setup
    assert "FAILURE:BATTERY_FIXTURE_ANCHOR_FAILED" in fine_setup
    assert fine_setup.index(
        "set_station_battery_kinematic("
    ) < fine_setup.index("bolt_camera_pose_for_target(")
    assert "STATION_BATTERY_ROOT_PATHS[station]" in preflight
    assert "battery rigid body 누락" in preflight
    assert "GetKinematicEnabledAttr()" not in preflight


def test_busbar_observation_uses_live_bolt_lines_and_actual_mesh_vertices():
    observe = _source_between(
        "def observe_busbar_hole_alignment(",
        "def validate_latched_busbar_asset(",
    )
    vertex_bounds = _source_between(
        "def transformed_mesh_vertex_world_bounds(",
        "def station_busbar_support_top_z(",
    )
    support_top = _source_between(
        "def station_busbar_support_top_z(",
        "def observe_busbar_hole_alignment(",
    )
    seating = _source_between(
        "def observe_latched_busbar_seating(",
        "def live_nut_busbar_seating_errors(",
    )
    setup = _source_between(
        'elif task == "ASSEMBLE_BUSBAR":',
        'elif task in ("SCAN_NUT1", "SCAN_NUT2"):',
    )

    assert "solve_busbar_bore_alignment(" in observe
    assert "world_xf(stage, bolt_path)" in observe
    assert "observation.bore_endpoint_residuals_m" in observe
    assert "top_endpoint_residuals_m" in observe
    assert "bottom_endpoint_residuals_m" in observe
    assert "emit_axis_diagnostics" in observe
    assert "projection XY=" in observe
    assert "ExtractTranslation()" not in observe

    assert "prim.IsA(UsdGeom.Mesh)" in vertex_bounds
    assert "GetPointsAttr().Get()" in vertex_bounds
    assert "transformed_usd_points_bounds(" in vertex_bounds
    assert "transformed_extent_world_bounds(" not in vertex_bounds
    assert "transformed_mesh_vertex_world_bounds(" in support_top
    assert "transformed_extent_world_bounds(" not in support_top
    assert "transformed_mesh_vertex_world_bounds(" in seating
    assert "transformed_extent_world_bounds(" not in seating
    assert "emit_axis_diagnostics=True" in setup


def test_support_top_uses_authored_vertices_and_fails_closed_on_invalid_points():
    def _translated_z(z_m):
        return (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, z_m, 1.0),
        )

    points = {
        "/support/a": (
            (-0.2, -0.1, -0.02),
            (0.2, 0.1, 0.10),
        ),
        "/support/b": (
            (-0.2, -0.1, -0.03),
            (0.2, 0.1, 0.20),
        ),
    }
    transforms = {
        "/support/a": _translated_z(1.0),
        "/support/b": _translated_z(0.9),
    }
    vertex_bounds, support_top, stage = _load_support_geometry_helpers(
        points,
        transforms,
    )

    np.testing.assert_allclose(
        vertex_bounds(stage, "/support/a"),
        (
            np.array([-0.2, -0.1, 0.98]),
            np.array([0.2, 0.1, 1.10]),
        ),
    )
    assert math.isclose(support_top(stage, 3), 1.10, abs_tol=1.0e-12)

    invalid_points = {
        **points,
        "/support/b": ((0.0, float("nan"), 0.0),),
    }
    invalid_vertex_bounds, invalid_support_top, invalid_stage = (
        _load_support_geometry_helpers(invalid_points, transforms)
    )
    with np.testing.assert_raises(RuntimeError):
        invalid_vertex_bounds(invalid_stage, "/support/b")
    with np.testing.assert_raises(RuntimeError):
        invalid_support_top(invalid_stage, 3)

    overflow_points = {
        **points,
        "/support/b": ((10 ** 10000, 0.0, 0.0),),
    }
    overflow_vertex_bounds, overflow_support_top, overflow_stage = (
        _load_support_geometry_helpers(overflow_points, transforms)
    )
    with np.testing.assert_raises(RuntimeError):
        overflow_vertex_bounds(overflow_stage, "/support/b")
    with np.testing.assert_raises(RuntimeError):
        overflow_support_top(overflow_stage, 3)


def test_busbar_assembly_closes_physical_holes_before_release():
    setup = _source_between(
        'elif task == "ASSEMBLE_BUSBAR":',
        'elif task in ("SCAN_NUT1", "SCAN_NUT2"):',
    )
    phases = _source_between(
        'elif phase == "BUSBAR_HOLE_ALIGNMENT":',
        "# ════════════════════════════════════════════════════════════════",
    )

    assert "observe_latched_busbar_seating(" in setup
    assert "initial_alignment.max_endpoint_residual_m" in setup
    assert "BUSBAR_HOLE_ALIGNMENT_TOL_M" in setup
    assert "observe_latched_busbar_seating(" in setup
    assert 'phase = "BUSBAR_HOLE_ALIGNMENT"' in setup
    assert "set_world_pose(" not in setup

    align = _source_between(
        'elif phase == "BUSBAR_HOLE_ALIGNMENT":',
        'elif phase == "BUSBAR_DESCEND_TO_BOLT":',
    )
    descend = _source_between(
        'elif phase == "BUSBAR_DESCEND_TO_BOLT":',
        'elif phase == "BUSBAR_RELEASE_SETTLE":',
    )
    release = _source_between(
        'elif phase == "BUSBAR_RELEASE_SETTLE":',
        "# ════════════════════════════════════════════════════════════════",
    )
    assert re.search(
        r"current_endpoint_max\s*<=\s*BUSBAR_HOLE_ALIGNMENT_TOL_M",
        align,
    )
    assert "busbar_hole_settle_count += 1" in align
    assert "BUSBAR_HOLE_ALIGNMENT_SETTLE_STEPS" in align
    assert "BUSBAR_ALIGNMENT_MAX_TRANSLATION_M" in align
    assert "BUSBAR_ALIGNMENT_MAX_YAW_RAD" in align
    assert "busbar_bore_unavailable_gate(" in align
    assert "top_endpoint_residuals" in align
    assert "bottom_endpoint_residuals" in align
    assert "BUSBAR_BORE_UNAVAILABLE_CONFIRM_STEPS" in align
    assert "FAILURE:BUSBAR_BORE_GEOMETRY_UNAVAILABLE" in align

    assert "busbar_min[2] - busbar_support_top_z" in descend
    command_index = descend.index("busbar_descent_z_command(")
    reentry_index = descend.index(
        'phase = "BUSBAR_HOLE_ALIGNMENT"',
        command_index,
    )
    detach_index = descend.index("detach_busbar_from_gripper(")
    assert command_index < reentry_index < detach_index
    assert "requires_hole_realignment" in descend
    assert "busbar_descent_realign_gate(" in descend
    assert "BUSBAR_DESCENT_REALIGN_CONFIRM_STEPS" in descend
    assert "BUSBAR_DESCENT_IMMEDIATE_REALIGN_M" in descend
    assert "residual_hold_required" in descend
    assert "residual_retract_required" in descend
    assert "requires_same_height_alignment" in descend
    assert "max(" in descend
    assert "requires_lead_stall_recovery" in descend
    assert "BUSBAR_DESCENT_LEAD_STALL_STEPS" in descend
    assert "BUSBAR_DESCENT_MIN_PROGRESS_M" in descend
    assert (
        "busbar_alignment_target_pos[2] ="
        in descend
    )
    assert "busbar_alignment_pose_settle_count = 0" in descend
    assert "busbar_hole_settle_count = 0" in descend
    assert "busbar_seat_settle_count = 0" in descend
    assert "busbar_release_previous_center = None" in descend
    assert "reserve_busbar_descent_retract_attempt(" in descend
    assert "retract_attempt_reserved" in descend
    assert "if not retract_attempt_reserved:" in descend
    assert "busbar_descent_realign_attempts +=" not in descend
    assert "BUSBAR_DESCENT_MAX_REALIGN_ATTEMPTS" in descend
    assert "FAILURE:BUSBAR_DESCENT_REALIGN_EXHAUSTED" in descend
    assert "seat_z_error <= BUSBAR_SEAT_Z_TOL_M" in descend
    assert "busbar_seat_settle_count += 1" in descend
    assert "detach_busbar_from_gripper(" in descend
    assert "joint_absent_after_detach" in descend
    assert "if not joint_absent_after_detach:" in descend
    assert (
        "busbar_release_pending_station ="
        in descend
    )
    reserve_index = descend.index(
        "reserve_busbar_descent_retract_attempt("
    )
    blocked_index = descend.index(
        "if not retract_attempt_reserved:",
        reserve_index,
    )
    action_index = descend.index(
        "actions = arm_controller.forward(",
        blocked_index,
    )
    assert reserve_index < blocked_index < action_index
    assert (
        descend.index("requires_same_height_alignment")
        < descend.index("performs_physical_retract")
        < reserve_index
    )
    assert descend.index("busbar_seat_settle_count") < descend.index(
        "detach_busbar_from_gripper("
    )

    assert "busbar_release_settle_count += 1" in release
    assert "BUSBAR_SEAT_SETTLE_STEPS" in release
    assert "and retract_pose_settled" in release
    success = release.index('publish_status("SUCCESS")')
    assert release.index("busbar_release_settle_count") < success
    assert "set_world_pose(" not in phases


def test_busbar_detach_is_idempotent_and_reports_the_absence_postcondition():
    detach = _source_between(
        "def detach_busbar_from_gripper(",
        "def busbar_grip_joint_absent(",
    )

    assert "for joint_path in joint_paths:" in detach
    assert "if not joint_prim.IsValid():" in detach
    assert "continue" in detach
    assert "stage.RemovePrim(joint_path)" in detach
    assert "return all(" in detach
    assert "not stage.GetPrimAtPath(joint_path).IsValid()" in detach
    assert "removed and" not in detach


def test_attached_or_detached_pending_busbar_blocks_unrelated_arm_tasks():
    dispatch = _source_between(
        "# A physical busbar FixedJoint is a transport latch",
        "# PICK 성공 뒤 실제 nut가 gripper에 남아 있는 동안은",
    )

    pending_index = dispatch.index(
        "if busbar_release_pending_station is not None:"
    )
    prelift_index = dispatch.index(
        "elif busbar_grasped and busbar_grasp_station is None:"
    )
    carried_index = dispatch.index("elif busbar_grasped:")
    assert pending_index < prelift_index < carried_index
    assert '{"ASSEMBLE_BUSBAR"}' in dispatch
    assert '{"PICK_BUSBAR"}' in dispatch
    for allowed_task in (
        '"MOVE_BATTERY_CENTER"',
        '"FINE_ALIGNMENT"',
        '"ASSEMBLE_BUSBAR"',
    ):
        assert allowed_task in dispatch
    assert "hold_current_arm_pose()" in dispatch
    assert "FAILURE:BUSBAR_TRANSPORT_TASK_MISMATCH" in dispatch


def test_cancel_preserves_attached_pick_latch_and_pick_can_resume_lift():
    cancel = _source_between(
        'if task == "CANCEL_ARM_TASK":',
        "# 안전장치: 팔 Task는 AMR이 도착해",
    )
    conflict = _source_between(
        "# ArmNode는 ExecuteArmTask를 직렬화하지만 /task_command 자체는",
        "# A physical busbar FixedJoint is a transport latch",
    )
    pick = _source_between(
        'elif task == "PICK_BUSBAR" and busbar_grasped:',
        'elif task == "MOVE_BATTERY_CENTER":',
    )
    fresh_pick = pick[
        pick.index('elif task == "PICK_BUSBAR":'):
    ]
    grasp = _source_between(
        'elif phase == "BUSBAR_GRASP":',
        'elif phase == "BUSBAR_LIFT":',
    )

    for abort in (cancel, conflict):
        assert "if not busbar_grasped:" in abort
        assert "busbar_pick_station = None" in abort
    assert "validate_latched_busbar_asset(" in pick
    assert "busbar_grip_joint_present(" in pick
    assert "busbar_grasp_start_z is None" in pick
    assert 'phase = "BUSBAR_LIFT"' in pick
    assert "FAILURE:BUSBAR_PICK_RESUME_INVALID" in pick
    assert "busbar_pick_station = None" in fresh_pick
    assert "busbar_grasp_start_z = None" in fresh_pick
    previous_latch_clear = grasp.index(
        "busbar_grasp_station = None"
    )
    new_payload_exposed = grasp.index("busbar_grasped = True")
    assert previous_latch_clear < new_payload_exposed


def test_post_detach_cancel_retries_release_without_reattaching_or_reinserting():
    setup = _source_between(
        'task == "ASSEMBLE_BUSBAR"\n'
        "                and busbar_release_pending_station is not None",
        'elif task in ("SCAN_NUT1", "SCAN_NUT2"):',
    )
    descend = _source_between(
        'elif phase == "BUSBAR_DESCEND_TO_BOLT":',
        'elif phase == "BUSBAR_RELEASE_SETTLE":',
    )
    release = _source_between(
        'elif phase == "BUSBAR_RELEASE_SETTLE":',
        "# ════════════════════════════════════════════════════════════════",
    )

    assert "busbar_release_pending_station is not None" in setup
    assert "busbar_grip_joint_absent(" in setup
    assert "observe_latched_busbar_seating(" in setup
    assert 'phase = "BUSBAR_RELEASE_SETTLE"' in setup
    pending_branch = setup[
        :setup.index('elif task == "ASSEMBLE_BUSBAR":')
    ]
    assert "attach_busbar_to_gripper(" not in pending_branch
    assert 'phase = "BUSBAR_HOLE_ALIGNMENT"' not in pending_branch
    assert (
        "busbar_release_pending_station ="
        in descend
    )
    success = release.index('publish_status("SUCCESS")')
    clear = release.index(
        "busbar_release_pending_station = None"
    )
    assert success < clear


def test_pick_success_latches_station_and_assembly_never_uses_mutable_station():
    pick = _source_between(
        'elif task == "PICK_BUSBAR":',
        'elif task == "MOVE_BATTERY_CENTER":',
    )
    lift = _source_between(
        'elif phase == "BUSBAR_LIFT":',
        'elif phase == "MOVE_BATTERY_CENTER_APPROACH":',
    )
    assembly_setup = _source_between(
        'elif task == "ASSEMBLE_BUSBAR":',
        'elif task in ("SCAN_NUT1", "SCAN_NUT2"):',
    )
    phases = _source_between(
        'elif phase == "BUSBAR_HOLE_ALIGNMENT":',
        "# ════════════════════════════════════════════════════════════════",
    )

    assert "busbar_pick_station = current_station" in pick
    latch = lift.index("busbar_grasp_station = busbar_pick_station")
    success = lift.index('publish_status("SUCCESS")')
    assert latch < success
    assert "validate_latched_busbar_asset(" in lift[:success]
    assert "busbar_grasp_station" in assembly_setup
    assert "observe_latched_busbar_seating(" in assembly_setup
    assert "current_station" not in phases
    assert phases.count("busbar_grasp_station") >= 3


def test_busbar_assembly_rotates_measured_downward_quaternion_and_settles_pose():
    setup = _source_between(
        'elif task == "ASSEMBLE_BUSBAR":',
        'elif task in ("SCAN_NUT1", "SCAN_NUT2"):',
    )
    move = _source_between(
        'elif phase == "MOVE_BATTERY_CENTER_APPROACH":',
        'elif phase == "FINE_ALIGNMENT":',
    )
    fine = _source_between(
        'elif phase == "FINE_ALIGNMENT":',
        "# [8단계] 버스바 수직 하강 체결",
    )
    align = _source_between(
        'elif phase == "BUSBAR_HOLE_ALIGNMENT":',
        'elif phase == "BUSBAR_DESCEND_TO_BOLT":',
    )

    assert "cur_ee, cur_ee_quat = _world_pose_for_path(" in setup
    assert "downward_tool_yaw(cur_ee_quat)" in setup
    assert "world_z_rotated_quat(" in setup
    assert "target_fine_yaw_rad" not in setup
    for phase_source in (move, fine, align):
        assert "consecutive_pose_motion_settle(" in phase_source
    assert "busbar_alignment_target_quat" in align
    assert "target_fine_yaw_rad" not in align


def test_downward_tool_yaw_and_world_z_composition_use_actual_quaternion():
    (
        quat_mul,
        quat_conj,
        quat_normalize,
        quat_rotate_vec,
        world_z_rotated_quat,
        downward_tool_yaw,
    ) = _load_pure_executor_functions(
        "_quat_mul",
        "_quat_conj",
        "_quat_normalize",
        "_quat_rotate_vec",
        "world_z_rotated_quat",
        "downward_tool_yaw",
    )
    del quat_mul, quat_conj, quat_normalize, quat_rotate_vec

    initial_yaw = math.radians(37.0)
    initial_quat = np.array(
        [
            0.0,
            -math.sin(0.5 * initial_yaw),
            math.cos(0.5 * initial_yaw),
            0.0,
        ]
    )
    assert math.isclose(
        downward_tool_yaw(initial_quat),
        initial_yaw,
        abs_tol=1.0e-9,
    )

    delta = math.radians(-4.5)
    rotated = world_z_rotated_quat(initial_quat, delta)
    assert math.isclose(
        downward_tool_yaw(rotated),
        initial_yaw + delta,
        abs_tol=1.0e-9,
    )

    with np.testing.assert_raises(ValueError):
        downward_tool_yaw(np.array([1.0, 0.0, 0.0, 0.0]))


def test_busbar_phases_fail_closed_on_missing_state_observation_and_joint_removal():
    align = _source_between(
        'elif phase == "BUSBAR_HOLE_ALIGNMENT":',
        'elif phase == "BUSBAR_DESCEND_TO_BOLT":',
    )
    descend = _source_between(
        'elif phase == "BUSBAR_DESCEND_TO_BOLT":',
        'elif phase == "BUSBAR_RELEASE_SETTLE":',
    )
    release = _source_between(
        'elif phase == "BUSBAR_RELEASE_SETTLE":',
        "# ════════════════════════════════════════════════════════════════",
    )

    assert "FAILURE:BUSBAR_ALIGNMENT_OBSERVATION_INVALID" in align
    assert "FAILURE:BUSBAR_DESCEND_STATE_MISSING" in descend
    assert "FAILURE:BUSBAR_DESCEND_OBSERVATION_INVALID" in descend
    assert "FAILURE:BUSBAR_JOINT_DETACH_FAILED" in descend
    assert "FAILURE:BUSBAR_RELEASE_STATE_MISSING" in release
    assert "FAILURE:BUSBAR_RELEASE_OBSERVATION_INVALID" in release
    assert "busbar_grip_joint_absent(" in release


def test_preflight_requires_busbar_hole_span_and_support_geometry():
    preflight = _source_between(
        "def validate_stage_contract(stage):",
        "def run_wheel_smoke_test(",
    )

    assert "STATION_BUSBAR_SUPPORT_PATHS[station]" in preflight
    assert "busbar 지지면 누락" in preflight
    assert "transformed_mesh_vertex_world_bounds(" in preflight
    assert "busbar 지지면 authored points " in preflight
    assert "검증 실패:" in preflight
    assert "station_busbar_support_top_z(stage, station)" in preflight
    assert "solve_busbar_hole_alignment(" in preflight
    assert "hole_fit.max_endpoint_residual_m" in preflight

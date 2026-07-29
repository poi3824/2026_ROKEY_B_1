"""Focused regressions for live rigid busbar hole-alignment retargeting."""

import ast
import math
from pathlib import Path

import numpy as np
import pytest


EXECUTOR_PATH = (
    Path(__file__).parents[3]
    / "isaacpjt"
    / "M0609"
    / "execute_isaac.py"
)
EXECUTOR_SOURCE = EXECUTOR_PATH.read_text(encoding="utf-8")


def _load_pure_helpers(*names):
    tree = ast.parse(EXECUTOR_SOURCE)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in selected} == set(names)
    namespace = {"math": math, "np": np}
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            filename="<busbar_alignment_helpers>",
            mode="exec",
        ),
        namespace,
    )
    return tuple(namespace[name] for name in names)


def _source_between(start_marker, end_marker):
    start = EXECUTOR_SOURCE.index(start_marker)
    end = EXECUTOR_SOURCE.index(end_marker, start)
    return EXECUTOR_SOURCE[start:end]


def test_rigid_retarget_rotates_the_actual_ee_to_hole_lever_arm():
    (
        _quat_mul,
        _quat_normalize,
        world_z_rotated_quat,
        rigid_target,
    ) = _load_pure_helpers(
        "_quat_mul",
        "_quat_normalize",
        "world_z_rotated_quat",
        "rigid_busbar_alignment_ee_target",
    )
    del _quat_mul, _quat_normalize, world_z_rotated_quat

    current_ee = np.array([1.20, -0.40, 0.61])
    current_quat = np.array([0.0, 0.0, 1.0, 0.0])
    hole_midpoint = np.array([1.00, -0.50])
    bolt_midpoint = np.array([1.01, -0.49])
    yaw_residual = math.radians(20.0)

    target_position, target_quat = rigid_target(
        current_ee,
        current_quat,
        hole_midpoint,
        bolt_midpoint,
        yaw_residual,
    )

    cosine = math.cos(yaw_residual)
    sine = math.sin(yaw_residual)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    expected_xy = (
        bolt_midpoint
        + rotation @ (current_ee[:2] - hole_midpoint)
    )
    assert target_position[:2] == pytest.approx(expected_xy)
    assert target_position[2] == pytest.approx(current_ee[2])
    assert np.linalg.norm(target_quat) == pytest.approx(1.0)

    # Applying the commanded EE rigid motion must land the actual hole
    # midpoint on the bolt midpoint exactly; a raw XY delta would not.
    corrected_hole_midpoint = (
        target_position[:2]
        + rotation @ (hole_midpoint - current_ee[:2])
    )
    assert corrected_hole_midpoint == pytest.approx(
        bolt_midpoint,
        abs=1.0e-12,
    )


def test_rigid_retarget_applies_midpoint_and_yaw_lead_as_one_atomic_transform():
    (
        _quat_mul,
        _quat_normalize,
        world_z_rotated_quat,
        rigid_target,
    ) = _load_pure_helpers(
        "_quat_mul",
        "_quat_normalize",
        "world_z_rotated_quat",
        "rigid_busbar_alignment_ee_target",
    )
    del _quat_mul, _quat_normalize, world_z_rotated_quat

    current_ee = np.array([0.30, -0.20, 0.61])
    current_quat = np.array([0.0, 0.0, 1.0, 0.0])
    hole_midpoint = np.array([0.10, -0.25])
    bolt_midpoint = np.array([0.11, -0.24])
    midpoint_lead = np.array([0.0002, -0.0001])
    yaw_residual = math.radians(-1.0)
    yaw_lead = math.radians(-0.05)

    target_position, _ = rigid_target(
        current_ee,
        current_quat,
        hole_midpoint,
        bolt_midpoint,
        yaw_residual,
        midpoint_lead_xy_m=midpoint_lead,
        yaw_lead_rad=yaw_lead,
    )

    commanded_yaw = yaw_residual + yaw_lead
    cosine = math.cos(commanded_yaw)
    sine = math.sin(commanded_yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    corrected_hole_midpoint = (
        target_position[:2]
        + rotation @ (hole_midpoint - current_ee[:2])
    )
    assert corrected_hole_midpoint == pytest.approx(
        bolt_midpoint + midpoint_lead,
        abs=1.0e-12,
    )


def test_bounded_lead_closes_a_fixed_three_millimetre_steady_state_shortfall():
    (update_lead,) = _load_pure_helpers(
        "bounded_busbar_alignment_lead_update",
    )

    lead_xy = np.zeros(2)
    lead_yaw = 0.0
    fixed_xy_shortfall = np.array([0.003, 0.0])
    fixed_yaw_shortfall = math.radians(0.5)

    for _ in range(12):
        # A deadband plant settles at command minus a fixed shortfall. Merely
        # reissuing the same exact target can never change this observation.
        actual_xy = lead_xy - fixed_xy_shortfall
        actual_yaw = lead_yaw - fixed_yaw_shortfall
        lead_xy, lead_yaw = update_lead(
            lead_xy,
            lead_yaw,
            -actual_xy,
            -actual_yaw,
            xy_step_m=0.00025,
            max_xy_lead_m=0.004,
            yaw_step_rad=math.radians(0.05),
            max_yaw_lead_rad=math.radians(0.75),
        )

    actual_xy = lead_xy - fixed_xy_shortfall
    actual_yaw = lead_yaw - fixed_yaw_shortfall
    assert np.linalg.norm(actual_xy) < 0.00075
    assert abs(actual_yaw) < math.radians(0.1)
    assert np.linalg.norm(lead_xy) == pytest.approx(0.003)
    assert lead_yaw == pytest.approx(math.radians(0.5))


def test_bounded_lead_never_exceeds_its_dedicated_xy_or_yaw_caps():
    (update_lead,) = _load_pure_helpers(
        "bounded_busbar_alignment_lead_update",
    )

    lead_xy = np.zeros(2)
    lead_yaw = 0.0
    for _ in range(100):
        lead_xy, lead_yaw = update_lead(
            lead_xy,
            lead_yaw,
            (1.0, 1.0),
            math.radians(20.0),
            xy_step_m=0.00025,
            max_xy_lead_m=0.004,
            yaw_step_rad=math.radians(0.05),
            max_yaw_lead_rad=math.radians(0.75),
        )

    assert np.linalg.norm(lead_xy) <= 0.004 + 1.0e-12
    assert abs(lead_yaw) <= math.radians(0.75) + 1.0e-12


def test_lead_response_uses_live_bore_improvement_not_rebased_ee_xy():
    (classify_response,) = _load_pure_helpers(
        "classify_busbar_alignment_lead_response",
    )

    # Live station-3 regression: the re-based EE XY projection was -0.233mm
    # (opposite the absolute target delta), but the two physical bore errors
    # improved from a 4.633mm maximum to 4.276mm and yaw followed -0.05deg.
    # A rigid pivot command must count that as a response.
    result = classify_response(
        (0.004633, 0.003500),
        (0.004276, 0.003300),
        (0.00025, 0.0),
        math.radians(-0.05),
        math.radians(0.9613),
        math.radians(0.9206),
        min_response_ratio=0.20,
    )

    assert result["geometry_responded"]
    assert result["yaw_responded"]
    assert result["endpoint_max_improvement_m"] == pytest.approx(
        0.000357
    )
    assert math.degrees(result["yaw_progress_rad"]) == pytest.approx(
        -0.0407,
        abs=1.0e-4,
    )


def test_lead_response_rejects_endpoint_noise_or_paired_bore_regression():
    (classify_response,) = _load_pure_helpers(
        "classify_busbar_alignment_lead_response",
    )

    noise = classify_response(
        (0.004000, 0.003000),
        (0.003960, 0.002980),
        (0.00025, 0.0),
        0.0,
        0.0,
        0.0,
        min_response_ratio=0.20,
    )
    assert not noise["geometry_responded"]
    assert not noise["yaw_responded"]

    # One bore cannot be traded for a much worse mate merely because an
    # individual endpoint appears to improve.
    regressed_pair = classify_response(
        (0.004633, 0.003000),
        (0.004276, 0.005000),
        (0.00025, 0.0),
        0.0,
        0.0,
        0.0,
        min_response_ratio=0.20,
    )
    assert not regressed_pair["geometry_responded"]


def test_lead_response_accepts_yaw_without_raw_xy_projection():
    (classify_response,) = _load_pure_helpers(
        "classify_busbar_alignment_lead_response",
    )

    response = classify_response(
        (0.004000, 0.003000),
        (0.004000, 0.003000),
        (0.00025, 0.0),
        math.radians(-0.05),
        math.radians(1.0),
        math.radians(0.96),
        min_response_ratio=0.20,
    )
    assert not response["geometry_responded"]
    assert response["yaw_responded"]


@pytest.mark.parametrize(
"invalid",
    [
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0, float("nan")]),
    ],
)
def test_rigid_retarget_rejects_invalid_ee_positions(invalid):
    (
        _quat_mul,
        _quat_normalize,
        world_z_rotated_quat,
        rigid_target,
    ) = _load_pure_helpers(
        "_quat_mul",
        "_quat_normalize",
        "world_z_rotated_quat",
        "rigid_busbar_alignment_ee_target",
    )
    del _quat_mul, _quat_normalize, world_z_rotated_quat

    with pytest.raises(ValueError):
        rigid_target(
            invalid,
            np.array([0.0, 0.0, 1.0, 0.0]),
            (0.0, 0.0),
            (0.0, 0.0),
            0.0,
        )


def test_alignment_loop_grows_bounded_lead_after_actual_motion_settles():
    alignment_loop = _source_between(
        'elif phase == "BUSBAR_HOLE_ALIGNMENT":',
        'elif phase == "BUSBAR_DESCEND_TO_BOLT":',
    )

    assert "rigid_busbar_alignment_ee_target(" in alignment_loop
    assert "alignment.hole_midpoint_xy_m" in alignment_loop
    assert "alignment.bolt_midpoint_xy_m" in alignment_loop
    assert "alignment.yaw_residual_rad" in alignment_loop
    assert "bounded_busbar_alignment_lead_update(" in alignment_loop
    assert "classify_busbar_alignment_lead_response(" in alignment_loop
    assert "busbar_alignment_last_retarget_endpoint_residuals" in alignment_loop
    assert "busbar_alignment_stationary_count" in alignment_loop
    assert "actual_alignment_stationary" in alignment_loop
    assert "midpoint_lead_xy_m=(" in alignment_loop
    assert "yaw_lead_rad=(" in alignment_loop
    assert "atomic_target_position" in alignment_loop
    assert "busbar_alignment_target_pos[:2] += lead_delta_xy" not in alignment_loop
    assert "BUSBAR_ALIGNMENT_CONTROLLER_LEAD_MAX_M" in alignment_loop
    assert "BUSBAR_ALIGNMENT_CONTROLLER_YAW_MAX_RAD" in alignment_loop
    assert "live_busbar_bore_tip_axial_clearance_m(" in alignment_loop
    assert "BUSBAR_DESCENT_FREE_BORE_CLEARANCE_M" in alignment_loop
    assert "BUSBAR_DESCENT_REALIGN_RETRACT_M" in alignment_loop
    assert "BUSBAR_ALIGNMENT_CONTROLLER_MAX_NO_RESPONSE_STEPS" in alignment_loop
    assert "FAILURE:BUSBAR_ALIGNMENT_CONTROLLER_UNRESPONSIVE" in alignment_loop
    assert "FAILURE:BUSBAR_ALIGNMENT_CONTROLLER_LEAD_EXHAUSTED" in alignment_loop
    assert "previous actual response=" in alignment_loop
    assert "previous_actual_xy_response" not in alignment_loop
    assert "0.00005 / correction_norm" not in alignment_loop
    assert "math.radians(0.01)" not in alignment_loop
    assert "BUSBAR_ALIGNMENT_MAX_TRANSLATION_M" in alignment_loop
    assert "BUSBAR_ALIGNMENT_MAX_YAW_RAD" in alignment_loop
    # A physically proven bore pose must become the complete descent baseline.
    # The previous RMP target can contain bounded XY/yaw lead and is not valid
    # to replay while the wrist begins moving down.
    assert "busbar_alignment_target_pos = cur_pos.copy()" in alignment_loop
    assert "busbar_alignment_target_quat = cur_orientation.copy()" in alignment_loop
    assert "bounded_busbar_descent_yaw_hold_delta(" not in alignment_loop
    assert "BUSBAR_DESCENT_YAW_HOLD_MAX_RAD" not in alignment_loop
    assert "target_mid_pos[:2] = (\n                                cur_pos[:2]" in alignment_loop
    assert "hold_current_arm_pose()" in alignment_loop

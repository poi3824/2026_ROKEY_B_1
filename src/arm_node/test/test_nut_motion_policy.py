import sys
from pathlib import Path

import numpy as np
import pytest


RMPFLOW_DIR = (
    Path(__file__).parents[3]
    / "isaacpjt"
    / "M0609"
    / "rmpflow"
)
sys.path.insert(0, str(RMPFLOW_DIR))

from nut_motion_policy import (  # noqa: E402
    compensated_ee_target_for_nut_axis,
    consecutive_planar_alignment,
    consecutive_pose_settle,
    exact_collision_filter_targets,
    lead_xy_target,
    lead_z_target,
    nut_ee_coupling_error,
    nut_from_ee_offset,
    planned_screw_depth,
    rate_limited_z_target,
    required_seating_depth,
    required_seating_depth_to_surface,
    screw_pass_target_z,
    signed_yaw_delta_wxyz,
    thread_entry_ee_z,
)


def test_thread_entry_ee_z_places_nut_bottom_at_live_bolt_clearance():
    bolt_tip_z = 0.153302
    nut_height_m = 0.00847725
    latched_nut_from_ee_z = -0.226
    entry_clearance_m = 0.001

    ee_z = thread_entry_ee_z(
        bolt_tip_z,
        nut_height_m,
        latched_nut_from_ee_z,
        entry_clearance_m,
    )
    nut_center_z = ee_z + latched_nut_from_ee_z
    nut_bottom_z = nut_center_z - 0.5 * nut_height_m

    assert nut_bottom_z == pytest.approx(
        bolt_tip_z + entry_clearance_m
    )
    assert ee_z == pytest.approx(0.384540625)


def test_thread_entry_target_tracks_bolt_and_latched_grasp_changes():
    baseline = thread_entry_ee_z(0.153, 0.008, -0.220, 0.001)

    assert thread_entry_ee_z(
        0.158,
        0.008,
        -0.220,
        0.001,
    ) == pytest.approx(baseline + 0.005)
    assert thread_entry_ee_z(
        0.153,
        0.008,
        -0.217,
        0.001,
    ) == pytest.approx(baseline - 0.003)


def test_dynamic_full_seating_depth_consumes_gap_and_nut_height():
    bolt_tip_z = 0.153302
    nut_height_m = 0.00847725
    entry_clearance_m = 0.001
    entry_nut_center_z = (
        bolt_tip_z + 0.5 * nut_height_m + entry_clearance_m
    )

    required = required_seating_depth(
        entry_nut_center_z,
        bolt_tip_z,
        nut_height_m,
    )
    tolerant_required = required_seating_depth(
        entry_nut_center_z,
        bolt_tip_z,
        nut_height_m,
        axial_tolerance_m=0.0005,
    )

    assert required == pytest.approx(
        entry_clearance_m + nut_height_m
    )
    assert required == pytest.approx(0.00947725)
    assert tolerant_required == pytest.approx(0.00897725)


def test_dynamic_seating_depth_accounts_for_existing_thread_overlap():
    bolt_tip_z = 0.15
    nut_height_m = 0.008
    target_overlap_m = 0.002
    # The entry pose is already 0.5 mm onto the thread.
    entry_nut_center_z = bolt_tip_z + 0.5 * nut_height_m - 0.0005

    required = required_seating_depth(
        entry_nut_center_z,
        bolt_tip_z,
        nut_height_m,
        target_thread_overlap_m=target_overlap_m,
    )

    assert required == pytest.approx(0.0015)


@pytest.mark.parametrize(
    "bolt_tip_z,expected_depth_m",
    [
        (0.160008, 0.009352),
        (0.160533, 0.009877),
    ],
)
def test_station5_surface_seating_depth_uses_actual_slot_tip(
    bolt_tip_z,
    expected_depth_m,
):
    busbar_top_z = 0.150856
    nut_height_m = 0.00847725
    entry_clearance_m = 0.0002
    entry_nut_center_z = (
        bolt_tip_z + entry_clearance_m + 0.5 * nut_height_m
    )

    required = required_seating_depth_to_surface(
        entry_nut_center_z,
        nut_height_m,
        busbar_top_z,
    )

    assert required == pytest.approx(expected_depth_m)
    assert required * 1000.0 == pytest.approx(
        expected_depth_m * 1000.0
    )


def test_surface_seating_depth_applies_axial_tolerance_and_clamps_at_zero():
    assert required_seating_depth_to_surface(
        0.155,
        0.008,
        0.15,
        axial_tolerance_m=0.0005,
    ) == pytest.approx(0.0005)
    assert required_seating_depth_to_surface(
        0.154,
        0.008,
        0.15,
        axial_tolerance_m=0.0005,
    ) == pytest.approx(0.0)


def test_six_350_degree_passes_plan_about_ten_mm_of_thread_lead():
    pass_angle_deg = 350.0
    pass_count = 6
    thread_pitch_m = 0.001714406

    total_depth = planned_screw_depth(
        pass_angle_deg,
        pass_count,
        thread_pitch_m,
    )
    per_pass_depth = screw_pass_target_z(
        0.4,
        pass_angle_deg,
        thread_pitch_m,
    )

    assert total_depth == pytest.approx(0.010000701666666665)
    assert total_depth * 1000.0 == pytest.approx(
        10.000701666666666
    )
    assert 0.4 - per_pass_depth == pytest.approx(
        total_depth / pass_count
    )


def test_six_pass_plan_covers_geometry_driven_full_seating_depth():
    nut_height_m = 0.00847725
    entry_clearance_m = 0.001
    bolt_tip_z = 0.153302
    entry_nut_center_z = (
        bolt_tip_z + 0.5 * nut_height_m + entry_clearance_m
    )

    planned = planned_screw_depth(350.0, 6, 0.001714406)
    required = required_seating_depth(
        entry_nut_center_z,
        bolt_tip_z,
        nut_height_m,
    )

    assert planned > required
    assert planned - required == pytest.approx(0.000523451666666665)


@pytest.mark.parametrize(
    "bolt_tip,nut_height,nut_from_ee,clearance",
    [
        (np.nan, 0.008, -0.2, 0.001),
        (0.15, 0.0, -0.2, 0.001),
        (0.15, -0.008, -0.2, 0.001),
        (0.15, 0.008, np.inf, 0.001),
        (0.15, 0.008, -0.2, -0.001),
    ],
)
def test_thread_entry_ee_z_rejects_invalid_geometry(
    bolt_tip,
    nut_height,
    nut_from_ee,
    clearance,
):
    with pytest.raises(ValueError):
        thread_entry_ee_z(
            bolt_tip,
            nut_height,
            nut_from_ee,
            clearance,
        )


@pytest.mark.parametrize(
    "entry_center,tip,height,overlap,tolerance,exception",
    [
        (np.nan, 0.15, 0.008, None, 0.0, ValueError),
        (0.155, 0.15, 0.0, None, 0.0, ValueError),
        (0.155, 0.15, 0.008, -0.001, 0.0, ValueError),
        (0.155, 0.15, 0.008, 0.009, 0.0, ValueError),
        (0.155, 0.15, 0.008, 0.002, -0.001, ValueError),
    ],
)
def test_required_seating_depth_rejects_invalid_geometry(
    entry_center,
    tip,
    height,
    overlap,
    tolerance,
    exception,
):
    with pytest.raises(exception):
        required_seating_depth(
            entry_center,
            tip,
            height,
            target_thread_overlap_m=overlap,
            axial_tolerance_m=tolerance,
        )


@pytest.mark.parametrize(
    "entry_center,height,surface,tolerance",
    [
        (np.nan, 0.008, 0.15, 0.0),
        (0.155, 0.0, 0.15, 0.0),
        (0.155, -0.008, 0.15, 0.0),
        (0.155, 0.008, np.inf, 0.0),
        (0.155, 0.008, 0.15, -0.001),
    ],
)
def test_surface_seating_depth_rejects_invalid_geometry(
    entry_center,
    height,
    surface,
    tolerance,
):
    with pytest.raises(ValueError):
        required_seating_depth_to_surface(
            entry_center,
            height,
            surface,
            axial_tolerance_m=tolerance,
        )


@pytest.mark.parametrize(
    "angle,passes,pitch,exception",
    [
        (np.nan, 6, 0.001714406, ValueError),
        (-1.0, 6, 0.001714406, ValueError),
        (350.0, 0, 0.001714406, ValueError),
        (350.0, 6.0, 0.001714406, TypeError),
        (350.0, True, 0.001714406, TypeError),
        (350.0, 6, 0.0, ValueError),
    ],
)
def test_planned_screw_depth_rejects_invalid_plan(
    angle,
    passes,
    pitch,
    exception,
):
    with pytest.raises(exception):
        planned_screw_depth(angle, passes, pitch)


def test_compensated_ee_target_subtracts_actual_grasp_offset():
    ee = np.array([0.5720, -0.0990, 0.9])
    nut = np.array([0.5720, -0.0940, 0.86])
    bolt_axis = np.array([1.0552117, 0.3722289, 0.1248])

    target = compensated_ee_target_for_nut_axis(
        bolt_axis,
        ee,
        nut,
        target_z=0.6,
    )

    # The nut is held +5 mm in world Y from the EE, so the EE must be sent
    # 5 mm to the opposite side of the bolt axis.
    np.testing.assert_allclose(
        target,
        np.array([bolt_axis[0], bolt_axis[1] - 0.005, 0.6]),
    )


@pytest.mark.parametrize(
    "bolt,ee,nut,target_z",
    [
        ([1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.6),
        ([1.0, 2.0], [0.0, 0.0], [0.0, 0.0, 0.0], 0.6),
        ([1.0, 2.0], [0.0, 0.0, 0.0], [0.0, np.nan, 0.0], 0.6),
        ([1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], np.inf),
    ],
)
def test_compensated_ee_target_rejects_invalid_state(
    bolt,
    ee,
    nut,
    target_z,
):
    with pytest.raises(ValueError):
        compensated_ee_target_for_nut_axis(
            bolt,
            ee,
            nut,
            target_z=target_z,
        )


def test_planar_alignment_requires_twelve_consecutive_physical_samples():
    bolt = np.array([1.0552, 0.3722])
    count = 0

    for _ in range(11):
        count, aligned, error = consecutive_planar_alignment(
            np.array([1.05569, 0.3722, 0.5]),
            bolt,
            count,
            tolerance_m=0.0005,
            required_steps=12,
        )
        assert not aligned
        assert error == pytest.approx(0.00049)

    count, aligned, _ = consecutive_planar_alignment(
        np.array([1.05569, 0.3722, 0.5]),
        bolt,
        count,
        tolerance_m=0.0005,
        required_steps=12,
    )
    assert count == 12
    assert aligned


def test_planar_alignment_resets_after_one_off_axis_sample():
    count, aligned, error = consecutive_planar_alignment(
        np.array([1.05621, 0.3722, 0.5]),
        np.array([1.0552, 0.3722]),
        11,
        tolerance_m=0.0005,
        required_steps=12,
    )

    assert count == 0
    assert not aligned
    assert error == pytest.approx(0.00101)


def test_each_regrasped_screw_pass_moves_nut_by_one_physical_lead():
    pitch = 0.001714406
    pass_angle = 350.0
    regrasp_height = 0.005
    initial_ee_z = 0.3697
    initial_nut_from_ee_z = -0.226

    first_ee_end = screw_pass_target_z(
        initial_ee_z,
        pass_angle,
        pitch,
    )
    first_nut_end = first_ee_end + initial_nut_from_ee_z

    # Regrasp closes 5 mm above the prior EE pose.  The physical nut stays
    # put while open, so the new nut-from-EE offset grows by the same 5 mm.
    second_start_ee_z = first_ee_end + regrasp_height
    second_nut_from_ee_z = (
        first_nut_end - second_start_ee_z
    )
    second_ee_end = screw_pass_target_z(
        second_start_ee_z,
        pass_angle,
        pitch,
    )
    second_nut_end = second_ee_end + second_nut_from_ee_z

    expected_pass_lead = (pass_angle / 360.0) * pitch
    assert first_nut_end == pytest.approx(
        initial_ee_z + initial_nut_from_ee_z - expected_pass_lead
    )
    assert second_nut_end == pytest.approx(
        first_nut_end - expected_pass_lead
    )


def test_signed_yaw_delta_accumulates_a_full_turn_across_wraparound():
    accumulated = 0.0
    previous_degrees = 0.0
    for current_degrees in range(2, 352, 2):
        previous = np.array([
            np.cos(np.radians(previous_degrees) / 2.0),
            0.0,
            0.0,
            np.sin(np.radians(previous_degrees) / 2.0),
        ])
        current = np.array([
            np.cos(np.radians(current_degrees) / 2.0),
            0.0,
            0.0,
            np.sin(np.radians(current_degrees) / 2.0),
        ])
        accumulated += signed_yaw_delta_wxyz(previous, current)
        previous_degrees = current_degrees

    assert accumulated == pytest.approx(np.radians(350.0))


@pytest.mark.parametrize(
    "previous,current",
    [
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
        ([1.0, 0.0, 0.0, np.nan], [1.0, 0.0, 0.0, 0.0]),
    ],
)
def test_signed_yaw_delta_rejects_invalid_quaternions(previous, current):
    with pytest.raises(ValueError):
        signed_yaw_delta_wxyz(previous, current)


@pytest.mark.parametrize(
    "start_z,angle,pitch",
    [
        (np.nan, 350.0, 0.0017),
        (0.37, -1.0, 0.0017),
        (0.37, 350.0, 0.0),
    ],
)
def test_screw_pass_target_rejects_invalid_state(
    start_z,
    angle,
    pitch,
):
    with pytest.raises(ValueError):
        screw_pass_target_z(start_z, angle, pitch)


def test_pose_settle_requires_twelve_consecutive_actual_samples():
    count = 0

    for _ in range(11):
        count, settled = consecutive_pose_settle(
            0.02,
            np.deg2rad(3.0),
            count,
            position_tolerance_m=0.02,
            orientation_tolerance_rad=np.deg2rad(3.0),
            required_steps=12,
        )
        assert not settled

    count, settled = consecutive_pose_settle(
        0.02,
        np.deg2rad(3.0),
        count,
        position_tolerance_m=0.02,
        orientation_tolerance_rad=np.deg2rad(3.0),
        required_steps=12,
    )

    assert count == 12
    assert settled


def test_pose_settle_resets_after_one_out_of_tolerance_sample():
    count, settled = consecutive_pose_settle(
        0.001,
        0.01,
        8,
        position_tolerance_m=0.02,
        orientation_tolerance_rad=np.deg2rad(3.0),
        required_steps=12,
    )
    assert count == 9
    assert not settled

    count, settled = consecutive_pose_settle(
        0.021,
        0.01,
        count,
        position_tolerance_m=0.02,
        orientation_tolerance_rad=np.deg2rad(3.0),
        required_steps=12,
    )
    assert count == 0
    assert not settled

    count, settled = consecutive_pose_settle(
        0.001,
        np.deg2rad(3.01),
        11,
        position_tolerance_m=0.02,
        orientation_tolerance_rad=np.deg2rad(3.0),
        required_steps=12,
    )
    assert count == 0
    assert not settled


@pytest.mark.parametrize(
    "position_error,orientation_error,previous_count,"
    "position_tolerance,orientation_tolerance,required_steps,exception",
    [
        (np.nan, 0.0, 0, 0.02, 0.1, 12, ValueError),
        (0.0, np.inf, 0, 0.02, 0.1, 12, ValueError),
        (-0.001, 0.0, 0, 0.02, 0.1, 12, ValueError),
        (0.0, -0.001, 0, 0.02, 0.1, 12, ValueError),
        (0.0, 0.0, -1, 0.02, 0.1, 12, ValueError),
        (0.0, 0.0, 0.5, 0.02, 0.1, 12, TypeError),
        (0.0, 0.0, 0, 0.02, 0.1, 0, ValueError),
        (0.0, 0.0, 0, 0.02, 0.1, 1.5, TypeError),
    ],
)
def test_pose_settle_rejects_invalid_state(
    position_error,
    orientation_error,
    previous_count,
    position_tolerance,
    orientation_tolerance,
    required_steps,
    exception,
):
    with pytest.raises(exception):
        consecutive_pose_settle(
            position_error,
            orientation_error,
            previous_count,
            position_tolerance_m=position_tolerance,
            orientation_tolerance_rad=orientation_tolerance,
            required_steps=required_steps,
        )


@pytest.mark.parametrize(
    "nut_index,peg_index",
    [
        (1, 5),
        (2, 4),
        (3, 3),
        (4, 1),
        (5, 0),
        (6, 2),
    ],
)
def test_each_nut_collision_filter_has_one_exact_own_peg_only(
    nut_index,
    peg_index,
):
    nut_root = f"/World/nut{nut_index}"
    body_path = f"{nut_root}/geo/PolyShape"
    peg_path = (
        "/World/Nova_Carter/chassis_link/carter_tray/"
        f"peg_{peg_index}"
    )
    body, targets = exact_collision_filter_targets(
        body_path,
        peg_path,
    )

    assert body == body_path
    assert targets == (peg_path,)
    assert nut_root not in (body, *targets)
    assert "/World/m0609" not in targets
    assert "/World/Nova_Carter" not in targets
    assert all(
        f"/World/Nova_Carter/chassis_link/carter_tray/peg_{index}"
        not in targets
        for index in range(6)
        if index != peg_index
    )
    assert all("finger" not in target for target in targets)


@pytest.mark.parametrize(
    "body,counterpart,exception",
    [
        ("/", "/World/peg", ValueError),
        ("World/nut2", "/World/peg", ValueError),
        ("/World/nut2/", "/World/peg", ValueError),
        ("/World/nut2", " /World/peg", ValueError),
        ("/World/nut2", "/World/nut2", ValueError),
        (None, "/World/peg", TypeError),
    ],
)
def test_exact_collision_filter_rejects_broad_or_invalid_pair(
    body,
    counterpart,
    exception,
):
    with pytest.raises(exception):
        exact_collision_filter_targets(body, counterpart)


def test_vertical_lift_command_cannot_jump_to_full_80mm_goal():
    start_z = 0.72
    command_z = rate_limited_z_target(
        start_z + 0.08,
        start_z,
        max_step=0.0005,
    )

    assert command_z == pytest.approx(start_z + 0.0005)


def test_vertical_command_converges_without_overshooting_goal():
    command_z = 0.7998
    command_z = rate_limited_z_target(
        0.8,
        command_z,
        max_step=0.0005,
    )

    assert command_z == pytest.approx(0.8)


@pytest.mark.parametrize("max_step", [0.0, -0.001, np.nan])
def test_vertical_command_rejects_unsafe_step(max_step):
    with pytest.raises(ValueError):
        rate_limited_z_target(0.8, 0.72, max_step=max_step)


def test_settled_reference_separates_safe_seating_from_transport_slip():
    ee = np.array([0.5, -0.2, 0.72])
    nut = np.array([0.5, -0.2, 0.68])
    release_reference = nut_from_ee_offset(ee, nut)

    # The newly dynamic nut self-seats 7 mm inside already-closed fingers.
    seated_nut = nut + np.array([0.007, 0.0, 0.0])
    assert nut_ee_coupling_error(
        ee,
        seated_nut,
        release_reference,
    ) == pytest.approx(0.007)
    settled_reference = nut_from_ee_offset(ee, seated_nut)

    # Transporting the settled grasp preserves the new relative offset.  The
    # original release reference would incorrectly accumulate seating drift.
    lift = np.array([0.0, 0.0, 0.08])
    assert nut_ee_coupling_error(
        ee + lift,
        seated_nut + lift,
        settled_reference,
    ) == pytest.approx(0.0)
    assert nut_ee_coupling_error(
        ee + lift,
        seated_nut + lift,
        release_reference,
    ) == pytest.approx(0.007)


def test_coupling_error_is_zero_when_nut_and_ee_move_together():
    initial_ee = np.array([0.5, -0.2, 0.72])
    initial_nut = np.array([0.5, -0.2, 0.68])
    reference = initial_nut - initial_ee

    error = nut_ee_coupling_error(
        initial_ee + np.array([0.0, 0.0, 0.08]),
        initial_nut + np.array([0.0, 0.0, 0.08]),
        reference,
    )

    assert error == pytest.approx(0.0)


def test_coupling_error_detects_nut_left_on_supply_peg():
    initial_ee = np.array([0.5, -0.2, 0.72])
    initial_nut = np.array([0.5, -0.2, 0.68])
    reference = initial_nut - initial_ee

    error = nut_ee_coupling_error(
        initial_ee + np.array([0.0, 0.0, 0.08]),
        initial_nut,
        reference,
    )

    assert error == pytest.approx(0.08)


@pytest.mark.parametrize(
    "end_effector,nut,reference",
    [
        ([0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, np.nan], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0], [0.0, np.inf, 0.0], [0.0, 0.0, 0.0]),
    ],
)
def test_coupling_error_rejects_invalid_positions(
    end_effector,
    nut,
    reference,
):
    with pytest.raises(ValueError):
        nut_ee_coupling_error(end_effector, nut, reference)


def test_far_target_is_not_modified():
    desired = np.array([1.0, 2.0, 0.9])
    current = np.array([0.9, 2.0, 0.8])

    command, lead = lead_xy_target(desired, current)

    np.testing.assert_allclose(command, desired)
    np.testing.assert_allclose(lead, np.zeros(2))


def test_near_target_leads_in_residual_direction_and_preserves_z():
    desired = np.array([1.0552, 0.3722, 0.9])
    current = np.array([1.05845, 0.35993, 0.90021])

    command, lead = lead_xy_target(
        desired,
        current,
        max_lead_step=1.0,
    )

    residual = desired[:2] - current[:2]
    np.testing.assert_allclose(
        command[:2] - desired[:2],
        2.0 * residual,
    )
    np.testing.assert_allclose(lead, 2.0 * residual)
    assert command[2] == desired[2]


def test_xy_lead_is_norm_bounded():
    desired = np.array([1.0, 2.0, 0.6])
    current = np.array([0.979, 1.979, 0.9])

    command, lead = lead_xy_target(
        desired,
        current,
        max_lead_step=1.0,
    )

    assert np.linalg.norm(command[:2] - desired[:2]) == pytest.approx(0.03)
    assert np.linalg.norm(lead) == pytest.approx(0.03)
    assert command[2] == 0.6


def test_lead_change_is_rate_limited():
    desired = np.array([1.0552, 0.3722, 0.9])
    current = np.array([1.05845, 0.35993, 0.90021])

    command, lead = lead_xy_target(
        desired,
        current,
        previous_lead=np.zeros(2),
    )

    assert np.linalg.norm(lead) == pytest.approx(0.001)
    np.testing.assert_allclose(command[:2], desired[:2] + lead)


def test_stale_lead_decays_at_the_same_rate_outside_activation_radius():
    desired = np.array([1.0, 2.0, 0.9])
    current = np.array([0.9, 2.0, 0.8])
    previous = np.array([0.01, 0.0])

    _, lead = lead_xy_target(desired, current, previous)

    np.testing.assert_allclose(lead, np.array([0.009, 0.0]))


@pytest.mark.parametrize(
    "desired,current",
    [
        ([1.0, 2.0], [1.0, 2.0, 0.9]),
        ([1.0, 2.0, 0.9], [1.0, np.nan, 0.9]),
    ],
)
def test_invalid_positions_are_rejected(desired, current):
    with pytest.raises(ValueError):
        lead_xy_target(desired, current)


def test_z_lead_is_positive_bounded_and_preserves_xy():
    desired = np.array([0.5728, -0.1229, 0.8169])
    current = np.array([0.5728, -0.1229, 0.8069])

    command, lead = lead_z_target(
        desired,
        current,
        max_lead_step=1.0,
    )

    assert lead == pytest.approx(0.02)
    np.testing.assert_allclose(command[:2], desired[:2])
    assert command[2] == pytest.approx(0.8369)


def test_z_lead_change_is_rate_limited():
    desired = np.array([0.5728, -0.1229, 0.8169])
    current = np.array([0.5728, -0.1229, 0.8069])

    command, lead = lead_z_target(desired, current)

    assert lead == pytest.approx(0.001)
    assert command[2] == pytest.approx(desired[2] + 0.001)


def test_z_lead_never_commands_downward_compensation():
    desired = np.array([0.5728, -0.1229, 0.8169])
    current = np.array([0.5728, -0.1229, 0.8200])

    command, lead = lead_z_target(
        desired,
        current,
        previous_lead=0.004,
    )

    assert lead == pytest.approx(0.003)
    np.testing.assert_allclose(command[:2], desired[:2])
    assert command[2] >= desired[2]


def test_z_lead_rejects_invalid_state():
    desired = np.array([0.5728, -0.1229, 0.8169])
    current = np.array([0.5728, -0.1229, 0.8069])

    with pytest.raises(ValueError):
        lead_z_target(desired, current, previous_lead=-0.001)

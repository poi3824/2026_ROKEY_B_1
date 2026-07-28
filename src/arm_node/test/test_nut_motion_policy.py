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
    consecutive_pose_settle,
    exact_collision_filter_targets,
    lead_xy_target,
    lead_z_target,
    nut_ee_coupling_error,
    nut_from_ee_offset,
    rate_limited_z_target,
)


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

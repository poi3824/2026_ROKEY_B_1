"""Pure motion helpers for the nut-to-bolt transfer phases."""

from __future__ import annotations

import numpy as np


def consecutive_pose_settle(
    position_error_m: float,
    orientation_error_rad: float,
    previous_count: int,
    *,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    required_steps: int,
):
    """Advance an actual-pose settle counter without time-based shortcuts.

    Nut scan motion deliberately separates wrist reorientation from the
    following planar move.  A position-only transition can otherwise complete
    immediately when the arm already happens to be at the requested height,
    while the wrist is still unwinding from screw insertion.
    """

    position_error = float(position_error_m)
    orientation_error = float(orientation_error_rad)
    position_tolerance = float(position_tolerance_m)
    orientation_tolerance = float(orientation_tolerance_rad)

    if not isinstance(previous_count, (int, np.integer)):
        raise TypeError("previous_count must be an integer")
    if not isinstance(required_steps, (int, np.integer)):
        raise TypeError("required_steps must be an integer")
    if previous_count < 0:
        raise ValueError("previous_count must be non-negative")
    if required_steps <= 0:
        raise ValueError("required_steps must be positive")
    if not np.all(np.isfinite([
        position_error,
        orientation_error,
        position_tolerance,
        orientation_tolerance,
    ])):
        raise ValueError("pose settle inputs must be finite")
    if (
        position_error < 0.0
        or orientation_error < 0.0
        or position_tolerance < 0.0
        or orientation_tolerance < 0.0
    ):
        raise ValueError("pose settle errors and tolerances must be non-negative")

    in_tolerance = (
        position_error <= position_tolerance
        and orientation_error <= orientation_tolerance
    )
    next_count = int(previous_count) + 1 if in_tolerance else 0
    return next_count, next_count >= int(required_steps)


def exact_collision_filter_targets(body_path: str, counterpart_path: str):
    """Return one validated, exact USD collision-filter relationship.

    Collision filtering is dangerous when either side names a broad parent
    prim: filtering ``/World/nut2`` against the robot or the whole cart also
    removes the finger and tray contacts that make a physical grasp possible.
    Keep the policy pure so its deliberately narrow scope can be tested
    without importing Isaac Sim.
    """

    paths = []
    for label, value in (
        ("body_path", body_path),
        ("counterpart_path", counterpart_path),
    ):
        if not isinstance(value, str):
            raise TypeError(f"{label} must be a string")
        if value != value.strip():
            raise ValueError(f"{label} must not contain outer whitespace")
        if not value.startswith("/") or value == "/" or value.endswith("/"):
            raise ValueError(f"{label} must be an absolute USD prim path")
        paths.append(value)

    body, counterpart = paths
    if body == counterpart:
        raise ValueError("collision-filter pair must name two distinct prims")
    return body, (counterpart,)


def nut_from_ee_offset(end_effector_position, nut_position):
    """Return the finite nut translation expressed relative to the EE."""

    end_effector = np.asarray(end_effector_position, dtype=float)
    nut = np.asarray(nut_position, dtype=float)
    if end_effector.shape != (3,) or nut.shape != (3,):
        raise ValueError("nut coupling positions must be 3-vectors")
    if (
        not np.all(np.isfinite(end_effector))
        or not np.all(np.isfinite(nut))
    ):
        raise ValueError("nut coupling positions must be finite")
    return nut - end_effector


def nut_ee_coupling_error(
    end_effector_position,
    nut_position,
    reference_nut_from_ee,
):
    """Measure translation drift of a grasped nut relative to the EE.

    Merely observing both objects above their starting Z does not prove that
    the gripper still carries the nut.  A coupled nut must preserve the
    nut-from-EE translation captured when the closed fingers release it from
    the tray glue.
    """

    current = nut_from_ee_offset(
        end_effector_position,
        nut_position,
    )
    reference = np.asarray(reference_nut_from_ee, dtype=float)
    if reference.shape != (3,):
        raise ValueError("nut coupling positions must be 3-vectors")
    if not np.all(np.isfinite(reference)):
        raise ValueError("nut coupling positions must be finite")

    return float(np.linalg.norm(current - reference))


def thread_entry_ee_z(
    bolt_tip_z: float,
    nut_height_m: float,
    latched_nut_from_ee_z: float,
    entry_clearance_m: float,
):
    """Return the EE Z that puts the nut bottom above the actual bolt tip.

    ``latched_nut_from_ee_z`` follows :func:`nut_from_ee_offset`: it is the
    physical nut-centre Z minus the EE Z captured after transport settles.
    At the returned target, the nut's lower face is exactly
    ``entry_clearance_m`` above ``bolt_tip_z``.  This keeps thread entry tied
    to the live collider geometry instead of a station-specific EE height.
    """

    tip_z = float(bolt_tip_z)
    nut_height = float(nut_height_m)
    nut_from_ee_z = float(latched_nut_from_ee_z)
    clearance = float(entry_clearance_m)
    if not np.all(np.isfinite([
        tip_z,
        nut_height,
        nut_from_ee_z,
        clearance,
    ])):
        raise ValueError("thread-entry geometry must be finite")
    if nut_height <= 0.0:
        raise ValueError("nut_height_m must be positive")
    if clearance < 0.0:
        raise ValueError("entry_clearance_m must be non-negative")

    entry_nut_center_z = tip_z + 0.5 * nut_height + clearance
    return entry_nut_center_z - nut_from_ee_z


def required_seating_depth(
    entry_nut_center_z: float,
    bolt_tip_z: float,
    nut_height_m: float,
    *,
    target_thread_overlap_m: float | None = None,
    axial_tolerance_m: float = 0.0,
):
    """Return required physical nut descent from its measured entry pose.

    The calculation assumes that usable thread continues downward from the
    measured bolt tip.  A nut initially above the tip must first consume its
    entry gap, then descend by the requested axial overlap.  If no overlap is
    supplied, the full nut height is required.  ``axial_tolerance_m`` lowers
    only the success threshold; it never changes the commanded screw travel.
    """

    entry_center_z = float(entry_nut_center_z)
    tip_z = float(bolt_tip_z)
    nut_height = float(nut_height_m)
    target_overlap = (
        nut_height
        if target_thread_overlap_m is None
        else float(target_thread_overlap_m)
    )
    tolerance = float(axial_tolerance_m)
    if not np.all(np.isfinite([
        entry_center_z,
        tip_z,
        nut_height,
        target_overlap,
        tolerance,
    ])):
        raise ValueError("seating geometry must be finite")
    if nut_height <= 0.0:
        raise ValueError("nut_height_m must be positive")
    if target_overlap < 0.0 or target_overlap > nut_height:
        raise ValueError(
            "target_thread_overlap_m must be within the nut height"
        )
    if tolerance < 0.0:
        raise ValueError("axial_tolerance_m must be non-negative")

    entry_nut_bottom_z = entry_center_z - 0.5 * nut_height
    geometric_depth = max(
        0.0,
        entry_nut_bottom_z - tip_z + target_overlap,
    )
    return max(0.0, geometric_depth - tolerance)


def required_seating_depth_to_surface(
    entry_nut_center_z: float,
    nut_height_m: float,
    seat_surface_z: float,
    *,
    axial_tolerance_m: float = 0.0,
):
    """Return nut descent needed for its lower face to reach a seat surface.

    Unlike thread-overlap depth, surface seating is not bounded by the nut
    height.  A bolt tip can stand more than one nut height above the busbar,
    so completion must use the measured entry lower-face Z minus the actual
    support-surface Z.  ``axial_tolerance_m`` relaxes only the verification
    threshold and cannot make the returned depth negative.
    """

    entry_center_z = float(entry_nut_center_z)
    nut_height = float(nut_height_m)
    surface_z = float(seat_surface_z)
    tolerance = float(axial_tolerance_m)
    if not np.all(np.isfinite([
        entry_center_z,
        nut_height,
        surface_z,
        tolerance,
    ])):
        raise ValueError("surface-seating geometry must be finite")
    if nut_height <= 0.0:
        raise ValueError("nut_height_m must be positive")
    if tolerance < 0.0:
        raise ValueError("axial_tolerance_m must be non-negative")

    entry_nut_bottom_z = entry_center_z - 0.5 * nut_height
    geometric_depth = max(0.0, entry_nut_bottom_z - surface_z)
    return max(0.0, geometric_depth - tolerance)


def planned_screw_depth(
    pass_angle_deg: float,
    pass_count: int,
    thread_pitch_m: float,
):
    """Return total axial lead commanded by equal-angle screw passes."""

    angle_deg = float(pass_angle_deg)
    pitch_m = float(thread_pitch_m)
    if (
        not isinstance(pass_count, (int, np.integer))
        or isinstance(pass_count, (bool, np.bool_))
    ):
        raise TypeError("pass_count must be an integer")
    if not np.all(np.isfinite([angle_deg, pitch_m])):
        raise ValueError("screw plan must be finite")
    if angle_deg < 0.0:
        raise ValueError("pass_angle_deg must be non-negative")
    if pass_count <= 0:
        raise ValueError("pass_count must be positive")
    if pitch_m <= 0.0:
        raise ValueError("thread_pitch_m must be positive")

    return (angle_deg / 360.0) * int(pass_count) * pitch_m


def compensated_ee_target_for_nut_axis(
    bolt_axis_position,
    end_effector_position,
    nut_position,
    *,
    target_z: float,
):
    """Return an EE target that places the *physical nut body* on a bolt axis.

    The gripper can hold a nut several millimetres away from the EE frame
    origin.  Sending the EE itself to the bolt axis therefore preserves that
    grasp offset and misses the thread.  Recompute the current nut-from-EE
    translation from the physical body poses and subtract it from the desired
    bolt axis.  Only XY is compensated; the calibrated EE insertion height
    remains an independent target.
    """

    bolt_axis = np.asarray(bolt_axis_position, dtype=float)
    end_effector = np.asarray(end_effector_position, dtype=float)
    nut = np.asarray(nut_position, dtype=float)
    z = float(target_z)
    if bolt_axis.shape not in {(2,), (3,)}:
        raise ValueError("bolt_axis_position must be a 2- or 3-vector")
    if end_effector.shape != (3,) or nut.shape != (3,):
        raise ValueError("nut alignment positions must be 3-vectors")
    if (
        not np.all(np.isfinite(bolt_axis))
        or not np.all(np.isfinite(end_effector))
        or not np.all(np.isfinite(nut))
        or not np.isfinite(z)
    ):
        raise ValueError("nut alignment positions must be finite")

    current_nut_from_ee = nut_from_ee_offset(end_effector, nut)
    target = np.array(
        [
            bolt_axis[0] - current_nut_from_ee[0],
            bolt_axis[1] - current_nut_from_ee[1],
            z,
        ],
        dtype=float,
    )
    return target


def consecutive_planar_alignment(
    nut_position,
    bolt_axis_position,
    previous_count: int,
    *,
    tolerance_m: float,
    required_steps: int,
):
    """Require consecutive physical nut-centre samples on the bolt axis."""

    nut = np.asarray(nut_position, dtype=float)
    bolt_axis = np.asarray(bolt_axis_position, dtype=float)
    tolerance = float(tolerance_m)
    if nut.shape != (3,):
        raise ValueError("nut_position must be a 3-vector")
    if bolt_axis.shape not in {(2,), (3,)}:
        raise ValueError("bolt_axis_position must be a 2- or 3-vector")
    if not isinstance(previous_count, (int, np.integer)):
        raise TypeError("previous_count must be an integer")
    if not isinstance(required_steps, (int, np.integer)):
        raise TypeError("required_steps must be an integer")
    if previous_count < 0:
        raise ValueError("previous_count must be non-negative")
    if required_steps <= 0:
        raise ValueError("required_steps must be positive")
    if (
        not np.all(np.isfinite(nut))
        or not np.all(np.isfinite(bolt_axis))
        or not np.isfinite(tolerance)
    ):
        raise ValueError("planar alignment inputs must be finite")
    if tolerance < 0.0:
        raise ValueError("tolerance_m must be non-negative")

    error_m = float(np.linalg.norm(nut[:2] - bolt_axis[:2]))
    next_count = int(previous_count) + 1 if error_m <= tolerance else 0
    return next_count, next_count >= int(required_steps), error_m


def screw_pass_target_z(
    pass_start_ee_z: float,
    pass_angle_deg: float,
    thread_pitch_m: float,
):
    """Return a pass-local screw Z target using the physical thread lead.

    A regrasp deliberately closes the fingers above the previous EE pose, so
    each pass must start from the *new* actual EE height.  Applying cumulative
    depth to that regrasp pose either consumes the grasp-height offset again or
    loses the previous insertion.  Pass-local lead keeps the carried nut's
    axial motion equal to ``angle / 360 * pitch`` on every pass.
    """

    start_z = float(pass_start_ee_z)
    angle_deg = float(pass_angle_deg)
    pitch_m = float(thread_pitch_m)
    if not np.all(np.isfinite([start_z, angle_deg, pitch_m])):
        raise ValueError("screw pass state must be finite")
    if angle_deg < 0.0:
        raise ValueError("pass_angle_deg must be non-negative")
    if pitch_m <= 0.0:
        raise ValueError("thread_pitch_m must be positive")
    return start_z - (angle_deg / 360.0) * pitch_m


def signed_yaw_delta_wxyz(previous_quaternion, current_quaternion):
    """Return the wrapped world-Z yaw change between two WXYZ quaternions.

    Screw verification integrates this value every physics tick. Comparing
    only the first and last quaternion cannot distinguish a real 350-degree
    turn from the equivalent shortest-path -10-degree pose.
    """

    previous = np.asarray(previous_quaternion, dtype=float)
    current = np.asarray(current_quaternion, dtype=float)
    if previous.shape != (4,) or current.shape != (4,):
        raise ValueError("screw quaternions must be 4-vectors in WXYZ order")
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
        raise ValueError("screw quaternions must be finite")

    previous_norm = float(np.linalg.norm(previous))
    current_norm = float(np.linalg.norm(current))
    if previous_norm <= 1.0e-9 or current_norm <= 1.0e-9:
        raise ValueError("screw quaternions must be non-zero")
    previous = previous / previous_norm
    current = current / current_norm

    def _yaw(quaternion):
        w, x, y, z = quaternion
        return float(np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        ))

    delta = _yaw(current) - _yaw(previous)
    return float(np.arctan2(np.sin(delta), np.cos(delta)))


def rate_limited_z_target(
    desired_z: float,
    previous_command_z: float,
    *,
    max_step: float,
):
    """Move a Cartesian Z command toward its goal by at most ``max_step``.

    The physical nut grasp is friction-based.  Sending the entire lift height
    as a target in one physics tick creates an avoidable contact impulse even
    though RMPFlow later smooths the joint trajectory.  Bounding the target
    motion makes that input deterministic without changing completion checks.
    """

    desired = float(desired_z)
    previous = float(previous_command_z)
    step = float(max_step)
    if (
        not np.isfinite(desired)
        or not np.isfinite(previous)
        or not np.isfinite(step)
    ):
        raise ValueError("Z target state must be finite")
    if step <= 0.0:
        raise ValueError("max_step must be positive")

    return previous + float(np.clip(
        desired - previous,
        -step,
        step,
    ))


def lead_xy_target(
    desired_position,
    current_position,
    previous_lead=None,
    *,
    activation_radius: float = 0.03,
    gain: float = 2.0,
    max_lead: float = 0.03,
    max_lead_step: float = 0.001,
):
    """Return a bounded, rate-limited XY command lead and its next state.

    RMPFlow can settle a few millimetres outside the requested Cartesian goal
    because of collision-repulsion terms.  Near the goal, commanding a small
    lead in the remaining-error direction closes that steady-state error.  The
    caller must still decide completion from the *actual* Cartesian position.
    """

    desired = np.asarray(desired_position, dtype=float)
    current = np.asarray(current_position, dtype=float)
    previous = (
        np.zeros(2, dtype=float)
        if previous_lead is None
        else np.asarray(previous_lead, dtype=float)
    )
    if desired.shape != (3,) or current.shape != (3,):
        raise ValueError(
            "desired_position and current_position must be 3-vectors"
        )
    if previous.shape != (2,):
        raise ValueError("previous_lead must be a 2-vector")
    if (
        not np.all(np.isfinite(desired))
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(previous))
    ):
        raise ValueError("nut motion positions must be finite")
    if (
        activation_radius <= 0.0
        or gain < 0.0
        or max_lead < 0.0
        or max_lead_step <= 0.0
    ):
        raise ValueError("lead policy limits must be non-negative")

    residual_xy = desired[:2] - current[:2]
    residual_norm = float(np.linalg.norm(residual_xy))
    desired_lead = np.zeros(2, dtype=float)
    if 0.0 < residual_norm <= activation_radius:
        desired_lead = gain * residual_xy

    lead_norm = float(np.linalg.norm(desired_lead))
    if lead_norm > max_lead and lead_norm > 0.0:
        desired_lead *= max_lead / lead_norm

    lead_delta = desired_lead - previous
    delta_norm = float(np.linalg.norm(lead_delta))
    if delta_norm > max_lead_step:
        lead_delta *= max_lead_step / delta_norm
    next_lead = previous + lead_delta

    command = desired.copy()
    command[:2] += next_lead
    return command, next_lead


def lead_z_target(
    desired_position,
    current_position,
    previous_lead: float = 0.0,
    *,
    activation_distance: float = 0.03,
    gain: float = 2.0,
    max_lead: float = 0.03,
    max_lead_step: float = 0.001,
):
    """Return a positive, bounded Z command lead and its next state.

    This is used while lifting a grasped nut vertically off its supply peg.
    Collision repulsion can leave the end effector just below the nominal
    clearance target.  The caller must verify clearance from the *actual*
    Cartesian displacement, never from this compensated command.
    """

    desired = np.asarray(desired_position, dtype=float)
    current = np.asarray(current_position, dtype=float)
    previous = float(previous_lead)
    if desired.shape != (3,) or current.shape != (3,):
        raise ValueError(
            "desired_position and current_position must be 3-vectors"
        )
    if (
        not np.all(np.isfinite(desired))
        or not np.all(np.isfinite(current))
        or not np.isfinite(previous)
    ):
        raise ValueError("nut motion positions must be finite")
    if (
        activation_distance <= 0.0
        or gain < 0.0
        or max_lead < 0.0
        or max_lead_step <= 0.0
        or previous < 0.0
    ):
        raise ValueError("lead policy limits must be non-negative")

    residual_z = float(desired[2] - current[2])
    desired_lead = 0.0
    if 0.0 < residual_z <= activation_distance:
        desired_lead = min(gain * residual_z, max_lead)

    lead_delta = float(np.clip(
        desired_lead - previous,
        -max_lead_step,
        max_lead_step,
    ))
    next_lead = float(np.clip(previous + lead_delta, 0.0, max_lead))

    command = desired.copy()
    command[2] += next_lead
    return command, next_lead

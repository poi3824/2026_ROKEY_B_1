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

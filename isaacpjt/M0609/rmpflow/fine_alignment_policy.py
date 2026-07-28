"""ROS/Isaac-free helpers for closed-loop fine-alignment motion steps."""

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class FineStepStatus:
    """Result of one executor-side settle observation."""

    stamp_ns: Optional[int]
    settle_count: int
    settled: bool


@dataclass(frozen=True)
class PlanarCommandLeadStatus:
    """Bounded XY controller lead retained around one fine-alignment step."""

    x_m: float
    y_m: float
    magnitude_m: float
    grew: bool
    saturated: bool


class BoundedPlanarCommandLead:
    """
    Add a bounded controller-only XY lead without changing the logical target.

    RMPFlow can settle just short of a submillimetre target.  While the
    measured command-direction response remains below the gate's real response
    requirement, this helper grows a small lead along the correction direction.
    It holds while the response is sufficient and resumes bounded growth if the
    response falls below the requirement before the real settle ACK.
    """

    def __init__(
        self,
        *,
        growth_step_m: float = 0.00005,
        max_lead_m: float = 0.001,
    ):
        values = (float(growth_step_m), float(max_lead_m))
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError(
                "command lead growth and maximum must be finite and positive"
            )
        if values[0] > values[1]:
            raise ValueError("command lead growth cannot exceed its maximum")
        self.growth_step_m = values[0]
        self.max_lead_m = values[1]
        self.reset()

    def reset(self):
        """Clear both retained lead and the active correction direction."""
        self._lead_x_m = 0.0
        self._lead_y_m = 0.0
        self._active_direction = None

    def begin(self, command_x_m: float, command_y_m: float):
        """Start one step, retaining only same-direction prior lead."""
        command_x_m = float(command_x_m)
        command_y_m = float(command_y_m)
        if not all(
            math.isfinite(value)
            for value in (command_x_m, command_y_m)
        ):
            raise ValueError("command lead direction must be finite")

        command_norm = math.hypot(command_x_m, command_y_m)
        if command_norm <= 1.0e-12:
            # A yaw-only step must not invent planar motion.  Keep the
            # previously settled controller offset so the camera observation
            # remains stationary, but do not grow it.
            self._active_direction = None
            return self.status()

        direction = (
            command_x_m / command_norm,
            command_y_m / command_norm,
        )
        retained_m = max(
            0.0,
            self._lead_x_m * direction[0]
            + self._lead_y_m * direction[1],
        )
        retained_m = min(retained_m, self.max_lead_m)
        self._lead_x_m = direction[0] * retained_m
        self._lead_y_m = direction[1] * retained_m
        self._active_direction = direction
        return self.status()

    def advance(
        self,
        *,
        observed_response_m: float,
        required_response_m: float,
    ):
        """Grow once when real response is short, otherwise freeze the lead."""
        observed_response_m = float(observed_response_m)
        required_response_m = float(required_response_m)
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (observed_response_m, required_response_m)
        ):
            raise ValueError(
                "command lead responses must be finite and non-negative"
            )

        previous_m = math.hypot(self._lead_x_m, self._lead_y_m)
        next_m = previous_m
        if (
            self._active_direction is not None
            and required_response_m > 0.0
            and observed_response_m < required_response_m
        ):
            next_m = min(
                self.max_lead_m,
                previous_m + self.growth_step_m,
            )
            self._lead_x_m = self._active_direction[0] * next_m
            self._lead_y_m = self._active_direction[1] * next_m

        return PlanarCommandLeadStatus(
            x_m=self._lead_x_m,
            y_m=self._lead_y_m,
            magnitude_m=next_m,
            grew=next_m > previous_m,
            saturated=next_m >= self.max_lead_m,
        )

    def finish_step(self):
        """Stop growth while retaining the settled controller offset."""
        self._active_direction = None
        return self.status()

    def status(self):
        """Return the current lead without changing it."""
        magnitude_m = math.hypot(self._lead_x_m, self._lead_y_m)
        return PlanarCommandLeadStatus(
            x_m=self._lead_x_m,
            y_m=self._lead_y_m,
            magnitude_m=magnitude_m,
            grew=False,
            saturated=magnitude_m >= self.max_lead_m,
        )

    def command_xy(self, logical_x_m: float, logical_y_m: float):
        """Return a controller XY target while leaving its logical input intact."""
        logical_x_m = float(logical_x_m)
        logical_y_m = float(logical_y_m)
        if not all(
            math.isfinite(value)
            for value in (logical_x_m, logical_y_m)
        ):
            raise ValueError("logical controller target must be finite")
        return (
            logical_x_m + self._lead_x_m,
            logical_y_m + self._lead_y_m,
        )

    def commit_xy(self, logical_x_m: float, logical_y_m: float):
        """Fold the current lead into the canonical XY without a target jump."""

        committed_xy = self.command_xy(logical_x_m, logical_y_m)
        self.reset()
        return committed_xy


class FineAlignmentStepGate:
    """Acknowledge one incremental command only after the arm has settled."""

    def __init__(
        self,
        *,
        position_tolerance_m: float = 0.005,
        orientation_tolerance_rad: float = math.radians(3.0),
        motion_tolerance_m: float = 0.0001,
        orientation_motion_tolerance_rad: float = math.radians(0.005),
        position_response_noise_margin_m: float = 0.0,
        orientation_response_noise_margin_rad: float = 0.0,
        settle_steps: int = 12,
    ):
        values = (
            position_tolerance_m,
            orientation_tolerance_rad,
            motion_tolerance_m,
            orientation_motion_tolerance_rad,
            position_response_noise_margin_m,
            orientation_response_noise_margin_rad,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("settle tolerances must be finite and non-negative")
        if not isinstance(settle_steps, int) or settle_steps <= 0:
            raise ValueError("settle_steps must be a positive integer")

        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(
            orientation_tolerance_rad
        )
        self.motion_tolerance_m = float(motion_tolerance_m)
        self.orientation_motion_tolerance_rad = float(
            orientation_motion_tolerance_rad
        )
        self.position_response_noise_margin_m = float(
            position_response_noise_margin_m
        )
        self.orientation_response_noise_margin_rad = float(
            orientation_response_noise_margin_rad
        )
        self.settle_steps = settle_steps
        self.reset()

    def reset(self):
        self.active_stamp_ns: Optional[int] = None
        self.settle_count = 0
        self.required_position_response_m = 0.0
        self.required_orientation_response_rad = 0.0

    def begin(
        self,
        stamp_ns: int,
        *,
        required_position_response_m: float = 0.0,
        required_orientation_response_rad: float = 0.0,
    ):
        stamp_ns = int(stamp_ns)
        if stamp_ns <= 0:
            raise ValueError("fine-alignment command stamp must be positive")
        if self.active_stamp_ns is not None:
            raise RuntimeError("a fine-alignment step is already active")
        responses = (
            float(required_position_response_m),
            float(required_orientation_response_rad),
        )
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in responses
        ):
            raise ValueError(
                "required step responses must be finite and non-negative"
            )

        self.active_stamp_ns = stamp_ns
        self.settle_count = 0
        self.required_position_response_m = max(
            0.0,
            responses[0] - self.position_response_noise_margin_m,
        )
        self.required_orientation_response_rad = max(
            0.0,
            responses[1] - self.orientation_response_noise_margin_rad,
        )

    def update(
        self,
        *,
        position_error_m: float,
        orientation_error_rad: float,
        motion_m: float,
        orientation_motion_rad: float,
        position_response_m: float,
        orientation_response_rad: float,
    ) -> FineStepStatus:
        values = (
            float(position_error_m),
            float(orientation_error_rad),
            float(motion_m),
            float(orientation_motion_rad),
            float(position_response_m),
            float(orientation_response_rad),
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("settle observations must be finite and non-negative")
        if self.active_stamp_ns is None:
            return FineStepStatus(None, 0, False)

        within_tolerance = (
            values[0] <= self.position_tolerance_m
            and values[1] <= self.orientation_tolerance_rad
            and values[2] <= self.motion_tolerance_m
            and values[3] <= self.orientation_motion_tolerance_rad
            and values[4] >= self.required_position_response_m
            and values[5] >= self.required_orientation_response_rad
        )
        if within_tolerance:
            self.settle_count += 1
        else:
            self.settle_count = 0

        stamp_ns = self.active_stamp_ns
        settled = self.settle_count >= self.settle_steps
        status = FineStepStatus(stamp_ns, self.settle_count, settled)
        if settled:
            self.reset()
        return status


def planar_command_response(
    *,
    start_x: float,
    start_y: float,
    current_x: float,
    current_y: float,
    command_x: float,
    command_y: float,
) -> float:
    """Return net EE displacement projected onto the XY command direction."""
    values = tuple(
        float(value)
        for value in (
            start_x,
            start_y,
            current_x,
            current_y,
            command_x,
            command_y,
        )
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("planar response inputs must be finite")

    start_x, start_y, current_x, current_y, command_x, command_y = values
    command_norm = math.hypot(command_x, command_y)
    if command_norm <= 1.0e-12:
        return 0.0
    response = (
        (current_x - start_x) * command_x
        + (current_y - start_y) * command_y
    ) / command_norm
    return max(0.0, response)


def planar_yaw_delta(
    *,
    quaternion_x: float,
    quaternion_y: float,
    quaternion_z: float,
    quaternion_w: float,
    max_abs_rad: float = math.radians(1.0),
) -> float:
    """Validate a planar delta quaternion and return its wrapped yaw."""
    values = tuple(
        float(value)
        for value in (
            quaternion_x,
            quaternion_y,
            quaternion_z,
            quaternion_w,
        )
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("fine-alignment quaternion contains NaN or inf")
    if not math.isfinite(max_abs_rad) or max_abs_rad <= 0.0:
        raise ValueError("max_abs_rad must be finite and positive")

    x, y, z, w = values
    if abs(x) > 1.0e-9 or abs(y) > 1.0e-9:
        raise ValueError("fine-alignment delta must be a planar yaw")
    norm = math.hypot(z, w)
    if norm <= 1.0e-12:
        raise ValueError("fine-alignment quaternion has zero norm")
    z /= norm
    w /= norm
    yaw = 2.0 * math.atan2(z, w)
    yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
    if abs(yaw) > max_abs_rad:
        raise ValueError("fine-alignment yaw delta exceeds the safety limit")
    return yaw

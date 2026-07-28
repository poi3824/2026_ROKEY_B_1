#!/usr/bin/env python3
"""Pure differential-drive go-to-pose control without ROS or Isaac imports."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the half-open interval [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    """Convert a yaw angle to a ROS-order quaternion."""
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def quaternion_to_yaw(
    x: float,
    y: float,
    z: float,
    w: float,
) -> float:
    """Return yaw from a finite, non-zero ROS-order quaternion."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1.0e-9:
        raise ValueError('orientation quaternion must be finite and non-zero')
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw, cos_yaw)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _rate_limit(value: float, previous: float, max_delta: float) -> float:
    return _clamp(value, previous - max_delta, previous + max_delta)


@dataclass(frozen=True)
class PoseState:
    """Measured chassis pose and motion used by the controller."""

    x: float
    y: float
    yaw: float
    z: float = 0.0
    up_z: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0
    linear_speed: float = 0.0
    angular_speed: float = 0.0


@dataclass(frozen=True)
class Goal2D:
    """Planar target pose and optional approach behavior."""

    x: float
    y: float
    yaw: float
    yaw_required: bool = True
    allow_reverse: bool = True
    align_yaw_before_drive: bool = False


@dataclass(frozen=True)
class ControllerConfig:
    """Physical and control limits for the differential-drive chassis."""

    position_tolerance: float = 0.02
    yaw_tolerance: float = 0.03
    final_approach_distance: float = 0.08
    turn_in_place_threshold: float = math.radians(5.0)
    max_linear: float = 0.35
    max_angular: float = 0.75
    min_angular: float = 0.30
    min_linear: float = 0.035
    linear_gain: float = 0.9
    heading_gain: float = 1.8
    final_yaw_gain: float = 1.8
    linear_acceleration: float = 0.35
    angular_acceleration: float = 1.2
    settle_linear_speed: float = 0.02
    settle_angular_speed: float = 0.05
    settle_steps: int = 12
    wheel_radius: float = 0.14
    track_width: float = 0.4132
    max_wheel_radps: float = 10.0
    allow_reverse: bool = True


@dataclass(frozen=True)
class DriveCommand:
    """One controller output in chassis velocity coordinates."""

    phase: str
    linear: float
    angular: float
    distance_error: float
    heading_error: float
    yaw_error: float
    travel_direction: str = 'forward'
    arrived: bool = False


class GoToPoseController:
    """Drive to a pose while latching the shortest travel direction."""

    def __init__(self, config: Optional[ControllerConfig] = None):
        """Create a controller with the approved physical defaults."""
        self.config = config or ControllerConfig()
        self._goal: Optional[Goal2D] = None
        self._last_linear = 0.0
        self._last_angular = 0.0
        self._settled_steps = 0
        self._drive_sign = 1.0
        self._direction_selected = False
        self._initial_alignment_complete = True

    @property
    def goal(self) -> Optional[Goal2D]:
        """Return the active target, if any."""
        return self._goal

    @property
    def travel_direction(self) -> str:
        """Return the direction latched for the active goal."""
        return 'reverse' if self._drive_sign < 0.0 else 'forward'

    def _select_direction(self, pose: PoseState) -> None:
        """Latch the travel direction requiring the smaller initial turn."""
        if self._goal is None or self._direction_selected:
            return

        goal = self._goal
        dx = goal.x - pose.x
        dy = goal.y - pose.y
        can_reverse = self.config.allow_reverse and goal.allow_reverse
        if (
            not can_reverse
            or math.hypot(dx, dy) <= self.config.position_tolerance
        ):
            self._drive_sign = 1.0
        else:
            bearing = math.atan2(dy, dx)
            forward_error = abs(normalize_angle(bearing - pose.yaw))
            reverse_error = abs(
                normalize_angle(bearing + math.pi - pose.yaw)
            )
            self._drive_sign = -1.0 if reverse_error < forward_error else 1.0
        self._direction_selected = True

    def set_goal(
        self,
        goal: Goal2D,
        current_pose: Optional[PoseState] = None,
    ) -> None:
        """Set a goal and optionally select its direction immediately."""
        if not all(math.isfinite(value) for value in (
            goal.x,
            goal.y,
            goal.yaw,
        )):
            raise ValueError('goal contains NaN or infinity')

        self._goal = Goal2D(
            x=float(goal.x),
            y=float(goal.y),
            yaw=normalize_angle(float(goal.yaw)),
            yaw_required=bool(goal.yaw_required),
            allow_reverse=bool(goal.allow_reverse),
            align_yaw_before_drive=bool(goal.align_yaw_before_drive),
        )
        self._last_linear = 0.0
        self._last_angular = 0.0
        self._settled_steps = 0
        self._drive_sign = 1.0
        self._direction_selected = False
        self._initial_alignment_complete = (
            not self._goal.align_yaw_before_drive
        )
        if current_pose is not None:
            self._select_direction(current_pose)

    def stop(self) -> None:
        """Reset velocity history and the arrival stability counter."""
        self._last_linear = 0.0
        self._last_angular = 0.0
        self._settled_steps = 0

    def clear_goal(self) -> None:
        """Clear the active goal and stop the controller."""
        self._goal = None
        self._drive_sign = 1.0
        self._direction_selected = False
        self._initial_alignment_complete = True
        self.stop()

    def _limited(
        self,
        linear: float,
        angular: float,
        dt: float,
    ) -> Tuple[float, float]:
        config = self.config
        linear = _rate_limit(
            linear,
            self._last_linear,
            config.linear_acceleration * max(dt, 0.0),
        )
        angular = _rate_limit(
            angular,
            self._last_angular,
            config.angular_acceleration * max(dt, 0.0),
        )
        self._last_linear = linear
        self._last_angular = angular
        return linear, angular

    def compute(self, pose: PoseState, dt: float) -> DriveCommand:
        """Compute the next chassis command for a measured pose."""
        if self._goal is None:
            return DriveCommand('idle', 0.0, 0.0, 0.0, 0.0, 0.0)
        if not math.isfinite(dt) or dt < 0.0:
            raise ValueError('dt must be finite and non-negative')
        if not all(math.isfinite(value) for value in (
            pose.x,
            pose.y,
            pose.yaw,
            pose.linear_speed,
            pose.angular_speed,
        )):
            raise ValueError('pose contains NaN or infinity')

        self._select_direction(pose)
        goal = self._goal
        dx = goal.x - pose.x
        dy = goal.y - pose.y
        distance = math.hypot(dx, dy)
        yaw_error = normalize_angle(goal.yaw - pose.yaw)

        if not self._initial_alignment_complete:
            if abs(yaw_error) > self.config.yaw_tolerance:
                raw_angular = _clamp(
                    self.config.final_yaw_gain * yaw_error,
                    -self.config.max_angular,
                    self.config.max_angular,
                )
                if abs(raw_angular) < self.config.min_angular:
                    raw_angular = math.copysign(
                        self.config.min_angular,
                        yaw_error,
                    )
                _, angular = self._limited(0.0, raw_angular, dt)
                self._last_linear = 0.0
                self._settled_steps = 0
                return DriveCommand(
                    phase='initial_align',
                    linear=0.0,
                    angular=angular,
                    distance_error=distance,
                    heading_error=0.0,
                    yaw_error=yaw_error,
                    travel_direction=self.travel_direction,
                )
            self._initial_alignment_complete = True
            self._last_angular = 0.0

        if distance > self.config.position_tolerance:
            bearing = math.atan2(dy, dx)
            if (
                goal.yaw_required
                and goal.align_yaw_before_drive
                and distance <= self.config.final_approach_distance
            ):
                path_yaw = goal.yaw
            elif self._drive_sign > 0.0:
                path_yaw = bearing
            else:
                path_yaw = normalize_angle(bearing + math.pi)
            heading_error = normalize_angle(path_yaw - pose.yaw)
            raw_angular = _clamp(
                self.config.heading_gain * heading_error,
                -self.config.max_angular,
                self.config.max_angular,
            )

            if abs(heading_error) > self.config.turn_in_place_threshold:
                phase = 'turn_to_path'
                raw_linear = 0.0
                if abs(raw_angular) < self.config.min_angular:
                    raw_angular = math.copysign(
                        self.config.min_angular,
                        heading_error,
                    )
            else:
                phase = 'drive'
                magnitude = min(
                    self.config.max_linear,
                    self.config.linear_gain * distance,
                )
                magnitude *= max(math.cos(heading_error), 0.0)
                if (
                    distance > self.config.position_tolerance * 2.0
                    and magnitude < self.config.min_linear
                ):
                    magnitude = self.config.min_linear
                raw_linear = self._drive_sign * magnitude

            linear, angular = self._limited(raw_linear, raw_angular, dt)
            if phase == 'turn_to_path':
                linear = 0.0
                self._last_linear = 0.0
            self._settled_steps = 0
            return DriveCommand(
                phase=phase,
                linear=linear,
                angular=angular,
                distance_error=distance,
                heading_error=heading_error,
                yaw_error=yaw_error,
                travel_direction=self.travel_direction,
            )

        if goal.yaw_required and abs(yaw_error) > self.config.yaw_tolerance:
            raw_angular = _clamp(
                self.config.final_yaw_gain * yaw_error,
                -self.config.max_angular,
                self.config.max_angular,
            )
            if abs(raw_angular) < self.config.min_angular:
                raw_angular = math.copysign(
                    self.config.min_angular,
                    yaw_error,
                )
            _, angular = self._limited(0.0, raw_angular, dt)
            self._last_linear = 0.0
            self._settled_steps = 0
            return DriveCommand(
                phase='final_align',
                linear=0.0,
                angular=angular,
                distance_error=distance,
                heading_error=0.0,
                yaw_error=yaw_error,
                travel_direction=self.travel_direction,
            )

        self._last_linear = 0.0
        self._last_angular = 0.0
        stationary = (
            abs(pose.linear_speed) <= self.config.settle_linear_speed
            and abs(pose.angular_speed) <= self.config.settle_angular_speed
        )
        if stationary:
            self._settled_steps += 1
        else:
            self._settled_steps = 0
        arrived = self._settled_steps >= self.config.settle_steps
        return DriveCommand(
            phase='arrived' if arrived else 'settling',
            linear=0.0,
            angular=0.0,
            distance_error=distance,
            heading_error=0.0,
            yaw_error=yaw_error,
            travel_direction=self.travel_direction,
            arrived=arrived,
        )

    def to_wheel_speeds(
        self,
        linear: float,
        angular: float,
        wheel_sign: float = 1.0,
        steer_sign: float = 1.0,
    ) -> Tuple[float, float]:
        """Convert chassis velocity to bounded left/right wheel rad/s."""
        half_track = self.config.track_width * 0.5
        left_linear = linear - steer_sign * angular * half_track
        right_linear = linear + steer_sign * angular * half_track
        left = wheel_sign * left_linear / self.config.wheel_radius
        right = wheel_sign * right_linear / self.config.wheel_radius

        peak = max(abs(left), abs(right))
        if peak > self.config.max_wheel_radps:
            scale = self.config.max_wheel_radps / peak
            left *= scale
            right *= scale
        return left, right

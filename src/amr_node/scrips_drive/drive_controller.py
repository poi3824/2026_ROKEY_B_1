#!/usr/bin/env python3
"""Isaac/ROS import가 없는 차동구동 go-to-pose 제어 수학."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple


def normalize_angle(angle: float) -> float:
    """각도를 [-pi, pi)로 정규화한다."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    """Z축 yaw를 ROS quaternion (x, y, z, w)으로 변환한다."""
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """정규화된 ROS quaternion에서 yaw를 구한다."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm < 1.0e-9:
        raise ValueError("orientation quaternion이 유효하지 않습니다")
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
    x: float
    y: float
    yaw: float
    yaw_required: bool = True
    allow_reverse: bool = True
    align_yaw_before_drive: bool = False


@dataclass(frozen=True)
class ControllerConfig:
    position_tolerance: float = 0.04
    yaw_tolerance: float = math.radians(3.0)
    # 명시적 reverse docking은 목표 바로 앞에서 최종 goal yaw를 경로 방향으로
    # 사용한다. 일반 forward goal은 위치에 먼저 도달한 뒤 제자리 yaw 정렬한다.
    final_approach_distance: float = 0.08
    # 큰 곡선을 그리며 출발하지 않도록 경로 방향을 먼저 거의 맞춘다.
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
    phase: str
    linear: float
    angular: float
    distance_error: float
    heading_error: float
    yaw_error: float
    travel_direction: str = "forward"
    arrived: bool = False


class GoToPoseController:
    """전진 기본 또는 명시적 후진 도킹 -> 최종 yaw 정렬 상태기."""

    def __init__(self, config: Optional[ControllerConfig] = None):
        self.config = config or ControllerConfig()
        self._goal: Optional[Goal2D] = None
        self._last_linear = 0.0
        self._last_angular = 0.0
        self._settled_steps = 0
        self._drive_sign = 1.0
        self._initial_alignment_complete = True

    @property
    def goal(self) -> Optional[Goal2D]:
        return self._goal

    def set_goal(self, goal: Goal2D):
        values = (goal.x, goal.y, goal.yaw)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("goal에 NaN/inf가 포함돼 있습니다")
        self._goal = Goal2D(
            goal.x,
            goal.y,
            normalize_angle(goal.yaw),
            yaw_required=bool(goal.yaw_required),
            allow_reverse=bool(goal.allow_reverse),
            align_yaw_before_drive=bool(goal.align_yaw_before_drive),
        )
        self._last_linear = 0.0
        self._last_angular = 0.0
        self._settled_steps = 0
        # 일반 목표는 항상 전진한다. 후진은 busbar 최종 docking처럼
        # allow_reverse와 align_yaw_before_drive를 모두 명시한 목표에만 쓴다.
        explicit_reverse = (
            self.config.allow_reverse
            and self._goal.allow_reverse
            and self._goal.align_yaw_before_drive
        )
        self._drive_sign = -1.0 if explicit_reverse else 1.0
        self._initial_alignment_complete = not explicit_reverse

    def stop(self):
        self._last_linear = 0.0
        self._last_angular = 0.0
        self._settled_steps = 0

    def clear_goal(self):
        self._goal = None
        self._drive_sign = 1.0
        self._initial_alignment_complete = True
        self.stop()

    def _limited(self, linear: float, angular: float, dt: float):
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
        if self._goal is None:
            return DriveCommand("idle", 0.0, 0.0, 0.0, 0.0, 0.0)

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
                    phase="initial_align",
                    linear=0.0,
                    angular=angular,
                    distance_error=distance,
                    heading_error=0.0,
                    yaw_error=yaw_error,
                    travel_direction="reverse",
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
            else:
                path_yaw = (
                    bearing
                    if self._drive_sign > 0.0
                    else normalize_angle(bearing + math.pi)
                )
            heading_error = normalize_angle(path_yaw - pose.yaw)
            raw_angular = _clamp(
                self.config.heading_gain * heading_error,
                -self.config.max_angular,
                self.config.max_angular,
            )

            if abs(heading_error) > self.config.turn_in_place_threshold:
                phase = "turn_to_path"
                raw_linear = 0.0
                # 작은 heading 오차에서 P 제어값이 chassis 정지 마찰보다
                # 작으면 5도 경계 바로 밖에서 영원히 멈춘다. 회전 전용
                # 단계에도 final_align과 같은 최소 각속도를 보장한다.
                if abs(raw_angular) < self.config.min_angular:
                    raw_angular = math.copysign(
                        self.config.min_angular,
                        heading_error,
                    )
            else:
                phase = "drive"
                linear_magnitude = min(
                    self.config.max_linear,
                    self.config.linear_gain * distance,
                )
                linear_magnitude *= max(math.cos(heading_error), 0.0)
                if (
                    distance > self.config.position_tolerance * 2.0
                    and linear_magnitude < self.config.min_linear
                ):
                    linear_magnitude = self.config.min_linear
                raw_linear = self._drive_sign * linear_magnitude

            linear, angular = self._limited(raw_linear, raw_angular, dt)
            # 회전 전용 단계에서 직진 가속도 제한을 그대로 적용하면, 직전 drive
            # 명령의 전진속도가 몇 frame 남아 경로를 벗어날 수 있다. 안전을 위해
            # 선속도만 즉시 0으로 만들고 각속도 ramp는 유지한다.
            if phase == "turn_to_path":
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
                travel_direction=(
                    "reverse" if self._drive_sign < 0.0 else "forward"
                ),
            )

        heading_error = 0.0
        if (
            goal.yaw_required
            and abs(yaw_error) > self.config.yaw_tolerance
        ):
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
            linear, angular = self._limited(0.0, raw_angular, dt)
            linear = 0.0
            self._last_linear = 0.0
            self._settled_steps = 0
            return DriveCommand(
                phase="final_align",
                linear=linear,
                angular=angular,
                distance_error=distance,
                heading_error=heading_error,
                yaw_error=yaw_error,
                travel_direction=(
                    "reverse" if self._drive_sign == -1.0 else "forward"
                ),
            )

        # pose 오차뿐 아니라 실제 root 속도도 낮은 상태가 연속 유지되어야 한다.
        self._last_linear = 0.0
        self._last_angular = 0.0
        stationary = (
            pose.linear_speed <= self.config.settle_linear_speed
            and pose.angular_speed <= self.config.settle_angular_speed
        )
        if stationary:
            self._settled_steps += 1
        else:
            self._settled_steps = 0
        arrived = self._settled_steps >= self.config.settle_steps
        return DriveCommand(
            phase="arrived" if arrived else "settling",
            linear=0.0,
            angular=0.0,
            distance_error=distance,
            heading_error=heading_error,
            yaw_error=yaw_error,
            travel_direction=(
                "reverse" if self._drive_sign == -1.0 else "forward"
            ),
            arrived=arrived,
        )

    def to_wheel_speeds(
        self,
        linear: float,
        angular: float,
        wheel_sign: float = 1.0,
        steer_sign: float = 1.0,
    ) -> Tuple[float, float]:
        """(v, w)를 좌우 wheel target(rad/s)으로 변환하고 함께 scale한다."""
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

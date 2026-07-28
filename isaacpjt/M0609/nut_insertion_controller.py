"""힘/토크 피드백 기반 너트 초기 삽입 제어기.

Isaac Sim/ROS 의존성을 두지 않아 센서 필터와 상태 전이를 단위 테스트할 수 있다.
좌표 오프셋은 너트 삽입을 시작한 TCP pose 기준 world XYZ, 회전은 시작 자세 기준
tool Z축 회전각(deg)이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Tuple


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


class InsertionState(str, Enum):
    SEEK_CONTACT = "seek_contact"
    CENTER_SEARCH = "center_search"
    THREAD_BACKOFF = "thread_backoff"
    THREAD_FORWARD = "thread_forward"
    RECOVER = "recover"
    ENGAGED = "engaged"
    FAILED = "failed"


@dataclass(frozen=True)
class RawNutSensors:
    """joint_6 child frame의 반력과 이름으로 찾은 DOF effort."""

    wrench: Tuple[float, float, float, float, float, float]
    wrist_effort: float
    gripper_effort: float
    tcp_z: float
    valid: bool = True


@dataclass(frozen=True)
class NutSensorSample:
    """영점과 저역통과 필터가 적용된 삽입 센서 값."""

    axial_force: float
    lateral_force: float
    torque: float
    gripper_force: float
    tcp_z: float
    valid: bool


class NutSensorFilter:
    """비접촉 자세의 하중을 영점으로 빼고 센서 노이즈를 완화한다."""

    def __init__(self, alpha: float = 0.2):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha는 (0, 1] 범위여야 합니다")
        self.alpha = float(alpha)
        self.reset()

    def reset(self):
        self._tare_wrench = [0.0] * 6
        self._tare_effort = 0.0
        self._tare_count = 0
        self._bias_wrench = [0.0] * 6
        self._bias_effort = 0.0
        self._filtered_wrench = [0.0] * 6
        self._filtered_effort = 0.0
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def tare_count(self) -> int:
        return self._tare_count

    def add_tare_sample(self, sample: RawNutSensors) -> bool:
        values = (*sample.wrench, sample.wrist_effort, sample.tcp_z)
        if not sample.valid or not _finite(values):
            return False
        self._tare_count += 1
        for index, value in enumerate(sample.wrench):
            self._tare_wrench[index] += float(value)
        self._tare_effort += float(sample.wrist_effort)
        return True

    def finish_tare(self) -> bool:
        if self._tare_count <= 0:
            return False
        scale = 1.0 / self._tare_count
        self._bias_wrench = [value * scale for value in self._tare_wrench]
        self._bias_effort = self._tare_effort * scale
        self._filtered_wrench = [0.0] * 6
        self._filtered_effort = 0.0
        self._ready = True
        return True

    def update(self, sample: RawNutSensors) -> NutSensorSample:
        values = (*sample.wrench, sample.wrist_effort,
                  sample.gripper_effort, sample.tcp_z)
        valid = sample.valid and self._ready and _finite(values)
        if not valid:
            return NutSensorSample(
                0.0, 0.0, 0.0, abs(float(sample.gripper_effort)),
                float(sample.tcp_z), False,
            )

        for index, value in enumerate(sample.wrench):
            unbiased = float(value) - self._bias_wrench[index]
            self._filtered_wrench[index] += self.alpha * (
                unbiased - self._filtered_wrench[index]
            )
        unbiased_effort = float(sample.wrist_effort) - self._bias_effort
        self._filtered_effort += self.alpha * (
            unbiased_effort - self._filtered_effort
        )

        fx, fy, fz, _, _, tz = self._filtered_wrench
        return NutSensorSample(
            axial_force=abs(fz),
            lateral_force=math.hypot(fx, fy),
            torque=max(abs(tz), abs(self._filtered_effort)),
            gripper_force=abs(float(sample.gripper_effort)),
            tcp_z=float(sample.tcp_z),
            valid=True,
        )


@dataclass(frozen=True)
class InsertionConfig:
    contact_force_n: float = 6.0
    contact_hold_steps: int = 5
    target_axial_force_n: float = 10.0
    max_axial_force_n: float = 55.0
    max_lateral_force_n: float = 20.0
    max_seek_depth_m: float = 0.003
    seek_speed_m_s: float = 0.0025
    force_compliance_m_per_n_s: float = 0.00003
    max_force_step_m: float = 0.00008
    spiral_radius_m: float = 0.0015
    spiral_turns: float = 2.0
    spiral_duration_s: float = 3.0
    center_min_time_s: float = 0.6
    centered_lateral_force_n: float = 5.0
    backoff_deg: float = 40.0
    thread_find_deg: float = 120.0
    thread_find_speed_deg_s: float = 40.0
    min_engagement_rotation_deg: float = 20.0
    engagement_drop_m: float = 0.00045
    min_thread_force_n: float = 2.0
    cross_thread_torque_nm: float = 25.0
    cross_thread_hold_steps: int = 8
    recovery_lift_m: float = 0.0015
    recovery_duration_s: float = 1.0
    max_attempts: int = 3
    invalid_sensor_hold_steps: int = 10


@dataclass(frozen=True)
class InsertionCommand:
    state: InsertionState
    x_offset: float
    y_offset: float
    z_offset: float
    rotation_deg: float
    engaged: bool = False
    failed: bool = False
    message: str = ""


class ForceGuidedNutInsertion:
    """초기 접촉부터 나사산 물림까지의 힘 기반 상태기."""

    def __init__(self, config: Optional[InsertionConfig] = None):
        self.config = config or InsertionConfig()
        self.reset(0.0)

    def reset(self, tcp_z: float):
        if not math.isfinite(tcp_z):
            raise ValueError("tcp_z가 유효하지 않습니다")
        self.state = InsertionState.SEEK_CONTACT
        self.origin_z = float(tcp_z)
        self.z_offset = 0.0
        self.rotation_deg = 0.0
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.elapsed = 0.0
        self.contact_steps = 0
        self.cross_steps = 0
        self.attempt = 1
        self.best_score = float("inf")
        self.best_xy = (0.0, 0.0)
        self.thread_start_z = float(tcp_z)
        self.invalid_steps = 0

    def _command(self, message: str = "") -> InsertionCommand:
        return InsertionCommand(
            state=self.state,
            x_offset=self.x_offset,
            y_offset=self.y_offset,
            z_offset=self.z_offset,
            rotation_deg=self.rotation_deg,
            engaged=self.state == InsertionState.ENGAGED,
            failed=self.state == InsertionState.FAILED,
            message=message,
        )

    def _change_state(self, state: InsertionState, sample: NutSensorSample):
        self.state = state
        self.elapsed = 0.0
        self.contact_steps = 0
        self.cross_steps = 0
        if state == InsertionState.CENTER_SEARCH:
            self.best_score = float("inf")
            self.best_xy = (self.x_offset, self.y_offset)
        elif state == InsertionState.THREAD_FORWARD:
            self.thread_start_z = sample.tcp_z

    def _force_control(self, sample: NutSensorSample, dt: float):
        if not sample.valid:
            return
        error = self.config.target_axial_force_n - sample.axial_force
        delta = self.config.force_compliance_m_per_n_s * error * dt
        # 힘이 부족하면 내려가야 하므로 world Z 오프셋의 부호는 반대다.
        delta = _clamp(
            delta,
            -self.config.max_force_step_m,
            self.config.max_force_step_m,
        )
        self.z_offset -= delta
        self.z_offset = _clamp(
            self.z_offset,
            -self.config.max_seek_depth_m,
            self.config.recovery_lift_m,
        )

    def update(self, sample: NutSensorSample, dt: float) -> InsertionCommand:
        if dt <= 0.0 or not math.isfinite(dt):
            raise ValueError("dt는 유한한 양수여야 합니다")
        if self.state in (InsertionState.ENGAGED, InsertionState.FAILED):
            return self._command()
        if not sample.valid:
            self.invalid_steps += 1
            if self.invalid_steps >= self.config.invalid_sensor_hold_steps:
                self.state = InsertionState.FAILED
                return self._command("joint_6 6축 반력 센서를 연속으로 읽을 수 없습니다")
            return self._command("센서 일시 누락, 현재 목표 유지")
        self.invalid_steps = 0

        self.elapsed += dt
        cfg = self.config

        if self.state != InsertionState.RECOVER and (
            sample.axial_force >= cfg.max_axial_force_n
            or sample.lateral_force >= cfg.max_lateral_force_n * 1.5
        ):
            self._change_state(InsertionState.RECOVER, sample)
            return self._command("삽입 안전 힘 한계 초과, 복구 시작")

        if self.state == InsertionState.SEEK_CONTACT:
            self.z_offset = max(
                self.z_offset - cfg.seek_speed_m_s * dt,
                -cfg.max_seek_depth_m,
            )
            if sample.axial_force >= cfg.contact_force_n:
                self.contact_steps += 1
            else:
                self.contact_steps = max(0, self.contact_steps - 1)
            if self.contact_steps >= cfg.contact_hold_steps:
                self._change_state(InsertionState.CENTER_SEARCH, sample)
                return self._command("축력 접촉 감지, 나선 중심 탐색 시작")
            if self.z_offset <= -cfg.max_seek_depth_m:
                self.state = InsertionState.FAILED
                return self._command("최대 탐색 깊이까지 접촉을 감지하지 못했습니다")

        elif self.state == InsertionState.CENTER_SEARCH:
            self._force_control(sample, dt)
            ratio = min(self.elapsed / cfg.spiral_duration_s, 1.0)
            radius = cfg.spiral_radius_m * ratio
            angle = 2.0 * math.pi * cfg.spiral_turns * ratio
            self.x_offset = radius * math.cos(angle)
            self.y_offset = radius * math.sin(angle)
            score = sample.lateral_force + 0.2 * abs(
                sample.axial_force - cfg.target_axial_force_n
            )
            if score < self.best_score:
                self.best_score = score
                self.best_xy = (self.x_offset, self.y_offset)
            centered = (
                self.elapsed >= cfg.center_min_time_s
                and sample.lateral_force <= cfg.centered_lateral_force_n
                and sample.axial_force >= cfg.min_thread_force_n
            )
            if centered or ratio >= 1.0:
                self.x_offset, self.y_offset = self.best_xy
                self._change_state(InsertionState.THREAD_BACKOFF, sample)
                return self._command("중심 탐색 완료, 역회전으로 나사산 시작점 탐색")

        elif self.state == InsertionState.THREAD_BACKOFF:
            self._force_control(sample, dt)
            self.rotation_deg = max(
                -cfg.backoff_deg,
                -cfg.thread_find_speed_deg_s * self.elapsed,
            )
            if self.rotation_deg <= -cfg.backoff_deg:
                self._change_state(InsertionState.THREAD_FORWARD, sample)
                return self._command("역회전 완료, 저속 정회전으로 나사산 물림 확인")

        elif self.state == InsertionState.THREAD_FORWARD:
            self._force_control(sample, dt)
            forward_deg = min(
                cfg.thread_find_speed_deg_s * self.elapsed,
                cfg.backoff_deg + cfg.thread_find_deg,
            )
            self.rotation_deg = -cfg.backoff_deg + forward_deg
            advance = self.thread_start_z - sample.tcp_z
            rotated = max(0.0, self.rotation_deg + cfg.backoff_deg)
            engaged = (
                rotated >= cfg.min_engagement_rotation_deg
                and advance >= cfg.engagement_drop_m
                and sample.axial_force >= cfg.min_thread_force_n
                and sample.torque < cfg.cross_thread_torque_nm
                and sample.lateral_force < cfg.max_lateral_force_n
            )
            if engaged:
                self.state = InsertionState.ENGAGED
                return self._command("Z 진행과 저토크 회전 확인, 나사산 정상 물림")

            shallow = advance < cfg.engagement_drop_m * 0.5
            if sample.torque >= cfg.cross_thread_torque_nm and shallow:
                self.cross_steps += 1
            else:
                self.cross_steps = max(0, self.cross_steps - 1)
            if (
                self.cross_steps >= cfg.cross_thread_hold_steps
                or forward_deg >= cfg.backoff_deg + cfg.thread_find_deg
            ):
                self._change_state(InsertionState.RECOVER, sample)
                return self._command("초기 토크 상승 또는 Z 진행 없음, 사선 물림 복구")

        elif self.state == InsertionState.RECOVER:
            ratio = min(self.elapsed / cfg.recovery_duration_s, 1.0)
            self.rotation_deg *= 1.0 - ratio
            self.z_offset = min(
                self.z_offset + cfg.recovery_lift_m * dt
                / cfg.recovery_duration_s,
                cfg.recovery_lift_m,
            )
            self.x_offset *= 1.0 - ratio
            self.y_offset *= 1.0 - ratio
            if ratio >= 1.0:
                if self.attempt >= cfg.max_attempts:
                    self.state = InsertionState.FAILED
                    return self._command("사선 물림 복구 횟수를 초과했습니다")
                self.attempt += 1
                self.origin_z = sample.tcp_z
                self.z_offset = 0.0
                self.rotation_deg = 0.0
                self.x_offset = 0.0
                self.y_offset = 0.0
                self._change_state(InsertionState.SEEK_CONTACT, sample)
                return self._command(f"복구 완료, 삽입 재시도 {self.attempt}")

        return self._command()


@dataclass(frozen=True)
class ScrewEvaluation:
    cross_thread: bool
    seated: bool


def evaluate_screw_state(
    *,
    planned_depth_m: float,
    actual_advance_m: float,
    torque_nm: float,
    lateral_force_n: float,
    z_movement_m: float,
    engage_length_m: float,
    seated_torque_nm: float,
    cross_thread_torque_nm: float,
    max_lateral_force_n: float,
    stall_z_movement_m: float,
) -> ScrewEvaluation:
    """사선 물림과 완착을 서로 배타적인 조건으로 판정한다."""

    shallow_limit = engage_length_m * 0.35
    shallow = planned_depth_m < shallow_limit
    expected_advance = max(planned_depth_m * 0.35, 0.0002)
    cross_thread = (
        shallow
        and torque_nm >= cross_thread_torque_nm
        and actual_advance_m < expected_advance
    )
    seated = (
        not cross_thread
        and planned_depth_m >= engage_length_m * 0.65
        and torque_nm >= seated_torque_nm
        and z_movement_m <= stall_z_movement_m
        and lateral_force_n <= max_lateral_force_n
    )
    return ScrewEvaluation(cross_thread=cross_thread, seated=seated)

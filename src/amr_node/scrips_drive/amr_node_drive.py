#!/usr/bin/env python3
"""Perception world 좌표만 Isaac PhysX AMR 목표로 전달하는 fail-closed relay.

좌표가 포함된 상위 goal은 받지 않는다. `/amr_physics/vision_request`에는
`target_kind`와 `request_id`만 들어오며, 실제 좌표는 namespaced
`Detection3DArray`에서 연속 검출·신뢰도·분산·신선도 검사를 통과한 값만 쓴다.

SUB /amr_physics/vision_request  std_msgs/String (JSON)
PUB /amr_physics/status          fms_interfaces/AmrStatus
PUB /amr_physics/goal_ack        std_msgs/String (JSON)
SUB /amr_physics/cancel_request  std_msgs/String (JSON)

SUB <busbar_detections_topic>    vision_msgs/Detection3DArray
SUB <battery_detections_topic>   vision_msgs/Detection3DArray

Isaac 저수준 API:
SUB /amr_physics/ready, /amr_physics/drive_state
PUB /amr_physics/command, /amr_physics/lease
"""

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray

from fms_interfaces.msg import AmrStatus

from vision_goal import (
    DetectionCandidate,
    DetectionWindow,
    VisionUnavailable,
    WorldGoal,
    compute_directional_standoff_goal,
    compute_standoff_goal,
    select_highest_confidence,
    select_nearest_valid_pair,
    select_nearest_to_reference,
)


TOPIC_PREFIX = "/amr_physics"
BRIDGE_STATE_TIMEOUT_SEC = 6.0
SUPPORTED_TARGETS = ("battery", "busbar")
READY_TOPIC = f"{TOPIC_PREFIX}/ready"
WORLD_POSE_TOPIC = f"{TOPIC_PREFIX}/world_pose"


@dataclass
class RelayGoal:
    goal_id: str
    request_id: str
    target_kind: str
    x: float
    y: float
    yaw: float
    vision_target_x: float
    vision_target_y: float
    vision_source_stamp_ns: int
    phase: str = "final"
    yaw_required: bool = True
    allow_reverse: bool = False
    align_yaw_before_drive: bool = False
    final_allow_reverse: bool = False
    final_x: Optional[float] = None
    final_y: Optional[float] = None
    final_yaw: Optional[float] = None
    frame_id: str = "world"
    command_sent: bool = False
    cancel_requested: bool = False
    created_wall: float = field(default_factory=time.monotonic)


@dataclass
class PendingVisionRequest:
    goal_id: str
    target_kind: str
    deadline_wall: Optional[float]


class AmrDriveNode(Node):
    def __init__(self):
        super().__init__("amr_physics_vision_relay")

        self.declare_parameter("vision_standoff_m", float("nan"))
        self.declare_parameter("battery_standoff_m", float("nan"))
        self.declare_parameter("busbar_preapproach_extra_m", 0.80)
        self.declare_parameter("busbar_scan_yaw_offset_deg", 180.0)
        self.declare_parameter("debug_no_timeouts", False)
        self.declare_parameter("vision_min_confidence", 0.6)
        self.declare_parameter("vision_min_samples", 3)
        self.declare_parameter("vision_window_size", 5)
        self.declare_parameter("vision_max_std_m", 0.01)
        self.declare_parameter("vision_max_age_sec", 1.0)
        self.declare_parameter("active_vision_timeout_sec", 6.0)
        self.declare_parameter("vision_max_target_drift_m", float("nan"))
        self.declare_parameter("vision_enforce_active_drift", False)
        self.declare_parameter("vision_request_timeout_sec", 8.0)
        self.declare_parameter("world_pose_max_age_sec", 1.0)
        self.declare_parameter("battery_bolt_pair_spacing_m", 0.418)
        self.declare_parameter(
            "battery_bolt_pair_spacing_tolerance_m",
            0.080,
        )
        self.declare_parameter(
            "busbar_detections_topic",
            "/busbar_cam/perception/detections_3d",
        )
        self.declare_parameter(
            "battery_detections_topic",
            "/bolt_cam/perception/detections_3d",
        )

        self._standoff_m = float(
            self.get_parameter("vision_standoff_m").value
        )
        self._battery_standoff_m = float(
            self.get_parameter("battery_standoff_m").value
        )
        self._busbar_preapproach_extra_m = float(
            self.get_parameter("busbar_preapproach_extra_m").value
        )
        scan_yaw_offset = math.radians(
            float(self.get_parameter("busbar_scan_yaw_offset_deg").value)
        )
        self._busbar_scan_yaw_offset_rad = (
            (scan_yaw_offset + math.pi) % (2.0 * math.pi) - math.pi
        )
        self._debug_no_timeouts = bool(
            self.get_parameter("debug_no_timeouts").value
        )
        min_confidence = float(
            self.get_parameter("vision_min_confidence").value
        )
        min_samples = int(self.get_parameter("vision_min_samples").value)
        window_size = int(self.get_parameter("vision_window_size").value)
        max_std_m = float(self.get_parameter("vision_max_std_m").value)
        max_age_sec = float(
            self.get_parameter("vision_max_age_sec").value
        )
        self._active_vision_timeout_sec = float(
            self.get_parameter("active_vision_timeout_sec").value
        )
        self._max_target_drift_m = float(
            self.get_parameter("vision_max_target_drift_m").value
        )
        self._enforce_active_drift = bool(
            self.get_parameter("vision_enforce_active_drift").value
        )
        self._request_timeout_sec = float(
            self.get_parameter("vision_request_timeout_sec").value
        )
        self._world_pose_max_age_sec = float(
            self.get_parameter("world_pose_max_age_sec").value
        )
        self._battery_pair_spacing_m = float(
            self.get_parameter("battery_bolt_pair_spacing_m").value
        )
        self._battery_pair_spacing_tolerance_m = float(
            self.get_parameter(
                "battery_bolt_pair_spacing_tolerance_m"
            ).value
        )
        self._busbar_topic = str(
            self.get_parameter("busbar_detections_topic").value
        )
        self._battery_topic = str(
            self.get_parameter("battery_detections_topic").value
        )

        if not math.isfinite(self._standoff_m) or self._standoff_m <= 0.0:
            raise ValueError(
                "vision_standoff_m을 busbar 로봇-작업물 캘리브레이션 값으로 "
                "명시해야 합니다. 기본값이나 하드코딩 폴백은 없습니다."
            )
        if (
            not math.isfinite(self._battery_standoff_m)
            or self._battery_standoff_m <= 0.0
        ):
            raise ValueError(
                "battery_standoff_m을 battery 스캔 캘리브레이션 값으로 "
                "명시해야 합니다. busbar stand-off를 재사용하지 않습니다."
            )
        if (
            not math.isfinite(self._busbar_preapproach_extra_m)
            or self._busbar_preapproach_extra_m < 0.0
        ):
            raise ValueError(
                "busbar_preapproach_extra_m은 유한한 0 이상의 거리여야 합니다"
            )
        if not math.isfinite(self._busbar_scan_yaw_offset_rad):
            raise ValueError(
                "busbar_scan_yaw_offset_deg는 유한한 값이어야 합니다"
            )
        if (
            not math.isfinite(self._max_target_drift_m)
            or self._max_target_drift_m <= 0.0
        ):
            raise ValueError(
                "vision_max_target_drift_m을 실제 비전 오차/작업 허용치로 "
                "명시해야 합니다. 임의 기본값은 없습니다."
            )
        positive_values = (
            max_std_m,
            max_age_sec,
            self._active_vision_timeout_sec,
            self._request_timeout_sec,
            self._world_pose_max_age_sec,
            self._battery_pair_spacing_m,
            self._battery_pair_spacing_tolerance_m,
        )
        if not all(math.isfinite(value) and value > 0.0
                   for value in positive_values):
            raise ValueError("vision timeout/std/age 파라미터는 양수여야 합니다")

        gate_options = {
            "min_score": min_confidence,
            "min_samples": min_samples,
            "max_samples": window_size,
            "max_age_sec": max_age_sec,
            "max_std_m": max_std_m,
        }
        self._vision_gates = {
            "busbar": DetectionWindow(
                expected_candidates=1,
                **gate_options,
            ),
            # battery 자체 라벨이 없으므로 배터리 위 bolt 두 개의 순수 중점만 쓴다.
            "battery": DetectionWindow(
                expected_candidates=2,
                **gate_options,
            ),
        }

        self._vision_request_sub = self.create_subscription(
            String,
            f"{TOPIC_PREFIX}/vision_request",
            self._on_vision_request,
            10,
        )
        self._status_pub = self.create_publisher(
            AmrStatus, f"{TOPIC_PREFIX}/status", 10
        )
        self._goal_ack_pub = self.create_publisher(
            String, f"{TOPIC_PREFIX}/goal_ack", 10
        )
        selected_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._selected_battery_pub = self.create_publisher(
            PoseStamped,
            f"{TOPIC_PREFIX}/selected_battery_midpoint",
            selected_qos,
        )
        self._selected_busbar_pub = self.create_publisher(
            PoseStamped,
            f"{TOPIC_PREFIX}/selected_busbar_pose",
            selected_qos,
        )
        self._cancel_request_sub = self.create_subscription(
            String,
            f"{TOPIC_PREFIX}/cancel_request",
            self._on_cancel_request,
            10,
        )
        self._command_pub = self.create_publisher(
            String, f"{TOPIC_PREFIX}/command", 10
        )
        self._lease_pub = self.create_publisher(
            String, f"{TOPIC_PREFIX}/lease", 10
        )

        self._busbar_detection_sub = self.create_subscription(
            Detection3DArray,
            self._busbar_topic,
            self._on_busbar_detections,
            10,
        )
        self._battery_detection_sub = self.create_subscription(
            Detection3DArray,
            self._battery_topic,
            self._on_battery_detections,
            10,
        )

        ready_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._ready_sub = self.create_subscription(
            String,
            READY_TOPIC,
            self._on_ready,
            ready_qos,
        )
        self._drive_state_sub = self.create_subscription(
            String,
            f"{TOPIC_PREFIX}/drive_state",
            self._on_drive_state,
            10,
        )
        self._world_pose_sub = self.create_subscription(
            PoseStamped,
            WORLD_POSE_TOPIC,
            self._on_world_pose,
            10,
        )

        self._session_id = ""
        self._busbar_approach_yaw: Optional[float] = None
        self._active: Optional[RelayGoal] = None
        self._pending: Optional[PendingVisionRequest] = None
        self._pending_reason = "비전 표본 대기 중"
        self._last_bridge_state_wall = 0.0
        self._current_world_pose: Optional[PoseStamped] = None
        self._last_world_pose_wall = 0.0
        self._watchdog = self.create_timer(0.25, self._on_watchdog)
        self._lease_timer = self.create_timer(0.8, self._publish_lease)

        self.get_logger().info(
            "AMR 비전 전용 relay 준비: "
            f"busbar_standoff={self._standoff_m:.3f}m, "
            f"battery_standoff={self._battery_standoff_m:.3f}m, "
            f"busbar_preapproach_extra="
            f"{self._busbar_preapproach_extra_m:.3f}m, "
            f"busbar_scan_yaw_offset="
            f"{math.degrees(self._busbar_scan_yaw_offset_rad):+.1f}deg, "
            f"active_drift_guard="
            f"{'on' if self._enforce_active_drift else 'off'}"
            f"({self._max_target_drift_m:.3f}m), "
            f"active_vision_timeout="
            f"{self._active_vision_timeout_sec:.1f}s, "
            f"busbar={self._busbar_topic}, "
            f"battery(bolt pair)={self._battery_topic}; "
            "Isaac READY와 안정된 비전 좌표 대기"
        )
        if self._debug_no_timeouts:
            self.get_logger().warn(
                "DEBUG 무제한 모드: 요청/pose/perception/bridge-state "
                "시간 제한을 적용하지 않습니다. source 수와 비전 품질 검사는 "
                "유지되며 active drift 취소는 별도 파라미터를 따릅니다."
            )

    @property
    def has_active_goal(self) -> bool:
        return self._active is not None or self._pending is not None

    @staticmethod
    def _decode_json(msg: String):
        payload = json.loads(msg.data)
        if not isinstance(payload, dict):
            raise ValueError("JSON object가 필요합니다")
        return payload

    @staticmethod
    def _exact_frame(frame_id: str) -> str:
        return str(frame_id)

    @staticmethod
    def _stamp_ns(stamp) -> int:
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)
        if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
            raise ValueError(
                f"유효하지 않은 source stamp: {sec}.{nanosec:09d}"
            )
        return sec * 1_000_000_000 + nanosec

    def _on_ready(self, msg: String):
        try:
            payload = self._decode_json(msg)
            if str(payload["state"]).upper() != "READY":
                raise ValueError("state가 READY가 아닙니다")
            session_id = str(payload["session_id"]).strip()
            if not session_id:
                raise ValueError("session_id가 비어 있습니다")
            workcell = payload.get("workcell")
            busbar_approach_yaw = None
            if workcell is not None:
                busbar_approach_yaw = float(
                    workcell["busbar_approach_yaw"]
                )
                if not math.isfinite(busbar_approach_yaw):
                    raise ValueError(
                        "workcell busbar_approach_yaw가 유한하지 않습니다"
                    )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().warn(f"잘못된 READY 무시: {exc}")
            return

        if session_id == self._session_id:
            return

        old_session = self._session_id
        active = self._active
        if active is not None and active.command_sent and old_session:
            self._publish_command(
                "CANCEL",
                active,
                session_id=old_session,
            )

        self._session_id = session_id
        self._busbar_approach_yaw = busbar_approach_yaw
        self._current_world_pose = None
        self._last_world_pose_wall = 0.0
        for gate in self._vision_gates.values():
            gate.reset(
                "Isaac session 전환 후 새 source frame 표본이 필요합니다"
            )
        approach = (
            "현재 AMR-검출점 직선"
            if busbar_approach_yaw is None
            else f"{math.degrees(busbar_approach_yaw):+.1f}deg 작업 셀 normal"
        )
        self.get_logger().info(
            f"Isaac bridge READY session={session_id}; "
            f"busbar 접근={approach}"
        )

        if active is not None:
            self._active = None
            self._publish_status(
                AmrStatus.STATE_ERROR,
                active.target_kind,
                "Isaac bridge가 주행 중 재시작되어 목표를 중단했습니다",
            )
        if self._pending is not None:
            self._pending_reason = (
                "Isaac READY 이후의 새 비전/world pose 표본 대기 중"
            )

    def _on_world_pose(self, msg: PoseStamped):
        frame_id = self._exact_frame(msg.header.frame_id)
        position = msg.pose.position
        if frame_id != "world":
            self.get_logger().warn(
                f"world_pose frame이 world가 아니어서 무시: {msg.header.frame_id!r}",
                throttle_duration_sec=3.0,
            )
            if self._active is not None:
                self.request_cancel("world_pose frame 오류")
            return
        if not all(
            math.isfinite(value)
            for value in (position.x, position.y, position.z)
        ):
            self.get_logger().warn(
                "world_pose에 NaN/inf가 있어 무시",
                throttle_duration_sec=3.0,
            )
            if self._active is not None:
                self.request_cancel("world_pose NaN/inf")
            return
        self._current_world_pose = msg
        self._last_world_pose_wall = time.monotonic()
        self._try_resolve_pending()

    def _on_busbar_detections(self, msg: Detection3DArray):
        self._on_detections(
            msg,
            label="busbar",
            target_kind="busbar",
        )

    def _on_battery_detections(self, msg: Detection3DArray):
        self._on_detections(
            msg,
            label="bolt",
            target_kind="battery",
        )

    def _on_detections(
        self,
        msg: Detection3DArray,
        *,
        label: str,
        target_kind: str,
    ):
        gate = self._vision_gates[target_kind]
        if self._exact_frame(msg.header.frame_id) != "world":
            reason = (
                f"{target_kind} detection frame이 world가 아닙니다: "
                f"{msg.header.frame_id!r}"
            )
            self._invalidate_vision(target_kind, reason)
            return

        try:
            source_stamp_ns = self._stamp_ns(msg.header.stamp)
            candidates = self._extract_candidates(
                msg,
                label,
                source_stamp_ns,
            )
            if target_kind == "battery":
                if len(candidates) < 2:
                    reason = (
                        f"볼트 후보 수 {len(candidates)}개 "
                        "(최소 2개 필요); 유효 볼트쌍 표본 누적 유지"
                    )
                    gate.note_transient_rejection(reason)
                    self._pending_reason = reason
                    self.get_logger().warn(
                        reason,
                        throttle_duration_sec=3.0,
                    )
                    return
                reference_xy = self._battery_pair_reference_xy()
                candidates = select_nearest_valid_pair(
                    candidates,
                    reference_xy,
                    expected_spacing_m=self._battery_pair_spacing_m,
                    spacing_tolerance_m=(
                        self._battery_pair_spacing_tolerance_m
                    ),
                )
                midpoint_x = 0.5 * (
                    candidates[0].x + candidates[1].x
                )
                midpoint_y = 0.5 * (
                    candidates[0].y + candidates[1].y
                )
                self.get_logger().info(
                    "battery 볼트 후보에서 AMR 최근접 유효 쌍 선택: "
                    f"mid=({midpoint_x:.4f}, {midpoint_y:.4f}), "
                    f"reference=({reference_xy[0]:.4f}, "
                    f"{reference_xy[1]:.4f})",
                    throttle_duration_sec=3.0,
                )
            if target_kind == "busbar" and len(candidates) > 1:
                original_candidates = candidates
                active = self._active
                if (
                    active is not None
                    and active.target_kind == target_kind
                ):
                    candidates = select_nearest_to_reference(
                        candidates,
                        (
                            active.vision_target_x,
                            active.vision_target_y,
                        ),
                    )
                    selection_reason = (
                        "주행 시작 때 선택한 대상과 가장 가까운 "
                        "후보를 유지"
                    )
                else:
                    candidates = select_highest_confidence(candidates)
                    selection_reason = "최고 confidence 1개 선택"
                ranked_scores = sorted(
                    (
                        candidate.score
                        for candidate in original_candidates
                    ),
                    reverse=True,
                )
                self.get_logger().warn(
                    f"busbar 후보 {len(original_candidates)}개 중 "
                    f"{selection_reason} "
                    f"(최고 {ranked_scores[0]:.3f}, "
                    f"차순위 {ranked_scores[1]:.3f}); "
                    "연속 표본 위치 안정성 검사는 계속 적용됩니다",
                    throttle_duration_sec=3.0,
                )
        except (AttributeError, TypeError, ValueError) as exc:
            reason = f"{target_kind} detection 메시지가 잘못됐습니다: {exc}"
            self._invalidate_vision(target_kind, reason)
            return

        accepted = gate.add_frame(
            candidates,
            time.monotonic(),
            source_stamp_ns,
        )
        if not accepted:
            self._pending_reason = gate.last_rejection
            if (
                self._active is not None
                and self._active.target_kind == target_kind
                and self._enforce_active_drift
            ):
                self.request_cancel(
                    f"active 비전 표본 무효: {gate.last_rejection}"
                )
                return
            if (
                self._pending is not None
                and self._pending.target_kind == target_kind
                and len(candidates) > gate.expected_candidates
            ):
                self._fail_pending_for(
                    target_kind,
                    f"복수 대상이 보여 식별할 수 없습니다: "
                    f"{gate.last_rejection}",
                )
            return
        self._try_resolve_pending()

    def _battery_pair_reference_xy(self):
        active = self._active
        if active is not None and active.target_kind == "battery":
            return (
                active.vision_target_x,
                active.vision_target_y,
            )
        if self._current_world_pose is None:
            raise VisionUnavailable(
                "AMR world pose가 없어 최근접 볼트쌍을 선택할 수 없습니다"
            )
        position = self._current_world_pose.pose.position
        return (float(position.x), float(position.y))

    def _extract_candidates(
        self,
        msg: Detection3DArray,
        label: str,
        array_stamp_ns: int,
    ):
        candidates = []
        for detection in msg.detections:
            detection_frame = self._exact_frame(
                detection.header.frame_id
            )
            if detection_frame != "world":
                raise ValueError(
                    f"detection frame={detection.header.frame_id!r}"
                )
            detection_stamp_ns = self._stamp_ns(detection.header.stamp)
            if detection_stamp_ns != array_stamp_ns:
                raise ValueError(
                    "array와 detection source timestamp가 다릅니다"
                )
            matching = [
                result
                for result in detection.results
                if str(result.hypothesis.class_id).strip().lower() == label
            ]
            if not matching:
                continue
            result = max(
                matching,
                key=lambda item: float(item.hypothesis.score),
            )
            position = result.pose.pose.position
            candidates.append(
                DetectionCandidate(
                    x=float(position.x),
                    y=float(position.y),
                    z=float(position.z),
                    score=float(result.hypothesis.score),
                )
            )
        return candidates

    def _on_vision_request(self, msg: String):
        target_kind = ""
        try:
            payload = self._decode_json(msg)
            if int(payload.get("version", 1)) != 1:
                raise ValueError("지원하지 않는 version입니다")
            goal_id = str(payload["request_id"]).strip()
            target_kind = str(payload["target_kind"]).strip().lower()
            if not goal_id or len(goal_id) > 128:
                raise ValueError("request_id는 1~128자여야 합니다")
            if target_kind not in SUPPORTED_TARGETS:
                raise ValueError(
                    f"target_kind는 {SUPPORTED_TARGETS} 중 하나여야 합니다"
                )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self._publish_status(
                AmrStatus.STATE_ERROR,
                target_kind,
                f"비전 요청 거부: {exc}",
            )
            return

        if (
            self._active is not None
            and self._active.request_id == goal_id
            and self._active.target_kind == target_kind
        ):
            self._publish_goal_ack(self._active)
            return
        if (
            self._pending is not None
            and self._pending.goal_id == goal_id
            and self._pending.target_kind == target_kind
        ):
            self._publish_goal_ack_fields(target_kind, goal_id)
            return
        if self._active is not None or self._pending is not None:
            self._publish_status(
                AmrStatus.STATE_ERROR,
                target_kind,
                "다른 비전 주행 요청이 진행 중이어서 새 요청을 거부했습니다",
            )
            return

        self._pending = PendingVisionRequest(
            goal_id=goal_id,
            target_kind=target_kind,
            deadline_wall=(
                None
                if self._debug_no_timeouts
                else time.monotonic() + self._request_timeout_sec
            ),
        )
        self._pending_reason = "안정된 비전 좌표 대기 중"
        self._publish_goal_ack_fields(target_kind, goal_id)
        self._publish_status(
            AmrStatus.STATE_MOVING,
            target_kind,
            "하드코딩 없이 안정된 perception world 좌표를 기다리는 중",
        )
        self._try_resolve_pending()

    def _target_topic(self, target_kind: str) -> str:
        return (
            self._busbar_topic
            if target_kind == "busbar"
            else self._battery_topic
        )

    def _try_resolve_pending(self):
        pending = self._pending
        if pending is None or self._active is not None:
            return

        if not self._session_id:
            self._pending_reason = "Isaac bridge READY가 아직 없습니다"
            return

        for source_topic in (READY_TOPIC, WORLD_POSE_TOPIC):
            source_count = self.count_publishers(source_topic)
            if source_count > 1:
                self._fail_pending_for(
                    pending.target_kind,
                    f"{source_topic} publisher가 {source_count}개라 "
                    "Isaac source가 모호합니다",
                )
                return
            if source_count < 1:
                self._pending_reason = (
                    f"{source_topic} publisher가 없습니다"
                )
                return

        topic = self._target_topic(pending.target_kind)
        publisher_count = self.count_publishers(topic)
        if publisher_count > 1:
            self._fail_pending_for(
                pending.target_kind,
                f"{topic} publisher가 {publisher_count}개라 source가 모호합니다",
            )
            return
        if publisher_count < 1:
            self._pending_reason = f"{topic} publisher가 없습니다"
            return

        now = time.monotonic()
        if self._current_world_pose is None:
            self._pending_reason = "Isaac world_pose가 아직 없습니다"
            return
        pose_age = now - self._last_world_pose_wall
        if (
            not self._debug_no_timeouts
            and pose_age > self._world_pose_max_age_sec
        ):
            self._pending_reason = (
                f"Isaac world_pose가 {pose_age:.2f}s 전 데이터로 오래됐습니다"
            )
            return

        gate = self._vision_gates[pending.target_kind]
        try:
            snapshot = gate.snapshot(
                now,
                enforce_age=not self._debug_no_timeouts,
            )
            current = self._current_world_pose.pose.position
            use_directional_busbar = (
                pending.target_kind == "busbar"
                and self._busbar_approach_yaw is not None
            )
            if use_directional_busbar:
                final_goal = compute_directional_standoff_goal(
                    target_xy=(snapshot.x, snapshot.y),
                    approach_yaw_rad=self._busbar_approach_yaw,
                    standoff_m=self._standoff_m,
                    docking_yaw_offset_rad=(
                        self._busbar_scan_yaw_offset_rad
                    ),
                )
                if self._busbar_preapproach_extra_m > 0.0:
                    world_goal = compute_directional_standoff_goal(
                        target_xy=(snapshot.x, snapshot.y),
                        approach_yaw_rad=self._busbar_approach_yaw,
                        standoff_m=(
                            self._standoff_m
                            + self._busbar_preapproach_extra_m
                        ),
                        docking_yaw_offset_rad=(
                            self._busbar_scan_yaw_offset_rad
                        ),
                    )
                    # 경유점에서는 최종 docking 자세로 90도 이상 제자리 회전할
                    # 필요가 없다. 현재 위치에서 경유점으로 향한 yaw로 도착한 뒤,
                    # 짧은 최종 선분에서 작업 셀 normal로 정렬한다.
                    pre_dx = world_goal.x - float(current.x)
                    pre_dy = world_goal.y - float(current.y)
                    world_goal = WorldGoal(
                        x=world_goal.x,
                        y=world_goal.y,
                        yaw=math.atan2(pre_dy, pre_dx),
                    )
                    phase = "preapproach"
                else:
                    world_goal = final_goal
                    phase = "final"
            else:
                standoff_m = (
                    self._battery_standoff_m
                    if pending.target_kind == "battery"
                    else self._standoff_m
                )
                world_goal = compute_standoff_goal(
                    current_xy=(float(current.x), float(current.y)),
                    target_xy=(snapshot.x, snapshot.y),
                    standoff_m=standoff_m,
                )
                final_goal = world_goal
                phase = "final"
        except VisionUnavailable as exc:
            self._pending_reason = str(exc)
            self.get_logger().warn(
                f"{pending.target_kind} 비전 목표 생성 대기: {exc}",
                throttle_duration_sec=3.0,
            )
            return

        selected = PoseStamped()
        selected.header.frame_id = "world"
        selected.header.stamp.sec = (
            snapshot.source_stamp_ns // 1_000_000_000
        )
        selected.header.stamp.nanosec = (
            snapshot.source_stamp_ns % 1_000_000_000
        )
        selected.pose.position.x = snapshot.x
        selected.pose.position.y = snapshot.y
        selected.pose.position.z = snapshot.z
        selected.pose.orientation.w = 1.0
        if pending.target_kind == "battery":
            self._selected_battery_pub.publish(selected)
        else:
            # relay가 confidence gate를 통과시켜 실제 주행에 사용한 바로 그
            # busbar 좌표를 ArmNode에도 전달한다. 고정 카메라를 재조회해 다른
            # 후보로 바뀌는 폴백은 사용하지 않는다.
            self._selected_busbar_pub.publish(selected)

        final_allow_reverse = (
            pending.target_kind == "busbar"
            and not math.isclose(
                self._busbar_scan_yaw_offset_rad,
                0.0,
                abs_tol=1.0e-9,
            )
        )
        self._pending = None
        self._active = RelayGoal(
            goal_id=uuid.uuid4().hex,
            request_id=pending.goal_id,
            target_kind=pending.target_kind,
            x=world_goal.x,
            y=world_goal.y,
            yaw=world_goal.yaw,
            vision_target_x=snapshot.x,
            vision_target_y=snapshot.y,
            vision_source_stamp_ns=snapshot.source_stamp_ns,
            phase=phase,
            # 배터리는 시작 pose가 이미 standoff 허용오차 안일 수 있으므로
            # 위치 도착 뒤 불필요한 chassis yaw 정렬을 강제하지 않는다.
            yaw_required=(
                pending.target_kind == "busbar"
                and phase != "preapproach"
            ),
            allow_reverse=(
                phase == "final" and final_allow_reverse
            ),
            align_yaw_before_drive=(
                phase == "final" and final_allow_reverse
            ),
            final_allow_reverse=final_allow_reverse,
            final_x=final_goal.x,
            final_y=final_goal.y,
            final_yaw=final_goal.yaw,
        )
        self.get_logger().info(
            f"비전 snapshot [{pending.target_kind}] "
            f"target=({snapshot.x:.4f}, {snapshot.y:.4f}, "
            f"{snapshot.z:.4f}), score>={snapshot.min_score:.3f}, "
            f"n={snapshot.sample_count}, "
            f"std={snapshot.std_xy_m * 1000.0:.1f}mm -> "
            f"chassis world "
            f"{'pre-approach' if phase == 'preapproach' else 'goal'}="
            f"({world_goal.x:.4f}, "
            f"{world_goal.y:.4f}, "
            f"{math.degrees(world_goal.yaw):.1f}deg)"
        )
        self._send_active_goal()

    def _publish_goal_ack(self, goal: RelayGoal):
        self._publish_goal_ack_fields(goal.target_kind, goal.request_id)

    def _publish_goal_ack_fields(self, target_kind: str, goal_id: str):
        ack = String()
        ack.data = json.dumps(
            {
                "station_id": target_kind,
                "goal_id": goal_id,
                "source": "perception",
            },
            separators=(",", ":"),
        )
        self._goal_ack_pub.publish(ack)

    def _send_active_goal(self):
        if self._active is None or not self._session_id:
            return
        self._publish_command("GOAL", self._active)
        self._active.command_sent = True
        self._active.cancel_requested = False
        self._last_bridge_state_wall = time.monotonic()
        self._publish_status(
            AmrStatus.STATE_MOVING,
            self._active.target_kind,
            "검증된 perception world 좌표로 Isaac 물리 주행 시작",
        )

    def _publish_command(
        self,
        command_type: str,
        goal: RelayGoal,
        *,
        session_id: Optional[str] = None,
    ):
        payload = {
            "version": 1,
            "type": command_type,
            "session_id": (
                self._session_id if session_id is None else session_id
            ),
            "goal_id": goal.goal_id,
        }
        if command_type == "GOAL":
            payload.update(
                {
                    "frame_id": goal.frame_id,
                    "x": goal.x,
                    "y": goal.y,
                    "yaw": goal.yaw,
                    "yaw_required": goal.yaw_required,
                    "allow_reverse": goal.allow_reverse,
                    "align_yaw_before_drive": (
                        goal.align_yaw_before_drive
                    ),
                    "lease_required": True,
                    "source": "perception",
                }
            )
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self._command_pub.publish(msg)

    def _publish_lease(self):
        if (
            self._active is None
            or not self._active.command_sent
            or self._active.cancel_requested
            or not self._session_id
        ):
            return
        msg = String()
        msg.data = json.dumps(
            {
                "version": 1,
                "session_id": self._session_id,
                "goal_id": self._active.goal_id,
            },
            separators=(",", ":"),
        )
        self._lease_pub.publish(msg)

    def _on_drive_state(self, msg: String):
        try:
            payload = self._decode_json(msg)
            session_id = str(payload["session_id"])
            goal_id = str(payload.get("goal_id", ""))
            active_goal_id = str(payload.get("active_goal_id", ""))
            state = str(payload["state"]).upper()
            message = str(payload.get("message", ""))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().warn(f"잘못된 drive_state 무시: {exc}")
            return

        if (
            self._active is None
            or session_id != self._session_id
            or goal_id != self._active.goal_id
        ):
            return

        self._last_bridge_state_wall = time.monotonic()
        if state in ("ACTIVE", "DRIVING"):
            return

        active = self._active
        if state == "REJECTED":
            if active_goal_id == active.goal_id:
                self.get_logger().warn(
                    "중복 command가 거부됐지만 기존 active goal은 계속됩니다: "
                    f"{message}"
                )
                return
            self._active = None
            self._publish_status(
                AmrStatus.STATE_ERROR,
                active.target_kind,
                message or "Isaac command 거부",
            )
            return
        if state == "ARRIVED":
            if active.phase == "preapproach":
                if (
                    active.final_x is None
                    or active.final_y is None
                    or active.final_yaw is None
                ):
                    self._active = None
                    self._publish_status(
                        AmrStatus.STATE_ERROR,
                        active.target_kind,
                        "pre-approach에 최종 docking goal이 없습니다",
                    )
                    return
                self._active = RelayGoal(
                    goal_id=uuid.uuid4().hex,
                    request_id=active.request_id,
                    target_kind=active.target_kind,
                    x=active.final_x,
                    y=active.final_y,
                    yaw=active.final_yaw,
                    vision_target_x=active.vision_target_x,
                    vision_target_y=active.vision_target_y,
                    vision_source_stamp_ns=(
                        active.vision_source_stamp_ns
                    ),
                    phase="final",
                    yaw_required=True,
                    allow_reverse=active.final_allow_reverse,
                    align_yaw_before_drive=active.final_allow_reverse,
                    final_allow_reverse=active.final_allow_reverse,
                    final_x=active.final_x,
                    final_y=active.final_y,
                    final_yaw=active.final_yaw,
                )
                self._publish_status(
                    AmrStatus.STATE_MOVING,
                    active.target_kind,
                    "안전 pre-approach 도착; 최종 docking 시작",
                )
                self.get_logger().info(
                    f"busbar pre-approach 도착 -> 최종 goal="
                    f"({active.final_x:.4f}, {active.final_y:.4f}, "
                    f"{math.degrees(active.final_yaw):.1f}deg)"
                )
                self._send_active_goal()
                return
            self._active = None
            self._publish_status(
                AmrStatus.STATE_ARRIVED,
                active.target_kind,
                message or "실제 chassis pose 도착",
            )
        elif state in ("FAILED", "CANCELED"):
            self._active = None
            self._publish_status(
                AmrStatus.STATE_ERROR,
                active.target_kind,
                message or state,
            )

    def _on_cancel_request(self, msg: String):
        try:
            payload = self._decode_json(msg)
            target_kind = str(payload["station_id"]).strip().lower()
            goal_id = str(payload["goal_id"]).strip()
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().warn(f"잘못된 cancel_request 무시: {exc}")
            return

        if (
            self._pending is not None
            and target_kind == self._pending.target_kind
            and goal_id == self._pending.goal_id
        ):
            pending = self._pending
            self._pending = None
            self._publish_status(
                AmrStatus.STATE_ERROR,
                pending.target_kind,
                "비전 좌표 대기 요청이 취소됐습니다",
            )
            return

        if (
            self._active is None
            or target_kind != self._active.target_kind
            or goal_id != self._active.request_id
        ):
            self.get_logger().warn(
                f"현재 goal과 다른 cancel_request 무시: "
                f"{target_kind}/{goal_id}"
            )
            return
        self.request_cancel("상위 cancel 요청")

    def request_cancel(self, reason: str):
        if self._pending is not None:
            pending = self._pending
            self._pending = None
            self._publish_status(
                AmrStatus.STATE_ERROR,
                pending.target_kind,
                f"{reason}: 비전 좌표 대기 취소",
            )
        if self._active is None:
            return
        active = self._active
        if not active.command_sent:
            self._active = None
            self._publish_status(
                AmrStatus.STATE_ERROR,
                active.target_kind,
                f"{reason}: READY 대기 목표 취소",
            )
            return
        if active.cancel_requested:
            return
        active.cancel_requested = True
        self._publish_command("CANCEL", active)
        self.get_logger().warn(f"CANCEL id={active.goal_id}: {reason}")

    def _invalidate_vision(self, target_kind: str, reason: str):
        self._vision_gates[target_kind].reset(reason)
        self._fail_pending_for(target_kind, reason)
        if (
            self._active is not None
            and self._active.target_kind == target_kind
        ):
            self.request_cancel(f"active perception source 무효: {reason}")

    def _fail_pending_for(self, target_kind: str, reason: str):
        if (
            self._pending is None
            or self._pending.target_kind != target_kind
        ):
            return
        pending = self._pending
        self._pending = None
        self._publish_status(
            AmrStatus.STATE_ERROR,
            pending.target_kind,
            reason,
        )

    def _on_watchdog(self):
        now = time.monotonic()
        if self._pending is not None:
            if (
                self._pending.deadline_wall is not None
                and now > self._pending.deadline_wall
            ):
                pending = self._pending
                self._pending = None
                self._publish_status(
                    AmrStatus.STATE_ERROR,
                    pending.target_kind,
                    "비전 목표 생성 시간 초과: "
                    f"{self._pending_reason}; 하드코딩 폴백 없이 정지",
                )
            else:
                self._try_resolve_pending()

        if self._active is None:
            return

        active = self._active
        if not active.command_sent:
            self._active = None
            self._publish_status(
                AmrStatus.STATE_ERROR,
                active.target_kind,
                "READY 검증 없이 생성된 내부 목표를 거부했습니다",
            )
            return

        source_topics = (
            READY_TOPIC,
            WORLD_POSE_TOPIC,
            self._target_topic(active.target_kind),
        )
        for source_topic in source_topics:
            source_count = self.count_publishers(source_topic)
            if source_count != 1:
                self.request_cancel(
                    f"{source_topic} publisher 수가 "
                    f"{source_count}개라 source 무결성이 깨졌습니다"
                )
                return

        pose_age = now - self._last_world_pose_wall
        if (
            not self._debug_no_timeouts
            and (
                self._current_world_pose is None
                or not math.isfinite(pose_age)
                or pose_age > self._world_pose_max_age_sec
            )
        ):
            self.request_cancel(
                f"Isaac world_pose freshness 위반: age={pose_age:.2f}s"
            )
            return

        if self._enforce_active_drift:
            gate = self._vision_gates[active.target_kind]
            try:
                # active drift guard를 명시적으로 켠 경우에만 새 표본으로
                # 처음 선택한 대상의 이동 여부를 검사한다. 기본 전체 공정은
                # 검증된 snapshot을 latch해 카메라 노이즈로 주행을 취소하지 않는다.
                snapshot = gate.snapshot(
                    now,
                    max_age_sec=self._active_vision_timeout_sec,
                    enforce_age=not self._debug_no_timeouts,
                )
            except VisionUnavailable as exc:
                self.request_cancel(
                    f"active perception freshness 위반: {exc}"
                )
                return
            target_drift = math.hypot(
                snapshot.x - active.vision_target_x,
                snapshot.y - active.vision_target_y,
            )
            if target_drift > self._max_target_drift_m:
                self.request_cancel(
                    f"perception target이 {target_drift:.3f}m 이동해 "
                    f"제한 {self._max_target_drift_m:.3f}m를 초과했습니다"
                )
                return

        if self._debug_no_timeouts:
            return

        silence = now - self._last_bridge_state_wall
        if silence <= BRIDGE_STATE_TIMEOUT_SEC:
            return
        active = self._active
        self._publish_command("CANCEL", active)
        self._active = None
        self._publish_status(
            AmrStatus.STATE_ERROR,
            active.target_kind,
            f"Isaac bridge state가 {silence:.1f}s 동안 없어 안전 취소",
        )

    def _publish_status(self, state: int, station_id: str, message: str):
        status = AmrStatus()
        status.state = state
        status.station_id = station_id
        status.message = message
        self._status_pub.publish(status)
        self.get_logger().info(
            f"STATUS {station_id} state={state}: {message}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = AmrDriveNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if rclpy.ok():
                node.request_cancel("relay 종료")
                deadline = time.monotonic() + 0.5
                while (
                    node.has_active_goal
                    and rclpy.ok()
                    and time.monotonic() < deadline
                ):
                    rclpy.spin_once(node, timeout_sec=0.05)
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()

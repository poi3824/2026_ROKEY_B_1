#!/usr/bin/env python3
"""
Run the ROS 2 arm action server with camera-specific perception.

- BehaviorNode의 /execute_arm_task Action 수신
- 타임아웃 없이 Isaac Sim 및 비전 서비스 응답을 무제한 대기
- [지원 Task]
  1. SCAN_BATTERY        : 배터리 스캔 위치로 이동 후 Perception 노드로부터 두 볼트의 중앙 좌표 수신 및 저장
  2. LATCH_BUSBAR        : 고정캠 버스바 좌표만 latch하고 손목 스캔 없이 완료
     SCAN_BUSBAR         : 명시적 legacy용 고정캠 + 손목캠 동일 대상 확인
  3. PICK_BUSBAR         : latch된 고정캠 snapshot만 사용해 파지 명령 중계
  4. MOVE_BATTERY_CENTER : Isaac Sim이 현재 station의 live bolt prim 중점을 읽어 이동하도록 명령 중계
  5. FINE_ALIGNMENT       : Isaac Sim 및 비전 노드의 정밀 1픽셀 오차 보정 명령 중계
  6. ASSEMBLE_BUSBAR      : Isaac Sim으로 버스바 하강 체결 및 그리퍼 해제 명령 중계
  7. SCAN_NUT1/SCAN_NUT2       : 너트 스캔 위치 이동 후 Perception 노드로부터 너트 좌표 수신 및 저장
  8. PICK_NUT1/PICK_NUT2       : 스캔된 너트 좌표로 Isaac Sim 파지 명령 중계
  9. ASSEMBLE_NUT1/ASSEMBLE_NUT2 : Isaac Sim으로 너트 Screwing 체결 명령 중계
"""

import math
import threading
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty, Float32, String

# Action 및 Custom Interfaces
from fms_interfaces.action import ExecuteArmTask
from fms_interfaces.srv import GetGraspPose, GetBoltPair
from fms_interfaces.msg import NutPose


class PerceptionProtocolError(RuntimeError):
    """The connected perception server lacks the reset-generation contract."""


class ArmNode(Node):
    """Coordinate perception requests with Isaac arm task commands."""

    WRIST_BUSBAR_REQUIRED_SAMPLES = 2
    WRIST_BUSBAR_MAX_TARGET_XY_M = 0.20
    WRIST_BUSBAR_MAX_TARGET_Z_M = 0.10
    WRIST_BUSBAR_MAX_SAMPLE_SHIFT_M = 0.05
    PERCEPTION_RETRY_INTERVAL_SEC = 0.2

    def __init__(self):
        """Create action, perception, reset, and Isaac bridge interfaces."""
        super().__init__('arm_node')
        self.get_logger().info("===========================================")
        self.get_logger().info(" ArmNode 활성화 (타임아웃 없이 무제한 대기 모드)")
        self.get_logger().info("===========================================")

        # Reentrant Callback Group 적용 (교착 상태 방지)
        self.cb_group = ReentrantCallbackGroup()
        # Reentrant executor에서도 arm task state는 한 goal만 소유해야 한다.
        self._execution_lock = threading.Lock()

        # 1. Action Server 생성
        self._action_server = ActionServer(
            self,
            ExecuteArmTask,
            '/execute_arm_task',
            execute_callback=self.execute_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.cb_group
        )

        # 2. Perception Node 서비스 클라이언트들
        self.client_get_wrist_pose = self.create_client(
            GetGraspPose,
            '/wrist/perception/get_grasp_pose',
            callback_group=self.cb_group,
        )
        self.client_get_busbar_pose = self.create_client(
            GetGraspPose,
            '/busbar_cam/perception/get_grasp_pose',
            callback_group=self.cb_group,
        )
        self.client_get_bolt_pair = self.create_client(
            GetBoltPair,
            '/bolt_cam/perception/get_bolt_pair',
            callback_group=self.cb_group,
        )

        # 카메라 이동 뒤 perception 캐시를 명시적으로 비운다.
        self.pub_wrist_reset = self.create_publisher(
            Empty, '/wrist/perception/reset_cache', 10
        )
        self.pub_busbar_reset = self.create_publisher(
            Empty, '/busbar_cam/perception/reset_cache', 10
        )
        self.pub_bolt_reset = self.create_publisher(
            Empty, '/bolt_cam/perception/reset_cache', 10
        )
        # Empty reset ACKs are kept for wire compatibility/diagnostics, but
        # cannot identify which Empty request they acknowledge. Causal reset
        # completion is verified below through the perception service's
        # generation metadata instead.
        self._perception_reset_ack_counts = {
            "wrist": 0,
            "busbar_cam": 0,
            "bolt_cam": 0,
        }
        self._required_perception_generation = {
            "wrist": None,
            "busbar_cam": None,
            "bolt_cam": None,
        }
        self._last_perception_reset_error = None
        self.sub_wrist_reset_ack = self.create_subscription(
            Empty,
            '/wrist/perception/reset_ack',
            lambda _msg: self._record_perception_reset_ack("wrist"),
            10,
            callback_group=self.cb_group,
        )
        self.sub_busbar_reset_ack = self.create_subscription(
            Empty,
            '/busbar_cam/perception/reset_ack',
            lambda _msg: self._record_perception_reset_ack("busbar_cam"),
            10,
            callback_group=self.cb_group,
        )
        self.sub_bolt_reset_ack = self.create_subscription(
            Empty,
            '/bolt_cam/perception/reset_ack',
            lambda _msg: self._record_perception_reset_ack("bolt_cam"),
            10,
            callback_group=self.cb_group,
        )

        # 3. 너트 legacy 진단 토픽 구독. SCAN_NUT 결과에는 사용하지 않는다.
        self.latest_nut_pose = None

        self.sub_nut_pose = self.create_subscription(
            NutPose, '/vision/nut_pose', self._on_nut_pose, 10, callback_group=self.cb_group
        )

        # 4. Isaac Sim 연동 Pub / Sub
        self.pub_target_pose = self.create_publisher(PoseStamped, '/target_pose', 10)
        self.pub_task_command = self.create_publisher(String, '/task_command', 10)

        self.sub_isaac_phase = self.create_subscription(
            String, '/isaac_phase', self._on_isaac_phase, 10, callback_group=self.cb_group
        )
        self.sub_isaac_progress = self.create_subscription(
            Float32, '/isaac_progress', self._on_isaac_progress, 10, callback_group=self.cb_group
        )
        self.sub_isaac_status = self.create_subscription(
            String, '/isaac_status', self._on_isaac_status, 10, callback_group=self.cb_group
        )

        # 내부 상태 및 저장 변수
        self.isaac_status = None
        self.isaac_phase = ""
        self.isaac_progress = 0.0

        # 배터리 스캔 시 저장할 두 볼트의 중앙 좌표 (PoseStamped)
        self.scanned_battery_midpoint = None

        # 버스바 스캔 시 저장할 버스바 파지 좌표 (PoseStamped)
        self.scanned_busbar_pose = None

        # 너트 스캔 시 저장할 너트 1번 / 2번 좌표 (PoseStamped)
        self.scanned_nut1_pose = None
        self.scanned_nut2_pose = None

    # =========================================================================
    # 콜백 함수들
    # =========================================================================
    def _on_nut_pose(self, msg: NutPose):
        self.latest_nut_pose = msg.pose

    def _on_isaac_phase(self, msg: String):
        self.isaac_phase = msg.data

    def _on_isaac_progress(self, msg: Float32):
        self.isaac_progress = msg.data

    def _on_isaac_status(self, msg: String):
        self.isaac_status = msg.data
        self.get_logger().info(f"[Isaac Status 수신]: {self.isaac_status}")

    # =========================================================================
    # Action Callback
    # =========================================================================
    def cancel_callback(self, _goal_handle):
        """Accept cancellation of an active arm task."""
        self.get_logger().info("ExecuteArmTask cancel 요청을 수락합니다.")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Serialize externally compatible ``ExecuteArmTask`` goals."""
        if not self._execution_lock.acquire(blocking=False):
            result_msg = ExecuteArmTask.Result()
            result_msg.success = False
            result_msg.error_code = "BUSY"
            result_msg.message = (
                "다른 ExecuteArmTask가 실행 중이므로 요청을 거부했습니다."
            )
            self._abort_or_cancel(goal_handle)
            return result_msg

        try:
            return self._execute_task(goal_handle)
        finally:
            self._execution_lock.release()

    def _execute_task(self, goal_handle):
        """Execute one goal while holding the arm execution lock."""
        task_type = goal_handle.request.task_type
        self.get_logger().info(
            "\n================ "
            f"[Action Goal Received: {task_type}] ================"
        )

        feedback_msg = ExecuteArmTask.Feedback()
        result_msg = ExecuteArmTask.Result()
        self.isaac_status = None

        # ---------------------------------------------------------------------
        # [Task 1] SCAN_BATTERY (배터리 스캔 지점 이동 & 두 볼트 중앙 좌표 저장)
        # ---------------------------------------------------------------------
        if task_type == "SCAN_BATTERY":
            self.get_logger().info(" -> [SCAN_BATTERY] Isaac Sim으로 스캔 이동 명령 전송")

            self.scanned_battery_midpoint = None
            cmd_msg = String()
            cmd_msg.data = "SCAN_BATTERY"
            self.pub_task_command.publish(cmd_msg)

            # 1. Isaac Sim이 스캔 고도로 이동 완료할 때까지 대기
            success = self.wait_for_isaac_completion(goal_handle, feedback_msg)

            if not success:
                result_msg.success = False
                if goal_handle.is_cancel_requested:
                    result_msg.error_code = "SCAN_BATTERY_CANCELED"
                    result_msg.message = "배터리 스캔이 취소됐습니다."
                else:
                    result_msg.error_code = "SCAN_BATTERY_FAILED"
                    result_msg.message = (
                        "배터리 스캔 위치 이동 실패 "
                        f"(Status: {self.isaac_status})"
                    )
                self._abort_or_cancel(goal_handle)
                return result_msg

            # 2. 이동 완료 뒤 과거 캐시를 버리고 고정 bolt camera의 새 좌표를 요청
            self.get_logger().info(" -> [SCAN_BATTERY] 배터리 스캔 위치 도착 완료. 비전 노드에 볼트 쌍 좌표 요청...")
            if not self._reset_perception_cache(
                self.pub_bolt_reset,
                goal_handle,
            ):
                result_msg.success = False
                if goal_handle.is_cancel_requested:
                    result_msg.error_code = "SCAN_BATTERY_CANCELED"
                    result_msg.message = (
                        "배터리 perception reset 대기 중 취소됐습니다."
                    )
                elif self._last_perception_reset_error is not None:
                    result_msg.error_code = (
                        "PERCEPTION_PROTOCOL_MISMATCH"
                    )
                    result_msg.message = self._last_perception_reset_error
                else:
                    result_msg.error_code = "BOLT_RESET_INTERRUPTED"
                    result_msg.message = (
                        "bolt camera perception reset generation 확인이 "
                        "중단됐습니다."
                    )
                self._abort_or_cancel(goal_handle)
                return result_msg
            try:
                found, midpoint_pose, msg = (
                    self.request_bolt_pair_midpoint_async(
                        goal_handle,
                    )
                )
            except PerceptionProtocolError as exc:
                result_msg.success = False
                result_msg.error_code = "PERCEPTION_PROTOCOL_MISMATCH"
                result_msg.message = (
                    self._record_perception_protocol_error(exc)
                )
                self._abort_or_cancel(goal_handle)
                return result_msg

            if found and midpoint_pose is not None:
                self.scanned_battery_midpoint = midpoint_pose
                self.get_logger().info(
                    f" ★ [배터리 볼트 중점 저장 완료] "
                    f"X: {midpoint_pose.pose.position.x:.4f}, "
                    f"Y: {midpoint_pose.pose.position.y:.4f}, "
                    f"Z: {midpoint_pose.pose.position.z:.4f}"
                )

                result_msg.success = True
                result_msg.message = f"배터리 스캔 및 볼트 중점 취득 성공 ({msg})"
                goal_handle.succeed()
            else:
                if goal_handle.is_cancel_requested:
                    result_msg.success = False
                    result_msg.error_code = "SCAN_BATTERY_CANCELED"
                    result_msg.message = "배터리 스캔이 취소됐습니다."
                    goal_handle.canceled()
                    return result_msg
                self.get_logger().error(f" -> 볼트 쌍 비전 검출 실패: {msg}")
                result_msg.success = False
                result_msg.error_code = "BOLT_PAIR_VISION_FAILED"
                result_msg.message = f"볼트 쌍 중점 취득 실패: {msg}"
                self._abort_or_cancel(goal_handle)

            return result_msg

        # ---------------------------------------------------------------------
        # [Task 2] LATCH/SCAN_BUSBAR (고정캠 버스바 비전 좌표 저장)
        # ---------------------------------------------------------------------
        elif task_type in {"LATCH_BUSBAR", "SCAN_BUSBAR"}:
            fixed_camera_only = task_type == "LATCH_BUSBAR"
            self.get_logger().info(
                f" -> [{task_type}] Isaac Sim으로 고정 버스바 카메라 "
                "이동 명령 전송"
            )

            # 이미 취소된 goal은 상태 변경이나 Isaac task를 시작하지 않는다.
            if goal_handle.is_cancel_requested:
                result_msg.success = False
                result_msg.error_code = "SCAN_BUSBAR_CANCELED"
                result_msg.message = "버스바 스캔이 취소됐습니다."
                goal_handle.canceled()
                return result_msg

            # 이전 cycle의 snapshot은 새 SCAN이 시작되는 즉시 무효화한다.
            self.scanned_busbar_pose = None
            # 직전 task의 phase를 새 handshake의 READY로 오인하지 않는다.
            self.isaac_phase = ""

            cmd_msg = String()
            cmd_msg.data = "SCAN_BUSBAR"
            self.pub_task_command.publish(cmd_msg)

            # 1. Isaac은 고정 카메라를 이동한 뒤 wrist 이동 전에 일시 정지한다.
            ready = self.wait_for_isaac_phase(
                "BUSBAR_CAMERA_READY",
                goal_handle,
                feedback_msg,
            )

            if not ready:
                # CONTINUE 전에 빠져나가면 Isaac의 pause를 반드시 해제한다.
                self._publish_cancel_arm_task()
                result_msg.success = False
                if goal_handle.is_cancel_requested:
                    result_msg.error_code = "SCAN_BUSBAR_CANCELED"
                    result_msg.message = "버스바 스캔이 취소됐습니다."
                else:
                    result_msg.error_code = "SCAN_BUSBAR_FAILED"
                    result_msg.message = (
                        "고정 카메라 준비 대기 실패 "
                        f"(Status: {self.isaac_status})"
                    )
                self._abort_or_cancel(goal_handle)
                return result_msg

            # 2. 고정 카메라의 새 snapshot을 먼저 latch한다.
            if not self._reset_perception_cache(
                self.pub_busbar_reset,
                goal_handle,
            ):
                self._publish_cancel_arm_task()
                result_msg.success = False
                if goal_handle.is_cancel_requested:
                    result_msg.error_code = "SCAN_BUSBAR_CANCELED"
                    result_msg.message = "버스바 스캔이 취소됐습니다."
                elif self._last_perception_reset_error is not None:
                    result_msg.error_code = (
                        "PERCEPTION_PROTOCOL_MISMATCH"
                    )
                    result_msg.message = self._last_perception_reset_error
                else:
                    result_msg.error_code = "BUSBAR_RESET_INTERRUPTED"
                    result_msg.message = (
                        "고정 카메라 perception reset이 중단됐습니다."
                    )
                self._abort_or_cancel(goal_handle)
                return result_msg

            try:
                found, fixed_pose, fixed_msg = (
                    self.request_grasp_pose_until_found(
                        self.client_get_busbar_pose,
                        "busbar",
                        goal_handle,
                    )
                )
            except PerceptionProtocolError as exc:
                self._publish_cancel_arm_task()
                result_msg.success = False
                result_msg.error_code = "PERCEPTION_PROTOCOL_MISMATCH"
                result_msg.message = (
                    self._record_perception_protocol_error(exc)
                )
                self._abort_or_cancel(goal_handle)
                return result_msg
            if not found or fixed_pose is None:
                self._publish_cancel_arm_task()
                result_msg.success = False
                if goal_handle.is_cancel_requested:
                    result_msg.error_code = "SCAN_BUSBAR_CANCELED"
                    result_msg.message = "버스바 스캔이 취소됐습니다."
                    goal_handle.canceled()
                else:
                    result_msg.error_code = "BUSBAR_VISION_INTERRUPTED"
                    result_msg.message = fixed_msg
                    self._abort_or_cancel(goal_handle)
                return result_msg

            # 3. fixed_pose를 지역 snapshot으로 보존한다. 기본 fleet 흐름의
            # LATCH_BUSBAR는 여기서 wrist 자세/검증을 건너뛰고 pause를
            # 완료한다. 명시적 legacy SCAN_BUSBAR만 기존 wrist 검증을 계속한다.
            if goal_handle.is_cancel_requested or self._isaac_has_failed():
                self._publish_cancel_arm_task()
                result_msg.success = False
                if goal_handle.is_cancel_requested:
                    result_msg.error_code = "SCAN_BUSBAR_CANCELED"
                    result_msg.message = "버스바 스캔이 취소됐습니다."
                else:
                    result_msg.error_code = "SCAN_BUSBAR_FAILED"
                    result_msg.message = (
                        "고정 카메라 latch 뒤 Isaac failure를 수신했습니다."
                    )
                self._abort_or_cancel(goal_handle)
                return result_msg

            if fixed_camera_only:
                self.scanned_busbar_pose = fixed_pose

                complete_msg = String()
                complete_msg.data = "COMPLETE_BUSBAR_FIXED_SCAN"
                self.pub_task_command.publish(complete_msg)

                success = self.wait_for_isaac_completion(
                    goal_handle,
                    feedback_msg,
                )
                if not success:
                    # wait_for_isaac_completion이 Action cancel이면 Isaac
                    # pause 해제 명령까지 발행한다.
                    self.scanned_busbar_pose = None
                    result_msg.success = False
                    if goal_handle.is_cancel_requested:
                        result_msg.error_code = "SCAN_BUSBAR_CANCELED"
                        result_msg.message = "버스바 latch가 취소됐습니다."
                    else:
                        result_msg.error_code = "SCAN_BUSBAR_FAILED"
                        result_msg.message = (
                            "고정 버스바 카메라 latch 완료 handshake 실패 "
                            f"(Status: {self.isaac_status})"
                        )
                    self._abort_or_cancel(goal_handle)
                    return result_msg

                self.get_logger().info(
                    " ★ [고정캠 버스바 snapshot latch 완료] "
                    f"fixed=({fixed_pose.pose.position.x:.4f}, "
                    f"{fixed_pose.pose.position.y:.4f}, "
                    f"{fixed_pose.pose.position.z:.4f}); "
                    "wrist scan 생략"
                )
                result_msg.success = True
                result_msg.message = (
                    "고정캠 버스바 snapshot latch 성공; "
                    f"손목 스캔 생략 ({fixed_msg})"
                )
                goal_handle.succeed()
                return result_msg

            continue_msg = String()
            continue_msg.data = "CONTINUE_BUSBAR_WRIST_SCAN"
            self.pub_task_command.publish(continue_msg)

            success = self.wait_for_isaac_completion(
                goal_handle,
                feedback_msg,
            )
            if not success:
                result_msg.success = False
                if goal_handle.is_cancel_requested:
                    result_msg.error_code = "SCAN_BUSBAR_CANCELED"
                    result_msg.message = "버스바 스캔이 취소됐습니다."
                else:
                    result_msg.error_code = "SCAN_BUSBAR_FAILED"
                    result_msg.message = (
                        "손목 카메라 스캔 자세 이동 실패 "
                        f"(Status: {self.isaac_status})"
                    )
                self._abort_or_cancel(goal_handle)
                return result_msg

            # 4. wrist 자세 도착 전에 생성된 표본은 사용하지 않는다.
            if not self._reset_perception_cache(
                self.pub_wrist_reset,
                goal_handle,
            ):
                result_msg.success = False
                if goal_handle.is_cancel_requested:
                    result_msg.error_code = "SCAN_BUSBAR_CANCELED"
                    result_msg.message = "버스바 스캔이 취소됐습니다."
                elif self._last_perception_reset_error is not None:
                    result_msg.error_code = (
                        "PERCEPTION_PROTOCOL_MISMATCH"
                    )
                    result_msg.message = self._last_perception_reset_error
                else:
                    result_msg.error_code = "WRIST_RESET_INTERRUPTED"
                    result_msg.message = (
                        "wrist perception reset이 중단됐습니다."
                    )
                self._abort_or_cancel(goal_handle)
                return result_msg

            try:
                confirmed, wrist_pose, wrist_msg = (
                    self.wait_for_wrist_busbar_confirmation(
                        fixed_pose,
                        goal_handle,
                    )
                )
            except PerceptionProtocolError as exc:
                result_msg.success = False
                result_msg.error_code = "PERCEPTION_PROTOCOL_MISMATCH"
                result_msg.message = (
                    self._record_perception_protocol_error(exc)
                )
                self._abort_or_cancel(goal_handle)
                return result_msg
            if not confirmed or wrist_pose is None:
                result_msg.success = False
                if goal_handle.is_cancel_requested:
                    result_msg.error_code = "SCAN_BUSBAR_CANCELED"
                    result_msg.message = "버스바 스캔이 취소됐습니다."
                    goal_handle.canceled()
                else:
                    result_msg.error_code = "WRIST_VISION_INTERRUPTED"
                    result_msg.message = wrist_msg
                    self._abort_or_cancel(goal_handle)
                return result_msg

            # PICK 좌표는 wrist 관측으로 덮지 않고 고정캠 snapshot 그대로 유지한다.
            self.scanned_busbar_pose = fixed_pose
            self.get_logger().info(
                " ★ [버스바 동일 대상 확인 완료] "
                f"fixed=({fixed_pose.pose.position.x:.4f}, "
                f"{fixed_pose.pose.position.y:.4f}, "
                f"{fixed_pose.pose.position.z:.4f}), "
                f"wrist=({wrist_pose.pose.position.x:.4f}, "
                f"{wrist_pose.pose.position.y:.4f}, "
                f"{wrist_pose.pose.position.z:.4f})"
            )
            result_msg.success = True
            result_msg.message = (
                "고정캠 버스바 snapshot latch 및 손목 카메라 동일 대상 "
                f"확인 성공 ({fixed_msg}; {wrist_msg})"
            )
            goal_handle.succeed()
            return result_msg

        # ---------------------------------------------------------------------
        # [Task 3] PICK_BUSBAR (버스바 파지 및 들어올리기)
        # ---------------------------------------------------------------------
        elif task_type == "PICK_BUSBAR":
            busbar_pose = self.scanned_busbar_pose
            if busbar_pose is None:
                result_msg.success = False
                result_msg.error_code = "NO_FIXED_CAMERA_BUSBAR_POSE"
                result_msg.message = (
                    "LATCH_BUSBAR/SCAN_BUSBAR에서 latch한 고정캠 "
                    "버스바 snapshot이 없습니다. wrist 또는 배터리 "
                    "좌표로 대체하지 않습니다."
                )
                self._abort_or_cancel(goal_handle)
                return result_msg

            # Isaac Sim 목표 좌표 및 파지 명령 퍼블리시
            self.pub_target_pose.publish(busbar_pose)
            if not self._wait_while_active(0.1, goal_handle):
                result_msg.success = False
                result_msg.error_code = "PICK_BUSBAR_CANCELED"
                result_msg.message = "버스바 파지 명령 전송 전에 취소됐습니다."
                self._abort_or_cancel(goal_handle)
                return result_msg

            cmd_msg = String()
            cmd_msg.data = "PICK_BUSBAR"
            self.pub_task_command.publish(cmd_msg)

            success = self.wait_for_isaac_completion(goal_handle, feedback_msg)

            if success:
                result_msg.success = True
                result_msg.message = "버스바 파지 및 들어올리기 성공"
                goal_handle.succeed()
            else:
                result_msg.success = False
                if goal_handle.is_cancel_requested:
                    result_msg.error_code = "PICK_BUSBAR_CANCELED"
                    result_msg.message = "버스바 파지가 취소됐습니다."
                else:
                    result_msg.error_code = "ISAAC_FAILED"
                    result_msg.message = (
                        "Isaac Sim 완료 실패 "
                        f"(Status: {self.isaac_status})"
                    )
                self._abort_or_cancel(goal_handle)

            return result_msg

        # ---------------------------------------------------------------------
        # [Task 4] MOVE_BATTERY_CENTER (현재 station의 live bolt 중점 상공으로 이동)
        # ---------------------------------------------------------------------
        elif task_type == "MOVE_BATTERY_CENTER":
            self.get_logger().info(
                " -> [MOVE_BATTERY_CENTER] Isaac Sim이 현재 station의 "
                "live bolt prim 중점을 읽도록 이동 명령 전송"
            )

            cmd_msg = String()
            cmd_msg.data = "MOVE_BATTERY_CENTER"
            self.pub_task_command.publish(cmd_msg)

            # 목표는 Isaac의 현재 physics stage에서 계산한다. 과거 camera
            # snapshot을 /target_pose로 다시 보내면 pack 이동·station 전환 뒤
            # stale 좌표를 사용할 수 있으므로 이 경로에서는 발행하지 않는다.
            success = self.wait_for_isaac_completion(goal_handle, feedback_msg)

            if success:
                result_msg.success = True
                result_msg.message = "배터리 볼트 중점 상공 이동 성공"
                goal_handle.succeed()
            else:
                result_msg.success = False
                result_msg.error_code = "MOVE_BATTERY_CENTER_FAILED"
                result_msg.message = f"배터리 중점 이동 실패 (Status: {self.isaac_status})"
                self._abort_or_cancel(goal_handle)

            return result_msg

        # ---------------------------------------------------------------------
        # [Task 5] FINE_ALIGNMENT (비전 기반 1픽셀 미세 오차 보정 중계)
        # ---------------------------------------------------------------------
        elif task_type == "FINE_ALIGNMENT":
            self.get_logger().info(" -> [FINE_ALIGNMENT] Isaac Sim으로 정밀 비전 오차 보정 시작 명령 전송")

            cmd_msg = String()
            cmd_msg.data = "FINE_ALIGNMENT"
            self.pub_task_command.publish(cmd_msg)

            success = self.wait_for_isaac_completion(goal_handle, feedback_msg)

            if success:
                result_msg.success = True
                result_msg.message = "비전 기반 1픽셀 정밀 오차 보정 완료"
                goal_handle.succeed()
            else:
                result_msg.success = False
                result_msg.error_code = "FINE_ALIGNMENT_FAILED"
                result_msg.message = f"비전 정밀 오차 보정 실패 (Status: {self.isaac_status})"
                self._abort_or_cancel(goal_handle)

            return result_msg

        # ---------------------------------------------------------------------
        # [Task 6] ASSEMBLE_BUSBAR (버스바 하강 체결 및 그리퍼 해제 중계)
        # ---------------------------------------------------------------------
        elif task_type == "ASSEMBLE_BUSBAR":
            self.get_logger().info(" -> [ASSEMBLE_BUSBAR] Isaac Sim으로 버스바 최종 체결 명령 전송")

            cmd_msg = String()
            cmd_msg.data = "ASSEMBLE_BUSBAR"
            self.pub_task_command.publish(cmd_msg)

            success = self.wait_for_isaac_completion(goal_handle, feedback_msg)

            if success:
                result_msg.success = True
                result_msg.message = "버스바 체결 및 파지 해제 완료"
                goal_handle.succeed()
            else:
                result_msg.success = False
                result_msg.error_code = "ASSEMBLE_BUSBAR_FAILED"
                result_msg.message = f"버스바 체결 실패 (Status: {self.isaac_status})"
                self._abort_or_cancel(goal_handle)

            return result_msg

        # ---------------------------------------------------------------------
        # [Task] RETURN_HOME (너트 작업 전 관절 각도만 초기 자세로 복귀, 스캔 이동 없음)
        # ---------------------------------------------------------------------
        elif task_type == "RETURN_HOME":
            self.get_logger().info(" -> [RETURN_HOME] Isaac Sim으로 초기 자세 복귀 명령 전송")

            cmd_msg = String()
            cmd_msg.data = "RETURN_HOME"
            self.pub_task_command.publish(cmd_msg)

            success = self.wait_for_isaac_completion(goal_handle, feedback_msg)

            if success:
                result_msg.success = True
                result_msg.message = "초기 자세 복귀 완료"
                goal_handle.succeed()
            else:
                result_msg.success = False
                result_msg.error_code = "RETURN_HOME_FAILED"
                result_msg.message = f"초기 자세 복귀 실패 (Status: {self.isaac_status})"
                self._abort_or_cancel(goal_handle)

            return result_msg

        # ---------------------------------------------------------------------
        # [Task 7] SCAN_NUT1 / SCAN_NUT2 (너트 스캔 지점 이동 & 비전 좌표 저장)
        # ---------------------------------------------------------------------
        elif task_type in ("SCAN_NUT1", "SCAN_NUT2"):
            self.get_logger().info(f" -> [{task_type}] Isaac Sim으로 너트 스캔 이동 명령 전송")
            if task_type == "SCAN_NUT1":
                self.scanned_nut1_pose = None
            else:
                self.scanned_nut2_pose = None

            cmd_msg = String()
            cmd_msg.data = task_type
            self.pub_task_command.publish(cmd_msg)

            # 1. Isaac Sim이 너트 스캔 위치로 이동 완료할 때까지 대기
            success = self.wait_for_isaac_completion(goal_handle, feedback_msg)

            if not success:
                result_msg.success = False
                result_msg.error_code = f"{task_type}_FAILED"
                result_msg.message = f"너트 스캔 위치 이동 실패 (Status: {self.isaac_status})"
                self._abort_or_cancel(goal_handle)
                return result_msg

            # 2. 이동 완료 후 Perception 노드에 너트 좌표 요청
            self.get_logger().info(f" -> [{task_type}] 너트 스캔 위치 도착 완료. 비전 노드에 너트 좌표 요청...")
            if not self._reset_perception_cache(
                self.pub_wrist_reset,
                goal_handle,
            ):
                result_msg.success = False
                if goal_handle.is_cancel_requested:
                    result_msg.error_code = f"{task_type}_CANCELED"
                    result_msg.message = "너트 perception reset 대기 중 취소됐습니다."
                elif self._last_perception_reset_error is not None:
                    result_msg.error_code = (
                        "PERCEPTION_PROTOCOL_MISMATCH"
                    )
                    result_msg.message = self._last_perception_reset_error
                else:
                    result_msg.error_code = "NUT_RESET_INTERRUPTED"
                    result_msg.message = (
                        "너트 perception reset generation 확인이 "
                        "중단됐습니다."
                    )
                self._abort_or_cancel(goal_handle)
                return result_msg
            try:
                found, nut_pose, msg = (
                    self.request_grasp_pose_until_found(
                        self.client_get_wrist_pose,
                        "nut",
                        goal_handle,
                    )
                )
            except PerceptionProtocolError as exc:
                result_msg.success = False
                result_msg.error_code = "PERCEPTION_PROTOCOL_MISMATCH"
                result_msg.message = (
                    self._record_perception_protocol_error(exc)
                )
                self._abort_or_cancel(goal_handle)
                return result_msg

            if found and nut_pose is not None:
                if task_type == "SCAN_NUT1":
                    self.scanned_nut1_pose = nut_pose
                else:
                    self.scanned_nut2_pose = nut_pose

                self.get_logger().info(
                    f" ★ [{task_type} 좌표 저장 완료] "
                    f"X: {nut_pose.pose.position.x:.4f}, "
                    f"Y: {nut_pose.pose.position.y:.4f}, "
                    f"Z: {nut_pose.pose.position.z:.4f}"
                )

                result_msg.success = True
                result_msg.message = f"너트 스캔 및 비전 좌표 취득 성공 ({msg})"
                goal_handle.succeed()
            else:
                if goal_handle.is_cancel_requested:
                    result_msg.success = False
                    result_msg.error_code = f"{task_type}_CANCELED"
                    result_msg.message = "너트 스캔이 취소됐습니다."
                    goal_handle.canceled()
                    return result_msg
                self.get_logger().error(f" -> 너트 비전 검출 실패: {msg}")
                result_msg.success = False
                result_msg.error_code = "NUT_VISION_FAILED"
                result_msg.message = f"너트 스캔 좌표 취득 실패: {msg}"
                self._abort_or_cancel(goal_handle)

            return result_msg

        # ---------------------------------------------------------------------
        # [Task 8] PICK_NUT1 / PICK_NUT2 (너트 물리 파지 및 들어올리기)
        # ---------------------------------------------------------------------
        elif task_type in ("PICK_NUT1", "PICK_NUT2"):
            nut_pose = (
                self.scanned_nut1_pose
                if task_type == "PICK_NUT1"
                else self.scanned_nut2_pose
            )

            if nut_pose is None:
                self.get_logger().error(f" -> [{task_type}] 스캔된 너트 좌표가 없습니다! 먼저 SCAN_NUT을 수행하세요.")
                result_msg.success = False
                result_msg.error_code = "NO_SCANNED_NUT_POSE"
                result_msg.message = "스캔된 너트 좌표가 존재하지 않습니다."
                self._abort_or_cancel(goal_handle)
                return result_msg

            # Isaac Sim 목표 좌표 및 파지 명령 퍼블리시
            self.pub_target_pose.publish(nut_pose)

            cmd_msg = String()
            cmd_msg.data = task_type
            self.pub_task_command.publish(cmd_msg)

            success = self.wait_for_isaac_completion(goal_handle, feedback_msg)

            if success:
                result_msg.success = True
                result_msg.message = "너트 파지 및 들어올리기 성공"
                goal_handle.succeed()
            else:
                result_msg.success = False
                result_msg.error_code = "ISAAC_FAILED"
                result_msg.message = f"Isaac Sim 완료 실패 (Status: {self.isaac_status})"
                self._abort_or_cancel(goal_handle)

            return result_msg

        # ---------------------------------------------------------------------
        # [Task 9] ASSEMBLE_NUT1 / ASSEMBLE_NUT2 (너트 Screwing 체결 명령 중계)
        # ---------------------------------------------------------------------
        elif task_type in ("ASSEMBLE_NUT1", "ASSEMBLE_NUT2"):
            self.get_logger().info(f" -> [{task_type}] Isaac Sim으로 너트 체결(Screwing) 명령 전송")

            cmd_msg = String()
            cmd_msg.data = task_type
            self.pub_task_command.publish(cmd_msg)

            # Isaac Sim 완료 응답만 무제한 대기
            success = self.wait_for_isaac_completion(goal_handle, feedback_msg)

            if success:
                result_msg.success = True
                result_msg.message = "너트 체결 및 파지 해제 완료"
                goal_handle.succeed()
            else:
                result_msg.success = False
                result_msg.error_code = f"{task_type}_FAILED"
                result_msg.message = f"너트 체결 실패 (Status: {self.isaac_status})"
                self._abort_or_cancel(goal_handle)

            return result_msg

        # ---------------------------------------------------------------------
        # [기타] 미지원 Task
        # ---------------------------------------------------------------------
        else:
            result_msg.success = False
            result_msg.error_code = "UNSUPPORTED_TASK"
            result_msg.message = f"지원하지 않는 Task입니다: {task_type}"
            self._abort_or_cancel(goal_handle)
            return result_msg

    def _abort_or_cancel(self, goal_handle):
        """요청된 cancel을 abort로 잘못 보고하지 않는다."""
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            goal_handle.abort()

    def _wait_while_active(self, duration_sec, goal_handle):
        """짧은 publisher ordering 대기 중에도 cancel/shutdown을 확인한다."""
        deadline = time.monotonic() + duration_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if (
                goal_handle.is_cancel_requested
                or self._isaac_has_failed()
            ):
                return False
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
        return (
            rclpy.ok()
            and not goal_handle.is_cancel_requested
            and not self._isaac_has_failed()
        )

    def _record_perception_reset_ack(self, camera_name):
        """Record compatibility ACK telemetry; do not use it as a causal ID."""
        self._perception_reset_ack_counts[camera_name] += 1

    def _isaac_has_failed(self):
        """Return whether Isaac reported a terminal failure."""
        return (
            self.isaac_status is not None
            and "FAILURE" in self.isaac_status
        )

    def _publish_cancel_arm_task(self):
        """Release an Isaac task that may be paused at an internal phase."""
        cancel_msg = String()
        cancel_msg.data = "CANCEL_ARM_TASK"
        self.pub_task_command.publish(cancel_msg)

    @staticmethod
    def _perception_generation(message):
        """Parse ``[generation=N]`` metadata from a service response."""
        marker = "[generation="
        if not isinstance(message, str) or not message.startswith(marker):
            return None
        end = message.find("]", len(marker))
        if end < 0:
            return None
        try:
            generation = int(message[len(marker):end])
        except ValueError:
            return None
        return generation if generation >= 0 else None

    def _camera_name_for_grasp_client(self, client):
        if client is self.client_get_wrist_pose:
            return "wrist"
        if client is self.client_get_busbar_pose:
            return "busbar_cam"
        raise ValueError("알 수 없는 GetGraspPose client입니다")

    def _validate_perception_response_generation(
        self,
        camera_name,
        message,
    ):
        """Validate one concrete service response after a proven reset."""
        required = self._required_perception_generation[camera_name]
        if required is None:
            # Requests made without this ArmNode initiating a reset retain the
            # compatibility behavior used by diagnostics and narrow tests.
            return None
        observed = self._perception_generation(message)
        if observed is None:
            raise PerceptionProtocolError(
                f"{camera_name} perception protocol mismatch after reset: "
                "service response에 '[generation=N]' metadata가 없습니다 "
                f"(required={required}, message={message!r})"
            )
        if observed < required:
            raise PerceptionProtocolError(
                f"{camera_name} perception generation regression after reset: "
                f"required>={required}, observed={observed}; "
                "perception service restart/rollback이 의심됩니다"
            )
        return observed

    def _record_perception_protocol_error(self, error):
        """Store and log a typed perception protocol failure."""
        message = str(error)
        self._last_perception_reset_error = message
        self.get_logger().error(message)
        return message

    def _probe_perception_generation(self, camera_name, goal_handle):
        """Read one camera's generation or reject an incompatible server."""
        last_message = "perception generation 응답 없음"
        while (
            rclpy.ok()
            and not goal_handle.is_cancel_requested
            and not self._isaac_has_failed()
        ):
            if camera_name == "bolt_cam":
                response, last_message = self._call_bolt_pair_once(
                    goal_handle)
                if response is not None:
                    last_message = response.message
            else:
                client = (
                    self.client_get_wrist_pose
                    if camera_name == "wrist"
                    else self.client_get_busbar_pose
                )
                response, last_message = (
                    self._call_grasp_pose_response_once(
                        client,
                        "__reset_generation_probe__",
                        goal_handle,
                    )
                )

            if response is None:
                self.get_logger().warn(
                    f"{camera_name} perception generation 응답 재시도: "
                    f"{last_message}",
                    throttle_duration_sec=2.0,
                )
                if not self._wait_while_active(
                    self.PERCEPTION_RETRY_INTERVAL_SEC,
                    goal_handle,
                ):
                    break
                continue

            generation = self._perception_generation(response.message)
            if generation is not None:
                return generation

            raise PerceptionProtocolError(
                f"{camera_name} perception protocol mismatch: "
                "service response에 '[generation=N]' metadata가 없습니다 "
                f"(message={response.message!r})"
            )
        return None

    def _reset_perception_cache(self, publisher, goal_handle):
        """Reset one cache and prove the generation advanced via its service."""
        if publisher is self.pub_wrist_reset:
            camera_name = "wrist"
        elif publisher is self.pub_busbar_reset:
            camera_name = "busbar_cam"
        elif publisher is self.pub_bolt_reset:
            camera_name = "bolt_cam"
        else:
            raise ValueError("알 수 없는 perception reset publisher입니다")

        self._last_perception_reset_error = None
        try:
            baseline_generation = self._probe_perception_generation(
                camera_name,
                goal_handle,
            )
        except PerceptionProtocolError as exc:
            self._last_perception_reset_error = str(exc)
            self.get_logger().error(str(exc))
            return False
        if baseline_generation is None:
            return False

        publisher.publish(Empty())
        while (
            rclpy.ok()
            and not goal_handle.is_cancel_requested
            and not self._isaac_has_failed()
        ):
            try:
                observed_generation = self._probe_perception_generation(
                    camera_name,
                    goal_handle,
                )
            except PerceptionProtocolError as exc:
                self._last_perception_reset_error = str(exc)
                self.get_logger().error(str(exc))
                return False
            if observed_generation is None:
                break
            if observed_generation > baseline_generation:
                self._required_perception_generation[camera_name] = (
                    observed_generation
                )
                return True
            self.get_logger().info(
                f"{camera_name} perception reset 처리 대기 "
                f"(generation={observed_generation})",
                throttle_duration_sec=2.0,
            )
            if not self._wait_while_active(
                self.PERCEPTION_RETRY_INTERVAL_SEC,
                goal_handle,
            ):
                break
        return False

    @staticmethod
    def _pose_stamp_ns(pose):
        stamp = pose.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def _pose_position_is_finite(pose):
        position = pose.pose.position
        return all(
            math.isfinite(value)
            for value in (position.x, position.y, position.z)
        )

    def wait_for_isaac_completion(self, goal_handle, feedback_msg) -> bool:
        """성공·실패·cancel·shutdown까지 타임아웃 없이 대기한다."""
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                self._publish_cancel_arm_task()
                return False

            feedback_msg.sub_phase = self.isaac_phase
            feedback_msg.progress_pct = float(self.isaac_progress)
            goal_handle.publish_feedback(feedback_msg)

            if self.isaac_status is not None:
                if self.isaac_status == "SUCCESS":
                    return True
                if "FAILURE" in self.isaac_status:
                    return False

            time.sleep(0.1)

        return False

    def wait_for_isaac_phase(
        self,
        expected_phase,
        goal_handle,
        feedback_msg,
    ) -> bool:
        """Wait indefinitely and cancelably for an Isaac handshake phase."""
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                return False

            feedback_msg.sub_phase = self.isaac_phase
            feedback_msg.progress_pct = float(self.isaac_progress)
            goal_handle.publish_feedback(feedback_msg)

            if self._isaac_has_failed():
                return False
            if self.isaac_status == "SUCCESS":
                self.get_logger().error(
                    f"'{expected_phase}' 전에 Isaac SUCCESS를 수신했습니다."
                )
                return False
            if self.isaac_phase == expected_phase:
                return True

            time.sleep(0.05)

        return False

    def _call_grasp_pose_response_once(
        self,
        client,
        target_label,
        goal_handle,
    ):
        """Return one concrete response while preserving transport failures."""
        while (
            rclpy.ok()
            and not goal_handle.is_cancel_requested
            and not self._isaac_has_failed()
        ):
            if client.wait_for_service(timeout_sec=1.0):
                break
            self.get_logger().warn(
                f"'{target_label}' GetGraspPose 서비스 준비 대기 중...",
                throttle_duration_sec=2.0,
            )
        else:
            return None, "Action cancel 또는 ROS 종료"

        req = GetGraspPose.Request()
        req.label = target_label
        try:
            future = client.call_async(req)
        except Exception as exc:
            return None, f"GetGraspPose 호출 시작 실패: {exc}"
        while (
            rclpy.ok()
            and not goal_handle.is_cancel_requested
            and not self._isaac_has_failed()
        ):
            if future.done():
                try:
                    response = future.result()
                except Exception as exc:  # service transport failure: retryable
                    return None, f"GetGraspPose 호출 실패: {exc}"
                if response is None:
                    return None, "GetGraspPose 응답 없음"
                return response, response.message
            time.sleep(0.05)
        return None, (
            "Action cancel"
            if goal_handle.is_cancel_requested
            else (
                "Isaac failure"
                if self._isaac_has_failed()
                else "ROS 종료"
            )
        )

    def _call_grasp_pose(self, client, target_label, goal_handle):
        """한 번의 GetGraspPose 응답을 cancel 가능한 방식으로 기다린다."""
        response, message = self._call_grasp_pose_response_once(
            client,
            target_label,
            goal_handle,
        )
        if response is None:
            return False, None, message
        self._validate_perception_response_generation(
            self._camera_name_for_grasp_client(client),
            response.message,
        )
        return (
            bool(response.found),
            response.pose if response.found else None,
            response.message,
        )

    def request_grasp_pose_until_found(
        self,
        client,
        target_label,
        goal_handle,
    ):
        """perception의 post-reset barrier가 준비될 때까지 무제한 재시도한다."""
        last_message = f"'{target_label}' 검출 대기"
        while (
            rclpy.ok()
            and not goal_handle.is_cancel_requested
            and not self._isaac_has_failed()
        ):
            found, pose, message = self._call_grasp_pose(
                client, target_label, goal_handle
            )
            last_message = message
            if found and pose is not None:
                if not self._pose_position_is_finite(pose):
                    last_message = "검출 좌표에 NaN/inf가 있습니다"
                else:
                    return True, pose, message

            self.get_logger().warn(
                f"'{target_label}' GetGraspPose 대기: {last_message}",
                throttle_duration_sec=2.0,
            )
            if not self._wait_while_active(
                self.PERCEPTION_RETRY_INTERVAL_SEC, goal_handle
            ):
                break

        return False, None, (
            "Action cancel로 GetGraspPose 재시도 중단"
            if goal_handle.is_cancel_requested
            else (
                "Isaac failure로 GetGraspPose 재시도 중단"
                if self._isaac_has_failed()
                else "ROS 종료로 GetGraspPose 재시도 중단"
            )
        )

    def wait_for_wrist_busbar_confirmation(
        self,
        fixed_pose,
        goal_handle,
    ):
        """고정캠 대상 근처의 서로 다른 최신 wrist 표본 2개를 확인한다."""
        fixed = fixed_pose.pose.position
        previous_pose = None
        latest_stamp_ns = None
        consecutive = 0
        last_message = "reset 뒤 새 wrist 버스바 검출을 기다리는 중"

        while (
            rclpy.ok()
            and not goal_handle.is_cancel_requested
            and not self._isaac_has_failed()
        ):
            found, wrist_pose, message = self._call_grasp_pose(
                self.client_get_wrist_pose,
                "busbar",
                goal_handle,
            )
            if not found or wrist_pose is None:
                last_message = message
            elif not self._pose_position_is_finite(wrist_pose):
                consecutive = 0
                previous_pose = None
                last_message = "wrist 좌표에 NaN/inf가 있습니다"
            else:
                stamp_ns = self._pose_stamp_ns(wrist_pose)
                if (
                    latest_stamp_ns is not None
                    and stamp_ns <= latest_stamp_ns
                ):
                    last_message = "중복·역행 wrist 표본을 무시했습니다"
                else:
                    latest_stamp_ns = stamp_ns
                    detected = wrist_pose.pose.position
                    target_xy = math.hypot(
                        detected.x - fixed.x,
                        detected.y - fixed.y,
                    )
                    target_z = abs(detected.z - fixed.z)
                    if target_xy > self.WRIST_BUSBAR_MAX_TARGET_XY_M:
                        consecutive = 0
                        previous_pose = None
                        last_message = (
                            f"wrist 후보 XY가 고정캠 대상에서 "
                            f"{target_xy:.3f}m 떨어져 있습니다"
                        )
                    elif target_z > self.WRIST_BUSBAR_MAX_TARGET_Z_M:
                        consecutive = 0
                        previous_pose = None
                        last_message = (
                            f"wrist 후보 Z가 고정캠 대상과 "
                            f"{target_z:.3f}m 차이 납니다"
                        )
                    else:
                        if previous_pose is None:
                            consecutive = 1
                        else:
                            previous = previous_pose.pose.position
                            sample_shift = math.sqrt(
                                (detected.x - previous.x) ** 2
                                + (detected.y - previous.y) ** 2
                                + (detected.z - previous.z) ** 2
                            )
                            consecutive = (
                                consecutive + 1
                                if sample_shift
                                <= self.WRIST_BUSBAR_MAX_SAMPLE_SHIFT_M
                                else 1
                            )
                            if consecutive == 1:
                                last_message = (
                                    f"wrist 표본 이동량 {sample_shift:.3f}m로 "
                                    "연속 표본을 다시 시작합니다"
                                )
                        previous_pose = wrist_pose
                        if (
                            consecutive
                            >= self.WRIST_BUSBAR_REQUIRED_SAMPLES
                        ):
                            return (
                                True,
                                wrist_pose,
                                f"XY {target_xy:.3f}m, Z {target_z:.3f}m, "
                                f"연속 {consecutive}표본",
                            )
                        last_message = (
                            f"유효 wrist 표본 {consecutive}/"
                            f"{self.WRIST_BUSBAR_REQUIRED_SAMPLES}"
                        )

            self.get_logger().warn(
                f"wrist 버스바 확인 대기: {last_message}",
                throttle_duration_sec=2.0,
            )
            if not self._wait_while_active(
                self.PERCEPTION_RETRY_INTERVAL_SEC, goal_handle
            ):
                break

        return False, None, (
            "Action cancel로 wrist 검증 중단"
            if goal_handle.is_cancel_requested
            else (
                "Isaac failure로 wrist 검증 중단"
                if self._isaac_has_failed()
                else "ROS 종료로 wrist 검증 중단"
            )
        )

    def _call_bolt_pair_once(self, goal_handle):
        """Call GetBoltPair once while remaining cancel/shutdown aware."""
        while (
            rclpy.ok()
            and not goal_handle.is_cancel_requested
            and not self._isaac_has_failed()
        ):
            if self.client_get_bolt_pair.wait_for_service(
                timeout_sec=1.0
            ):
                break
            self.get_logger().warn(
                "GetBoltPair 서비스 준비 대기 중...",
                throttle_duration_sec=2.0,
            )
        else:
            return None, (
                "Action cancel 또는 Isaac failure 또는 ROS 종료"
            )

        try:
            future = self.client_get_bolt_pair.call_async(
                GetBoltPair.Request()
            )
        except Exception as exc:
            return None, f"GetBoltPair 호출 시작 실패: {exc}"

        while (
            rclpy.ok()
            and not goal_handle.is_cancel_requested
            and not self._isaac_has_failed()
        ):
            if future.done():
                try:
                    response = future.result()
                except Exception as exc:
                    return None, f"GetBoltPair 호출 실패: {exc}"
                if response is None:
                    return None, "GetBoltPair 응답 없음"
                return response, response.message
            time.sleep(0.05)
        return None, (
            "Action cancel"
            if goal_handle.is_cancel_requested
            else (
                "Isaac failure"
                if self._isaac_has_failed()
                else "ROS 종료"
            )
        )

    def request_bolt_pair_midpoint_async(
        self,
        goal_handle,
    ):
        """고정 bolt camera의 새 볼트쌍이 올 때까지 무제한 재시도한다."""
        last_message = "GetBoltPair 검출 대기"
        while (
            rclpy.ok()
            and not goal_handle.is_cancel_requested
            and not self._isaac_has_failed()
        ):
            if not self.client_get_bolt_pair.wait_for_service(
                timeout_sec=1.0
            ):
                self.get_logger().warn(
                    "GetBoltPair 서비스 준비 대기 중...",
                    throttle_duration_sec=2.0,
                )
                continue

            try:
                future = self.client_get_bolt_pair.call_async(
                    GetBoltPair.Request()
                )
            except Exception as exc:
                last_message = f"GetBoltPair 호출 시작 실패: {exc}"
                if not self._wait_while_active(
                    self.PERCEPTION_RETRY_INTERVAL_SEC, goal_handle
                ):
                    break
                continue
            while (
                rclpy.ok()
                and not goal_handle.is_cancel_requested
                and not self._isaac_has_failed()
            ):
                if future.done():
                    try:
                        response = future.result()
                    except Exception as exc:
                        response = None
                        last_message = f"GetBoltPair 호출 실패: {exc}"
                    break
                time.sleep(0.05)
            else:
                break

            if response is not None:
                self._validate_perception_response_generation(
                    "bolt_cam",
                    response.message,
                )

            if (
                response is not None
                and response.found
            ):
                pose_a_msg = response.pose_a
                pose_b_msg = response.pose_b
                stamps = (
                    self._pose_stamp_ns(pose_a_msg),
                    self._pose_stamp_ns(pose_b_msg),
                )
                if (
                    self._pose_position_is_finite(pose_a_msg)
                    and self._pose_position_is_finite(pose_b_msg)
                    and stamps[0] >= 0
                    and stamps[1] >= 0
                ):
                    pose_a = pose_a_msg.pose.position
                    pose_b = pose_b_msg.pose.position
                    midpoint_pose = PoseStamped()
                    midpoint_pose.header = pose_a_msg.header
                    midpoint_pose.pose.position.x = (
                        pose_a.x + pose_b.x
                    ) / 2.0
                    midpoint_pose.pose.position.y = (
                        pose_a.y + pose_b.y
                    ) / 2.0
                    midpoint_pose.pose.position.z = (
                        pose_a.z + pose_b.z
                    ) / 2.0
                    midpoint_pose.pose.orientation = (
                        pose_a_msg.pose.orientation
                    )
                    return True, midpoint_pose, response.message
                last_message = (
                    "볼트쌍 좌표가 유효하지 않거나 reset 이전 표본입니다"
                )
            elif response is not None:
                last_message = response.message

            self.get_logger().warn(
                f"GetBoltPair 대기: {last_message}",
                throttle_duration_sec=2.0,
            )
            if not self._wait_while_active(
                self.PERCEPTION_RETRY_INTERVAL_SEC, goal_handle
            ):
                break

        return False, None, (
            "Action cancel로 GetBoltPair 재시도 중단"
            if goal_handle.is_cancel_requested
            else (
                "Isaac failure로 GetBoltPair 재시도 중단"
                if self._isaac_has_failed()
                else "ROS 종료로 GetBoltPair 재시도 중단"
            )
        )


def main(args=None):
    """Run the arm action server until shutdown."""
    rclpy.init(args=args)
    node = ArmNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("ArmNode 종료")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

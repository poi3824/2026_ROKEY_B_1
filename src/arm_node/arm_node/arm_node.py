#!/usr/bin/env python3
"""
arm_node.py - ROS 2 Arm Control Node (MultiThreaded Executor & Reentrant Group)
- BehaviorNode의 /execute_arm_task Action 수신
- [지원 Task]
  1. SCAN_BATTERY        : 배터리 스캔 위치로 이동 후 Perception 노드로부터 두 볼트의 중앙 좌표 수신 및 저장
  2. SCAN_BUSBAR         : 고정캠 선택 좌표로 이동 후 손목캠으로 동일 버스바 여부 확인
  3. PICK_BUSBAR         : AMR relay가 latch한 고정캠 좌표로 Isaac Sim 파지 명령 중계
  4. MOVE_BATTERY_CENTER : 스캔 시 저장한 배터리 중점 좌표를 Isaac Sim으로 퍼블리시 및 이동 명령 중계
  5. FINE_ALIGNMENT       : Isaac Sim 및 비전 노드의 정밀 1픽셀 오차 보정 명령 중계
  6. ASSEMBLE_BUSBAR      : Isaac Sim으로 버스바 하강 체결 및 그리퍼 해제 명령 중계 (추가됨)
  7. SCAN_NUT1/SCAN_NUT2       : 너트 스캔 위치 이동 후 Perception 노드로부터 너트 좌표 수신 및 저장 (신규)
  8. PICK_NUT1/PICK_NUT2       : 스캔된 너트 좌표로 Isaac Sim 파지 명령 중계 (신규)
  9. ASSEMBLE_NUT1/ASSEMBLE_NUT2 : Isaac Sim으로 너트 Screwing 체결 명령 중계 (신규)
"""

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float32

# Action 및 Custom Interfaces
from fms_interfaces.action import ExecuteArmTask
from fms_interfaces.srv import GetGraspPose, GetBoltPair
from fms_interfaces.msg import NutPose


class ArmNode(Node):
    def __init__(self):
        super().__init__('arm_node')
        self.get_logger().info("===========================================")
        self.get_logger().info(" ArmNode 활성화 (배터리/버스바 스캔, 파지, 이동, 정밀 보정 및 체결 지원)")
        self.get_logger().info("===========================================")

        # Reentrant Callback Group 적용 (교착 상태 방지)
        self.cb_group = ReentrantCallbackGroup()

        # 1. Action Server 생성
        self._action_server = ActionServer(
            self,
            ExecuteArmTask,
            '/execute_arm_task',
            execute_callback=self.execute_callback,
            callback_group=self.cb_group
        )

        # 2. Perception Node 서비스 클라이언트들
        self.client_get_nut_pose = self.create_client(
            GetGraspPose, '/perception/get_grasp_pose', callback_group=self.cb_group
        )
        self.client_get_bolt_pair = self.create_client(
            GetBoltPair, '/bolt_cam/perception/get_bolt_pair', callback_group=self.cb_group
        )

        # 3. Perception Node 토픽 백업 구독
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
        selected_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.sub_selected_battery = self.create_subscription(
            PoseStamped,
            '/amr_physics/selected_battery_midpoint',
            self._on_selected_battery_midpoint,
            selected_qos,
            callback_group=self.cb_group,
        )
        self.sub_selected_busbar = self.create_subscription(
            PoseStamped,
            '/amr_physics/selected_busbar_pose',
            self._on_selected_busbar_pose,
            selected_qos,
            callback_group=self.cb_group,
        )
        self.sub_wrist_busbar = self.create_subscription(
            PoseStamped,
            '/wrist/vision/busbar_grasp_pose',
            self._on_wrist_busbar_pose,
            10,
            callback_group=self.cb_group,
        )
        self.declare_parameter(
            'use_amr_selected_battery_midpoint',
            False,
        )
        self.declare_parameter('debug_no_timeouts', False)
        self.declare_parameter(
            'wrist_busbar_required_samples',
            2,
        )
        self.declare_parameter(
            'wrist_busbar_observation_sec',
            2.0,
        )
        self.declare_parameter(
            'wrist_busbar_max_target_distance_m',
            0.20,
        )
        self.declare_parameter(
            'wrist_busbar_max_target_z_distance_m',
            0.10,
        )
        self.declare_parameter(
            'wrist_busbar_max_sample_shift_m',
            0.05,
        )
        self._use_amr_selected_battery_midpoint = bool(
            self.get_parameter(
                'use_amr_selected_battery_midpoint'
            ).value
        )
        self._debug_no_timeouts = bool(
            self.get_parameter('debug_no_timeouts').value
        )
        self._wrist_busbar_required_samples = int(
            self.get_parameter(
                'wrist_busbar_required_samples'
            ).value
        )
        self._wrist_busbar_observation_sec = float(
            self.get_parameter(
                'wrist_busbar_observation_sec'
            ).value
        )
        self._wrist_busbar_max_target_distance_m = float(
            self.get_parameter(
                'wrist_busbar_max_target_distance_m'
            ).value
        )
        self._wrist_busbar_max_target_z_distance_m = float(
            self.get_parameter(
                'wrist_busbar_max_target_z_distance_m'
            ).value
        )
        self._wrist_busbar_max_sample_shift_m = float(
            self.get_parameter(
                'wrist_busbar_max_sample_shift_m'
            ).value
        )
        if self._wrist_busbar_required_samples < 1:
            raise ValueError('wrist_busbar_required_samples는 1 이상이어야 합니다')
        positive_wrist_values = (
            self._wrist_busbar_observation_sec,
            self._wrist_busbar_max_target_distance_m,
            self._wrist_busbar_max_target_z_distance_m,
            self._wrist_busbar_max_sample_shift_m,
        )
        if not all(
            math.isfinite(value) and value > 0.0
            for value in positive_wrist_values
        ):
            raise ValueError('wrist busbar 검증 파라미터는 유한한 양수여야 합니다')

        # 내부 상태 및 저장 변수
        self.isaac_status = None
        self.isaac_phase = ""
        self.isaac_progress = 0.0

        # 배터리 스캔 시 저장할 두 볼트의 중앙 좌표 (PoseStamped)
        self.scanned_battery_midpoint = None
        self.latest_amr_selected_battery_midpoint = None

        # 버스바 스캔 시 저장할 버스바 파지 좌표 (PoseStamped)
        self.scanned_busbar_pose = None
        self.latest_amr_selected_busbar_pose = None
        self.latest_wrist_busbar_pose = None
        self._wrist_busbar_sequence = 0

        # 너트 스캔 시 저장할 너트 1번 / 2번 좌표 (PoseStamped) (신규 추가)
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

    def _on_selected_battery_midpoint(self, msg: PoseStamped):
        if msg.header.frame_id != 'world':
            self.get_logger().warn(
                '선택된 배터리 중점 frame이 world가 아니어서 무시합니다')
            return
        self.latest_amr_selected_battery_midpoint = msg

    def _on_selected_busbar_pose(self, msg: PoseStamped):
        if msg.header.frame_id != 'world':
            self.get_logger().warn(
                '선택된 버스바 frame이 world가 아니어서 무시합니다')
            return
        position = msg.pose.position
        if not all(
            math.isfinite(value)
            for value in (position.x, position.y, position.z)
        ):
            self.get_logger().warn(
                '선택된 버스바 좌표에 NaN/inf가 있어 무시합니다')
            return
        self.latest_amr_selected_busbar_pose = msg

    def _on_wrist_busbar_pose(self, msg: PoseStamped):
        if msg.header.frame_id != 'world':
            return
        position = msg.pose.position
        if not all(
            math.isfinite(value)
            for value in (position.x, position.y, position.z)
        ):
            return
        self.latest_wrist_busbar_pose = msg
        self._wrist_busbar_sequence += 1

    # =========================================================================
    # Action Callback
    # =========================================================================
    def execute_callback(self, goal_handle):
        task_type = goal_handle.request.task_type
        self.get_logger().info(f"\n================ [Action Goal Received: {task_type}] ================")
        
        feedback_msg = ExecuteArmTask.Feedback()
        result_msg = ExecuteArmTask.Result()
        self.isaac_status = None

        # ---------------------------------------------------------------------
        # [안전 Task] STOW_ARM (AMR 주행 전에 로봇팔을 초기 관절 자세로 복귀)
        # ---------------------------------------------------------------------
        if task_type == "STOW_ARM":
            self.get_logger().info(
                " -> [STOW_ARM] AMR 주행용 안전 관절 자세로 복귀")
            cmd_msg = String()
            cmd_msg.data = "STOW_ARM"
            self.pub_task_command.publish(cmd_msg)

            success = self.wait_for_isaac_completion(
                goal_handle,
                feedback_msg,
            )
            if success:
                result_msg.success = True
                result_msg.message = "AMR 주행용 팔 안전 자세 복귀 완료"
                goal_handle.succeed()
            else:
                result_msg.success = False
                result_msg.error_code = "STOW_ARM_FAILED"
                result_msg.message = (
                    f"팔 안전 자세 복귀 실패 (Status: {self.isaac_status})"
                )
                goal_handle.abort()
            return result_msg

        # ---------------------------------------------------------------------
        # [Task 1] SCAN_BATTERY (배터리 스캔 지점 이동 & 두 볼트 중앙 좌표 저장)
        # ---------------------------------------------------------------------
        if task_type == "SCAN_BATTERY":
            self.get_logger().info(" -> [SCAN_BATTERY] Isaac Sim으로 스캔 이동 명령 전송")
            
            cmd_msg = String()
            cmd_msg.data = "SCAN_BATTERY"
            self.pub_task_command.publish(cmd_msg)

            # 1. Isaac Sim이 스캔 고도로 이동 완료할 때까지 대기
            success = self.wait_for_isaac_completion(goal_handle, feedback_msg)
            
            if not success:
                result_msg.success = False
                result_msg.error_code = "SCAN_BATTERY_FAILED"
                result_msg.message = f"배터리 스캔 위치 이동 실패 (Status: {self.isaac_status})"
                goal_handle.abort()
                return result_msg

            # 2. 이동 완료 후 Perception 노드에 볼트 페어 비전 좌표 요청
            self.get_logger().info(" -> [SCAN_BATTERY] 배터리 스캔 위치 도착 완료. 비전 노드에 볼트 쌍 좌표 요청...")
            if self._use_amr_selected_battery_midpoint:
                midpoint_pose = self.latest_amr_selected_battery_midpoint
                found = midpoint_pose is not None
                msg = (
                    'AMR 주행이 선택한 동일 볼트쌍 중점'
                    if found
                    else 'AMR 선택 볼트쌍 중점이 없습니다'
                )
            else:
                found, midpoint_pose, msg = (
                    self.request_bolt_pair_midpoint_async(
                        timeout_sec=5.0
                    )
                )

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
                self.get_logger().error(f" -> 볼트 쌍 비전 검출 실패: {msg}")
                result_msg.success = False
                result_msg.error_code = "BOLT_PAIR_VISION_FAILED"
                result_msg.message = f"볼트 쌍 중점 취득 실패: {msg}"
                goal_handle.abort()

            return result_msg

        # ---------------------------------------------------------------------
        # [Task 2] SCAN_BUSBAR (버스바 스캔 지점 이동 & 버스바 비전 좌표 저장)
        # ---------------------------------------------------------------------
        elif task_type == "SCAN_BUSBAR":
            selected_pose = self.latest_amr_selected_busbar_pose
            if selected_pose is None:
                result_msg.success = False
                result_msg.error_code = "NO_SELECTED_BUSBAR_POSE"
                result_msg.message = (
                    "AMR relay가 선택해 주행에 사용한 버스바 좌표가 없습니다"
                )
                goal_handle.abort()
                return result_msg

            self.scanned_busbar_pose = None
            attempt = 0
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    result_msg.success = False
                    result_msg.error_code = "SCAN_BUSBAR_CANCELED"
                    result_msg.message = "버스바 손목 스캔이 취소됐습니다"
                    goal_handle.canceled()
                    return result_msg

                target = selected_pose.pose.position
                self.get_logger().info(
                    " -> [SCAN_BUSBAR] AMR이 선택한 동일 버스바를 "
                    f"손목 카메라 탐색 {attempt + 1}회차: "
                    f"({target.x:.4f}, {target.y:.4f}, {target.z:.4f})"
                )
                self.pub_target_pose.publish(selected_pose)
                # 서로 다른 ROS topic의 target/command 순서를 executor가 확실히
                # 처리하도록 짧게 양보한다. 좌표를 새로 계산하거나 폴백하지 않는다.
                time.sleep(0.1)

                self.isaac_status = None
                cmd_msg = String()
                cmd_msg.data = f"SCAN_BUSBAR:{attempt}"
                self.pub_task_command.publish(cmd_msg)
                success = self.wait_for_isaac_completion(
                    goal_handle,
                    feedback_msg,
                )
                if not success:
                    self.get_logger().warn(
                        " -> 버스바 스캔 자세가 수렴하지 않았습니다. "
                        "다음 제한 범위 탐색 자세를 계속 시도합니다."
                    )
                    attempt += 1
                    continue

                # 이동 중 보였던 과거 프레임은 성공으로 쓰지 않는다. 스캔 자세에
                # 도착한 뒤 손목 카메라가 새로 발행한 연속 표본만 확인한다.
                start_sequence = self._wrist_busbar_sequence
                found, busbar_pose, msg = self._wait_for_wrist_busbar(
                    selected_pose,
                    start_sequence,
                    goal_handle,
                )
                if found:
                    # 손목 카메라는 동일한 버스바가 시야에 들어왔는지만 확인한다.
                    # PICK XY는 관측 시점/카메라가 바뀌며 옆으로 이동하지 않도록
                    # AMR 주행에 사용한 고정 카메라 snapshot을 그대로 유지한다.
                    self.scanned_busbar_pose = selected_pose
                    self.get_logger().info(
                        " ★ [손목 버스바 확인 완료] "
                        f"wrist=({busbar_pose.pose.position.x:.4f}, "
                        f"{busbar_pose.pose.position.y:.4f}, "
                        f"{busbar_pose.pose.position.z:.4f}), "
                        "PICK 고정캠="
                        f"({selected_pose.pose.position.x:.4f}, "
                        f"{selected_pose.pose.position.y:.4f}, "
                        f"{selected_pose.pose.position.z:.4f})"
                    )
                    result_msg.success = True
                    result_msg.message = (
                        "손목 카메라 동일 버스바 확인 성공; "
                        f"PICK은 고정캠 좌표 사용 ({msg})"
                    )
                    goal_handle.succeed()
                    return result_msg

                self.get_logger().warn(
                    f" -> 손목 카메라에 선택 버스바가 아직 없습니다: {msg}; "
                    "다음 탐색 자세를 계속 시도합니다"
                )
                attempt += 1

            result_msg.success = False
            result_msg.error_code = "SCAN_BUSBAR_INTERRUPTED"
            result_msg.message = "ROS 종료로 버스바 손목 스캔을 중단했습니다"
            goal_handle.abort()
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
                    "손목 카메라로 동일 대상을 확인한 고정캠 버스바 "
                    "snapshot이 없습니다."
                )
                goal_handle.abort()
                return result_msg
            self.get_logger().info(
                " -> [PICK_BUSBAR] AMR 주행에 사용한 고정캠 버스바 "
                f"좌표를 사용합니다: ({busbar_pose.pose.position.x:.4f}, "
                f"{busbar_pose.pose.position.y:.4f}, "
                f"{busbar_pose.pose.position.z:.4f})")

            # Isaac Sim 목표 좌표 및 파지 명령 퍼블리시
            self.pub_target_pose.publish(busbar_pose)
            # 서로 다른 ROS topic의 target/command 처리 순서를 보장한다.
            time.sleep(0.1)

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
                result_msg.error_code = "ISAAC_FAILED_OR_TIMEOUT"
                result_msg.message = f"Isaac Sim 완료 실패 (Status: {self.isaac_status})"
                goal_handle.abort()

            return result_msg

        # ---------------------------------------------------------------------
        # [Task 4] MOVE_BATTERY_CENTER (저장된 배터리 중점 좌표 상공으로 이동)
        # ---------------------------------------------------------------------
        elif task_type == "MOVE_BATTERY_CENTER":
            if self.scanned_battery_midpoint is None:
                self.get_logger().error(" -> [MOVE_BATTERY_CENTER] 스캔 저장된 배터리 볼트 중점 좌표가 없습니다!")
                result_msg.success = False
                result_msg.error_code = "NO_SCANNED_BATTERY_MIDPOINT"
                result_msg.message = "저장된 배터리 볼트 중점 좌표가 존재하지 않습니다."
                goal_handle.abort()
                return result_msg

            self.get_logger().info(" -> [MOVE_BATTERY_CENTER] 스캔 시 저장한 배터리 중점 좌표를 Isaac Sim으로 퍼블리시")
            self.get_logger().info(
                f"    [목표 좌표] X: {self.scanned_battery_midpoint.pose.position.x:.4f}, "
                f"Y: {self.scanned_battery_midpoint.pose.position.y:.4f}, "
                f"Z: {self.scanned_battery_midpoint.pose.position.z:.4f}"
            )

            # 1. 저장된 배터리 볼트 중점 좌표 퍼블리시
            self.pub_target_pose.publish(self.scanned_battery_midpoint)

            # 2. Isaac Sim으로 이동 명령 전달
            cmd_msg = String()
            cmd_msg.data = "MOVE_BATTERY_CENTER"
            self.pub_task_command.publish(cmd_msg)

            # 3. Isaac Sim 완료 대기
            success = self.wait_for_isaac_completion(goal_handle, feedback_msg)

            if success:
                result_msg.success = True
                result_msg.message = "배터리 볼트 중점 상공 이동 성공"
                goal_handle.succeed()
            else:
                result_msg.success = False
                result_msg.error_code = "MOVE_BATTERY_CENTER_FAILED"
                result_msg.message = f"배터리 중점 이동 실패 (Status: {self.isaac_status})"
                goal_handle.abort()

            return result_msg

        # ---------------------------------------------------------------------
        # [Task 5] FINE_ALIGNMENT (비전 기반 1픽셀 미세 오차 보정 중계)
        # ---------------------------------------------------------------------
        elif task_type == "FINE_ALIGNMENT":
            self.get_logger().info(" -> [FINE_ALIGNMENT] Isaac Sim으로 정밀 비전 오차 보정 시작 명령 전송")

            cmd_msg = String()
            cmd_msg.data = "FINE_ALIGNMENT"
            self.pub_task_command.publish(cmd_msg)

            # Isaac Sim이 비전 보정을 거쳐 ALIGNMENT_SUCCESS 신호를 보낼 때까지 대기 (타임아웃 45초 지정)
            success = self.wait_for_isaac_completion(goal_handle, feedback_msg, timeout_sec=45.0)

            if success:
                result_msg.success = True
                result_msg.message = "비전 기반 1픽셀 정밀 오차 보정 완료"
                goal_handle.succeed()
            else:
                result_msg.success = False
                result_msg.error_code = "FINE_ALIGNMENT_FAILED"
                result_msg.message = f"비전 정밀 오차 보정 실패 (Status: {self.isaac_status})"
                goal_handle.abort()

            return result_msg

        # ---------------------------------------------------------------------
        # 🔥 [Task 6] ASSEMBLE_BUSBAR (버스바 하강 체결 및 그리퍼 해제 중계)
        # ---------------------------------------------------------------------
        elif task_type == "ASSEMBLE_BUSBAR":
            self.get_logger().info(" -> [ASSEMBLE_BUSBAR] Isaac Sim으로 버스바 최종 체결 명령 전송")

            cmd_msg = String()
            cmd_msg.data = "ASSEMBLE_BUSBAR"
            self.pub_task_command.publish(cmd_msg)

            # Isaac Sim이 하강 체결 후 완료 신호를 보낼 때까지 대기
            success = self.wait_for_isaac_completion(goal_handle, feedback_msg, timeout_sec=30.0)

            if success:
                result_msg.success = True
                result_msg.message = "버스바 체결 및 파지 해제 완료"
                goal_handle.succeed()
            else:
                result_msg.success = False
                result_msg.error_code = "ASSEMBLE_BUSBAR_FAILED"
                result_msg.message = f"버스바 체결 실패 (Status: {self.isaac_status})"
                goal_handle.abort()

            return result_msg

        # ---------------------------------------------------------------------
        # 🔥 [Task 7] SCAN_NUT1 / SCAN_NUT2 (너트 스캔 지점 이동 & 비전 좌표 저장) (신규 추가)
        # ---------------------------------------------------------------------
        elif task_type in ("SCAN_NUT1", "SCAN_NUT2"):
            self.get_logger().info(f" -> [{task_type}] Isaac Sim으로 너트 스캔 이동 명령 전송")

            cmd_msg = String()
            cmd_msg.data = task_type
            self.pub_task_command.publish(cmd_msg)

            # 1. Isaac Sim이 너트 스캔 위치로 이동 완료할 때까지 대기
            success = self.wait_for_isaac_completion(goal_handle, feedback_msg)

            if not success:
                result_msg.success = False
                result_msg.error_code = f"{task_type}_FAILED"
                result_msg.message = f"너트 스캔 위치 이동 실패 (Status: {self.isaac_status})"
                goal_handle.abort()
                return result_msg

            # 2. 이동 완료 후 Perception 노드에 너트 좌표 요청
            self.get_logger().info(f" -> [{task_type}] 너트 스캔 위치 도착 완료. 비전 노드에 너트 좌표 요청...")
            found, nut_pose, msg = self.request_vision_pose_async("nut")

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
                self.get_logger().error(f" -> 너트 비전 검출 실패: {msg}")
                result_msg.success = False
                result_msg.error_code = "NUT_VISION_FAILED"
                result_msg.message = f"너트 스캔 좌표 취득 실패: {msg}"
                goal_handle.abort()

            return result_msg

        # ---------------------------------------------------------------------
        # 🔥 [Task 8] PICK_NUT1 / PICK_NUT2 (너트 물리 파지 및 들어올리기) (신규 추가)
        # ---------------------------------------------------------------------
        elif task_type in ("PICK_NUT1", "PICK_NUT2"):
            nut_pose = self.scanned_nut1_pose if task_type == "PICK_NUT1" else self.scanned_nut2_pose

            if nut_pose is None:
                self.get_logger().error(f" -> [{task_type}] 스캔된 너트 좌표가 없습니다! 먼저 SCAN_NUT을 수행하세요.")
                result_msg.success = False
                result_msg.error_code = "NO_SCANNED_NUT_POSE"
                result_msg.message = "스캔된 너트 좌표가 존재하지 않습니다."
                goal_handle.abort()
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
                result_msg.error_code = "ISAAC_FAILED_OR_TIMEOUT"
                result_msg.message = f"Isaac Sim 완료 실패 (Status: {self.isaac_status})"
                goal_handle.abort()

            return result_msg

        # ---------------------------------------------------------------------
        # 🔥 [Task 9] ASSEMBLE_NUT1 / ASSEMBLE_NUT2 (너트 Screwing 체결 명령 중계) (신규 추가)
        # ---------------------------------------------------------------------
        elif task_type in ("ASSEMBLE_NUT1", "ASSEMBLE_NUT2"):
            self.get_logger().info(f" -> [{task_type}] Isaac Sim으로 너트 체결(Screwing) 명령 전송")

            cmd_msg = String()
            cmd_msg.data = task_type
            self.pub_task_command.publish(cmd_msg)

            # Isaac Sim이 착좌/Screwing/Regrasp/이탈까지 완료 신호를 보낼 때까지 대기
            success = self.wait_for_isaac_completion(goal_handle, feedback_msg, timeout_sec=60.0)

            if success:
                result_msg.success = True
                result_msg.message = "너트 체결 및 파지 해제 완료"
                goal_handle.succeed()
            else:
                result_msg.success = False
                result_msg.error_code = f"{task_type}_FAILED"
                result_msg.message = f"너트 체결 실패 (Status: {self.isaac_status})"
                goal_handle.abort()

            return result_msg

        # ---------------------------------------------------------------------
        # [기타] 미지원 Task
        # ---------------------------------------------------------------------
        else:
            result_msg.success = False
            result_msg.error_code = "UNSUPPORTED_TASK"
            result_msg.message = f"지원하지 않는 Task입니다: {task_type}"
            goal_handle.abort()
            return result_msg

    def _wait_for_wrist_busbar(
        self,
        selected_pose,
        start_sequence,
        goal_handle,
    ):
        deadline = (
            time.monotonic() + self._wrist_busbar_observation_sec
        )
        observed_sequence = start_sequence
        consecutive = 0
        previous_pose = None
        last_reason = "스캔 자세 도착 뒤 새 wrist 검출이 없습니다"

        while rclpy.ok() and time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                return False, None, "Action cancel 요청"

            sequence = self._wrist_busbar_sequence
            pose = self.latest_wrist_busbar_pose
            if sequence <= observed_sequence or pose is None:
                time.sleep(0.05)
                continue
            observed_sequence = sequence

            selected = selected_pose.pose.position
            detected = pose.pose.position
            target_distance = math.hypot(
                detected.x - selected.x,
                detected.y - selected.y,
            )
            if (
                target_distance
                > self._wrist_busbar_max_target_distance_m
            ):
                consecutive = 0
                previous_pose = None
                last_reason = (
                    f"wrist 후보가 선택 대상에서 "
                    f"{target_distance:.3f}m 떨어져 있습니다"
                )
                continue

            target_z_distance = abs(detected.z - selected.z)
            if (
                target_z_distance
                > self._wrist_busbar_max_target_z_distance_m
            ):
                consecutive = 0
                previous_pose = None
                last_reason = (
                    f"wrist 후보 Z가 선택 대상과 "
                    f"{target_z_distance:.3f}m 차이 납니다"
                )
                continue

            if previous_pose is None:
                consecutive = 1
            else:
                previous = previous_pose.pose.position
                sample_shift = math.hypot(
                    detected.x - previous.x,
                    detected.y - previous.y,
                )
                if (
                    sample_shift
                    <= self._wrist_busbar_max_sample_shift_m
                ):
                    consecutive += 1
                else:
                    consecutive = 1
                    last_reason = (
                        f"wrist 연속 표본 이동량 "
                        f"{sample_shift:.3f}m가 큽니다"
                    )
            previous_pose = pose

            if consecutive >= self._wrist_busbar_required_samples:
                return (
                    True,
                    pose,
                    f"선택 대상과 {target_distance:.3f}m, "
                    f"연속 {consecutive}표본",
                )
            last_reason = (
                f"유효 wrist 표본 {consecutive}/"
                f"{self._wrist_busbar_required_samples}"
            )
            time.sleep(0.05)

        return False, None, last_reason

    def wait_for_isaac_completion(
        self,
        goal_handle,
        feedback_msg,
        timeout_sec: float = 30.0,
    ) -> bool:
        start_time = time.monotonic()
        while rclpy.ok():
            feedback_msg.sub_phase = self.isaac_phase
            feedback_msg.progress_pct = float(self.isaac_progress)
            goal_handle.publish_feedback(feedback_msg)

            if goal_handle.is_cancel_requested:
                return False

            if self.isaac_status is not None:
                if self.isaac_status == "SUCCESS":
                    return True
                elif "FAILURE" in self.isaac_status:
                    return False

            if (
                not self._debug_no_timeouts
                and time.monotonic() - start_time > timeout_sec
            ):
                self.get_logger().error("Isaac Sim 응답 시간 초과 (Timeout)")
                return False

            time.sleep(0.1)

        return False

    def request_bolt_pair_midpoint_async(self, timeout_sec: float = 5.0):
        """Perception 노드의 /perception/get_bolt_pair 서비스를 호출하여 두 볼트의 중앙 좌표 계산"""
        if not self.client_get_bolt_pair.wait_for_service(timeout_sec=2.0):
            return False, None, "GetBoltPair 서비스 응답 없음 (Timeout)"

        req = GetBoltPair.Request()
        future = self.client_get_bolt_pair.call_async(req)

        start = time.time()
        while not future.done():
            if time.time() - start > timeout_sec:
                return False, None, "GetBoltPair 서비스 호출 시간 초과"
            time.sleep(0.05)

        res = future.result()
        if res is not None and res.found:
            pose_a = res.pose_a.pose.position
            pose_b = res.pose_b.pose.position

            # 두 볼트의 3D 중앙 좌표 연산
            midpoint_pose = PoseStamped()
            midpoint_pose.header = res.pose_a.header
            midpoint_pose.pose.position.x = (pose_a.x + pose_b.x) / 2.0
            midpoint_pose.pose.position.y = (pose_a.y + pose_b.y) / 2.0
            midpoint_pose.pose.position.z = (pose_a.z + pose_b.z) / 2.0

            # Orientation은 기본 세팅 사용
            midpoint_pose.pose.orientation = res.pose_a.pose.orientation

            return True, midpoint_pose, res.message

        return False, None, res.message if res else "GetBoltPair 수신 결과 없음"

    def request_vision_pose_async(self, target_label: str, timeout_sec: float = 3.0):
        """Action Thread 안전한 Service 비동기 호출"""
        if target_label == "nut":
            client = self.client_get_nut_pose
            if client.wait_for_service(timeout_sec=1.0):
                req = GetGraspPose.Request()
                req.label = target_label
                future = client.call_async(req)
                
                start = time.time()
                while not future.done():
                    if time.time() - start > timeout_sec:
                        break
                    time.sleep(0.05)

                if future.done() and future.result() is not None:
                    res = future.result()
                    if res.found:
                        return True, res.pose, res.message

            # Fallback 토픽 사용
            if self.latest_nut_pose is not None:
                return True, self.latest_nut_pose, "토픽 데이터 사용"

            return False, None, f"'{target_label}' 검출 실패 (서비스/토픽 모두 없음)"

        return False, None, "알 수 없는 타겟 라벨"


def main(args=None):
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

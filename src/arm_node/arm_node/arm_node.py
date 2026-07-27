#!/usr/bin/env python3
"""
arm_node.py - ROS 2 Arm Control Node (MultiThreaded Executor & Reentrant Group)
- BehaviorNode의 /execute_arm_task Action 수신
- [지원 Task]
  1. SCAN_BATTERY        : 배터리 스캔 위치로 이동 후 Perception 노드로부터 두 볼트의 중앙 좌표 수신 및 저장
  2. SCAN_BUSBAR         : 버스바 스캔 위치로 이동 후 Perception 노드로부터 버스바 파지 좌표 미리 수신 및 저장
  3. PICK_BUSBAR         : 버스바 비전 위치 수신(또는 저장된 좌표 활용) 및 Isaac Sim 파지 명령 중계
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
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float32

# Action 및 Custom Interfaces
from fms_interfaces.action import ExecuteArmTask
from fms_interfaces.srv import GetGraspPose, GetBoltPair
from fms_interfaces.msg import BusbarGrasp, NutPose


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
        self.client_get_grasp_pose = self.create_client(
            GetGraspPose, '/perception/get_grasp_pose', callback_group=self.cb_group
        )
        self.client_get_bolt_pair = self.create_client(
            GetBoltPair, '/perception/get_bolt_pair', callback_group=self.cb_group
        )

        # 3. Perception Node 토픽 백업 구독
        self.latest_busbar_grasp = None
        self.latest_nut_pose = None

        self.sub_busbar_grasp = self.create_subscription(
            BusbarGrasp, '/vision/busbar_grasp', self._on_busbar_grasp, 10, callback_group=self.cb_group
        )
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

        # 너트 스캔 시 저장할 너트 1번 / 2번 좌표 (PoseStamped) (신규 추가)
        self.scanned_nut1_pose = None
        self.scanned_nut2_pose = None

        # ★ 배터리 중심 기준 볼트1/2 예상 오프셋 (execute_isaac.py의 BOLT1/2_OFFSET_FROM_CENTER와
        # 반드시 동일하게 유지) - get_bolt_pair가 반환하는 A/B 두 볼트 중 어느 쪽이 이
        # nut_index에 해당하는지 구분(가까운 쪽 매칭)하는 기준점으로만 쓴다.
        self._BOLT_OFFSET_FROM_CENTER = {
            1: (-0.1042, 0.1812),
            2: (0.1042, -0.1812),
        }

    # =========================================================================
    # 콜백 함수들
    # =========================================================================
    def _on_busbar_grasp(self, msg: BusbarGrasp):
        self.latest_busbar_grasp = msg.pose

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
    def execute_callback(self, goal_handle):
        task_type = goal_handle.request.task_type
        self.get_logger().info(f"\n================ [Action Goal Received: {task_type}] ================")
        
        feedback_msg = ExecuteArmTask.Feedback()
        result_msg = ExecuteArmTask.Result()
        self.isaac_status = None

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
            found, midpoint_pose, msg = self.request_bolt_pair_midpoint_async(timeout_sec=5.0)

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
            self.get_logger().info(" -> [SCAN_BUSBAR] Isaac Sim으로 버스바 스캔 이동 명령 전송")

            cmd_msg = String()
            cmd_msg.data = "SCAN_BUSBAR"
            self.pub_task_command.publish(cmd_msg)

            # 1. Isaac Sim이 버스바 스캔 위치로 이동 완료할 때까지 대기
            success = self.wait_for_isaac_completion(goal_handle, feedback_msg)

            if not success:
                result_msg.success = False
                result_msg.error_code = "SCAN_BUSBAR_FAILED"
                result_msg.message = f"버스바 스캔 위치 이동 실패 (Status: {self.isaac_status})"
                goal_handle.abort()
                return result_msg

            # 2. 이동 완료 후 Perception 노드에 버스바 좌표 요청
            self.get_logger().info(" -> [SCAN_BUSBAR] 버스바 스캔 위치 도착 완료. 비전 노드에 버스바 좌표 요청...")
            found, busbar_pose, msg = self.request_vision_pose_async("busbar", timeout_sec=5.0)

            if found and busbar_pose is not None:
                self.scanned_busbar_pose = busbar_pose
                self.get_logger().info(
                    f" ★ [버스바 좌표 저장 완료] "
                    f"X: {busbar_pose.pose.position.x:.4f}, "
                    f"Y: {busbar_pose.pose.position.y:.4f}, "
                    f"Z: {busbar_pose.pose.position.z:.4f}"
                )

                result_msg.success = True
                result_msg.message = f"버스바 스캔 및 비전 좌표 취득 성공 ({msg})"
                goal_handle.succeed()
            else:
                self.get_logger().error(f" -> 버스바 비전 검출 실패: {msg}")
                result_msg.success = False
                result_msg.error_code = "BUSBAR_VISION_FAILED"
                result_msg.message = f"버스바 스캔 좌표 취득 실패: {msg}"
                goal_handle.abort()

            return result_msg

        # ---------------------------------------------------------------------
        # [Task 3] PICK_BUSBAR (버스바 파지 및 들어올리기)
        # ---------------------------------------------------------------------
        elif task_type == "PICK_BUSBAR":
            busbar_pose = None

            # 1순위: SCAN_BUSBAR 단계에서 미리 취득해 둔 좌표 활용
            if self.scanned_busbar_pose is not None:
                self.get_logger().info(" -> [PICK_BUSBAR] 미리 스캔한 버스바 좌표를 활용합니다.")
                busbar_pose = self.scanned_busbar_pose
            else:
                # 2순위: 실시간 비전 노드에 버스바 좌표 직접 요청
                found, busbar_pose, msg = self.request_vision_pose_async("busbar")
                
                # 3순위: 백업으로 배터리 스캔 시 저장한 볼트 중점 좌표 활용
                if not found:
                    if self.scanned_battery_midpoint is not None:
                        self.get_logger().warn(" -> 버스바 직접 검출 실패. 배터리 스캔 시 저장한 볼트 중점 좌표를 백업으로 사용합니다.")
                        busbar_pose = self.scanned_battery_midpoint
                    else:
                        result_msg.success = False
                        result_msg.error_code = "BUSBAR_VISION_FAILED"
                        result_msg.message = msg
                        goal_handle.abort()
                        return result_msg

            # Isaac Sim 목표 좌표 및 파지 명령 퍼블리시
            self.pub_target_pose.publish(busbar_pose)
            
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

            # 2. 이동 완료 후 Perception 노드에 너트 좌표 요청 (참고용/비필수)
            # ★ PICK_NUT1/2의 실제 파지 좌표는 이제 execute_isaac.py의 고정좌표
            # (NUT1_PICK_XY/NUT2_PICK_XY)를 쓰므로, 여기서 검출 실패해도 전체 태스크를
            # abort하지 않는다 - 스캔 위치 이동 자체는 이미 성공했고, 이 값은 로그/
            # 참고용으로만 저장한다.
            self.get_logger().info(f" -> [{task_type}] 너트 스캔 위치 도착 완료. 비전 노드에 너트 좌표 지속 요청(참고용)...")
            found, nut_pose, msg = self.request_vision_pose_async("nut", retry_timeout_sec=5.0)

            if found and nut_pose is not None:
                if task_type == "SCAN_NUT1":
                    self.scanned_nut1_pose = nut_pose
                else:
                    self.scanned_nut2_pose = nut_pose

                self.get_logger().info(
                    f" ★ [{task_type} 좌표 저장 완료(참고용)] "
                    f"X: {nut_pose.pose.position.x:.4f}, "
                    f"Y: {nut_pose.pose.position.y:.4f}, "
                    f"Z: {nut_pose.pose.position.z:.4f}"
                )
                result_msg.message = f"너트 스캔 및 비전 좌표 취득 성공 ({msg})"
            else:
                self.get_logger().warn(f" -> 너트 비전 검출 실패(비필수, 고정좌표로 진행): {msg}")
                result_msg.message = f"너트 스캔 위치 이동 성공 (비전 좌표는 미취득: {msg})"

            result_msg.success = True
            goal_handle.succeed()

            return result_msg

        # ---------------------------------------------------------------------
        # 🔥 [Task 8] PICK_NUT1 / PICK_NUT2 (너트 물리 파지 및 들어올리기) (신규 추가)
        # ---------------------------------------------------------------------
        elif task_type in ("PICK_NUT1", "PICK_NUT2"):
            # ★ 실제 파지 좌표는 execute_isaac.py의 고정좌표(NUT1_PICK_XY/NUT2_PICK_XY)를
            # 쓰므로 scanned_nut1_pose/scanned_nut2_pose(SCAN_NUT의 참고용 비전값)가 없어도
            # 더 이상 막지 않는다. 있으면 참고용으로만 같이 퍼블리시.
            nut_pose = self.scanned_nut1_pose if task_type == "PICK_NUT1" else self.scanned_nut2_pose
            if nut_pose is not None:
                self.pub_target_pose.publish(nut_pose)

            cmd_msg = String()
            cmd_msg.data = task_type
            self.pub_task_command.publish(cmd_msg)

            # ★ eye-in-hand 실시간 추적(track_vision_label="nut")은 접근할수록 목표가
            # 표류해 미달하는 문제가 있어서 되돌림 - execute_isaac.py가 이제
            # NUT1_PICK_XY/NUT2_PICK_XY 고정좌표를 쓰므로 여기서도 재발행 불필요.
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
            nut_index = 1 if task_type == "ASSEMBLE_NUT1" else 2

            # ★ 볼트 위치는 원래 (배터리 중심 + 고정 오프셋) 가정만으로 계산돼서 실측과
            # 어긋나면 체결 위치를 못 잡는 문제가 있었음 - 착좌 하강 진입 전 get_bolt_pair로
            # 딱 1회 실측 보정을 시도한다. 실패했다고 /target_pose를 그냥 안 보내면
            # execute_isaac.py의 latest_target_pose에 이전 태스크(PICK_NUT 등)의 낡은 값이
            # 그대로 남아있어 엉뚱한 좌표로 오인될 위험이 있다 - 실패 시에도 반드시
            # 같은 (배터리 중심 + 고정 오프셋) 기준 좌표를 대신 발행해서 최신값을 덮어쓴다.
            found, bolt_pose, vmsg = self.request_bolt_pose_by_index_async(nut_index, timeout_sec=5.0)
            if found and bolt_pose is not None:
                self.pub_target_pose.publish(bolt_pose)
                self.get_logger().info(
                    f" -> [{task_type}] get_bolt_pair 실측 보정 적용 -> "
                    f"X={bolt_pose.pose.position.x:.4f}, Y={bolt_pose.pose.position.y:.4f} ({vmsg})"
                )
            else:
                self.get_logger().warn(f" -> [{task_type}] 볼트 실측 실패({vmsg}), 고정 오프셋 좌표로 진행")
                if self.scanned_battery_midpoint is not None:
                    offset = self._BOLT_OFFSET_FROM_CENTER[nut_index]
                    center = self.scanned_battery_midpoint.pose.position
                    fallback_pose = PoseStamped()
                    fallback_pose.header = self.scanned_battery_midpoint.header
                    fallback_pose.pose.position.x = center.x + offset[0]
                    fallback_pose.pose.position.y = center.y + offset[1]
                    fallback_pose.pose.position.z = center.z
                    fallback_pose.pose.orientation = self.scanned_battery_midpoint.pose.orientation
                    self.pub_target_pose.publish(fallback_pose)

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

    def wait_for_isaac_completion(self, goal_handle, feedback_msg, timeout_sec: float = 30.0,
                                   track_vision_label: str = None) -> bool:
        """track_vision_label(예: "nut")을 주면, 대기하는 동안 계속 갱신되는 최신 비전
        검출값(latest_nut_pose 등)을 매 폴링 주기(0.1s)마다 /target_pose로 재발행한다.
        그리퍼 장착 카메라(eye-in-hand)처럼 팔이 움직이면서 시점이 계속 바뀌는 대상은
        스캔 시점의 스냅샷 좌표 하나로는 하강 중 오차가 누적되기 때문 - Isaac Sim
        쪽(execute_isaac.py)이 매 스텝 latest_target_pose를 다시 읽어 목표를 갱신한다.
        고정 카메라 대상 태스크는 기본값(None)으로 두면 기존과 동일하게 동작한다.
        """
        start_time = time.time()
        while rclpy.ok():
            feedback_msg.sub_phase = self.isaac_phase
            feedback_msg.progress_pct = float(self.isaac_progress)
            goal_handle.publish_feedback(feedback_msg)

            if track_vision_label == "nut" and self.latest_nut_pose is not None:
                self.pub_target_pose.publish(self.latest_nut_pose)

            if self.isaac_status is not None:
                if self.isaac_status == "SUCCESS":
                    return True
                elif "FAILURE" in self.isaac_status:
                    return False

            if time.time() - start_time > timeout_sec:
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

    def request_bolt_pose_by_index_async(self, nut_index: int, timeout_sec: float = 5.0):
        """너트 체결(ASSEMBLE_NUT1/2) 직전에 get_bolt_pair로 실측한 볼트 좌표를 nut_index에
        매칭해서 반환한다. get_bolt_pair는 A/B 두 볼트를 이전 tick과의 최근접 매칭으로만
        구분하고 bolt1/bolt2로 라벨링하지 않으므로, scanned_battery_midpoint(SCAN_BATTERY에서
        저장) + _BOLT_OFFSET_FROM_CENTER로 계산한 예상 좌표에 더 가까운 쪽을 그 nut_index의
        볼트로 판정한다.
        """
        if self.scanned_battery_midpoint is None:
            return False, None, "scanned_battery_midpoint 없음 (SCAN_BATTERY 먼저 필요)"

        offset = self._BOLT_OFFSET_FROM_CENTER.get(nut_index)
        if offset is None:
            return False, None, f"알 수 없는 nut_index={nut_index!r}"

        center = self.scanned_battery_midpoint.pose.position
        expected_x = center.x + offset[0]
        expected_y = center.y + offset[1]

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
        if res is None or not res.found:
            return False, None, res.message if res else "GetBoltPair 수신 결과 없음"

        pos_a = res.pose_a.pose.position
        pos_b = res.pose_b.pose.position
        dist_a = math.hypot(pos_a.x - expected_x, pos_a.y - expected_y)
        dist_b = math.hypot(pos_b.x - expected_x, pos_b.y - expected_y)
        matched = res.pose_a if dist_a <= dist_b else res.pose_b
        matched_dist = min(dist_a, dist_b)

        return True, matched, f"{res.message} (예상좌표 대비 매칭거리 {matched_dist:.4f}m)"

    def request_vision_pose_async(self, target_label: str, timeout_sec: float = 3.0,
                                   retry_timeout_sec: float = 0.0, retry_interval_sec: float = 0.3):
        """Action Thread 안전한 Service 비동기 호출.

        retry_timeout_sec=0(기본)이면 기존과 동일하게 단발 요청 후 실패 시 바로 포기한다.
        retry_timeout_sec > 0으로 주면, 첫 시도에서 found=False가 나와도 바로 포기하지 않고
        그 시간 동안 retry_interval_sec 간격으로 계속 재요청한다 - perception_node의 롤링
        평균 캐시(SMOOTHING_WINDOW_SEC)가 스캔 도착 직후엔 아직 안 채워져 있거나, 순간적으로
        검출을 놓친 경우를 대비한 것.
        """
        if target_label not in ["busbar", "nut"]:
            return False, None, "알 수 없는 타겟 라벨"

        deadline = time.time() + max(retry_timeout_sec, 0.0)
        last_message = f"'{target_label}' 검출 실패 (서비스/토픽 모두 없음)"

        while True:
            if self.client_get_grasp_pose.wait_for_service(timeout_sec=1.0):
                req = GetGraspPose.Request()
                req.label = target_label
                future = self.client_get_grasp_pose.call_async(req)

                start = time.time()
                while not future.done():
                    if time.time() - start > timeout_sec:
                        break
                    time.sleep(0.05)

                if future.done() and future.result() is not None:
                    res = future.result()
                    if res.found:
                        return True, res.pose, res.message
                    last_message = res.message

            # Fallback 토픽 사용 (서비스는 못 받았어도 그 사이 토픽에 새 값이 왔을 수 있음)
            if target_label == "busbar" and self.latest_busbar_grasp is not None:
                return True, self.latest_busbar_grasp, "토픽 데이터 사용"
            elif target_label == "nut" and self.latest_nut_pose is not None:
                return True, self.latest_nut_pose, "토픽 데이터 사용"

            if time.time() >= deadline:
                if retry_timeout_sec > 0.0:
                    return False, None, f"'{target_label}' 검출 실패 ({retry_timeout_sec:.1f}s 재시도 후 포기: {last_message})"
                return False, None, last_message

            time.sleep(retry_interval_sec)


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
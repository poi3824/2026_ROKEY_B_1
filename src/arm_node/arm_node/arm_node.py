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

    def wait_for_isaac_completion(self, goal_handle, feedback_msg, timeout_sec: float = 30.0) -> bool:
        start_time = time.time()
        while rclpy.ok():
            feedback_msg.sub_phase = self.isaac_phase
            feedback_msg.progress_pct = float(self.isaac_progress)
            goal_handle.publish_feedback(feedback_msg)

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

    def request_vision_pose_async(self, target_label: str, timeout_sec: float = 3.0):
        """Action Thread 안전한 Service 비동기 호출"""
        if target_label in ["busbar", "nut"]:
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

            # Fallback 토픽 사용
            if target_label == "busbar" and self.latest_busbar_grasp is not None:
                return True, self.latest_busbar_grasp, "토픽 데이터 사용"
            elif target_label == "nut" and self.latest_nut_pose is not None:
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
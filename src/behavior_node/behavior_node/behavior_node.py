#!/usr/bin/env python3
import sys
from enum import Enum, auto
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from fms_interfaces.action import ExecuteArmTask


class ProcessState(Enum):
    IDLE = auto()                   # 대기 상태
    SCAN_BATTERY = auto()           # 배터리 스캔 지점으로 이동
    SCAN_BUSBAR = auto()            # 버스바 스캔 지점으로 이동
    PICK_BUSBAR = auto()            # 버스바 파지 및 상승
    MOVE_BATTERY_CENTER = auto()    # 배터리 중점 상공으로 이동
    FINE_ALIGNMENT = auto()         # 비전 기반 1픽셀 미세 오차 보정
    ASSEMBLE_BUSBAR = auto()        # 버스바 체결 (하강 및 그리퍼 해제) - 추가됨
    SCAN_NUT1 = auto()              # 너트 스캔 위치 이동 및 너트 1번 스캔 (신규)
    PICK_NUT1 = auto()              # 너트 1번 물리 파지 (신규)
    ASSEMBLE_NUT1 = auto()          # 볼트 1번 위치로 이동 및 너트 1번 체결 (신규)
    SCAN_NUT2 = auto()              # 너트 스캔 위치 재이동 및 너트 2번 스캔 (신규)
    PICK_NUT2 = auto()              # 너트 2번 물리 파지 (신규)
    ASSEMBLE_NUT2 = auto()          # 볼트 2번 위치로 이동 및 너트 2번 체결 (신규)
    SUCCESS = auto()                # 성공
    FAILURE = auto()                # 실패


class BehaviorNode(Node):
    def __init__(self):
        super().__init__('behavior_node')
        self.get_logger().info("===========================================")
        self.get_logger().info(" Behavior 노드 활성화 (스캔 -> 파지 -> 중점 이동 -> 미세 보정 -> 버스바 체결 -> 너트1/2 체결 시퀀스)")
        self.get_logger().info("===========================================")

        # Arm Node 액션 클라이언트 생성
        self._action_client = ActionClient(self, ExecuteArmTask, '/execute_arm_task')

        # FSM 상태 변수 
        self.state = ProcessState.IDLE
        self.is_waiting_action = False
        self.next_state_on_success = ProcessState.IDLE

        # 메인 FSM 루프 실행 (10Hz)
        self.create_timer(0.1, self.fsm_loop)

        # 테스트용: 노드 실행 후 3초 뒤 한 번만 시동
        self.start_timer = self.create_timer(3.0, self.auto_start_trigger)
        self.has_started = False

    def auto_start_trigger(self):
        """테스트용 트리거: 타이머 취소 후 배터리 스캔 이동부터 시퀀스 시작"""
        self.start_timer.cancel()
        if not self.has_started:
            self.has_started = True
            self.get_logger().info("\n[Behavior] 전체 공정 시퀀스를 시작합니다. (스캔 지점 이동 시작)")
            self.state = ProcessState.SCAN_BATTERY

    def fsm_loop(self):
        """FSM 상태 관리 루프"""
        if self.is_waiting_action or self.state == ProcessState.IDLE:
            return

        # [STEP 0] 배터리 스캔 지점으로 이동
        if self.state == ProcessState.SCAN_BATTERY:
            self.get_logger().info("\n>>> [STEP 0] 배터리 스캔 지점 이동 (SCAN_BATTERY) 요청")
            self.send_arm_goal(
                task_type="SCAN_BATTERY", 
                next_state=ProcessState.SCAN_BUSBAR
            )

        # [STEP 1] 버스바 스캔 지점으로 이동
        elif self.state == ProcessState.SCAN_BUSBAR:
            self.get_logger().info("\n>>> [STEP 1] 버스바 스캔 지점 이동 (SCAN_BUSBAR) 요청")
            self.send_arm_goal(
                task_type="SCAN_BUSBAR", 
                next_state=ProcessState.PICK_BUSBAR
            )

        # [STEP 2] 버스바 위치 접근 & 파지
        elif self.state == ProcessState.PICK_BUSBAR:
            self.get_logger().info("\n>>> [STEP 2] 버스바 파지 (PICK_BUSBAR) 요청")
            self.send_arm_goal(
                task_type="PICK_BUSBAR", 
                next_state=ProcessState.MOVE_BATTERY_CENTER
            )

        # [STEP 3] 스캔했던 배터리 볼트 중점 상공으로 이동
        elif self.state == ProcessState.MOVE_BATTERY_CENTER:
            self.get_logger().info("\n>>> [STEP 3] 배터리 중점 상공 이동 (MOVE_BATTERY_CENTER) 요청")
            self.send_arm_goal(
                task_type="MOVE_BATTERY_CENTER", 
                next_state=ProcessState.FINE_ALIGNMENT
            )

        # [STEP 4] 비전 노드 기반 1픽셀 정밀 오차 보정 수행
        elif self.state == ProcessState.FINE_ALIGNMENT:
            self.get_logger().info("\n>>> [STEP 4] 정밀 비전 오차 보정 (FINE_ALIGNMENT) 수행 중...")
            self.send_arm_goal(
                task_type="FINE_ALIGNMENT", 
                next_state=ProcessState.ASSEMBLE_BUSBAR
            )

        # [STEP 5] 버스바 하강 및 최종 체결 수행 (추가됨)
        elif self.state == ProcessState.ASSEMBLE_BUSBAR:
            self.get_logger().info("\n>>> [STEP 5] 버스바 체결 및 하강 (ASSEMBLE_BUSBAR) 요청")
            self.send_arm_goal(
                task_type="ASSEMBLE_BUSBAR",
                next_state=ProcessState.SCAN_NUT1
            )

        # [STEP 6] 너트 스캔 위치 이동 및 너트 1번 스캔 (신규)
        elif self.state == ProcessState.SCAN_NUT1:
            self.get_logger().info("\n>>> [STEP 6] 너트 스캔 위치 이동 및 너트 1번 스캔 (SCAN_NUT1) 요청")
            self.send_arm_goal(
                task_type="SCAN_NUT1",
                next_state=ProcessState.PICK_NUT1
            )

        # [STEP 7] 너트 1번 물리 파지 (신규)
        elif self.state == ProcessState.PICK_NUT1:
            self.get_logger().info("\n>>> [STEP 7] 너트 1번 파지 (PICK_NUT1) 요청")
            self.send_arm_goal(
                task_type="PICK_NUT1",
                next_state=ProcessState.ASSEMBLE_NUT1
            )

        # [STEP 8] 볼트 1번 위치로 이동 및 너트 1번 체결 (신규)
        elif self.state == ProcessState.ASSEMBLE_NUT1:
            self.get_logger().info("\n>>> [STEP 8] 볼트 1번 위치 이동 및 너트 1번 체결 (ASSEMBLE_NUT1) 요청")
            self.send_arm_goal(
                task_type="ASSEMBLE_NUT1",
                next_state=ProcessState.SCAN_NUT2
            )

        # [STEP 9] 너트 스캔 위치 재이동 및 너트 2번 스캔 (신규)
        elif self.state == ProcessState.SCAN_NUT2:
            self.get_logger().info("\n>>> [STEP 9] 너트 스캔 위치 재이동 및 너트 2번 스캔 (SCAN_NUT2) 요청")
            self.send_arm_goal(
                task_type="SCAN_NUT2",
                next_state=ProcessState.PICK_NUT2
            )

        # [STEP 10] 너트 2번 물리 파지 (신규)
        elif self.state == ProcessState.PICK_NUT2:
            self.get_logger().info("\n>>> [STEP 10] 너트 2번 파지 (PICK_NUT2) 요청")
            self.send_arm_goal(
                task_type="PICK_NUT2",
                next_state=ProcessState.ASSEMBLE_NUT2
            )

        # [STEP 11] 볼트 2번 위치로 이동 및 너트 2번 체결 (신규)
        elif self.state == ProcessState.ASSEMBLE_NUT2:
            self.get_logger().info("\n>>> [STEP 11] 볼트 2번 위치 이동 및 너트 2번 체결 (ASSEMBLE_NUT2) 요청")
            self.send_arm_goal(
                task_type="ASSEMBLE_NUT2",
                next_state=ProcessState.SUCCESS
            )

        # 최종 성공
        elif self.state == ProcessState.SUCCESS:
            self.get_logger().info("\n===========================================")
            self.get_logger().info("  ★ [공정 완수] 스캔, 파지, 중점 이동, 정밀 오차 보정, 버스바 체결 및 너트 1/2번 체결 완료 ★")
            self.get_logger().info("===========================================")
            self.state = ProcessState.IDLE

        # 오류 및 중단
        elif self.state == ProcessState.FAILURE:
            self.get_logger().error("\n===========================================")
            self.get_logger().error("  X [공정 중단] 오류 발생으로 작업을 중단합니다.")
            self.get_logger().error("===========================================")
            self.state = ProcessState.IDLE

    def send_arm_goal(self, task_type: str, next_state: ProcessState):
        """Arm 노드로 Action Goal 전송"""
        if not self._action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Arm Action Server가 응답하지 않습니다! (Timeout)")
            self.state = ProcessState.FAILURE
            return

        goal_msg = ExecuteArmTask.Goal()
        goal_msg.task_type = task_type

        self.is_waiting_action = True
        self.next_state_on_success = next_state
        
        send_goal_future = self._action_client.send_goal_async(
            goal_msg, 
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Goal 수락 여부 확인"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(" -> Arm Node가 Goal 수락을 거부했습니다.")
            self.is_waiting_action = False
            self.state = ProcessState.FAILURE
            return

        self.get_logger().info(" -> Goal 수락됨. 하위 작업 수행 중...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """피드백 모니터링"""
        fb = feedback_msg.feedback
        sys.stdout.write(
            f"\r    [Feedback] Phase: {fb.sub_phase:<20} | Progress: {fb.progress_pct:5.1f}%"
        )
        sys.stdout.flush()

    def get_result_callback(self, future):
        """최종 결과 처리"""
        result = future.result().result
        self.is_waiting_action = False

        print()

        if result.success:
            self.get_logger().info(f" -> [{self.state.name}] 작업 성공 완료!")
            self.state = self.next_state_on_success
        else:
            self.get_logger().error(f" -> [{self.state.name}] 작업 실패!")
            self.get_logger().error(f"    [Error Code] : {result.error_code}")
            self.get_logger().error(f"    [Error Message]: {result.message}")
            self.state = ProcessState.FAILURE


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("\nBehavior Manager 키보드 입력에 의해 종료됨.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
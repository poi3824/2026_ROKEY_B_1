#!/usr/bin/env python3
import sys
import time
from enum import Enum, auto
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from fms_interfaces.action import ExecuteArmTask
from fms_interfaces.msg import AmrGoal, AmrStatus, FleetJob, FleetReport


X_BATTERY = 0.6667
Y_BUSBAR = (1.2392 + 1.205 + 1.2367) / 3

# amr_node의 amr_baseline 좌표계 기준 접근 지점.
AMR_STATION_POSES = {
    "station_5": {
        "battery": ("battery5", X_BATTERY, -1.1964, -1.5707),
        "busbar": ("busbar5", -1.1298, Y_BUSBAR, 0.0),
    },
    "station_4": {
        "battery": ("battery4", X_BATTERY, -0.6617, -1.5707),
        "busbar": ("busbar4", -0.6606, Y_BUSBAR, 0.0),
    },
    "station_3": {
        "battery": ("battery3", X_BATTERY, -0.0382, -1.5707),
        "busbar": ("busbar3", -0.1911, Y_BUSBAR, 0.0),
    },
}


class ProcessState(Enum):
    IDLE = auto()                   # 대기 상태
    MOVE_AMR_BATTERY_SCAN = auto()  # AMR을 배터리 스캔 접근 지점으로 이동
    SCAN_BATTERY = auto()           # 배터리 스캔 지점으로 이동
    MOVE_AMR_BUSBAR = auto()        # AMR을 버스바 접근 지점으로 이동
    SCAN_BUSBAR = auto()            # 버스바 스캔 지점으로 이동
    PICK_BUSBAR = auto()            # 버스바 파지 및 상승
    MOVE_AMR_BATTERY_ASSEMBLY = auto()  # 파지 후 배터리 조립 지점 복귀
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

        # amr_node 연동
        self._amr_goal_pub = self.create_publisher(AmrGoal, '/amr/goal', 10)
        self._amr_status_sub = self.create_subscription(
            AmrStatus, '/amr/status', self._on_amr_status, 10)

        # fleet_manager_node 연동
        self._fleet_job_sub = self.create_subscription(
            FleetJob, '/fleet/job', self._on_fleet_job, 10)
        self._fleet_report_pub = self.create_publisher(
            FleetReport, '/fleet/report', 10)

        self.declare_parameter('work_station', 'station_4')
        self.declare_parameter('amr_move_timeout_sec', 120.0)
        self.declare_parameter('auto_start', False)
        self._work_station = self.get_parameter('work_station').value
        self._amr_move_timeout_sec = float(
            self.get_parameter('amr_move_timeout_sec').value)
        self._auto_start = bool(self.get_parameter('auto_start').value)
        if self._work_station not in AMR_STATION_POSES:
            raise ValueError(
                f"work_station must be one of {sorted(AMR_STATION_POSES)}")
        if self._amr_move_timeout_sec <= 0.0:
            raise ValueError("amr_move_timeout_sec must be positive")
        self._amr_goal_sent = False
        self._waiting_amr_station = None
        self._amr_deadline = 0.0
        self._active_job = None

        # FSM 상태 변수
        self.state = ProcessState.IDLE
        self.is_waiting_action = False
        self.next_state_on_success = ProcessState.IDLE

        # 메인 FSM 루프 실행 (10Hz)
        self.create_timer(0.1, self.fsm_loop)

        # standalone 점검에서만 사용하는 선택적 자동 시작.
        self.start_timer = self.create_timer(3.0, self.auto_start_trigger)
        self.has_started = False

    def auto_start_trigger(self):
        """테스트용 트리거: 타이머 취소 후 배터리 스캔 이동부터 시퀀스 시작"""
        self.start_timer.cancel()
        if self._auto_start and not self.has_started:
            self.has_started = True
            self.get_logger().info(
                "\n[Behavior] 전체 공정 시퀀스를 시작합니다. "
                "(AMR 배터리 접근 지점 이동 시작)")
            self.state = ProcessState.MOVE_AMR_BATTERY_SCAN

    def _on_fleet_job(self, msg: FleetJob):
        """Fleet 작업 하나를 받아 해당 station의 전체 조립 FSM을 시작한다."""
        if self.state != ProcessState.IDLE or self._active_job is not None:
            self.get_logger().warn(
                f"작업 수행 중이므로 /fleet/job 무시: {msg.job_id}")
            return

        if msg.job_type != 'ASSEMBLE':
            self._publish_fleet_report(
                msg, False, f"지원하지 않는 job_type: {msg.job_type}")
            return
        if msg.station_id not in AMR_STATION_POSES:
            self._publish_fleet_report(
                msg, False, f"알 수 없는 station_id: {msg.station_id}")
            return

        self._active_job = msg
        self._work_station = msg.station_id
        self._amr_goal_sent = False
        self._waiting_amr_station = None
        self.is_waiting_action = False
        self.state = ProcessState.MOVE_AMR_BATTERY_SCAN
        self.get_logger().info(
            f"SUB /fleet/job <- {msg.job_id} "
            f"({msg.station_id}, {msg.job_type}); 전체 공정 시작")

    def _publish_fleet_report(
        self, job: FleetJob, success: bool, message: str
    ):
        report = FleetReport()
        report.job_id = job.job_id
        report.station_id = job.station_id
        report.success = success
        report.message = message
        report.stamp = self.get_clock().now().to_msg()
        self._fleet_report_pub.publish(report)
        self.get_logger().info(
            f"PUB /fleet/report -> {job.job_id} "
            f"{'SUCCESS' if success else 'FAILED'}: {message}")

    def _finish_active_job(self, success: bool, message: str):
        if self._active_job is not None:
            self._publish_fleet_report(
                self._active_job, success, message)
            self._active_job = None

    def fsm_loop(self):
        """FSM 상태 관리 루프"""
        if self.is_waiting_action or self.state == ProcessState.IDLE:
            return

        if self.state in (
            ProcessState.MOVE_AMR_BATTERY_SCAN,
            ProcessState.MOVE_AMR_BUSBAR,
            ProcessState.MOVE_AMR_BATTERY_ASSEMBLY,
        ):
            if not self._amr_goal_sent:
                target_kind = (
                    "busbar"
                    if self.state == ProcessState.MOVE_AMR_BUSBAR
                    else "battery"
                )
                self._send_amr_goal(target_kind)
            elif time.monotonic() > self._amr_deadline:
                self.get_logger().error(
                    f"AMR 이동 Timeout: {self._waiting_amr_station}")
                self._amr_goal_sent = False
                self._waiting_amr_station = None
                self.state = ProcessState.FAILURE
            return

        # [STEP 0] 배터리 스캔 지점으로 이동
        if self.state == ProcessState.SCAN_BATTERY:
            self.get_logger().info("\n>>> [STEP 0] 배터리 스캔 지점 이동 (SCAN_BATTERY) 요청")
            self.send_arm_goal(
                task_type="SCAN_BATTERY",
                next_state=ProcessState.MOVE_AMR_BUSBAR
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
                next_state=ProcessState.MOVE_AMR_BATTERY_ASSEMBLY
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
            self._finish_active_job(True, "전체 조립 공정 완료")
            self.state = ProcessState.IDLE

        # 오류 및 중단
        elif self.state == ProcessState.FAILURE:
            self.get_logger().error("\n===========================================")
            self.get_logger().error("  X [공정 중단] 오류 발생으로 작업을 중단합니다.")
            self.get_logger().error("===========================================")
            self._finish_active_job(False, "하위 AMR/Arm 공정 실패")
            self.state = ProcessState.IDLE

    def _send_amr_goal(self, target_kind: str):
        station_id, x, y, theta = AMR_STATION_POSES[
            self._work_station][target_kind]
        goal = AmrGoal()
        goal.station_id = station_id
        goal.x = x
        goal.y = y
        goal.theta = theta
        self._waiting_amr_station = station_id
        self._amr_goal_sent = True
        self._amr_deadline = (
            time.monotonic() + self._amr_move_timeout_sec)
        self._amr_goal_pub.publish(goal)
        self.get_logger().info(
            f"PUB /amr/goal -> {station_id} ({x:.4f}, {y:.4f})")

    def _on_amr_status(self, msg: AmrStatus):
        if (
            not self._amr_goal_sent
            or msg.station_id != self._waiting_amr_station
        ):
            return
        if msg.state == AmrStatus.STATE_MOVING:
            self.get_logger().info(
                f"AMR 이동 시작 확인: {msg.station_id}")
            return
        if msg.state == AmrStatus.STATE_ERROR:
            self.get_logger().error(
                f"AMR 이동 실패: {msg.station_id}: {msg.message}")
            self._amr_goal_sent = False
            self._waiting_amr_station = None
            self.state = ProcessState.FAILURE
            return
        if msg.state != AmrStatus.STATE_ARRIVED:
            return

        self.get_logger().info(
            f"AMR 도착 확인: {msg.station_id}: {msg.message}")
        self._amr_goal_sent = False
        self._waiting_amr_station = None
        if self.state == ProcessState.MOVE_AMR_BATTERY_SCAN:
            self.state = ProcessState.SCAN_BATTERY
        elif self.state == ProcessState.MOVE_AMR_BUSBAR:
            self.state = ProcessState.SCAN_BUSBAR
        elif self.state == ProcessState.MOVE_AMR_BATTERY_ASSEMBLY:
            self.state = ProcessState.MOVE_BATTERY_CENTER

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
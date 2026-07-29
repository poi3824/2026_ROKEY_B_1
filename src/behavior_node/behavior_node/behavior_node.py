#!/usr/bin/env python3
"""Coordinate the AMR and arm through the complete assembly process."""

import sys
import time
from enum import Enum, auto
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from fms_interfaces.action import ExecuteArmTask
from fms_interfaces.msg import AmrGoal, AmrStatus, FleetJob, FleetReport
from .amr_timeout import (
    DEFAULT_AMR_MOVE_TIMEOUT_SEC,
    amr_move_deadline,
    amr_move_timed_out,
)


X_BATTERY = 0.6667
Y_BUSBAR = 1.9078

# amr_node의 amr_baseline 좌표계 기준 접근 지점.
AMR_STATION_POSES = {
    "station_5": {
        "battery": ("battery5", X_BATTERY, -1.1964, -1.5707),
        "busbar": ("busbar5", -0.9586, Y_BUSBAR, -1.5707),
    },
    "station_4": {
        "battery": ("battery4", X_BATTERY, -0.6617, -1.5707),
        "busbar": ("busbar4", -0.2271, Y_BUSBAR, -1.5707),
    },
    "station_3": {
        "battery": ("battery3", X_BATTERY, -0.0382, -1.5707),
        "busbar": ("busbar3", 0.5867, Y_BUSBAR, -1.5707),
    },
}


class ProcessState(Enum):
    """States in one station's busbar-and-nut assembly process."""

    IDLE = auto()                   # 대기 상태
    MOVE_AMR_BATTERY_SCAN = auto()  # AMR을 배터리 스캔 접근 지점으로 이동
    SCAN_BATTERY = auto()           # 배터리 스캔 지점으로 이동
    MOVE_AMR_BUSBAR = auto()        # AMR을 버스바 접근 지점으로 이동
    INITIALIZE_ARM = auto()          # 버스바 정차 후 초기 안전 관절 자세로 복귀
    SCAN_BUSBAR = auto()            # 고정 버스바 카메라 pose latch
    PICK_BUSBAR = auto()            # 버스바 파지 및 상승
    MOVE_AMR_BATTERY_ASSEMBLY = auto()  # 파지 후 배터리 조립 지점 복귀
    MOVE_BATTERY_CENTER = auto()    # 배터리 중점 상공으로 이동
    FINE_ALIGNMENT = auto()         # 비전 기반 1픽셀 미세 오차 보정
    ASSEMBLE_BUSBAR = auto()        # 버스바 체결 (하강 및 그리퍼 해제) - 추가됨
    RETURN_HOME_FOR_NUT = auto()    # 너트 작업 전 초기 관절 자세로 복귀 (HOME_EE_POS 갱신)
    SCAN_NUT1 = auto()              # 너트 스캔 위치 이동 및 너트 1번 스캔 (신규)
    PICK_NUT1 = auto()              # 너트 1번 물리 파지 (신규)
    ASSEMBLE_NUT1 = auto()          # 볼트 1번 위치로 이동 및 너트 1번 체결 (신규)
    SCAN_NUT2 = auto()              # 너트 스캔 위치 재이동 및 너트 2번 스캔 (신규)
    PICK_NUT2 = auto()              # 너트 2번 물리 파지 (신규)
    ASSEMBLE_NUT2 = auto()          # 볼트 2번 위치로 이동 및 너트 2번 체결 (신규)
    SUCCESS = auto()                # 성공
    FAILURE = auto()                # 실패


class BehaviorNode(Node):
    """Run the station-level AMR and arm state machine."""

    def __init__(self):
        """Create action, fleet, AMR, and timer interfaces."""
        super().__init__('behavior_node')
        self.get_logger().info("===========================================")
        self.get_logger().info(
            " Behavior 노드 활성화 (스캔 -> 파지 -> 중점 이동 -> "
            "미세 보정 -> 버스바 체결 -> 너트1/2 체결 시퀀스)"
        )
        self.get_logger().info("===========================================")

        # Arm Node 액션 클라이언트 생성
        self._action_client = ActionClient(
            self, ExecuteArmTask, '/execute_arm_task')

        # amr_node 연동
        self._amr_goal_pub = self.create_publisher(AmrGoal, '/amr/goal', 10)
        self._amr_status_sub = self.create_subscription(
            AmrStatus, '/amr/status', self._on_amr_status, 10)

        # fleet_manager_node 연동
        self._fleet_job_sub = self.create_subscription(
            FleetJob, '/fleet/job', self._on_fleet_job, 10)
        self._fleet_report_pub = self.create_publisher(
            FleetReport, '/fleet/report', 10)

        self.declare_parameter('work_station', 'station_3')
        self.declare_parameter(
            'amr_move_timeout_sec',
            DEFAULT_AMR_MOVE_TIMEOUT_SEC,
        )
        self.declare_parameter('auto_start', False)
        self._work_station = self.get_parameter('work_station').value
        self._amr_move_timeout_sec = float(
            self.get_parameter('amr_move_timeout_sec').value)
        self._auto_start = bool(self.get_parameter('auto_start').value)
        if self._work_station not in AMR_STATION_POSES:
            raise ValueError(
                f"work_station must be one of {sorted(AMR_STATION_POSES)}")
        if self._amr_move_timeout_sec < 0.0:
            raise ValueError(
                "amr_move_timeout_sec must be zero or greater")
        self._amr_goal_sent = False
        self._waiting_amr_station = None
        self._amr_deadline = None
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
        """타이머 취소 후 버스바 접근 이동부터 테스트 시퀀스를 시작한다."""
        self.start_timer.cancel()
        if self._auto_start and not self.has_started:
            self.has_started = True
            self.get_logger().info(
                "\n[Behavior] 전체 공정 시퀀스를 시작합니다. "
                "(AMR 버스바 접근 지점으로 바로 이동 시작)")
            self.state = ProcessState.MOVE_AMR_BUSBAR

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
        self.state = ProcessState.MOVE_AMR_BUSBAR
        self.get_logger().info(
            f"SUB /fleet/job <- {msg.job_id} "
            f"({msg.station_id}, {msg.job_type}); "
            "버스바 접근 지점으로 바로 이동")

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
        """FSM 상태를 한 번 진행한다."""
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
            elif amr_move_timed_out(
                self._amr_deadline,
                time.monotonic(),
            ):
                self.get_logger().error(
                    f"AMR 이동 Timeout: {self._waiting_amr_station}")
                self._amr_goal_sent = False
                self._waiting_amr_station = None
                self._amr_deadline = None
                self.state = ProcessState.FAILURE
            return

        # [STEP 0] 배터리 스캔 지점으로 이동
        if self.state == ProcessState.SCAN_BATTERY:
            self.get_logger().info("\n>>> [STEP 0] 배터리 스캔 지점 이동 (SCAN_BATTERY) 요청")
            self.send_arm_goal(
                task_type="SCAN_BATTERY",
                next_state=ProcessState.MOVE_AMR_BUSBAR
            )

        # 기본 공정은 초기 배터리 스캔 결과를 요청하지 않으므로 배터리 방문과
        # SCAN_BATTERY/GetBoltPair를 건너뛴다. 버스바 정차 뒤에는 기존 RETURN_HOME
        # task로 안전 관절 자세만 먼저 만든다.
        elif self.state == ProcessState.INITIALIZE_ARM:
            self.get_logger().info(
                "\n>>> [STEP 0] 버스바 정차 후 초기 안전 자세 "
                "(RETURN_HOME) 요청")
            self.send_arm_goal(
                task_type="RETURN_HOME",
                next_state=ProcessState.SCAN_BUSBAR
            )

        # [STEP 1] 고정 버스바 카메라 좌표 latch
        elif self.state == ProcessState.SCAN_BUSBAR:
            self.get_logger().info(
                "\n>>> [STEP 1] 고정 버스바 카메라 좌표 "
                "latch (LATCH_BUSBAR) 요청"
            )
            self.send_arm_goal(
                task_type="LATCH_BUSBAR",
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
                next_state=ProcessState.RETURN_HOME_FOR_NUT
            )

        # [STEP 5.5] 너트 작업 전 초기 관절 자세로 복귀 (신규)
        # 너트 위치는 execute_isaac.py에서 HOME_EE_POS 기준 상대 오프셋으로 계산되는데,
        # 버스바 체결까지 마친 시점의 팔 위치는 HOME이 아니다(배터리 중점 상공 쪽에
        # 남아있음). AMR이 다른 배터리 스테이션으로 이동한 뒤라 HOME_EE_POS도 예전
        # 스테이션 기준 값으로 남아있을 수 있으므로, RETURN_HOME task로 관절 각도만
        # 초기 자세로 되돌려 HOME_EE_POS를 현재 AMR 위치 기준으로 새로 갱신한다
        # (SCAN_BATTERY를 재사용하면 배터리 스캔 고도까지 다시 올라가버려서 안 됨).
        elif self.state == ProcessState.RETURN_HOME_FOR_NUT:
            self.get_logger().info("\n>>> [STEP 5.5] 너트 작업 전 초기 자세 복귀 (RETURN_HOME_FOR_NUT) 요청")
            self.send_arm_goal(
                task_type="RETURN_HOME",
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
        self._amr_deadline = amr_move_deadline(
            time.monotonic(),
            self._amr_move_timeout_sec,
        )
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
            self._amr_deadline = None
            self.state = ProcessState.FAILURE
            return
        if msg.state != AmrStatus.STATE_ARRIVED:
            return

        self.get_logger().info(
            f"AMR 도착 확인: {msg.station_id}: {msg.message}")
        self._amr_goal_sent = False
        self._waiting_amr_station = None
        self._amr_deadline = None
        if self.state == ProcessState.MOVE_AMR_BATTERY_SCAN:
            self.state = ProcessState.SCAN_BATTERY
        elif self.state == ProcessState.MOVE_AMR_BUSBAR:
            self.state = ProcessState.INITIALIZE_ARM
        elif self.state == ProcessState.MOVE_AMR_BATTERY_ASSEMBLY:
            self.state = ProcessState.MOVE_BATTERY_CENTER

    def send_arm_goal(self, task_type: str, next_state: ProcessState):
        """Arm 노드로 Action Goal을 전송한다."""
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
        """Goal 수락 여부를 확인한다."""
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
        """Arm action 피드백을 출력한다."""
        fb = feedback_msg.feedback
        sys.stdout.write(
            f"\r    [Feedback] Phase: {fb.sub_phase:<20} | "
            f"Progress: {fb.progress_pct:5.1f}%"
        )
        sys.stdout.flush()

    def get_result_callback(self, future):
        """Arm action의 최종 결과를 처리한다."""
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
    """Run the behavior node until shutdown."""
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

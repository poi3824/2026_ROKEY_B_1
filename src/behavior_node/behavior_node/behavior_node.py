"""behavior_node
지휘 계층 · job 하나를 받아 한 스테이션의 조립을 끝까지 지휘.

SUB /fleet/job                                         (fms_interfaces/FleetJob)
PUB /fleet/report                                       (fms_interfaces/FleetReport)

PUB /amr/goal            SUB /amr/status                (이동)
PUB /busbar/command · /busbar/target                     (버스바 파지·삽입)
PUB /fasten/command       SUB /busbar/result · /fasten/result
SUB /vision/stud_pose · /vision/busbar_grasp · /vision/nut_pose
"""
from enum import Enum, auto

import rclpy
from rclpy.node import Node

from fms_interfaces.msg import (
    FleetJob, FleetReport,
    AmrGoal, AmrStatus,
    BusbarCommand, BusbarTarget, BusbarResult,
    FastenCommand, FastenResult,
    StudPose, BusbarGrasp, NutPose,
)

# 스테이션 좌표 (Isaac Sim 월드 기준). TODO: 실제 스테이션 배치로 교체.
STATION_POSES = {
    'station_1': (1.0, 0.0, 0.0),
    'station_2': (2.0, 0.0, 0.0),
    'station_3': (3.0, 0.0, 0.0),
}

MAX_RETRY = 3


class State(Enum):
    IDLE = auto()
    MOVE_TO_STATION = auto()
    WAIT_BUSBAR_VISION = auto()
    GRASP_BUSBAR = auto()
    INSERT_BUSBAR = auto()
    WAIT_NUT_VISION = auto()
    FASTEN_APPROACH = auto()
    FASTEN = auto()
    RECOVER = auto()
    REPORT = auto()


class BehaviorNode(Node):

    def __init__(self):
        super().__init__('behavior_node')

        # FMS 인터페이스
        self._job_sub = self.create_subscription(FleetJob, '/fleet/job', self._on_job, 10)
        self._report_pub = self.create_publisher(FleetReport, '/fleet/report', 10)

        # amr_node 인터페이스
        self._amr_goal_pub = self.create_publisher(AmrGoal, '/amr/goal', 10)
        self._amr_status_sub = self.create_subscription(
            AmrStatus, '/amr/status', self._on_amr_status, 10)

        # arm_node 인터페이스
        self._busbar_cmd_pub = self.create_publisher(BusbarCommand, '/busbar/command', 10)
        self._busbar_target_pub = self.create_publisher(BusbarTarget, '/busbar/target', 10)
        self._fasten_cmd_pub = self.create_publisher(FastenCommand, '/fasten/command', 10)
        self._busbar_result_sub = self.create_subscription(
            BusbarResult, '/busbar/result', self._on_busbar_result, 10)
        self._fasten_result_sub = self.create_subscription(
            FastenResult, '/fasten/result', self._on_fasten_result, 10)

        # perception_node 인터페이스
        self._stud_pose_sub = self.create_subscription(
            StudPose, '/vision/stud_pose', self._on_stud_pose, 10)
        self._busbar_grasp_sub = self.create_subscription(
            BusbarGrasp, '/vision/busbar_grasp', self._on_busbar_grasp, 10)
        self._nut_pose_sub = self.create_subscription(
            NutPose, '/vision/nut_pose', self._on_nut_pose, 10)

        self._state = State.IDLE
        self._job = None
        self._retry_count = 0
        self._recover_target_state = None
        self._latest_busbar_grasp = None
        self._latest_stud_pose = None
        self._latest_nut_pose = None

        self._timer = self.create_timer(0.5, self._step)

        self.get_logger().info('behavior_node started')

    # --- job 해석 ---------------------------------------------------------
    def _on_job(self, msg: FleetJob):
        if self._state != State.IDLE:
            self.get_logger().warn(
                f'job {msg.job_id} 수신했지만 이미 {self._job.job_id if self._job else "?"} 처리 중, 무시')
            return

        self.get_logger().info(
            f'SUB /fleet/job <- {msg.job_id} ({msg.station_id}, {msg.job_type})')
        self._job = msg
        self._retry_count = 0
        self._latest_busbar_grasp = None
        self._latest_stud_pose = None
        self._latest_nut_pose = None
        self._set_state(State.MOVE_TO_STATION)

    # --- 조립 FSM 상태 전이 -------------------------------------------------
    def _set_state(self, new_state: State):
        self.get_logger().info(f'[FSM] {self._state.name} -> {new_state.name}')
        self._state = new_state

    def _step(self):
        if self._job is None:
            return

        if self._state == State.MOVE_TO_STATION:
            self._enter_move_to_station()
        elif self._state == State.WAIT_BUSBAR_VISION:
            if self._latest_busbar_grasp is not None:
                self._set_state(State.GRASP_BUSBAR)
                self._send_busbar_command('GRASP')
        elif self._state == State.WAIT_NUT_VISION:
            if self._latest_stud_pose is not None and self._latest_nut_pose is not None:
                self._set_state(State.FASTEN_APPROACH)
                self._send_fasten_command('APPROACH')
        elif self._state == State.REPORT:
            self._send_report(success=True, message='조립 완료')
            self._set_state(State.IDLE)
            self._job = None

    # --- 이동 -------------------------------------------------------------
    def _enter_move_to_station(self):
        if getattr(self, '_move_goal_sent', False):
            return
        x, y, theta = STATION_POSES.get(self._job.station_id, (0.0, 0.0, 0.0))
        goal = AmrGoal()
        goal.station_id = self._job.station_id
        goal.x, goal.y, goal.theta = x, y, theta
        self._amr_goal_pub.publish(goal)
        self._move_goal_sent = True
        self.get_logger().info(f'PUB /amr/goal -> {goal.station_id}')

    def _on_amr_status(self, msg: AmrStatus):
        if self._state != State.MOVE_TO_STATION:
            return
        if msg.state == AmrStatus.STATE_ARRIVED:
            self._move_goal_sent = False
            self._set_state(State.WAIT_BUSBAR_VISION)
        elif msg.state == AmrStatus.STATE_ERROR:
            self._move_goal_sent = False
            self._enter_recover(State.MOVE_TO_STATION, msg.message)

    # --- 버스바 파지 · 삽입 --------------------------------------------------
    def _send_busbar_command(self, command: str):
        cmd = BusbarCommand()
        cmd.command = command
        cmd.station_id = self._job.station_id
        self._busbar_cmd_pub.publish(cmd)

        target = BusbarTarget()
        target.station_id = self._job.station_id
        target.target_pose = self._latest_busbar_grasp.pose.pose
        self._busbar_target_pub.publish(target)
        self.get_logger().info(f'PUB /busbar/command -> {command}')

    def _on_busbar_grasp(self, msg: BusbarGrasp):
        self._latest_busbar_grasp = msg

    def _on_busbar_result(self, msg: BusbarResult):
        if not msg.success:
            self._enter_recover(self._state, msg.message)
            return

        if self._state == State.GRASP_BUSBAR:
            self._set_state(State.INSERT_BUSBAR)
            self._send_busbar_command('INSERT')
        elif self._state == State.INSERT_BUSBAR:
            self._set_state(State.WAIT_NUT_VISION)

    # --- 너트 체결 시퀀스 ----------------------------------------------------
    def _on_stud_pose(self, msg: StudPose):
        self._latest_stud_pose = msg

    def _on_nut_pose(self, msg: NutPose):
        self._latest_nut_pose = msg

    def _send_fasten_command(self, command: str):
        cmd = FastenCommand()
        cmd.command = command
        cmd.nut_id = str(self._latest_nut_pose.id) if self._latest_nut_pose else ''
        self._fasten_cmd_pub.publish(cmd)
        self.get_logger().info(f'PUB /fasten/command -> {command}')

    def _on_fasten_result(self, msg: FastenResult):
        if not msg.success:
            self._enter_recover(self._state, msg.message)
            return

        if self._state == State.FASTEN_APPROACH:
            self._set_state(State.FASTEN)
            self._send_fasten_command('FASTEN')
        elif self._state == State.FASTEN:
            self.get_logger().info(f'체결 토크 확인 완료: {msg.torque:.2f} Nm')
            self._set_state(State.REPORT)

    # --- 복구 로직 ----------------------------------------------------------
    def _enter_recover(self, failed_state: State, reason: str):
        self._retry_count += 1
        self.get_logger().warn(
            f'{failed_state.name} 실패 ({reason}), 재시도 {self._retry_count}/{MAX_RETRY}')

        if self._retry_count > MAX_RETRY:
            self._send_report(success=False, message=f'{failed_state.name} 재시도 초과: {reason}')
            self._set_state(State.IDLE)
            self._job = None
            return

        self._recover_target_state = failed_state
        self._set_state(State.RECOVER)
        # TODO: 실제 후퇴(retreat) 동작은 arm_node/amr_node에 별도 커맨드로 위임해야 함.
        # 지금은 동일 단계를 즉시 재시도한다.
        self._set_state(self._recover_target_state)
        if failed_state == State.MOVE_TO_STATION:
            self._enter_move_to_station()
        elif failed_state in (State.GRASP_BUSBAR, State.INSERT_BUSBAR):
            self._send_busbar_command('GRASP' if failed_state == State.GRASP_BUSBAR else 'INSERT')
        elif failed_state in (State.FASTEN_APPROACH, State.FASTEN):
            self._send_fasten_command('APPROACH' if failed_state == State.FASTEN_APPROACH else 'FASTEN')

    # --- FMS 보고 -----------------------------------------------------------
    def _send_report(self, success: bool, message: str):
        report = FleetReport()
        report.job_id = self._job.job_id
        report.station_id = self._job.station_id
        report.success = success
        report.message = message
        report.stamp = self.get_clock().now().to_msg()
        self._report_pub.publish(report)
        self.get_logger().info(f'PUB /fleet/report -> {report.job_id} success={success}')


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

"""behavior_node
지휘 계층 · job 하나를 받아 한 스테이션의 조립을 끝까지 지휘.

SUB /fleet/job                                         (fms_interfaces/FleetJob)
PUB /fleet/report                                       (fms_interfaces/FleetReport)

PUB /amr/goal            SUB /amr/status                (이동)
ACTION /busbar_insert                                    (버스바 파지·삽입, GRASP -> INSERT)
ACTION /nut_fasten     (너트 체결, NUT_APPROACH -> NUT_GRASP -> FASTEN_APPROACH -> FASTEN)
SUB /vision/stud_pose · /vision/busbar_grasp · /vision/nut_pose

arm_node가 없거나 응답이 없어도 무한 대기하지 않도록, goal마다 서버 연결(server_is_ready) ->
accept 응답 -> 실행 결과 세 단계에 각각 타임아웃을 둔다(_PendingGoal, _check_pending_timeout).
실행 타임아웃 발생 시 goal 취소와 최종 result를 확인한 뒤에만 RECOVER로 진입한다.
취소 여부를 확인할 수 없으면 중복 로봇 동작을 막기 위해 해당 job을 안전 실패 처리한다.
"""
from enum import Enum, auto

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from fms_interfaces.action import BusbarInsert, NutFasten
from fms_interfaces.msg import (
    FleetJob, FleetReport,
    AmrGoal, AmrStatus,
    StudPose, BusbarGrasp, NutPose,
)

# 스테이션 좌표 (Isaac Sim 월드 기준). TODO: 실제 스테이션 배치로 교체.
STATION_POSES = {
    'station_1': (1.0, 0.0, 0.0),
    'station_2': (2.0, 0.0, 0.0),
    'station_3': (3.0, 0.0, 0.0),
}

MAX_RETRY = 3

# --- arm_node 연결/응답/실행 결과 타임아웃 ---------------------------------
SERVER_CONNECT_TIMEOUT_SEC = 3.0        # server_is_ready() 대기 한도 (2~5초 권장)
GOAL_RESPONSE_TIMEOUT_SEC = 5.0         # goal accept/reject 응답 대기 한도
GOAL_RESPONSE_CLEANUP_TIMEOUT_SEC = 5.0 # accept 지연 시 응답을 기다려 고아 goal을 정리할 시간
CANCEL_COMPLETION_TIMEOUT_SEC = 10.0    # cancel 수락 후 최종 CANCELED result 대기 한도
BUSBAR_GRASP_RESULT_TIMEOUT_SEC = 35.0  # Cartesian 이동 3회 + 그리퍼 동작
BUSBAR_INSERT_RESULT_TIMEOUT_SEC = 60.0 # 이동/정지/볼트조회/점진하강/이탈
# 너트 체결은 동작 사이 정지/정착 시간을 두기 위해 busbar처럼 goal 4개로 나눠 보낸다
# (NUT_APPROACH -> NUT_GRASP -> FASTEN_APPROACH -> FASTEN). 앞 3개는 Cartesian 이동
# (_move_to_pose 1회당 최대 8초), FASTEN만 기록 궤적 재생(1895프레임, physics_dt=1/60초).
NUT_APPROACH_RESULT_TIMEOUT_SEC = 15.0        # 이동 1회
NUT_GRASP_RESULT_TIMEOUT_SEC = 20.0           # 하강+그리퍼+상승 (이동 2회 + 그리퍼 대기)
FASTEN_APPROACH_RESULT_TIMEOUT_SEC = 30.0     # 이동 2회, 자세 수렴까지 확인(각 12초) + 여유
NUT_FASTEN_RESULT_TIMEOUT_SEC = 45.0          # 자세 정렬 램프 1초 + JSON 재생 31.58초 + 여유


class _PendingGoal:
    """action 서버 연결 대기 -> goal 전송 -> accept 대기 -> 실행 결과 대기 각 단계의
    타임아웃을 추적한다. behavior_node는 한 번에 하나의 goal만 진행하므로 인스턴스 하나로 충분.

    이 인스턴스 자체가 "현재 유효한 goal"의 identity 역할도 한다 -- 재시도로 새 _PendingGoal이
    self._pending에 들어가면, 이전 _PendingGoal을 참조하고 있는 콜백/feedback은 모두
    `self._pending is not pending` 비교로 걸러져 무시된다 (attempt_id는 이 무효화를 사람이
    읽을 수 있는 로그로 남기기 위한 것일 뿐, 식별 자체는 객체 identity로 한다)."""

    def __init__(self, client, action_name, state_on_timeout, phase, deadline,
                 goal, on_result, result_timeout_sec, attempt_id):
        self.client = client
        self.action_name = action_name
        self.state_on_timeout = state_on_timeout
        # waiting_server | waiting_accept | accept_expired | executing | canceling
        self.phase = phase
        self.deadline = deadline
        self.goal = goal
        self.on_result = on_result
        self.result_timeout_sec = result_timeout_sec
        self.attempt_id = attempt_id  # 예: "job_1/GRASP/attempt_2"
        self.goal_handle = None
        # feedback 로그 스팸 방지용 - phase 변경 또는 10% 진행마다만 로그를 남긴다.
        self.last_logged_phase = None
        self.last_logged_decile = -1
        self.failure_reason = None


# perception_node가 아직 없어 vision 토픽이 발행되지 않는 동안은 WAIT_*_VISION에서
# 무한 대기하지 않도록 건너뛴다. arm_node 쪽 좌표가 이미 하드코딩돼 있어 target_pose
# 없이도 GRASP/FASTEN 커맨드는 그대로 보낼 수 있다. perception_node 붙으면 True로.
VISION_ENABLED = False


class State(Enum):
    IDLE = auto()
    FAULT = auto()
    MOVE_TO_STATION = auto()
    WAIT_BUSBAR_VISION = auto()
    GRASP_BUSBAR = auto()
    INSERT_BUSBAR = auto()
    WAIT_NUT_VISION = auto()
    NUT_APPROACH = auto()
    NUT_GRASP = auto()
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

        # arm_node 인터페이스 (ROS2 Action - goal 실행 중 feedback으로 vision 보정값 수신)
        self._busbar_action_client = ActionClient(self, BusbarInsert, 'busbar_insert')
        self._fasten_action_client = ActionClient(self, NutFasten, 'nut_fasten')

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
        self._pending = None  # type: _PendingGoal | None
        self._attempt_counts = {}  # command(str) -> 이번 job에서 몇 번째 시도인지

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
        self._pending = None
        self._attempt_counts = {}
        self._set_state(State.MOVE_TO_STATION)

    # --- 조립 FSM 상태 전이 -------------------------------------------------
    def _set_state(self, new_state: State):
        self.get_logger().info(f'[FSM] {self._state.name} -> {new_state.name}')
        self._state = new_state

    def _step(self):
        if self._job is None:
            return

        self._check_pending_timeout()

        if self._state == State.MOVE_TO_STATION:
            self._enter_move_to_station()
        elif self._state == State.WAIT_BUSBAR_VISION:
            if self._latest_busbar_grasp is not None:
                self._set_state(State.GRASP_BUSBAR)
                self._send_busbar_goal('GRASP')
        elif self._state == State.WAIT_NUT_VISION:
            if self._latest_stud_pose is not None and self._latest_nut_pose is not None:
                self._set_state(State.NUT_APPROACH)
                self._send_fasten_goal('NUT_APPROACH')
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
            if VISION_ENABLED:
                self._set_state(State.WAIT_BUSBAR_VISION)
            else:
                self._set_state(State.GRASP_BUSBAR)
                self._send_busbar_goal('GRASP')
        elif msg.state == AmrStatus.STATE_ERROR:
            self._move_goal_sent = False
            self._enter_recover(State.MOVE_TO_STATION, msg.message)

    # --- arm_node 서버 연결 · 응답 · 실행 결과 타임아웃 -------------------------
    def _next_attempt_id(self, command: str) -> str:
        """job_id/command/attempt_N 형태의 실행 ID. 로그 추적용이며, 재시도 이후 이전
        goal의 콜백/feedback을 사람이 로그에서 구분할 수 있게 해준다."""
        n = self._attempt_counts.get(command, 0) + 1
        self._attempt_counts[command] = n
        job_id = self._job.job_id if self._job is not None else '?'
        return f'{job_id}/{command}/attempt_{n}'

    def _send_action_goal(self, client, goal, action_name, command, on_result,
                           result_timeout_sec):
        """server_is_ready() 확인부터 goal 전송, accept/실행 결과까지 타임아웃을 추적하며
        진행한다. 서버가 아직 준비 안 됐으면 즉시 블로킹하지 않고 _step()이 폴링하며 재시도."""
        attempt_id = self._next_attempt_id(command)
        if client.server_is_ready():
            self._dispatch_goal(client, goal, action_name, on_result, result_timeout_sec,
                                 attempt_id)
            return

        self.get_logger().warn(f'[{attempt_id}] {action_name} 서버 연결 대기 중...')
        deadline = self.get_clock().now() + Duration(seconds=SERVER_CONNECT_TIMEOUT_SEC)
        self._pending = _PendingGoal(
            client, action_name, self._state, 'waiting_server', deadline,
            goal, on_result, result_timeout_sec, attempt_id)

    def _dispatch_goal(self, client, goal, action_name, on_result, result_timeout_sec,
                        attempt_id):
        deadline = self.get_clock().now() + Duration(seconds=GOAL_RESPONSE_TIMEOUT_SEC)
        pending = _PendingGoal(
            client, action_name, self._state, 'waiting_accept', deadline,
            goal, on_result, result_timeout_sec, attempt_id)
        self._pending = pending
        self.get_logger().info(f'[{attempt_id}] goal 전송')
        send_future = client.send_goal_async(
            goal, feedback_callback=lambda msg: self._on_feedback(msg, pending))
        send_future.add_done_callback(lambda f: self._on_goal_response(f, pending))

    def _on_feedback(self, feedback_msg, pending: _PendingGoal):
        if self._pending is not pending:
            self.get_logger().debug(f'[{pending.attempt_id}] 지연된 feedback 무시 (이미 재시도됨)')
            return
        # 콘솔 부하를 줄이기 위해 phase가 바뀌거나 진행률이 새 10% 단위를 넘을 때만 로그.
        fb = feedback_msg.feedback
        decile = int(fb.progress * 10)
        if fb.phase != pending.last_logged_phase or decile != pending.last_logged_decile:
            self.get_logger().info(
                f'[{pending.attempt_id}] {pending.action_name} phase={fb.phase} '
                f'progress={fb.progress:.2f}')
            pending.last_logged_phase = fb.phase
            pending.last_logged_decile = decile

    def _on_goal_response(self, future, pending: _PendingGoal):
        try:
            goal_handle = future.result()
        except Exception as exc:
            if self._pending is not pending:
                return
            self.get_logger().error(f'[{pending.attempt_id}] goal 응답 처리 중 예외: {exc}')
            # 요청이 서버에 도착했는지 확정할 수 없으므로 자동 재시도하지 않는다.
            self._fail_job_safely(
                f'{pending.action_name} goal 수락 여부 확인 실패: {exc}')
            return

        # 이미 다른 요청으로 넘어간 뒤 늦게 accept된 goal은 고아 상태로 실행되지 않도록
        # 반드시 취소한다.
        if self._pending is not pending:
            if goal_handle.accepted:
                self.get_logger().error(
                    f'[{pending.attempt_id}] 무효화 뒤 늦게 accept됨 -> 고아 goal 취소 요청')
                goal_handle.cancel_goal_async()
            return

        if not goal_handle.accepted:
            self._pending = None
            self._enter_recover(pending.state_on_timeout, f'{pending.action_name} goal이 거부됨')
            return

        pending.goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda f: self._on_goal_result(f, pending))

        if pending.phase == 'accept_expired':
            self.get_logger().warn(
                f'[{pending.attempt_id}] 타임아웃 뒤 goal accept 확인 -> 실행 취소 요청')
            self._begin_cancel(pending, pending.failure_reason)
            return

        pending.phase = 'executing'
        pending.deadline = self.get_clock().now() + Duration(seconds=pending.result_timeout_sec)

    def _on_goal_result(self, future, pending: _PendingGoal):
        if self._pending is not pending:
            self.get_logger().debug(f'[{pending.attempt_id}] 지연된 result 무시 (이미 재시도됨)')
            return

        try:
            result_response = future.result()
            result = result_response.result
        except Exception as exc:
            self._pending = None
            self.get_logger().error(f'[{pending.attempt_id}] 실행 결과 처리 중 예외: {exc}')
            if pending.phase == 'canceling':
                self._fail_job_safely(
                    f'{pending.action_name} 취소 후 최종 상태 확인 실패: {exc}')
            else:
                self._enter_recover(
                    pending.state_on_timeout, f'{pending.action_name} 결과 예외: {exc}')
            return

        self._pending = None
        if pending.phase == 'canceling':
            # timeout/cancel 요청과 실제 완료가 엇갈린 경우, 서버가 성공으로 확정한
            # 물리 동작을 다시 실행하지 말고 정상 결과로 이어간다.
            if result_response.status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(
                    f'[{pending.attempt_id}] 취소 요청 전 정상 완료 확인, 재시도하지 않음')
                pending.on_result(result)
                return
            self.get_logger().info(f'[{pending.attempt_id}] 기존 goal 종료 확인, 이제 재시도')
            self._enter_recover(
                pending.state_on_timeout,
                pending.failure_reason or f'{pending.action_name} 실행 취소됨')
            return

        pending.on_result(result)

    def _begin_cancel(self, pending: _PendingGoal, reason: str):
        """실행 중 goal을 취소한다. 최종 result 전에는 새 goal을 보내지 않는다."""
        if self._pending is not pending:
            return
        if pending.goal_handle is None:
            self._fail_job_safely(
                f'{pending.action_name} goal handle이 없어 취소 상태를 확인할 수 없음')
            return

        pending.phase = 'canceling'
        pending.failure_reason = reason
        pending.deadline = (
            self.get_clock().now() + Duration(seconds=CANCEL_COMPLETION_TIMEOUT_SEC))
        try:
            cancel_future = pending.goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(
                lambda f: self._on_cancel_response(f, pending))
        except Exception as exc:
            self._fail_job_safely(f'{pending.action_name} 취소 요청 실패: {exc}')

    def _on_cancel_response(self, future, pending: _PendingGoal):
        if self._pending is not pending:
            return
        try:
            response = future.result()
        except Exception as exc:
            self._fail_job_safely(f'{pending.action_name} 취소 응답 실패: {exc}')
            return

        if response.goals_canceling:
            self.get_logger().info(
                f'[{pending.attempt_id}] 취소 수락됨, 최종 result 대기 중')
        else:
            # result 콜백과 취소 응답 순서가 뒤바뀔 수 있으므로 timeout까지 기다린다.
            self.get_logger().warn(
                f'[{pending.attempt_id}] 취소 대상 없음, 최종 result 확인 대기')

    def _check_pending_timeout(self):
        pending = self._pending
        if pending is None:
            return

        if pending.phase == 'waiting_server':
            if pending.client.server_is_ready():
                self._dispatch_goal(pending.client, pending.goal, pending.action_name,
                                     pending.on_result, pending.result_timeout_sec,
                                     pending.attempt_id)
                return

        if self.get_clock().now() < pending.deadline:
            return

        if pending.phase == 'waiting_server':
            self._pending = None
            self._enter_recover(pending.state_on_timeout,
                                 f'{pending.action_name} 서버 연결 타임아웃 [{pending.attempt_id}]')
        elif pending.phase == 'waiting_accept':
            # 수락 여부를 모르는 상태로 재시도하면 첫 goal과 동작이 겹칠 수 있다.
            pending.phase = 'accept_expired'
            pending.failure_reason = (
                f'{pending.action_name} goal 응답 타임아웃 [{pending.attempt_id}]')
            pending.deadline = (
                self.get_clock().now()
                + Duration(seconds=GOAL_RESPONSE_CLEANUP_TIMEOUT_SEC))
            self.get_logger().error(
                f'[{pending.attempt_id}] goal 응답 타임아웃, 늦은 accept 정리 대기')
        elif pending.phase == 'accept_expired':
            self._fail_job_safely(
                f'{pending.action_name} goal 수락 여부를 확인할 수 없어 중복 실행 방지 중단')
        elif pending.phase == 'executing':
            self.get_logger().warn(f'[{pending.attempt_id}] 실행 결과 타임아웃 -> 취소 요청')
            self._begin_cancel(
                pending,
                f'{pending.action_name} 실행 결과 타임아웃 [{pending.attempt_id}]')
        elif pending.phase == 'canceling':
            self._fail_job_safely(
                f'{pending.action_name} 취소 후 최종 상태 확인 타임아웃 '
                f'[{pending.attempt_id}]')

    # --- 버스바 파지 · 삽입 --------------------------------------------------
    def _on_busbar_grasp(self, msg: BusbarGrasp):
        self._latest_busbar_grasp = msg

    def _send_busbar_goal(self, command: str):
        goal = BusbarInsert.Goal()
        goal.command = command
        goal.station_id = self._job.station_id
        if self._latest_busbar_grasp is not None:
            goal.target_pose = self._latest_busbar_grasp.pose.pose
        self.get_logger().info(f'ACTION /busbar_insert 요청 -> {command}')
        result_timeout = (
            BUSBAR_GRASP_RESULT_TIMEOUT_SEC if command == 'GRASP'
            else BUSBAR_INSERT_RESULT_TIMEOUT_SEC)
        self._send_action_goal(
            self._busbar_action_client, goal, 'busbar_insert', command,
            on_result=self._on_busbar_result, result_timeout_sec=result_timeout)

    def _on_busbar_result(self, result):
        if not result.success:
            self._enter_recover(self._state, result.message)
            return

        if self._state == State.GRASP_BUSBAR:
            self._set_state(State.INSERT_BUSBAR)
            self._send_busbar_goal('INSERT')
        elif self._state == State.INSERT_BUSBAR:
            if VISION_ENABLED:
                self._set_state(State.WAIT_NUT_VISION)
            else:
                self._set_state(State.NUT_APPROACH)
                self._send_fasten_goal('NUT_APPROACH')

    # --- 너트 체결 시퀀스 ----------------------------------------------------
    def _on_stud_pose(self, msg: StudPose):
        self._latest_stud_pose = msg

    def _on_nut_pose(self, msg: NutPose):
        self._latest_nut_pose = msg

    _FASTEN_RESULT_TIMEOUTS = {
        'NUT_APPROACH': NUT_APPROACH_RESULT_TIMEOUT_SEC,
        'NUT_GRASP': NUT_GRASP_RESULT_TIMEOUT_SEC,
        'FASTEN_APPROACH': FASTEN_APPROACH_RESULT_TIMEOUT_SEC,
        'FASTEN': NUT_FASTEN_RESULT_TIMEOUT_SEC,
    }

    def _send_fasten_goal(self, command: str):
        goal = NutFasten.Goal()
        goal.command = command
        goal.nut_id = str(self._latest_nut_pose.id) if self._latest_nut_pose else ''
        self.get_logger().info(f'ACTION /nut_fasten 요청 -> {command}')
        result_timeout = self._FASTEN_RESULT_TIMEOUTS[command]
        self._send_action_goal(
            self._fasten_action_client, goal, 'nut_fasten', command,
            on_result=self._on_fasten_result, result_timeout_sec=result_timeout)

    def _on_fasten_result(self, result):
        if not result.success:
            self._enter_recover(self._state, result.message)
            return

        if self._state == State.NUT_APPROACH:
            self._set_state(State.NUT_GRASP)
            self._send_fasten_goal('NUT_GRASP')
        elif self._state == State.NUT_GRASP:
            self._set_state(State.FASTEN_APPROACH)
            self._send_fasten_goal('FASTEN_APPROACH')
        elif self._state == State.FASTEN_APPROACH:
            self._set_state(State.FASTEN)
            self._send_fasten_goal('FASTEN')
        elif self._state == State.FASTEN:
            self.get_logger().info(
                f'너트 체결 Action 완료: {result.message} (reported torque={result.torque:.2f} Nm)')
            self._set_state(State.REPORT)

    # --- 복구 로직 ----------------------------------------------------------
    def _fail_job_safely(self, reason: str):
        """Action 종료 여부가 불명확하면 재시도와 새 job 수락을 모두 차단한다."""
        self.get_logger().error(f'안전 중단: {reason}')
        self._pending = None
        if self._job is not None:
            self._send_report(success=False, message=f'안전 중단: {reason}')
        self._move_goal_sent = False
        self._set_state(State.FAULT)
        self._job = None
        self.get_logger().error(
            'Action 상태가 불명확해 FAULT로 잠금. arm/behavior 노드 상태 확인 후 재시작 필요')

    def _enter_recover(self, failed_state: State, reason: str):
        self._retry_count += 1
        self.get_logger().warn(
            f'{failed_state.name} 실패 ({reason}), 재시도 {self._retry_count}/{MAX_RETRY}')

        if self._retry_count > MAX_RETRY:
            self._send_report(success=False, message=f'{failed_state.name} 재시도 초과: {reason}')
            self._move_goal_sent = False
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
            self._send_busbar_goal('GRASP' if failed_state == State.GRASP_BUSBAR else 'INSERT')
        elif failed_state in (State.NUT_APPROACH, State.NUT_GRASP,
                               State.FASTEN_APPROACH, State.FASTEN):
            # state 이름이 곧 nut_fasten의 command 문자열이라 그대로 재사용한다.
            self._send_fasten_goal(failed_state.name)

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
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            try:
                node.destroy_node()
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()

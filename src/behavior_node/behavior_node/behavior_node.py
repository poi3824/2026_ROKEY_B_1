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
import math
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

# 아래 값은 Isaac Sim Property 패널에서 읽은 고정 월드 좌표다. behavior_node는 이 값을
# 변경하지 않고 AmrGoal로 보낸다. /odom의 HOME 상대 좌표를 월드 좌표로 바꾸는 책임은
# amr_node에 두어, 사용자가 지정한 목표 좌표와 로그에 표시되는 목표 좌표가 항상 같게 한다.
STATION_POSES = {
    'station_1': (0.66594, -0.04555, -math.pi / 2),
    'station_2': (0.66594, -0.64732, -math.pi / 2),
    'station_3': (0.66594, -1.17289, -math.pi / 2),
    'station_4': (1.98852, -1.17289, -math.pi / 2),
    'station_5': (1.98852, -0.64732, -math.pi / 2),
    'station_6': (1.98852, -0.1118, -math.pi / 2),
    # ST3→ST4 직선은 배터리 모듈팩 하단 모서리를 관통한다. ST3에서 남쪽으로 충분히
    # 이탈한 뒤 가로질러, ST4 남쪽에서 -90도 자세로 직선 후진 진입한다.
    'scan_3_exit': (0.66594, -1.75, 0.0),
    # ST4 열보다 10cm 더 전진한 뒤 반시계 정렬을 시작해야 차체 후미가 모듈팩
    # 모서리를 충분히 벗어나고, 회전 후 ST4~ST6 직선과 나란히 설 수 있다.
    'scan_4_approach': (2.08852, -1.75, math.pi / 2),
    # 스캔 구간에서는 ST4 아래에서 반시계로 정렬한 +90도 자세를 유지한 채
    # ST4→ST5→ST6을 한 줄로 전진한다. 실제 체결용 station_4~6 자세(-90도)는 보존한다.
    'scan_station_4': (1.98852, -1.17289, math.pi / 2),
    'scan_station_5': (1.98852, -0.64732, math.pi / 2),
    'scan_station_6': (1.98852, -0.1118, math.pi / 2),
    # 전체 스캔 후 ST4 쪽에서 ST3 쪽으로 되돌아올 때 사용하는 동일 위치/복귀 자세.
    # 여기서 -90도로 정렬하면 ST3까지 직선 후진할 수 있어 스테이션 앞 회전을 피한다.
    'scan_3_return': (0.66594, -1.75, -math.pi / 2),
    # 전체 스캔 종료 후 그림의 빨간 경로처럼 모듈팩 오른쪽과 위쪽을 돌아 BUS1으로 간다.
    # 각 코너에서 다음 직선 구간 방향으로 미리 자세를 맞춰 전진 주행을 유지한다.
    # ST6 스캔 직후에는 회전하지 않는다. ST4→ST5→ST6에서 유지한 +90도 방향으로
    # 모듈팩 끝보다 0.66m 더 직진한 뒤에만 BUS1 상단 우회를 시작한다.
    'scan_6_straight_exit': (1.98852, 0.55, math.pi / 2),
    'scan_6_right_exit': (3.10, 0.55, 0.0),
    'scan_6_top_right': (3.10, 1.92483, math.pi / 2),
    'busbar_return_approach': (-1.47361, 1.92483, math.pi),
    # ST1과 BUS1 사이 안전 중간점. 여기서 바로 BUS1로 가지 않고 아래 도킹 정렬점을 거친다.
    'busbar_approach': (-1.47361, 1.92483, 0.0),
    # BUS1 왼쪽 1.2m 지점. 기존 -1.20211은 테이블 왼쪽 모서리와 너무 가까워 팔이
    # 제자리 회전 중 테이블에 닿았다. 충분히 왼쪽으로 빠진 이 위치에서 -180도로 정렬한 뒤
    # BUS1까지 직선 후진하여 실제 파지 자세를 유지한다.
    'busbar_dock_approach': (-1.70211, 2.37274, -math.pi),
    'busbar_table': (-0.50211, 2.37274, -math.pi),  # BUS1
    # BUS1 파지 후 복귀 전용 통과점. 진입용 좌표와 위치는 같지만 amr_node에서 최종
    # 제자리 회전을 생략하도록 별도 ID를 사용한다.
    'busbar_dock_exit': (-1.70211, 2.37274, -math.pi),
    'busbar_return_mid': (-1.47361, 1.92483, 0.0),
    'station_1_return': (0.66594, -0.04555, -math.pi / 2),
    # 수정본 좌표: 평행주차 없이 -90도 자세로 남쪽에서 북쪽 방향으로 직선 후진 진입.
    'busbar_2_approach': (0.48533, 1.47778, -math.pi / 2),
    'busbar_table_2': (0.48533, 2.07778, -math.pi / 2),
    'busbar_3_approach': (1.38165, 1.47778, -math.pi / 2),
    'busbar_table_3': (1.38165, 2.07778, -math.pi / 2),
}

# arm_node/perception_node를 이번 라운드에서는 아직 실행/연결하지 않는다 - 스캔/버스바 파지/
# 체결은 SCAN_STUB/GRASP_STUB/FASTEN_STUB(제자리 대기)로 대체한다. 새 씬(Collected_Busbar3,
# onrobot_rg2 그리퍼)에 맞게 arm_node의 하드코딩 좌표를 재보정한 뒤 True로 바꾸면, 아래
# _on_amr_status의 분기를 통해 기존 GRASP_BUSBAR/NUT_* 액션 흐름이 그대로 다시 활성화된다.
ARM_ACTIONS_ENABLED = False

STUB_WAIT_SEC = 2.0

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
    MOVE_TO_SCAN_STATION = auto()
    SCAN_STUB = auto()
    MOVE_TO_BUS_APPROACH = auto()
    MOVE_TO_BUS_DOCK_APPROACH = auto()
    MOVE_TO_BUSBAR_TABLE = auto()
    GRASP_STUB = auto()
    MOVE_BACK_TO_BUS_DOCK_APPROACH = auto()
    MOVE_BACK_TO_BUS_APPROACH = auto()
    MOVE_BACK_TO_STATION = auto()
    FASTEN_STUB = auto()
    MOVE_TO_BUSBAR2_APPROACH = auto()
    MOVE_TO_BUSBAR2_TABLE = auto()
    GRASP2_STUB = auto()
    MOVE_TO_STATION2 = auto()
    FASTEN2_STUB = auto()
    MOVE_TO_BUSBAR3_APPROACH = auto()
    MOVE_TO_BUSBAR3_TABLE = auto()
    GRASP3_STUB = auto()
    MOVE_TO_STATION3 = auto()
    FASTEN3_STUB = auto()
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


STATE_DISPLAY = {
    State.IDLE: ('대기', '-'),
    State.FAULT: ('오류 정지', '!'),
    State.MOVE_TO_STATION: ('ST1 스캔 위치로 이동', '1/31'),
    State.MOVE_TO_SCAN_STATION: ('다음 스캔 위치로 이동', '스캔'),
    State.SCAN_STUB: ('비전 스캔', '스캔'),
    State.MOVE_TO_BUS_APPROACH: ('BUS1 중간 접근점으로 이동', '13/31'),
    State.MOVE_TO_BUS_DOCK_APPROACH: ('BUS1 도킹 자세 정렬점으로 이동', '14/31'),
    State.MOVE_TO_BUSBAR_TABLE: ('BUS1 파지 위치로 정밀 이동', '15/31'),
    State.GRASP_STUB: ('BUS1 버스바 파지', '16/31'),
    State.MOVE_BACK_TO_BUS_DOCK_APPROACH: ('BUS1에서 직선으로 이탈', '17/31'),
    State.MOVE_BACK_TO_BUS_APPROACH: ('BUS1 중간점으로 복귀', '18/31'),
    State.MOVE_BACK_TO_STATION: ('ST1으로 복귀', '19/31'),
    State.FASTEN_STUB: ('ST1 버스바·너트 체결', '20/31'),
    State.MOVE_TO_BUSBAR2_APPROACH: ('BUS2 직선 접근점으로 이동', '21/31'),
    State.MOVE_TO_BUSBAR2_TABLE: ('BUS2 파지 위치로 정밀 이동', '22/31'),
    State.GRASP2_STUB: ('BUS2 버스바 파지', '23/31'),
    State.MOVE_TO_STATION2: ('ST2로 이동', '24/31'),
    State.FASTEN2_STUB: ('ST2 버스바·너트 체결', '25/31'),
    State.MOVE_TO_BUSBAR3_APPROACH: ('BUS3 직선 접근점으로 이동', '26/31'),
    State.MOVE_TO_BUSBAR3_TABLE: ('BUS3 파지 위치로 정밀 이동', '27/31'),
    State.GRASP3_STUB: ('BUS3 버스바 파지', '28/31'),
    State.MOVE_TO_STATION3: ('ST3로 이동', '29/31'),
    State.FASTEN3_STUB: ('ST3 버스바·너트 체결', '30/31'),
    State.REPORT: ('1·2·3차 작업 완료 보고', '31/31'),
}


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
        self._stub_deadline = None  # SCAN_STUB/GRASP_STUB/FASTEN_STUB 공용 대기 타이머
        self._stub_label = None
        self._stub_last_remaining_sec = None
        self._scan_station_number = 1
        self._scan_route_queue = []
        self._active_scan_waypoint = None
        self._bus1_route_queue = []
        self._active_bus1_waypoint = None
        self._bus3_route_queue = []
        self._active_bus3_waypoint = None
        self._station3_route_queue = []
        self._active_station3_waypoint = None

        self._timer = self.create_timer(0.5, self._step)

        self.get_logger().info(
            '\n' + '=' * 72 +
            '\n[BEHAVIOR] 작업 지휘 노드 시작 — 새 작업을 기다립니다.' +
            '\n' + '=' * 72)

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
        self._scan_station_number = 1
        self._scan_route_queue = []
        self._active_scan_waypoint = None
        self._bus1_route_queue = []
        self._active_bus1_waypoint = None
        self._bus3_route_queue = []
        self._active_bus3_waypoint = None
        self._station3_route_queue = []
        self._active_station3_waypoint = None
        self._set_state(State.MOVE_TO_STATION)

    # --- 조립 FSM 상태 전이 -------------------------------------------------
    def _set_state(self, new_state: State):
        old_label = STATE_DISPLAY.get(self._state, (self._state.name, '?'))[0]
        new_label, step = STATE_DISPLAY.get(new_state, (new_state.name, '?'))
        if new_state == State.MOVE_TO_SCAN_STATION:
            new_label = f'ST{self._scan_station_number} 스캔 위치로 이동'
            step = f'{self._scan_station_number * 2 - 1}/31'
        elif new_state == State.SCAN_STUB:
            new_label = f'ST{self._scan_station_number} 비전 스캔'
            step = f'{self._scan_station_number * 2}/31'
        self.get_logger().info(
            '\n' + '=' * 72 +
            f'\n[단계 {step}] {new_label}' +
            f'\n상태 전환: {old_label}  →  {new_label}' +
            '\n' + '=' * 72)
        self._state = new_state

    def _step(self):
        if self._job is None:
            return

        self._check_pending_timeout()

        if self._state == State.MOVE_TO_STATION:
            self._enter_move_to_station()
        elif self._state == State.MOVE_TO_SCAN_STATION:
            self._enter_move_to_scan_station()
        elif self._state == State.SCAN_STUB:
            self._enter_scan_stub()
        elif self._state == State.MOVE_TO_BUS_APPROACH:
            self._enter_move_to_bus_approach()
        elif self._state == State.MOVE_TO_BUS_DOCK_APPROACH:
            self._enter_move_to_bus_dock_approach()
        elif self._state == State.MOVE_TO_BUSBAR_TABLE:
            self._enter_move_to_busbar_table()
        elif self._state == State.GRASP_STUB:
            self._enter_grasp_stub()
        elif self._state == State.MOVE_BACK_TO_BUS_DOCK_APPROACH:
            self._enter_move_back_to_bus_dock_approach()
        elif self._state == State.MOVE_BACK_TO_BUS_APPROACH:
            self._enter_move_back_to_bus_approach()
        elif self._state == State.MOVE_BACK_TO_STATION:
            self._enter_move_back_to_station()
        elif self._state == State.FASTEN_STUB:
            self._enter_fasten_stub()
        elif self._state == State.MOVE_TO_BUSBAR2_APPROACH:
            self._enter_move_to_busbar2_approach()
        elif self._state == State.MOVE_TO_BUSBAR2_TABLE:
            self._enter_move_to_busbar2_table()
        elif self._state == State.GRASP2_STUB:
            self._enter_grasp2_stub()
        elif self._state == State.MOVE_TO_STATION2:
            self._enter_move_to_station2()
        elif self._state == State.FASTEN2_STUB:
            self._enter_fasten2_stub()
        elif self._state == State.MOVE_TO_BUSBAR3_APPROACH:
            self._enter_move_to_busbar3_approach()
        elif self._state == State.MOVE_TO_BUSBAR3_TABLE:
            self._enter_move_to_busbar3_table()
        elif self._state == State.GRASP3_STUB:
            self._enter_grasp3_stub()
        elif self._state == State.MOVE_TO_STATION3:
            self._enter_move_to_station3()
        elif self._state == State.FASTEN3_STUB:
            self._enter_fasten3_stub()
        elif self._state == State.WAIT_BUSBAR_VISION:
            if self._latest_busbar_grasp is not None:
                self._set_state(State.GRASP_BUSBAR)
                self._send_busbar_goal('GRASP')
        elif self._state == State.WAIT_NUT_VISION:
            if self._latest_stud_pose is not None and self._latest_nut_pose is not None:
                self._set_state(State.NUT_APPROACH)
                self._send_fasten_goal('NUT_APPROACH')
        elif self._state == State.REPORT:
            self._send_report(success=True, message='전체 비전 스캔 및 ST1·ST2·ST3 버스바·너트 체결 완료')
            self._set_state(State.IDLE)
            self._job = None

    # --- 이동 -------------------------------------------------------------
    # MOVE_TO_STATION / MOVE_TO_BUSBAR_TABLE / MOVE_BACK_TO_STATION 3개 state가 공유하는
    # "waypoint_id로 AmrGoal 1회 발행 후 도착 대기" 로직.
    def _enter_travel(self, waypoint_id: str):
        if getattr(self, '_move_goal_sent', False):
            return
        pose = STATION_POSES.get(waypoint_id)
        if pose is None:
            self.get_logger().error(f'알 수 없는 AMR waypoint: {waypoint_id}')
            return
        x, y, theta = pose
        goal = AmrGoal()
        goal.station_id = waypoint_id
        goal.x, goal.y, goal.theta = x, y, theta
        self._amr_goal_pub.publish(goal)
        self._move_goal_sent = True
        self.get_logger().info(
            f'[이동 명령] 목적지={goal.station_id} | '
            f'월드 좌표 X={x:.3f}, Y={y:.3f}, 방향={math.degrees(theta):.1f}°')

    def _enter_move_to_station(self):
        self._enter_travel(self._job.station_id)

    def _enter_move_to_scan_station(self):
        if self._active_scan_waypoint is None:
            if not self._scan_route_queue:
                waypoint_prefix = (
                    'scan_station'
                    if self._scan_station_number >= 4
                    else 'station'
                )
                self._scan_route_queue = [
                    f'{waypoint_prefix}_{self._scan_station_number}'
                ]
            self._active_scan_waypoint = self._scan_route_queue.pop(0)
        self._enter_travel(self._active_scan_waypoint)

    def _enter_move_to_bus_approach(self):
        if self._active_bus1_waypoint is None:
            if not self._bus1_route_queue:
                self._bus1_route_queue = ['busbar_approach']
            self._active_bus1_waypoint = self._bus1_route_queue.pop(0)
        self._enter_travel(self._active_bus1_waypoint)

    def _enter_move_to_bus_dock_approach(self):
        self._enter_travel('busbar_dock_approach')

    def _enter_move_to_busbar_table(self):
        self._enter_travel('busbar_table')

    def _enter_move_back_to_bus_dock_approach(self):
        self._enter_travel('busbar_dock_exit')

    def _enter_move_back_to_bus_approach(self):
        self._enter_travel('busbar_return_mid')

    def _enter_move_back_to_station(self):
        self._enter_travel('station_1_return')

    def _enter_move_to_busbar2_approach(self):
        self._enter_travel('busbar_2_approach')

    def _enter_move_to_busbar2_table(self):
        self._enter_travel('busbar_table_2')

    def _enter_move_to_station2(self):
        self._enter_travel('station_2')

    def _enter_move_to_busbar3_approach(self):
        if self._active_bus3_waypoint is None:
            if not self._bus3_route_queue:
                self._bus3_route_queue = [
                    'station_1',
                    'busbar_2_approach',
                    'busbar_3_approach',
                ]
            self._active_bus3_waypoint = self._bus3_route_queue.pop(0)
        self._enter_travel(self._active_bus3_waypoint)

    def _enter_move_to_busbar3_table(self):
        self._enter_travel('busbar_table_3')

    def _enter_move_to_station3(self):
        if self._active_station3_waypoint is None:
            if not self._station3_route_queue:
                self._station3_route_queue = [
                    'busbar_3_approach',
                    'busbar_2_approach',
                    'station_1',
                    'station_2',
                    'station_3',
                ]
            self._active_station3_waypoint = self._station3_route_queue.pop(0)
        self._enter_travel(self._active_station3_waypoint)

    # state 도착 시 다음 state로 - ARM_ACTIONS_ENABLED=True로 arm_node 연동을 켜기 전까지는
    # MOVE_TO_STATION 도착 후에도 (WAIT_BUSBAR_VISION/GRASP_BUSBAR 대신) SCAN_STUB으로 간다.
    _TRAVEL_NEXT_STATE = {
        State.MOVE_TO_STATION: State.SCAN_STUB,
        State.MOVE_TO_SCAN_STATION: State.SCAN_STUB,
        State.MOVE_TO_BUS_APPROACH: State.MOVE_TO_BUS_DOCK_APPROACH,
        State.MOVE_TO_BUS_DOCK_APPROACH: State.MOVE_TO_BUSBAR_TABLE,
        State.MOVE_TO_BUSBAR_TABLE: State.GRASP_STUB,
        State.MOVE_BACK_TO_BUS_DOCK_APPROACH: State.MOVE_BACK_TO_BUS_APPROACH,
        State.MOVE_BACK_TO_BUS_APPROACH: State.MOVE_BACK_TO_STATION,
        State.MOVE_BACK_TO_STATION: State.FASTEN_STUB,
        State.MOVE_TO_BUSBAR2_APPROACH: State.MOVE_TO_BUSBAR2_TABLE,
        State.MOVE_TO_BUSBAR2_TABLE: State.GRASP2_STUB,
        State.MOVE_TO_STATION2: State.FASTEN2_STUB,
        State.MOVE_TO_BUSBAR3_APPROACH: State.MOVE_TO_BUSBAR3_TABLE,
        State.MOVE_TO_BUSBAR3_TABLE: State.GRASP3_STUB,
        State.MOVE_TO_STATION3: State.FASTEN3_STUB,
    }

    def _on_amr_status(self, msg: AmrStatus):
        if self._state not in self._TRAVEL_NEXT_STATE:
            return
        if msg.state == AmrStatus.STATE_ARRIVED:
            self._move_goal_sent = False
            if self._state == State.MOVE_TO_SCAN_STATION:
                self._active_scan_waypoint = None
                if self._scan_route_queue:
                    self.get_logger().info(
                        f'[안전 우회] 다음 경유지로 이동 | '
                        f'남은 경유지={len(self._scan_route_queue)}개')
                    return
            elif self._state == State.MOVE_TO_BUS_APPROACH:
                self._active_bus1_waypoint = None
                if self._bus1_route_queue:
                    self.get_logger().info(
                        f'[ST6→BUS1 안전 복귀] 외곽 경로의 다음 지점으로 이동 | '
                        f'남은 경유지={len(self._bus1_route_queue)}개')
                    return
            elif self._state == State.MOVE_TO_BUSBAR3_APPROACH:
                self._active_bus3_waypoint = None
                if self._bus3_route_queue:
                    self.get_logger().info(
                        f'[ST2→BUS3 안전 이동] 외곽 경로의 다음 지점으로 이동 | '
                        f'남은 경유지={len(self._bus3_route_queue)}개')
                    return
            elif self._state == State.MOVE_TO_STATION3:
                self._active_station3_waypoint = None
                if self._station3_route_queue:
                    self.get_logger().info(
                        f'[BUS3→ST3 안전 복귀] 외곽 경로의 다음 지점으로 이동 | '
                        f'남은 경유지={len(self._station3_route_queue)}개')
                    return
            if self._state == State.MOVE_TO_STATION and ARM_ACTIONS_ENABLED:
                if VISION_ENABLED:
                    self._set_state(State.WAIT_BUSBAR_VISION)
                else:
                    self._set_state(State.GRASP_BUSBAR)
                    self._send_busbar_goal('GRASP')
            else:
                next_state = self._TRAVEL_NEXT_STATE[self._state]
                self._set_state(next_state)
                # BUS1 복귀 통과점은 다음 0.5초 FSM tick을 기다리지 않고 즉시 다음
                # 목표를 발행해 정지 시간이 눈에 띄지 않게 한다.
                if next_state == State.MOVE_BACK_TO_BUS_APPROACH:
                    self._enter_move_back_to_bus_approach()
                elif next_state == State.MOVE_BACK_TO_STATION:
                    self._enter_move_back_to_station()
        elif msg.state == AmrStatus.STATE_ERROR:
            self._move_goal_sent = False
            self._enter_recover(self._state, msg.message)

    # --- 스캔/파지/체결 스텁 (arm_node/perception_node 연결 전 임시 동작) --------------
    # STUB_WAIT_SEC만큼 제자리 대기만 하고 다음 단계로 넘어간다. arm_node 쪽 좌표가 새 씬
    # (Collected_Busbar3)에 맞게 준비되면, 아래 3개 메서드 내부만 실제 액션 호출
    # (_send_busbar_goal/_send_fasten_goal)로 바꾸면 되고 FSM 흐름 자체는 안 건드려도 된다.
    def _enter_stub_wait(self, label: str) -> bool:
        """대기가 끝난 첫 tick에만 True. 반환 후 타이머를 리셋해 다음 스텁 state가 이어서
        새로 잡을 수 있게 한다."""
        if self._stub_deadline is None:
            self._stub_deadline = self.get_clock().now() + Duration(seconds=STUB_WAIT_SEC)
            self._stub_label = label
            self._stub_last_remaining_sec = math.ceil(STUB_WAIT_SEC)
            self.get_logger().info(
                f'[{label} 중...] 로봇 정지 유지 | 예상 시간 {STUB_WAIT_SEC:.1f}초')
            return False

        now = self.get_clock().now()
        if now >= self._stub_deadline:
            self.get_logger().info(f'[{label} 완료] 다음 단계로 이동합니다.')
            self._stub_deadline = None
            self._stub_label = None
            self._stub_last_remaining_sec = None
            return True

        remaining = max(0.0, (self._stub_deadline - now).nanoseconds / 1e9)
        remaining_sec = math.ceil(remaining)
        if remaining_sec != self._stub_last_remaining_sec:
            self._stub_last_remaining_sec = remaining_sec
            self.get_logger().info(f'[{label} 중...] 남은 시간 약 {remaining_sec}초')
        return False

    def _enter_scan_stub(self):
        station_number = self._scan_station_number
        if self._enter_stub_wait(f'ST{station_number} 비전 스캔'):
            if station_number < 6:
                self._scan_station_number += 1
                if station_number == 3:
                    self._scan_route_queue = [
                        'scan_3_exit',
                        'scan_4_approach',
                        'scan_station_4',
                    ]
                    self.get_logger().info(
                        '[ST3→ST4 안전 우회] 모듈팩 하단을 피해 '
                        'ST3 이탈점 → ST4 접근점 → ST4 순서로 이동합니다.')
                self._set_state(State.MOVE_TO_SCAN_STATION)
            else:
                self.get_logger().info(
                    '[전체 스캔 완료] ST1 → ST2 → ST3 → ST4 → ST5 → ST6')
                self._bus1_route_queue = [
                    'scan_6_straight_exit',
                    'busbar_return_approach',
                ]
                self.get_logger().info(
                    '[ST6→BUS1 직행] ST6에서 모듈팩 끝까지 짧게 직진한 뒤 '
                    '반시계 방향으로 BUS1 방향을 맞추며 기존 BUS1 중간점까지 '
                    '곧바로 전진합니다.')
                self._set_state(State.MOVE_TO_BUS_APPROACH)

    def _enter_grasp_stub(self):
        if self._enter_stub_wait('버스바 파지'):
            self._set_state(State.MOVE_BACK_TO_BUS_DOCK_APPROACH)

    def _enter_fasten_stub(self):
        if self._enter_stub_wait('ST1 버스바·너트 체결'):
            self._set_state(State.MOVE_TO_BUSBAR2_APPROACH)

    def _enter_grasp2_stub(self):
        if self._enter_stub_wait('BUS2 버스바 파지'):
            self._set_state(State.MOVE_TO_STATION2)

    def _enter_fasten2_stub(self):
        if self._enter_stub_wait('ST2 버스바·너트 체결'):
            self._set_state(State.MOVE_TO_BUSBAR3_APPROACH)

    def _enter_grasp3_stub(self):
        if self._enter_stub_wait('BUS3 버스바 파지'):
            self._set_state(State.MOVE_TO_STATION3)

    def _enter_fasten3_stub(self):
        if self._enter_stub_wait('ST3 버스바·너트 체결'):
            self._set_state(State.REPORT)

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
        elif failed_state == State.MOVE_TO_SCAN_STATION:
            self._enter_move_to_scan_station()
        elif failed_state == State.MOVE_TO_BUS_APPROACH:
            self._enter_move_to_bus_approach()
        elif failed_state == State.MOVE_TO_BUS_DOCK_APPROACH:
            self._enter_move_to_bus_dock_approach()
        elif failed_state == State.MOVE_TO_BUSBAR_TABLE:
            self._enter_move_to_busbar_table()
        elif failed_state == State.MOVE_BACK_TO_BUS_DOCK_APPROACH:
            self._enter_move_back_to_bus_dock_approach()
        elif failed_state == State.MOVE_BACK_TO_BUS_APPROACH:
            self._enter_move_back_to_bus_approach()
        elif failed_state == State.MOVE_BACK_TO_STATION:
            self._enter_move_back_to_station()
        elif failed_state == State.MOVE_TO_BUSBAR2_APPROACH:
            self._enter_move_to_busbar2_approach()
        elif failed_state == State.MOVE_TO_BUSBAR2_TABLE:
            self._enter_move_to_busbar2_table()
        elif failed_state == State.MOVE_TO_STATION2:
            self._enter_move_to_station2()
        elif failed_state == State.MOVE_TO_BUSBAR3_APPROACH:
            self._enter_move_to_busbar3_approach()
        elif failed_state == State.MOVE_TO_BUSBAR3_TABLE:
            self._enter_move_to_busbar3_table()
        elif failed_state == State.MOVE_TO_STATION3:
            self._enter_move_to_station3()
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

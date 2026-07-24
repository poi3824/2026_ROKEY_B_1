"""arm_node
실행 계층 · 버스바 파지·삽입·너트 체결 (M0609 매니퓰레이터).

ACTION /busbar_insert  (fms_interfaces/action/BusbarInsert)  <- behavior_node
ACTION /nut_fasten     (fms_interfaces/action/NutFasten)     <- behavior_node

SUB /vision/busbar_grasp · /vision/nut_pose · /vision/stud_pose
    (fms_interfaces/BusbarGrasp, NutPose, StudPose) <- perception_node (또는 dummy_executor_node)
    goal 진행 중 최신 값을 각 액션의 feedback(vision_target_pose)에 계속 실어 보낸다.

버스바 파지·삽입(GRASP/INSERT)은 /arm/target_pose(PoseStamped)를 Isaac Sim의 RMPFlow에
발행하고 /arm/current_pose(PoseStamped)로 수렴을 확인하는 실시간 Cartesian 제어로 수행한다.
GRASP 목표 좌표는 behavior_node가 goal.target_pose로 실어 보내는 /vision/busbar_grasp
검출값(없으면 하드코딩 fallback)을 쓴다.

INSERT 목표는 상공 접근(POS_INSERT_ABOVE) 도달 후 정지를 확인하고
perception_node의 /perception/get_bolt_pair 서비스를 동기 호출해, 버스바가 다리를
걸치는 볼트 2개의 실측 XY 중간점으로 하드코딩된 target_mid_pos의 XY를 대체한다
(볼트 미검출/서비스 실패 시 하드코딩 좌표로 폴백). vision_offset_grasp_x/y,
vision_offset_insert_x/y 파라미터는 TCP(그리퍼 기준점)와 비전이 알려주는 목표 지점
사이의 체계적 오차를 보정하는 값이다. GRASP/INSERT는 카메라가 보는 각도·거리가 달라
실측 오차 크기도 다르게 나와(예: GRASP 수십 mm대 vs INSERT 1mm 이하) 용도별로 별도
파라미터를 쓴다 (측정 방법은 선언부 주석 참고).

너트 체결(APPROACH/FASTEN)은 scripts/record_nut_fasten_trajectory.py로 World0123.usd에서
미리 녹화해둔 관절 궤적(data/nut_fasten_trajectory.json)을 그대로 재생해 /arm/joint_command로
발행한다 — 대상 nut1/peg_0 -> bolt_2 위치가 고정이라 실시간 IK 없이도 충분하기 때문.
녹화 파일이 없으면 더미 딜레이 + 임의 토크로 대체(_execute_nut_fasten_dummy).
"""
import collections
import json
import os
import sys
import threading
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int32, String

from fms_interfaces.action import BusbarInsert, NutFasten
from fms_interfaces.msg import BusbarGrasp, NutPose, StudPose
from fms_interfaces.srv import GetBoltPair

# feedback.phase에 실어 보낼 단계 이름. GRAB_BUSBAR/INSERT_BUSBAR 실제 동작 단계와 1:1 대응.
BUSBAR_GRASP_PHASES = ['BUSBAR_APPROACH', 'BUSBAR_DESCEND', 'BUSBAR_GRASP', 'BUSBAR_LIFT']
# 너트 체결은 busbar GRASP/INSERT와 동일한 절대좌표 + RMPFlow Cartesian 제어로 수행하되
# (기록 궤적 재생 대신), 동작 사이 물리적 충돌 여지를 줄이기 위해 busbar처럼 별도
# goal 4개로 나눈다 - 한 goal 안에 다 몰아넣으면 동작 간 정지/정착 시간이 없어진다.
# goal.command 값이 곧 이 순서이자 진행 단계 검증 키다.
NUT_FASTEN_STAGE_ORDER = ['NUT_APPROACH', 'NUT_GRASP', 'FASTEN_APPROACH', 'FASTEN']
BUSBAR_INSERT_PHASES = ['MOVE_TO_BOLT_APPROACH', 'BOLT_PAIR_SCAN', 'BUSBAR_DESCEND_TO_BOLT',
                         'BUSBAR_RELEASE_AND_RETRACT']

# setup.py가 JSON을 share/arm_node/data에 설치하므로 설치된 패키지 기준으로 찾는다.
# 소스 직접 실행 등 ament index를 쓸 수 없는 경우에만 모듈 옆 경로로 폴백한다.
try:
    _ARM_SHARE_DIR = get_package_share_directory('arm_node')
except Exception:
    _ARM_SHARE_DIR = os.path.dirname(__file__)
TRAJECTORY_PATH = os.path.join(_ARM_SHARE_DIR, 'data', 'nut_fasten_trajectory.json')
REPLAY_JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6',
                       'finger_joint', 'right_inner_knuckle_joint']

# 정지 판정: 최근 이 개수만큼의 연속 샘플이 모두 속도 임계값 이하여야 "정지"로 본다.
# (arm_node_scan_test.py의 검증된 패턴 이식). 선속도뿐 아니라 각속도도 봐야 한다 -
# FixedJoint 결속/FASTEN 재생 진입 순간 상대속도(선형+각)가 남아있으면 새 조인트가
# 그 운동을 순간적으로 멈추려는 큰 반력을 만든다.
STATIONARY_MIN_SAMPLES = 3
STATIONARY_SPEED_THRESHOLD_M_S = 0.005
STATIONARY_ANGULAR_SPEED_THRESHOLD_DEG_S = 3.0
STATIONARY_WAIT_TIMEOUT_SEC = 3.0

BOLT_PAIR_SERVICE_TIMEOUT_SEC = 3.0

# /arm/joint_command는 60Hz로 유지하고 Action feedback만 이 주기로 제한한다.
NUT_FASTEN_FEEDBACK_HZ = 7.5

# FASTEN 재생 진입 직전, 라이브 NUT_ALIGN 관절 배치 -> 녹화 궤적 첫 프레임 사이를
# 완충 보간하는 램프 설정 (급격한 관절값 점프로 인한 PD 드라이브 튕김 방지).
NUT_FASTEN_HANDOFF_RAMP_SEC = 1.0
NUT_FASTEN_HANDOFF_STEPS = 60

# FASTEN 재생 페이싱: wall-clock time.sleep()만으로 프레임 간격을 맞추면, Isaac Sim의
# 실제 스텝 속도가 60Hz를 못 따라갈 때(world.step(render=True)의 RTX 렌더링 부하 등)
# ROS2SubscribeJointState의 구독 큐(queueSize 기본값 10, ~166ms 버퍼)가 넘쳐서 오래된
# 프레임이 조용히 버려진다(Isaac Sim ROS2 브리지 소스 rcl_take 확인함) - 그 결과 로봇이
# 몇 프레임을 건너뛰며 튄다. 다음 프레임을 보내기 전 실제로 이전 프레임에 도달했는지
# /joint_states로 확인하면 큐에 한 번에 1개 이상 쌓일 일이 없어 이 문제가 사라진다.
NUT_FASTEN_FRAME_TOLERANCE_RAD = 0.02
NUT_FASTEN_FRAME_MAX_WAIT_SEC = 0.1


class ArmNode(Node):

    def __init__(self):
        super().__init__('arm_node')

        self._cb_group = ReentrantCallbackGroup()
        self._shutdown_requested = threading.Event()

        # 두 Action이 동일한 팔을 공유하므로 한 번에 하나만 실행한다.
        self._busy_lock = threading.Lock()
        self._busy = False
        self._busbar_grasp_done = False
        # 너트 체결 4단계(NUT_APPROACH -> NUT_GRASP -> FASTEN_APPROACH -> FASTEN) 진행
        # 순서를 강제하기 위해 마지막으로 완료된 단계 이름을 기록한다 (busbar의
        # _busbar_grasp_done과 동일한 목적, 단계가 4개라 문자열로 추적).
        self._nut_last_completed = None

        # 1. ACTION 서버 (behavior_node -> arm_node)
        self._busbar_action_server = ActionServer(
            self, BusbarInsert, 'busbar_insert',
            execute_callback=self._execute_busbar_insert,
            goal_callback=self._busbar_goal_callback,
            cancel_callback=self._accept_cancel,
            callback_group=self._cb_group,
        )
        self._fasten_action_server = ActionServer(
            self, NutFasten, 'nut_fasten',
            execute_callback=self._execute_nut_fasten,
            goal_callback=self._fasten_goal_callback,
            cancel_callback=self._accept_cancel,
            callback_group=self._cb_group,
        )

        # 2. 버스바 파지·삽입용 Cartesian 제어 인터페이스 (RMPFlow 경유)
        self._target_pose_pub = self.create_publisher(PoseStamped, '/arm/target_pose', 10)
        self._gripper_pub = self.create_publisher(Int32, '/arm/gripper_command', 10)
        # execute_isaac.py의 RMPFlow 루프는 /arm/target_pose 마지막 값을 계속 재적용하므로,
        # nut_fasten이 /arm/joint_command로 직접 관절을 제어하는 동안은 꺼둬야 서로 충돌하지 않는다.
        self._rmpflow_enable_pub = self.create_publisher(Bool, '/arm/rmpflow_enable', 10)
        # 그리퍼 마찰만으로는 너트를 안정적으로 붙잡지 못해, 실제 FixedJoint 결속/해제를
        # execute_isaac.py에 요청한다 (data=nut_id 결속, data=''(빈 문자열) 해제).
        self._nut_grasp_attach_pub = self.create_publisher(String, '/arm/nut_grasp_attach', 10)
        self._current_ee_pose = None
        # 정지 판정(_is_stationary)용 최근 EE 위치 샘플 이력.
        self._recent_pose_samples = collections.deque(maxlen=STATIONARY_MIN_SAMPLES)
        self._current_pose_sub = self.create_subscription(
            PoseStamped, '/arm/current_pose', self._current_pose_callback, 10,
            callback_group=self._cb_group,
        )

        # perception_node에 볼트 2개 위치를 동기 요청하는 서비스 클라이언트 (INSERT 정렬용).
        self._bolt_pair_client = self.create_client(
            GetBoltPair, '/perception/get_bolt_pair', callback_group=self._cb_group)

        # 3. 너트 체결 궤적 재생 및 Isaac Sim 조인트 상태 인터페이스
        self._joint_states_sub = self.create_subscription(
            JointState, '/joint_states', self._on_joint_states, 10,
            callback_group=self._cb_group)
        self._joint_command_pub = self.create_publisher(JointState, '/arm/joint_command', 10)
        self._latest_joint_states = None

        # 4. perception_node(또는 dummy_executor_node) 인터페이스 - feedback용 최신 vision pose 보관
        self._busbar_grasp_sub = self.create_subscription(
            BusbarGrasp, '/vision/busbar_grasp', self._on_busbar_grasp, 10,
            callback_group=self._cb_group)
        self._nut_pose_sub = self.create_subscription(
            NutPose, '/vision/nut_pose', self._on_nut_pose, 10,
            callback_group=self._cb_group)
        self._stud_pose_sub = self.create_subscription(
            StudPose, '/vision/stud_pose', self._on_stud_pose, 10,
            callback_group=self._cb_group)
        self._latest_busbar_grasp = None
        self._latest_nut_pose = None
        self._latest_stud_pose = None

        # TCP(그리퍼 기준점)와 비전이 알려주는 목표 지점 사이의 체계적 오차 보정값.
        # 카메라 캘리브레이션/그리퍼 형상 등에서 오는 고정 오차라고 보고 적용한다.
        # 측정 방법: offset=0으로 두고 perception_node 터미널 로그(평균 좌표)와 이미
        # 검증된 하드코딩 기준 좌표(_DEFAULT_POS_GRAB_PICK, target_mid_pos)를 비교 ->
        # offset = 기준좌표 - 비전평균좌표 -> 아래 파라미터로 설정 (재빌드 없이
        # `--ros-args -p vision_offset_grasp_x:=...` 또는 `ros2 param set`으로 조정 가능).
        #
        # GRASP(버스바 픽업 트레이)와 INSERT(볼트 트레이)는 카메라가 보는 각도/거리가
        # 서로 달라 실측해보니 오차 크기가 확연히 다르다 (예: GRASP는 수십 mm대,
        # INSERT/볼트는 1mm 이하) — 그래서 하나로 공유하지 않고 용도별로 분리한다.
        self.declare_parameter('vision_offset_grasp_x', 0.0)
        self.declare_parameter('vision_offset_grasp_y', 0.0)
        self.declare_parameter('vision_offset_insert_x', 0.0)
        self.declare_parameter('vision_offset_insert_y', 0.0)
        self._vision_offset_grasp_xy = np.array([
            self.get_parameter('vision_offset_grasp_x').value,
            self.get_parameter('vision_offset_grasp_y').value,
        ])
        self._vision_offset_insert_xy = np.array([
            self.get_parameter('vision_offset_insert_x').value,
            self.get_parameter('vision_offset_insert_y').value,
        ])

        # ★ [좌표 및 파라미터 정의 - 01_pick_and_lift.py 동일]
        # vision 검출이 없을 때 쓰는 대체(fallback) 파지 좌표.
        # self._DEFAULT_POS_GRAB_PICK = np.array([0.5128, 0.4477, 0.455, 0.0, 3.1415, 1.5708])
        self._DEFAULT_POS_GRAB_PICK = np.array([0.5128, 0.7299, 0.455, 0.0, 3.1415, 1.5708])

        # ★ 수정됨: 볼트 1, 2의 World 프레임 중심 좌표 (World0123.usd에서 실측, 미터 단위)
        # 이전 값은 PolyShape 메시의 로컬(mm 스케일 서브에셋) 좌표를 잘못 대입한 것이었음 (z=8.0은 world 높이가 아님)
        # X: 1.15447, Y: 0.17451, Z: 0.13423
        target_mid_pos = np.array([1.15935, 0.18325, 0.18])  # 그립 지점(볼트/표면 기준) Z, link_6(EE) 좌표 아님

        # ★ [EE 오프셋] /arm/target_pose는 link_6(EE) 좌표를 그대로 명령하며, 그리퍼가
        # link_6보다 아래로 뻗어나가는 길이를 보정하지 않으면 그립 지점보다 훨씬 더 깊이
        # 들어간다. m0609 + onrobot_rg2ft 조합에서 확립된 값(isaacpjt/M0609의
        # 9~12번 스크립트가 동일하게 사용: "그립점 = EE - EE_OFFSET").
        EE_OFFSET_Z = 0.185

        # ★ 수정됨: 그리퍼 Z축 90도 회전을 위해 기존 1.5708(90도)에서 0.0(0도)으로 변경
        yaw_angle = 0.0

        self._POS_INSERT_ABOVE = np.array([target_mid_pos[0], target_mid_pos[1], 0.6, 0.0, 3.1415, yaw_angle])
        self._POS_INSERT_PLACE = np.array([target_mid_pos[0], target_mid_pos[1], target_mid_pos[2] + EE_OFFSET_Z, 0.0, 3.1415, yaw_angle])

        # ★ 너트 파지(nut_id별 절대좌표, World0123.usd 실측) — busbar와 동일한
        # "그립점 = EE - EE_OFFSET_Z" 컨벤션. Z는 nut bottom_z + 그립 오프셋(기록
        # 스크립트의 NUT_GRASP_Z_LOCAL_OFFSET=0.03 + nut 두께 약 8.5mm)에 EE_OFFSET_Z를 더한 값
        # (record_nut_fasten_trajectory.py 재녹화 로그, 2026-07-24 실측과 대조 검증됨).
        NUT_GRASP_XY_Z = {
            'nut1': (0.5746, -0.1008, 0.5339),
            'nut2': (0.6643, -0.1031, 0.5336),
        }
        self._NUT_PICK_POSITIONS = {
            nut_id: np.array([x, y, grip_z + EE_OFFSET_Z, 0.0, 3.1415, 0.0])
            for nut_id, (x, y, grip_z) in NUT_GRASP_XY_Z.items()
        }
        self._NUT_GRASP_HOVER_OFFSET = 0.145  # busbar GRASP와 동일한 상공 접근 여유고도

        # ★ 볼트_2 정렬 목표(World0123.usd 실측, 2026-07-24) — busbar의 target_mid_pos와는
        # 별개 값이다(현재 씬 기준 서로 어긋나 있음이 확인됨, target_mid_pos 재보정 필요 별도 확인 요망).
        BOLT2_XY = (1.2636, 0.0019)
        BOLT2_TIP_Z = 0.1689
        NUT_ALIGN_GRIP_Z = BOLT2_TIP_Z + 0.001 + 0.05  # SCREW_HOVER_CLEAR + NUT_GRASP_Z_LOCAL
        self._NUT_ALIGN_POSITION = np.array([
            BOLT2_XY[0], BOLT2_XY[1], NUT_ALIGN_GRIP_Z + EE_OFFSET_Z, 0.0, 3.1415, 0.0])

        # ★ [단계별 차등 허용 오차]
        self._PICK_TOLERANCE_STRICT = 0.01    # Pick 단계: 0.01m (10mm)
        self._INSERT_TOLERANCE_STRICT = 0.015
        self._BUSBAR_RELEASE_Z = 0.36        # 그리퍼 해제 임계 높이
        self._INSERT_SPEED = 0.0015           # step당 수직 하강 속도 (m/step)

        # 초기 상태: 그리퍼 열림 (0)
        self._current_gripper_state = 0
        self._publish_gripper_state(0)

        self._trajectory = self._load_trajectory()

        self.get_logger().info('arm_node started (ACTION: busbar_insert, nut_fasten)')

    # --- 궤적 로딩 & Isaac Sim 인터페이스 콜백 --------------------------------
    def _load_trajectory(self):
        try:
            with open(TRAJECTORY_PATH) as f:
                trajectory = json.load(f)
            dof_names = trajectory['dof_names']
            period = float(trajectory['physics_dt'])
            if period <= 0.0:
                raise ValueError(f'physics_dt는 양수여야 함: {period}')

            indices = [dof_names.index(name) for name in REPLAY_JOINT_NAMES]
            required_position_count = max(indices) + 1
            for segment in ('approach', 'fasten'):
                frames = trajectory[segment]
                if not frames:
                    raise ValueError(f'{segment} 프레임이 비어 있음')
                for frame_index, frame in enumerate(frames):
                    positions = frame['positions']
                    if len(positions) < required_position_count:
                        raise ValueError(
                            f'{segment}[{frame_index}] positions 길이 부족: '
                            f'{len(positions)} < {required_position_count}')
            trajectory['replay_indices'] = indices

            # FASTEN 재생 중 그리퍼가 최종적으로 열리는(너트를 놓는) 프레임을 미리 찾아둔다 -
            # 그 프레임에서 NUT_GRASP 때 건 FixedJoint 결속을 같은 타이밍에 해제해야
            # 그리퍼가 열려도 너트가 계속 딸려오지 않는다. 라쳇 방식(REGRASP_CYCLES)이라
            # 중간에 열림/닫힘이 여러 번 반복되므로, 첫 번째가 아니라 "마지막" 열림
            # 전환(그 뒤로 다시 닫히지 않는 시점)을 찾아야 한다.
            finger_pos = REPLAY_JOINT_NAMES.index('finger_joint')
            finger_dof_index = indices[finger_pos]
            was_closed = False
            release_idx = None
            for frame_index, frame in enumerate(trajectory['fasten']):
                value = frame['positions'][finger_dof_index]
                closed_now = value >= 0.5
                if was_closed and not closed_now:
                    release_idx = frame_index
                was_closed = closed_now
            trajectory['fasten_release_idx'] = release_idx
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().error(
                f'궤적 파일이 없거나 유효하지 않음 ({TRAJECTORY_PATH}): {exc}')
            return None

        self.get_logger().info(
            f'너트 궤적 로드 완료: approach={len(trajectory["approach"])}프레임 '
            f'({len(trajectory["approach"]) * period:.2f}s), '
            f'fasten={len(trajectory["fasten"])}프레임 '
            f'({len(trajectory["fasten"]) * period:.2f}s), '
            f'fasten_release_idx={trajectory["fasten_release_idx"]}')
        return trajectory

    def _on_joint_states(self, msg: JointState):
        self._latest_joint_states = msg

    def _current_pose_callback(self, msg: PoseStamped):
        self._current_ee_pose = msg.pose
        self._recent_pose_samples.append((
            np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
            np.array([msg.pose.orientation.w, msg.pose.orientation.x,
                      msg.pose.orientation.y, msg.pose.orientation.z]),
            time.time(),
        ))

    def _is_stationary(self) -> bool:
        if len(self._recent_pose_samples) < STATIONARY_MIN_SAMPLES:
            return False

        samples = list(self._recent_pose_samples)
        for (pos_a, quat_a, t_a), (pos_b, quat_b, t_b) in zip(samples, samples[1:]):
            dt = t_b - t_a
            if dt <= 0:
                continue
            speed = np.linalg.norm(pos_b - pos_a) / dt
            if speed > STATIONARY_SPEED_THRESHOLD_M_S:
                return False
            dot = abs(float(np.clip(np.dot(quat_a, quat_b), -1.0, 1.0)))
            angle_deg = np.degrees(2.0 * np.arccos(dot))
            angular_speed = angle_deg / dt
            if angular_speed > STATIONARY_ANGULAR_SPEED_THRESHOLD_DEG_S:
                return False
        return True

    def _wait_until_stationary(
        self, timeout_sec: float = STATIONARY_WAIT_TIMEOUT_SEC, goal_handle=None
    ) -> bool:
        self.get_logger().info('  -> 정지 판정 대기 중 (속도 <= '
                                f'{STATIONARY_SPEED_THRESHOLD_M_S} m/s, 연속 {STATIONARY_MIN_SAMPLES}샘플)...')
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            if self._shutdown_requested.is_set() or not rclpy.ok():
                return False
            if goal_handle is not None and goal_handle.is_cancel_requested:
                self._publish_cartesian_hold()
                return False
            if self._is_stationary():
                self.get_logger().info('  -> 정지 확인 완료')
                return True
            time.sleep(0.05)
        self.get_logger().warn(f'  -> 정지 판정 타임아웃({timeout_sec}s), 계속 진행')
        return True

    def _request_bolt_pair(self, label: str = 'bolt', goal_handle=None):
        if not self._bolt_pair_client.wait_for_service(timeout_sec=BOLT_PAIR_SERVICE_TIMEOUT_SEC):
            self.get_logger().warn('/perception/get_bolt_pair 서비스 대기 타임아웃')
            return None

        request = GetBoltPair.Request()
        request.label = label
        future = self._bolt_pair_client.call_async(request)

        start_time = time.time()
        while not future.done():
            if self._shutdown_requested.is_set() or not rclpy.ok():
                return None
            if goal_handle is not None and goal_handle.is_cancel_requested:
                self._publish_cartesian_hold()
                return None
            if time.time() - start_time > BOLT_PAIR_SERVICE_TIMEOUT_SEC:
                self.get_logger().warn('/perception/get_bolt_pair 응답 타임아웃')
                return None
            time.sleep(0.05)

        response = future.result()
        if response is None or not response.found:
            message = response.message if response is not None else 'no response'
            self.get_logger().warn(f'/perception/get_bolt_pair 검출 없음: {message}')
            return None

        a = (response.pose_a.pose.position.x, response.pose_a.pose.position.y)
        b = (response.pose_b.pose.position.x, response.pose_b.pose.position.y)
        self.get_logger().info(
            f'/perception/get_bolt_pair 응답 좌표 사용 -> A=({a[0]:.4f},{a[1]:.4f}) '
            f'B=({b[0]:.4f},{b[1]:.4f}) ({response.message})')
        return a, b

    def _apply_vision_offset(self, xy: np.ndarray, offset_xy: np.ndarray, tag: str) -> np.ndarray:
        corrected = np.asarray(xy, dtype=float) + offset_xy
        if np.any(offset_xy != 0.0):
            self.get_logger().info(
                f'[{tag}] vision_offset 적용 -> 원본=({xy[0]:.4f},{xy[1]:.4f}) '
                f'보정후=({corrected[0]:.4f},{corrected[1]:.4f}) '
                f'(offset=({offset_xy[0]:.4f},{offset_xy[1]:.4f}))')
        return corrected

    # --- vision 토픽 콜백 ------------------------------------------------------
    def _on_busbar_grasp(self, msg: BusbarGrasp):
        self._latest_busbar_grasp = msg

    def _on_nut_pose(self, msg: NutPose):
        self._latest_nut_pose = msg

    def _on_stud_pose(self, msg: StudPose):
        self._latest_stud_pose = msg

    def _latest_busbar_vision_pose(self) -> Pose:
        return self._latest_busbar_grasp.pose.pose if self._latest_busbar_grasp is not None else Pose()

    def _latest_nut_vision_pose(self) -> Pose:
        if self._latest_nut_pose is not None:
            return self._latest_nut_pose.pose.pose
        if self._latest_stud_pose is not None:
            return self._latest_stud_pose.pose.pose
        return Pose()

    # --- Action goal/cancel 및 공유 로봇 실행권 --------------------------------
    def _accept_cancel(self, goal_handle):
        if goal_handle.status in (
            GoalStatus.STATUS_ACCEPTED,
            GoalStatus.STATUS_EXECUTING,
        ):
            return CancelResponse.ACCEPT
        return CancelResponse.REJECT

    def _busbar_goal_callback(self, goal_request):
        with self._busy_lock:
            if self._busy:
                self.get_logger().warn(
                    f'busbar_insert goal 거부: 로봇 busy (command={goal_request.command})')
                return GoalResponse.REJECT
            if goal_request.command not in ('GRASP', 'INSERT'):
                self.get_logger().warn(
                    f'busbar_insert goal 거부: 알 수 없는 command={goal_request.command}')
                return GoalResponse.REJECT
            if goal_request.command == 'INSERT' and not self._busbar_grasp_done:
                self.get_logger().warn(
                    'busbar_insert goal 거부: GRASP 완료 전 INSERT 요청')
                return GoalResponse.REJECT
            self._busy = True
            return GoalResponse.ACCEPT

    def _fasten_goal_callback(self, goal_request):
        with self._busy_lock:
            if self._busy:
                self.get_logger().warn(
                    f'nut_fasten goal 거부: 로봇 busy (command={goal_request.command})')
                return GoalResponse.REJECT
            if goal_request.command not in NUT_FASTEN_STAGE_ORDER:
                self.get_logger().warn(
                    f'nut_fasten goal 거부: 알 수 없는 command={goal_request.command}')
                return GoalResponse.REJECT
            stage_idx = NUT_FASTEN_STAGE_ORDER.index(goal_request.command)
            if stage_idx > 0 and self._nut_last_completed != NUT_FASTEN_STAGE_ORDER[stage_idx - 1]:
                self.get_logger().warn(
                    f'nut_fasten goal 거부: {NUT_FASTEN_STAGE_ORDER[stage_idx - 1]} 완료 전 '
                    f'{goal_request.command} 요청')
                return GoalResponse.REJECT
            self._busy = True
            return GoalResponse.ACCEPT

    def _release_robot(self):
        with self._busy_lock:
            self._busy = False

    def request_shutdown(self):
        self._shutdown_requested.set()

    def _publish_cartesian_hold(self):
        if self._current_ee_pose is None:
            self.get_logger().warn('Cartesian 정지: 현재 EE pose가 없어 새 목표를 발행하지 않음')
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose = self._current_ee_pose
        self._target_pose_pub.publish(msg)
        self._publish_gripper_state(self._current_gripper_state)

    def _publish_joint_hold(self, names=None, positions=None):
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        if names is not None and positions is not None:
            cmd.name = list(names)
            cmd.position = list(positions)
        elif self._latest_joint_states is not None:
            cmd.name = list(self._latest_joint_states.name)
            cmd.position = list(self._latest_joint_states.position)
        else:
            self.get_logger().warn('Joint 정지: 참조할 관절 위치가 없어 명령을 발행하지 않음')
            return
        self._joint_command_pub.publish(cmd)

    def _resolve_grab_pick_pose(self, target_pose: Pose) -> np.ndarray:
        if target_pose.position.x != 0.0 or target_pose.position.y != 0.0:
            pick_pose = self._DEFAULT_POS_GRAB_PICK.copy()
            pick_pose[0], pick_pose[1] = self._apply_vision_offset(
                np.array([target_pose.position.x, target_pose.position.y]),
                self._vision_offset_grasp_xy, 'GRASP')
            self.get_logger().info(
                f'goal.target_pose(vision) 좌표 사용 -> x={pick_pose[0]:.4f}, y={pick_pose[1]:.4f}')
            return pick_pose
        self.get_logger().warn('goal.target_pose가 비어있음(vision 미검출), 기본 좌표로 대체')
        return self._DEFAULT_POS_GRAB_PICK.copy()

    # --- 저수준 Cartesian 제어 (target_pose/current_pose, RMPFlow 경유) --------
    def _publish_gripper_state(self, state: int):
        self._current_gripper_state = state
        msg = Int32()
        msg.data = int(state)
        self._gripper_pub.publish(msg)

    def _publish_target_pose(self, target):
        x, y, z, roll, pitch, yaw = target
        q_wxyz = self._euler_to_quaternion(roll, pitch, yaw)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"

        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)

        msg.pose.orientation.w = float(q_wxyz[0])
        msg.pose.orientation.x = float(q_wxyz[1])
        msg.pose.orientation.y = float(q_wxyz[2])
        msg.pose.orientation.z = float(q_wxyz[3])

        self._target_pose_pub.publish(msg)

    def _move_to_pose(
        self, target_pose, step_name, pos_tolerance=0.001, timeout_sec=8.0,
        goal_handle=None, orientation_tolerance_deg=None,
    ):
        """orientation_tolerance_deg를 주면 위치뿐 아니라 목표 자세(roll/pitch/yaw)
        수렴도 확인한다. 기본(None)은 기존과 동일하게 위치만 본다 - RMPFlow가 위치는
        먼저 맞추고 자세 수렴이 뒤처지는 경우, 위치만 보고 "도달"로 판단하면 실제로는
        자세가 덜 맞은 채로 다음 동작(특히 FASTEN 재생 관절값과의 이어붙임)으로 넘어가
        관절 공간에서 큰 불연속이 생길 수 있어 이 값이 중요한 지점(FASTEN_ALIGN 등)에만
        선택적으로 켠다."""
        self.get_logger().info(f'  -> [하위동작] {step_name} Target Pose 발행 (목표 오차: {pos_tolerance:.4f} m)...')
        self._publish_target_pose(target_pose)

        target_pos = np.array(target_pose[:3])
        target_quat = self._euler_to_quaternion(*target_pose[3:6]) if orientation_tolerance_deg is not None else None
        start_time = time.time()

        while True:
            if self._shutdown_requested.is_set() or not rclpy.ok():
                return False
            if goal_handle is not None and goal_handle.is_cancel_requested:
                self._publish_cartesian_hold()
                return False
            self._publish_gripper_state(self._current_gripper_state)

            if self._current_ee_pose is not None:
                curr_pos = np.array([
                    self._current_ee_pose.position.x,
                    self._current_ee_pose.position.y,
                    self._current_ee_pose.position.z
                ])

                dist_error = np.linalg.norm(target_pos - curr_pos)

                orient_error_deg = 0.0
                orient_ok = True
                if target_quat is not None:
                    curr_quat = np.array([
                        self._current_ee_pose.orientation.w,
                        self._current_ee_pose.orientation.x,
                        self._current_ee_pose.orientation.y,
                        self._current_ee_pose.orientation.z,
                    ])
                    dot = abs(float(np.clip(np.dot(target_quat, curr_quat), -1.0, 1.0)))
                    orient_error_deg = float(np.degrees(2.0 * np.arccos(dot)))
                    orient_ok = orient_error_deg < orientation_tolerance_deg

                sys.stdout.write(
                    f"\r     [이동 중...] 현재오차: {dist_error:.4f} m "
                    f"(목표: <{pos_tolerance:.4f} m)"
                    + (f" | 자세오차: {orient_error_deg:.1f}deg (목표: <{orientation_tolerance_deg:.1f}deg)"
                       if target_quat is not None else "")
                    + f" | 경과시간: {time.time() - start_time:4.1f}s"
                )
                sys.stdout.flush()

                if dist_error < pos_tolerance and orient_ok:
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                    self.get_logger().info(
                        f'  -> [도달 완료] {step_name} (최종 오차: {dist_error:.4f} m'
                        + (f', 자세오차: {orient_error_deg:.1f}deg)' if target_quat is not None else ')'))
                    return True

            if time.time() - start_time > timeout_sec:
                sys.stdout.write('\n')
                sys.stdout.flush()
                if self._current_ee_pose is None:
                    self.get_logger().error(
                        f'  -> [시간 초과({timeout_sec:.1f}s)] {step_name}: '
                        '/arm/current_pose 미수신')
                else:
                    self.get_logger().error(
                        f'  -> [시간 초과({timeout_sec:.1f}s)] {step_name}: '
                        f'목표 오차 {dist_error:.4f} m'
                        + (f', 자세오차 {orient_error_deg:.1f}deg' if target_quat is not None else '')
                        + ' 미달')
                return False

            time.sleep(0.05)

    def _descend_step_by_step(
        self, start_above_pose, final_target_pose, step_name, goal_handle=None,
        timeout_sec=15.0,
    ):
        self.get_logger().info(f'  -> [하위동작] {step_name} 점진적 하강 시작 (-0.0015m/step)...')

        current_target_z = start_above_pose[2]
        final_target_z = final_target_pose[2]
        final_pos = np.array(final_target_pose[:3])
        start_time = time.time()

        while True:
            if self._shutdown_requested.is_set() or not rclpy.ok():
                return False
            if goal_handle is not None and goal_handle.is_cancel_requested:
                self._publish_cartesian_hold()
                return False
            if time.time() - start_time > timeout_sec:
                self.get_logger().warn(
                    f'  -> [시간 초과({timeout_sec:.1f}s)] {step_name} 중단')
                self._publish_cartesian_hold()
                return False
            self._publish_gripper_state(self._current_gripper_state)

            if current_target_z > final_target_z:
                current_target_z = max(final_target_z, current_target_z - self._INSERT_SPEED)

            step_pose = final_target_pose.copy()
            step_pose[2] = current_target_z
            self._publish_target_pose(step_pose)

            if self._current_ee_pose is not None:
                curr_pos = np.array([
                    self._current_ee_pose.position.x,
                    self._current_ee_pose.position.y,
                    self._current_ee_pose.position.z
                ])

                dist_error = np.linalg.norm(final_pos - curr_pos)
                elapsed_time = time.time() - start_time

                # ★ TCP 좌표 실시간 출력: 명령값(step_pose)과 실측값(curr_pos) 비교용
                sys.stdout.write(
                    f"\r     [TCP] 명령 z={step_pose[2]:.4f} | "
                    f"실측=({curr_pos[0]:.4f}, {curr_pos[1]:.4f}, {curr_pos[2]:.4f}) | "
                    f"오차={dist_error:.4f}m"
                )
                sys.stdout.flush()

                if curr_pos[2] <= self._BUSBAR_RELEASE_Z or dist_error < self._INSERT_TOLERANCE_STRICT:
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                    self.get_logger().info(f'  -> [하강 완료] {step_name} (EE Z: {curr_pos[2]:.4f}m | 최종 오차: {dist_error:.4f}m)')
                    return True

            time.sleep(0.05)

    def _control_gripper(self, close: bool, goal_handle=None):
        state_val = 1 if close else 0
        action_str = "닫기" if close else "열기"

        self.get_logger().info(f'  -> 그리퍼 {action_str}')
        self._publish_gripper_state(state_val)
        for _ in range(10):
            if self._shutdown_requested.is_set() or not rclpy.ok():
                return False
            if goal_handle is not None and goal_handle.is_cancel_requested:
                self._publish_cartesian_hold()
                return False
            time.sleep(0.1)
        return True

    def _euler_to_quaternion(self, roll, pitch, yaw):
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * sp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.array([w, x, y, z])

    # --- 버스바 파지 · 삽입 ACTION -----------------------------------------
    def _publish_busbar_feedback(self, goal_handle, phase: str, progress: float):
        feedback = BusbarInsert.Feedback()
        feedback.phase = phase
        feedback.progress = progress
        feedback.vision_target_pose = self._latest_busbar_vision_pose()
        goal_handle.publish_feedback(feedback)

    def _execute_busbar_insert(self, goal_handle):
        goal = goal_handle.request
        self.get_logger().info(f'ACTION /busbar_insert 시작 <- {goal.command} ({goal.station_id})')

        # nut_fasten이 이전 실행에서 비정상 종료(취소/크래시)돼 rmpflow_enable=False로
        # 고착된 채 남아있을 수 있으므로, Cartesian 제어가 필요한 busbar 동작 시작 시
        # 항상 다시 True로 재확인한다.
        self._rmpflow_enable_pub.publish(Bool(data=True))

        try:
            if goal.command == 'GRASP':
                with self._busy_lock:
                    self._busbar_grasp_done = False
                    self._nut_last_completed = None
                success, message = self._run_busbar_grasp(goal, goal_handle)
            else:
                success, message = self._run_busbar_insert(goal, goal_handle)

            if self._shutdown_requested.is_set() or not rclpy.ok():
                return BusbarInsert.Result(success=False, message='노드 종료 중')

            if goal_handle.is_cancel_requested:
                self._publish_cartesian_hold()
                goal_handle.canceled()
                return BusbarInsert.Result(success=False, message='취소됨')

            if success:
                goal_handle.succeed()
            else:
                goal_handle.abort()
            with self._busy_lock:
                if goal.command == 'GRASP':
                    self._busbar_grasp_done = success
                elif success:
                    self._busbar_grasp_done = False
            self.get_logger().info(
                f'ACTION /busbar_insert 완료 -> success={success} ({message})')
            return BusbarInsert.Result(success=success, message=message)
        finally:
            self._release_robot()

    def _run_busbar_grasp(self, goal, goal_handle):
        pos_grab_pick = self._resolve_grab_pick_pose(goal.target_pose)
        pos_grab_above = pos_grab_pick + np.array([0.0, 0.0, 0.145, 0.0, 0.0, 0.0])
        busbar_lift_move_pos = pos_grab_pick + np.array([0.0, 0.1, 0.145, 0.0, 0.0, 0.0])
        phases = BUSBAR_GRASP_PHASES

        self._publish_gripper_state(0)

        self._publish_busbar_feedback(goal_handle, phases[0], 1 / 4)
        if not self._move_to_pose(
            pos_grab_above, '1. 버스바 상공 접근',
            pos_tolerance=self._PICK_TOLERANCE_STRICT, goal_handle=goal_handle,
        ):
            return False, '상공 접근 실패'

        self._publish_busbar_feedback(goal_handle, phases[1], 2 / 4)
        if not self._move_to_pose(
            pos_grab_pick, '2. 버스바 파지점 하강',
            pos_tolerance=self._PICK_TOLERANCE_STRICT, goal_handle=goal_handle,
        ):
            return False, '파지점 하강 실패'

        self._publish_busbar_feedback(goal_handle, phases[2], 3 / 4)
        if not self._control_gripper(close=True, goal_handle=goal_handle):
            return False, '그리퍼 닫기 취소'

        self._publish_busbar_feedback(goal_handle, phases[3], 4 / 4)
        if not self._move_to_pose(
            busbar_lift_move_pos, '3. 버스바 상승 및 이동',
            pos_tolerance=self._PICK_TOLERANCE_STRICT, goal_handle=goal_handle,
        ):
            return False, '상승 이동 실패'

        return True, 'GRAB_BUSBAR 시퀀스 최종 완료'

    def _resolve_insert_poses(self, goal_handle=None) -> tuple:
        """★ 수정됨: 재보정된 하드코딩 좌표를 우선 사용한다. 볼트 쌍은 계속 실측 조회해
        로그로 비교만 하고(모니터링용), 반환값 자체는 vision 보정을 적용하지 않는다 —
        하드코딩 좌표가 실측 기준으로 이미 맞춰져 있어 당분간 vision 보정 없이도
        충분하다고 판단했기 때문. 필요해지면 아래 mid_xy를 다시 반환값에 반영하면 된다."""
        pos_insert_above = self._POS_INSERT_ABOVE.copy()
        pos_insert_place = self._POS_INSERT_PLACE.copy()

        bolt_pair = self._request_bolt_pair(goal_handle=goal_handle)
        if bolt_pair is None:
            self.get_logger().warn('볼트 쌍 미검출/서비스 실패, 하드코딩 체결 좌표로 대체')
        else:
            (ax, ay), (bx, by) = bolt_pair
            mid_xy = self._apply_vision_offset(
                np.array([(ax + bx) / 2.0, (ay + by) / 2.0]),
                self._vision_offset_insert_xy, 'INSERT')
            self.get_logger().info(
                f'[참고용] 볼트 쌍 실측 중간점 -> x={mid_xy[0]:.4f}, y={mid_xy[1]:.4f} '
                f'(하드코딩: x={pos_insert_above[0]:.4f}, y={pos_insert_above[1]:.4f})')

        self.get_logger().info(
            f'비전 보정을 적용하지 않고 하드코딩 좌표(볼트 중심)를 사용합니다 -> '
            f'ABOVE(TCP)=({pos_insert_above[0]:.4f}, {pos_insert_above[1]:.4f}, {pos_insert_above[2]:.4f}) '
            f'PLACE(TCP)=({pos_insert_place[0]:.4f}, {pos_insert_place[1]:.4f}, {pos_insert_place[2]:.4f})'
        )

        return pos_insert_above, pos_insert_place

    def _run_busbar_insert(self, goal, goal_handle):
        phases = BUSBAR_INSERT_PHASES

        self._publish_busbar_feedback(goal_handle, phases[0], 1 / 4)
        if not self._move_to_pose(
            self._POS_INSERT_ABOVE, '1. 체결위치 상공 접근',
            pos_tolerance=self._INSERT_TOLERANCE_STRICT, goal_handle=goal_handle,
        ):
            return False, '체결 상공 접근 실패'

        self._publish_busbar_feedback(goal_handle, phases[1], 2 / 4)
        if not self._wait_until_stationary(goal_handle=goal_handle):
            return False, '정지 대기 취소'
        
        # ★ 수정된 하드코딩 적용 부분 호출
        pos_insert_above, pos_insert_place = self._resolve_insert_poses(
            goal_handle=goal_handle)
            
        if goal_handle.is_cancel_requested:
            return False, '볼트쌍 조회 취소'
        if not self._move_to_pose(
            pos_insert_above, '1b. 보정된 체결위치 재접근',
            pos_tolerance=self._INSERT_TOLERANCE_STRICT, goal_handle=goal_handle,
        ):
            return False, '보정된 체결 상공 접근 실패'

        self._publish_busbar_feedback(goal_handle, phases[2], 3 / 4)
        if not self._descend_step_by_step(
            pos_insert_above, pos_insert_place, '2. 체결 위치 점진적 하강 및 삽입',
            goal_handle=goal_handle,
        ):
            return False, '체결 위치 하강 실패'

        if not self._control_gripper(close=False, goal_handle=goal_handle):
            return False, '그리퍼 열기 취소'
        if not self._move_to_pose(
            pos_insert_above, '3. 상공 이탈',
            pos_tolerance=self._INSERT_TOLERANCE_STRICT, goal_handle=goal_handle,
        ):
            return False, '상공 이탈 실패'
        self._publish_busbar_feedback(goal_handle, phases[3], 4 / 4)

        return True, 'INSERT_BUSBAR 시퀀스 최종 완료'

    # --- 너트 파지(APPROACH) ACTION: busbar GRASP와 동일한 절대좌표 + Cartesian 제어 ---
    def _publish_nut_feedback(self, goal_handle, phase: str, progress: float):
        feedback = NutFasten.Feedback()
        feedback.phase = phase
        feedback.progress = progress
        feedback.vision_target_pose = self._latest_nut_vision_pose()
        goal_handle.publish_feedback(feedback)

    def _resolve_nut_pick_pose(self, nut_id: str) -> np.ndarray:
        pose = self._NUT_PICK_POSITIONS.get(nut_id)
        if pose is None:
            self.get_logger().warn(
                f'알 수 없는 nut_id={nut_id!r}, 기본값(nut1)으로 대체')
            pose = self._NUT_PICK_POSITIONS['nut1']
        return pose.copy()

    def _run_nut_approach(self, goal, goal_handle):
        """1단계: 너트 상공까지만 이동 (파지는 아직 안 함)."""
        nut_id = goal.nut_id or 'nut1'
        pos_pick = self._resolve_nut_pick_pose(nut_id)
        pos_above = pos_pick + np.array([0.0, 0.0, self._NUT_GRASP_HOVER_OFFSET, 0.0, 0.0, 0.0])

        self._publish_gripper_state(0)
        self._publish_nut_feedback(goal_handle, 'NUT_APPROACH', 1.0)
        if not self._move_to_pose(
            pos_above, f'너트({nut_id}) 상공 접근',
            pos_tolerance=self._PICK_TOLERANCE_STRICT, goal_handle=goal_handle,
        ):
            return False, '너트 상공 접근 실패'

        return True, 'NUT_APPROACH 완료'

    def _run_nut_grasp(self, goal, goal_handle):
        """2단계: 하강 -> 그리퍼 닫기(+FixedJoint 결속) -> 상승. NUT_APPROACH가 끝난
        상공 위치에서 시작한다고 가정한다."""
        nut_id = goal.nut_id or 'nut1'
        pos_pick = self._resolve_nut_pick_pose(nut_id)
        pos_above = pos_pick + np.array([0.0, 0.0, self._NUT_GRASP_HOVER_OFFSET, 0.0, 0.0, 0.0])

        self._publish_nut_feedback(goal_handle, 'NUT_DESCEND', 1 / 3)
        if not self._move_to_pose(
            pos_pick, f'너트({nut_id}) 파지점 하강',
            pos_tolerance=self._PICK_TOLERANCE_STRICT, goal_handle=goal_handle,
        ):
            return False, '너트 파지점 하강 실패'

        self._publish_nut_feedback(goal_handle, 'NUT_GRASP', 2 / 3)
        if not self._control_gripper(close=True, goal_handle=goal_handle):
            return False, '그리퍼 닫기 취소'

        # FixedJoint 결속 직전 상대속도(선형+각)가 남아있으면 새 조인트가 그 운동을
        # 순간적으로 멈추려는 큰 반력을 만든다 - 결속 전 반드시 정지를 확인한다.
        if not self._wait_until_stationary(goal_handle=goal_handle):
            return False, '결속 전 정지 대기 취소'
        # 핑거 마찰만으로는 불안정하므로 현재 상대 자세로 FixedJoint 결속을 요청한다.
        self._nut_grasp_attach_pub.publish(String(data=nut_id))

        self._publish_nut_feedback(goal_handle, 'NUT_LIFT', 3 / 3)
        if not self._move_to_pose(
            pos_above, '너트 상승',
            pos_tolerance=self._PICK_TOLERANCE_STRICT, goal_handle=goal_handle,
        ):
            return False, '너트 상승 실패'

        return True, 'NUT_GRASP 완료'

    def _run_fasten_approach(self, goal, goal_handle):
        """3단계: 파지한 너트를 볼트 상공으로 옮기고 정렬 위치까지 하강. FASTEN(기록
        궤적 재생)이 기대하는 자세로 도달한다."""
        pos_align = self._NUT_ALIGN_POSITION.copy()
        pos_align_above = pos_align + np.array([0.0, 0.0, self._NUT_GRASP_HOVER_OFFSET, 0.0, 0.0, 0.0])

        self._publish_nut_feedback(goal_handle, 'FASTEN_CARRY', 1 / 2)
        if not self._move_to_pose(
            pos_align_above, '볼트 상공 이동',
            pos_tolerance=self._PICK_TOLERANCE_STRICT, goal_handle=goal_handle,
            orientation_tolerance_deg=5.0, timeout_sec=12.0,
        ):
            return False, '볼트 상공 이동 실패'

        # FASTEN(기록 궤적 재생)으로 넘어가기 직전 마지막 위치 - 여기서 자세까지
        # 확실히 맞춰야 관절 공간 불연속(자세 정렬 램프의 큰 오차, 팔 튕김)을 줄인다.
        self._publish_nut_feedback(goal_handle, 'FASTEN_ALIGN', 2 / 2)
        if not self._move_to_pose(
            pos_align, '볼트 위 정렬 하강',
            pos_tolerance=self._PICK_TOLERANCE_STRICT, goal_handle=goal_handle,
            orientation_tolerance_deg=3.0, timeout_sec=12.0,
        ):
            return False, '볼트 위 정렬 실패'

        return True, 'FASTEN_APPROACH 완료'

    # --- 너트 체결 ACTION (NUT_APPROACH/NUT_GRASP/FASTEN_APPROACH: Cartesian,
    # FASTEN(SEAT~SCREW)은 기록 궤적 재생) ------------------------------------
    def _execute_nut_fasten(self, goal_handle):
        goal = goal_handle.request
        self.get_logger().info(f'ACTION /nut_fasten 시작 <- {goal.command} (nut_id={goal.nut_id})')

        try:
            if goal.command in ('NUT_APPROACH', 'NUT_GRASP', 'FASTEN_APPROACH'):
                # 이전 FASTEN 실행이 취소/크래시로 비정상 종료돼 rmpflow_enable=False로
                # 고착돼 있을 수 있으므로, Cartesian 제어를 시작하기 전 재확인한다.
                self._rmpflow_enable_pub.publish(Bool(data=True))
                if goal.command == 'NUT_APPROACH':
                    success, message = self._run_nut_approach(goal, goal_handle)
                elif goal.command == 'NUT_GRASP':
                    success, message = self._run_nut_grasp(goal, goal_handle)
                else:
                    success, message = self._run_fasten_approach(goal, goal_handle)

                if self._shutdown_requested.is_set() or not rclpy.ok():
                    return NutFasten.Result(success=False, torque=0.0, message='노드 종료 중')
                if goal_handle.is_cancel_requested:
                    self._publish_cartesian_hold()
                    goal_handle.canceled()
                    return NutFasten.Result(success=False, torque=0.0, message='취소됨')

                if success:
                    goal_handle.succeed()
                else:
                    goal_handle.abort()
                result = NutFasten.Result(success=success, torque=0.0, message=message)
            elif self._trajectory is None:
                goal_handle.abort()
                result = NutFasten.Result(
                    success=False,
                    torque=0.0,
                    message='너트 궤적 JSON이 없거나 유효하지 않아 실행하지 않음',
                )
            else:
                self._rmpflow_enable_pub.publish(Bool(data=False))
                try:
                    result = self._execute_nut_fasten_replay(goal_handle)
                finally:
                    # 노드 종료(Ctrl-C 등) 도중이면 publisher의 rcl 컨텍스트가 이미
                    # 무효화돼 있어 publish 자체가 RCLError를 던진다 - 종료 중엔 굳이
                    # 복구할 필요도 없으므로 건너뛴다.
                    if rclpy.ok() and not self._shutdown_requested.is_set():
                        self._rmpflow_enable_pub.publish(Bool(data=True))

            if result.success:
                # 실패 시엔 건드리지 않는다 - 같은 단계를 재시도할 때 이전 단계 완료
                # 기록이 남아있어야 goal_callback의 순서 검증(predecessor 확인)을 통과한다
                # (busbar의 _busbar_grasp_done과 동일한 패턴).
                with self._busy_lock:
                    self._nut_last_completed = goal.command
            return result
        finally:
            self._release_robot()

    def _ramp_to_frame(self, target_frame, indices, goal_handle):
        """recorded FASTEN 궤적 재생 시작 전, 실제 관절 상태에서 첫 프레임 목표값까지
        부드럽게 보간한다. 라이브 NUT_ALIGN(RMPFlow IK)이 녹화 당시와 world 위치는
        같아도 관절 구성(팔꿈치 각도 등)이 정확히 같으리라는 보장이 없어, 그 차이를
        곧바로 절대 관절값으로 커맨드하면 강성(1e8) 드라이브가 한 스텝만에 따라잡으려다
        튕겨나간다(재현됨) -- 그래서 짧게 램프를 깔아 완충한다."""
        target_positions = [target_frame['positions'][i] for i in indices]

        if self._latest_joint_states is None:
            self.get_logger().warn('  -> 현재 관절 상태 없음(/joint_states 미수신), 보간 없이 즉시 시작')
            return True

        name_to_pos = dict(zip(self._latest_joint_states.name, self._latest_joint_states.position))
        try:
            current_positions = [name_to_pos[name] for name in REPLAY_JOINT_NAMES]
        except KeyError as exc:
            self.get_logger().warn(f'  -> 현재 관절 상태에 {exc} 없음, 보간 없이 즉시 시작')
            return True

        max_delta = max(abs(t - c) for t, c in zip(target_positions, current_positions))
        self.get_logger().info(
            f'  -> FASTEN 진입 전 자세 정렬 보간 시작 (최대 관절 오차: {max_delta:.4f} rad)...')

        for step in range(1, NUT_FASTEN_HANDOFF_STEPS + 1):
            if self._shutdown_requested.is_set() or not rclpy.ok():
                return False
            if goal_handle.is_cancel_requested:
                return False
            frac = step / NUT_FASTEN_HANDOFF_STEPS
            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name = REPLAY_JOINT_NAMES
            cmd.position = [c + frac * (t - c) for c, t in zip(current_positions, target_positions)]
            self._joint_command_pub.publish(cmd)
            time.sleep(NUT_FASTEN_HANDOFF_RAMP_SEC / NUT_FASTEN_HANDOFF_STEPS)

        self.get_logger().info('  -> 자세 정렬 완료, FASTEN 궤적 재생 시작')
        return True

    def _wait_for_joint_target(self, target_positions, max_wait_sec):
        """/joint_states가 target_positions에 수렴할 때까지, 또는 max_wait_sec
        안전장치에 도달할 때까지 대기한다 (NUT_FASTEN_FRAME_MAX_WAIT_SEC 설명 참고)."""
        target_arr = np.array(target_positions)
        deadline = time.time() + max_wait_sec
        while time.time() < deadline:
            if self._shutdown_requested.is_set() or not rclpy.ok():
                return
            if self._latest_joint_states is not None:
                name_to_pos = dict(zip(
                    self._latest_joint_states.name, self._latest_joint_states.position))
                try:
                    current_arr = np.array([name_to_pos[n] for n in REPLAY_JOINT_NAMES])
                except KeyError:
                    return  # 조인트 이름 정보 없음 - 안전 폴백으로 그냥 진행
                if np.max(np.abs(current_arr - target_arr)) < NUT_FASTEN_FRAME_TOLERANCE_RAD:
                    return
            time.sleep(0.002)

    def _execute_nut_fasten_replay(self, goal_handle):
        segment = 'fasten'
        frames = self._trajectory[segment]
        indices = self._trajectory['replay_indices']
        period = self._trajectory['physics_dt']
        total = max(len(frames), 1)
        feedback_every = max(1, round((1.0 / NUT_FASTEN_FEEDBACK_HZ) / period))
        release_idx = self._trajectory.get('fasten_release_idx')

        # 라이브 Cartesian 제어(FASTEN_APPROACH)에서 관절 재생으로 넘어가는 순간에도
        # 상대속도가 남아있으면 같은 이유로 위험하다 - 램프 시작 전 정지를 확인한다.
        if not self._wait_until_stationary(goal_handle=goal_handle):
            return NutFasten.Result(success=False, torque=0.0, message='체결 진입 전 정지 대기 취소')

        if not self._ramp_to_frame(frames[0], indices, goal_handle):
            return NutFasten.Result(success=False, torque=0.0, message='체결 시작 자세 정렬 실패/취소')

        for idx, frame in enumerate(frames):
            if idx == release_idx:
                # 그리퍼가 열리는 프레임과 같은 타이밍에 NUT_GRASP 때 건 FixedJoint를 해제한다.
                self._nut_grasp_attach_pub.publish(String(data=''))
            if self._shutdown_requested.is_set() or not rclpy.ok():
                return NutFasten.Result(
                    success=False, torque=0.0, message='노드 종료 중')
            if goal_handle.is_cancel_requested:
                if idx > 0:
                    last_positions = [
                        frames[idx - 1]['positions'][i] for i in indices
                    ]
                    self._publish_joint_hold(REPLAY_JOINT_NAMES, last_positions)
                else:
                    self._publish_joint_hold()
                goal_handle.canceled()
                return NutFasten.Result(success=False, torque=0.0, message='취소됨')

            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name = REPLAY_JOINT_NAMES
            cmd.position = [frame['positions'][i] for i in indices]
            self._joint_command_pub.publish(cmd)

            if idx % feedback_every == 0 or idx == total - 1:
                feedback = NutFasten.Feedback()
                feedback.phase = f'{segment.upper()}_REPLAY'
                feedback.progress = (idx + 1) / total
                feedback.vision_target_pose = self._latest_nut_vision_pose()
                goal_handle.publish_feedback(feedback)
            self._wait_for_joint_target(cmd.position, NUT_FASTEN_FRAME_MAX_WAIT_SEC)

        goal_handle.succeed()
        message = '체결 궤적 재생 완료 (토크 미측정)'
        result = NutFasten.Result(success=True, torque=0.0, message=message)

        self.get_logger().info(f'ACTION /nut_fasten 완료 -> {message}')
        return result


def main(args=None):
    rclpy.init(args=args)
    node = ArmNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.request_shutdown()
        try:
            executor.shutdown(timeout_sec=5.0)
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            try:
                node.destroy_node()
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()
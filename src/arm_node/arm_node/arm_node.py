"""arm_node
실행 계층 · 버스바 파지·삽입·너트 체결 (M0609 매니퓰레이터).

ACTION /busbar_insert  (fms_interfaces/action/BusbarInsert)  <- behavior_node
ACTION /nut_fasten     (fms_interfaces/action/NutFasten)     <- behavior_node

SUB /vision/busbar_grasp · /vision/nut_pose · /vision/stud_pose
    (fms_interfaces/BusbarGrasp, NutPose, StudPose) <- perception_node (또는 dummy_executor_node)
    goal 진행 중 최신 값을 각 액션의 feedback(vision_target_pose)에 계속 실어 보낸다.

SUB /joint_states        (sensor_msgs/JointState)  <- Isaac Sim (관절 상태·접촉 정보)
PUB /arm/joint_command    (sensor_msgs/JointState)  -> Isaac Sim (관절 제어)

너트 체결(APPROACH/FASTEN)은 scripts/record_nut_fasten_trajectory.py로 World0123.usd에서
미리 녹화해둔 관절 궤적(data/nut_fasten_trajectory.json)을 그대로 재생한다 — 대상
nut1/peg_0 -> bolt_2 위치가 고정이라 실시간 IK 없이도 충분하기 때문. 버스바 파지·삽입은
아직 녹화된 궤적이 없어 10_busbar_assembly.py의 BUSBAR_* phase 이름만 빌려 진행 상황을
feedback으로 흘려보내는 placeholder 상태다 (TODO: 실제 IK/모션 플래닝, record_busbar_insert
_trajectory.py 같은 녹화 스크립트로 교체).
"""
import json
import os
import random
import time

import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from fms_interfaces.action import BusbarInsert, NutFasten
from fms_interfaces.msg import BusbarGrasp, NutPose, StudPose

FASTEN_TORQUE_MIN_NM = 8.0
FASTEN_TORQUE_MAX_NM = 12.0
FASTEN_SUCCESS_THRESHOLD_NM = 9.0

# TODO 로직 처리 시뮬레이션 시간 (실제로는 IK/모션 플래닝 완료 시점에 결과 발행)
ACTION_DELAY_SEC = 1.5

# 10_busbar_assembly.py의 phase 이름을 그대로 재사용 (BUSBAR_APPROACH ... RELEASE_AND_RETRACT)
BUSBAR_GRASP_PHASES = ['BUSBAR_APPROACH', 'BUSBAR_DESCEND', 'BUSBAR_GRASP', 'BUSBAR_LIFT']
BUSBAR_INSERT_PHASES = ['MOVE_TO_BOLT_APPROACH', 'BUSBAR_DESCEND_TO_BOLT', 'BUSBAR_RELEASE_AND_RETRACT']

TRAJECTORY_PATH = os.path.join(os.path.dirname(__file__), 'data', 'nut_fasten_trajectory.json')
REPLAY_JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6',
                       'finger_joint', 'right_inner_knuckle_joint']


class ArmNode(Node):

    def __init__(self):
        super().__init__('arm_node')

        self._cb_group = ReentrantCallbackGroup()

        self._busbar_action_server = ActionServer(
            self, BusbarInsert, 'busbar_insert',
            execute_callback=self._execute_busbar_insert,
            callback_group=self._cb_group,
        )
        self._fasten_action_server = ActionServer(
            self, NutFasten, 'nut_fasten',
            execute_callback=self._execute_nut_fasten,
            callback_group=self._cb_group,
        )

        # perception_node(또는 dummy_executor_node) 인터페이스 - feedback용 최신 vision pose 보관
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

        # Isaac Sim 연동 인터페이스
        self._joint_states_sub = self.create_subscription(
            JointState, '/joint_states', self._on_joint_states, 10,
            callback_group=self._cb_group)
        self._joint_command_pub = self.create_publisher(JointState, '/arm/joint_command', 10)
        self._latest_joint_states = None

        self._trajectory = self._load_trajectory()

        self.get_logger().info('arm_node started')

    def _load_trajectory(self):
        try:
            with open(TRAJECTORY_PATH) as f:
                trajectory = json.load(f)
        except OSError as exc:
            self.get_logger().warn(f'궤적 파일을 불러오지 못함 ({TRAJECTORY_PATH}): {exc}')
            return None
        indices = [trajectory['dof_names'].index(name) for name in REPLAY_JOINT_NAMES]
        trajectory['replay_indices'] = indices
        return trajectory

    def _on_joint_states(self, msg: JointState):
        # TODO: 실제 접촉/토크 판정에 조인트 effort를 활용.
        self._latest_joint_states = msg

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

    # --- 버스바 파지 · 삽입 --------------------------------------------------
    def _execute_busbar_insert(self, goal_handle):
        goal = goal_handle.request
        self.get_logger().info(f'ACTION /busbar_insert 시작 <- {goal.command} ({goal.station_id})')

        if goal.command == 'GRASP':
            phases = BUSBAR_GRASP_PHASES
        elif goal.command == 'INSERT':
            phases = BUSBAR_INSERT_PHASES
        else:
            self.get_logger().warn(f'알 수 없는 busbar command: {goal.command}')
            goal_handle.abort()
            return BusbarInsert.Result(success=False, message=f'알 수 없는 command: {goal.command}')

        # TODO: 실제 파지 IK/모션 플래닝, 스터드 삽입·접촉 탐색(force feedback) 로직.
        # 지금은 target_pose를 그대로 흘려보내기만 하는 placeholder.
        self._send_joint_command_towards_target(goal.target_pose)

        tick_period = ACTION_DELAY_SEC / len(phases)
        for i, phase in enumerate(phases):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return BusbarInsert.Result(success=False, message='취소됨')

            feedback = BusbarInsert.Feedback()
            feedback.phase = phase
            feedback.progress = (i + 1) / len(phases)
            feedback.vision_target_pose = self._latest_busbar_vision_pose()
            goal_handle.publish_feedback(feedback)
            time.sleep(tick_period)

        goal_handle.succeed()
        message = '버스바 파지 완료' if goal.command == 'GRASP' else '버스바 삽입 완료'
        self.get_logger().info(f'ACTION /busbar_insert 완료 -> success=True ({message})')
        return BusbarInsert.Result(success=True, message=message)

    def _send_joint_command_towards_target(self, target_pose: Pose):
        # TODO: target_pose에 대한 역기구학 계산 결과를 조인트 커맨드로 변환.
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        self._joint_command_pub.publish(cmd)

    # --- 너트 체결 시퀀스 ----------------------------------------------------
    def _execute_nut_fasten(self, goal_handle):
        goal = goal_handle.request
        self.get_logger().info(f'ACTION /nut_fasten 시작 <- {goal.command} (nut_id={goal.nut_id})')

        if goal.command not in ('APPROACH', 'FASTEN'):
            self.get_logger().warn(f'알 수 없는 fasten command: {goal.command}')
            goal_handle.abort()
            return NutFasten.Result(success=False, torque=0.0, message=f'알 수 없는 command: {goal.command}')

        if self._trajectory is None:
            return self._execute_nut_fasten_dummy(goal_handle, goal)

        return self._execute_nut_fasten_replay(goal_handle, goal)

    def _execute_nut_fasten_dummy(self, goal_handle, goal):
        # TODO: 실제 토크 센서 기반 체결. 궤적 데이터가 없을 때만 쓰는 대체 경로.
        self.get_logger().warn('궤적 데이터 없음 -> 더미 딜레이로 대체')
        time.sleep(ACTION_DELAY_SEC)

        if goal.command == 'APPROACH':
            goal_handle.succeed()
            return NutFasten.Result(success=True, torque=0.0, message='너트 접근 완료')

        torque = random.uniform(FASTEN_TORQUE_MIN_NM, FASTEN_TORQUE_MAX_NM)
        success = torque >= FASTEN_SUCCESS_THRESHOLD_NM
        if success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        message = '체결 완료' if success else f'토크 부족 ({torque:.2f} Nm)'
        return NutFasten.Result(success=success, torque=torque, message=message)

    def _execute_nut_fasten_replay(self, goal_handle, goal):
        segment = 'approach' if goal.command == 'APPROACH' else 'fasten'
        frames = self._trajectory[segment]
        indices = self._trajectory['replay_indices']
        period = self._trajectory['physics_dt']
        total = max(len(frames), 1)

        for idx, frame in enumerate(frames):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return NutFasten.Result(success=False, torque=0.0, message='취소됨')

            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name = REPLAY_JOINT_NAMES
            cmd.position = [frame['positions'][i] for i in indices]
            self._joint_command_pub.publish(cmd)

            feedback = NutFasten.Feedback()
            feedback.phase = f'{segment.upper()}_REPLAY'
            feedback.progress = (idx + 1) / total
            feedback.vision_target_pose = self._latest_nut_vision_pose()
            goal_handle.publish_feedback(feedback)
            time.sleep(period)

        goal_handle.succeed()
        if segment == 'approach':
            message = '너트 접근 완료 (기록 궤적 재생)'
            result = NutFasten.Result(success=True, torque=0.0, message=message)
        else:
            # TODO: 실제 토크 센서 기반 판정. 지금은 궤적 재생 완료를 성공 신호로 사용.
            torque = random.uniform(FASTEN_SUCCESS_THRESHOLD_NM, FASTEN_TORQUE_MAX_NM)
            message = '체결 완료 (기록 궤적 재생)'
            result = NutFasten.Result(success=True, torque=torque, message=message)

        self.get_logger().info(f'ACTION /nut_fasten 완료 -> {message}')
        return result


def main(args=None):
    rclpy.init(args=args)
    node = ArmNode()
    executor = MultiThreadedExecutor()
    try:
        rclpy.spin(node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

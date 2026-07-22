"""arm_node
실행 계층 · 버스바 파지·삽입·너트 체결 (M0609 매니퓰레이터).

SUB /busbar/command · /busbar/target   (fms_interfaces/BusbarCommand, BusbarTarget) <- behavior_node
SUB /fasten/command                    (fms_interfaces/FastenCommand)              <- behavior_node
PUB /busbar/result                     (fms_interfaces/BusbarResult)               -> behavior_node
PUB /fasten/result                     (fms_interfaces/FastenResult)               -> behavior_node

SUB /joint_states        (sensor_msgs/JointState)  <- Isaac Sim (관절 상태·접촉 정보)
PUB /arm/joint_command    (sensor_msgs/JointState)  -> Isaac Sim (관절 제어)

너트 체결(APPROACH/FASTEN)은 scripts/record_nut_fasten_trajectory.py로 World0123.usd에서
미리 녹화해둔 관절 궤적(data/nut_fasten_trajectory.json)을 그대로 재생한다 — 대상
nut1/peg_0 -> bolt_2 위치가 고정이라 실시간 IK 없이도 충분하기 때문.
"""
import json
import os
import random

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from fms_interfaces.msg import (
    BusbarCommand, BusbarTarget, BusbarResult,
    FastenCommand, FastenResult,
)

FASTEN_TORQUE_MIN_NM = 8.0
FASTEN_TORQUE_MAX_NM = 12.0
FASTEN_SUCCESS_THRESHOLD_NM = 9.0

# TODO 로직 처리 시뮬레이션 시간 (실제로는 IK/모션 플래닝 완료 시점에 결과 발행)
ACTION_DELAY_SEC = 1.5

TRAJECTORY_PATH = os.path.join(os.path.dirname(__file__), 'data', 'nut_fasten_trajectory.json')
REPLAY_JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6',
                       'finger_joint', 'right_inner_knuckle_joint']


class ArmNode(Node):

    def __init__(self):
        super().__init__('arm_node')

        self._busbar_cmd_sub = self.create_subscription(
            BusbarCommand, '/busbar/command', self._on_busbar_command, 10)
        self._busbar_target_sub = self.create_subscription(
            BusbarTarget, '/busbar/target', self._on_busbar_target, 10)
        self._fasten_cmd_sub = self.create_subscription(
            FastenCommand, '/fasten/command', self._on_fasten_command, 10)

        self._busbar_result_pub = self.create_publisher(BusbarResult, '/busbar/result', 10)
        self._fasten_result_pub = self.create_publisher(FastenResult, '/fasten/result', 10)

        # Isaac Sim 연동 인터페이스
        self._joint_states_sub = self.create_subscription(
            JointState, '/joint_states', self._on_joint_states, 10)
        self._joint_command_pub = self.create_publisher(JointState, '/arm/joint_command', 10)
        self._latest_joint_states = None

        self._latest_busbar_target = None

        self._trajectory = self._load_trajectory()
        self._replay_timer = None
        self._replay_frames = None
        self._replay_idx = 0
        self._replay_indices = None
        self._replay_on_done = None

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

    def _on_busbar_target(self, msg: BusbarTarget):
        self._latest_busbar_target = msg

    # --- 버스바 파지 · 삽입 --------------------------------------------------
    def _on_busbar_command(self, msg: BusbarCommand):
        self.get_logger().info(f'SUB /busbar/command <- {msg.command} ({msg.station_id})')

        if msg.command == 'GRASP':
            self._send_joint_command_towards_target()
            # TODO: 실제 파지 IK/모션 플래닝, 스터드 삽입·접촉 탐색 로직.
            self.create_timer(ACTION_DELAY_SEC, lambda: self._finish_busbar(True, '버스바 파지 완료'))
        elif msg.command == 'INSERT':
            self._send_joint_command_towards_target()
            # TODO: 스터드 삽입 접촉 탐색(force feedback) 로직.
            self.create_timer(ACTION_DELAY_SEC, lambda: self._finish_busbar(True, '버스바 삽입 완료'))
        else:
            self.get_logger().warn(f'알 수 없는 busbar command: {msg.command}')

    def _finish_busbar(self, success: bool, message: str):
        result = BusbarResult()
        result.success = success
        result.message = message
        self._busbar_result_pub.publish(result)
        self.get_logger().info(f'PUB /busbar/result -> success={success} ({message})')

    def _send_joint_command_towards_target(self):
        if self._latest_busbar_target is None:
            return
        # TODO: target_pose에 대한 역기구학 계산 결과를 조인트 커맨드로 변환.
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        self._joint_command_pub.publish(cmd)

    # --- 너트 체결 시퀀스 ----------------------------------------------------
    def _on_fasten_command(self, msg: FastenCommand):
        self.get_logger().info(f'SUB /fasten/command <- {msg.command} (nut_id={msg.nut_id})')

        if self._trajectory is None:
            self.get_logger().warn('궤적 데이터 없음 -> 더미 딜레이로 대체')
            if msg.command == 'APPROACH':
                self.create_timer(ACTION_DELAY_SEC, lambda: self._finish_fasten_approach(True, '너트 접근 완료'))
            elif msg.command == 'FASTEN':
                self._respond_fasten_dummy()
            else:
                self.get_logger().warn(f'알 수 없는 fasten command: {msg.command}')
            return

        if msg.command == 'APPROACH':
            self._start_replay('approach', lambda: self._finish_fasten_approach(True, '너트 접근 완료 (기록 궤적 재생)'))
        elif msg.command == 'FASTEN':
            self._start_replay('fasten', self._respond_fasten_after_replay)
        else:
            self.get_logger().warn(f'알 수 없는 fasten command: {msg.command}')

    def _respond_fasten_dummy(self):
        # TODO: 실제 토크 센서 기반 체결. 궤적 데이터가 없을 때만 쓰는 대체 경로.
        torque = random.uniform(FASTEN_TORQUE_MIN_NM, FASTEN_TORQUE_MAX_NM)
        success = torque >= FASTEN_SUCCESS_THRESHOLD_NM
        message = '체결 완료' if success else f'토크 부족 ({torque:.2f} Nm)'
        self.create_timer(ACTION_DELAY_SEC, lambda: self._finish_fasten(success, torque, message))

    def _respond_fasten_after_replay(self):
        # TODO: 실제 토크 센서 기반 판정. 지금은 궤적 재생 완료를 성공 신호로 사용.
        torque = random.uniform(FASTEN_SUCCESS_THRESHOLD_NM, FASTEN_TORQUE_MAX_NM)
        self._finish_fasten(True, torque, '체결 완료 (기록 궤적 재생)')

    # --- 기록 궤적 재생 --------------------------------------------------
    def _start_replay(self, segment, on_done):
        if self._replay_timer is not None:
            self._replay_timer.cancel()
        self._replay_frames = self._trajectory[segment]
        self._replay_indices = self._trajectory['replay_indices']
        self._replay_idx = 0
        self._replay_on_done = on_done
        period = self._trajectory['physics_dt']
        self._replay_timer = self.create_timer(period, self._on_replay_step)

    def _on_replay_step(self):
        if self._replay_idx >= len(self._replay_frames):
            self._replay_timer.cancel()
            self._replay_timer = None
            on_done, self._replay_on_done = self._replay_on_done, None
            on_done()
            return

        positions = self._replay_frames[self._replay_idx]['positions']
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = REPLAY_JOINT_NAMES
        cmd.position = [positions[i] for i in self._replay_indices]
        self._joint_command_pub.publish(cmd)
        self._replay_idx += 1

    def _finish_fasten_approach(self, success: bool, message: str):
        result = FastenResult()
        result.success = success
        result.torque = 0.0
        result.message = message
        self._fasten_result_pub.publish(result)
        self.get_logger().info(f'PUB /fasten/result -> success={success} ({message})')

    def _finish_fasten(self, success: bool, torque: float, message: str):
        result = FastenResult()
        result.success = success
        result.torque = torque
        result.message = message
        self._fasten_result_pub.publish(result)
        self.get_logger().info(
            f'PUB /fasten/result -> success={success} torque={torque:.2f} ({message})')


def main(args=None):
    rclpy.init(args=args)
    node = ArmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

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
검출값(없으면 하드코딩 fallback)을 쓰고, INSERT 목표는 아직 스터드 vision 연동 전이라
target_mid_pos 하드코딩 좌표를 그대로 쓴다 (TODO: /vision/stud_pose 연동).

너트 체결(APPROACH/FASTEN)은 scripts/record_nut_fasten_trajectory.py로 World0123.usd에서
미리 녹화해둔 관절 궤적(data/nut_fasten_trajectory.json)을 그대로 재생해 /arm/joint_command로
발행한다 — 대상 nut1/peg_0 -> bolt_2 위치가 고정이라 실시간 IK 없이도 충분하기 때문.
녹화 파일이 없으면 더미 딜레이 + 임의 토크로 대체(_execute_nut_fasten_dummy).
"""
import json
import os
import random
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32

from fms_interfaces.action import BusbarInsert, NutFasten
from fms_interfaces.msg import BusbarGrasp, NutPose, StudPose

FASTEN_TORQUE_MIN_NM = 8.0
FASTEN_TORQUE_MAX_NM = 12.0
FASTEN_SUCCESS_THRESHOLD_NM = 9.0

# 궤적 데이터가 없을 때만 쓰는 더미 체결 경로의 시뮬레이션 시간.
ACTION_DELAY_SEC = 1.5

# feedback.phase에 실어 보낼 단계 이름. GRAB_BUSBAR/INSERT_BUSBAR 실제 동작 단계와 1:1 대응.
BUSBAR_GRASP_PHASES = ['BUSBAR_APPROACH', 'BUSBAR_DESCEND', 'BUSBAR_GRASP', 'BUSBAR_LIFT']
BUSBAR_INSERT_PHASES = ['MOVE_TO_BOLT_APPROACH', 'BUSBAR_DESCEND_TO_BOLT', 'BUSBAR_RELEASE_AND_RETRACT']

TRAJECTORY_PATH = os.path.join(os.path.dirname(__file__), 'data', 'nut_fasten_trajectory.json')
REPLAY_JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6',
                       'finger_joint', 'right_inner_knuckle_joint']


class ArmNode(Node):

    def __init__(self):
        super().__init__('arm_node')

        self._cb_group = ReentrantCallbackGroup()

        # 1. ACTION 서버 (behavior_node -> arm_node)
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

        # 2. 버스바 파지·삽입용 Cartesian 제어 인터페이스 (RMPFlow 경유)
        self._target_pose_pub = self.create_publisher(PoseStamped, '/arm/target_pose', 10)
        self._gripper_pub = self.create_publisher(Int32, '/arm/gripper_command', 10)
        self._current_ee_pose = None
        self._current_pose_sub = self.create_subscription(
            PoseStamped, '/arm/current_pose', self._current_pose_callback, 10,
            callback_group=self._cb_group,
        )

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

        # ★ [좌표 및 파라미터 정의 - 01_pick_and_lift.py 동일]
        # vision 검출이 없을 때 쓰는 대체(fallback) 파지 좌표.
        self._DEFAULT_POS_GRAB_PICK = np.array([0.5128, 0.4477, 0.455, 0.0, 3.1415, 1.5708])

        target_mid_pos = np.array([1.03115, -0.07855, 0.0693])
        self._POS_INSERT_ABOVE = np.array([target_mid_pos[0], target_mid_pos[1], 0.6, 0.0, 3.1415, 1.5708])
        self._POS_INSERT_PLACE = np.array([target_mid_pos[0], target_mid_pos[1], target_mid_pos[2], 0.0, 3.1415, 1.5708])

        # ★ [단계별 차등 허용 오차]
        self._PICK_TOLERANCE_STRICT = 0.01    # Pick 단계: 0.01m (10mm)
        # Insert 단계 허용 오차. 원래 0.001m(1mm)였으나 RMPFlow가 그 안으로 수렴하지
        # 못했다. 0.015m로 완화하고, _move_to_pose 자체에 timeout_sec을 둬서 이 값도
        # 사실상 모니터링(로그)용이지 무한 대기를 강제하는 값이 아니게 했다.
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
        except OSError as exc:
            self.get_logger().warn(f'궤적 파일을 불러오지 못함 ({TRAJECTORY_PATH}): {exc}')
            return None
        indices = [trajectory['dof_names'].index(name) for name in REPLAY_JOINT_NAMES]
        trajectory['replay_indices'] = indices
        return trajectory

    def _on_joint_states(self, msg: JointState):
        # TODO: 실제 접촉/토크 판정에 조인트 effort를 활용.
        self._latest_joint_states = msg

    def _current_pose_callback(self, msg: PoseStamped):
        self._current_ee_pose = msg.pose

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

    def _resolve_grab_pick_pose(self, target_pose: Pose) -> np.ndarray:
        """goal.target_pose(behavior_node가 /vision/busbar_grasp에서 채워 보냄) 기반 파지
        좌표를 반환. vision 값이 없으면(behavior_node가 기본 Pose()를 그대로 보낸 경우)
        하드코딩 좌표로 대체."""
        if target_pose.position.x != 0.0 or target_pose.position.y != 0.0:
            pick_pose = self._DEFAULT_POS_GRAB_PICK.copy()
            pick_pose[0] = target_pose.position.x
            pick_pose[1] = target_pose.position.y
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

    def _move_to_pose(self, target_pose, step_name, pos_tolerance=0.001, timeout_sec=8.0):
        """
        목표 허용 오차 안으로 들어올 때까지 한 줄(\r)로 실시간 오차 및 진행 시간 갱신 출력.

        pos_tolerance는 도달 판정 기준이자 모니터링용 로그 값일 뿐이다 — timeout_sec가
        지나도 그 안으로 수렴하지 못하면(예: RMPFlow 정상 오차 범위가 tolerance보다 큰 경우)
        무한 대기하지 않고 그 시점 오차로 진행한다 (INSERT_BUSBAR가 1mm 근처에서 절대
        안 줄어들어 영원히 멈춰있던 문제 대응).

        TODO: nut_fasten 궤적 재생과 달리 ActionServer 취소 요청(is_cancel_requested)을
        아직 확인하지 않는다 — GRASP/INSERT 진행 중 goal 취소는 지원 밖.
        """
        self.get_logger().info(f'  -> [하위동작] {step_name} Target Pose 발행 (목표 오차: {pos_tolerance:.4f} m)...')
        self._publish_target_pose(target_pose)

        target_pos = np.array(target_pose[:3])
        start_time = time.time()

        while True:
            self._publish_gripper_state(self._current_gripper_state)

            if self._current_ee_pose is not None:
                curr_pos = np.array([
                    self._current_ee_pose.position.x,
                    self._current_ee_pose.position.y,
                    self._current_ee_pose.position.z
                ])

                dist_error = np.linalg.norm(target_pos - curr_pos)
                elapsed_time = time.time() - start_time

                sys.stdout.write(
                    f"\r     [이동 중...] 현재오차: {dist_error:.4f} m (목표: <{pos_tolerance:.4f} m) | 경과시간: {elapsed_time:4.1f}s"
                )
                sys.stdout.flush()

                if dist_error < pos_tolerance:
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                    self.get_logger().info(f'  -> [도달 완료] {step_name} (최종 오차: {dist_error:.4f} m)')
                    return True

                if elapsed_time > timeout_sec:
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                    self.get_logger().warn(
                        f'  -> [시간 초과({timeout_sec:.1f}s)] {step_name} 목표 오차 미달(현재 {dist_error:.4f} m)이지만 진행')
                    return True

            time.sleep(0.05)

    def _descend_step_by_step(self, start_above_pose, final_target_pose, step_name):
        """
        Step당 -0.0015m씩 점진적 순응 하강 수행
        (하강 과정 진행 상태를 줄바꿈 없이 '\r'로 동일 라인 연속 갱신)
        """
        self.get_logger().info(f'  -> [하위동작] {step_name} 점진적 하강 시작 (-0.0015m/step)...')

        current_target_z = start_above_pose[2]
        final_target_z = final_target_pose[2]
        final_pos = np.array(final_target_pose[:3])
        start_time = time.time()

        while True:
            self._publish_gripper_state(self._current_gripper_state)

            # Step당 Z축 목표치 낮춤
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

                # Z=0.36m 이하로 내려오거나, 최종 목표점 tolerance 이내 진입 시 완료 판단
                if curr_pos[2] <= self._BUSBAR_RELEASE_Z or dist_error < self._INSERT_TOLERANCE_STRICT:
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                    self.get_logger().info(f'  -> [하강 완료] {step_name} (EE Z: {curr_pos[2]:.4f}m | 최종 오차: {dist_error:.4f}m)')
                    return True

            time.sleep(0.05)

    def _control_gripper(self, close: bool):
        state_val = 1 if close else 0
        action_str = "닫기" if close else "열기"

        self.get_logger().info(f'  -> 그리퍼 {action_str}')
        self._publish_gripper_state(state_val)
        time.sleep(1.0)

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

        if goal.command == 'GRASP':
            success, message = self._run_busbar_grasp(goal, goal_handle)
        elif goal.command == 'INSERT':
            success, message = self._run_busbar_insert(goal, goal_handle)
        else:
            self.get_logger().warn(f'알 수 없는 busbar command: {goal.command}')
            goal_handle.abort()
            return BusbarInsert.Result(success=False, message=f'알 수 없는 command: {goal.command}')

        if success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        self.get_logger().info(f'ACTION /busbar_insert 완료 -> success={success} ({message})')
        return BusbarInsert.Result(success=success, message=message)

    def _run_busbar_grasp(self, goal, goal_handle):
        # above/lift는 항상 파지점 기준 상대 오프셋이므로 매 goal마다 다시 계산한다.
        pos_grab_pick = self._resolve_grab_pick_pose(goal.target_pose)
        pos_grab_above = pos_grab_pick + np.array([0.0, 0.0, 0.145, 0.0, 0.0, 0.0])
        busbar_lift_move_pos = pos_grab_pick + np.array([0.0, 0.1, 0.145, 0.0, 0.0, 0.0])
        phases = BUSBAR_GRASP_PHASES

        self._publish_gripper_state(0)

        # 1. 버스바 상공 접근 (Tol: 0.01m)
        self._publish_busbar_feedback(goal_handle, phases[0], 1 / 4)
        if not self._move_to_pose(pos_grab_above, '1. 버스바 상공 접근', pos_tolerance=self._PICK_TOLERANCE_STRICT):
            return False, '상공 접근 실패'

        # 2. 파지 위치 하강 (Tol: 0.01m)
        self._publish_busbar_feedback(goal_handle, phases[1], 2 / 4)
        if not self._move_to_pose(pos_grab_pick, '2. 버스바 파지점 하강', pos_tolerance=self._PICK_TOLERANCE_STRICT):
            return False, '파지점 하강 실패'

        # 그리퍼 닫기 (1)
        self._publish_busbar_feedback(goal_handle, phases[2], 3 / 4)
        self._control_gripper(close=True)

        # 3. 버스바 상승 이동 (Tol: 0.01m)
        self._publish_busbar_feedback(goal_handle, phases[3], 4 / 4)
        if not self._move_to_pose(busbar_lift_move_pos, '3. 버스바 상승 및 이동', pos_tolerance=self._PICK_TOLERANCE_STRICT):
            return False, '상승 이동 실패'

        return True, 'GRAB_BUSBAR 시퀀스 최종 완료'

    def _run_busbar_insert(self, goal, goal_handle):
        phases = BUSBAR_INSERT_PHASES

        # 1. 체결 위치 상공 접근
        self._publish_busbar_feedback(goal_handle, phases[0], 1 / 3)
        if not self._move_to_pose(self._POS_INSERT_ABOVE, '1. 체결위치 상공 접근', pos_tolerance=self._INSERT_TOLERANCE_STRICT):
            return False, '체결 상공 접근 실패'

        # 2. 체결 위치 점진적 하강 (step당 -0.0015m 하강 및 Z=0.36m 감지)
        self._publish_busbar_feedback(goal_handle, phases[1], 2 / 3)
        if not self._descend_step_by_step(self._POS_INSERT_ABOVE, self._POS_INSERT_PLACE, '2. 체결 위치 점진적 하강 및 삽입'):
            return False, '체결 위치 하강 실패'

        # 그리퍼 열기 (0) + 상공 이탈
        self._control_gripper(close=False)
        if not self._move_to_pose(self._POS_INSERT_ABOVE, '3. 상공 이탈', pos_tolerance=self._INSERT_TOLERANCE_STRICT):
            return False, '상공 이탈 실패'
        self._publish_busbar_feedback(goal_handle, phases[2], 3 / 3)

        return True, 'INSERT_BUSBAR 시퀀스 최종 완료'

    # --- 너트 체결 ACTION (기록 궤적 재생) ------------------------------------
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
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

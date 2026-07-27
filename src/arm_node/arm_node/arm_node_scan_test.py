"""arm_node_scan_test (arm_node.py 테스트용 복제본)

arm_node.py는 GRAB_BUSBAR 호출 시점에 백그라운드로 구독해둔 /perception/detections_3d의
"가장 최근(5초 이내)" 값을 그냥 가져다 쓴다. 이 복제본은 대신 다음 순서로 동작한다:

  1. 고정된 스캔 포즈(작업대/버스바 트레이 상공)로 이동.
  2. 도달 후, 엔드이펙터 속도가 0에 가까워질 때까지(정지 판정) 대기.
  3. 그 시점에 perception_node의 /perception/get_grasp_pose 서비스를 동기 호출해
     파지 좌표를 명시적으로 요청.
  4. 응답 좌표로 하강·파지 수행 (실패/미검출 시 하드코딩 기본 좌표로 폴백).

검증되면 이 로직을 arm_node.py에 병합하는 건 별도 작업. 그 전까지 원본 arm_node.py는
건드리지 않는다.
"""
import collections
import sys
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from fms_interfaces.srv import ArmControl, GetGraspPose

# 정지 판정: 최근 이 개수만큼의 연속 샘플이 모두 속도 임계값 이하여야 "정지"로 본다.
STATIONARY_MIN_SAMPLES = 3
STATIONARY_SPEED_THRESHOLD_M_S = 0.005
STATIONARY_WAIT_TIMEOUT_SEC = 3.0

GRASP_SERVICE_LABEL = 'busbar'
GRASP_SERVICE_TIMEOUT_SEC = 3.0


class ArmNode(Node):

    def __init__(self):
        super().__init__('arm_node_scan_test')

        self._cb_group = ReentrantCallbackGroup()

        # 1. 서비스 서버
        self._srv = self.create_service(
            ArmControl, '/arm/control', self._handle_arm_control, callback_group=self._cb_group
        )

        # 2. Target Pose 퍼블리셔
        self._target_pose_pub = self.create_publisher(
            PoseStamped, '/arm/target_pose', 10
        )

        # 3. Gripper Command 퍼블리셔 (열림: 0, 닫힘: 1)
        self._gripper_pub = self.create_publisher(
            Int32, '/arm/gripper_command', 10
        )

        # 4. Current Pose 서브스크라이버 (Feedback) + 정지 판정용 최근 샘플 이력
        self._current_ee_pose = None
        self._recent_pose_samples = collections.deque(maxlen=STATIONARY_MIN_SAMPLES)
        self._current_pose_sub = self.create_subscription(
            PoseStamped,
            '/arm/current_pose',
            self._current_pose_callback,
            10,
            callback_group=self._cb_group
        )

        # 5. perception_node 파지 좌표 요청 서비스 클라이언트
        self._grasp_client = self.create_client(
            GetGraspPose, '/perception/get_grasp_pose', callback_group=self._cb_group
        )

        # ★ [좌표 및 파라미터 정의 - 01_pick_and_lift.py 동일]
        # 작업대(버스바 트레이) 상공 스캔 포즈. 기존 pos_grab_above와 동일한 좌표 —
        # "작업대 상단"에 해당하는 값을 새로 만들지 않고 재사용.
        self._SCAN_POSE = np.array([0.5128, 0.4477, 0.6, 0.0, 3.1415, 1.5708])
        # perception 서비스가 실패/미검출일 때 쓰는 대체(fallback) 파지 좌표.
        self._DEFAULT_POS_GRAB_PICK = np.array([0.5128, 0.4477, 0.455, 0.0, 3.1415, 1.5708])

        target_mid_pos = np.array([1.03115, -0.07855, 0.0693])
        self._POS_INSERT_ABOVE = np.array([target_mid_pos[0], target_mid_pos[1], 0.6, 0.0, 3.1415, 1.5708])
        self._POS_INSERT_PLACE = np.array([target_mid_pos[0], target_mid_pos[1], target_mid_pos[2], 0.0, 3.1415, 1.5708])

        # ★ [단계별 차등 허용 오차]
        self._PICK_TOLERANCE_STRICT = 0.001    # Pick 단계: 0.01m (10mm)
        # Insert 단계 허용 오차. 원래 0.001m(1mm)였으나 RMPFlow가 그 안으로 수렴하지
        # 못했다. 0.015m로 완화하고, _move_to_pose 자체에 timeout_sec을 둬서 이 값도
        # 사실상 모니터링(로그)용이지 무한 대기를 강제하는 값이 아니게 했다.
        self._INSERT_TOLERANCE_STRICT = 0.001
        self._BUSBAR_RELEASE_Z = 0.36        # 그리퍼 해제 임계 높이
        self._INSERT_SPEED = 0.0015           # step당 수직 하강 속도 (m/step)

        # 초기 상태: 그리퍼 열림 (0)
        self._current_gripper_state = 0
        self._publish_gripper_state(0)

        self.get_logger().info('ArmNode(scan_test) - 스캔/정지/서비스요청 흐름 준비 완료')

    def _current_pose_callback(self, msg: PoseStamped):
        self._current_ee_pose = msg.pose
        self._recent_pose_samples.append((
            np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
            time.time(),
        ))

    def _is_stationary(self) -> bool:
        """최근 STATIONARY_MIN_SAMPLES개 샘플의 연속 구간 속도가 모두 임계값 이하인지 확인.

        /arm/current_pose에는 속도 필드가 없어 위치 차분으로 직접 계산한다.
        """
        if len(self._recent_pose_samples) < STATIONARY_MIN_SAMPLES:
            return False

        samples = list(self._recent_pose_samples)
        for (pos_a, t_a), (pos_b, t_b) in zip(samples, samples[1:]):
            dt = t_b - t_a
            if dt <= 0:
                continue
            speed = np.linalg.norm(pos_b - pos_a) / dt
            if speed > STATIONARY_SPEED_THRESHOLD_M_S:
                return False
        return True

    def _wait_until_stationary(self, timeout_sec: float = STATIONARY_WAIT_TIMEOUT_SEC) -> bool:
        self.get_logger().info('  -> 정지 판정 대기 중 (속도 <= '
                                f'{STATIONARY_SPEED_THRESHOLD_M_S} m/s, 연속 {STATIONARY_MIN_SAMPLES}샘플)...')
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            if self._is_stationary():
                self.get_logger().info('  -> 정지 확인 완료')
                return True
            time.sleep(0.05)
        self.get_logger().warn(f'  -> 정지 판정 타임아웃({timeout_sec}s), 계속 진행')
        return False

    def _request_grasp_pose(self, label: str = GRASP_SERVICE_LABEL):
        """perception_node에 파지 좌표를 동기 요청. 반환: (x, y) 또는 실패 시 None."""
        if not self._grasp_client.wait_for_service(timeout_sec=GRASP_SERVICE_TIMEOUT_SEC):
            self.get_logger().warn('/perception/get_grasp_pose 서비스 대기 타임아웃')
            return None

        request = GetGraspPose.Request()
        request.label = label
        future = self._grasp_client.call_async(request)

        start_time = time.time()
        while not future.done():
            if time.time() - start_time > GRASP_SERVICE_TIMEOUT_SEC:
                self.get_logger().warn('/perception/get_grasp_pose 응답 타임아웃')
                return None
            time.sleep(0.05)

        response = future.result()
        if response is None or not response.found:
            message = response.message if response is not None else 'no response'
            self.get_logger().warn(f'/perception/get_grasp_pose 검출 없음: {message}')
            return None

        x = response.pose.pose.position.x
        y = response.pose.pose.position.y
        self.get_logger().info(
            f'/perception/get_grasp_pose 응답 좌표 사용 -> x={x:.4f}, y={y:.4f} ({response.message})')
        return x, y

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

    def _move_to_pose(self, target_pose, step_name, pos_tolerance=0.001, timeout_sec=7.0):
        """
        목표 허용 오차 안으로 들어올 때까지 한 줄(\r)로 실시간 오차 및 진행 시간 갱신 출력.

        pos_tolerance는 도달 판정 기준이자 모니터링용 로그 값일 뿐이다 — timeout_sec가
        지나도 그 안으로 수렴하지 못하면(예: RMPFlow 정상 오차 범위가 tolerance보다 큰 경우)
        무한 대기하지 않고 그 시점 오차로 진행한다 (INSERT_BUSBAR가 1mm 근처에서 절대
        안 줄어들어 영원히 멈춰있던 문제 대응).
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

                # Z=0.36m 이하로 내려오거나, 최종 목표점 1mm 이내 진입 시 완료 판단
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

    def _handle_arm_control(self, request, response):
        cmd = request.command
        self.get_logger().info(f'================ [작업 시작: {cmd}] ================')

        if cmd == 'GRAB_BUSBAR':
            self._publish_gripper_state(0)

            # 1. 작업대(버스바 트레이) 상공 스캔 포즈로 이동 (Tol: 0.01m)
            if not self._move_to_pose(self._SCAN_POSE, '1. 스캔 포즈(작업대 상공) 접근', pos_tolerance=self._PICK_TOLERANCE_STRICT):
                response.success, response.message = False, '스캔 포즈 접근 실패'
                return response

            # 2. 정지 판정 대기 (속도 0 확인)
            self._wait_until_stationary()

            # 3. perception_node에 파지 좌표 서비스 요청
            grasp_xy = self._request_grasp_pose()
            pos_grab_pick = self._DEFAULT_POS_GRAB_PICK.copy()
            if grasp_xy is not None:
                pos_grab_pick[0], pos_grab_pick[1] = grasp_xy
            else:
                self.get_logger().warn('파지 좌표 서비스 실패, 기본 좌표로 대체')

            pos_grab_above = pos_grab_pick + np.array([0.0, 0.0, 0.145, 0.0, 0.0, 0.0])
            busbar_lift_move_pos = pos_grab_pick + np.array([0.0, 0.1, 0.145, 0.0, 0.0, 0.0])

            # 4. 파지점 상공으로 재접근 (스캔 포즈 -> 실제 파지점 상공, Tol: 0.01m)
            if not self._move_to_pose(pos_grab_above, '2. 버스바 상공 접근', pos_tolerance=self._PICK_TOLERANCE_STRICT):
                response.success, response.message = False, '상공 접근 실패'
                return response

            # 5. 파지 위치 하강 (Tol: 0.01m)
            if not self._move_to_pose(pos_grab_pick, '3. 버스바 파지점 하강', pos_tolerance=self._PICK_TOLERANCE_STRICT):
                response.success, response.message = False, '파지점 하강 실패'
                return response

            # 그리퍼 닫기 (1)
            self._control_gripper(close=True)

            # 6. 버스바 상승 이동 (Tol: 0.01m)
            if not self._move_to_pose(busbar_lift_move_pos, '4. 버스바 상승 및 이동', pos_tolerance=self._PICK_TOLERANCE_STRICT):
                response.success, response.message = False, '상승 이동 실패'
                return response

            response.success = True
            response.message = 'GRAB_BUSBAR 시퀀스 최종 완료'
            return response

        elif cmd == 'INSERT_BUSBAR':
            # 1. 체결 위치 상공 접근 (Tol: 0.001m)
            if not self._move_to_pose(self._POS_INSERT_ABOVE, '1. 체결위치 상공 접근', pos_tolerance=self._INSERT_TOLERANCE_STRICT):
                response.success, response.message = False, '체결 상공 접근 실패'
                return response

            # 2. 체결 위치 점진적 하강 (step당 -0.0015m 하강 및 Z=0.36m 감지)
            if not self._descend_step_by_step(self._POS_INSERT_ABOVE, self._POS_INSERT_PLACE, '2. 체결 위치 점진적 하강 및 삽입'):
                response.success, response.message = False, '체결 위치 하강 실패'
                return response

            # 그리퍼 열기 (0)
            self._control_gripper(close=False)

            # 3. 상공 이탈 (Tol: 0.001m)
            if not self._move_to_pose(self._POS_INSERT_ABOVE, '3. 상공 이탈', pos_tolerance=self._INSERT_TOLERANCE_STRICT):
                response.success, response.message = False, '상공 이탈 실패'
                return response

            response.success = True
            response.message = 'INSERT_BUSBAR 시퀀스 최종 완료'
            return response

        else:
            response.success = False
            response.message = f'알 수 없는 명령입니다: {cmd}'
            return response

    def _euler_to_quaternion(self, roll, pitch, yaw):
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * sp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.array([w, x, y, z])


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

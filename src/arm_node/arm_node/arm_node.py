import sys
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from fms_interfaces.srv import ArmControl


class ArmNode(Node):

    def __init__(self):
        super().__init__('arm_node')

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

        # 4. Current Pose 서브스크라이버 (Feedback)
        self._current_ee_pose = None
        self._current_pose_sub = self.create_subscription(
            PoseStamped,
            '/arm/current_pose',
            self._current_pose_callback,
            10,
            callback_group=self._cb_group
        )

        # ★ [좌표 및 파라미터 정의 - 01_pick_and_lift.py 동일]
        self._POS_GRAB_PICK = np.array([0.5128, 0.4477, 0.455, 0.0, 3.1415, 1.5708])
        self._POS_GRAB_ABOVE = self._POS_GRAB_PICK + np.array([0.0, 0.0, 0.145, 0.0, 0.0, 0.0])
        self._BUSBAR_LIFT_MOVE_POS = self._POS_GRAB_PICK + np.array([0.0, 0.1, 0.145, 0.0, 0.0, 0.0])

        target_mid_pos = np.array([1.03115, -0.07855, 0.0693])
        self._POS_INSERT_ABOVE = np.array([target_mid_pos[0], target_mid_pos[1], 0.6, 0.0, 3.1415, 1.5708])
        self._POS_INSERT_PLACE = np.array([target_mid_pos[0], target_mid_pos[1], target_mid_pos[2], 0.0, 3.1415, 1.5708])

        # ★ [단계별 차등 허용 오차]
        self._PICK_TOLERANCE_STRICT = 0.01    # Pick 단계: 0.01m (10mm)
        self._INSERT_TOLERANCE_STRICT = 0.001 # Insert 단계: 0.001m (1mm)
        self._BUSBAR_RELEASE_Z = 0.36        # 그리퍼 해제 임계 높이
        self._INSERT_SPEED = 0.0015           # step당 수직 하강 속도 (m/step)

        # 초기 상태: 그리퍼 열림 (0)
        self._current_gripper_state = 0
        self._publish_gripper_state(0)

        self.get_logger().info('ArmNode (점진적 하강 & 완벽 피드백 제어 모드) 준비 완료')

    def _current_pose_callback(self, msg: PoseStamped):
        self._current_ee_pose = msg.pose

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

    def _move_to_pose(self, target_pose, step_name, pos_tolerance=0.001):
        """
        목표 허용 오차 안으로 들어올 때까지 한 줄(\r)로 실시간 오차 및 진행 시간 갱신 출력
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

                # ★ 동일한 터미널 라인에 실시간 Z 높이 및 오차/시간 한 줄 연속 갱신 (\r)
                # sys.stdout.write(
                #     f"\r     [점진하강 중...] 현재Z: {curr_pos[2]:.4f}m (목표Z: <=0.3600m) | 최종오차: {dist_error:.4f}m | 경과시간: {elapsed_time:4.1f}s"
                # )
                # sys.stdout.flush()

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
            
            # 1. 버스바 상공 접근 (Tol: 0.01m)
            if not self._move_to_pose(self._POS_GRAB_ABOVE, '1. 버스바 상공 접근', pos_tolerance=self._PICK_TOLERANCE_STRICT):
                response.success, response.message = False, '상공 접근 실패'
                return response

            # 2. 파지 위치 하강 (Tol: 0.01m)
            if not self._move_to_pose(self._POS_GRAB_PICK, '2. 버스바 파지점 하강', pos_tolerance=self._PICK_TOLERANCE_STRICT):
                response.success, response.message = False, '파지점 하강 실패'
                return response

            # 그리퍼 닫기 (1)
            self._control_gripper(close=True)

            # 3. 버스바 상승 이동 (Tol: 0.01m)
            if not self._move_to_pose(self._BUSBAR_LIFT_MOVE_POS, '3. 버스바 상승 및 이동', pos_tolerance=self._PICK_TOLERANCE_STRICT):
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
"""amr_node
실행 계층 · 목표 스테이션으로 이동, 도착·이동·오류 상태 보고.

SUB /amr/goal      (fms_interfaces/AmrGoal)     <- behavior_node
PUB /amr/status    (fms_interfaces/AmrStatus)   -> behavior_node

PUB /amr/cmd_vel   (geometry_msgs/Twist)        -> Isaac Sim (이동 명령)
SUB /amr/sim_pose  (geometry_msgs/Pose2D)       <- Isaac Sim (실제 위치, execute_isaac_stations.py)

목표 지점까지 단순 P 제어로 이동한다: 각오차(heading_error)가 클수록 cos(heading_error)로
전진 속도를 자연스럽게 줄이면서 제자리에 가깝게 회전하고, 오차가 작아질수록 그만큼 직진
비중을 늘린다 (회전/직진을 문턱값으로 딱 자르면 그 경계에서 진동하며 멈추는 문제가 있었음).
목표까지 거리가 ARRIVAL_TOLERANCE_M 이내면 정지하고 도착 보고한다.
"""
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose2D

from fms_interfaces.msg import AmrGoal, AmrStatus

ARRIVAL_TOLERANCE_M = 0.05

MAX_LINEAR_SPEED = 0.3   # m/s
MAX_ANGULAR_SPEED = 0.8  # rad/s
K_LINEAR = 0.8
K_ANGULAR = 1.5

CONTROL_PERIOD_SEC = 0.1  # 10Hz


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class AmrNode(Node):

    def __init__(self):
        super().__init__('amr_node')

        self._goal_sub = self.create_subscription(AmrGoal, '/amr/goal', self._on_goal, 10)
        self._status_pub = self.create_publisher(AmrStatus, '/amr/status', 10)

        self._cmd_vel_pub = self.create_publisher(Twist, '/amr/cmd_vel', 10)
        self._sim_pose_sub = self.create_subscription(Pose2D, '/amr/sim_pose', self._on_sim_pose, 10)

        self._current_pose = None  # Isaac Sim으로부터 실제 피드백을 받기 전까지는 알 수 없음
        self._goal = None
        self._moving = False
        self._control_timer = None

        self.get_logger().info('amr_node started')

    def _on_sim_pose(self, msg: Pose2D):
        self._current_pose = (msg.x, msg.y, msg.theta)

    def _on_goal(self, msg: AmrGoal):
        self.get_logger().info(f'SUB /amr/goal <- {msg.station_id} ({msg.x:.2f}, {msg.y:.2f})')

        if self._current_pose is None:
            self.get_logger().warn('/amr/sim_pose 피드백 없음 - Isaac Sim 연동 확인 필요, 이동 보류')
            self._publish_status(AmrStatus.STATE_ERROR, msg.station_id, 'sim_pose 피드백 없음')
            return

        self._goal = msg
        self._moving = True
        self._publish_status(AmrStatus.STATE_MOVING, msg.station_id, '이동 중')

        if self._control_timer is not None:
            self._control_timer.cancel()
        self._control_timer = self.create_timer(CONTROL_PERIOD_SEC, self._control_step)

    def _control_step(self):
        if self._goal is None or self._current_pose is None:
            return

        x, y, theta = self._current_pose
        dx = self._goal.x - x
        dy = self._goal.y - y
        distance = math.hypot(dx, dy)

        if distance <= ARRIVAL_TOLERANCE_M:
            self._on_arrived()
            return

        heading_to_goal = math.atan2(dy, dx)
        heading_error = _normalize_angle(heading_to_goal - theta)

        # 헤딩 오차가 크면 cos()로 자연스럽게 전진 속도를 줄인다 (0.2rad 문턱에서 회전/직진을
        # 딱 자르면 문턱 근처에서 진동하며 멈추는 문제가 있었음 - 부드러운 전환으로 해결).
        forward_scale = max(0.0, math.cos(heading_error))
        twist = Twist()
        twist.linear.x = max(0.0, min(MAX_LINEAR_SPEED, K_LINEAR * distance * forward_scale))
        twist.angular.z = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, K_ANGULAR * heading_error))
        self._cmd_vel_pub.publish(twist)

    def _on_arrived(self):
        if self._control_timer is not None:
            self._control_timer.cancel()
            self._control_timer = None
        self._cmd_vel_pub.publish(Twist())  # 정지
        self._moving = False
        station_id = self._goal.station_id if self._goal is not None else ''
        self._goal = None
        self._publish_status(AmrStatus.STATE_ARRIVED, station_id, '도착')

    def _publish_status(self, state: int, station_id: str, message: str):
        status = AmrStatus()
        status.state = state
        status.station_id = station_id
        status.message = message
        self._status_pub.publish(status)
        self.get_logger().info(f'PUB /amr/status -> {station_id} state={state}')


def main(args=None):
    rclpy.init(args=args)
    node = AmrNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

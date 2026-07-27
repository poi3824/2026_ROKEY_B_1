"""amr_node
실행 계층 · 목표 스테이션으로 이동, 도착·이동·오류 상태 보고.

SUB /amr/goal      (fms_interfaces/AmrGoal)     <- behavior_node
PUB /amr/status    (fms_interfaces/AmrStatus)   -> behavior_node

PUB /amr/cmd_vel   (geometry_msgs/Twist)        -> Isaac Sim (이동 명령)
SUB /amr/sim_pose  (geometry_msgs/Pose2D)       <- Isaac Sim (위치 상태)
"""
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import Twist

from fms_interfaces.msg import AmrGoal, AmrStatus

ARRIVAL_TOLERANCE_M = 0.05


class AmrNode(Node):

    def __init__(self):
        super().__init__('amr_node')

        self._goal_sub = self.create_subscription(AmrGoal, '/amr/goal', self._on_goal, 10)
        self._status_pub = self.create_publisher(AmrStatus, '/amr/status', 10)

        # Isaac Sim 연동 인터페이스
        self._cmd_vel_pub = self.create_publisher(Twist, '/amr/cmd_vel', 10)
        # TODO: Isaac Sim에서 실제 AMR 위치를 퍼블리시하면 이 콜백에서 도착 판정을 갱신.
        self._current_pose = (0.0, 0.0, 0.0)

        self._goal = None
        self._moving = False

        self.declare_parameter('sim_move_duration_sec', 3.0)

        self.get_logger().info('amr_node started')

    def _on_goal(self, msg: AmrGoal):
        self.get_logger().info(f'SUB /amr/goal <- {msg.station_id} ({msg.x:.2f}, {msg.y:.2f})')
        self._goal = msg
        self._moving = True
        self._publish_status(AmrStatus.STATE_MOVING, msg.station_id, '이동 중')

        # TODO: 실제 경로 계획/제어 대신, 지금은 목표 방향으로 cmd_vel을 흉내내어 발행하고
        # 일정 시간 뒤 Isaac Sim의 위치 피드백 없이 도착으로 간주한다.
        dx = msg.x - self._current_pose[0]
        dy = msg.y - self._current_pose[1]
        twist = Twist()
        twist.linear.x = math.copysign(0.3, dx) if abs(dx) > ARRIVAL_TOLERANCE_M else 0.0
        twist.linear.y = math.copysign(0.3, dy) if abs(dy) > ARRIVAL_TOLERANCE_M else 0.0
        self._cmd_vel_pub.publish(twist)

        duration = self.get_parameter('sim_move_duration_sec').value
        self._arrival_timer = self.create_timer(duration, self._on_arrived)

    def _on_arrived(self):
        self._arrival_timer.cancel()
        if self._goal is None:
            return
        self._current_pose = (self._goal.x, self._goal.y, self._goal.theta)
        self._cmd_vel_pub.publish(Twist())  # 정지
        self._moving = False
        self._publish_status(AmrStatus.STATE_ARRIVED, self._goal.station_id, '도착')

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

"""
FMS의 AMR 명령과 Isaac Sim AMR 브리지를 연결한다.

다른 ``src`` 노드는 커스텀 메시지인 ``/amr/goal``과 ``/amr/status``만 사용한다.
Isaac Sim 프로세스는 커스텀 인터페이스를 import하지 않고 표준 메시지 토픽만
사용한다. 이 노드가 두 영역 사이에서 메시지 변환, 상태 추적, 오류 처리를 맡는다.

SUB /amr/goal        (fms_interfaces/AmrGoal)     <- behavior/test node
PUB /amr/status      (fms_interfaces/AmrStatus)   -> behavior/test node
PUB /amr/goal_pose   (geometry_msgs/PoseStamped)  -> Isaac Sim
SUB /amr/sim_pose    (geometry_msgs/Pose2D)       <- Isaac Sim
SUB /amr/drive_state (std_msgs/String, optional)  <- physics bridge
PUB /amr/cancel      (std_msgs/Empty)              -> physics bridge

``/amr/sim_pose``는 실제 pose 피드백용이다. 도착 여부는 위치·yaw·정지 상태를
12틱 동안 함께 검사한 물리 브리지의 ``/amr/drive_state``만 신뢰한다.
"""
import json
import math
import time

import rclpy
from geometry_msgs.msg import Pose2D, PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Empty, String

from fms_interfaces.msg import AmrGoal, AmrStatus
from .timeout_policy import (
    DEFAULT_BRIDGE_TIMEOUT_SEC,
    bridge_timeout_expired,
)

DEFAULT_ARRIVAL_TOLERANCE_M = 0.05
GOAL_FRAME_ID = 'amr_baseline'
ACTIVE_DRIVE_STATES = {'ACTIVE', 'DRIVING'}
TERMINAL_DRIVE_STATES = {'ARRIVED', 'FAILED', 'CANCELED'}


def _stamp_key(msg: PoseStamped) -> str:
    """Return the bridge correlation key encoded in a goal header."""
    return f'{int(msg.header.stamp.sec)}:{int(msg.header.stamp.nanosec)}'


def _goal_error(station_id: str, x: float, y: float, theta: float):
    """Return a user-facing validation error, or ``None`` for a valid goal."""
    if not station_id.strip():
        return 'station_id가 비어 있습니다'
    if not all(math.isfinite(float(value)) for value in (x, y, theta)):
        return '목표 좌표에 NaN/inf가 포함돼 있습니다'
    return None


class AmrNode(Node):
    """Translate FMS goals and report the corresponding Isaac drive result."""

    def __init__(self):
        """Create the FMS-facing and Isaac-facing ROS interfaces."""
        super().__init__('amr_node')

        self.declare_parameter(
            'arrival_tolerance_m', DEFAULT_ARRIVAL_TOLERANCE_M)
        self.declare_parameter(
            'bridge_timeout_sec', DEFAULT_BRIDGE_TIMEOUT_SEC)
        self._arrival_tolerance = float(
            self.get_parameter('arrival_tolerance_m').value)
        self._bridge_timeout = float(
            self.get_parameter('bridge_timeout_sec').value)
        if self._arrival_tolerance <= 0.0:
            raise ValueError('arrival_tolerance_m must be greater than zero')
        if self._bridge_timeout < 0.0:
            raise ValueError('bridge_timeout_sec must be zero or greater')

        self._goal_sub = self.create_subscription(
            AmrGoal, '/amr/goal', self._on_goal, 10)
        self._status_pub = self.create_publisher(
            AmrStatus, '/amr/status', 10)
        self._goal_pose_pub = self.create_publisher(
            PoseStamped, '/amr/goal_pose', 10)
        self._cancel_pub = self.create_publisher(
            Empty, '/amr/cancel', 10)
        self._sim_pose_sub = self.create_subscription(
            Pose2D, '/amr/sim_pose', self._on_sim_pose, 10)
        self._drive_state_sub = self.create_subscription(
            String, '/amr/drive_state', self._on_drive_state, 10)

        self._current_pose = (0.0, 0.0, 0.0)
        self._goal_station_id = ''
        self._goal_xy = None
        self._goal_stamp = ''
        self._moving = False
        self._drive_state_seen = False
        self._last_bridge_activity = 0.0
        self._watchdog = self.create_timer(1.0, self._on_watchdog)

        self.get_logger().info(
            'amr_node started: FMS(/amr/goal, /amr/status) <-> '
            'Isaac(/amr/goal_pose, /amr/sim_pose)')

    def _on_goal(self, msg: AmrGoal):
        error = _goal_error(msg.station_id, msg.x, msg.y, msg.theta)
        if error is not None:
            self.get_logger().error(
                f'/amr/goal 거부 ({msg.station_id!r}): {error}')
            self._publish_status(
                AmrStatus.STATE_ERROR, msg.station_id, error)
            return

        station_id = msg.station_id.strip()
        self.get_logger().info(
            f'SUB /amr/goal <- {station_id} '
            f'({msg.x:.4f}, {msg.y:.4f}, {msg.theta:.4f})')

        if self._moving:
            old_station = self._goal_station_id
            self._moving = False
            self.get_logger().warn(
                f'새 목표 {station_id} 수신: 기존 목표 {old_station} 교체')

        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = GOAL_FRAME_ID
        goal_pose.pose.position.x = float(msg.x)
        goal_pose.pose.position.y = float(msg.y)
        goal_pose.pose.orientation.z = math.sin(float(msg.theta) * 0.5)
        goal_pose.pose.orientation.w = math.cos(float(msg.theta) * 0.5)

        self._goal_station_id = station_id
        self._goal_xy = (float(msg.x), float(msg.y))
        self._goal_stamp = _stamp_key(goal_pose)
        self._moving = True
        self._drive_state_seen = False
        self._last_bridge_activity = time.monotonic()

        self._publish_status(
            AmrStatus.STATE_MOVING,
            station_id,
            'Isaac Sim에 이동 목표 전송')
        self._goal_pose_pub.publish(goal_pose)
        self.get_logger().info(
            f'PUB /amr/goal_pose [{GOAL_FRAME_ID}] -> '
            f'({msg.x:.4f}, {msg.y:.4f}, {msg.theta:.4f}), '
            f'stamp={self._goal_stamp}')

    def _on_sim_pose(self, msg: Pose2D):
        self._current_pose = (msg.x, msg.y, msg.theta)

        if not self._moving or self._goal_xy is None:
            return
        self._last_bridge_activity = time.monotonic()

        # sim_pose만으로는 yaw/정지 12틱 조건을 증명할 수 없다. 특히 새 목표 직후
        # 이전 목표의 마지막 pose가 먼저 도착해도 조기 ARRIVED가 되지 않아야 한다.
        # 물리 브리지가 동일 goal_stamp로 발행하는 drive_state가 유일한 terminal
        # source다.

    def _on_drive_state(self, msg: String):
        try:
            payload = json.loads(msg.data)
            state = str(payload['state']).upper()
            stamp = str(payload.get('goal_stamp', ''))
            message = str(payload.get('message', ''))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().warn(
                f'잘못된 /amr/drive_state JSON 무시: {exc}')
            return

        if not self._moving or stamp != self._goal_stamp:
            return
        if state not in ACTIVE_DRIVE_STATES | TERMINAL_DRIVE_STATES:
            self.get_logger().warn(
                f'알 수 없는 /amr/drive_state 상태 무시: {state}')
            return

        self._drive_state_seen = True
        self._last_bridge_activity = time.monotonic()
        if state == 'ARRIVED':
            self._finish(
                AmrStatus.STATE_ARRIVED,
                message or 'Isaac Sim 주행 완료')
        elif state in {'FAILED', 'CANCELED'}:
            self._finish(
                AmrStatus.STATE_ERROR,
                message or f'Isaac Sim 주행 {state.lower()}')

    def _on_watchdog(self):
        now = time.monotonic()
        if not bridge_timeout_expired(
            moving=self._moving,
            timeout_sec=self._bridge_timeout,
            last_activity=self._last_bridge_activity,
            now=now,
        ):
            return

        silence = now - self._last_bridge_activity
        self._cancel_pub.publish(Empty())
        self._finish(
            AmrStatus.STATE_ERROR,
            f'Isaac Sim 브리지 응답이 {silence:.1f}초 동안 없습니다')

    def _finish(self, state: int, message: str):
        if not self._moving:
            return
        station_id = self._goal_station_id
        self._moving = False
        self._goal_xy = None
        self._goal_stamp = ''
        self._drive_state_seen = False
        self._publish_status(state, station_id, message)

    def cancel(self):
        """Cancel an active bridge goal during node shutdown."""
        if not self._moving:
            return
        self._moving = False
        self._cancel_pub.publish(Empty())

    def _publish_status(self, state: int, station_id: str, message: str):
        status = AmrStatus()
        status.state = state
        status.station_id = station_id
        status.message = message
        self._status_pub.publish(status)
        self.get_logger().info(
            f'PUB /amr/status -> {station_id} state={state}: {message}')


def main(args=None):
    """Run the AMR integration node."""
    rclpy.init(args=args)
    node = AmrNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if rclpy.ok():
                node.cancel()
                node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()

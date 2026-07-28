"""amr_test_sequencer
개발/테스트 전용 노드 - amr_node의 실제 이동 로직을 손으로 하나씩 pub 하지 않고,
정해진 스테이션 순서를 자동으로 돌며 검증하기 위한 임시 스크립트.

PUB /amr/goal   (fms_interfaces/AmrGoal)   -> amr_node
SUB /amr/status (fms_interfaces/AmrStatus) <- amr_node

한 목표를 보내고 STATE_ARRIVED가 올 때까지 기다린 다음 다음 목표로 넘어간다.
station_1/station_2는 behavior_node.STATION_POSES와 동일한 값을 그대로 가져와 쓴다.
"""
import sys
from pathlib import Path

import rclpy
from rclpy.node import Node

from fms_interfaces.msg import AmrGoal, AmrStatus

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src' / 'behavior_node' / 'behavior_node'))
from behavior_node import STATION_POSES, BATTERY_SLOT_POSES  # noqa: E402

ARRIVAL_TIMEOUT_SEC = 150.0

DEFAULT_ROUTE = ['home', 'station_1', 'station_2']

ROUTE_POSES = {
    'home': (-1.093, -0.203, 0.0),
    **STATION_POSES,
    **BATTERY_SLOT_POSES,
}


class AmrTestSequencer(Node):

    def __init__(self, route):
        super().__init__('amr_test_sequencer')
        self._route = route
        self._idx = -1
        self._arrived = False

        self._goal_pub = self.create_publisher(AmrGoal, '/amr/goal', 10)
        self._status_sub = self.create_subscription(AmrStatus, '/amr/status', self._on_status, 10)

        self._timer = self.create_timer(0.5, self._tick)
        self._wait_start = self.get_clock().now()

    def _on_status(self, msg: AmrStatus):
        if msg.state == AmrStatus.STATE_ARRIVED and msg.station_id == self._route[self._idx]:
            self.get_logger().info(f'도착 확인: {msg.station_id}')
            self._arrived = True
        elif msg.state == AmrStatus.STATE_ERROR:
            self.get_logger().error(f'에러 상태 수신: {msg.station_id} - {msg.message}')

    def _send_next_goal(self):
        self._idx += 1
        if self._idx >= len(self._route):
            self.get_logger().info('전체 경로 완료. 종료합니다.')
            rclpy.shutdown()
            return

        station_id = self._route[self._idx]
        x, y, theta = ROUTE_POSES[station_id]
        goal = AmrGoal(station_id=station_id, x=x, y=y, theta=theta)
        self._arrived = False
        self._wait_start = self.get_clock().now()
        self.get_logger().info(f'PUB /amr/goal -> {station_id} ({x:.3f}, {y:.3f})')
        self._goal_pub.publish(goal)

    def _tick(self):
        if self._idx == -1:
            self._send_next_goal()
            return

        if self._arrived:
            self._send_next_goal()
            return

        elapsed = (self.get_clock().now() - self._wait_start).nanoseconds / 1e9
        if elapsed > ARRIVAL_TIMEOUT_SEC:
            self.get_logger().error(
                f'{self._route[self._idx]} 도착 대기 타임아웃({ARRIVAL_TIMEOUT_SEC}s) - 중단합니다.'
            )
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    route = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_ROUTE
    unknown = [s for s in route if s not in ROUTE_POSES]
    if unknown:
        print(f'알 수 없는 station_id: {unknown} (사용 가능: {list(ROUTE_POSES.keys())})')
        return
    node = AmrTestSequencer(route)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()

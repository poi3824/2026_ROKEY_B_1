"""fleet_manager_node
FMS 계층 · 전체 작업을 보고 다음 job을 결정하는 관리자.

PUB /fleet/job      (fms_interfaces/FleetJob)
SUB /fleet/report   (fms_interfaces/FleetReport)
"""
import itertools

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from fms_interfaces.msg import FleetJob, FleetReport


class FleetManagerNode(Node):

    def __init__(self):
        super().__init__('fleet_manager_node')

        self._job_pub = self.create_publisher(FleetJob, '/fleet/job', 10)
        self._report_sub = self.create_subscription(
            FleetReport, '/fleet/report', self._on_report, 10)

        self._job_id_gen = itertools.count(1)
        self._pending_jobs = []
        self._job_in_progress = False

        self.declare_parameter('job_dispatch_period_sec', 5.0)
        period = self.get_parameter('job_dispatch_period_sec').value
        self._timer = self.create_timer(period, self._tick)

        self.get_logger().info('fleet_manager_node started')

    # --- 남은 작업 판단 --------------------------------------------------
    def _refresh_pending_jobs(self):
        """어느 단자(스테이션)가 비어 있는지 확인.

        TODO: 비전 스캔 결과(전체 스테이션 상태)를 받아 실제 빈 단자를 판단.
        지금은 데모용으로 station_1~3에 순차적으로 ASSEMBLE job을 채워 넣는다.
        """
        if not self._pending_jobs and not self._job_in_progress:
            for station_id in ('station_1', 'station_2', 'station_3'):
                self._pending_jobs.append({
                    'station_id': station_id,
                    'job_type': 'ASSEMBLE',
                    'target': 'busbar_and_nut',
                })

    # --- job 생성 · 작업 할당 ---------------------------------------------
    def _tick(self):
        if self._job_in_progress:
            return

        self._refresh_pending_jobs()
        if not self._pending_jobs:
            return

        job_spec = self._pending_jobs.pop(0)
        msg = FleetJob()
        msg.job_id = f'job_{next(self._job_id_gen)}'
        msg.station_id = job_spec['station_id']
        msg.job_type = job_spec['job_type']
        msg.target = job_spec['target']
        msg.stamp = self.get_clock().now().to_msg()

        self._job_in_progress = True
        self._job_pub.publish(msg)
        self.get_logger().info(
            f'PUB /fleet/job -> {msg.job_id} ({msg.station_id}, {msg.job_type})')

    def _on_report(self, msg: FleetReport):
        status = 'SUCCESS' if msg.success else 'FAILED'
        self.get_logger().info(
            f'SUB /fleet/report <- {msg.job_id} {status}: {msg.message}')
        self._job_in_progress = False


def main(args=None):
    rclpy.init(args=args)
    node = FleetManagerNode()
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

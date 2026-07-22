#!/usr/bin/env python3
"""
behavior_node.py (수정본)
- 타이머 취소(cancel) 처리 추가
"""

import rclpy
from rclpy.node import Node
from fms_interfaces.srv import ArmControl


class BehaviorNode(Node):

    def __init__(self):
        super().__init__('behavior_node')

        self._arm_client = self.create_client(ArmControl, '/arm/control')

        self._job_sequence = [
            'GRAB_BUSBAR',
            'INSERT_BUSBAR',
        ]
        self._current_step_idx = 0

        self.get_logger().info('behavior_node 시작')

        # 1회성 실행을 위한 타이머 저장
        self._timer = self.create_timer(1.0, self._start_sequence_once)

    def _start_sequence_once(self):
        # 타이머 중복 실행 방지를 위한 cancel
        self._timer.cancel()
        self._send_next_command()

    def _send_next_command(self):
        if self._current_step_idx >= len(self._job_sequence):
            self.get_logger().info('🎉 모든 버스바 및 너트 체결 공정이 성공적으로 완료되었습니다!')
            return

        cmd = self._job_sequence[self._current_step_idx]

        if not self._arm_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('/arm/control 서비스 대기 중...')
            # 재시도용 1초 뒤 호출
            self._timer = self.create_timer(1.0, self._start_sequence_once)
            return

        req = ArmControl.Request()
        req.command = cmd
        self.get_logger().info(f'[{self._current_step_idx + 1}/{len(self._job_sequence)}] 서비스 요청 전달 -> {cmd}')

        future = self._arm_client.call_async(req)
        future.add_done_callback(self._on_arm_response)

    def _on_arm_response(self, future):
        try:
            res = future.result()
            if res.success:
                self.get_logger().info(f'작업 성공 응답 수신: {res.message}')
                self._current_step_idx += 1
                self._send_next_command()
            else:
                self.get_logger().error(f'작업 실패 응답 수신: {res.message}. 시퀀스를 중단합니다.')
        except Exception as e:
            self.get_logger().error(f'서비스 통신 예외 발생: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
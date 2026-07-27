#!/usr/bin/env python3
"""ping_pong_demo.py -- amr_node <-> execute_isaac.py 노드 통신 파이프라인이
정해진 순서대로 반복 이동하는지 시연하는 테스트 노드.

behavior_node를 거치지 않고 /amr/goal(fms_interfaces/AmrGoal)을 정해진 스테이션
목록 순서대로 번갈아 직접 발행한다. amr_node가 /amr/sim_pose 피드백으로 도착
판정해서 /amr/status(ARRIVED)를 찍으면, 그걸 보고 다음 목표를 보낸다 (고정
타이머가 아니라 실제 도착 확인 기반 -- 닫힌 루프). 목록 끝까지 가면 처음으로
돌아가 계속 반복한다.

기본 순서(인자 없이 실행하면 이대로 동작, 3바퀴 반복):
  배터리3 -> 버스바3 -> 배터리2 -> 버스바2 -> 배터리1 -> 버스바1 -> (반복, 총 3바퀴)

각 지점 좌표는 2026-07-27 세션 중 GUI로 직접 눈 맞춰서 스크린샷으로 확인한
6개 지점을 그대로 등록한 것 (사용자가 스크린샷 순서 기준으로 지정). 배터리팩과
버스바가 같은 열이어도 X/Y가 서로 다른 별개 지점이다 -- col1/2/3(각 열의 대표
지점 하나)과는 다른, 더 세분화된 좌표 세트.

전제: amr_node와 execute_isaac.py(Isaac Sim 쪽 브릿지)가 이미 떠 있어야 한다.
물리 모드는 아직 조향이 완벽히 수렴하지 않을 때가 있어서, ARRIVAL_WATCHDOG_SEC
안에 도착 확인이 안 오면 무한대기하지 않고 경고를 찍고 다음 스테이션으로 넘어간다
(지금은 FORCE_ANIMATION_MODE=True라 execute_isaac.py가 애니메이션으로 동작해서
해당될 일은 거의 없음).

사용:
  python3 ping_pong_demo.py                                  # 기본 6지점 순서, 3바퀴
  python3 ping_pong_demo.py --stations battery3,busbar3 --cycles 5 --wait-sec 2
"""
import argparse

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from fms_interfaces.msg import AmrGoal, AmrStatus

# (x, y) -- BASELINE 기준 delta. execute_isaac.py가 --usd 경로 보고 파일별
# baseline_offset(Busbar_amr는 -0.3 X)을 알아서 더하니 여기선 그대로 두면 됨.
#
# 6개 지점 원본 실측값(스크린샷 순서): battery3=(-0.1911,0.0117), busbar3=(-0.1911,1.2392),
# battery2=(-0.1911,-0.5457), busbar2=(-0.6606,1.205), battery1=(-0.1259,-1.1478),
# busbar1=(-1.1298,1.2367). 배터리팩은 X가 거의 일정(Y로 열을 구분), 버스바는 Y가
# 거의 일정(X로 접근 깊이를 구분)하는 실측 패턴이라 -- 그룹별로 "고정축"을 하나로
# 통일한다: 배터리팩은 X_BATTERY 하나 + 원본 Y, 버스바는 Y_BUSBAR 하나 + 원본 X.
X_BATTERY = -0.1911    # battery3/2 실측값 (배터리팩 접근 라인)
Y_BUSBAR = (1.2392 + 1.205 + 1.2367) / 3  # busbar1/2/3 실측 Y 평균 (버스바 라인)

STATIONS = {
    "battery3": (X_BATTERY, 0.0117),
    "busbar3":  (-0.1911, Y_BUSBAR),
    "battery2": (X_BATTERY, -0.5457),
    "busbar2":  (-0.6606, Y_BUSBAR),
    "battery1": (X_BATTERY, -1.1478),
    "busbar1":  (-1.1298, Y_BUSBAR),
}

DEFAULT_STATIONS = ["battery3", "busbar3", "battery2", "busbar2", "battery1", "busbar1"]
DEFAULT_CYCLES = 3
ARRIVAL_WATCHDOG_SEC = 40.0  # execute_isaac.py 물리 모드 자체 타임아웃(30s)보다 넉넉하게


class PingPongDemo(Node):

    def __init__(self, stations: list, cycles: int, wait_sec: float):
        super().__init__('ping_pong_demo')
        self._goal_pub = self.create_publisher(AmrGoal, '/amr/goal', 10)
        self._status_sub = self.create_subscription(
            AmrStatus, '/amr/status', self._on_status, 10)

        self._targets = [(name, STATIONS[name]) for name in stations]
        self._idx = 0
        self._max_sends = cycles * len(stations) if cycles > 0 else -1  # -1 = 무한
        self._sent = 0
        self._waiting_station = None
        self._wait_sec = wait_sec
        self._watchdog_timer = None

        self.get_logger().info(
            f"ping_pong_demo 시작: {' -> '.join(stations)} -> (반복), "
            f"{'무한' if cycles <= 0 else f'{cycles}바퀴'}")
        self._send_next()

    def _send_next(self):
        if self._max_sends >= 0 and self._sent >= self._max_sends:
            self.get_logger().info("설정한 반복 횟수 완료 -- 종료")
            self.create_timer(0.5, self._shutdown_once)
            return

        name, (x, y) = self._targets[self._idx]
        msg = AmrGoal()
        msg.station_id = name
        msg.x = x
        msg.y = y
        msg.theta = 0.0
        self._waiting_station = name
        self._goal_pub.publish(msg)
        self._sent += 1
        self.get_logger().info(f"PUB /amr/goal -> {name} (x={x:.4f}, y={y:.4f}) [{self._sent}번째 이동]")

        self._arm_watchdog()

    def _arm_watchdog(self):
        if self._watchdog_timer is not None:
            self._watchdog_timer.cancel()
        self._watchdog_timer = self.create_timer(ARRIVAL_WATCHDOG_SEC, self._on_watchdog_timeout)

    def _disarm_watchdog(self):
        if self._watchdog_timer is not None:
            self._watchdog_timer.cancel()
            self._watchdog_timer = None

    def _on_watchdog_timeout(self):
        self.get_logger().warn(
            f"{ARRIVAL_WATCHDOG_SEC:.0f}s 안에 {self._waiting_station} 도착 확인이 안 옴 -- "
            f"execute_isaac.py/amr_node가 떠 있는지, 물리 모드면 조향이 수렴했는지 확인. "
            f"일단 다음 스테이션으로 넘어간다.")
        self._disarm_watchdog()
        self._advance()

    def _shutdown_once(self):
        rclpy.shutdown()

    def _on_status(self, msg: AmrStatus):
        if msg.state != AmrStatus.STATE_ARRIVED:
            return
        if self._waiting_station is None or msg.station_id != self._waiting_station:
            return

        self._disarm_watchdog()
        self.get_logger().info(
            f"도착 확인: {msg.station_id} -- {self._wait_sec:.1f}s 후 다음으로")
        self._waiting_station = None
        self._advance()

    def _advance(self):
        self._idx = (self._idx + 1) % len(self._targets)

        timer = None

        def _fire():
            timer.cancel()
            self._send_next()

        timer = self.create_timer(self._wait_sec, _fire)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stations", default=",".join(DEFAULT_STATIONS),
                         help=f"쉼표로 구분한 순서 목록 (기본: {','.join(DEFAULT_STATIONS)})")
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES,
                         help=f"전체 순서 반복 횟수 (기본 {DEFAULT_CYCLES}, 0 = 무한)")
    parser.add_argument("--wait-sec", type=float, default=2.0, help="도착 후 다음 이동까지 대기 시간")
    parsed = parser.parse_args(args)

    stations = parsed.stations.split(",")
    for name in stations:
        if name not in STATIONS:
            raise SystemExit(f"알 수 없는 스테이션: {name} (선택 가능: {sorted(STATIONS.keys())})")
    if len(stations) < 2:
        raise SystemExit("--stations는 최소 2개 이상이어야 한다")

    rclpy.init()
    node = PingPongDemo(stations, parsed.cycles, parsed.wait_sec)
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


if __name__ == "__main__":
    main()

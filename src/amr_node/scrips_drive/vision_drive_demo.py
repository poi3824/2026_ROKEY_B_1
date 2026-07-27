#!/usr/bin/env python3
"""비전 좌표 기반 AMR 주행 또는 AMR+Arm 전체 공정을 실행하는 데모 노드."""

import argparse
import json
import uuid

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from fms_interfaces.action import ExecuteArmTask
from fms_interfaces.msg import AmrStatus


SUPPORTED_TARGETS = ("battery", "busbar")
FULL_PROCESS_STEPS = (
    ("drive", "battery"),
    ("arm", "SCAN_BATTERY"),
    ("arm", "STOW_ARM"),
    ("drive", "busbar"),
    ("arm", "SCAN_BUSBAR"),
    ("arm", "PICK_BUSBAR"),
    ("drive", "battery"),
    ("arm", "MOVE_BATTERY_CENTER"),
    ("arm", "FINE_ALIGNMENT"),
    ("arm", "ASSEMBLE_BUSBAR"),
    ("arm", "SCAN_NUT1"),
    ("arm", "PICK_NUT1"),
    ("arm", "ASSEMBLE_NUT1"),
    ("arm", "SCAN_NUT2"),
    ("arm", "PICK_NUT2"),
    ("arm", "ASSEMBLE_NUT2"),
)


class VisionDriveDemo(Node):
    def __init__(
        self,
        targets,
        cycles,
        wait_sec,
        timeout_sec,
        full_process,
    ):
        super().__init__("vision_drive_demo")
        self._request_pub = self.create_publisher(
            String, "/amr_physics/vision_request", 10
        )
        self._status_sub = self.create_subscription(
            AmrStatus,
            "/amr_physics/status",
            self._on_status,
            10,
        )
        self._ack_sub = self.create_subscription(
            String,
            "/amr_physics/goal_ack",
            self._on_ack,
            10,
        )
        self._cancel_pub = self.create_publisher(
            String, "/amr_physics/cancel_request", 10
        )
        self._arm_client = ActionClient(
            self,
            ExecuteArmTask,
            "/execute_arm_task",
        )

        self._targets = tuple(targets)
        self._full_process = bool(full_process)
        self._process_step = 0
        self._waiting_arm_task = None
        self._arm_goal_handle = None
        self._index = 0
        self._max_requests = (
            cycles * len(self._targets) if cycles > 0 else -1
        )
        self._sent = 0
        self._wait_sec = wait_sec
        self._timeout_sec = timeout_sec
        self._waiting_target = None
        self._waiting_goal_id = None
        self._request_acked = False
        self._watchdog = None
        self._retry_timer = None
        self._finished = False
        self._startup_timer = self.create_timer(1.0, self._start_once)

        if self._full_process:
            self.get_logger().info(
                "비전 물리 주행 + Arm 전체 공정 준비: "
                "battery 주행/스캔 -> busbar 주행/스캔/파지 -> "
                "battery 복귀/조립 -> nut1/2 체결; "
                "AMR 요청에는 좌표가 포함되지 않습니다"
            )
        else:
            self.get_logger().info(
                "비전 전용 주행 시험 준비: "
                f"{' -> '.join(self._targets)}, "
                f"{'무한' if cycles <= 0 else f'{cycles}회'}; "
                "요청에는 좌표가 포함되지 않습니다"
            )

    def _start_once(self):
        self._startup_timer.cancel()
        if self._full_process:
            self._run_process_step()
        else:
            self._send_next()

    def _run_process_step(self):
        if self._finished:
            return
        if self._process_step >= len(FULL_PROCESS_STEPS):
            self._finish("전체 조립 공정을 완료했습니다")
            return

        step_kind, command = FULL_PROCESS_STEPS[self._process_step]
        self.get_logger().info(
            f"전체 공정 {self._process_step + 1}/"
            f"{len(FULL_PROCESS_STEPS)}: {step_kind} {command}"
        )
        if step_kind == "drive":
            self._send_drive_request(command)
        else:
            self._send_arm_task(command)

    def _send_next(self):
        if self._finished:
            return
        if self._request_pub.get_subscription_count() != 1:
            self.get_logger().warn(
                "vision_request subscriber가 정확히 1개가 될 때까지 대기"
            )
            self._schedule_retry()
            return
        if self._max_requests >= 0 and self._sent >= self._max_requests:
            self._finish("설정한 비전 주행 요청을 모두 완료했습니다")
            return

        target_kind = self._targets[self._index]
        self._send_drive_request(target_kind)
        self._sent += 1

    def _send_drive_request(self, target_kind):
        if self._request_pub.get_subscription_count() != 1:
            self.get_logger().warn(
                "vision_request subscriber가 정확히 1개가 될 때까지 대기"
            )
            self._schedule_retry(
                self._run_process_step if self._full_process
                else self._send_next
            )
            return
        goal_id = uuid.uuid4().hex
        request = String()
        request.data = json.dumps(
            {
                "version": 1,
                "request_id": goal_id,
                "target_kind": target_kind,
            },
            separators=(",", ":"),
        )
        self._waiting_target = target_kind
        self._waiting_goal_id = goal_id
        self._request_acked = False
        self._request_pub.publish(request)
        self.get_logger().info(
            f"VISION_REQUEST -> {target_kind}, id={goal_id}; "
            "perception 좌표 대기"
        )
        self._arm_watchdog()

    def _schedule_retry(self, callback=None):
        if self._retry_timer is not None:
            return
        callback = callback or self._send_next

        timer = None

        def retry():
            self._retry_timer = None
            timer.cancel()
            callback()

        timer = self.create_timer(0.5, retry)
        self._retry_timer = timer

    def _on_ack(self, msg: String):
        try:
            payload = json.loads(msg.data)
            target_kind = str(payload["station_id"])
            goal_id = str(payload["goal_id"])
            source = str(payload.get("source", ""))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return
        if (
            target_kind == self._waiting_target
            and goal_id == self._waiting_goal_id
            and source == "perception"
        ):
            self._request_acked = True

    def _on_status(self, msg: AmrStatus):
        if (
            self._waiting_target is None
            or msg.station_id != self._waiting_target
        ):
            return
        if msg.state == AmrStatus.STATE_ERROR:
            self._finish(
                f"{msg.station_id} 비전 주행 실패: {msg.message}",
                error=True,
            )
            return
        if msg.state == AmrStatus.STATE_MOVING:
            self.get_logger().info(
                f"{msg.station_id} 대기/주행 상태: {msg.message}",
                throttle_duration_sec=3.0,
            )
            return
        if msg.state != AmrStatus.STATE_ARRIVED:
            return

        self._disarm_watchdog()
        self.get_logger().info(
            f"도착 확인: {msg.station_id}; "
            f"{self._wait_sec:.1f}s 후 다음 비전 요청"
        )
        self._waiting_target = None
        self._waiting_goal_id = None
        self._request_acked = False
        if self._full_process:
            self._process_step += 1
            self._schedule_process_step()
            return
        self._index = (self._index + 1) % len(self._targets)
        self._schedule_next()

    def _send_arm_task(self, task_type):
        if not self._arm_client.server_is_ready():
            self.get_logger().warn(
                "/execute_arm_task Action server 대기 중"
            )
            self._schedule_retry(self._run_process_step)
            return

        goal = ExecuteArmTask.Goal()
        goal.task_type = task_type
        self._waiting_arm_task = task_type
        future = self._arm_client.send_goal_async(
            goal,
            feedback_callback=self._on_arm_feedback,
        )
        future.add_done_callback(self._on_arm_goal_response)

    def _on_arm_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._finish(
                f"{self._waiting_arm_task} Action 전송 실패: {exc}",
                error=True,
            )
            return
        if not goal_handle.accepted:
            self._finish(
                f"{self._waiting_arm_task} Action이 거부됐습니다",
                error=True,
            )
            return
        self._arm_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_arm_result)

    def _on_arm_feedback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f"{self._waiting_arm_task}: "
            f"{feedback.sub_phase} {feedback.progress_pct:.1f}%",
            throttle_duration_sec=2.0,
        )

    def _on_arm_result(self, future):
        task = self._waiting_arm_task
        self._waiting_arm_task = None
        self._arm_goal_handle = None
        try:
            result = future.result().result
        except Exception as exc:
            self._finish(f"{task} Action 결과 오류: {exc}", error=True)
            return
        if not result.success:
            self._finish(
                f"{task} 실패 [{result.error_code}]: {result.message}",
                error=True,
            )
            return
        self.get_logger().info(f"{task} 완료: {result.message}")
        self._process_step += 1
        self._schedule_process_step()

    def _schedule_process_step(self):
        timer = None

        def fire():
            timer.cancel()
            self._run_process_step()

        timer = self.create_timer(self._wait_sec, fire)

    def _schedule_next(self):
        timer = None

        def fire():
            timer.cancel()
            self._send_next()

        timer = self.create_timer(self._wait_sec, fire)

    def _arm_watchdog(self):
        self._disarm_watchdog()
        if self._timeout_sec is None:
            return
        self._watchdog = self.create_timer(
            self._timeout_sec,
            self._on_timeout,
        )

    def _disarm_watchdog(self):
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None

    def _on_timeout(self):
        target = self._waiting_target
        goal_id = self._waiting_goal_id
        self._disarm_watchdog()
        if self._request_acked and target and goal_id:
            cancel = String()
            cancel.data = json.dumps(
                {
                    "station_id": target,
                    "goal_id": goal_id,
                },
                separators=(",", ":"),
            )
            self._cancel_pub.publish(cancel)
        self._finish(
            f"{target} 요청이 {self._timeout_sec:.1f}s 안에 완료되지 않았습니다",
            error=True,
        )

    def _finish(self, message: str, error: bool = False):
        if self._finished:
            return
        self._finished = True
        self._disarm_watchdog()
        if error:
            self.get_logger().error(message)
        else:
            self.get_logger().info(message)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        default="battery,busbar",
        help="좌표가 아닌 검출 종류 순서: battery,busbar",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="전체 순서 반복 횟수, 0은 무한",
    )
    parser.add_argument(
        "--wait-sec",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=120.0,
    )
    parser.add_argument(
        "--debug-no-timeouts",
        action="store_true",
        help="디버깅 중 요청 완료 watchdog를 끄고 Ctrl+C까지 대기",
    )
    process_mode = parser.add_mutually_exclusive_group()
    process_mode.add_argument(
        "--full-process",
        dest="full_process",
        action="store_true",
        help=(
            "배터리/버스바 물리 주행과 로봇팔 스캔·파지·조립·너트 "
            "체결까지 전체 공정을 1회 실행"
        ),
    )
    process_mode.add_argument(
        "--drive-only",
        dest="full_process",
        action="store_false",
        help="로봇팔 공정 없이 --targets의 AMR 주행만 시험",
    )
    parser.set_defaults(full_process=True)
    parsed = parser.parse_args(args)

    targets = tuple(
        item.strip().lower()
        for item in parsed.targets.split(",")
        if item.strip()
    )
    if not targets:
        raise SystemExit("--targets가 비어 있습니다")
    unknown = [item for item in targets if item not in SUPPORTED_TARGETS]
    if unknown:
        raise SystemExit(
            f"지원하지 않는 target: {unknown}; "
            f"선택 가능: {SUPPORTED_TARGETS}"
        )
    if parsed.timeout_sec <= 0.0 or parsed.wait_sec < 0.0:
        raise SystemExit("timeout은 양수, wait는 0 이상이어야 합니다")

    rclpy.init()
    node = VisionDriveDemo(
        targets,
        parsed.cycles,
        parsed.wait_sec,
        None if parsed.debug_no_timeouts else parsed.timeout_sec,
        parsed.full_process,
    )
    try:
        while rclpy.ok() and not node._finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()

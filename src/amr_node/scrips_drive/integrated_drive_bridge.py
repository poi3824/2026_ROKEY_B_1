#!/usr/bin/env python3
"""팔 executor와 같은 SimulationApp에서 쓰는 물리 AMR ROS 브릿지."""

import json
import math
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String

from drive_controller import (
    ControllerConfig,
    Goal2D,
    GoToPoseController,
    PoseState,
    normalize_angle,
)


TOPIC_PREFIX = "/amr_physics"


@dataclass(frozen=True)
class IntegratedDriveConfig:
    physics_hz: float = 60.0
    wheel_radius: float = 0.14
    track_width: float = 0.4132
    wheel_sign: float = 1.0
    steer_sign: float = 1.0
    max_linear: float = 0.35
    max_angular: float = 0.75
    min_angular: float = 0.30
    max_wheel_radps: float = 10.0
    position_tolerance: float = 0.04
    yaw_tolerance_deg: float = 3.0
    goal_timeout_sec: float = 90.0
    stuck_timeout_sec: float = 8.0
    lease_timeout_sec: float = 4.0
    workspace_radius: float = 3.5
    debug_no_timeouts: bool = False
    busbar_workcell_yaw_deg: float = 90.0
    busbar_workcell_pivot_xyz: tuple = (0.0, 0.0, 0.0)
    busbar_approach_yaw_rad: float = math.pi


@dataclass
class ActiveGoal:
    goal_id: str
    x: float
    y: float
    yaw: float
    world_goal: Goal2D
    last_lease_wall: float


class IntegratedPhysicsDriveBridge(Node):
    """표준 ROS 메시지만 사용해 동일 stage의 wheel articulation을 제어한다."""

    def __init__(
        self,
        *,
        measure_pose: Callable[[], PoseState],
        apply_wheels: Callable[[float, float], None],
        config: IntegratedDriveConfig,
        on_arrived: Optional[Callable[[], None]] = None,
    ):
        super().__init__("amr_physics_bridge")
        self._measure_pose = measure_pose
        self._apply_wheels = apply_wheels
        self._config = config
        self._on_arrived_callback = on_arrived
        self._session_id = uuid.uuid4().hex
        self._active = None
        self._elapsed = 0.0
        self._best_metric = float("inf")
        self._last_progress = 0.0
        self._last_phase = ""
        self._last_state_wall = 0.0
        self._frame_count = 0

        start = self._measure_pose()
        self._origin_x = start.x
        self._origin_y = start.y
        self._origin_z = start.z
        self._origin_yaw = start.yaw

        self._controller = GoToPoseController(
            ControllerConfig(
                position_tolerance=config.position_tolerance,
                yaw_tolerance=math.radians(config.yaw_tolerance_deg),
                max_linear=config.max_linear,
                max_angular=config.max_angular,
                min_angular=config.min_angular,
                max_wheel_radps=config.max_wheel_radps,
                allow_reverse=True,
                wheel_radius=config.wheel_radius,
                track_width=config.track_width,
            )
        )

        ready_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._ready_pub = self.create_publisher(
            String, f"{TOPIC_PREFIX}/ready", ready_qos
        )
        self._world_pose_pub = self.create_publisher(
            PoseStamped, f"{TOPIC_PREFIX}/world_pose", 10
        )
        self._state_pub = self.create_publisher(
            String, f"{TOPIC_PREFIX}/drive_state", 10
        )
        self._command_sub = self.create_subscription(
            String, f"{TOPIC_PREFIX}/command", self._on_command, 10
        )
        self._lease_sub = self.create_subscription(
            String, f"{TOPIC_PREFIX}/lease", self._on_lease, 10
        )
        self._ready_timer = self.create_timer(1.0, self._publish_ready)
        self._publish_ready()
        self.get_logger().info(
            "전체 공정 executor 물리 AMR 브릿지 준비: "
            f"session={self._session_id}"
        )

    @property
    def driving(self):
        return self._active is not None

    def _publish_ready(self):
        msg = String()
        msg.data = json.dumps(
            {
                "version": 1,
                "state": "READY",
                "session_id": self._session_id,
                "command_topic": f"{TOPIC_PREFIX}/command",
                "lease_timeout_sec": self._config.lease_timeout_sec,
                "debug_no_timeouts": self._config.debug_no_timeouts,
                "drive_direction": "forward_or_explicit_reverse_docking",
                "workcell": {
                    "busbar_yaw_deg": (
                        self._config.busbar_workcell_yaw_deg
                    ),
                    "pivot_world": {
                        "x": self._config.busbar_workcell_pivot_xyz[0],
                        "y": self._config.busbar_workcell_pivot_xyz[1],
                        "z": self._config.busbar_workcell_pivot_xyz[2],
                    },
                    "busbar_approach_yaw": (
                        self._config.busbar_approach_yaw_rad
                    ),
                },
                "baseline": {
                    "frame_id": "amr_baseline",
                    "parent_frame_id": "world",
                    "origin_x": self._origin_x,
                    "origin_y": self._origin_y,
                    "origin_z": self._origin_z,
                    "origin_yaw": self._origin_yaw,
                },
            },
            separators=(",", ":"),
        )
        self._ready_pub.publish(msg)

    @staticmethod
    def _decode(msg):
        payload = json.loads(msg.data)
        if not isinstance(payload, dict):
            raise ValueError("JSON object가 필요합니다")
        return payload

    def _on_command(self, msg):
        goal_id = ""
        try:
            payload = self._decode(msg)
            command_type = str(payload["type"]).upper()
            session_id = str(payload["session_id"])
            goal_id = str(payload["goal_id"])
            if session_id != self._session_id:
                return
            if command_type == "CANCEL":
                self._cancel(goal_id, "취소 요청을 받았습니다")
                return
            if command_type != "GOAL":
                raise ValueError("type은 GOAL 또는 CANCEL이어야 합니다")
            if str(payload.get("source", "")).lower() != "perception":
                raise ValueError("source=perception 목표만 허용합니다")
            if str(payload["frame_id"]) != "world":
                raise ValueError("world frame 목표만 허용합니다")
            if payload.get("lease_required") is not True:
                raise ValueError("lease_required=true가 필요합니다")

            x = float(payload["x"])
            y = float(payload["y"])
            yaw = normalize_angle(float(payload["yaw"]))
            if not all(math.isfinite(value) for value in (x, y, yaw)):
                raise ValueError("목표에 NaN/inf가 있습니다")
            if (
                math.hypot(x - self._origin_x, y - self._origin_y)
                > self._config.workspace_radius
            ):
                raise ValueError("목표가 workspace 안전 반경 밖입니다")

            world_goal = Goal2D(
                x=x,
                y=y,
                yaw=yaw,
                yaw_required=bool(payload.get("yaw_required", True)),
                allow_reverse=bool(payload.get("allow_reverse", False)),
                align_yaw_before_drive=bool(
                    payload.get("align_yaw_before_drive", False)
                ),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self._publish_state(
                "REJECTED",
                str(exc),
                goal_id=goal_id,
            )
            return

        if self._active is not None:
            self._cancel(
                self._active.goal_id,
                "새 목표가 도착해 기존 목표를 중단했습니다",
            )
        self._active = ActiveGoal(
            goal_id=goal_id,
            x=x,
            y=y,
            yaw=yaw,
            world_goal=world_goal,
            last_lease_wall=time.monotonic(),
        )
        self._controller.set_goal(world_goal)
        self._elapsed = 0.0
        self._best_metric = float("inf")
        self._last_progress = 0.0
        self._last_phase = ""
        self._publish_state("ACTIVE", "목표를 수락했습니다")

    def _on_lease(self, msg):
        try:
            payload = self._decode(msg)
            session_id = str(payload["session_id"])
            goal_id = str(payload["goal_id"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return
        if (
            self._active is not None
            and session_id == self._session_id
            and goal_id == self._active.goal_id
        ):
            self._active.last_lease_wall = time.monotonic()

    def _publish_pose(self, pose):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = pose.x
        msg.pose.position.y = pose.y
        msg.pose.position.z = pose.z
        msg.pose.orientation.x = pose.qx
        msg.pose.orientation.y = pose.qy
        msg.pose.orientation.z = pose.qz
        msg.pose.orientation.w = pose.qw
        self._world_pose_pub.publish(msg)

    def _publish_state(
        self,
        state,
        message,
        *,
        goal=None,
        goal_id=None,
        pose=None,
        command=None,
    ):
        active = self._active if goal is None else goal
        payload = {
            "version": 1,
            "state": state,
            "message": message,
            "session_id": self._session_id,
            "goal_id": goal_id or (active.goal_id if active else ""),
            "active_goal_id": self._active.goal_id if self._active else "",
        }
        if pose is not None:
            payload["pose"] = {
                "world_x": pose.x,
                "world_y": pose.y,
                "world_yaw": pose.yaw,
            }
        if command is not None:
            payload["control"] = {
                "phase": command.phase,
                "linear": command.linear,
                "angular": command.angular,
                "distance_error": command.distance_error,
                "heading_error": command.heading_error,
                "yaw_error": command.yaw_error,
                "travel_direction": command.travel_direction,
            }
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self._state_pub.publish(msg)

    def stop(self):
        self._controller.stop()
        self._apply_wheels(0.0, 0.0)

    def _cancel(self, goal_id, reason):
        if self._active is None or self._active.goal_id != goal_id:
            return
        goal = self._active
        self.stop()
        self._publish_state("CANCELED", reason, goal=goal)
        self._active = None

    def _fail(self, reason, pose=None, command=None):
        goal = self._active
        self.stop()
        if goal is not None:
            self._publish_state(
                "FAILED",
                reason,
                goal=goal,
                pose=pose,
                command=command,
            )
        self._active = None
        self.get_logger().error(reason)

    @staticmethod
    def _progress_metric(command):
        if command.phase == "drive":
            return command.distance_error
        if command.phase in (
            "turn_to_path",
            "initial_align",
            "final_align",
        ):
            return abs(
                command.heading_error
                if command.phase == "turn_to_path"
                else command.yaw_error
            )
        return 0.0

    def control_step(self):
        pose = self._measure_pose()
        self._frame_count += 1
        if self._frame_count % max(
            1, int(round(self._config.physics_hz / 10.0))
        ) == 0:
            self._publish_pose(pose)
        if self._active is None:
            return

        now = time.monotonic()
        if (
            not self._config.debug_no_timeouts
            and now - self._active.last_lease_wall
            > self._config.lease_timeout_sec
        ):
            self._fail("relay lease가 끊겨 안전 정지", pose=pose)
            return
        if (
            math.hypot(pose.x - self._origin_x, pose.y - self._origin_y)
            > self._config.workspace_radius
        ):
            self._fail("AMR이 workspace 안전 반경 밖으로 이탈", pose=pose)
            return
        if pose.up_z < math.cos(math.radians(40.0)):
            self._fail("AMR 기울기 초과", pose=pose)
            return

        dt = 1.0 / self._config.physics_hz
        self._elapsed += dt
        if (
            not self._config.debug_no_timeouts
            and self._elapsed > self._config.goal_timeout_sec
        ):
            self._fail("AMR 목표 시간 초과", pose=pose)
            return

        command = self._controller.compute(pose, dt)
        if command.arrived:
            goal = self._active
            self.stop()
            self._publish_pose(pose)
            self._publish_state(
                "ARRIVED",
                "실제 chassis pose가 위치/yaw 허용오차에 도달했습니다",
                goal=goal,
                pose=pose,
                command=command,
            )
            self._active = None
            if self._on_arrived_callback is not None:
                self._on_arrived_callback()
            return

        left, right = self._controller.to_wheel_speeds(
            command.linear,
            command.angular,
            wheel_sign=self._config.wheel_sign,
            steer_sign=self._config.steer_sign,
        )
        self._apply_wheels(left, right)

        metric = self._progress_metric(command)
        if command.phase != self._last_phase:
            self._last_phase = command.phase
            self._best_metric = metric
            self._last_progress = self._elapsed
        else:
            threshold = (
                0.01
                if command.phase == "drive"
                else math.radians(2.0)
            )
            if metric < self._best_metric - threshold:
                self._best_metric = metric
                self._last_progress = self._elapsed
            elif (
                not self._config.debug_no_timeouts
                and command.phase != "settling"
                and self._elapsed - self._last_progress
                > self._config.stuck_timeout_sec
            ):
                self._fail(
                    f"{command.phase} 진전 없음",
                    pose=pose,
                    command=command,
                )
                return

        if now - self._last_state_wall >= 1.0:
            self._last_state_wall = now
            self._publish_state(
                "DRIVING",
                command.phase,
                pose=pose,
                command=command,
            )

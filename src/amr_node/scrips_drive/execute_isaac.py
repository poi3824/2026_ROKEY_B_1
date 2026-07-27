#!/usr/bin/env python3
"""Isaac Sim에서 Nova Carter를 실제 PhysX 바퀴로 목표 pose까지 주행시킨다.

물리 articulation과 좌우 wheel drive가 없으면 시작 단계에서 실패한다.

입력
  /amr_physics/command std_msgs/String
    - READY가 제공한 session_id와 UUID goal_id를 넣은 JSON GOAL/CANCEL
    - GOAL은 source=perception, frame_id=world만 허용
  /amr_physics/lease   std_msgs/String
    - active goal의 session_id/goal_id lease

출력
  /amr_physics/ready   std_msgs/String (transient-local)
  /amr_physics/baseline_pose geometry_msgs/PoseStamped
  /amr_physics/world_pose    geometry_msgs/PoseStamped
    - 실제 chassis_link의 USD 절대 world pose.
  /amr_physics/drive_state   std_msgs/String
    - JSON 상태(ACTIVE, DRIVING, ARRIVED, FAILED, CANCELED).

주행은 매 physics tick에서 실제 chassis_link pose를 다시 읽는 폐루프 방식이다.
목표 방향 회전 -> 직진/조향 -> 최종 yaw 정렬 순으로 진행하며, 새 goal은 기존 goal을
안전하게 preempt한다. 이 구현은 좌표 기반 point-to-point 물리 주행이며, lidar/map을
이용한 Nav2 장애물 회피는 포함하지 않는다.
"""

import argparse
import json
import math
import os
import sys
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _ensure_isaac_humble_runtime():
    """Isaac 5.1 내장 Humble의 shared library 경로를 프로세스 시작 전에 적용."""
    isaac_path = os.environ.get("ISAAC_PATH")
    if not isaac_path:
        return
    ros_lib = (
        Path(isaac_path)
        / "exts"
        / "isaacsim.ros2.bridge"
        / "humble"
        / "lib"
    )
    if not ros_lib.is_dir():
        return

    environment = os.environ.copy()
    changed = False
    if environment.get("ROS_DISTRO") != "humble":
        environment["ROS_DISTRO"] = "humble"
        changed = True
    if environment.get("RMW_IMPLEMENTATION") != "rmw_fastrtps_cpp":
        environment["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
        changed = True

    current_paths = [
        value
        for value in environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if value
    ]
    if str(ros_lib) not in current_paths:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            current_paths + [str(ros_lib)]
        )
        changed = True

    if changed and environment.get("AMR_ISAAC_ROS_ENV_READY") != "1":
        environment["AMR_ISAAC_ROS_ENV_READY"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)


_ensure_isaac_humble_runtime()


def _default_usd() -> str:
    src_dir = Path(__file__).resolve().parents[2]
    return str(src_dir / "Collected_Busbar_amr" / "Busbar.usd")


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--usd", default=_default_usd(), help="대상 USD 파일")
    parser.add_argument(
        "--headless",
        action="store_true",
        default=os.environ.get("AMR_BRIDGE_HEADLESS") == "1",
        help="창 없이 Isaac Sim 실행",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="씬/조인트/DOF 검사만 한 뒤 주행하지 않고 종료",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="짧은 전진/회전 물리 시험 후 초기 pose로 reset하고 종료",
    )
    parser.add_argument(
        "--calibrate-signs",
        action="store_true",
        help="짧은 전진/회전 시험으로 wheel/steer 부호를 측정하고 reset한 뒤 계속 실행",
    )
    parser.add_argument(
        "--busbar-workcell-yaw-deg",
        type=float,
        default=0.0,
        help=(
            "현재 USD 자세에서 추가로 적용할 session-layer 회전. "
            "Busbar.usd에는 이미 +90도가 영구 반영되어 기본값 0 사용"
        ),
    )
    parser.add_argument("--wheel-radius", type=float, default=0.14)
    parser.add_argument("--track-width", type=float, default=0.4132)
    parser.add_argument("--wheel-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--steer-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--max-linear", type=float, default=0.35)
    parser.add_argument("--max-angular", type=float, default=0.75)
    parser.add_argument(
        "--min-angular",
        type=float,
        default=0.30,
        help="최종 yaw 정렬에서 chassis 정지 마찰을 이기기 위한 최소 각속도",
    )
    parser.add_argument(
        "--drive-direction",
        choices=("auto", "forward"),
        default="auto",
        help=(
            "auto는 goal 시작 시 전진/후진 중 초기 회전이 작은 방향을 "
            "선택하고 goal이 끝날 때까지 유지"
        ),
    )
    parser.add_argument("--max-wheel-radps", type=float, default=10.0)
    parser.add_argument(
        "--max-wheel-force",
        type=float,
        default=200.0,
        help="각 구동 wheel DriveAPI의 유한 최대 force/torque",
    )
    parser.add_argument("--position-tolerance", type=float, default=0.04)
    parser.add_argument("--yaw-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--goal-timeout-sec", type=float, default=90.0)
    parser.add_argument("--stuck-timeout-sec", type=float, default=8.0)
    parser.add_argument(
        "--debug-no-timeouts",
        action="store_true",
        help=(
            "디버깅 중 goal/stuck/relay lease 시간 제한을 끕니다. "
            "workspace·기울기·높이 안전 검사는 유지됩니다"
        ),
    )
    parser.add_argument(
        "--lease-timeout-sec",
        type=float,
        default=4.0,
        help="active relay lease가 끊겼을 때 물리 AMR 정지까지의 wall-clock 시간",
    )
    parser.add_argument(
        "--workspace-radius",
        type=float,
        default=3.5,
        help="executor 시작 chassis pose에서 허용할 최대 XY 거리",
    )
    args = parser.parse_args()

    positive = (
        "wheel_radius",
        "track_width",
        "max_linear",
        "max_angular",
        "min_angular",
        "max_wheel_radps",
        "max_wheel_force",
        "position_tolerance",
        "goal_timeout_sec",
        "stuck_timeout_sec",
        "lease_timeout_sec",
        "workspace_radius",
    )
    for name in positive:
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            parser.error(
                f"--{name.replace('_', '-')} 값은 유한한 양수여야 합니다"
            )
    if (
        not math.isfinite(args.yaw_tolerance_deg)
        or args.yaw_tolerance_deg <= 0.0
        or args.yaw_tolerance_deg >= 180.0
    ):
        parser.error("--yaw-tolerance-deg 값은 0~180도 사이여야 합니다")
    if args.max_linear > 2.0 or args.max_angular > 6.0:
        parser.error("--max-linear/--max-angular가 안전 상한을 초과합니다")
    if args.min_angular > args.max_angular:
        parser.error("--min-angular는 --max-angular 이하여야 합니다")
    if args.max_wheel_radps > 50.0 or args.max_wheel_force > 5000.0:
        parser.error("--max-wheel-radps/--max-wheel-force가 안전 상한을 초과합니다")
    if args.workspace_radius > 20.0:
        parser.error("--workspace-radius가 안전 상한 20m를 초과합니다")
    if (
        not math.isfinite(args.busbar_workcell_yaw_deg)
        or abs(args.busbar_workcell_yaw_deg) > 360.0
    ):
        parser.error("--busbar-workcell-yaw-deg 값은 -360~360도여야 합니다")
    return args


ARGS = _parse_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": ARGS.headless})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from pxr import UsdGeom, UsdPhysics  # noqa: E402

enable_extension("isaacsim.ros2.bridge")
for _ in range(3):
    simulation_app.update()

import rclpy  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from drive_controller import (  # noqa: E402
    ControllerConfig,
    Goal2D,
    GoToPoseController,
    PoseState,
    normalize_angle,
)
from workcell_transform import rotate_busbar_workcell  # noqa: E402
PHYSICS_HZ = 60.0
TOPIC_PREFIX = "/amr_physics"
CHASSIS_PATH = "/World/Nova_Carter/chassis_link"
LEFT_WHEEL_JOINT = "/World/Nova_Carter/joint_wheel_left"
RIGHT_WHEEL_JOINT = "/World/Nova_Carter/joint_wheel_right"
LEFT_WHEEL_NAME = "joint_wheel_left"
RIGHT_WHEEL_NAME = "joint_wheel_right"

# Busbar.usd에는 팔 작업용 lock_amr_base() 결과가 저장돼 있다. 주행을 시작할 때
# Nova Carter 원본 nova_carter_physics.usd의 authored 값으로만 되돌린다.
FACTORY_DRIVE_SETTINGS = {
    "/World/Nova_Carter/joint_caster_base": {
        "stiffness": 100.0,
        "damping": 10.0,
        "max_force": float("inf"),
        "target_position": 0.0,
    },
    "/World/Nova_Carter/joint_swing_left": {
        "stiffness": 0.0,
        "damping": 1.0e-5,
        "max_force": float("inf"),
        "target_position": 0.0,
    },
    "/World/Nova_Carter/joint_swing_right": {
        "stiffness": 0.0,
        "damping": 1.0e-5,
        "max_force": float("inf"),
        "target_position": 0.0,
    },
    LEFT_WHEEL_JOINT: {
        "stiffness": 0.0,
        "damping": 1.0e6,
        "max_force": float("inf"),
        "target_position": 0.0,
    },
    RIGHT_WHEEL_JOINT: {
        "stiffness": 0.0,
        "damping": 1.0e6,
        "max_force": float("inf"),
        "target_position": 0.0,
    },
}


def _valid_prim(stage, path):
    prim = stage.GetPrimAtPath(path)
    return prim if prim and prim.IsValid() else None


def _measure_pose(articulation) -> PoseState:
    """PhysX articulation tensor에서 실제 root pose를 읽는다.

    headless physics step에서는 USD Xform writeback이 늦거나 비활성일 수 있으므로
    제어 피드백에 ComputeLocalToWorldTransform을 사용하지 않는다.
    """
    position, orientation = articulation.get_world_pose()
    linear_velocity = articulation.get_linear_velocity()
    angular_velocity = articulation.get_angular_velocity()
    values = [
        float(value)
        for value in (
            *position,
            *orientation,
            *linear_velocity,
            *angular_velocity,
        )
    ]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("articulation pose에 NaN/inf가 포함돼 있습니다")

    # Isaac Sim quaternion은 scalar-first (w, x, y, z).
    w, x, y, z = (float(value) for value in orientation)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1.0e-9:
        raise RuntimeError("articulation orientation quaternion이 유효하지 않습니다")
    w /= norm
    x /= norm
    y /= norm
    z /= norm
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return PoseState(
        x=float(position[0]),
        y=float(position[1]),
        yaw=yaw,
        z=float(position[2]),
        up_z=up_z,
        qx=x,
        qy=y,
        qz=z,
        qw=w,
        linear_speed=math.sqrt(
            sum(float(value) ** 2 for value in linear_velocity)
        ),
        angular_speed=math.sqrt(
            sum(float(value) ** 2 for value in angular_velocity)
        ),
    )


def _physics_scenes(stage):
    return [str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]


def configure_ros_domain(stage):
    """USD의 모든 ROS2Context를 executor의 ROS_DOMAIN_ID로 강제 일치시킨다."""
    raw_domain_id = os.environ.get("ROS_DOMAIN_ID", "0")
    try:
        domain_id = int(raw_domain_id)
    except ValueError as exc:
        raise RuntimeError(
            f"ROS_DOMAIN_ID가 정수가 아닙니다: {raw_domain_id!r}"
        ) from exc
    if not 0 <= domain_id <= 232:
        raise RuntimeError("ROS_DOMAIN_ID는 0~232 범위여야 합니다")

    context_paths = []
    for prim in stage.Traverse():
        node_type = prim.GetAttribute("node:type")
        if (
            not node_type.IsValid()
            or node_type.Get() != "isaacsim.ros2.bridge.ROS2Context"
        ):
            continue
        domain_attr = prim.GetAttribute("inputs:domain_id")
        env_attr = prim.GetAttribute("inputs:useDomainIDEnvVar")
        if not domain_attr.IsValid() or not env_attr.IsValid():
            raise RuntimeError(
                f"ROS2Context 입력 속성이 없습니다: {prim.GetPath()}"
            )
        domain_attr.Set(domain_id)
        env_attr.Set(True)
        context_paths.append(str(prim.GetPath()))

    if not context_paths:
        raise RuntimeError("USD에서 ROS2Context prim을 찾지 못했습니다")
    return domain_id, tuple(context_paths)


def validate_stage(stage):
    errors = []
    warnings = []

    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        errors.append("스테이지 up-axis가 Z가 아닙니다")
    meters = UsdGeom.GetStageMetersPerUnit(stage)
    if abs(float(meters) - 1.0) > 1.0e-6:
        errors.append(f"metersPerUnit={meters}; 1.0m 단위가 필요합니다")

    scenes = _physics_scenes(stage)
    if not scenes:
        errors.append("PhysicsScene이 없습니다")

    chassis = _valid_prim(stage, CHASSIS_PATH)
    if chassis is None:
        errors.append(f"chassis 없음: {CHASSIS_PATH}")
    elif not chassis.HasAPI(UsdPhysics.ArticulationRootAPI):
        errors.append(f"ArticulationRootAPI 없음: {CHASSIS_PATH}")
    elif not chassis.HasAPI(UsdPhysics.RigidBodyAPI):
        errors.append(f"RigidBodyAPI 없음: {CHASSIS_PATH}")

    wheel_positions = []
    for joint_path, wheel_path in (
        (LEFT_WHEEL_JOINT, "/World/Nova_Carter/wheel_left"),
        (RIGHT_WHEEL_JOINT, "/World/Nova_Carter/wheel_right"),
    ):
        joint = _valid_prim(stage, joint_path)
        if joint is None:
            errors.append(f"wheel joint 없음: {joint_path}")
            continue
        if not joint.IsA(UsdPhysics.RevoluteJoint):
            errors.append(f"RevoluteJoint가 아님: {joint_path}")
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if not drive:
            errors.append(f"angular DriveAPI 없음: {joint_path}")
        body0 = joint.GetRelationship("physics:body0").GetTargets()
        body1 = joint.GetRelationship("physics:body1").GetTargets()
        if not body0 or str(body0[0]) != CHASSIS_PATH:
            errors.append(f"{joint_path} body0가 chassis_link가 아닙니다")
        if not body1 or str(body1[0]) != wheel_path:
            errors.append(f"{joint_path} body1이 {wheel_path}가 아닙니다")
        pos_attr = joint.GetAttribute("physics:localPos0")
        if pos_attr:
            wheel_positions.append(float(pos_attr.Get()[1]))

    if len(wheel_positions) == 2:
        measured_track = abs(wheel_positions[0] - wheel_positions[1])
        if abs(measured_track - ARGS.track_width) > 0.005:
            warnings.append(
                f"USD track={measured_track:.4f}m, 설정={ARGS.track_width:.4f}m"
            )

    arm_joint = _valid_prim(stage, "/World/m0609/FixedJoint")
    if arm_joint is None or not arm_joint.IsA(UsdPhysics.FixedJoint):
        warnings.append("m0609 FixedJoint가 없어 물리 주행 때 팔이 AMR을 따라가지 않을 수 있습니다")

    if errors:
        raise RuntimeError("USD 물리 preflight 실패:\n  - " + "\n  - ".join(errors))
    return scenes, warnings


def restore_factory_amr_drives(stage):
    for path, values in FACTORY_DRIVE_SETTINGS.items():
        prim = _valid_prim(stage, path)
        if prim is None:
            raise RuntimeError(f"AMR drive joint 없음: {path}")
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            raise RuntimeError(f"angular DriveAPI 없음: {path}")
        drive.GetStiffnessAttr().Set(values["stiffness"])
        drive.GetDampingAttr().Set(values["damping"])
        max_force = values["max_force"]
        if path in (LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT):
            # 원본 asset은 inf지만 충돌/구속 시 무제한 토크를 허용하지 않는다.
            max_force = ARGS.max_wheel_force
        drive.GetMaxForceAttr().Set(max_force)
        drive.GetTargetPositionAttr().Set(values["target_position"])
        drive.GetTargetVelocityAttr().Set(0.0)


class BaselineFrame:
    """런타임 시작 chassis pose와 USD world 사이의 telemetry transform."""

    def __init__(self, _stage, articulation):
        current_pose = _measure_pose(articulation)
        self.origin_x = current_pose.x
        self.origin_y = current_pose.y
        self.origin_z = current_pose.z
        self.origin_yaw = current_pose.yaw

        cosine = math.cos(self.origin_yaw)
        sine = math.sin(self.origin_yaw)
        self._bx_x = cosine
        self._bx_y = sine
        self._by_x = -sine
        self._by_y = cosine

    def pose_from_world(self, pose: PoseState) -> PoseState:
        dx = pose.x - self.origin_x
        dy = pose.y - self.origin_y
        local_x = self._bx_x * dx + self._bx_y * dy
        local_y = self._by_x * dx + self._by_y * dy
        return PoseState(
            x=local_x,
            y=local_y,
            yaw=normalize_angle(pose.yaw - self.origin_yaw),
            z=pose.z - self.origin_z,
            up_z=pose.up_z,
            qx=pose.qx,
            qy=pose.qy,
            qz=pose.qz,
            qw=pose.qw,
            linear_speed=pose.linear_speed,
            angular_speed=pose.angular_speed,
        )


@dataclass
class ActiveGoal:
    goal_id: str
    frame_id: str
    requested_x: float
    requested_y: float
    requested_yaw: float
    yaw_required: bool
    allow_reverse: bool
    align_yaw_before_drive: bool
    world_goal: Goal2D
    lease_required: bool
    last_lease_wall: float


class AmrPhysicsBridge(Node):
    def __init__(
        self,
        articulation,
        wheel_indices,
        baseline_frame,
        wheel_sign,
        steer_sign,
        workcell_rotation,
    ):
        super().__init__("amr_physics_bridge")
        self._articulation = articulation
        self._wheel_indices = np.asarray(wheel_indices, dtype=np.int64)
        self._baseline = baseline_frame
        self._wheel_sign = float(wheel_sign)
        self._steer_sign = float(steer_sign)
        self._workcell_rotation = workcell_rotation
        self._session_id = uuid.uuid4().hex

        config = ControllerConfig(
            position_tolerance=ARGS.position_tolerance,
            yaw_tolerance=math.radians(ARGS.yaw_tolerance_deg),
            max_linear=ARGS.max_linear,
            max_angular=ARGS.max_angular,
            min_angular=ARGS.min_angular,
            max_wheel_radps=ARGS.max_wheel_radps,
            allow_reverse=ARGS.drive_direction == "auto",
            wheel_radius=ARGS.wheel_radius,
            track_width=ARGS.track_width,
        )
        self._controller = GoToPoseController(config)

        ready_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._ready_pub = self.create_publisher(
            String, f"{TOPIC_PREFIX}/ready", ready_qos
        )
        self._baseline_pose_pub = self.create_publisher(
            PoseStamped, f"{TOPIC_PREFIX}/baseline_pose", 10
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

        self._active: Optional[ActiveGoal] = None
        self._terminal_results = OrderedDict()
        self._goal_elapsed = 0.0
        self._last_progress_elapsed = 0.0
        self._best_progress_metric = float("inf")
        self._progress_phase = ""
        self._last_heartbeat_wall = time.monotonic()
        self._last_log_elapsed = -1.0
        self._frame_count = 0

        self._publish_ready()
        self.get_logger().info(
            "물리 AMR 브릿지 준비 완료: "
            f"session={self._session_id}, "
            f"wheel_sign={self._wheel_sign:+.0f}, "
            f"steer_sign={self._steer_sign:+.0f}, "
            f"busbar_workcell_yaw="
            f"{self._workcell_rotation.yaw_deg:+.1f}deg, "
            f"origin=({baseline_frame.origin_x:.4f}, "
            f"{baseline_frame.origin_y:.4f}, "
            f"{math.degrees(baseline_frame.origin_yaw):.1f}deg)"
        )
        if ARGS.debug_no_timeouts:
            self.get_logger().warn(
                "DEBUG 무제한 모드: goal/stuck/relay lease 시간 제한을 "
                "적용하지 않습니다"
            )

    def _publish_ready(self):
        msg = String()
        msg.data = json.dumps(
            {
                "version": 1,
                "state": "READY",
                "session_id": self._session_id,
                "command_topic": f"{TOPIC_PREFIX}/command",
                "lease_timeout_sec": ARGS.lease_timeout_sec,
                "debug_no_timeouts": ARGS.debug_no_timeouts,
                "drive_direction": ARGS.drive_direction,
                "workcell": {
                    "busbar_yaw_deg": self._workcell_rotation.yaw_deg,
                    "pivot_world": {
                        "x": self._workcell_rotation.pivot_world[0],
                        "y": self._workcell_rotation.pivot_world[1],
                        "z": self._workcell_rotation.pivot_world[2],
                    },
                    "busbar_approach_yaw": (
                        self._workcell_rotation.busbar_approach_yaw_rad
                    ),
                },
                "baseline": {
                    "frame_id": "amr_baseline",
                    "parent_frame_id": "world",
                    "origin_x": self._baseline.origin_x,
                    "origin_y": self._baseline.origin_y,
                    "origin_z": self._baseline.origin_z,
                    "origin_yaw": self._baseline.origin_yaw,
                },
            },
            separators=(",", ":"),
        )
        self._ready_pub.publish(msg)

    def _apply_wheels(self, left_radps: float, right_radps: float):
        action = ArticulationAction(
            joint_velocities=np.asarray([left_radps, right_radps], dtype=np.float32),
            joint_indices=self._wheel_indices,
        )
        self._articulation.apply_action(action)

    def stop(self):
        self._controller.stop()
        self._apply_wheels(0.0, 0.0)

    def _state_payload(
        self,
        state,
        message,
        pose=None,
        command=None,
        active=None,
        goal_id=None,
    ):
        goal = active if active is not None else self._active
        payload = {
            "version": 1,
            "state": state,
            "message": message,
            "session_id": self._session_id,
            "goal_id": (
                goal_id
                if goal_id is not None
                else (goal.goal_id if goal else "")
            ),
            "active_goal_id": (
                self._active.goal_id if self._active is not None else ""
            ),
        }
        if goal:
            payload["goal"] = {
                "frame_id": goal.frame_id,
                "x": goal.requested_x,
                "y": goal.requested_y,
                "yaw": goal.requested_yaw,
                "world_x": goal.world_goal.x,
                "world_y": goal.world_goal.y,
                "world_yaw": goal.world_goal.yaw,
                "align_yaw_before_drive": goal.align_yaw_before_drive,
            }
        if pose is not None:
            baseline_pose = self._baseline.pose_from_world(pose)
            payload["pose"] = {
                "x": baseline_pose.x,
                "y": baseline_pose.y,
                "yaw": baseline_pose.yaw,
                "world_x": pose.x,
                "world_y": pose.y,
                "world_yaw": pose.yaw,
                "linear_speed": pose.linear_speed,
                "angular_speed": pose.angular_speed,
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
        return payload

    def _publish_state(self, state, message, **kwargs):
        msg = String()
        msg.data = json.dumps(
            self._state_payload(state, message, **kwargs),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self._state_pub.publish(msg)

    @staticmethod
    def _goal_signature(goal: ActiveGoal):
        return (
            goal.frame_id,
            goal.requested_x,
            goal.requested_y,
            goal.requested_yaw,
            goal.yaw_required,
            goal.allow_reverse,
            goal.align_yaw_before_drive,
            goal.lease_required,
        )

    def _publish_terminal(
        self,
        state,
        message,
        active,
        pose=None,
        command=None,
    ):
        self._terminal_results[active.goal_id] = (active, state, message)
        self._terminal_results.move_to_end(active.goal_id)
        while len(self._terminal_results) > 64:
            self._terminal_results.popitem(last=False)
        self._publish_state(
            state,
            message,
            pose=pose,
            command=command,
            active=active,
        )

    @staticmethod
    def _decode_json(msg: String):
        payload = json.loads(msg.data)
        if not isinstance(payload, dict):
            raise ValueError("JSON object가 필요합니다")
        return payload

    def _on_command(self, msg: String):
        goal_id = ""
        try:
            payload = self._decode_json(msg)
            command_type = str(payload["type"]).strip().upper()
            session_id = str(payload["session_id"]).strip()
            goal_id = str(payload["goal_id"]).strip()
            if session_id != self._session_id:
                self.get_logger().warn(
                    "다른/만료된 bridge session command를 무시합니다"
                )
                return
            if not goal_id or len(goal_id) > 128:
                raise ValueError("goal_id는 1~128자여야 합니다")

            if command_type == "CANCEL":
                self._cancel_goal(goal_id)
                return
            if command_type != "GOAL":
                raise ValueError("command type은 GOAL 또는 CANCEL이어야 합니다")

            frame_id = str(payload["frame_id"])
            x = float(payload["x"])
            y = float(payload["y"])
            yaw = float(payload["yaw"])
            if not all(math.isfinite(value) for value in (x, y, yaw)):
                raise ValueError("goal에 NaN/inf가 포함돼 있습니다")
            yaw = normalize_angle(yaw)
            yaw_required = payload.get("yaw_required", True)
            if not isinstance(yaw_required, bool):
                raise ValueError("yaw_required는 bool이어야 합니다")
            allow_reverse = payload.get("allow_reverse", False)
            if not isinstance(allow_reverse, bool):
                raise ValueError("allow_reverse는 bool이어야 합니다")
            align_yaw_before_drive = payload.get(
                "align_yaw_before_drive",
                False,
            )
            if not isinstance(align_yaw_before_drive, bool):
                raise ValueError("align_yaw_before_drive는 bool이어야 합니다")
            lease_required = payload.get("lease_required", True)
            if lease_required is not True:
                raise ValueError("안전을 위해 lease_required=true가 필요합니다")
            source = str(payload.get("source", "")).strip().lower()
            if source != "perception":
                raise ValueError("source=perception인 목표만 허용합니다")
            if frame_id != "world":
                raise ValueError(
                    "perception의 world frame 목표만 허용합니다"
                )
            world_goal = Goal2D(
                x,
                y,
                yaw,
                yaw_required=yaw_required,
                allow_reverse=allow_reverse,
                align_yaw_before_drive=align_yaw_before_drive,
            )
            goal_radius = math.hypot(
                world_goal.x - self._baseline.origin_x,
                world_goal.y - self._baseline.origin_y,
            )
            if goal_radius > ARGS.workspace_radius:
                raise ValueError(
                    f"goal이 workspace 반경 {ARGS.workspace_radius:.2f}m 밖입니다"
                )
            candidate = ActiveGoal(
                goal_id=goal_id,
                frame_id=frame_id,
                requested_x=x,
                requested_y=y,
                requested_yaw=yaw,
                yaw_required=yaw_required,
                allow_reverse=allow_reverse,
                align_yaw_before_drive=align_yaw_before_drive,
                world_goal=world_goal,
                lease_required=lease_required,
                last_lease_wall=time.monotonic(),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().error(f"command 거부: {exc}")
            self._publish_state("REJECTED", str(exc), goal_id=goal_id)
            return

        terminal = self._terminal_results.get(goal_id)
        if terminal is not None:
            previous, state, message = terminal
            if self._goal_signature(previous) != self._goal_signature(candidate):
                self._publish_state(
                    "REJECTED",
                    "이미 종료된 goal_id를 다른 payload로 재사용할 수 없습니다",
                    goal_id=goal_id,
                )
                return
            self._publish_state(
                state,
                f"terminal 결과 재전송: {message}",
                active=previous,
            )
            return

        if self._active is not None and self._active.goal_id == goal_id:
            if self._goal_signature(self._active) != self._goal_signature(candidate):
                self._publish_state(
                    "REJECTED",
                    "active goal_id의 payload가 기존 command와 다릅니다",
                    goal_id=goal_id,
                )
                return
            self._active.last_lease_wall = time.monotonic()
            self._publish_state("ACTIVE", "중복 GOAL을 idempotent하게 확인했습니다")
            return

        # 새 command를 완전히 검증한 뒤에만 기존 물리 목표를 preempt한다.
        if self._active is not None:
            old = self._active
            self.stop()
            self._publish_terminal(
                "CANCELED",
                "새 목표가 도착해 기존 목표를 중단했습니다",
                old,
            )

        self._active = candidate
        self._controller.set_goal(world_goal)
        self._goal_elapsed = 0.0
        self._last_progress_elapsed = 0.0
        self._best_progress_metric = float("inf")
        self._progress_phase = ""
        self._last_heartbeat_wall = time.monotonic()
        self._last_log_elapsed = -1.0
        self.get_logger().info(
            f"goal 수락 id={goal_id} [{frame_id}] ({x:.4f}, {y:.4f}, "
            f"{math.degrees(yaw):.1f}deg) -> world "
            f"({world_goal.x:.4f}, {world_goal.y:.4f}, "
            f"{math.degrees(world_goal.yaw):.1f}deg)"
        )
        self._publish_state("ACTIVE", "목표를 수락했습니다")

    def _cancel_goal(self, goal_id: str):
        if self._active is None or self._active.goal_id != goal_id:
            self.get_logger().warn(
                f"현재 goal과 일치하지 않는 CANCEL 무시: {goal_id}"
            )
            return
        active = self._active
        self.stop()
        self._publish_terminal(
            "CANCELED",
            "취소 요청을 받았습니다",
            active,
        )
        self.get_logger().warn("현재 AMR 목표를 취소했습니다")
        self._active = None

    def _on_lease(self, msg: String):
        try:
            payload = self._decode_json(msg)
            session_id = str(payload["session_id"]).strip()
            goal_id = str(payload["goal_id"]).strip()
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().warn(f"잘못된 lease 무시: {exc}")
            return
        if (
            self._active is not None
            and session_id == self._session_id
            and goal_id == self._active.goal_id
        ):
            self._active.last_lease_wall = time.monotonic()

    @staticmethod
    def _quaternion_multiply(first, second):
        x1, y1, z1, w1 = first
        x2, y2, z2, w2 = second
        return (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )

    def _publish_pose(self, pose):
        baseline_pose = self._baseline.pose_from_world(pose)

        stamp = self.get_clock().now().to_msg()
        baseline_msg = PoseStamped()
        baseline_msg.header.stamp = stamp
        baseline_msg.header.frame_id = "amr_baseline"
        baseline_msg.pose.position.x = baseline_pose.x
        baseline_msg.pose.position.y = baseline_pose.y
        baseline_msg.pose.position.z = baseline_pose.z
        half = -0.5 * self._baseline.origin_yaw
        relative_q = self._quaternion_multiply(
            (0.0, 0.0, math.sin(half), math.cos(half)),
            (pose.qx, pose.qy, pose.qz, pose.qw),
        )
        (
            baseline_msg.pose.orientation.x,
            baseline_msg.pose.orientation.y,
            baseline_msg.pose.orientation.z,
            baseline_msg.pose.orientation.w,
        ) = relative_q
        self._baseline_pose_pub.publish(baseline_msg)

        world_msg = PoseStamped()
        world_msg.header.stamp = stamp
        world_msg.header.frame_id = "world"
        world_msg.pose.position.x = pose.x
        world_msg.pose.position.y = pose.y
        world_msg.pose.position.z = pose.z
        world_msg.pose.orientation.x = pose.qx
        world_msg.pose.orientation.y = pose.qy
        world_msg.pose.orientation.z = pose.qz
        world_msg.pose.orientation.w = pose.qw
        self._world_pose_pub.publish(world_msg)

    def _fail(self, message, pose=None, command=None):
        active = self._active
        self.stop()
        self.get_logger().error(message)
        if active is not None:
            self._publish_terminal(
                "FAILED",
                message,
                active,
                pose=pose,
                command=command,
            )
        self._active = None

    @staticmethod
    def _progress_metric(command):
        if command.phase == "turn_to_path":
            return abs(command.heading_error)
        if command.phase == "drive":
            return command.distance_error
        if command.phase in ("initial_align", "final_align"):
            return abs(command.yaw_error)
        return 0.0

    def control_step(self, dt: float):
        try:
            pose = _measure_pose(self._articulation)
        except Exception as exc:
            if self._active is not None:
                self._fail(f"articulation pose 측정 실패: {exc}")
            raise
        self._frame_count += 1
        if self._frame_count % 6 == 0:
            self._publish_pose(pose)

        if self._active is None:
            return

        baseline_pose = self._baseline.pose_from_world(pose)
        if math.hypot(baseline_pose.x, baseline_pose.y) > ARGS.workspace_radius:
            self._fail(
                f"AMR이 workspace 반경 {ARGS.workspace_radius:.2f}m 밖으로 이탈",
                pose=pose,
            )
            return

        if (
            not ARGS.debug_no_timeouts
            and self._active.lease_required
            and time.monotonic() - self._active.last_lease_wall
            > ARGS.lease_timeout_sec
        ):
            self._fail(
                f"relay lease가 {ARGS.lease_timeout_sec:.1f}s 동안 없어 안전 정지",
                pose=pose,
            )
            return

        self._goal_elapsed += dt
        if pose.up_z < math.cos(math.radians(40.0)):
            self._fail(
                f"AMR 기울기 초과(up_z={pose.up_z:.3f}); 전복/충돌 가능성",
                pose=pose,
            )
            return
        if abs(pose.z - self._baseline.origin_z) > 0.30:
            self._fail(
                f"AMR 높이 이탈(z={pose.z:.3f}m); 전복/충돌 가능성",
                pose=pose,
            )
            return
        if (
            not ARGS.debug_no_timeouts
            and self._goal_elapsed > ARGS.goal_timeout_sec
        ):
            self._fail(
                f"목표 시간 초과({ARGS.goal_timeout_sec:.1f}s)",
                pose=pose,
            )
            return

        command = self._controller.compute(pose, dt)
        if command.arrived:
            active = self._active
            self.stop()
            self._publish_pose(pose)
            self.get_logger().info(
                f"도착: position_error={command.distance_error:.3f}m, "
                f"yaw_error={math.degrees(command.yaw_error):.2f}deg"
            )
            self._publish_terminal(
                "ARRIVED",
                "실제 chassis pose가 위치/yaw 허용오차에 도달했습니다",
                active,
                pose=pose,
                command=command,
            )
            self._active = None
            return

        left, right = self._controller.to_wheel_speeds(
            command.linear,
            command.angular,
            wheel_sign=self._wheel_sign,
            steer_sign=self._steer_sign,
        )
        try:
            self._apply_wheels(left, right)
        except Exception as exc:
            self._fail(f"wheel action 적용 실패: {exc}", pose=pose, command=command)
            return

        if command.phase != self._progress_phase:
            self._progress_phase = command.phase
            self._best_progress_metric = self._progress_metric(command)
            self._last_progress_elapsed = self._goal_elapsed
        elif command.phase != "settling":
            metric = self._progress_metric(command)
            threshold = 0.01 if command.phase == "drive" else math.radians(2.0)
            if metric < self._best_progress_metric - threshold:
                self._best_progress_metric = metric
                self._last_progress_elapsed = self._goal_elapsed
            elif (
                not ARGS.debug_no_timeouts
                and self._goal_elapsed - self._last_progress_elapsed
                > ARGS.stuck_timeout_sec
            ):
                self._fail(
                    f"{ARGS.stuck_timeout_sec:.1f}s 동안 {command.phase} 진전 없음; "
                    "바퀴 잠금, 충돌 또는 마찰 설정을 확인하세요",
                    pose=pose,
                    command=command,
                )
                return

        now_wall = time.monotonic()
        if now_wall - self._last_heartbeat_wall >= 1.0:
            self._last_heartbeat_wall = now_wall
            self._publish_state(
                "DRIVING",
                command.phase,
                pose=pose,
                command=command,
            )

        if int(self._goal_elapsed) > int(self._last_log_elapsed):
            self._last_log_elapsed = self._goal_elapsed
            self.get_logger().info(
                f"[{command.phase}] dist={command.distance_error:.3f}m "
                f"dir={command.travel_direction} "
                f"heading_err={math.degrees(command.heading_error):+.1f}deg "
                f"yaw_err={math.degrees(command.yaw_error):+.1f}deg "
                f"v={command.linear:.3f}m/s w={command.angular:+.3f}rad/s "
                f"wheel=({left:+.2f}, {right:+.2f})rad/s"
            )


def _raw_wheel_action(articulation, wheel_indices, left, right):
    articulation.apply_action(
        ArticulationAction(
            joint_velocities=np.asarray([left, right], dtype=np.float32),
            joint_indices=np.asarray(wheel_indices, dtype=np.int64),
        )
    )


def calibrate_drive_signs(world, stage, articulation, wheel_indices):
    """짧게 움직여 부호를 실측하고 매 probe 뒤 hard reset으로 초기 pose를 복원."""
    probe_radps = 0.8
    probe_steps = 18
    settle_steps = 4

    start = _measure_pose(articulation)
    _raw_wheel_action(
        articulation, wheel_indices, probe_radps, probe_radps
    )
    for _ in range(probe_steps):
        world.step(render=not ARGS.headless)
    _raw_wheel_action(articulation, wheel_indices, 0.0, 0.0)
    for _ in range(settle_steps):
        world.step(render=not ARGS.headless)
    end = _measure_pose(articulation)

    forward_x = math.cos(start.yaw)
    forward_y = math.sin(start.yaw)
    projection = (end.x - start.x) * forward_x + (end.y - start.y) * forward_y
    if abs(projection) < 0.003:
        raise RuntimeError(
            f"전진 probe 변위가 {projection:.4f}m로 너무 작습니다. "
            "wheel drive/ground collision을 확인하세요"
        )
    wheel_sign = 1.0 if projection > 0.0 else -1.0
    print(
        f"[smoke] raw wheel +{probe_radps:.2f}rad/s -> "
        f"forward projection={projection:+.4f}m, wheel_sign={wheel_sign:+.0f}",
        flush=True,
    )

    _raw_wheel_action(articulation, wheel_indices, 0.0, 0.0)
    world.reset()
    restore_factory_amr_drives(stage)
    world.play()

    start = _measure_pose(articulation)
    _raw_wheel_action(
        articulation,
        wheel_indices,
        -wheel_sign * probe_radps,
        -wheel_sign * probe_radps,
    )
    for _ in range(probe_steps):
        world.step(render=not ARGS.headless)
    _raw_wheel_action(articulation, wheel_indices, 0.0, 0.0)
    for _ in range(settle_steps):
        world.step(render=not ARGS.headless)
    end = _measure_pose(articulation)

    forward_x = math.cos(start.yaw)
    forward_y = math.sin(start.yaw)
    reverse_projection = (
        (end.x - start.x) * forward_x
        + (end.y - start.y) * forward_y
    )
    if reverse_projection > -0.003:
        raise RuntimeError(
            f"후진 probe 변위가 {reverse_projection:.4f}m입니다. "
            "음의 wheel target/ground collision을 확인하세요"
        )
    print(
        f"[smoke] signed wheel reverse probe -> "
        f"forward projection={reverse_projection:+.4f}m",
        flush=True,
    )

    _raw_wheel_action(articulation, wheel_indices, 0.0, 0.0)
    world.reset()
    restore_factory_amr_drives(stage)
    world.play()

    start = _measure_pose(articulation)
    _raw_wheel_action(
        articulation,
        wheel_indices,
        -wheel_sign * probe_radps,
        wheel_sign * probe_radps,
    )
    for _ in range(probe_steps):
        world.step(render=not ARGS.headless)
    _raw_wheel_action(articulation, wheel_indices, 0.0, 0.0)
    for _ in range(settle_steps):
        world.step(render=not ARGS.headless)
    end = _measure_pose(articulation)

    yaw_delta = normalize_angle(end.yaw - start.yaw)
    if abs(yaw_delta) < math.radians(1.0):
        raise RuntimeError(
            f"회전 probe 변화가 {math.degrees(yaw_delta):.2f}deg로 너무 작습니다. "
            "caster/마찰 설정을 확인하세요"
        )
    steer_sign = 1.0 if yaw_delta > 0.0 else -1.0
    print(
        f"[smoke] physical L-/R+ probe -> yaw={math.degrees(yaw_delta):+.2f}deg, "
        f"steer_sign={steer_sign:+.0f}",
        flush=True,
    )

    _raw_wheel_action(articulation, wheel_indices, 0.0, 0.0)
    world.reset()
    restore_factory_amr_drives(stage)
    world.play()
    return wheel_sign, steer_sign


def smoke_test_reverse_controller(
    world,
    stage,
    articulation,
    wheel_indices,
    wheel_sign,
    steer_sign,
):
    """실제 controller가 목표 뒤쪽으로 후진하고 허용오차 안에 정차하는지 확인."""
    controller = GoToPoseController(
        ControllerConfig(
            position_tolerance=ARGS.position_tolerance,
            yaw_tolerance=math.radians(ARGS.yaw_tolerance_deg),
            max_linear=ARGS.max_linear,
            max_angular=ARGS.max_angular,
            min_angular=ARGS.min_angular,
            max_wheel_radps=ARGS.max_wheel_radps,
            allow_reverse=True,
            wheel_radius=ARGS.wheel_radius,
            track_width=ARGS.track_width,
        )
    )
    start = _measure_pose(articulation)
    reverse_distance = 0.60
    goal = Goal2D(
        x=start.x - math.cos(start.yaw) * reverse_distance,
        y=start.y - math.sin(start.yaw) * reverse_distance,
        yaw=start.yaw,
        yaw_required=False,
        allow_reverse=True,
    )
    controller.set_goal(goal)

    command = None
    max_steps = int(PHYSICS_HZ * 30.0)
    for step in range(max_steps):
        pose = _measure_pose(articulation)
        command = controller.compute(pose, 1.0 / PHYSICS_HZ)
        if step % int(PHYSICS_HZ * 2.0) == 0:
            print(
                "[smoke] auto reverse 진행: "
                f"t={step / PHYSICS_HZ:.1f}s, "
                f"phase={command.phase}, "
                f"dist={command.distance_error:.3f}m, "
                f"yaw_error={math.degrees(command.yaw_error):+.2f}deg, "
                f"v={command.linear:+.3f}m/s, "
                f"w={command.angular:+.3f}rad/s",
                flush=True,
            )
        if command.arrived:
            break
        left, right = controller.to_wheel_speeds(
            command.linear,
            command.angular,
            wheel_sign=wheel_sign,
            steer_sign=steer_sign,
        )
        _raw_wheel_action(
            articulation,
            wheel_indices,
            left,
            right,
        )
        world.step(render=not ARGS.headless)
    else:
        pose = _measure_pose(articulation)
        raise RuntimeError(
            "자동 후진 controller smoke test가 30초 안에 도착하지 못했습니다: "
            f"phase={command.phase if command else 'none'}, "
            f"position_error={math.hypot(goal.x - pose.x, goal.y - pose.y):.3f}m, "
            f"yaw_error="
            f"{math.degrees(normalize_angle(goal.yaw - pose.yaw)):.2f}deg"
        )

    _raw_wheel_action(articulation, wheel_indices, 0.0, 0.0)
    pose = _measure_pose(articulation)
    position_error = math.hypot(goal.x - pose.x, goal.y - pose.y)
    yaw_error = abs(normalize_angle(goal.yaw - pose.yaw))
    if command.travel_direction != "reverse":
        raise RuntimeError(
            "뒤쪽 목표인데 controller가 후진을 선택하지 않았습니다"
        )
    print(
        "[smoke] auto reverse controller 도착: "
        f"max_linear={ARGS.max_linear:.2f}m/s, "
        f"max_angular={ARGS.max_angular:.2f}rad/s, "
        f"position_error={position_error:.4f}m, "
        f"yaw_error={math.degrees(yaw_error):.2f}deg",
        flush=True,
    )

    world.reset()
    restore_factory_amr_drives(stage)
    world.play()


def main():
    usd_path = Path(ARGS.usd).expanduser().resolve()
    if not usd_path.is_file():
        raise FileNotFoundError(f"USD 파일 없음: {usd_path}")

    context = omni.usd.get_context()
    context.open_stage(str(usd_path))
    for _ in range(30):
        simulation_app.update()

    stage = context.get_stage()
    if stage is None:
        raise RuntimeError(f"USD stage 로드 실패: {usd_path}")

    workcell_rotation = rotate_busbar_workcell(
        stage,
        ARGS.busbar_workcell_yaw_deg,
    )
    ros_domain_id, ros_context_paths = configure_ros_domain(stage)
    print(
        "[workcell] USD 기본 + 추가 session 회전: "
        f"yaw={workcell_rotation.yaw_deg:+.1f}deg, "
        f"pivot=({workcell_rotation.pivot_world[0]:.4f}, "
        f"{workcell_rotation.pivot_world[1]:.4f}), "
        f"approach_yaw="
        f"{math.degrees(workcell_rotation.busbar_approach_yaw_rad):+.1f}deg",
        flush=True,
    )
    print(
        f"[ROS2] domain={ros_domain_id}, "
        f"Context={list(ros_context_paths)}",
        flush=True,
    )

    scenes, warnings = validate_stage(stage)
    print(
        f"[preflight] USD={usd_path}\n"
        f"[preflight] PhysicsScene={scenes}, Z-up, metersPerUnit=1",
        flush=True,
    )
    for warning in warnings:
        print(f"[preflight 경고] {warning}", flush=True)

    restore_factory_amr_drives(stage)
    print(
        "[preflight] AMR wheel/caster drive를 Nova Carter 원본 물리값으로 복원",
        flush=True,
    )

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / PHYSICS_HZ,
        rendering_dt=1.0 / PHYSICS_HZ,
    )
    articulation = world.scene.add(
        SingleArticulation(
            prim_path=CHASSIS_PATH,
            name="amr_drive_articulation",
            reset_xform_properties=False,
        )
    )
    world.reset()
    world.play()

    left_index = articulation.get_dof_index(LEFT_WHEEL_NAME)
    right_index = articulation.get_dof_index(RIGHT_WHEEL_NAME)
    if (
        left_index is None
        or right_index is None
        or int(left_index) < 0
        or int(right_index) < 0
    ):
        raise RuntimeError("articulation에서 좌우 wheel DOF index를 찾지 못했습니다")
    wheel_indices = (int(left_index), int(right_index))
    print(
        f"[preflight] articulation DOF={articulation.num_dof}, "
        f"wheel indices={wheel_indices}, radius={ARGS.wheel_radius:.4f}m, "
        f"track={ARGS.track_width:.4f}m, "
        f"wheel max force={ARGS.max_wheel_force:.1f}",
        flush=True,
    )

    baseline_frame = BaselineFrame(stage, articulation)
    print(
        f"[preflight] runtime start pose="
        f"({baseline_frame.origin_x:.6f}, {baseline_frame.origin_y:.6f}, "
        f"{baseline_frame.origin_z:.6f}), "
        f"yaw={math.degrees(baseline_frame.origin_yaw):.4f}deg",
        flush=True,
    )

    if ARGS.preflight_only:
        _raw_wheel_action(articulation, wheel_indices, 0.0, 0.0)
        print("[preflight] 완료 (로봇을 움직이지 않았습니다)", flush=True)
        return

    wheel_sign = ARGS.wheel_sign
    steer_sign = ARGS.steer_sign
    if ARGS.calibrate_signs or ARGS.smoke_test:
        wheel_sign, steer_sign = calibrate_drive_signs(
            world, stage, articulation, wheel_indices
        )
        # hard reset 뒤에도 DOF handle을 명시적으로 다시 초기화한다.
        articulation.initialize(world.physics_sim_view)
        left_index = articulation.get_dof_index(LEFT_WHEEL_NAME)
        right_index = articulation.get_dof_index(RIGHT_WHEEL_NAME)
        if (
            left_index is None
            or right_index is None
            or int(left_index) < 0
            or int(right_index) < 0
        ):
            raise RuntimeError(
                "hard reset 뒤 좌우 wheel DOF index를 다시 찾지 못했습니다"
            )
        wheel_indices = (int(left_index), int(right_index))
        print(
            "[smoke] 두 probe 뒤 hard reset 완료; 시작 pose 복원",
            flush=True,
        )
    if ARGS.smoke_test:
        smoke_test_reverse_controller(
            world,
            stage,
            articulation,
            wheel_indices,
            wheel_sign,
            steer_sign,
        )
        print(
            "[smoke] 자동 후진·고속 정차 시험 뒤 hard reset 완료",
            flush=True,
        )
        return

    rclpy.init()
    node = AmrPhysicsBridge(
        articulation=articulation,
        wheel_indices=wheel_indices,
        baseline_frame=baseline_frame,
        wheel_sign=wheel_sign,
        steer_sign=steer_sign,
        workcell_rotation=workcell_rotation,
    )

    try:
        while simulation_app.is_running() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            node.control_step(1.0 / PHYSICS_HZ)
            # headless는 창만 숨긴다. 고정 RGB-D 카메라가 perception source이므로
            # headless에서도 렌더 tick을 끄면 안 된다.
            world.step(render=True)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop()
            world.step(render=False)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[AMR drive fatal] {exc}", file=sys.stderr, flush=True)
        raise
    finally:
        simulation_app.close()

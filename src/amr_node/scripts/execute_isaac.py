#!/usr/bin/env python3
"""execute_isaac.py (amr_node) -- Isaac Sim에서 /amr/goal_pose(geometry_msgs/PoseStamped)를
구독해 Nova_Carter를 목표까지 이동시키고, 결과를 /amr/sim_pose(geometry_msgs/Pose2D)로
amr_node에 피드백하는 상주 프로세스.

시작 시 chassis_link에 PhysicsArticulationRootAPI가 있는지로 두 가지 모드 중 하나를
자동 선택한다 (단, FORCE_ANIMATION_MODE=True면 무조건 애니메이션):

  [물리 모드] (예: Collected_Busbar_amr/Busbar.usd -- ArticulationRootAPI + 바퀴
  PhysicsRevoluteJoint+DriveAPI가 실제로 존재) -- Play를 걸고 joint_wheel_left/right의
  drive:angular:physics:targetVelocity를 직접 제어하는 차동 조향 go-to-point 루프.
  실제 PhysX 시뮬레이션으로 바퀴가 굴러서 AMR이 이동한다.

  [애니메이션 모드] (물리 조인트 없어도 동작, FORCE_ANIMATION_MODE=True면 항상 이쪽) --
  물리 없이 X축 이동 -> Y축 이동 순차 2단계로 여러 프레임에 걸쳐 구간 보간 이동한다
  (대각선 동시 이동 금지 -- 조립 디버깅 중 "지금 X/Y 중 뭐가 움직이는지" 헷갈리는 걸
  줄이기 위함). 바퀴는 이동 거리만큼 orient만 시각적으로 회전(구름 애니메이션),
  몸체 자체는 방향 전환 없이 항상 같은 자세를 유지한다 -- 몸체 회전 애니메이션을
  넣었다가 회전 후 직진 구간에서 트레이/작업대를 파고드는 문제가 반복돼서 단순
  직선 이동 버전으로 롤백함 (2026-07-27).

WHEEL_VELOCITY_SIGN: 바퀴 회전축(axis=Z) 기준 양수 targetVelocity가 실제로 전진
방향인지는 Isaac Sim 실행 전까지 확정 못 했다 -- 반대로(목표에서 멀어지는 방향으로)
굴러가면 이 상수만 -1.0으로 뒤집으면 된다.

isaacpjt/M0609/execute_isaac.py(팔 조립 파이프라인, 별도 프로세스)와 이름은 같지만
경로가 다른 별개 파일이다 -- 서로 겹치지 않는다.

fms_interfaces(커스텀 msg)는 Isaac Sim 내부 rclpy(ROS Distro humble 내장 빌드)의
ament index에서 안 잡혀서 ModuleNotFoundError가 난다 (팀원 execute_isaac.py도 이래서
커스텀 msg 없이 geometry_msgs/std_msgs 표준 메시지만 쓴다). 그래서 이 스크립트도
표준 메시지만 쓴다:
  amr_node.py가 /amr/goal(fms_interfaces/AmrGoal, 커스텀) <- behavior_node를 받아서
  /amr/goal_pose(geometry_msgs/PoseStamped, 표준) -> 이 스크립트로 중계 발행한다.

move_amr_group.py와 같은 GROUP_PRIMS/BASELINE을 그대로 import해서 재사용한다.
차이점: move_amr_group.py는 파일을 열고 수정하고 저장하고 끝나는 1회성 스크립트지만,
이 스크립트는 Isaac Sim을 계속 띄워둔 채로 ROS2 요청이 올 때마다 라이브 스테이지를
즉시 수정한다 (저장은 안 함 -- 세션이 끝나면 사라지는 임시 위치 변경).

*** isaacpjt/M0609/execute_isaac.py(팔 조립 파이프라인)와는 별도의 독립 Isaac Sim
프로세스다. *** GPU 하나에서 SimulationApp 두 개를 동시에 못 띄우고, 띄워도 상태가
공유되지 않는다. "이동된 좌표에서 실제로 조립까지 되는지" 검증하려면:
  1) move_amr_group.py --save로 좌표를 파일에 저장한 뒤
  2) 팔 조립 스크립트가 그 저장된 파일을 열게 하는 방식이 지금은 더 안전하다.
이 스크립트는 "ROS2 요청 -> 실시간 이동 -> 피드백"이라는 node 통신 자체를 시연/검증하기 위함.

사용:
  AMR_BRIDGE_HEADLESS=1 <isaac_sim_python> execute_isaac.py [--usd /path/to/Busbar.usd]

  # 다른 터미널에서 요청 발행 (BASELINE 기준 delta, move_amr_group.py의 PRESETS와 동일 값):
  ros2 topic pub -1 /amr/goal_pose geometry_msgs/msg/PoseStamped \
      "{pose: {position: {x: 0.0, y: -0.5983, z: 0.0}}}"

  # 또는 amr_node.py를 통해 (behavior_node -> /amr/goal -> amr_node가 중계):
  ros2 topic pub -1 /amr/goal fms_interfaces/msg/AmrGoal \
      "{station_id: 'col2', x: 0.0, y: -0.5983, theta: 0.0}"
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

_HEADLESS = os.environ.get("AMR_BRIDGE_HEADLESS") == "1"
simulation_app = SimulationApp({"headless": _HEADLESS})

from omni.isaac.core.utils.extensions import enable_extension  # noqa: E402
enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402
from isaacsim.core.api import World  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from geometry_msgs.msg import Pose2D, PoseStamped  # noqa: E402

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from move_amr_group import GROUP_PRIMS, BASELINE  # noqa: E402

DEFAULT_USD = "/home/rokey/EV_combine/src/Collected_Busbar_amr/Busbar.usd"

CHASSIS_PATH = "/World/Nova_Carter/chassis_link"
WHEEL_JOINT_LEFT = "/World/Nova_Carter/joint_wheel_left"
WHEEL_JOINT_RIGHT = "/World/Nova_Carter/joint_wheel_right"

# 이동 애니메이션 파라미터 (애니메이션 모드용)
DRIVE_SPEED_MPS = 0.15      # 너무 빨라서 가짜처럼 보인다는 피드백으로 0.3 -> 0.15 (2026-07-27)
ANIM_STEPS_PER_SEC = 30.0
WHEEL_RADIUS_M = 0.125      # nova_carter_physics.usd의 wheel_collider Cylinder 실측값
WHEEL_PRIMS = [
    "/World/Nova_Carter/wheel_left",
    "/World/Nova_Carter/wheel_right",
]

# 조립 디버깅용 핸드오프를 빨리 넘기기 위해 물리 모드를 임시로 꺼둔다 (2026-07-27).
# 물리 모드(DriveAPI+Play)는 조향이 아직 안정적으로 수렴하지 않고, 씬이 꼬이면
# AMR이 물리적으로 끼어 아예 안 움직이는 문제까지 있었다. True로 두면 chassis_link에
# ArticulationRootAPI가 있어도(Collected_Busbar_amr) 무조건 애니메이션 모드로 동작한다.
# 물리 디버깅을 재개하려면 이 값만 False로 되돌리면 된다.
FORCE_ANIMATION_MODE = True

TRACK_WIDTH_M = 0.4132
MAX_LINEAR_MPS = 0.3
MAX_ANGULAR_RADPS = 1.0
K_LINEAR = 1.5
K_ANGULAR = 2.5
PHYSICS_ARRIVAL_TOLERANCE_M = 0.03
PHYSICS_DRIVE_TIMEOUT_SEC = 30.0
PHYSICS_STEP_HZ = 60.0
WHEEL_VELOCITY_SIGN = -1.0   # 반대 방향으로 구르면 -1.0으로 (실측 결과 반대였음, 2026-07-27 확인)


def _normalize_angle(angle_rad: float) -> float:
    """[-pi, pi] 범위로 정규화."""
    while angle_rad > math.pi:
        angle_rad -= 2 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2 * math.pi
    return angle_rad


def _get_translate_op(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"prim not found: {path}")
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    raise RuntimeError(f"{path}: xformOp:translate 없음 -- move_amr_group.py가 가정한 구조와 다름")


def _get_orient_op(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"prim not found: {path}")
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            return op
    raise RuntimeError(f"{path}: xformOp:orient 없음")


def _set_orient(op, quat: Gf.Quatd):
    """orient 어트리뷰트가 quatf(단정밀도)인 prim도 있어서(nut1/nut2 실측 확인,
    2026-07-27) 타입 보고 맞춰서 Set한다 -- 안 그러면 Tf.ErrorException(타입 불일치)."""
    if op.GetAttr().GetTypeName() == "quatf":
        op.Set(Gf.Quatf(quat))
    else:
        op.Set(quat)


def _has_physics_articulation(stage) -> bool:
    prim = stage.GetPrimAtPath(CHASSIS_PATH)
    return prim.IsValid() and prim.HasAPI(UsdPhysics.ArticulationRootAPI)


# 캐스터 스위블(조향) 조인트 -- 에셋 기본값으로 stiffness=1e9(사실상 고정)가 이미
# 걸려있다 (World0123의 lock_carter_base()가 거는 파킹브레이크와 동일한 값, 원래
# 에셋 기본값). 진짜 캐스터처럼 자유 회전해야 주행 중 끌림/저항이 안 생긴다.
CASTER_SWIVEL_JOINTS = [
    "/World/Nova_Carter/joint_caster_base",
    "/World/Nova_Carter/joint_swing_left",
    "/World/Nova_Carter/joint_swing_right",
]


def _unlock_casters(stage):
    """완전 고정(1e9, 에셋 기본값)도 완전 자유(stiffness=0)도 아닌 중간값으로.

    완전 자유로 풀었더니(2026-07-27 최초 시도) 전진 드리프트 진단에서 heading이
    실제 주행과 같은 방향으로 틀어지는 게 확인됐다 -- 진짜 캐스터처럼 진행 방향으로
    스스로 되돌아오려는 약한 복원력(self-centering)이 없어서 자유롭게 흔들리며
    반작용 토크를 만드는 것으로 보인다. 약한 스프링 형태로 0도(직진 정렬)를
    부드럽게 유지하도록 바꾼다. joint_swing_left는 에셋 기본 targetPosition이
    -12.8도로 비대칭이었던 것도 0으로 맞춰서 좌우 대칭으로 만든다.
    """
    for path in CASTER_SWIVEL_JOINTS:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        stiffness = prim.GetAttribute("drive:angular:physics:stiffness")
        damping = prim.GetAttribute("drive:angular:physics:damping")
        target_position = prim.GetAttribute("drive:angular:physics:targetPosition")
        if stiffness:
            stiffness.Set(100.0)
        if damping:
            damping.Set(50.0)
        if target_position:
            target_position.Set(0.0)


def _chassis_world_xy_and_forward(stage):
    """chassis_link의 현재 world (x, y)와 로컬 +X가 가리키는 world 방향(단위벡터, xy)."""
    prim = stage.GetPrimAtPath(CHASSIS_PATH)
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pos = xf.ExtractTranslation()
    fwd = xf.TransformDir(Gf.Vec3d(1, 0, 0))
    fwd_xy = Gf.Vec2d(fwd[0], fwd[1])
    length = fwd_xy.GetLength()
    if length > 1e-9:
        fwd_xy /= length
    return pos[0], pos[1], fwd_xy


def _set_wheel_velocities(stage, left_radps: float, right_radps: float):
    for path, v in ((WHEEL_JOINT_LEFT, left_radps), (WHEEL_JOINT_RIGHT, right_radps)):
        prim = stage.GetPrimAtPath(path)
        attr = prim.GetAttribute("drive:angular:physics:targetVelocity")
        if not attr:
            raise RuntimeError(f"{path}: drive:angular:physics:targetVelocity 없음")
        attr.Set(math.degrees(v))  # PhysX angular drive velocity 단위는 deg/s


STEER_CALIBRATION_PROBE_MPS = 0.15
STEER_CALIBRATION_STEPS = 45  # ~0.75s @ 60Hz


def _measure_heading(stage):
    _, _, fwd = _chassis_world_xy_and_forward(stage)
    return math.atan2(fwd[1], fwd[0])


def _calibrate_steering_sign(stage, world, log=print):
    """오른쪽 바퀴만 더 빠르게(직진 부호는 그대로 사용) 잠깐 굴려보고 실제 heading이
    어느 쪽으로 바뀌는지 측정해서 차동 조향 부호를 정한다."""
    heading0 = _measure_heading(stage)

    wl = WHEEL_VELOCITY_SIGN * 0.0 / WHEEL_RADIUS_M
    wr = WHEEL_VELOCITY_SIGN * STEER_CALIBRATION_PROBE_MPS / WHEEL_RADIUS_M
    _set_wheel_velocities(stage, wl, wr)
    for _ in range(STEER_CALIBRATION_STEPS):
        world.step(render=True)
    _set_wheel_velocities(stage, 0.0, 0.0)
    for _ in range(10):
        world.step(render=True)

    heading1 = _measure_heading(stage)
    delta_heading_deg = math.degrees(_normalize_angle(heading1 - heading0))

    steer_sign = 1.0 if delta_heading_deg >= 0 else -1.0

    log(
        f"[조향 캘리브레이션] 오른쪽 바퀴만 {STEER_CALIBRATION_PROBE_MPS}m/s로 "
        f"{STEER_CALIBRATION_STEPS / PHYSICS_STEP_HZ:.2f}s 굴림 -> heading "
        f"{math.degrees(heading0):.1f}deg -> {math.degrees(heading1):.1f}deg "
        f"(delta={delta_heading_deg:+.1f}deg) -> STEER_SIGN={steer_sign:+.0f}")
    if abs(delta_heading_deg) < 1.0:
        log(
            f"[조향 캘리브레이션] 경고: heading 변화가 {delta_heading_deg:.2f}deg로 너무 작다 -- "
            f"바퀴 차동이 요(yaw)에 미치는 영향이 약하거나 다른 저항이 지배적일 수 있음.")

    return steer_sign


FORWARD_PROBE_MPS = 0.2
FORWARD_PROBE_STEPS = 90  # ~1.5s @ 60Hz


def _diagnose_forward_drift(stage, world, log=print):
    """차동 없이 양쪽 바퀴를 똑같이 굴려서 순수 전진만 시켜보고 heading이 얼마나
    틀어지는지 측정한다."""
    heading0 = _measure_heading(stage)

    w = WHEEL_VELOCITY_SIGN * FORWARD_PROBE_MPS / WHEEL_RADIUS_M
    _set_wheel_velocities(stage, w, w)
    for _ in range(FORWARD_PROBE_STEPS):
        world.step(render=True)
    _set_wheel_velocities(stage, 0.0, 0.0)
    for _ in range(10):
        world.step(render=True)

    heading1 = _measure_heading(stage)
    delta_heading_deg = math.degrees(_normalize_angle(heading1 - heading0))

    log(
        f"[전진 드리프트 진단] 양쪽 바퀴 동일하게 {FORWARD_PROBE_MPS}m/s로 "
        f"{FORWARD_PROBE_STEPS / PHYSICS_STEP_HZ:.2f}s 전진 -> heading "
        f"{math.degrees(heading0):.1f}deg -> {math.degrees(heading1):.1f}deg "
        f"(delta={delta_heading_deg:+.1f}deg)")
    if abs(delta_heading_deg) > 3.0:
        log(
            "[전진 드리프트 진단] 차동 없이도 heading이 크게 틀어짐 -- "
            "조향 문제가 아니라 전진 자체(캐스터 재정렬 반작용 등)가 원인일 가능성 높음.")


class AmrBridgeNode(Node):

    def __init__(self, stage, world=None, steer_sign=1.0):
        super().__init__("amr_bridge_node")
        self._stage = stage
        self._world = world
        self._physics_mode = world is not None
        self._steer_sign = steer_sign
        self._sim_pose_pub = self.create_publisher(Pose2D, "/amr/sim_pose", 10)
        self._goal_sub = self.create_subscription(
            PoseStamped, "/amr/goal_pose", self._on_goal_pose, 10)

        if not self._physics_mode:
            # 바퀴 굴림 애니메이션은 이 base orient 위에 로컬 Z축 회전을 얹는 식으로
            # 계산한다 (매번 현재값을 누적하면 오차가 쌓이므로, 항상 이 고정된 기준에서
            # 다시 계산). 물리 모드에서는 안 씀 (실제 DriveAPI가 바퀴를 굴림).
            self._wheel_base_orient = {
                path: Gf.Quatd(_get_orient_op(stage, path).Get()) for path in WHEEL_PRIMS
            }
            self._wheel_roll_deg = 0.0

        mode_str = "물리(DriveAPI+Play)" if self._physics_mode else "애니메이션(Xform 보간, 직선만)"
        self.get_logger().info(f"AMR 브릿지 준비 완료 [{mode_str}] -- /amr/goal_pose 대기 중")

    def _current_group_delta(self) -> Gf.Vec3d:
        current = _get_translate_op(self._stage, "/World/Nova_Carter").Get()
        return Gf.Vec3d(current) - BASELINE["/World/Nova_Carter"]

    def _apply_group_delta(self, delta: Gf.Vec3d):
        for path in GROUP_PRIMS:
            _get_translate_op(self._stage, path).Set(Gf.Vec3d(BASELINE[path]) + delta)

    def _spin_wheels(self, distance_delta_m: float):
        self._wheel_roll_deg += math.degrees(distance_delta_m / WHEEL_RADIUS_M)
        roll = Gf.Rotation(Gf.Vec3d(0, 0, 1), self._wheel_roll_deg).GetQuat()
        for path in WHEEL_PRIMS:
            new_orient = self._wheel_base_orient[path] * roll
            _set_orient(_get_orient_op(self._stage, path), Gf.Quatd(new_orient))

    def _on_goal_pose(self, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.get_logger().info(f"SUB /amr/goal_pose <- ({x:.4f}, {y:.4f})")
        if self._physics_mode:
            self._drive_physics(x, y)
        else:
            self._drive_animation(x, y)

    def _drive_animation(self, x: float, y: float):
        """X 이동 -> Y 이동 순차 2단계로 진행한다 (대각선 동시 이동 금지). 단순
        직선 보간만 한다 -- 몸체 회전 애니메이션을 넣었다가 회전 후 직진 구간에서
        트레이/작업대를 파고드는 문제가 반복돼서 롤백함 (2026-07-27)."""
        try:
            start_delta = self._current_group_delta()
        except RuntimeError as e:
            self.get_logger().error(f"이동 실패: {e}")
            return

        via_delta = Gf.Vec3d(x, start_delta[1], 0.0)  # 1단계: X만 이동
        target_delta = Gf.Vec3d(x, y, 0.0)             # 2단계: Y만 이동

        try:
            after_x = self._animate_segment(start_delta, via_delta, "X")
            self._animate_segment(after_x, target_delta, "Y")
        except RuntimeError as e:
            self.get_logger().error(f"이동 중 실패: {e}")
            return

        # X/Y 둘 다 거리 0(이미 그 위치)이면 스텝 루프가 아예 안 돌아서 /amr/sim_pose가
        # 한 번도 발행 안 되고, amr_node가 도착 판정을 영영 못 한다 (2026-07-27 확인)
        # -- 그래서 이동 여부와 무관하게 항상 마지막에 한 번 더 발행.
        pose = Pose2D()
        pose.x, pose.y = x, y
        self._sim_pose_pub.publish(pose)
        self.get_logger().info(f"PUB /amr/sim_pose -> ({x:.4f}, {y:.4f}) 도착")

    def _animate_segment(self, start_delta: Gf.Vec3d, target_delta: Gf.Vec3d, axis_label: str) -> Gf.Vec3d:
        distance = (target_delta - start_delta).GetLength()
        if distance < 1e-6:
            return start_delta

        duration_sec = max(distance / DRIVE_SPEED_MPS, 1.0 / ANIM_STEPS_PER_SEC)
        n_steps = max(int(round(duration_sec * ANIM_STEPS_PER_SEC)), 1)
        self.get_logger().info(
            f"[애니메이션 {axis_label}축] 이동 시작: {distance:.3f}m, {n_steps}스텝 (~{duration_sec:.1f}s)")

        prev_delta = start_delta
        cur_delta = start_delta
        frame_dt = 1.0 / ANIM_STEPS_PER_SEC
        for i in range(1, n_steps + 1):
            frame_start = time.monotonic()
            t = i / n_steps
            cur_delta = start_delta + (target_delta - start_delta) * t
            self._apply_group_delta(cur_delta)
            self._spin_wheels((cur_delta - prev_delta).GetLength())
            prev_delta = cur_delta

            pose = Pose2D()
            pose.x, pose.y = cur_delta[0], cur_delta[1]
            self._sim_pose_pub.publish(pose)

            simulation_app.update()
            # simulation_app.update()는 자체적으로 프레임레이트를 제한하지 않아서
            # (실측 175Hz+, 2026-07-27) 명시적으로 페이싱해야 DRIVE_SPEED_MPS가 실제
            # 벽시계 속도로 반영된다.
            time.sleep(max(0.0, frame_dt - (time.monotonic() - frame_start)))

        return cur_delta

    def _drive_physics(self, target_x: float, target_y: float):
        """joint_wheel_left/right의 DriveAPI targetVelocity를 직접 제어하는 차동 조향
        (heading 오차 기반 좌우 바퀴 속도 차이) go-to-point 루프. Play 상태에서
        world.step(render=True)로 실제 PhysX를 스텝한다.

        target_x/target_y는 move_amr_group.py와 동일하게 BASELINE(그룹 최초 위치) 기준
        delta다 -- Nova_Carter 자체 world 좌표가 아니라 group의 BASELINE + delta가 목표.
        """
        nova_carter_baseline = Gf.Vec3d(BASELINE["/World/Nova_Carter"])
        target_wx = nova_carter_baseline[0] + target_x
        target_wy = nova_carter_baseline[1] + target_y

        if not self._world.is_playing():
            self._world.play()

        max_steps = int(PHYSICS_DRIVE_TIMEOUT_SEC * PHYSICS_STEP_HZ)
        self.get_logger().info(
            f"[물리] 목표 world=({target_wx:.4f}, {target_wy:.4f}), 최대 {PHYSICS_DRIVE_TIMEOUT_SEC:.0f}s")

        arrived = False
        for step in range(max_steps):
            cur_x, cur_y, fwd_xy = _chassis_world_xy_and_forward(self._stage)
            remaining = Gf.Vec2d(target_wx - cur_x, target_wy - cur_y)
            distance = remaining.GetLength()

            if distance <= PHYSICS_ARRIVAL_TOLERANCE_M:
                arrived = True
                break

            target_bearing = math.atan2(remaining[1], remaining[0])
            current_heading = math.atan2(fwd_xy[1], fwd_xy[0])
            heading_error = _normalize_angle(target_bearing - current_heading)

            angular_w = max(-MAX_ANGULAR_RADPS, min(MAX_ANGULAR_RADPS, K_ANGULAR * heading_error))
            forward_gain = max(math.cos(heading_error), 0.0)
            linear_v = max(0.0, min(MAX_LINEAR_MPS, K_LINEAR * distance * forward_gain))

            v_left = linear_v - self._steer_sign * angular_w * TRACK_WIDTH_M / 2.0
            v_right = linear_v + self._steer_sign * angular_w * TRACK_WIDTH_M / 2.0
            wheel_left_radps = WHEEL_VELOCITY_SIGN * v_left / WHEEL_RADIUS_M
            wheel_right_radps = WHEEL_VELOCITY_SIGN * v_right / WHEEL_RADIUS_M

            try:
                _set_wheel_velocities(self._stage, wheel_left_radps, wheel_right_radps)
            except RuntimeError as e:
                self.get_logger().error(f"바퀴 속도 설정 실패: {e}")
                break

            self._world.step(render=True)

            if step % int(PHYSICS_STEP_HZ) == 0:
                pose = Pose2D()
                pose.x, pose.y = cur_x - nova_carter_baseline[0], \
                    cur_y - nova_carter_baseline[1]
                self._sim_pose_pub.publish(pose)
                self.get_logger().info(
                    f"  진행: dist={distance:.3f}m heading={math.degrees(current_heading):.1f}deg "
                    f"heading_err={math.degrees(heading_error):.1f}deg v={linear_v:.2f} w={angular_w:.2f}")

        _set_wheel_velocities(self._stage, 0.0, 0.0)  # 정지
        self._world.step(render=True)

        final_x, final_y, _ = _chassis_world_xy_and_forward(self._stage)
        pose = Pose2D()
        pose.x = final_x - nova_carter_baseline[0]
        pose.y = final_y - nova_carter_baseline[1]
        self._sim_pose_pub.publish(pose)

        if arrived:
            self.get_logger().info(f"[물리] 도착 (world=({final_x:.4f}, {final_y:.4f}))")
        else:
            self.get_logger().error(
                f"[물리] 타임아웃 -- 목표 미도달 (world=({final_x:.4f}, {final_y:.4f})). "
                f"WHEEL_VELOCITY_SIGN이 반대일 수 있음.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", default=DEFAULT_USD, help="대상 Busbar.usd 경로")
    args = parser.parse_args()

    if not os.path.isfile(args.usd):
        print(f"USD 파일 없음: {args.usd}", file=sys.stderr)
        simulation_app.close()
        sys.exit(1)

    ctx = omni.usd.get_context()
    ctx.open_stage(args.usd)
    for _ in range(20):
        simulation_app.update()

    stage = ctx.get_stage()
    if stage is None:
        print(f"스테이지 로드 실패: {args.usd}", file=sys.stderr)
        simulation_app.close()
        sys.exit(1)

    world = None
    steer_sign = 1.0
    if not FORCE_ANIMATION_MODE and _has_physics_articulation(stage):
        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / PHYSICS_STEP_HZ)
        world.reset()
        _unlock_casters(stage)
        for _ in range(5):
            simulation_app.update()
        steer_sign = _calibrate_steering_sign(stage, world)
        _diagnose_forward_drift(stage, world)

    rclpy.init()
    node = AmrBridgeNode(stage, world=world, steer_sign=steer_sign)

    try:
        while simulation_app.is_running():
            if world is not None:
                world.step(render=True)
            else:
                simulation_app.update()
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()

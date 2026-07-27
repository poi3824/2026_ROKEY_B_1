#!/usr/bin/env python3
"""
execute_isaac.py - Busbar Automation Sequence for Isaac Sim
BehaviorNode (FSM) 및 ArmNode, Vision Correction Node 통신 연동 버전

[동작 방식]
0. INIT_POSE            : 관절값 [0, 90, 0, 0, 90, 0] 도 단위로 초기 위치 정렬
1. SCAN_BATTERY         : 초기 위치 기준 Z=0.7m 스캔 위치로 이동
2. SCAN_BUSBAR          : 상대 위치 이동 및 버스바 스캔
3. PICK_BUSBAR          : Z=0.6m 상공 접근 -> Z=0.455m 파지 위치 하강 -> Kinematic 파지 및 상승
4. MOVE_BATTERY_CENTER  : ArmNode에서 보낸 배터리 중점 좌표 상공(Z=0.7m)으로 이동
5. FINE_ALIGNMENT       : 비전 노드의 START_ERRORFIX_CORRECTION 트리거 발송 후, 1픽셀 오차 보정 피드백에 맞춰 미세 정렬
6. ASSEMBLE_BUSBAR      : 정렬된 XY 상태를 유지하며 수직 하강 안착, 버스바 고정 해제, 그리퍼 개방 및 상공 이탈
7. SCAN_NUT1 / SCAN_NUT2   : 너트 스캔 위치(버스바 체결 위치 기준 상대 이동)로 이동
8. PICK_NUT1 / PICK_NUT2   : 초기 위치(HOME_EE_POS) 기준 고정 상대좌표로 접근(너트는 Nova
   Carter에 고정되어 비전 없이도 위치가 일정함), 물리 파지(Gripper Close) 및 상승
9. ASSEMBLE_NUT1 / ASSEMBLE_NUT2 : 볼트 1/2번의 실측 월드 좌표(BOLT1/2_WORLD_POS, test_isaac
   씬 고정 배치 기준 하드코딩)로 이동, 착좌 및 Screwing 체결(Regrasp 포함)
10. RETURN_HOME_JOINTS  : 너트 2번 체결 완료 후 초기 관절 각도 [0, 0, 90, 0, 90, 0]로 완전 복귀
"""

import os
import sys
import math
import time
import numpy as np
from pathlib import Path

# 1. Isaac SimulationApp 초기화
from isaacsim import SimulationApp

_HEADLESS = os.environ.get("AMR_HEADLESS") == "1"
simulation_app = SimulationApp({"headless": _HEADLESS})

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

sys.stdout.reconfigure(line_buffering=True)


class _Tee:
    """print()/stdout 출력을 원본 스트림과 로그 파일에 동시에 기록한다."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


# 매 실행마다 콘솔 출력(단계별 phase, [Dynamic Target Set] 등)을 로그 파일에도
# 자동 기록한다 - 별도로 `| tee`를 붙이지 않아도 사후 디버깅에 사용할 수 있다.
_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_PATH = _LOG_DIR / f"isaac_run_{time.strftime('%Y%m%d_%H%M%S')}.log"
_log_file = open(_LOG_PATH, "a", buffering=1)
sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)
print(f"[Log] 콘솔 출력을 다음 파일에도 기록합니다: {_LOG_PATH}")

# 2. USD 및 Isaac Core Imports
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.core.utils.types import ArticulationAction

# 3. ROS 2 Imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose2D
from std_msgs.msg import String, Float32, Empty

# RMPFlow Controller 경로 설정
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "rmpflow"))
from m0609_rmpflow_controller import RMPFlowController

# ══════════════════════════════════════════════════════════════════════════
#  [A] 설정 및 파라미터
# ══════════════════════════════════════════════════════════════════════════
USD_PATH = "/home/rokey/EV_combine/src/Collected_Busbar/Busbar.usd"

NOVA_CARTER_ROOT = "/World/Nova_Carter/chassis_link"
M0609_PATH       = "/World/m0609"
EE_LINK_NAME     = "link_6"
GRIPPER_JOINTS   = ["finger_joint", "right_inner_knuckle_joint"]

# AMR 이동 파라미터 (amr_node의 arrival_tolerance_m 기본값 0.05m보다 더 정확히 세워서 도착)
AMR_LINEAR_SPEED  = 0.3    # m/s
AMR_ANGULAR_SPEED = 1.0    # rad/s
AMR_POS_TOL       = 0.02   # m
AMR_YAW_TOL       = 0.03   # rad

BUSBAR_ROOT_PATH      = "/World/busbar"
BUSBAR_POLYSHAPE_PATH = "/World/busbar/geo/PolyShape"

# 너트 Prim 경로
NUT1_ROOT_PATH      = "/World/nut1"
NUT2_ROOT_PATH      = "/World/nut2"
NUT1_POLYSHAPE_PATH = "/World/nut1/geo/PolyShape"
NUT2_POLYSHAPE_PATH = "/World/nut2/geo/PolyShape"

# 그리퍼 파라미터
GRIPPER_OPEN      = np.array([0.0, 0.0])
GRIPPER_CLOSE     = np.array([0.8, 0.8])
GRIPPER_CLOSE_NUT = np.array([0.96, 0.96])
GRIPPER_DELTA     = np.array([-0.5, -0.5])
GRIP_CLOSE_RAMP_STEPS = 50

# Kinematic Pose-Glue 파라미터
EE_OFFSET = np.array([0.0, 0.0, 0.185])
BUSBAR_HEIGHT = 0.003
BUSBAR_GRASP_Z_LOCAL = BUSBAR_HEIGHT + 0.02
BUSBAR_REST_ORIENTATION = np.array([0.5, -0.5, 0.5, 0.5])

# 기본 좌표 및 높이 정의
TARGET_INIT_JOINTS   = np.array([0.0, 0.0, np.radians(90.0), 0.0, np.radians(90.0), 0.0])

_POS_GRAB_PICK       = 0.455
SCAN_APPROACH_Z      = 0.7     # 배터리 스캔 고도
BUSBAR_SCAN_Z        = 0.7     # 버스바 스캔 고도
BUSBAR_APPROACH_Z    = 0.60   # 버스바 파지 접근 고도 (Z = 0.6m)
BATTERY_CENTER_Z     = 0.40   # 배터리 중점 이동 고도 (Z = 0.7m)

# 버스바 체결 및 하강 제어 파라미터
INSERT_SPEED            = 0.0005   # Step당 수직 하강 거리
BUSBAR_RELEASE_Z        = 0.37     # 그리퍼 해제 및 체결 완료 임계 Z 높이
INSERT_TOLERANCE_STRICT = 0.001    # Insert 단계 오차 허용범위 (1mm)

# ══════════════════════════════════════════════════════════════════════════
#  너트 조립(Nut Assembly) 파라미터
# ══════════════════════════════════════════════════════════════════════════
NUT_SCAN_OFFSET_X = -0.5
NUT_SCAN_OFFSET_Y = -0.3
NUT_SCAN_Z        = 0.9

NUT1_OFFSET_FROM_HOME = np.array([-0.4587, -0.2957])
NUT2_OFFSET_FROM_HOME = np.array([-0.3683, -0.2961])

NUT_HEIGHT         = 0.0095
NUT_GRASP_Z_LOCAL  = NUT_HEIGHT + 0.023
NUT_SUPPLY_TABLE_Z = 0.72                                   # 너트 공급대 높이
NUT_PICK_Z         = NUT_SUPPLY_TABLE_Z - (NUT_GRASP_Z_LOCAL - 0.0395)
NUT_APPROACH_Z     = 0.8                                     # 너트 파지 상공 고도
BOLT_APPROACH_Z    = 0.6                                     # 너트 체결 상공 고도

# ★ 볼트 1/2번 실측 월드 좌표 (station별 XY, test_isaac 씬 고정 배치 기준) ★
# AMR_WORK_STATION 환경변수로 대상 station 선택. 기본값 station_3 (기존 동작 그대로 유지).
STATION_BOLT_POSITIONS = {
    "station_3": {
        "bolt1": np.array([1.0552, 0.3722]),
        "bolt2": np.array([1.2636, 0.0098]),
    },
    "station_4": {
        "bolt1": np.array([1.0552, -0.2047]),
        "bolt2": np.array([1.2636, -0.5671]),
    },
}
_WORK_STATION = os.environ.get("AMR_WORK_STATION", "station_3")
if _WORK_STATION not in STATION_BOLT_POSITIONS:
    raise ValueError(
        f"AMR_WORK_STATION must be one of {sorted(STATION_BOLT_POSITIONS)}, "
        f"got {_WORK_STATION!r}")
BOLT1_WORLD_POS = STATION_BOLT_POSITIONS[_WORK_STATION]["bolt1"]
BOLT2_WORLD_POS = STATION_BOLT_POSITIONS[_WORK_STATION]["bolt2"]

# 너트 체결(Screwing) 파라미터
ENGAGE_LEN           = 0.0125     # 체결 깊이 (12.5mm)
SCREW_TURNS_DEG      = 350.0      # 1회전당 350도
REGRASP_CYCLES       = 1          # 총 2회전 (Regrasp 1회)
SCREW_OMEGA_DEG_S    = 120.0      # 초당 120도 회전
REGRASP_LIFT_HEIGHT  = 0.06       # Regrasp 시 수직 상승 높이 (6cm)
REGRASP_Z_OFFSET     = 0.005      # Regrasp 하강/재파지 시 높이 보정

TOTAL_REV   = (SCREW_TURNS_DEG / 360.0) * (1 + REGRASP_CYCLES)
NUT_PITCH_M = ENGAGE_LEN / TOTAL_REV

TORQUE_THRESHOLD     = 45.0       # 6번 조인트 반력 임계값 (Nm)
STUCK_Z_DELTA_THRESH = 0.0001     # Z축 하강 멈춤 판정 기준 (0.1mm)
STUCK_STEP_LIMIT     = 12         # Z축 변화 없이 토크 지속되는 Step 수
TCP_FORCE_CHECK_Z    = 0.378      # TCP(EE) 높이가 이 값 이하일 때만 힘/토크 감지 활성화

# 위치 저장용 변수
HOME_EE_POS          = None
SCAN_POS             = None
BUSBAR_SCAN_POS      = None
BATTERY_CENTER_POS   = None
target_fine_pos      = None

PICK_TOLERANCE_STRICT    = 0.01     # 10mm
JOINT_TOLERANCE          = 0.02     # 관절 오차 허용범위 (rad)
PICK_TOLERANCE_LOOSE_VAL = 0.015
MAX_STUCK_STEPS          = 1000000  # Phase 타임아웃 기준
PHYSICS_DT               = 1.0 / 60.0

URDF_PATH        = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
ROBOT_DESC_PATH  = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
RMPFLOW_CFG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")

# ══════════════════════════════════════════════════════════════════════════
#  [B] 비전 브릿지 및 유틸리티 함수
# ══════════════════════════════════════════════════════════════════════════
class Execute_Isaac_Busar(Node):
    """ArmNode / BehaviorNode / VisionNode 통신 브릿지"""
    def __init__(self):
        super().__init__("execute_isaac_busar")
        self.latest_target_pose = None
        self.requested_task = None
        self.alignment_success = False

        self.sub_target_pose = self.create_subscription(
            PoseStamped, '/target_pose', self._on_target_pose, 10
        )
        self.sub_task_cmd = self.create_subscription(
            String, '/task_command', self._on_task_command, 10
        )

        # amr_node 연동 (amr_node.py: PUB /amr/goal_pose, SUB /amr/sim_pose, SUB /amr/cancel)
        self.amr_goal_pose = None
        self.amr_cancel_requested = False
        self.sub_amr_goal_pose = self.create_subscription(
            PoseStamped, '/amr/goal_pose', self._on_amr_goal_pose, 10
        )
        self.sub_amr_cancel = self.create_subscription(
            Empty, '/amr/cancel', self._on_amr_cancel, 10
        )
        self.pub_amr_sim_pose = self.create_publisher(Pose2D, '/amr/sim_pose', 10)

        self.pub_phase = self.create_publisher(String, '/isaac_phase', 10)
        self.pub_progress = self.create_publisher(Float32, '/isaac_progress', 10)
        self.pub_status = self.create_publisher(String, '/isaac_status', 10)
        self.pub_errorfix_command = self.create_publisher(String, '/errorfix_command', 10)

    def _on_target_pose(self, msg: PoseStamped):
        self.latest_target_pose = msg

    def _on_amr_goal_pose(self, msg: PoseStamped):
        self.amr_goal_pose = msg

    def _on_amr_cancel(self, msg: Empty):
        self.amr_cancel_requested = True

    def _on_task_command(self, msg: String):
        self.get_logger().info(f"[Task Command 수신]: {msg.data}")
        if msg.data == "ALIGNMENT_SUCCESS":
            self.alignment_success = True
        else:
            self.requested_task = msg.data


def euler_to_quaternion_wxyz(roll, pitch, yaw):
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * sp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])


def disable_physics_recursively(stage, prim_path):
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        return
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)


def enable_physics_recursively(stage, prim_path):
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        return
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(True)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(True)


def world_xf(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim 경로가 올바르지 않습니다: {path}")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)


def find_prim_path(stage, root_path, name):
    root = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def lock_amr_base(stage, amr_root_path):
    """로봇팔 작업 중 AMR 바퀴를 고정(브레이크)한다 - 팔의 반력으로 베이스가
    흔들리는 것을 방지."""
    amr_prim = stage.GetPrimAtPath(amr_root_path).GetParent()
    if not amr_prim.IsValid():
        amr_prim = stage.GetPrimAtPath(amr_root_path)

    for prim in Usd.PrimRange(amr_prim):
        for dt in ("angular", "linear"):
            drive = UsdPhysics.DriveAPI.Get(prim, dt)
            if drive:
                drive.GetStiffnessAttr().Set(1.0e9)
                drive.GetDampingAttr().Set(1.0e6)
                drive.GetMaxForceAttr().Set(1.0e9)
                if drive.GetTargetVelocityAttr():
                    drive.GetTargetVelocityAttr().Set(0.0)


def unlock_amr_base(stage, amr_root_path):
    """AMR 이동 시작 전 바퀴 구동부를 풀어준다 (lock_amr_base의 반대) - 베이스를
    set_world_pose로 직접 이동시키므로 바퀴 조인트가 그 이동에 저항하지 않도록 함."""
    amr_prim = stage.GetPrimAtPath(amr_root_path).GetParent()
    if not amr_prim.IsValid():
        amr_prim = stage.GetPrimAtPath(amr_root_path)

    for prim in Usd.PrimRange(amr_prim):
        for dt in ("angular", "linear"):
            drive = UsdPhysics.DriveAPI.Get(prim, dt)
            if drive:
                drive.GetStiffnessAttr().Set(0.0)
                drive.GetDampingAttr().Set(0.0)
                drive.GetMaxForceAttr().Set(0.0)


def quat_wxyz_to_yaw(quat_wxyz):
    """월드 Z축 기준 yaw(rad)만 추출 (AMR은 평면 위를 움직이므로 roll/pitch는 무시)."""
    w, x, y, z = quat_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def compute_local_offset(parent_pos, parent_quat_wxyz, child_pos, child_quat_wxyz):
    """parent(AMR) 월드 포즈 기준 child(너트)의 로컬 오프셋 포즈(위치+회전) 계산."""
    parent_rot = Gf.Rotation(Gf.Quatd(
        float(parent_quat_wxyz[0]),
        Gf.Vec3d(float(parent_quat_wxyz[1]), float(parent_quat_wxyz[2]), float(parent_quat_wxyz[3]))))
    parent_mat = Gf.Matrix4d().SetRotate(parent_rot)
    parent_mat.SetTranslateOnly(Gf.Vec3d(*[float(v) for v in parent_pos]))

    child_rot = Gf.Rotation(Gf.Quatd(
        float(child_quat_wxyz[0]),
        Gf.Vec3d(float(child_quat_wxyz[1]), float(child_quat_wxyz[2]), float(child_quat_wxyz[3]))))
    child_mat = Gf.Matrix4d().SetRotate(child_rot)
    child_mat.SetTranslateOnly(Gf.Vec3d(*[float(v) for v in child_pos]))

    local_mat = child_mat * parent_mat.GetInverse()
    local_pos = local_mat.ExtractTranslation()
    local_quat = local_mat.ExtractRotationQuat()
    return (
        np.array([local_pos[0], local_pos[1], local_pos[2]]),
        np.array([local_quat.GetReal(), *local_quat.GetImaginary()]),
    )


def compose_world_pose(parent_pos, parent_quat_wxyz, local_pos, local_quat_wxyz):
    """parent(AMR) 현재 월드 포즈에 로컬 오프셋을 합성해 child(너트)의 월드 포즈 계산."""
    parent_rot = Gf.Rotation(Gf.Quatd(
        float(parent_quat_wxyz[0]),
        Gf.Vec3d(float(parent_quat_wxyz[1]), float(parent_quat_wxyz[2]), float(parent_quat_wxyz[3]))))
    parent_mat = Gf.Matrix4d().SetRotate(parent_rot)
    parent_mat.SetTranslateOnly(Gf.Vec3d(*[float(v) for v in parent_pos]))

    local_rot = Gf.Rotation(Gf.Quatd(
        float(local_quat_wxyz[0]),
        Gf.Vec3d(float(local_quat_wxyz[1]), float(local_quat_wxyz[2]), float(local_quat_wxyz[3]))))
    local_mat = Gf.Matrix4d().SetRotate(local_rot)
    local_mat.SetTranslateOnly(Gf.Vec3d(*[float(v) for v in local_pos]))

    world_mat = local_mat * parent_mat
    world_pos = world_mat.ExtractTranslation()
    world_quat = world_mat.ExtractRotationQuat()
    return (
        np.array([world_pos[0], world_pos[1], world_pos[2]]),
        np.array([world_quat.GetReal(), *world_quat.GetImaginary()]),
    )


def glue_busbar_to_ee(robot, busbar_xform, rest_pick_pos, blend):
    if busbar_xform is None or rest_pick_pos is None:
        return
    ee_pos, ee_quat = robot.end_effector.get_world_pose()
    grasp_point_pos = np.asarray(ee_pos) - EE_OFFSET
    target_pos = grasp_point_pos - np.array([0.0, 0.0, BUSBAR_GRASP_Z_LOCAL])

    busbar_pos = rest_pick_pos + blend * (target_pos - rest_pick_pos)
    busbar_xform.set_world_pose(position=busbar_pos, orientation=np.asarray(ee_quat))


def yaw_rotated_quat(base_wxyz, delta_deg):
    """base_wxyz 오리엔테이션을 월드 Z축 기준으로 delta_deg 만큼 추가 회전시킨 쿼터니언 반환 (Screwing 회전용)"""
    base_q = Gf.Quatd(float(base_wxyz[0]), Gf.Vec3d(float(base_wxyz[1]), float(base_wxyz[2]), float(base_wxyz[3])))
    base_rot = Gf.Rotation(base_q)
    extra_rot = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), float(delta_deg))
    combined = extra_rot * base_rot
    q = combined.GetQuat()
    return np.array([q.GetReal(), *q.GetImaginary()])


def resolve_nut_assets(nut_index, nut1_xform, nut2_xform):
    """nut_index(1 또는 2)에 따라 대상 너트 Xform과 라벨을 반환"""
    if nut_index == 1:
        return nut1_xform, "너트 1번"
    return nut2_xform, "너트 2번"


def update_target_positions(target_pose_msg: PoseStamped):
    global BUSBAR_APPROACH_POS, BUSBAR_PICK_POS, BUSBAR_LIFT_MOVE_POS

    pos = target_pose_msg.pose.position
    
    # 1) 실제 파지 좌표 (Z = 0.455m)
    pick_z = _POS_GRAB_PICK
    new_pick = np.array([pos.x, pos.y, pick_z])
    BUSBAR_PICK_POS = new_pick

    # 2) 상공 접근 좌표 (Z = 0.600m 고정)
    BUSBAR_APPROACH_POS = np.array([pos.x, pos.y, BUSBAR_APPROACH_Z])

    # 3) 상승 이동 좌표 (Z = 0.600m 고정)
    BUSBAR_LIFT_MOVE_POS = np.array([pos.x, pos.y + 0.1, BUSBAR_APPROACH_Z])

    print(f"[Dynamic Target Set] Approach Pos (Z={BUSBAR_APPROACH_Z:.2f}m): ({BUSBAR_APPROACH_POS[0]:.4f}, {BUSBAR_APPROACH_POS[1]:.4f}, {BUSBAR_APPROACH_POS[2]:.4f})")
    print(f"[Dynamic Target Set] Pick Pos     (Z={BUSBAR_PICK_POS[2]:.4f}m): ({BUSBAR_PICK_POS[0]:.4f}, {BUSBAR_PICK_POS[1]:.4f}, {BUSBAR_PICK_POS[2]:.4f})")


# ══════════════════════════════════════════════════════════════════════════
#  [C] 메인 파이프라인
# ══════════════════════════════════════════════════════════════════════════
def main():
    global BUSBAR_APPROACH_POS, BUSBAR_PICK_POS, BUSBAR_LIFT_MOVE_POS, SCAN_POS, HOME_EE_POS, BUSBAR_SCAN_POS, BATTERY_CENTER_POS, target_fine_pos

    usd_file_path = Path(USD_PATH).resolve()
    if not usd_file_path.is_file():
        raise FileNotFoundError(f"[ERROR] USD 파일을 찾을 수 없습니다: {usd_file_path}")

    ctx = omni.usd.get_context()
    ctx.open_stage(str(usd_file_path))
    for _ in range(15):
        simulation_app.update()

    stage = ctx.get_stage()
    if not stage:
        raise RuntimeError(f"[ERROR] Stage를 로드하지 못했습니다: {usd_file_path}")

    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT)
    lock_amr_base(stage, NOVA_CARTER_ROOT)

    busbar_xform = SingleXFormPrim(BUSBAR_POLYSHAPE_PATH, name="busbar_poly") if stage.GetPrimAtPath(BUSBAR_POLYSHAPE_PATH).IsValid() else None
    init_busbar_pos, init_busbar_quat = busbar_xform.get_world_pose() if busbar_xform else (None, None)

    nut1_xform = SingleXFormPrim(NUT1_POLYSHAPE_PATH, name="nut1_poly") if stage.GetPrimAtPath(NUT1_POLYSHAPE_PATH).IsValid() else None
    nut2_xform = SingleXFormPrim(NUT2_POLYSHAPE_PATH, name="nut2_poly") if stage.GetPrimAtPath(NUT2_POLYSHAPE_PATH).IsValid() else None
    init_nut1_pos, init_nut1_quat = nut1_xform.get_world_pose() if nut1_xform else (None, None)
    init_nut2_pos, init_nut2_quat = nut2_xform.get_world_pose() if nut2_xform else (None, None)

    ee_path = find_prim_path(stage, M0609_PATH, EE_LINK_NAME)
    gripper = ParallelGripper(
            end_effector_prim_path=ee_path,
            joint_prim_names=GRIPPER_JOINTS,
            joint_opened_positions=GRIPPER_OPEN,
            joint_closed_positions=GRIPPER_CLOSE,
            action_deltas=GRIPPER_DELTA,
        )
    
    robot = world.scene.add(SingleManipulator(
        prim_path=NOVA_CARTER_ROOT, 
        name="mobile_manipulator",
        end_effector_prim_path=ee_path,
        gripper=gripper,
    ))

    world.reset()
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )

    for _ in range(30):
        world.step(render=True)

    quat_busbar = euler_to_quaternion_wxyz(0.0, 3.1415, 1.5708)
    quat_nut = euler_to_quaternion_wxyz(0.0, 3.1415, 0.0)

    rclpy.init()
    isaac_node = Execute_Isaac_Busar()

    arm_controller = RMPFlowController(
        name="m0609_busbar_controller",
        robot_articulation=robot,
        urdf_path=URDF_PATH,
        robot_description_path=ROBOT_DESC_PATH,
        rmpflow_config_path=RMPFLOW_CFG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
    )

    base_link_xf = world_xf(stage, f"{M0609_PATH}/base_link")
    base_pos = base_link_xf.ExtractTranslation()
    base_quat = base_link_xf.ExtractRotationQuat()
    arm_controller._motion_policy.set_robot_base_pose(
        robot_position=np.array([base_pos[0], base_pos[1], base_pos[2]]),
        robot_orientation=np.array([base_quat.GetReal(), *[float(x) for x in base_quat.GetImaginary()]]),
    )

    print("\nIsaac Sim 준비 완료 - BehaviorNode 명령을 대기합니다.")

    step_count = 0
    grasp_timer = 0
    was_playing = False
    phase = "IDLE"
    busbar_start_grasp_pos = None
    descend_target_z = None
    target_mid_pos = None

    # ── 너트 조립(Nut Assembly) 상태 변수 ──
    nut_index      = 0        # 1: 너트 1번, 2: 너트 2번
    NUT_SCAN_POS   = None     # 너트 스캔 위치 (최초 1회 계산 후 재사용)
    nut_pick_pos   = None     # 현재 너트의 물리 파지 좌표
    nut_approach_pos = None   # 현재 너트 파지 상공 접근 좌표
    bolt_target_pos  = None   # 체결 목표 좌표
    bolt_touch_pos   = None   # 착좌(Screwing 시작) 목표 좌표

    screw_sub          = "rotate"
    screw_pass_idx      = 0
    screw_pass_theta    = 0.0
    screw_seat_ee_pos   = None
    screw_start_quat    = None
    screw_pass_end_pos  = None
    screw_release_step  = 0
    screw_regrasp_step  = 0
    screw_unwind_deg    = 0.0
    prev_ee_z           = 0.0
    stuck_counter       = 0

    # ── AMR 이동 상태 (amr_node <-> /amr/goal_pose, /amr/sim_pose) ──
    amr_moving = False
    amr_target_xy_theta = None
    wheels_locked = True  # main() 시작 시 lock_amr_base() 이미 호출됨

    # ── 너트 AMR 고정(Glue) 상태 ──
    # PICK_NUT1/2로 실제 파지되기 전까지는 AMR 위 고정 오프셋을 계속 따라가도록
    # 매 프레임 강제로 포즈를 갱신한다(물리 마찰에 의존 X). AMR이 set_world_pose로
    # 직접 텔레포트 이동하기 때문에(바퀴 조인트 저항 없음), 얹혀있기만 한 너트는
    # 접촉 마찰만으로는 AMR 이동/회전을 따라가지 못하고 떨어질 수 있어서 필요하다.
    nut1_local_offset = None
    nut2_local_offset = None
    nut1_released = False
    nut2_released = False

    def publish_status(status_str: str):
        msg = String()
        msg.data = status_str
        isaac_node.pub_status.publish(msg)

    def publish_progress(phase_name: str, pct: float):
        p_msg = String()
        p_msg.data = phase_name
        isaac_node.pub_phase.publish(p_msg)

        prog_msg = Float32()
        prog_msg.data = pct
        isaac_node.pub_progress.publish(prog_msg)

    while simulation_app.is_running():
        world.step(render=True)
        rclpy.spin_once(isaac_node, timeout_sec=0.0)
        playing = world.is_playing()

        # 1. Play / Stop 상태 보정
        if playing and not was_playing:
            world.reset()
            enable_physics_recursively(stage, BUSBAR_ROOT_PATH)
            if busbar_xform and init_busbar_pos is not None:
                busbar_xform.set_world_pose(position=init_busbar_pos, orientation=init_busbar_quat)
            if nut1_xform and init_nut1_pos is not None:
                nut1_xform.set_world_pose(position=init_nut1_pos, orientation=init_nut1_quat)
            if nut2_xform and init_nut2_pos is not None:
                nut2_xform.set_world_pose(position=init_nut2_pos, orientation=init_nut2_quat)

            # 너트는 PICK 전까지 물리를 끄고 AMR에 Kinematic Glue로 고정한다
            # (얹혀있기만 하면 AMR set_world_pose 텔레포트 이동 시 마찰만으로
            #  따라가지 못하고 떨어진다 -> 로컬 오프셋을 계산해 매 프레임 강제 추종).
            disable_physics_recursively(stage, NUT1_ROOT_PATH)
            disable_physics_recursively(stage, NUT2_ROOT_PATH)
            amr_pos0, amr_quat0 = robot.get_world_pose()
            nut1_local_offset = (
                compute_local_offset(amr_pos0, amr_quat0, init_nut1_pos, init_nut1_quat)
                if nut1_xform and init_nut1_pos is not None else None)
            nut2_local_offset = (
                compute_local_offset(amr_pos0, amr_quat0, init_nut2_pos, init_nut2_quat)
                if nut2_xform and init_nut2_pos is not None else None)
            nut1_released = False
            nut2_released = False

            step_count = 0
            grasp_timer = 0

            nut_index = 0
            NUT_SCAN_POS = None
            nut_pick_pos = None
            nut_approach_pos = None
            bolt_target_pos = None
            bolt_touch_pos = None
            screw_sub = "rotate"
            screw_pass_idx = 0
            screw_pass_theta = 0.0
            stuck_counter = 0

            amr_moving = False
            amr_target_xy_theta = None

        # 1.4. 너트 AMR 고정(Glue) 추종 - PICK 되기 전까지 매 프레임 강제 갱신
        if playing:
            amr_cur_pos, amr_cur_quat = robot.get_world_pose()
            if not nut1_released and nut1_xform is not None and nut1_local_offset is not None:
                glued_pos, glued_quat = compose_world_pose(amr_cur_pos, amr_cur_quat, *nut1_local_offset)
                nut1_xform.set_world_pose(position=glued_pos, orientation=glued_quat)
            if not nut2_released and nut2_xform is not None and nut2_local_offset is not None:
                glued_pos, glued_quat = compose_world_pose(amr_cur_pos, amr_cur_quat, *nut2_local_offset)
                nut2_xform.set_world_pose(position=glued_pos, orientation=glued_quat)

        # 1.5. AMR 이동 처리 (behavior_node -> amr_node -> /amr/goal_pose)
        #      팔 작업 중에는 바퀴를 잠그고(lock), AMR 이동 중에는 풀어준다(unlock).
        if playing:
            if isaac_node.amr_cancel_requested:
                isaac_node.amr_cancel_requested = False
                if amr_moving:
                    print("\n[AMR] 이동 취소 수신 -> 정지 및 바퀴 잠금")
                amr_moving = False
                amr_target_xy_theta = None
                if not wheels_locked:
                    lock_amr_base(stage, NOVA_CARTER_ROOT)
                    wheels_locked = True

            if isaac_node.amr_goal_pose is not None:
                goal_msg = isaac_node.amr_goal_pose
                isaac_node.amr_goal_pose = None
                g_pos = goal_msg.pose.position
                g_ori = goal_msg.pose.orientation
                g_theta = quat_wxyz_to_yaw([g_ori.w, g_ori.x, g_ori.y, g_ori.z])
                amr_target_xy_theta = (g_pos.x, g_pos.y, g_theta)
                amr_moving = True
                if wheels_locked:
                    unlock_amr_base(stage, NOVA_CARTER_ROOT)
                    wheels_locked = False
                print(f"\n>>> [AMR] 이동 목표 수신 (X={g_pos.x:.4f}, Y={g_pos.y:.4f}, "
                      f"Theta={g_theta:.4f}) -> 바퀴 잠금 해제, 이동 시작")

            amr_pos, amr_quat = robot.get_world_pose()
            amr_yaw = quat_wxyz_to_yaw(amr_quat)

            if amr_moving and amr_target_xy_theta is not None:
                tx, ty, ttheta = amr_target_xy_theta
                dx, dy = tx - amr_pos[0], ty - amr_pos[1]
                dist = math.hypot(dx, dy)
                dyaw = math.atan2(math.sin(ttheta - amr_yaw), math.cos(ttheta - amr_yaw))

                if dist > 1e-6:
                    step_dist = min(dist, AMR_LINEAR_SPEED * PHYSICS_DT)
                    new_x = amr_pos[0] + dx / dist * step_dist
                    new_y = amr_pos[1] + dy / dist * step_dist
                else:
                    new_x, new_y = amr_pos[0], amr_pos[1]

                step_yaw = max(-AMR_ANGULAR_SPEED * PHYSICS_DT, min(AMR_ANGULAR_SPEED * PHYSICS_DT, dyaw))
                new_yaw = amr_yaw + step_yaw

                robot.set_world_pose(
                    position=np.array([new_x, new_y, amr_pos[2]]),
                    orientation=euler_to_quaternion_wxyz(0.0, 0.0, new_yaw),
                )
                amr_pos = np.array([new_x, new_y, amr_pos[2]])
                amr_yaw = new_yaw

                if dist < AMR_POS_TOL and abs(dyaw) < AMR_YAW_TOL:
                    amr_moving = False
                    amr_target_xy_theta = None
                    print(f"\n[AMR] 목표 지점 도착 (X={new_x:.4f}, Y={new_y:.4f}) -> 바퀴 잠금")
                    lock_amr_base(stage, NOVA_CARTER_ROOT)
                    wheels_locked = True

                    # AMR이 이동했으므로 팔 IK가 가정하는 베이스 위치도 갱신해야 한다.
                    base_link_xf = world_xf(stage, f"{M0609_PATH}/base_link")
                    base_pos = base_link_xf.ExtractTranslation()
                    base_quat = base_link_xf.ExtractRotationQuat()
                    arm_controller._motion_policy.set_robot_base_pose(
                        robot_position=np.array([base_pos[0], base_pos[1], base_pos[2]]),
                        robot_orientation=np.array(
                            [base_quat.GetReal(), *[float(v) for v in base_quat.GetImaginary()]]
                        ),
                    )

            sim_pose_msg = Pose2D()
            sim_pose_msg.x = float(amr_pos[0])
            sim_pose_msg.y = float(amr_pos[1])
            sim_pose_msg.theta = float(amr_yaw)
            isaac_node.pub_amr_sim_pose.publish(sim_pose_msg)

        # 2. BehaviorNode / ArmNode 명령 분기 처리
        if playing and isaac_node.requested_task:
            task = isaac_node.requested_task
            isaac_node.requested_task = None

            # 안전장치: 팔 Task는 AMR이 도착해 바퀴가 잠긴 상태에서만 수행되어야 한다.
            if not wheels_locked:
                print(f"\n[WARN] [{task}] 바퀴가 아직 안 잠긴 상태 -> 강제 잠금 후 진행")
                lock_amr_base(stage, NOVA_CARTER_ROOT)
                wheels_locked = True
                amr_moving = False
                amr_target_xy_theta = None

            if task == "SCAN_BATTERY":
                phase = "INIT_POSE"
                step_count = 0
                print(f"\n>>> [{task}] 배터리 스캔 요청 수신 -> 1) 초기 관절 정렬 시작")

            elif task == "SCAN_BUSBAR":
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                BUSBAR_SCAN_POS = np.array([cur_pos[0] - 0.5, cur_pos[1] + 0.5, BUSBAR_SCAN_Z])
                phase = "SCAN_BUSBAR_APPROACH"
                step_count = 0
                print(f"\n>>> [{task}] 버스바 스캔 위치 이동 시작 (Target: X={BUSBAR_SCAN_POS[0]:.3f}, Y={BUSBAR_SCAN_POS[1]:.3f}, Z={BUSBAR_SCAN_POS[2]:.3f})")

            elif task == "PICK_BUSBAR":
                if isaac_node.latest_target_pose is not None:
                    update_target_positions(isaac_node.latest_target_pose)
                phase = "BUSBAR_APPROACH"
                step_count = 0
                print(f"\n>>> [{task}] 버스바 상공 접근(Z=0.6m) 시작")

            elif task == "MOVE_BATTERY_CENTER":
                if isaac_node.latest_target_pose is not None:
                    pos = isaac_node.latest_target_pose.pose.position
                    BATTERY_CENTER_POS = np.array([pos.x, pos.y, BATTERY_CENTER_Z])
                    phase = "MOVE_BATTERY_CENTER_APPROACH"
                    step_count = 0
                    print(f"\n>>> [{task}] 배터리 볼트 중점 상공 이동 시작 (Target: X={BATTERY_CENTER_POS[0]:.3f}, Y={BATTERY_CENTER_POS[1]:.3f}, Z={BATTERY_CENTER_POS[2]:.3f})")
                else:
                    print(f"\n[ERROR] [{task}] 수신된 Target Pose가 없습니다.")
                    publish_status("FAILURE:NO_TARGET_POSE")

            elif task == "FINE_ALIGNMENT":
                print("\n>>> [{task}] 비전 정밀 오차 보정 시작")
                print(" -> 🚀 오차 보정 비전 노드에 시작 트리거 전송 [START_ERRORFIX_CORRECTION]")
                
                start_cmd = String()
                start_cmd.data = "START_ERRORFIX_CORRECTION"
                isaac_node.pub_errorfix_command.publish(start_cmd)

                isaac_node.alignment_success = False
                
                cur_ee = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                target_fine_pos = np.array([cur_ee[0], cur_ee[1], BATTERY_CENTER_Z])
                phase = "FINE_ALIGNMENT"
                step_count = 0

            elif task == "ASSEMBLE_BUSBAR":
                cur_ee = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                if target_fine_pos is not None:
                    target_mid_pos = np.array([target_fine_pos[0], target_fine_pos[1], 0.0693])
                else:
                    target_mid_pos = np.array([cur_ee[0], cur_ee[1], 0.0693])

                descend_target_z = cur_ee[2]
                phase = "BUSBAR_DESCEND_TO_BOLT"
                step_count = 0
                print(f"\n>>> [{task}] 버스바 수직 하강 안착 시작 (Target Mid Pos: X={target_mid_pos[0]:.4f}, Y={target_mid_pos[1]:.4f})")

            elif task in ("SCAN_NUT1", "SCAN_NUT2"):
                nut_index = 1 if task == "SCAN_NUT1" else 2
                if NUT_SCAN_POS is None:
                    cur_ee = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                    NUT_SCAN_POS = np.array([cur_ee[0] + NUT_SCAN_OFFSET_X, cur_ee[1] + NUT_SCAN_OFFSET_Y, NUT_SCAN_Z])
                phase = "NUT_SCAN_APPROACH"
                step_count = 0
                print(f"\n>>> [{task}] 너트 스캔 위치 이동 시작 (Target: X={NUT_SCAN_POS[0]:.3f}, Y={NUT_SCAN_POS[1]:.3f}, Z={NUT_SCAN_POS[2]:.3f})")

            elif task in ("PICK_NUT1", "PICK_NUT2"):
                nut_index = 1 if task == "PICK_NUT1" else 2
                # 실제 파지 시작 -> AMR Glue 해제하고 정상 다이나믹 바디로 전환
                # (이후 그리퍼의 실제 접촉/마찰 기반 파지 물리는 기존 로직 그대로).
                if nut_index == 1:
                    nut1_released = True
                    enable_physics_recursively(stage, NUT1_ROOT_PATH)
                else:
                    nut2_released = True
                    enable_physics_recursively(stage, NUT2_ROOT_PATH)
                if HOME_EE_POS is not None:
                    nut_offset = NUT1_OFFSET_FROM_HOME if nut_index == 1 else NUT2_OFFSET_FROM_HOME
                    pick_x = HOME_EE_POS[0] + nut_offset[0]
                    pick_y = HOME_EE_POS[1] + nut_offset[1]
                    nut_pick_pos = np.array([pick_x, pick_y, NUT_PICK_Z])
                    nut_approach_pos = np.array([pick_x, pick_y, NUT_APPROACH_Z])
                    phase = "NUT_APPROACH"
                    step_count = 0
                    print(f"\n>>> [{task}] 너트 {nut_index}번 상공 접근 시작 (Target: X={nut_approach_pos[0]:.3f}, Y={nut_approach_pos[1]:.3f})")
                else:
                    print(f"\n[ERROR] [{task}] 초기 위치(HOME_EE_POS)가 없습니다. 먼저 INIT_POSE가 수행되어야 합니다.")
                    publish_status("FAILURE:NO_HOME_POSE")

            elif task in ("ASSEMBLE_NUT1", "ASSEMBLE_NUT2"):
                nut_index = 1 if task == "ASSEMBLE_NUT1" else 2
                bolt_world_xy = BOLT1_WORLD_POS if nut_index == 1 else BOLT2_WORLD_POS
                bolt_target_pos = np.array([bolt_world_xy[0], bolt_world_xy[1], 0.0])
                bolt_touch_pos = np.array([bolt_target_pos[0], bolt_target_pos[1], 0.3697])
                phase = "MOVE_TO_BOLT_NUT"
                step_count = 0
                print(f"\n>>> [{task}] 너트 {nut_index}번 -> 볼트 {nut_index}번 체결 시작 [하드코딩 월드 좌표] "
                      f"(Target: X={bolt_target_pos[0]:.4f}, Y={bolt_target_pos[1]:.4f})")

        # 3. FSM 제어 루프
        if playing and phase != "IDLE" and phase != "DONE":

            # [STEP 0-A] 초기 관절 자세 정렬
            if phase == "INIT_POSE":
                publish_progress("INIT_ALIGN", 10.0)
                
                arm_joint_names = [
                    "joint_1", "joint_2", "joint_3", 
                    "joint_4", "joint_5", "joint_6"
                ]
                arm_dof_indices = [robot.get_dof_index(name) for name in arm_joint_names]

                robot.apply_action(
                    ArticulationAction(
                        joint_positions=TARGET_INIT_JOINTS,
                        joint_indices=arm_dof_indices
                    )
                )
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                all_joints = robot.get_joint_positions()
                cur_arm_joints = all_joints[arm_dof_indices]
                
                joint_err = np.linalg.norm(cur_arm_joints - TARGET_INIT_JOINTS)

                if joint_err < JOINT_TOLERANCE or step_count > 60:
                    HOME_EE_POS = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                    SCAN_POS = np.array([HOME_EE_POS[0], HOME_EE_POS[1], SCAN_APPROACH_Z])
                    
                    print(f"[OK] 1) 관절 정렬 완료! (EE Pos: {HOME_EE_POS[0]:.3f}, {HOME_EE_POS[1]:.3f}, {HOME_EE_POS[2]:.3f})")
                    print(f" -> 2) 초기 위치 기준 Z={SCAN_APPROACH_Z}m 상승 시작 (Target: {SCAN_POS[0]:.3f}, {SCAN_POS[1]:.3f}, {SCAN_POS[2]:.3f})")
                    
                    phase = "SCAN_APPROACH"
                    step_count = 0

            # [STEP 0-B] 배터리 스캔 고도 상승 (Z = 0.7m)
            elif phase == "SCAN_APPROACH":
                publish_progress("SCAN_NAV", 50.0)
                actions = arm_controller.forward(
                    target_end_effector_position=SCAN_POS,
                    target_end_effector_orientation=euler_to_quaternion_wxyz(0.0, 3.1415, -1.5708)
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(SCAN_POS))

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 배터리 스캔 위치 도착 완료! ({cur_pos[0]:.3f}, {cur_pos[1]:.3f}, {cur_pos[2]:.3f})")
                    publish_progress("SCAN_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"

            # [STEP 1] 버스바 스캔 위치 이동
            elif phase == "SCAN_BUSBAR_APPROACH":
                publish_progress("SCAN_BUSBAR_NAV", 50.0)
                actions = arm_controller.forward(
                    target_end_effector_position=BUSBAR_SCAN_POS,
                    target_end_effector_orientation=quat_busbar
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(BUSBAR_SCAN_POS))

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 버스바 스캔 위치 도착 완료! ({cur_pos[0]:.3f}, {cur_pos[1]:.3f}, {cur_pos[2]:.3f})")
                    publish_progress("SCAN_BUSBAR_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"

            # [2단계] 버스바 상공 위치 접근 (Z = 0.6m)
            elif phase == "BUSBAR_APPROACH":
                publish_progress("APPROACH", 20.0)
                actions = arm_controller.forward(
                    target_end_effector_position=BUSBAR_APPROACH_POS,
                    target_end_effector_orientation=quat_busbar
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(BUSBAR_APPROACH_POS))

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 1. 버스바 상공(Z={BUSBAR_APPROACH_POS[2]:.3f}m) 접근 완료 -> 2. 파지점(Z={BUSBAR_PICK_POS[2]:.3f}m) 수직 하강 시작")
                    phase = "BUSBAR_DESCEND"
                    step_count = 0

            # [3단계] 파지 위치 수직 하강 (Z = 0.455m)
            elif phase == "BUSBAR_DESCEND":
                publish_progress("DESCEND", 50.0)
                actions = arm_controller.forward(
                    target_end_effector_position=BUSBAR_PICK_POS,
                    target_end_effector_orientation=quat_busbar
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(BUSBAR_PICK_POS))

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print("[OK] 2. 버스바 파지점 도착 -> Kinematic 파지 시작")
                    phase = "BUSBAR_GRASP"
                    grasp_timer = 0

            # [4단계] Kinematic Pose-Glue 파지
            elif phase == "BUSBAR_GRASP":
                publish_progress("GRASPING", 75.0)
                if grasp_timer == 0:
                    disable_physics_recursively(stage, BUSBAR_ROOT_PATH)
                    if busbar_xform is not None:
                        real_pos, _ = busbar_xform.get_world_pose()
                        busbar_start_grasp_pos = np.array(real_pos)
                    else:
                        busbar_start_grasp_pos = BUSBAR_PICK_POS

                actions = arm_controller.forward(
                    target_end_effector_position=BUSBAR_PICK_POS,
                    target_end_effector_orientation=quat_busbar
                )
                robot.apply_action(actions)

                grasp_timer += 1
                ramp_frac = min(grasp_timer / GRIP_CLOSE_RAMP_STEPS, 1.0)
                grip_target = ramp_frac * GRIPPER_CLOSE
                robot.gripper.apply_action(ArticulationAction(joint_positions=grip_target))

                glue_busbar_to_ee(robot, busbar_xform, busbar_start_grasp_pos, blend=ramp_frac)

                if grasp_timer >= GRIP_CLOSE_RAMP_STEPS:
                    print("[OK] 3. 버스바 파지 완료 -> 4. 안전 고도 상승")
                    phase = "BUSBAR_LIFT"
                    step_count = 0

            # [5단계] 안전 고도 들어올리기 (Z = 0.6m)
            elif phase == "BUSBAR_LIFT":
                publish_progress("LIFTING", 90.0)
                actions = arm_controller.forward(
                    target_end_effector_position=BUSBAR_LIFT_MOVE_POS,
                    target_end_effector_orientation=quat_busbar
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))

                glue_busbar_to_ee(robot, busbar_xform, busbar_start_grasp_pos, blend=1.0)

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(BUSBAR_LIFT_MOVE_POS))

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print("\n★ [PICK_BUSBAR SUCCESS] 버스바 파지 및 상승 완수!")
                    publish_progress("COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"

            # [6단계] 배터리 중점 상공으로 이동 (Z = 0.7m, 버스바 고정 상태 유지)
            elif phase == "MOVE_BATTERY_CENTER_APPROACH":
                publish_progress("MOVING_BATTERY_CENTER", 50.0)
                actions = arm_controller.forward(
                    target_end_effector_position=BATTERY_CENTER_POS,
                    target_end_effector_orientation=euler_to_quaternion_wxyz(0.0, 3.1415, 0.0)
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))
                glue_busbar_to_ee(robot, busbar_xform, busbar_start_grasp_pos, blend=1.0)

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(BATTERY_CENTER_POS))

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 배터리 중점 상공 위치 도착 완료! ({cur_pos[0]:.3f}, {cur_pos[1]:.3f}, {cur_pos[2]:.3f})")
                    publish_progress("MOVE_BATTERY_CENTER_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"

            # [7단계] 비전 보정 노드 피드백 기반 미세 오차 정렬
            elif phase == "FINE_ALIGNMENT":
                publish_progress("FINE_ALIGNMENT_TRACKING", 85.0)

                cur_ee_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()

                if isaac_node.latest_target_pose is not None:
                    offset = isaac_node.latest_target_pose.pose.position
                    if abs(offset.x) <= 0.0025 and abs(offset.y) <= 0.0025:             
                        target_fine_pos[0] += offset.x
                        target_fine_pos[1] += offset.y
                        target_fine_pos[2] = BATTERY_CENTER_Z
                        
                        print(f"\n[FINE_ALIGNMENT] Vision Offset Received -> dx: {offset.x:+.4f}m, dy: {offset.y:+.4f}m")
                        print(f"               └─ New Target Pos -> X: {target_fine_pos[0]:.4f}, Y: {target_fine_pos[1]:.4f}, Z: {target_fine_pos[2]:.4f}")

                    isaac_node.latest_target_pose = None

                sys.stdout.write(
                    f"\r\033[K[FINE_ALIGNMENT Loop] Cur EE: ({cur_ee_pos[0]:.4f}, {cur_ee_pos[1]:.4f}) "
                    f"-> Target: ({target_fine_pos[0]:.4f}, {target_fine_pos[1]:.4f})"
                )
                sys.stdout.flush()

                actions = arm_controller.forward(
                    target_end_effector_position=target_fine_pos,
                    target_end_effector_orientation=euler_to_quaternion_wxyz(0.0, 3.1415, 0.0)
                )
                robot.apply_action(actions)

                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))
                glue_busbar_to_ee(robot, busbar_xform, busbar_start_grasp_pos, blend=1.0)

                if isaac_node.alignment_success:
                    print("\n\n★ [FINE_ALIGNMENT SUCCESS] 미세 오차 보정 성공 및 정렬 완료!")
                    publish_progress("FINE_ALIGNMENT_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"

            # [8단계] 버스바 수직 하강 체결
            elif phase == "BUSBAR_DESCEND_TO_BOLT":
                publish_progress("BUSBAR_DESCEND_INSERT", 90.0)

                descend_target_z = max(descend_target_z - INSERT_SPEED, target_mid_pos[2])
                step_target_pos = np.array([target_mid_pos[0], target_mid_pos[1], descend_target_z])

                actions = arm_controller.forward(
                    target_end_effector_position=step_target_pos,
                    target_end_effector_orientation=euler_to_quaternion_wxyz(0.0, 3.1415, 0.0)
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))

                glue_busbar_to_ee(robot, busbar_xform, busbar_start_grasp_pos, blend=1.0)

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                dist_err = math.dist(cur_pos, tuple(target_mid_pos))

                if cur_pos[2] <= BUSBAR_RELEASE_Z or dist_err < INSERT_TOLERANCE_STRICT:
                    if busbar_xform is not None:
                        busbar_xform.set_world_pose(position=target_mid_pos, orientation=BUSBAR_REST_ORIENTATION)
                    
                    print(f"\n[OK] 버스바 안착 체결 완료 (EE Z: {cur_pos[2]:.4f}m)!")

                    phase = "BUSBAR_RELEASE_AND_RETRACT"
                    step_count = 0

            # [9단계] 버스바 해제 및 안전 상공 이탈
            elif phase == "BUSBAR_RELEASE_AND_RETRACT":
                publish_progress("BUSBAR_RETRACT", 95.0)

                retract_pos = np.array([target_mid_pos[0], target_mid_pos[1], BATTERY_CENTER_Z])
                actions = arm_controller.forward(
                    target_end_effector_position=retract_pos,
                    target_end_effector_orientation=euler_to_quaternion_wxyz(0.0, 3.1415, 0.0)
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                glue_busbar_to_ee(robot, busbar_xform, busbar_start_grasp_pos, blend=0.0)

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(retract_pos))

                if current_err < PICK_TOLERANCE_STRICT or step_count > MAX_STUCK_STEPS:
                    print("\n★ [ASSEMBLE_BUSBAR SUCCESS] 버스바 체결 및 안전 이탈 완료!")
                    publish_progress("ASSEMBLE_BUSBAR_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"

            # ════════════════════════════════════════════════════════════════
            # 너트 조립(Nut Assembly) 공용 Phase
            # ════════════════════════════════════════════════════════════════
            # [10단계] 너트 스캔 위치 이동
            elif phase == "NUT_SCAN_APPROACH":
                publish_progress("NUT_SCAN_NAV", 50.0)
                actions = arm_controller.forward(target_end_effector_position=NUT_SCAN_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(NUT_SCAN_POS))

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 너트 스캔 위치 도착 완료! ({cur_pos[0]:.3f}, {cur_pos[1]:.3f}, {cur_pos[2]:.3f})")
                    publish_progress("NUT_SCAN_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"

            # [11단계] 너트 상공 접근
            elif phase == "NUT_APPROACH":
                publish_progress("NUT_APPROACH", 20.0)
                actions = arm_controller.forward(target_end_effector_position=nut_approach_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(nut_approach_pos))

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    nut_label = resolve_nut_assets(nut_index, nut1_xform, nut2_xform)[1]
                    print(f"[OK] {nut_label} 상공 도착! -> 하강 시작")
                    phase = "NUT_DESCEND"
                    step_count = 0

            # [12단계] 너트 파지점 하강
            elif phase == "NUT_DESCEND":
                publish_progress("NUT_DESCEND", 40.0)
                actions = arm_controller.forward(target_end_effector_position=nut_pick_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(nut_pick_pos))

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    nut_label = resolve_nut_assets(nut_index, nut1_xform, nut2_xform)[1]
                    print(f"[OK] {nut_label} 하강 완료! -> 물리 파지(Gripper Close) 시작")
                    phase = "NUT_GRASP"
                    grasp_timer = 0

            # [13단계] 너트 물리 파지
            elif phase == "NUT_GRASP":
                publish_progress("NUT_GRASPING", 60.0)
                actions = arm_controller.forward(target_end_effector_position=nut_pick_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)

                grasp_timer += 1
                ramp_frac = min(grasp_timer / GRIP_CLOSE_RAMP_STEPS, 1.0)
                grip_target = ramp_frac * GRIPPER_CLOSE_NUT
                robot.gripper.apply_action(ArticulationAction(joint_positions=grip_target))

                if grasp_timer >= GRIP_CLOSE_RAMP_STEPS:
                    nut_label = resolve_nut_assets(nut_index, nut1_xform, nut2_xform)[1]
                    print(f"[OK] {nut_label} 물리 파지 완료! -> 상공({NUT_APPROACH_Z}m)으로 상승")
                    phase = "NUT_LIFT"
                    step_count = 0

            # [14단계] 너트 파지 후 상공 상승
            elif phase == "NUT_LIFT":
                publish_progress("NUT_LIFTING", 80.0)
                actions = arm_controller.forward(target_end_effector_position=nut_approach_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(nut_approach_pos))

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    nut_label = resolve_nut_assets(nut_index, nut1_xform, nut2_xform)[1]
                    print(f"\n★ [PICK_{nut_label} SUCCESS] 너트 파지 및 상승 완수!")
                    publish_progress("NUT_PICK_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"

            # [15단계] 볼트 상공 이동
            elif phase == "MOVE_TO_BOLT_NUT":
                publish_progress("MOVE_TO_BOLT", 20.0)
                bolt_approach_pos = np.array([bolt_target_pos[0], bolt_target_pos[1], BOLT_APPROACH_Z])
                actions = arm_controller.forward(target_end_effector_position=bolt_approach_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(bolt_approach_pos))

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    nut_label = resolve_nut_assets(nut_index, nut1_xform, nut2_xform)[1]
                    print(f"[OK] 볼트 {nut_index}번 상공 도착! ({nut_label}) -> 착좌 하강 시작")
                    phase = "NUT_DESCEND_TO_BOLT"
                    step_count = 0
                    descend_target_z = BOLT_APPROACH_Z

            # [16단계] 볼트 착좌 하강
            elif phase == "NUT_DESCEND_TO_BOLT":
                publish_progress("NUT_DESCEND_TO_BOLT", 40.0)
                descend_target_z = max(descend_target_z - INSERT_SPEED, bolt_touch_pos[2])
                step_target_pos = np.array([bolt_touch_pos[0], bolt_touch_pos[1], descend_target_z])

                actions = arm_controller.forward(target_end_effector_position=step_target_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()

                if abs(cur_pos[2] - bolt_touch_pos[2]) < PICK_TOLERANCE_LOOSE_VAL or descend_target_z <= bolt_touch_pos[2]:
                    ee_now_pos, ee_now_quat = robot.end_effector.get_world_pose()
                    screw_start_quat = np.asarray(ee_now_quat).copy()
                    screw_seat_ee_pos = np.asarray(ee_now_pos).copy()

                    screw_sub = "rotate"
                    screw_pass_idx = 0
                    screw_pass_theta = 0.0
                    stuck_counter = 0
                    prev_ee_z = ee_now_pos[2]

                    nut_label = resolve_nut_assets(nut_index, nut1_xform, nut2_xform)[1]
                    print(f"[OK] 볼트 {nut_index}번 착좌 완료 ({nut_label})! -> Screwing 시작")
                    phase = "NUT_SCREW"
                    step_count = 0

            # [17단계] Screwing
            elif phase == "NUT_SCREW":
                publish_progress("NUT_SCREWING", 70.0)
                if screw_sub == "rotate":
                    screw_pass_theta = min(screw_pass_theta + SCREW_OMEGA_DEG_S * PHYSICS_DT, SCREW_TURNS_DEG)
                    pass_done = (screw_pass_theta >= SCREW_TURNS_DEG)

                    total_deg = screw_pass_idx * SCREW_TURNS_DEG + screw_pass_theta
                    depth_m = min((total_deg / 360.0) * NUT_PITCH_M, ENGAGE_LEN)

                    if screw_pass_idx > 0:
                        regrasp_extra = REGRASP_Z_OFFSET * (1.0 - screw_pass_theta / SCREW_TURNS_DEG)
                    else:
                        regrasp_extra = 0.0

                    target_pos = screw_seat_ee_pos.copy()
                    target_pos[2] = screw_seat_ee_pos[2] - depth_m + regrasp_extra
                    target_quat = yaw_rotated_quat(screw_start_quat, screw_pass_theta)

                    actions = arm_controller.forward(target_end_effector_position=target_pos, target_end_effector_orientation=target_quat)
                    robot.apply_action(actions)
                    robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                    # 실시간 토크 및 Stuck 감지
                    cur_ee_pos, _ = robot.end_effector.get_world_pose()
                    z_movement = abs(prev_ee_z - cur_ee_pos[2])
                    prev_ee_z = cur_ee_pos[2]

                    joint_efforts = robot.get_measured_joint_efforts()
                    curr_torque = abs(joint_efforts[-1]) if joint_efforts is not None and len(joint_efforts) > 0 else 0.0

                    if cur_ee_pos[2] <= TCP_FORCE_CHECK_Z:
                        if z_movement < STUCK_Z_DELTA_THRESH and depth_m > 0.003:
                            stuck_counter += 1
                        else:
                            stuck_counter = max(0, stuck_counter - 1)
                        is_seated_by_torque = (curr_torque > TORQUE_THRESHOLD) or (stuck_counter >= STUCK_STEP_LIMIT)
                    else:
                        is_seated_by_torque = False

                    if step_count % 20 == 0:
                        print(f"  [NUT{nut_index} SCREW] Pass {screw_pass_idx+1}/{1+REGRASP_CYCLES} | Theta: {screw_pass_theta:.1f}° | 깊이: {depth_m*1000:.2f}mm / 목표 {ENGAGE_LEN*1000:.1f}mm | TCP Z: {cur_ee_pos[2]:.4f}m | 토크: {curr_torque:.1f}Nm")

                    if pass_done or is_seated_by_torque:
                        if is_seated_by_torque:
                            print(f"  [체결 감지] 너트 {nut_index}번 완착(Seating) 감지! (TCP Z: {cur_ee_pos[2]:.4f}m, 토크: {curr_torque:.1f}Nm) -> Screwing 조기 종료 및 그리퍼 해제")

                        if depth_m >= ENGAGE_LEN or screw_pass_idx >= REGRASP_CYCLES or is_seated_by_torque:
                            robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
                            print(f"\n[OK] 너트 {nut_index}번 체결 완료! -> 꼬인 방향 유지한 채 수직 상승 시작")
                            phase = "NUT_RETRACT_LIFT"
                            step_count = 0
                        else:
                            screw_pass_end_pos = target_pos.copy()
                            screw_sub = "release"
                            screw_release_step = 0

                elif screw_sub == "release":
                    screw_release_step += 1
                    rf = min(screw_release_step / GRIP_CLOSE_RAMP_STEPS, 1.0)
                    release_target = (1.0 - rf) * GRIPPER_CLOSE_NUT[0]
                    robot.gripper.apply_action(ArticulationAction(joint_positions=np.array([release_target, release_target])))

                    hold_quat = yaw_rotated_quat(screw_start_quat, screw_pass_theta)
                    actions = arm_controller.forward(target_end_effector_position=screw_pass_end_pos, target_end_effector_orientation=hold_quat)
                    robot.apply_action(actions)

                    if rf >= 1.0:
                        screw_sub = "lift_up"
                        screw_release_step = 0

                elif screw_sub == "lift_up":
                    lift_target_pos = screw_pass_end_pos + np.array([0.0, 0.0, REGRASP_LIFT_HEIGHT])
                    hold_quat = yaw_rotated_quat(screw_start_quat, screw_pass_theta)
                    actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=hold_quat)
                    robot.apply_action(actions)
                    robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                    cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                    if math.dist(cur_pos, tuple(lift_target_pos)) < PICK_TOLERANCE_STRICT or screw_release_step > 40:
                        screw_sub = "unwind"
                        screw_unwind_deg = screw_pass_theta
                    screw_release_step += 1

                elif screw_sub == "unwind":
                    screw_unwind_deg = max(screw_unwind_deg - SCREW_OMEGA_DEG_S * PHYSICS_DT, 0.0)
                    unwind_done = (screw_unwind_deg <= 0.0)

                    lift_target_pos = screw_pass_end_pos + np.array([0.0, 0.0, REGRASP_LIFT_HEIGHT])
                    target_quat = yaw_rotated_quat(screw_start_quat, screw_unwind_deg)
                    actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=target_quat)
                    robot.apply_action(actions)
                    robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                    if unwind_done:
                        screw_sub = "descend_down"
                        screw_release_step = 0

                elif screw_sub == "descend_down":
                    regrasp_descend_target = screw_pass_end_pos + np.array([0.0, 0.0, REGRASP_Z_OFFSET])
                    actions = arm_controller.forward(target_end_effector_position=regrasp_descend_target, target_end_effector_orientation=screw_start_quat)
                    robot.apply_action(actions)
                    robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                    cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                    if math.dist(cur_pos, tuple(regrasp_descend_target)) < PICK_TOLERANCE_STRICT or screw_release_step > 40:
                        screw_sub = "regrasp"
                        screw_regrasp_step = 0
                    screw_release_step += 1

                elif screw_sub == "regrasp":
                    screw_regrasp_step += 1
                    rf = min(screw_regrasp_step / GRIP_CLOSE_RAMP_STEPS, 1.0)
                    grip_target = rf * GRIPPER_CLOSE_NUT[0]
                    robot.gripper.apply_action(ArticulationAction(joint_positions=np.array([grip_target, grip_target])))

                    regrasp_target_pos = screw_pass_end_pos + np.array([0.0, 0.0, REGRASP_Z_OFFSET])
                    actions = arm_controller.forward(target_end_effector_position=regrasp_target_pos, target_end_effector_orientation=screw_start_quat)
                    robot.apply_action(actions)

                    if rf >= 1.0:
                        screw_pass_idx += 1
                        screw_pass_theta = 0.0
                        stuck_counter = 0
                        screw_sub = "rotate"

            # [18단계] 체결 후 수직 상승 (꼬인 방향 유지)
            elif phase == "NUT_RETRACT_LIFT":
                publish_progress("NUT_RETRACT_LIFT", 90.0)
                ee_now_pos, _ = robot.end_effector.get_world_pose()
                lift_target_pos = np.array([ee_now_pos[0], ee_now_pos[1], NUT_APPROACH_Z])

                last_screw_quat = yaw_rotated_quat(screw_start_quat, screw_pass_theta)
                actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=last_screw_quat)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(lift_target_pos))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 수직 이탈 완료! -> 상공 안전 지대에서 6번 조인트 0도로 되감기(Unwind) 시작")
                    phase = "NUT_RETRACT_UNWIND"
                    screw_unwind_deg = screw_pass_theta
                    step_count = 0

            # [19단계] 상공에서 6번 조인트 0도로 되감기
            elif phase == "NUT_RETRACT_UNWIND":
                publish_progress("NUT_RETRACT_UNWIND", 95.0)
                screw_unwind_deg = max(screw_unwind_deg - SCREW_OMEGA_DEG_S * PHYSICS_DT, 0.0)
                ee_now_pos, _ = robot.end_effector.get_world_pose()
                lift_target_pos = np.array([ee_now_pos[0], ee_now_pos[1], NUT_APPROACH_Z])

                target_quat = yaw_rotated_quat(screw_start_quat, screw_unwind_deg)
                actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=target_quat)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                if screw_unwind_deg <= 0.0:
                    print(f"[OK] 6번 조인트 0도 원점 되감기 완료! -> 기본 방향(quat_nut) 정렬")
                    phase = "NUT_RETRACT_ROTATE"
                    step_count = 0

            # [20단계] 기본 오리엔테이션 정렬
            elif phase == "NUT_RETRACT_ROTATE":
                publish_progress("NUT_RETRACT_ROTATE", 98.0)
                ee_now_pos, _ = robot.end_effector.get_world_pose()
                lift_target_pos = np.array([ee_now_pos[0], ee_now_pos[1], NUT_APPROACH_Z])

                actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(lift_target_pos))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    nut_label = resolve_nut_assets(nut_index, nut1_xform, nut2_xform)[1]
                    print(f"\n[OK] {nut_label} 상공 이탈 및 회전 정렬 완료!")
                    
                    # 너트 2번 체결까지 전체 조립 작업이 끝난 경우 초기 관절각으로 완전 복귀
                    if nut_index == 2:
                        print(" -> 🚀 너트 2번 체결 완료. 초기 관절 각도 [0, 0, 90°, 0, 90°, 0] 복귀 시작...")
                        phase = "RETURN_HOME_JOINTS"
                        step_count = 0
                    else:
                        print(f"★ [ASSEMBLE_{nut_label} SUCCESS] 너트 1번 체결 완료!")
                        publish_progress("ASSEMBLE_NUT_COMPLETE", 100.0)
                        publish_status("SUCCESS")
                        phase = "IDLE"

            # [21단계 - 신규] 너트 2번 완료 후 초기 관절 각도복귀 [0, 0, 90°, 0, 90°, 0]
            elif phase == "RETURN_HOME_JOINTS":
                publish_progress("RETURNING_HOME_JOINTS", 99.0)

                arm_joint_names = [
                    "joint_1", "joint_2", "joint_3", 
                    "joint_4", "joint_5", "joint_6"
                ]
                arm_dof_indices = [robot.get_dof_index(name) for name in arm_joint_names]

                robot.apply_action(
                    ArticulationAction(
                        joint_positions=TARGET_INIT_JOINTS,
                        joint_indices=arm_dof_indices
                    )
                )
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                all_joints = robot.get_joint_positions()
                cur_arm_joints = all_joints[arm_dof_indices]
                joint_err = np.linalg.norm(cur_arm_joints - TARGET_INIT_JOINTS)

                if joint_err < JOINT_TOLERANCE or step_count > 120:
                    print("\n★ [ALL PROCESS COMPLETE & HOME RETURNED] 모든 너트 체결 및 초기 관절 각도 복귀 완수!")
                    publish_progress("ASSEMBLE_NUT_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"

            # Phase 타임아웃 예외 처리
            if step_count > MAX_STUCK_STEPS * 4:
                print(f"\n[ERROR] Phase '{phase}' 진행 시간 초과 - 실패 처리")
                publish_status("FAILURE:TIMEOUT")
                phase = "DONE"

            step_count += 1

        was_playing = playing

    # 리소스 해제
    if 'world' in locals() and world is not None:
        world.clear_instance()
    omni.usd.get_context().close_stage()
    isaac_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()
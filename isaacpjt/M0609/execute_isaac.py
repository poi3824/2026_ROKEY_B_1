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
7. PICK_NUT1 / PICK_NUT2   : 공급대 고정좌표(NUT1_PICK_XY/NUT2_PICK_XY)로 바로 접근,
   물리 파지(Gripper Close) 및 상승 (스캔/비전 없음 - assembly_nut_fraction1.py와 동일 방식)
8. ASSEMBLE_NUT1 / ASSEMBLE_NUT2 : 실시간 배터리 중심 좌표(target_mid_pos) 기준 사전 계산된
   상대 오프셋(+ get_bolt_pair 실측 보정)을 적용해 산출한 볼트 목표 좌표로 이동, 착좌 및
   Screwing 체결(Regrasp 포함)
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
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String, Float32

# RMPFlow Controller 경로 설정
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "rmpflow"))
from m0609_rmpflow_controller import RMPFlowController

# ══════════════════════════════════════════════════════════════════════════
#  [A] 설정 및 파라미터
# ══════════════════════════════════════════════════════════════════════════
USD_PATH = "/home/rokey/junhyeok_version/isaacpjt/M0609/Collected_Busbar/Busbar.usd"

NOVA_CARTER_ROOT = "/World/Nova_Carter/chassis_link"
M0609_PATH       = "/World/m0609"
EE_LINK_NAME     = "link_6"
GRIPPER_JOINTS   = ["finger_joint", "right_inner_knuckle_joint"]

BUSBAR_ROOT_PATH      = "/World/busbar"
BUSBAR_POLYSHAPE_PATH = "/World/busbar/geo/PolyShape"

# 너트 Prim 경로 (신규 추가)
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
#  너트 조립(Nut Assembly) 파라미터 (신규 추가)
# ══════════════════════════════════════════════════════════════════════════
NUT_HEIGHT         = 0.0095
NUT_GRASP_Z_LOCAL  = NUT_HEIGHT + 0.023
NUT_SUPPLY_TABLE_Z = 0.72                                   # 너트 공급대 높이
NUT_PICK_Z         = NUT_SUPPLY_TABLE_Z - (NUT_GRASP_Z_LOCAL - 0.0395)
NUT_APPROACH_Z     = 0.8                                     # 너트 파지 상공 고도
BOLT_APPROACH_Z    = 0.6                                     # 볼트 체결 상공 고도

# ★ 체결 착좌(bolt_touch_pos) 전용 높이 오프셋 - NUT_GRASP_Z_LOCAL은 너트 파지
# 깊이(NUT_PICK_Z)에도 같이 쓰이는 공용 상수라, 착좌 높이만 따로 튜닝하기 위해
# 분리했다. target_mid_pos[2](0.0693) + EE_OFFSET[2](0.185) + 이 값 = 목표 착좌
# 높이(EE Z) - 실측 기준 0.3695m가 되도록 역산한 값 (2026-07-27).
NUT_FASTEN_TOUCH_Z_LOCAL = 0.1152

# ★ 너트 1/2번 파지 XY (assembly_nut_fraction1.py의 NUT1_PICK_POS/NUT2_PICK_POS와 동일값) ★
# 그리퍼 장착 카메라(eye-in-hand)로 하강 중 실시간 추적을 시도했더니 접근할수록
# 목표가 표류해 완전히 미달하는 문제가 있었음(2026-07-27 실측) - 원인 규명 전까지는
# 공급대에 항상 같은 자리로 놓이는 너트 특성상 고정좌표가 더 안정적이라 되돌린다.
NUT1_PICK_XY = (0.5746, -0.1008)
NUT2_PICK_XY = (0.6643, -0.1031)

# ★ 볼트 1/2번 참고 좌표 (절대좌표로 직접 이동에 사용 금지!) ★
# 아래 두 좌표는 버스바 체결 시 사용되는 기준 배터리 중심 좌표(_REF_BATTERY_CENTER_POS) 대비
# 상대 오프셋(Offset)을 미리 계산해 두기 위한 참고값일 뿐이며, 실제 체결 시에는
# 실시간 배터리 중심 좌표(target_mid_pos)에 이 오프셋을 더해 동적으로 목표 좌표를 산출한다.
BOLT1_OFFSET_FROM_CENTER = np.array([-0.1042, 0.1812, 0.0])
BOLT2_OFFSET_FROM_CENTER = np.array([0.1042, -0.1812, 0.0])

# 너트 체결(Screwing) 파라미터
ENGAGE_LEN           = 0.0125     # 체결 깊이 (12.5mm)
SCREW_TURNS_DEG      = 350.0      # 1회전당 350도
REGRASP_CYCLES       = 1          # 총 2회전 (Regrasp 1회)
SCREW_OMEGA_DEG_S    = 120.0      # 초당 120도 회전
REGRASP_LIFT_HEIGHT  = 0.05       # Regrasp 시 수직 상승 높이 (5cm)
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
MAX_STUCK_STEPS          = 100000      # Phase 타임아웃 기준
PHYSICS_DT               = 1.0 / 60.0

# ★ 버스바 PI 정렬(assembly_nut_fraction1.py에서 이식) - /busbar_alignment_error
# 유효 시간, 이보다 오래된 값은 사용 안 함 (error_fix_depth1.py 미기동/끊김 대비)
BUSBAR_ALIGNMENT_MAX_AGE = 1.0

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

        self.pub_phase = self.create_publisher(String, '/isaac_phase', 10)
        self.pub_progress = self.create_publisher(Float32, '/isaac_progress', 10)
        self.pub_status = self.create_publisher(String, '/isaac_status', 10)
        self.pub_errorfix_command = self.create_publisher(String, '/errorfix_command', 10)

        # ★ assembly_nut_fraction1.py에서 끌어온 PI 정렬 로직용 - error_fix_depth1.py가
        # 퍼블리시하는 world dx,dy,dz[m] / dTheta[rad]를 계속 받아둔다 (기존 /target_pose+
        # /errorfix_command 경로는 error_fix_depth1.py가 안 쓰고 있어 실제로 연결이 안 돼있었음).
        self.busbar_align_dx = 0.0
        self.busbar_align_dy = 0.0
        self.busbar_align_dtheta = 0.0
        self.busbar_align_last_stamp = 0.0
        self.sub_busbar_alignment = self.create_subscription(
            Twist, '/busbar_alignment_error', self._on_busbar_alignment_error, 10
        )

    def _on_target_pose(self, msg: PoseStamped):
        self.latest_target_pose = msg

    def _on_task_command(self, msg: String):
        self.get_logger().info(f"[Task Command 수신]: {msg.data}")
        if msg.data == "ALIGNMENT_SUCCESS":
            self.alignment_success = True
        else:
            self.requested_task = msg.data

    def _on_busbar_alignment_error(self, msg: Twist):
        self.busbar_align_dx = msg.linear.x
        self.busbar_align_dy = msg.linear.y
        self.busbar_align_dtheta = msg.angular.z
        self.busbar_align_last_stamp = time.time()


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


class YawAligner:
    """버스바 yaw 정렬 오차(dTheta)를 0으로 미는 증분형(incremental) PI 보정기.
    assembly_nut_fraction1.py의 YawAligner와 동일 (SIMC 튜닝 근거도 동일하게 유지).

    이 문제는 "고정된 기하학적 오정렬(D) - 지금까지 명령한 누적 보정량(C)" = 잔여오차
    구조라, 매 스텝 오차를 새로 계산해서 명령을 통째로 덮어쓰는 표준 위치형(positional)
    PID는 맞지 않는다 - 위치형 P만으로는 0으로 수렴하지 않고, D항은 물리스텝 dt(1/60s)에서
    게인이 과도하게 커져 ±한계각에 튕기며 진동한다(assembly_nut_fraction1.py에서 실측됨).
    대신 증분형(velocity-form) PI를 쓴다: 매 "새" 비전 샘플마다
        C += Ki * error + Kp * (error - 직전 error)

    ★ 게인은 RMPFlow 실제 플랜트(목표 yaw 커맨드 -> EE 실제 yaw) 스텝응답 실측 기반
    SIMC 설계값 그대로 재사용 (τ≈0.283s, K≈1.0, λ=τ 보수적 선택 -> Kp=1.0, Ki≈0.0589).
    비전 왕복 지연시간(θ)을 실측 못 해 θ=0 가정, 안전 마진 큰 λ=τ 사용.
    """

    def __init__(self, kp, ki, out_limit_deg):
        self.kp = kp
        self.ki = ki
        self.out_limit_deg = out_limit_deg
        self.correction_deg = 0.0
        self.prev_error = None

    def reset(self):
        self.correction_deg = 0.0
        self.prev_error = None

    def update(self, error_deg):
        delta_error = 0.0 if self.prev_error is None else (error_deg - self.prev_error)
        self.correction_deg += self.ki * error_deg + self.kp * delta_error
        self.correction_deg = max(-self.out_limit_deg, min(self.out_limit_deg, self.correction_deg))
        self.prev_error = error_deg
        return self.correction_deg


def resolve_nut_assets(nut_index, nut1_xform, nut2_xform):
    """nut_index(1 또는 2)에 따라 대상 너트 Xform과 라벨을 반환 (Nut1/Nut2 공용 로직 재사용용)"""
    if nut_index == 1:
        return nut1_xform, "너트 1번"
    return nut2_xform, "너트 2번"


def compute_bolt_target_pos(nut_index, battery_center_pos):
    """사전 계산된 배터리 중심 기준 오프셋을 실시간 배터리 중심 좌표(Anchor)에 적용하여
    볼트 1/2번 체결 목표 3D 월드 좌표를 동적으로 산출한다 (절대좌표 하드코딩 금지)."""
    offset = BOLT1_OFFSET_FROM_CENTER if nut_index == 1 else BOLT2_OFFSET_FROM_CENTER
    return np.asarray(battery_center_pos) + offset


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

    print("=" * 60)
    print("  [PI_bus] execute_isaac.py 실행 중 (버스바 yaw PI 누적 + 고정좌표 너트 파지 버전)")
    print("=" * 60)

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

    # ★ 버스바 yaw 보정 누적기 - error_fix.py가 /target_pose로 보내는 미세 yaw 스텝을
    # 누적하는 용도 (assembly_nut_fraction1.py의 YawAligner 클래스를 accumulator로 재사용)
    yaw_aligner = YawAligner(kp=1.0, ki=0.0589, out_limit_deg=15.0)

    # ── 너트 조립(Nut Assembly) 상태 변수 (신규 추가) ──
    nut_index      = 0        # 1: 너트 1번, 2: 너트 2번 (NUT_*/MOVE_TO_BOLT_NUT 공용 Phase에서 참조)
    nut_pick_pos   = None      # NUT1_PICK_XY/NUT2_PICK_XY 기준 현재 너트의 물리 파지 좌표
    nut_approach_pos = None    # 현재 너트 파지 상공 접근 좌표
    bolt_target_pos  = None    # compute_bolt_target_pos()로 동적 산출된 체결 목표 좌표
    bolt_touch_pos   = None    # 착좌(Screwing 시작) 목표 좌표

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
        # ★ spin_once는 한 번 호출에 콜백을 1개만 처리하고 리턴한다 - error_fix.py가
        # 10Hz로 /target_pose를 계속 보내는데 한 틱에 여러 개가 몰리면 일부가 씹혀서
        # 유실될 수 있다(구독 콜백이 latest-value 덮어쓰기 방식이라 처리 전에 다음
        # 메시지가 오면 이전 건 사라짐 - yaw는 스텝이 0.01°로 작아서 이게 누적되면
        # 최종 각도가 몇 도 부족해질 수 있음, 2026-07-27 실측 2도 잔차 확인).
        # 큐가 빌 때까지 반복 처리해서 유실을 없앤다.
        for _ in range(20):
            rclpy.spin_once(isaac_node, timeout_sec=0.0)
        playing = world.is_playing()

        # 1. Play / Stop 상태 보정
        if playing and not was_playing:
            world.reset()
            enable_physics_recursively(stage, BUSBAR_ROOT_PATH)
            enable_physics_recursively(stage, NUT1_ROOT_PATH)
            enable_physics_recursively(stage, NUT2_ROOT_PATH)
            if busbar_xform and init_busbar_pos is not None:
                busbar_xform.set_world_pose(position=init_busbar_pos, orientation=init_busbar_quat)
            if nut1_xform and init_nut1_pos is not None:
                nut1_xform.set_world_pose(position=init_nut1_pos, orientation=init_nut1_quat)
            if nut2_xform and init_nut2_pos is not None:
                nut2_xform.set_world_pose(position=init_nut2_pos, orientation=init_nut2_quat)

            step_count = 0
            grasp_timer = 0

            yaw_aligner.reset()

            nut_index = 0
            nut_pick_pos = None
            nut_approach_pos = None
            bolt_target_pos = None
            bolt_touch_pos = None
            screw_sub = "rotate"
            screw_pass_idx = 0
            screw_pass_theta = 0.0
            stuck_counter = 0

        # 2. BehaviorNode / ArmNode 명령 분기 처리
        if playing and isaac_node.requested_task:
            task = isaac_node.requested_task
            isaac_node.requested_task = None

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
                isaac_node.latest_target_pose = None
                yaw_aligner.reset()

                # FINE_ALIGNMENT 진입 시 실시간 EE 위치 기준으로 Target 초기화
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

            # ── [신규] 너트 1/2번 파지 -> 체결 (Nut Assembly) ──
            # main의 assembly_nut_fraction1.py와 동일하게 스캔 단계 없이 공급대
            # 고정좌표로 바로 접근한다 (아래 PICK_NUT1/2에서 NUT1_PICK_XY/NUT2_PICK_XY 사용).
            elif task in ("PICK_NUT1", "PICK_NUT2"):
                nut_index = 1 if task == "PICK_NUT1" else 2
                # ★ 비전(latest_target_pose) 대신 assembly_nut_fraction1.py와 동일한
                # 고정좌표 사용 - 공급대 너트 위치는 항상 같아서 eye-in-hand 실시간
                # 추적보다 이쪽이 더 안정적 (위 NUT1_PICK_XY/NUT2_PICK_XY 주석 참고).
                pick_xy = NUT1_PICK_XY if nut_index == 1 else NUT2_PICK_XY
                nut_pick_pos = np.array([pick_xy[0], pick_xy[1], NUT_PICK_Z])
                nut_approach_pos = np.array([pick_xy[0], pick_xy[1], NUT_APPROACH_Z])
                phase = "NUT_APPROACH"
                step_count = 0
                print(f"\n>>> [{task}] 너트 {nut_index}번 상공 접근 시작 (Target: X={nut_approach_pos[0]:.3f}, Y={nut_approach_pos[1]:.3f})")

            elif task in ("ASSEMBLE_NUT1", "ASSEMBLE_NUT2"):
                nut_index = 1 if task == "ASSEMBLE_NUT1" else 2
                if target_mid_pos is not None:
                    bolt_target_pos = compute_bolt_target_pos(nut_index, target_mid_pos)
                    # ★ arm_node가 착좌 진입 직전 get_bolt_pair 실측값을 /target_pose로
                    # 먼저 보내뒀으면 그 XY로 대체 (Z는 기존 고정 오프셋 계산값 유지 -
                    # depth/체결 깊이 계산은 별도로 튜닝된 값이라 그대로 둔다). 못 받았으면
                    # 기존처럼 고정 오프셋 계산값 그대로 사용.
                    if isaac_node.latest_target_pose is not None:
                        vpos = isaac_node.latest_target_pose.pose.position
                        bolt_target_pos = np.array([vpos.x, vpos.y, bolt_target_pos[2]])
                        print(f"    [Vision] get_bolt_pair 실측 좌표로 대체 -> X={vpos.x:.4f}, Y={vpos.y:.4f}")
                    bolt_touch_pos = np.array([
                        bolt_target_pos[0], bolt_target_pos[1],
                        bolt_target_pos[2] + EE_OFFSET[2] + NUT_FASTEN_TOUCH_Z_LOCAL
                    ])
                    phase = "MOVE_TO_BOLT_NUT"
                    step_count = 0
                    print(f"\n>>> [{task}] 너트 {nut_index}번 -> 볼트 {nut_index}번 체결 시작 (Target: X={bolt_target_pos[0]:.4f}, Y={bolt_target_pos[1]:.4f})")
                else:
                    print(f"\n[ERROR] [{task}] 배터리 중심 좌표(target_mid_pos)가 없습니다. 먼저 ASSEMBLE_BUSBAR가 수행되어야 합니다.")
                    publish_status("FAILURE:NO_BATTERY_CENTER")

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

            # [7단계] 비전 보정 노드 피드백 기반 미세 오차 정렬 (Target 상태 유지 보완)
            elif phase == "FINE_ALIGNMENT":
                # ★ 실제 기동 중인 error_fix 패키지(src/error_fix/error_fix/error_fix.py)는
                # world dx,dy를 안 보낸다 - FIXED_STEP(0.2mm)짜리 방향 신호만 /target_pose로
                # 계속 보내고, 받는 쪽이 누적해야 하는 프로토콜이다(error_fix_depth1.py의
                # /busbar_alignment_error 프로토콜과는 다름 - 처음에 그걸로 착각해서
                # 아무 보정도 안 먹히고 있었음, 2026-07-27 실측 확인). 완료는
                # /task_command="ALIGNMENT_SUCCESS"로 온다(픽셀 0오차+각도 0.5도 이하를
                # 30프레임 연속 유지했을 때).
                publish_progress("FINE_ALIGNMENT_TRACKING", 85.0)

                if isaac_node.latest_target_pose is not None:
                    offset = isaac_node.latest_target_pose.pose.position
                    offset_quat = isaac_node.latest_target_pose.pose.orientation
                    if abs(offset.x) <= 0.0025 and abs(offset.y) <= 0.0025:
                        target_fine_pos[0] += offset.x
                        target_fine_pos[1] += offset.y
                        target_fine_pos[2] = BATTERY_CENTER_Z
                        # ★ 기존엔 orientation(yaw 스텝)을 통째로 무시하고 있었음 -
                        # 쿼터니언에서 yaw만 뽑아 누적(±out_limit_deg로 clamp)한다.
                        step_yaw_deg = math.degrees(2.0 * math.atan2(offset_quat.z, offset_quat.w))
                        yaw_aligner.correction_deg = max(
                            -yaw_aligner.out_limit_deg,
                            min(yaw_aligner.out_limit_deg, yaw_aligner.correction_deg + step_yaw_deg)
                        )
                    isaac_node.latest_target_pose = None

                quat_fine = yaw_rotated_quat(euler_to_quaternion_wxyz(0.0, 3.1415, 0.0), yaw_aligner.correction_deg)

                actions = arm_controller.forward(
                    target_end_effector_position=target_fine_pos,
                    target_end_effector_orientation=quat_fine
                )
                robot.apply_action(actions)

                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))
                glue_busbar_to_ee(robot, busbar_xform, busbar_start_grasp_pos, blend=1.0)

                if step_count % 30 == 0:
                    cur_ee_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                    print(f"    [FINE_ALIGNMENT] cur=({cur_ee_pos[0]:.4f},{cur_ee_pos[1]:.4f}) "
                          f"target=({target_fine_pos[0]:.4f},{target_fine_pos[1]:.4f}) "
                          f"yaw_corr={yaw_aligner.correction_deg:+.2f}deg")

                if isaac_node.alignment_success:
                    print(f"\n★ [FINE_ALIGNMENT SUCCESS] 미세 오차 보정 성공 및 정렬 완료! "
                          f"(yaw_corr={yaw_aligner.correction_deg:+.2f}deg)")
                    isaac_node.alignment_success = False
                    publish_progress("FINE_ALIGNMENT_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"

            # [8단계] 버스바 수직 하강 체결
            elif phase == "BUSBAR_DESCEND_TO_BOLT":
                publish_progress("BUSBAR_DESCEND_INSERT", 90.0)

                descend_target_z = max(descend_target_z - INSERT_SPEED, target_mid_pos[2])

                # ★ error_fix.py는 ALIGNMENT_SUCCESS를 보내고 나면 is_active=False가 되어
                # 더 이상 /target_pose를 안 보낸다 - 하강 중엔 새 보정이 안 들어오는 게
                # 정상이라 XY는 FINE_ALIGNMENT에서 확정된 target_mid_pos를 그대로 쓴다.
                # yaw만 FINE_ALIGNMENT에서 얻은 보정각을 그대로 유지해서 하강 내내 적용한다
                # (기존엔 하강 시작하자마자 0도로 초기화돼서 애써 맞춘 각도가 사라졌었음).
                step_target_pos = np.array([target_mid_pos[0], target_mid_pos[1], descend_target_z])
                quat_fine = yaw_rotated_quat(euler_to_quaternion_wxyz(0.0, 3.1415, 0.0), yaw_aligner.correction_deg)

                actions = arm_controller.forward(
                    target_end_effector_position=step_target_pos,
                    target_end_effector_orientation=quat_fine
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))

                glue_busbar_to_ee(robot, busbar_xform, busbar_start_grasp_pos, blend=1.0)

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                dist_err = math.dist(cur_pos, tuple(target_mid_pos))

                if step_count % 30 == 0:
                    print(f"    [BUSBAR_DESCEND_TO_BOLT] cur=({cur_pos[0]:.4f},{cur_pos[1]:.4f},{cur_pos[2]:.4f}) "
                          f"target=({target_mid_pos[0]:.4f},{target_mid_pos[1]:.4f}) "
                          f"yaw_corr={yaw_aligner.correction_deg:+.2f}deg err={dist_err*1000:.1f}mm")

                if cur_pos[2] <= BUSBAR_RELEASE_Z or dist_err < INSERT_TOLERANCE_STRICT:
                    if busbar_xform is not None:
                        busbar_xform.set_world_pose(position=target_mid_pos, orientation=BUSBAR_REST_ORIENTATION)
                    
                    # -------------------------------------------------------------
                    # 🔥 [추가 필요] 너트 조립용 기준 좌표(Anchor) 저장 및 볼트 3D 좌표 산출
                    # -------------------------------------------------------------
                    global assembled_battery_center, calculated_bolt1_pos, calculated_bolt2_pos
                    
                    # 1) 실제 안착된 배터리 중심 좌표 확정 저장
                    assembled_battery_center = np.array(target_mid_pos, dtype=float)
                    
                    # 2) 볼트 1, 2의 오프셋 벡터 (미리 정의된 상대 거리 오프셋)
                    # 예: BOLT1_OFFSET = np.array([-0.1042, 0.1812, 0.0])
                    #     BOLT2_OFFSET = np.array([0.1042, -0.1812, 0.0])
                    calculated_bolt1_pos = assembled_battery_center + BOLT1_OFFSET_FROM_CENTER 
                    calculated_bolt2_pos = assembled_battery_center + BOLT2_OFFSET_FROM_CENTER 
                    
                    print(f"\n[OK] 버스바 안착 체결 완료 (EE Z: {cur_pos[2]:.4f}m)!")
                    print(f" -> 기준 안착 좌표 저장: {assembled_battery_center}")
                    print(f" -> 동적 계산된 볼트1 좌표: {calculated_bolt1_pos}")
                    print(f" -> 동적 계산된 볼트2 좌표: {calculated_bolt2_pos}")
                    # -------------------------------------------------------------

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

                # Kinematic Pose-Glue 해제
                glue_busbar_to_ee(robot, busbar_xform, busbar_start_grasp_pos, blend=0.0)

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(retract_pos))

                if current_err < PICK_TOLERANCE_STRICT or step_count > MAX_STUCK_STEPS:
                    print("\n★ [ASSEMBLE_BUSBAR SUCCESS] 버스바 체결 및 안전 이탈 완료!")
                    publish_progress("ASSEMBLE_BUSBAR_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"

            # ════════════════════════════════════════════════════════════════
            # [신규] 너트 조립(Nut Assembly) 공용 Phase (nut_index로 Nut1/Nut2 재사용)
            # ════════════════════════════════════════════════════════════════
            # [11단계] 너트 상공 접근
            elif phase == "NUT_APPROACH":
                publish_progress("NUT_APPROACH", 20.0)
                # ★ eye-in-hand 실시간 추적은 접근할수록 목표가 표류해 미달하는 문제가
                # 있어서(위 NUT1_PICK_XY 주석 참고) 되돌림 - nut_approach_pos는
                # PICK_NUT1/2 커맨드 수신 시 고정좌표로 한 번만 설정되고 여기서는 갱신 안 함.
                actions = arm_controller.forward(target_end_effector_position=nut_approach_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(nut_approach_pos))

                # ★ 진단용: eye-in-hand 추적값이 계속 표류하는지, 그냥 못 따라가는지
                # 구분하려면 이 로그로 목표(vision)와 실제 EE 위치를 같이 봐야 한다.
                if step_count % 30 == 0:
                    print(f"    [NUT_APPROACH] cur=({cur_pos[0]:.4f},{cur_pos[1]:.4f}) "
                          f"target=({nut_approach_pos[0]:.4f},{nut_approach_pos[1]:.4f}) err={current_err*1000:.1f}mm")

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    nut_label = resolve_nut_assets(nut_index, nut1_xform, nut2_xform)[1]
                    print(f"[OK] {nut_label} 상공 도착! (err={current_err*1000:.1f}mm) -> 하강 시작")
                    phase = "NUT_DESCEND"
                    step_count = 0

            # [12단계] 너트 파지점 하강
            elif phase == "NUT_DESCEND":
                publish_progress("NUT_DESCEND", 40.0)
                # ★ NUT_APPROACH와 동일한 이유로 하강 중에도 고정좌표(nut_pick_pos)를
                # 그대로 유지, 매 스텝 갱신하지 않음.
                actions = arm_controller.forward(target_end_effector_position=nut_pick_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(nut_pick_pos))

                if step_count % 30 == 0:
                    print(f"    [NUT_DESCEND] cur=({cur_pos[0]:.4f},{cur_pos[1]:.4f},{cur_pos[2]:.4f}) "
                          f"target=({nut_pick_pos[0]:.4f},{nut_pick_pos[1]:.4f},{nut_pick_pos[2]:.4f}) err={current_err*1000:.1f}mm")

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    nut_label = resolve_nut_assets(nut_index, nut1_xform, nut2_xform)[1]
                    print(f"[OK] {nut_label} 하강 완료! (err={current_err*1000:.1f}mm) -> 물리 파지(Gripper Close) 시작")
                    phase = "NUT_GRASP"
                    grasp_timer = 0

            # [13단계] 너트 물리 파지 (실제 그리퍼로 파지, Kinematic Glue 미사용)
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

            # [15단계] 볼트 상공 이동 (compute_bolt_target_pos()로 동적 산출된 bolt_target_pos 사용)
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

            # [17단계] Screwing (Nut1/Nut2 공용 - 회전 체결 / Regrasp / 토크·Stuck 감지)
            elif phase == "NUT_SCREW":
                publish_progress("NUT_SCREWING", 70.0)
                if screw_sub == "rotate":
                    screw_pass_theta = min(screw_pass_theta + SCREW_OMEGA_DEG_S * PHYSICS_DT, SCREW_TURNS_DEG)
                    pass_done = (screw_pass_theta >= SCREW_TURNS_DEG)

                    total_deg = screw_pass_idx * SCREW_TURNS_DEG + screw_pass_theta
                    depth_m = min((total_deg / 360.0) * NUT_PITCH_M, ENGAGE_LEN)

                    target_pos = screw_seat_ee_pos.copy()
                    target_pos[2] = screw_seat_ee_pos[2] - depth_m
                    target_quat = yaw_rotated_quat(screw_start_quat, screw_pass_theta)

                    actions = arm_controller.forward(target_end_effector_position=target_pos, target_end_effector_orientation=target_quat)
                    robot.apply_action(actions)
                    robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                    # 실시간 토크 감지 및 Z축 정지(Stuck) 모니터링
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

            # [20단계] 기본 오리엔테이션 정렬 후 완료
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
                    print(f"\n★ [ASSEMBLE_{nut_label} SUCCESS] 너트 체결 및 안전 이탈 완료!")
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
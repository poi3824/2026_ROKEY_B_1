#!/usr/bin/env python3
"""
execute_isaac.py - Busbar Automation Sequence for Isaac Sim
BehaviorNode (FSM) 및 ArmNode, Vision Correction Node 통신 연동 버전

[동작 방식]
0. INIT_POSE            : 관절값 [0, 0, 90, 0, 90, 0] 도 단위로 초기 위치 정렬
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

# 2. USD 및 Isaac Core Imports
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
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
USD_PATH = "/home/rokey/junhyeok_version/isaacpjt/M0609/Collected_Busbar_AMR/Busbar.usd"

NOVA_CARTER_ROOT = "/World/Nova_Carter/chassis_link"
M0609_PATH       = "/World/m0609"
EE_LINK_NAME     = "link_6"
BOLT_CAMERA_PATH = "/World/Camera_bolt"  # 볼트쌍 인식용 고정 카메라 - 스테이션마다 위치가
                                          # 달라서 작업 중인 스테이션의 2번 볼트 좌표로
                                          # 옮겨줘야 한다
GRIPPER_JOINTS   = ["finger_joint", "right_inner_knuckle_joint"]
GRIPPER_ROOT_PATH = f"{M0609_PATH}/onrobot_rg2ft"

# AMR 이동 파라미터 (amr_node의 arrival_tolerance_m 기본값 0.05m보다 더 정확히 세워서 도착)
AMR_LINEAR_SPEED  = 0.3    # m/s (목표에서 멀 때 최고 속도)
AMR_ANGULAR_SPEED = 1.0    # rad/s (목표에서 멀 때 최고 각속도)
AMR_POS_TOL       = 0.02   # m
AMR_YAW_TOL       = 0.03   # rad
AMR_DECEL_DIST    = 0.3    # m - 목표까지 이 거리 안에서는 속도를 선형으로 줄인다
AMR_DECEL_YAW     = 0.3    # rad - 각도도 동일하게 감속
AMR_MIN_LINEAR_SPEED  = 0.02   # m/s - 감속해도 이 이하로는 안 느려짐(끝없이 기어가는 것 방지)
AMR_MIN_ANGULAR_SPEED = 0.05   # rad/s
AMR_ACCEL_TIME = 1.0   # s - 이동 시작 후 이 시간 동안 속도를 0에서 최고속까지 서서히 올린다

BUSBAR_ROOT_PATH      = "/World/Z_busbar3"
BUSBAR_POLYSHAPE_PATH = "/World/Z_busbar3/Mesh"

# 너트 Prim 경로 (체결에 실제로 쓰이는 건 nut1/nut2뿐이지만, 씬에는 여분 너트
# nut3~nut6도 있고 이것들도 전부 AMR에 실린 부품 트레이 위 물체라 이동 시 같이
# 옮겨줘야 한다). 예전엔 nut1_01/nut2_01/02/03이라는 이름이었는데 씬에서 nut3~nut6로
# 순차 개명됨.
NUT1_ROOT_PATH      = "/World/nut1"
NUT2_ROOT_PATH      = "/World/nut2"
NUT1_POLYSHAPE_PATH = "/World/nut1/geo/PolyShape"
NUT2_POLYSHAPE_PATH = "/World/nut2/geo/PolyShape"
EXTRA_NUT_ROOT_PATHS = ["/World/nut3", "/World/nut4", "/World/nut5", "/World/nut6"]
EXTRA_NUT_POLYSHAPE_PATHS = [f"{p}/geo/PolyShape" for p in EXTRA_NUT_ROOT_PATHS]

# 그리퍼 파라미터
GRIPPER_OPEN      = np.array([0.0, 0.0])
GRIPPER_CLOSE     = np.array([0.85, 0.85])
# battery4_main에서 실제 너트 파지에 사용한 닫힘 값
GRIPPER_CLOSE_NUT = np.array([0.96, 0.96])
GRIPPER_DELTA     = np.array([-0.5, -0.5])
GRIP_CLOSE_RAMP_STEPS = 50
NUT_AMR_DETACH_SETTLE_STEPS = 60  # 하강 완료 직후(손가락 닫기 전) 너트가 중력/잔류
                                   # 속도로 살짝 튀거나 흔들릴 수 있으니, 완전히 정지할
                                   # 때까지(1초) 대기한 뒤에야 그리퍼를 닫기 시작한다.
GRIP_SETTLE_STEPS = 15  # 손가락이 다 닫힌 뒤 FixedJoint를 만들기 전 안정화 대기 틱 수
BUSBAR_MIN_LIFT_RISE = 0.03  # 실제 버스바가 최소 30mm 상승해야 PICK 성공으로 인정

# Kinematic Pose-Glue 파라미터
EE_OFFSET = np.array([0.0, 0.0, 0.185])
BUSBAR_HEIGHT = 0.003
BUSBAR_GRASP_Z_LOCAL = BUSBAR_HEIGHT + 0.02
BUSBAR_REST_ORIENTATION = np.array([0.5, -0.5, 0.5, 0.5])

# 기본 좌표 및 높이 정의
TARGET_INIT_JOINTS   = np.array([0.0, 0.0, np.radians(90.0), 0.0, np.radians(90.0), np.radians(90.0)])

# INIT_POSE/RETURN_HOME_JOINTS 관절 속도 제한 (rad/s) - 목표 각도를 한 틱에 통째로
# 명령하면 오차가 클 때 PD 드라이브가 최대 속도로 꽂아버려 반력으로 AMR까지 흔들리므로,
# 이 속도로 매 틱 목표를 조금씩만 전진시켜 부드럽게 도달하게 한다.
ARM_INIT_MAX_JOINT_SPEED = math.radians(20.0)  # rad/s

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
NUT_SCAN_Z        = 0.9

# 초기 자세(INIT_POSE 직후 HOME_EE_POS) TCP 좌표 기준, 너트 1~6번까지의 실측 상대
# 오프셋(X,Y만 - Z는 NUT_PICK_Z로 고정). 초기 TCP=(1.0335, 0.1931, 0.8658), 너트1~6
# 실측 월드좌표를 각각 빼서 계산함 - AMR이 어느 배터리 스테이션에 있든, INIT_POSE로
# 되돌아가 새로 구한 HOME_EE_POS에 이 오프셋만 더하면 그 스테이션에서의 정확한 너트
# 위치가 나온다(월드 절대좌표를 하드코딩하면 스테이션이 바뀔 때마다 다 틀어짐).
NUT_OFFSET_FROM_HOME = {
    1: np.array([-0.4587, -0.2957]),
    2: np.array([-0.3683, -0.2961]),
    3: np.array([-0.2781, -0.2949]),
    4: np.array([-0.2781, -0.2093]),
    5: np.array([-0.3692, -0.2107]),
    6: np.array([-0.4589, -0.2083]),
}

NUT_HEIGHT         = 0.0095
NUT_GRASP_Z_LOCAL  = NUT_HEIGHT + 0.023
NUT_SUPPLY_TABLE_Z = 0.72                                   # 너트 공급대 높이
NUT_PICK_Z         = NUT_SUPPLY_TABLE_Z - (NUT_GRASP_Z_LOCAL - 0.0395)
NUT_APPROACH_Z     = 0.9                                     # 너트 파지 상공 고도 - 볼트 펙 위로
                                                              # 완전히 빠져나올 여유를 더 주기 위해
                                                              # 0.8->0.9로 올림(실측: 0.8은 부족해서
                                                              # 들어올리다 펙에 걸려 놓쳤음)
NUT_PEG_CLEARANCE_Z = 0.08                                   # 파지 직후 XY 이동 없이 먼저
                                                              # 수직으로 빠져나올 거리(80mm,
                                                              # 공급대 peg 길이보다 큰 안전값)
NUT_PEG_CLEAR_TOLERANCE = 0.003                              # 수직 이탈 완료 허용오차(3mm)
NUT_PEG_CLEAR_HOLD_STEPS = 15                                # 이탈 높이 도착 후 파지 안정화
                                                              # 대기(60Hz 기준 0.25초)
BOLT_APPROACH_Z    = 0.6                                     # 너트 체결 상공 고도

# ★ 스테이션별 볼트 1/2번 실측 월드 좌표 (test_isaac 씬 고정 배치 기준) ★
# 딕셔너리 키(1,2)는 "이번 스테이션의 몇 번째 볼트인가"이지, 트레이의 물리적 nut_index가
# 아니다(그건 STATION_NUT_INDICES가 따로 정한다).
STATION_BOLT_WORLD_POS = {
    3: {1: np.array([1.0552, 0.3722]), 2: np.array([1.2636, 0.0098])},
    4: {1: np.array([1.0552, -0.2047]), 2: np.array([1.2636, -0.5671])},
    5: {1: np.array([1.0552, -0.2047]), 2: np.array([1.2636, -1.0936])},
}

# 스테이션별로 6슬롯 너트 트레이 중 어느 물리적 너트(1~6)를 쓸지 - 배터리팩4는 3,4번,
# 배터리팩5는 5,6번 너트를 쓴다.
STATION_NUT_INDICES = {
    3: (1, 2),
    4: (3, 4),
    5: (5, 6),
}

# behavior_node.py의 AMR_STATION_POSES와 동일한 좌표 - /amr/goal_pose로 어느 스테이션에
# 와있는지 자동 판별하는 데 쓴다(battery/busbar 두 지점 다 등록해서 어느 태스크
# 시점이든 매칭되게 함).
STATION_AMR_POINTS = {
    3: [(0.6667, -0.0382), (0.5867, 1.9078)],
    4: [(0.6667, -0.6617), (-0.2271, 1.9078)],
    5: [(0.6667, -1.1964), (-0.9586, 1.9078)],
}

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


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def _quat_rotate_vec(q, v):
    qv = np.array([0.0, v[0], v[1], v[2]])
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[1:]


def compute_local_offset(parent_pos, parent_quat, child_pos, child_quat):
    """battery4_main 방식: AMR 기준 너트의 로컬 위치/자세를 계산한다."""
    parent_pos = np.asarray(parent_pos, dtype=float)
    parent_quat = np.asarray(parent_quat, dtype=float)
    child_pos = np.asarray(child_pos, dtype=float)
    child_quat = np.asarray(child_quat, dtype=float)
    parent_inv = _quat_conj(parent_quat)
    return (
        _quat_rotate_vec(parent_inv, child_pos - parent_pos),
        _quat_mul(parent_inv, child_quat),
    )


def compose_world_pose(parent_pos, parent_quat, local_pos, local_quat):
    """battery4_main 방식: AMR 월드 포즈와 로컬 오프셋으로 너트 포즈를 복원한다."""
    parent_pos = np.asarray(parent_pos, dtype=float)
    parent_quat = np.asarray(parent_quat, dtype=float)
    return (
        parent_pos + _quat_rotate_vec(parent_quat, np.asarray(local_pos, dtype=float)),
        _quat_mul(parent_quat, np.asarray(local_quat, dtype=float)),
    )


def disable_rigidbody_sleep(stage, prim_path):
    """너트처럼 AMR에 용접된 채 오래 정지해 있다가 그리퍼로 옮겨지는 RigidBody는, PhysX가
    "안 움직인다"고 판단해 자동으로 재워버릴 수 있다 - 잠든 채로는 그리퍼가 손가락을
    닫아도 접촉/마찰이 제대로 반영 안 될 수 있어서, sleepThreshold를 0으로 낮춰
    애초에 잠들지 않게 한다."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rb.CreateSleepThresholdAttr().Set(0.0)


def find_rigidbody_ancestor(stage, start_path):
    """start_path부터 위로 올라가며 실제 UsdPhysics.RigidBodyAPI가 적용된 조상 프림을 찾는다."""
    prim = stage.GetPrimAtPath(start_path)
    while prim.IsValid():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(prim.GetPath())
        if prim.GetPath() == Sdf.Path.absoluteRootPath:
            break
        prim = prim.GetParent()
    return None


NUT_AMR_JOINT_NAME = "PhysicsFixedJoint_AMR"


def remove_nut_amr_joint(stage, nut_root_path, joint_name=NUT_AMR_JOINT_NAME):
    """attach_nut_to_amr이 만든 조인트를 완전히 지운다 (비활성화만으로는 부족 - 남은 조인트
    프림이 옛 상대위치를 들고 있다가 다음 attach 때 재사용되면 두 프레임이 어긋난다)."""
    joint_path = f"{nut_root_path}/{joint_name}"
    joint_prim = stage.GetPrimAtPath(joint_path)
    if joint_prim.IsValid():
        UsdPhysics.FixedJoint(joint_prim).GetJointEnabledAttr().Set(False)
        stage.RemovePrim(joint_path)


def attach_nut_to_amr(stage, nut_root_path, nut_rigidbody_path, amr_link_path, nut_xform, robot, joint_name=NUT_AMR_JOINT_NAME):
    """너트를 현재 위치 그대로 AMR 링크에 물리 FixedJoint로 고정한다 (순간이동 없이 그 자리에서 부착).
    world.reset() 직후에만 호출한다 - 이 시점 이후 시뮬레이션 도중에 새로 만드는 조인트는
    PhysX 라이브 반영이 안 되는 문제가 있었다(그리퍼 조인트에서 실측 확인됨)."""
    amr_prim = stage.GetPrimAtPath(amr_link_path)
    nut_prim = stage.GetPrimAtPath(nut_rigidbody_path)
    if not amr_prim.IsValid() or not nut_prim.IsValid():
        return None

    remove_nut_amr_joint(stage, nut_root_path, joint_name)

    amr_pos, amr_quat = robot.get_world_pose()
    amr_pos = np.asarray(amr_pos, dtype=float)
    amr_quat = np.asarray(amr_quat, dtype=float)

    nut_pos, nut_quat = nut_xform.get_world_pose()
    nut_pos = np.asarray(nut_pos, dtype=float)
    nut_quat = np.asarray(nut_quat, dtype=float)

    amr_quat_conj = _quat_conj(amr_quat)
    rel_pos = _quat_rotate_vec(amr_quat_conj, nut_pos - amr_pos)
    rel_quat = _quat_mul(amr_quat_conj, nut_quat)

    # 검증: local0을 다시 world로 복원했을 때 실제 너트 위치와 일치하는지 확인.
    frame0_world_pos = amr_pos + _quat_rotate_vec(amr_quat, rel_pos)
    frame_error = float(np.linalg.norm(frame0_world_pos - nut_pos))
    joint_path = f"{nut_root_path}/{joint_name}"
    if frame_error > 0.001:
        print(f"[ERROR] {joint_path} FixedJoint frame 불일치 {frame_error*1000:.2f}mm -> 생성 취소")
        return None

    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateJointEnabledAttr(False)
    joint.CreateBody0Rel().SetTargets([amr_link_path])
    joint.CreateBody1Rel().SetTargets([nut_rigidbody_path])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in rel_pos]))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(rel_quat[0]), Gf.Vec3f(*[float(v) for v in rel_quat[1:]])))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.GetJointEnabledAttr().Set(True)
    print(f"[NUT JOINT ATTACH] {joint_path} frame_error={frame_error*1000:.3f}mm")
    return joint


def detach_nut_from_amr(stage, nut_root_path, joint_name=NUT_AMR_JOINT_NAME):
    """attach_nut_to_amr으로 만든 조인트를 비활성화해서 너트를 AMR에서 풀어준다 - 그리퍼가
    물리적으로 붙잡아 들어올리기 직전(NUT_GRASP 시작 전)에 호출한다."""
    joint_path = f"{nut_root_path}/{joint_name}"
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid():
        print(f"[WARN] {joint_path} 조인트가 없습니다 (이미 해제됐거나 생성된 적 없음)")
        return False
    enabled_attr = UsdPhysics.FixedJoint(joint_prim).GetJointEnabledAttr()
    if enabled_attr:
        enabled_attr.Set(False)
    print(f"[NUT JOINT DETACH] {joint_path}")
    return True




BUSBAR_GRIP_JOINT_NAME = "PhysicsFixedJoint_Gripper"


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def _quat_rotate_vec(q, v):
    qv = np.array([0.0, v[0], v[1], v[2]])
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[1:]


def find_busbar_body_near(stage, target_xy, max_distance=0.30):
    """비전 파지점 근처의 버스바 Mesh rigid body를 찾는다.

    상위 Xform/그룹 프림까지 후보에 넣으면 엉뚱한 body frame으로 조인트가 생성되어
    버스바가 튈 수 있으므로, 실제 형상 body인 Mesh/PolyShape만 허용한다.
    """
    candidates = []
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        return None

    for prim in Usd.PrimRange(world_prim):
        path = str(prim.GetPath())
        if "busbar" not in path.lower():
            continue
        if prim.GetName().lower() not in {"mesh", "polyshape"}:
            continue
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue

        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        pos = xf.ExtractTranslation()
        distance = math.hypot(
            float(pos[0]) - float(target_xy[0]),
            float(pos[1]) - float(target_xy[1]),
        )
        if distance <= max_distance:
            candidates.append((distance, path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    distance, path = candidates[0]
    print(f"[BUSBAR SELECT] target=({target_xy[0]:.4f},{target_xy[1]:.4f}) "
          f"-> {path} (distance={distance*1000:.1f}mm)")
    return path


def busbar_root_path_for_body(busbar_body_path):
    """선택된 Mesh/PolyShape 경로에서 해당 버스바의 최상위 root 경로를 구한다."""
    parts = busbar_body_path.rstrip("/").split("/")
    for index in range(len(parts) - 1, 1, -1):
        if "busbar" in parts[index].lower():
            return "/".join(parts[:index + 1])
    return busbar_body_path.rsplit("/", 1)[0]


def remove_all_busbar_grip_joints(stage):
    """이전 작업에서 남은 버스바-그리퍼 FixedJoint를 전부 제거한다."""
    joint_paths = []
    world_prim = stage.GetPrimAtPath("/World")
    if world_prim.IsValid():
        for prim in Usd.PrimRange(world_prim):
            if prim.GetName() == BUSBAR_GRIP_JOINT_NAME:
                joint_paths.append(str(prim.GetPath()))

    for joint_path in joint_paths:
        joint_prim = stage.GetPrimAtPath(joint_path)
        if joint_prim.IsValid():
            UsdPhysics.FixedJoint(joint_prim).GetJointEnabledAttr().Set(False)
            stage.RemovePrim(joint_path)
            print(f"[BUSBAR JOINT REMOVE] {joint_path}")


def attach_busbar_to_gripper(stage, gripper_link_path, robot, busbar_body_path):
    """버스바를 그리퍼(EE 링크)에 FixedJoint로 고정한다.

    find_rigidbody_path로 찾은 부모 프림(/World/Z_busbar3)과, 위치를 실제로 측정한
    Mesh 프림(/World/Z_busbar3/Mesh, busbar_xform)이 서로 다른 로컬 좌표계라 조인트가
    엉뚱한 곳으로 튕겼다(실측 확인됨) - 부모와 Mesh 사이에 자체 상대 오프셋이 있으면
    "Mesh 기준으로 잰 상대변환"을 "부모 프림"에 그대로 적용했을 때 그 오프셋만큼
    어긋난다. 그래서 위치를 재는 프림과 조인트를 실제로 거는 프림을 Mesh 하나로
    통일한다. 선택된 Mesh/PolyShape를 위치 측정과 Body1에 동일하게 사용한다."""
    gripper_prim = stage.GetPrimAtPath(gripper_link_path)
    busbar_prim = stage.GetPrimAtPath(busbar_body_path)
    if not gripper_prim.IsValid() or not busbar_prim.IsValid():
        return None

    ee_pos, ee_quat = robot.end_effector.get_world_pose()
    busbar_world_xf = UsdGeom.Xformable(busbar_prim).ComputeLocalToWorldTransform(0)
    real_pos = np.asarray(busbar_world_xf.ExtractTranslation(), dtype=float)
    busbar_quat = busbar_world_xf.ExtractRotationQuat()
    real_quat = np.array(
        [busbar_quat.GetReal(), *busbar_quat.GetImaginary()], dtype=float
    )
    ee_pos = np.asarray(ee_pos, dtype=float)
    ee_quat = np.asarray(ee_quat, dtype=float)
    real_pos = np.asarray(real_pos, dtype=float)
    real_quat = np.asarray(real_quat, dtype=float)

    ee_quat_conj = _quat_conj(ee_quat)
    rel_pos = _quat_rotate_vec(ee_quat_conj, real_pos - ee_pos)
    rel_quat = _quat_mul(ee_quat_conj, real_quat)

    joint_path = f"{busbar_body_path}/{BUSBAR_GRIP_JOINT_NAME}"
    joint_prim = stage.GetPrimAtPath(joint_path)
    if joint_prim.IsValid():
        UsdPhysics.FixedJoint(joint_prim).GetJointEnabledAttr().Set(False)
        stage.RemovePrim(joint_path)
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)

    joint.CreateBody0Rel().SetTargets([gripper_link_path])
    joint.CreateBody1Rel().SetTargets([busbar_body_path])

    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in rel_pos]))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(rel_quat[0]), Gf.Vec3f(*[float(v) for v in rel_quat[1:]])))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateCollisionEnabledAttr().Set(False)
    joint.CreateJointEnabledAttr().Set(True)

    print(f"[BUSBAR ATTACH] EE=({ee_pos[0]:.4f},{ee_pos[1]:.4f},{ee_pos[2]:.4f}) "
          f"Busbar={busbar_body_path} "
          f"({real_pos[0]:.4f},{real_pos[1]:.4f},{real_pos[2]:.4f}) rel_pos={rel_pos}")
    return joint


def detach_busbar_from_gripper(stage, busbar_body_path):
    joint_path = f"{busbar_body_path}/{BUSBAR_GRIP_JOINT_NAME}"
    joint_prim = stage.GetPrimAtPath(joint_path)
    if joint_prim.IsValid():
        UsdPhysics.FixedJoint(joint_prim).GetJointEnabledAttr().Set(False)
        stage.RemovePrim(joint_path)
        print(f"[BUSBAR JOINT DETACH+REMOVE] {joint_path}")



def nut_paths_for_index(nut_index):
    """nut_index(1~6)에 따라 (root_path, polyshape_path) 반환. 3~6번은 EXTRA_NUT_*
    (스테이션4=3,4번 / 스테이션5=5,6번 너트) 목록에서 가져온다."""
    if nut_index == 1:
        return NUT1_ROOT_PATH, NUT1_POLYSHAPE_PATH
    if nut_index == 2:
        return NUT2_ROOT_PATH, NUT2_POLYSHAPE_PATH
    idx = nut_index - 3
    return EXTRA_NUT_ROOT_PATHS[idx], EXTRA_NUT_POLYSHAPE_PATHS[idx]


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


def _set_runtime_gains(robot, joint_names, stiffness, damping, max_force):
    """robot._articulation_view(런타임 PhysX 텐서 뷰)에 직접 gain을 적용한다.

    UsdPhysics.DriveAPI의 stiffness/damping 속성을 시뮬레이션 시작 후에 stage에서
    직접 Set()해도 이미 초기화된 ArticulationView는 그 값을 다시 읽어가지 않는다
    (Articulation.set_gains의 save_to_usd=False 기본값이 보여주듯, USD와 런타임
    physics view는 서로 다른 저장소다). 그래서 "강성을 올렸는데도 계속 풀린다"는
    증상은 실제로는 런타임에 전혀 반영이 안 된 것이었다. robot.apply_action이
    쓰는 것과 동일한 _articulation_view.set_gains/set_max_efforts를 써야 실제
    시뮬레이션에 적용된다."""
    if robot is None:
        return
    view = getattr(robot, "_articulation_view", None)
    if view is None:
        return
    # view.set_gains(..., joint_names=...)의 내부 이름->인덱스 변환이 dof_names 순서와
    # 어긋나 있어(IndexError로 실측 확인됨) 범위를 벗어난 인덱스를 만들어냈다. 스크립트
    # 전역에서 이미 잘 쓰이고 있는 robot.get_dof_index()로 직접 인덱스를 구해
    # joint_indices로 넘기면 이 문제를 피할 수 있다.
    joint_indices = [robot.get_dof_index(name) for name in joint_names]
    n = len(joint_indices)
    kps = np.full((1, n), float(stiffness))
    kds = np.full((1, n), float(damping))
    efforts = np.full((1, n), float(max_force))
    view.set_gains(kps=kps, kds=kds, joint_indices=joint_indices, save_to_usd=False)
    view.set_max_efforts(values=efforts, joint_indices=joint_indices)


def lock_amr_base(stage, amr_root_path, robot=None):
    """로봇팔 작업 중 AMR 바퀴를 고정(브레이크)한다 - 팔의 반력으로 베이스가
    흔들리는 것을 방지.

    stiffness만 확 높이고 targetPosition을 안 맞추면, 드라이브가 "지금 위치"가
    아니라 예전에 남아있던 targetPosition(보통 0)으로 확 스냅되면서 바퀴가 휙
    돌아간다. 그래서 robot이 주어지면 잠그기 직전에 각 관절의 현재 각도를 읽어
    그대로 targetPosition으로 넣어, "지금 있는 자리 그대로" 잠기도록 한다.

    아래 stage 기반 DriveAPI Set()은 USD에 값을 남겨두기 위한 것일 뿐 런타임
    시뮬레이션에는 반영되지 않는다 - 실제 잠금 효과는 이 함수 끝에서
    robot._articulation_view.set_gains/set_max_efforts로 낸다(_set_runtime_gains)."""
    amr_prim = stage.GetPrimAtPath(amr_root_path).GetParent()
    if not amr_prim.IsValid():
        amr_prim = stage.GetPrimAtPath(amr_root_path)

    dof_names = list(robot.dof_names) if robot is not None else None
    joint_positions = robot.get_joint_positions() if robot is not None else None
    locked_joint_names = []

    for prim in Usd.PrimRange(amr_prim):
        for dt in ("angular", "linear"):
            drive = UsdPhysics.DriveAPI.Get(prim, dt)
            if drive:
                if dof_names is not None and prim.GetName() in dof_names:
                    idx = dof_names.index(prim.GetName())
                    cur_val = float(joint_positions[idx])
                    pos_attr = drive.GetTargetPositionAttr()
                    if pos_attr:
                        # USD Physics DriveAPI의 각도 관련 targetPosition은 degree 단위다
                        # (Isaac Sim의 관절 각도는 radian이라 변환 필요).
                        pos_attr.Set(math.degrees(cur_val) if dt == "angular" else cur_val)
                    locked_joint_names.append(prim.GetName())
                drive.GetStiffnessAttr().Set(1.0e9)
                drive.GetDampingAttr().Set(1.0e6)
                drive.GetMaxForceAttr().Set(1.0e9)
                if drive.GetTargetVelocityAttr():
                    drive.GetTargetVelocityAttr().Set(0.0)

    if robot is not None and locked_joint_names:
        locked_indices = [dof_names.index(n) for n in locked_joint_names]
        # 드라이브 목표 위치도 "지금 있는 자리"로 맞춰준다 (apply_action은 position
        # target을 설정할 뿐 순간이동시키지 않으므로 set_joint_positions과 다르다).
        robot.apply_action(
            ArticulationAction(
                joint_positions=joint_positions[locked_indices],
                joint_indices=locked_indices,
            )
        )
        _set_runtime_gains(robot, locked_joint_names, 1.0e9, 1.0e6, 1.0e9)


def unlock_amr_base(stage, amr_root_path, robot=None):
    """AMR 이동 시작 전 바퀴 구동부를 풀어준다 (lock_amr_base의 반대) - 베이스를
    set_world_pose로 직접 이동시키므로 바퀴 조인트가 그 이동에 저항하지 않도록 함.

    주의: 여기서는 일부러 _set_runtime_gains를 쓰지 않는다. 바퀴 강성을 런타임에서
    진짜로 0까지 내리면, 바퀴가 몸체(그 위의 팔 포함)를 떠받치는 힘까지 사라져서
    로봇이 바닥으로 무너지는 사고가 발생했다(실측 확인됨). AMR 이동은 set_world_pose로
    좌표를 직접 덮어써서 하기 때문에 바퀴 드라이브를 실제로 풀어줄 필요가 애초에 없다 -
    이 함수는 이전부터 stage 속성만 바꾸는(런타임에는 반영 안 되는) 안전한 무동작이었고,
    그 상태를 그대로 유지한다. robot 파라미터는 lock_amr_base와 시그니처를 맞추기 위해
    남겨두되 사용하지 않는다."""
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


def stiffen_gripper_grip(robot, joint_names=None):
    """물체(버스바 등)를 옮기는 동안 그리퍼 관절 강성을 확 높인다 - AMR 이동 중
    반동(순간이동으로 인한 물리적 충격)에도 그리퍼가 살짝 벌어지지 않도록 하기 위함.
    GRIPPER_OPEN/CLOSE 자체(목표 위치)는 그대로 동작하고, 그 목표를 얼마나 세게
    붙잡을지(강성/최대 힘)만 올라간다.

    반드시 robot._articulation_view.set_gains/set_max_efforts(런타임 physics view)로
    적용해야 한다 - 이전에는 UsdPhysics.DriveAPI를 stage에 직접 Set()했는데, 시뮬레이션이
    이미 시작된 뒤에는 그 변경이 실제 물리 계산에 반영되지 않아 "강성을 올려도 계속
    벌어지는" 문제가 있었다."""
    if joint_names is None:
        joint_names = GRIPPER_JOINTS
    _set_runtime_gains(robot, joint_names, 1.0e9, 1.0e6, 1.0e9)


def quat_wxyz_to_yaw(quat_wxyz):
    """월드 Z축 기준 yaw(rad)만 추출 (AMR은 평면 위를 움직이므로 roll/pitch는 무시)."""
    w, x, y, z = quat_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def resolve_station_from_amr_xy(x, y, tolerance=0.3):
    """AMR 목표좌표(x,y)를 STATION_AMR_POINTS와 비교해 가장 가까운 스테이션 번호를 찾는다.
    behavior_node.py가 station_id 문자열을 execute_isaac.py에 직접 전달하는 경로가 없어서,
    대신 이미 받고 있는 /amr/goal_pose로 역추정한다. tolerance(m) 밖이면 매칭 실패로 본다."""
    best_station, best_dist = None, None
    for station, points in STATION_AMR_POINTS.items():
        for px, py in points:
            d = math.hypot(x - px, y - py)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_station = station
    if best_dist is not None and best_dist <= tolerance:
        return best_station
    return None


def yaw_rotated_quat(base_wxyz, delta_deg):
    """base_wxyz 오리엔테이션을 월드 Z축 기준으로 delta_deg 만큼 추가 회전시킨 쿼터니언 반환 (Screwing 회전용)"""
    base_q = Gf.Quatd(float(base_wxyz[0]), Gf.Vec3d(float(base_wxyz[1]), float(base_wxyz[2]), float(base_wxyz[3])))
    base_rot = Gf.Rotation(base_q)
    extra_rot = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), float(delta_deg))
    combined = extra_rot * base_rot
    q = combined.GetQuat()
    return np.array([q.GetReal(), *q.GetImaginary()])


def resolve_nut_assets(nut_index, nut1_xform, nut2_xform, extra_nut_xforms=None):
    """nut_index(1~6)에 따라 대상 너트 Xform과 라벨을 반환. 3~6번은 extra_nut_xforms
    (EXTRA_NUT_ROOT_PATHS와 같은 순서) 목록에서 가져온다 - 안 넘겨주면 라벨만 반환."""
    if nut_index == 1:
        return nut1_xform, "너트 1번"
    if nut_index == 2:
        return nut2_xform, "너트 2번"
    xform = None
    if extra_nut_xforms is not None:
        idx = nut_index - 3
        if 0 <= idx < len(extra_nut_xforms):
            xform = extra_nut_xforms[idx]
    return xform, f"너트 {nut_index}번"


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

    bolt_camera_xform = SingleXFormPrim(BOLT_CAMERA_PATH, name="bolt_camera") if stage.GetPrimAtPath(BOLT_CAMERA_PATH).IsValid() else None
    bolt_camera_init_pos, bolt_camera_init_quat = bolt_camera_xform.get_world_pose() if bolt_camera_xform else (None, None)
    if bolt_camera_xform is None:
        print(f"[WARN] bolt_camera 프림을 못 찾았습니다: {BOLT_CAMERA_PATH} - 스테이션 4/5 볼트 인식이 이전 위치를 볼 수 있습니다.")

    nut1_xform = SingleXFormPrim(NUT1_POLYSHAPE_PATH, name="nut1_poly") if stage.GetPrimAtPath(NUT1_POLYSHAPE_PATH).IsValid() else None
    nut2_xform = SingleXFormPrim(NUT2_POLYSHAPE_PATH, name="nut2_poly") if stage.GetPrimAtPath(NUT2_POLYSHAPE_PATH).IsValid() else None
    init_nut1_pos, init_nut1_quat = nut1_xform.get_world_pose() if nut1_xform else (None, None)
    init_nut2_pos, init_nut2_quat = nut2_xform.get_world_pose() if nut2_xform else (None, None)

    # 진단용: NUT1/2_POLYSHAPE_PATH에 실제 RigidBodyAPI가 있는지, 아니면 조상 프림에
    # 있는지 확인 - 다르면 조인트 Body1 대상을 재검토해야 한다.
    print(f"[PHYSICS BODY] AMR  = {find_rigidbody_ancestor(stage, NOVA_CARTER_ROOT)}")
    print(f"[PHYSICS BODY] NUT1 = {find_rigidbody_ancestor(stage, NUT1_POLYSHAPE_PATH)} (used: {NUT1_POLYSHAPE_PATH})")
    print(f"[PHYSICS BODY] NUT2 = {find_rigidbody_ancestor(stage, NUT2_POLYSHAPE_PATH)} (used: {NUT2_POLYSHAPE_PATH})")

    # 너트가 AMR에 용접된 채 오래 정지해 있다가 그리퍼 조인트로 넘어갈 때 잠들어있으면
    # 안 된다 - 애초에 안 잠들도록 설정.
    disable_rigidbody_sleep(stage, NUT1_POLYSHAPE_PATH)
    disable_rigidbody_sleep(stage, NUT2_POLYSHAPE_PATH)
    for extra_poly in EXTRA_NUT_POLYSHAPE_PATHS:
        disable_rigidbody_sleep(stage, extra_poly)

    # 체결에 직접 쓰이진 않지만 AMR 부품 트레이 위 여분 너트들 - 이동 시 nut1/2와 똑같이 같이 옮겨야 한다.
    extra_nut_xforms = [
        SingleXFormPrim(p, name=f"extra_nut_poly_{i}") if stage.GetPrimAtPath(p).IsValid() else None
        for i, p in enumerate(EXTRA_NUT_POLYSHAPE_PATHS)
    ]
    extra_nut_inits = [
        (xf.get_world_pose() if xf else (None, None)) for xf in extra_nut_xforms
    ]

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

    # battery4_main의 검증된 너트 운반 방식: PICK 전에는 물리를 끄고 AMR 기준 로컬
    # 오프셋으로 매 프레임 포즈를 갱신한다. PICK 명령을 받은 너트만 이 glue에서 풀고
    # 물리를 켜서 그리퍼의 실제 접촉/마찰로 집는다.
    nut_xforms_all = [nut1_xform, nut2_xform, *extra_nut_xforms]
    nut_roots_all = [NUT1_ROOT_PATH, NUT2_ROOT_PATH, *EXTRA_NUT_ROOT_PATHS]
    nut_polys_all = [NUT1_POLYSHAPE_PATH, NUT2_POLYSHAPE_PATH, *EXTRA_NUT_POLYSHAPE_PATHS]
    nut_local_offsets = [None] * len(nut_xforms_all)
    nut_released = [False] * len(nut_xforms_all)
    amr_glue_pos, amr_glue_quat = robot.get_world_pose()
    for i, (nut_xf, nut_root) in enumerate(zip(nut_xforms_all, nut_roots_all)):
        if nut_xf is None:
            continue
        remove_nut_amr_joint(stage, nut_root)
        disable_physics_recursively(stage, nut_root)
        nut_pos, nut_quat = nut_xf.get_world_pose()
        nut_local_offsets[i] = compute_local_offset(
            amr_glue_pos, amr_glue_quat, nut_pos, nut_quat
        )

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

    def sync_rmpflow_base_pose():
        """AMR이 이동한 뒤(또는 강제로 멈춘 뒤) RMPFlow가 가정하는 베이스 위치/자세를
        m0609/base_link의 실제 world transform으로 다시 맞춘다. 이걸 안 하면 RMPFlow는
        팔이 예전 베이스 위치에 있는 줄 알고 IK를 풀어서 엉뚱한 자세가 나온다."""
        base_link_xf = world_xf(stage, f"{M0609_PATH}/base_link")
        base_pos = base_link_xf.ExtractTranslation()
        base_quat = base_link_xf.ExtractRotationQuat()
        arm_controller._motion_policy.set_robot_base_pose(
            robot_position=np.array([base_pos[0], base_pos[1], base_pos[2]]),
            robot_orientation=np.array([base_quat.GetReal(), *[float(x) for x in base_quat.GetImaginary()]]),
        )

    sync_rmpflow_base_pose()

    print("\nIsaac Sim 준비 완료 - BehaviorNode 명령을 대기합니다.")

    step_count = 0
    grasp_timer = 0
    nut_release_timer = 0
    was_playing = False
    phase = "IDLE"
    init_pose_only = False  # True면 INIT_POSE 완료 후 SCAN_APPROACH로 안 이어지고 바로 종료
    busbar_grasped = False  # 그리퍼-버스바 FixedJoint가 걸려있는 상태인지
    active_busbar_body_path = BUSBAR_POLYSHAPE_PATH  # 현재 작업에서 실제로 집을 버스바
    busbar_attach_z = None  # FixedJoint 생성 순간의 실제 버스바 Z (상승 검증 기준)
    descend_target_z = None
    target_mid_pos = None
    scan_hold_quat = None  # INIT_POSE 완료 시점의 실제 EE 자세 (SCAN_APPROACH가 그대로 유지)

    # ── 너트 조립(Nut Assembly) 상태 변수 ──
    nut_index      = 0        # 1: 너트 1번, 2: 너트 2번
    NUT_SCAN_POS   = None     # 너트 스캔 위치 (SCAN_NUT1/2마다 해당 너트 오프셋으로 새로 계산)
    NUT_SCAN_LIFT_POS = None  # 방향 정렬용 중간 경유 위치 (현재 XY, 스캔 고도)
    BUSBAR_SCAN_LIFT_POS = None  # 버스바 스캔용 방향 정렬 중간 경유 위치 (현재 XY, 스캔 고도)
    nut_pick_pos   = None     # 현재 너트의 물리 파지 좌표
    nut_approach_pos = None   # 현재 너트 파지 상공 접근 좌표
    nut_peg_clear_pos = None  # 파지 직후 peg에서 수직으로 빠져나올 중간 목표
    nut_peg_clear_hold = 0    # 중간 목표 도착 후 안정화 카운터
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
    amr_move_step = 0  # 이번 이동 시작 후 지난 스텝 수 (가속 램프용)
    current_station = 3  # /amr/goal_pose로 자동 판별, 기본값은 station 3
    wheels_locked = True  # main() 시작 시 lock_amr_base() 이미 호출됨

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
            remove_all_busbar_grip_joints(stage)
            busbar_grasped = False
            active_busbar_body_path = BUSBAR_POLYSHAPE_PATH
            busbar_attach_z = None
            if bolt_camera_xform and bolt_camera_init_pos is not None:
                bolt_camera_xform.set_world_pose(position=bolt_camera_init_pos, orientation=bolt_camera_init_quat)
            if busbar_xform and init_busbar_pos is not None:
                busbar_xform.set_world_pose(position=init_busbar_pos, orientation=init_busbar_quat)
            if nut1_xform and init_nut1_pos is not None:
                nut1_xform.set_world_pose(position=init_nut1_pos, orientation=init_nut1_quat)
            if nut2_xform and init_nut2_pos is not None:
                nut2_xform.set_world_pose(position=init_nut2_pos, orientation=init_nut2_quat)
            for extra_xf, (extra_pos, extra_quat), extra_root, extra_poly in zip(
                extra_nut_xforms, extra_nut_inits, EXTRA_NUT_ROOT_PATHS, EXTRA_NUT_POLYSHAPE_PATHS
            ):
                if extra_xf and extra_pos is not None:
                    extra_xf.set_world_pose(position=extra_pos, orientation=extra_quat)

            # battery4_main과 동일하게 재생 재시작 시 모든 너트를 kinematic glue 상태로
            # 되돌리고 현재 AMR 포즈 기준 로컬 오프셋을 다시 계산한다.
            amr_glue_pos, amr_glue_quat = robot.get_world_pose()
            nut_released = [False] * len(nut_xforms_all)
            for i, (nut_xf, nut_root) in enumerate(zip(nut_xforms_all, nut_roots_all)):
                if nut_xf is None:
                    nut_local_offsets[i] = None
                    continue
                remove_nut_amr_joint(stage, nut_root)
                disable_physics_recursively(stage, nut_root)
                nut_pos, nut_quat = nut_xf.get_world_pose()
                nut_local_offsets[i] = compute_local_offset(
                    amr_glue_pos, amr_glue_quat, nut_pos, nut_quat
                )

            step_count = 0
            grasp_timer = 0
            nut_release_timer = 0

            nut_index = 0
            NUT_SCAN_POS = None
            nut_pick_pos = None
            nut_approach_pos = None
            nut_peg_clear_pos = None
            nut_peg_clear_hold = 0
            bolt_target_pos = None
            bolt_touch_pos = None
            screw_sub = "rotate"
            screw_pass_idx = 0
            screw_pass_theta = 0.0
            stuck_counter = 0

            amr_moving = False
            amr_target_xy_theta = None

        # battery4_main의 너트 AMR glue 추종. PICK된 너트는 nut_released=True가 되어
        # 여기서 제외되고 이후부터 정상 다이나믹 바디로 움직인다.
        if playing:
            amr_glue_pos, amr_glue_quat = robot.get_world_pose()
            for i, nut_xf in enumerate(nut_xforms_all):
                if nut_xf is None or nut_released[i] or nut_local_offsets[i] is None:
                    continue
                glued_pos, glued_quat = compose_world_pose(
                    amr_glue_pos, amr_glue_quat, *nut_local_offsets[i]
                )
                nut_xf.set_world_pose(position=glued_pos, orientation=glued_quat)

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
                    lock_amr_base(stage, NOVA_CARTER_ROOT, robot=robot)
                    wheels_locked = True

            if isaac_node.amr_goal_pose is not None:
                goal_msg = isaac_node.amr_goal_pose
                isaac_node.amr_goal_pose = None
                g_pos = goal_msg.pose.position
                g_ori = goal_msg.pose.orientation
                g_theta = quat_wxyz_to_yaw([g_ori.w, g_ori.x, g_ori.y, g_ori.z])
                amr_target_xy_theta = (g_pos.x, g_pos.y, g_theta)
                amr_moving = True
                amr_move_step = 0
                if wheels_locked:
                    unlock_amr_base(stage, NOVA_CARTER_ROOT, robot=robot)
                    wheels_locked = False
                resolved_station = resolve_station_from_amr_xy(g_pos.x, g_pos.y)
                if resolved_station is not None:
                    current_station = resolved_station
                    print(f"[STATION] 목표좌표로 스테이션 {current_station}번 인식")
                else:
                    print(f"[WARN] 목표좌표({g_pos.x:.4f},{g_pos.y:.4f})가 알려진 스테이션과 안 맞음 -> current_station={current_station} 유지")
                print(f"\n>>> [AMR] 이동 목표 수신 (X={g_pos.x:.4f}, Y={g_pos.y:.4f}, "
                      f"Theta={g_theta:.4f}) -> 바퀴 잠금 해제, 이동 시작")

            amr_pos, amr_quat = robot.get_world_pose()
            amr_yaw = quat_wxyz_to_yaw(amr_quat)

            if amr_moving and amr_target_xy_theta is not None:
                tx, ty, ttheta = amr_target_xy_theta
                dx, dy = tx - amr_pos[0], ty - amr_pos[1]
                dist = math.hypot(dx, dy)
                dyaw = math.atan2(math.sin(ttheta - amr_yaw), math.cos(ttheta - amr_yaw))

                # 목표에 가까워질수록(감속) + 이동을 막 시작했을 때도(가속) 속도를
                # 서서히 바꾼다 - 시작할 때 0에서 최고속으로 한 틱만에 확 튀는 것도,
                # 끝에서 뚝 멈추는 것도 둘 다 관성/물리 반력으로 차체가 흔들리는 원인이다.
                accel_factor = min(1.0, amr_move_step * PHYSICS_DT / AMR_ACCEL_TIME)

                if dist > 1e-6:
                    linear_speed = AMR_LINEAR_SPEED
                    if dist < AMR_DECEL_DIST:
                        linear_speed = AMR_LINEAR_SPEED * (dist / AMR_DECEL_DIST)
                    linear_speed = max(linear_speed * accel_factor, AMR_MIN_LINEAR_SPEED)
                    step_dist = min(dist, linear_speed * PHYSICS_DT)
                    new_x = amr_pos[0] + dx / dist * step_dist
                    new_y = amr_pos[1] + dy / dist * step_dist
                else:
                    new_x, new_y = amr_pos[0], amr_pos[1]

                angular_speed = AMR_ANGULAR_SPEED
                if abs(dyaw) < AMR_DECEL_YAW:
                    angular_speed = AMR_ANGULAR_SPEED * (abs(dyaw) / AMR_DECEL_YAW)
                angular_speed = max(angular_speed * accel_factor, AMR_MIN_ANGULAR_SPEED)
                step_yaw = max(-angular_speed * PHYSICS_DT, min(angular_speed * PHYSICS_DT, dyaw))
                new_yaw = amr_yaw + step_yaw
                amr_move_step += 1

                robot.set_world_pose(
                    position=np.array([new_x, new_y, amr_pos[2]]),
                    orientation=euler_to_quaternion_wxyz(0.0, 0.0, new_yaw),
                )

                # 너트 1/2번은 AMR 섀시에 FixedJoint로 고정되어 있으므로(attach_nut_to_amr),
                # AMR이 set_world_pose로 움직이면 물리 솔버가 알아서 같이 끌고 간다.

                amr_pos = np.array([new_x, new_y, amr_pos[2]])
                amr_yaw = new_yaw

                if dist < AMR_POS_TOL and abs(dyaw) < AMR_YAW_TOL:
                    amr_moving = False
                    amr_target_xy_theta = None
                    print(f"\n[AMR] 목표 지점 도착 (X={new_x:.4f}, Y={new_y:.4f}) -> 바퀴 잠금")
                    try:
                        robot.set_linear_velocity(np.zeros(3))
                        robot.set_angular_velocity(np.zeros(3))
                    except Exception:
                        pass
                    lock_amr_base(stage, NOVA_CARTER_ROOT, robot=robot)
                    wheels_locked = True
                    sync_rmpflow_base_pose()

            sim_pose_msg = Pose2D()
            sim_pose_msg.x = float(amr_pos[0])
            sim_pose_msg.y = float(amr_pos[1])
            sim_pose_msg.theta = float(amr_yaw)
            isaac_node.pub_amr_sim_pose.publish(sim_pose_msg)

            # 버스바를 실제 물리(마찰) 그립으로 물고 있는 동안은 phase="IDLE" 구간
            # (예: PICK_BUSBAR 완료 후 AMR이 배터리 스테이션으로 이동하는 중)에도
            # 계속 CLOSE 명령을 넣어줘야 한다 - 관절 액션이 안 들어가면 드라이브가
            # 느슨해져 그 사이 손아귀 힘이 빠질 수 있다. 위치는 더 이상 스크립트로
            # 덮어쓰지 않는다 - 그리퍼(팔+AMR과 같이 강체로 움직이는 관절 체인)가
            # 실제로 물고 있는 물리 그립이 알아서 따라간다.
            if busbar_grasped:
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))

        # 2. BehaviorNode / ArmNode 명령 분기 처리
        if playing and isaac_node.requested_task:
            task = isaac_node.requested_task
            isaac_node.requested_task = None

            # 안전장치: 팔 Task는 AMR이 도착해 바퀴가 잠긴 상태에서만 수행되어야 한다.
            if not wheels_locked:
                print(f"\n[WARN] [{task}] 바퀴가 아직 안 잠긴 상태 -> 강제 잠금 후 진행")
                lock_amr_base(stage, NOVA_CARTER_ROOT, robot=robot)
                wheels_locked = True
                amr_moving = False
                amr_target_xy_theta = None
                sync_rmpflow_base_pose()

            if task == "SCAN_BATTERY":
                # 볼트쌍 인식 카메라는 씬에 고정돼 있어서 스테이션마다 다른 볼트 위치를
                # 못 본다 - 지금 작업 중인 스테이션의 2번 볼트 좌표 위로 옮겨준다(높이/자세는
                # 원래 값 유지). 이후 Isaac이 SCAN_BATTERY 완료를 보고하면 arm_node가 평소처럼
                # /perception/get_bolt_pair를 호출해서 이 새 위치 기준으로 볼트쌍을 인식한다.
                #
                # 카메라만 옮기는 걸로는 부족하다 - error_fix_node(error_fix.py)는 자기
                # 프로세스 수명 중 딱 한 번만 고정 볼트를 검출하고(bolts_detected 플래그)
                # 그 뒤로는 다시 검출을 안 해서, 카메라를 옮겨도 예전 스테이션에서 잡은
                # 픽셀좌표를 계속 재사용하는 버그가 있었다(실측 확인됨) - 그래서 카메라를
                # 옮긴 직후 RESET_BOLT_DETECTION 명령으로 강제 재탐색시킨다.
                if bolt_camera_xform is not None:
                    bolt2_xy = STATION_BOLT_WORLD_POS.get(current_station, STATION_BOLT_WORLD_POS[3])[2]
                    bolt_camera_xform.set_world_pose(
                        position=np.array([bolt2_xy[0], bolt2_xy[1], bolt_camera_init_pos[2]]),
                        orientation=bolt_camera_init_quat,
                    )
                    reset_cmd = String()
                    reset_cmd.data = "RESET_BOLT_DETECTION"
                    isaac_node.pub_errorfix_command.publish(reset_cmd)
                    print(f"[BOLT CAMERA] 스테이션{current_station} 2번 볼트 좌표로 이동: ({bolt2_xy[0]:.4f}, {bolt2_xy[1]:.4f}) -> 재탐색 명령 전송")
                phase = "INIT_POSE"
                init_pose_only = False
                step_count = 0
                print(f"\n>>> [{task}] 배터리 스캔 요청 수신 -> 1) 초기 관절 정렬 시작")

            elif task == "RETURN_HOME":
                # 너트 작업 전처럼 "관절 각도만 초기 자세로 되돌리고 싶을 때" 쓴다.
                # SCAN_BATTERY와 달리 완료 후 SCAN_APPROACH(배터리 스캔 고도 상승)로
                # 이어지지 않고 바로 끝난다.
                phase = "INIT_POSE"
                init_pose_only = True
                step_count = 0
                print(f"\n>>> [{task}] 초기 관절 자세 복귀 시작")

            elif task == "SCAN_BUSBAR":
                # AMR 차체의 정차 좌표를 로봇 팔 EE 목표로 사용하면 스테이션 4/5에서
                # 팔이 엉뚱한 월드 좌표로 향한다. 모든 스테이션에서 현재 EE 기준의
                # 검증된 상대 오프셋으로 버스바 스캔 위치를 계산한다.
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                BUSBAR_SCAN_POS = np.array([
                    cur_pos[0] - 0.5,
                    cur_pos[1] + 0.5,
                    BUSBAR_SCAN_Z,
                ])
                # 너트1 때와 같은 문제 - XY 이동과 orientation 변경(quat_busbar)을 동시에
                # 크게 시키면 RMPFlow가 팔꿈치/손목을 꼬아서 돌아가는 경로를 잡는다(스테이션4
                # 버스바 스캔에서 실측 확인됨). 제자리에서 방향+고도만 먼저 맞추고
                # (SCAN_BUSBAR_LIFT), 그 다음 같은 높이/방향을 유지한 채 옆으로만 이동
                # (SCAN_BUSBAR_APPROACH)시켜서 회전과 이동을 분리한다.
                BUSBAR_SCAN_LIFT_POS = np.array([cur_pos[0], cur_pos[1], BUSBAR_SCAN_Z])
                phase = "SCAN_BUSBAR_LIFT"
                step_count = 0
                print(f"\n>>> [{task}] 1) 방향 정렬 및 스캔 고도 상승 시작 (스테이션{current_station}, Target: X={BUSBAR_SCAN_LIFT_POS[0]:.3f}, Y={BUSBAR_SCAN_LIFT_POS[1]:.3f}, Z={BUSBAR_SCAN_LIFT_POS[2]:.3f})")

            elif task == "PICK_BUSBAR":
                if isaac_node.latest_target_pose is not None:
                    update_target_positions(isaac_node.latest_target_pose)

                # 스테이션 3은 검증된 기존 경로를 유지한다. 스테이션 4/5는 비전이
                # 가리킨 파지점 근처의 실제 버스바 body를 찾아야 Z_busbar3가 대신
                # 그리퍼에 묶이는 문제가 생기지 않는다.
                remove_all_busbar_grip_joints(stage)
                if current_station == 3:
                    active_busbar_body_path = (
                        BUSBAR_POLYSHAPE_PATH
                        if stage.GetPrimAtPath(BUSBAR_POLYSHAPE_PATH).IsValid()
                        else None
                    )
                else:
                    active_busbar_body_path = find_busbar_body_near(
                        stage, BUSBAR_PICK_POS[:2]
                    )

                if active_busbar_body_path is None:
                    print(f"\n[ERROR] 스테이션{current_station} 파지점 근처에서 "
                          "버스바 Mesh rigid body를 찾지 못했습니다.")
                    publish_status("FAILURE:BUSBAR_PRIM_NOT_FOUND")
                    phase = "IDLE"
                    continue

                # 성공본은 Z_busbar3만 물리를 켰기 때문에 스테이션 4/5 body에 조인트가
                # 생성돼도 정적 상태로 남을 수 있었다. 실제 선택된 버스바 전체 root의
                # RigidBody/Collision을 활성화한 뒤에 접근한다.
                active_busbar_root_path = busbar_root_path_for_body(
                    active_busbar_body_path
                )
                enable_physics_recursively(stage, active_busbar_root_path)
                busbar_attach_z = None
                print(f"[BUSBAR PHYSICS ENABLE] {active_busbar_root_path}")

                phase = "BUSBAR_APPROACH"
                step_count = 0
                print(f"\n>>> [{task}] 버스바 상공 접근(Z=0.6m) 시작 "
                      f"(Prim: {active_busbar_body_path})")

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
                nut_slot = 1 if task == "SCAN_NUT1" else 2
                nut_index = STATION_NUT_INDICES.get(current_station, (1, 2))[nut_slot - 1]
                if HOME_EE_POS is not None:
                    # NUT_OFFSET_FROM_HOME(하드코딩 오프셋) 대신 지금 씬의 너트 실제 world
                    # 위치를 직접 읽는다 - 너트를 옮기거나(PolyShape 물리 위치 정렬 등) 씬을
                    # 재배치할 때마다 오프셋 표를 다시 실측/갱신해야 하는 문제가 있었다.
                    nut_xf, nut_label = resolve_nut_assets(nut_index, nut1_xform, nut2_xform, extra_nut_xforms)
                    if nut_xf is None:
                        print(f"\n[ERROR] [{task}] 너트 {nut_index}번 Xform을 찾을 수 없습니다.")
                        publish_status("FAILURE:NUT_XFORM_NOT_FOUND")
                    else:
                        nut_live_pos, _ = nut_xf.get_world_pose()
                        NUT_SCAN_POS = np.array([
                            float(nut_live_pos[0]),
                            float(nut_live_pos[1]),
                            NUT_SCAN_Z,
                        ])
                        # XY와 orientation을 동시에 크게 바꾸면 RMPFlow가 팔꿈치/손목을 꼬아서
                        # 돌아가는 경로를 잡는 문제가 있었다(실측 확인됨) - 먼저 지금 있는
                        # XY에서 방향만 quat_nut로 맞추고 높이를 스캔 고도로 올린 뒤(NUT_SCAN_LIFT),
                        # 그 다음에 같은 높이/방향을 유지한 채 옆으로만 이동(NUT_SCAN_APPROACH)
                        # 시켜서 회전과 이동을 분리한다.
                        cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                        NUT_SCAN_LIFT_POS = np.array([cur_pos[0], cur_pos[1], NUT_SCAN_Z])
                        phase = "NUT_SCAN_LIFT"
                        step_count = 0
                        print(f"\n>>> [{task}] 1) 방향 정렬 및 스캔 고도 상승 시작 (Target: X={NUT_SCAN_LIFT_POS[0]:.3f}, Y={NUT_SCAN_LIFT_POS[1]:.3f}, Z={NUT_SCAN_LIFT_POS[2]:.3f})")
                else:
                    print(f"\n[ERROR] [{task}] 초기 위치(HOME_EE_POS)가 없습니다. 먼저 INIT_POSE가 수행되어야 합니다.")
                    publish_status("FAILURE:NO_HOME_POSE")

            elif task in ("PICK_NUT1", "PICK_NUT2"):
                nut_slot = 1 if task == "PICK_NUT1" else 2
                nut_index = STATION_NUT_INDICES.get(current_station, (1, 2))[nut_slot - 1]
                if HOME_EE_POS is not None:
                    # battery4_main 방식: 실제 파지를 시작할 때 대상 너트만 AMR glue에서
                    # 해제하고 물리/콜리전을 활성화한다.
                    nut_array_index = nut_index - 1
                    nut_released[nut_array_index] = True
                    remove_nut_amr_joint(stage, nut_roots_all[nut_array_index])
                    enable_physics_recursively(stage, nut_roots_all[nut_array_index])

                    # glue 해제 직전의 실측 위치를 그대로 파지 XY로 사용한다.
                    nut_xf, nut_label = resolve_nut_assets(nut_index, nut1_xform, nut2_xform, extra_nut_xforms)
                    if nut_xf is None:
                        print(f"\n[ERROR] [{task}] 너트 {nut_index}번 Xform을 찾을 수 없습니다.")
                        publish_status("FAILURE:NUT_XFORM_NOT_FOUND")
                    else:
                        nut_live_pos, _ = nut_xf.get_world_pose()
                        pick_x, pick_y = float(nut_live_pos[0]), float(nut_live_pos[1])
                        nut_pick_pos = np.array([pick_x, pick_y, NUT_PICK_Z])
                        nut_approach_pos = np.array([pick_x, pick_y, NUT_APPROACH_Z])
                        phase = "NUT_APPROACH"
                        step_count = 0
                        print(f"\n>>> [{task}] 너트 {nut_index}번 상공 접근 시작 (Target: X={nut_approach_pos[0]:.3f}, Y={nut_approach_pos[1]:.3f})")
                else:
                    print(f"\n[ERROR] [{task}] 초기 위치(HOME_EE_POS)가 없습니다. 먼저 INIT_POSE가 수행되어야 합니다.")
                    publish_status("FAILURE:NO_HOME_POSE")

            elif task in ("ASSEMBLE_NUT1", "ASSEMBLE_NUT2"):
                nut_slot = 1 if task == "ASSEMBLE_NUT1" else 2
                nut_index = STATION_NUT_INDICES.get(current_station, (1, 2))[nut_slot - 1]
                bolt_world_xy = STATION_BOLT_WORLD_POS.get(current_station, STATION_BOLT_WORLD_POS[3])[nut_slot]
                bolt_target_pos = np.array([bolt_world_xy[0], bolt_world_xy[1], 0.0])
                bolt_touch_pos = np.array([bolt_target_pos[0], bolt_target_pos[1], 0.3697])
                phase = "MOVE_TO_BOLT_NUT"
                step_count = 0
                print(f"\n>>> [{task}] 너트 {nut_index}번(스테이션{current_station} 슬롯{nut_slot}) -> 볼트 {nut_slot}번 체결 시작 "
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

                all_joints = robot.get_joint_positions()
                cur_arm_joints = all_joints[arm_dof_indices]

                max_step = ARM_INIT_MAX_JOINT_SPEED * PHYSICS_DT
                delta = TARGET_INIT_JOINTS - cur_arm_joints
                ramped_target = cur_arm_joints + np.clip(delta, -max_step, max_step)

                robot.apply_action(
                    ArticulationAction(
                        joint_positions=ramped_target,
                        joint_indices=arm_dof_indices
                    )
                )
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                joint_err = np.linalg.norm(cur_arm_joints - TARGET_INIT_JOINTS)

                if joint_err < JOINT_TOLERANCE:
                    ee_xf = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}")
                    HOME_EE_POS = np.array(ee_xf.ExtractTranslation(), dtype=float)
                    ee_quat = ee_xf.ExtractRotationQuat()
                    SCAN_POS = np.array([HOME_EE_POS[0], HOME_EE_POS[1], SCAN_APPROACH_Z])
                    # AMR이 어느 방향을 보고 있든, INIT_POSE가 만들어낸 실제 EE 자세를
                    # 그대로 유지한 채 Z만 올린다 (world 절대 방향을 새로 강제하지 않음).
                    scan_hold_quat = np.array(
                        [ee_quat.GetReal(), *[float(v) for v in ee_quat.GetImaginary()]]
                    )

                    print(f"[OK] 1) 관절 정렬 완료! (EE Pos: {HOME_EE_POS[0]:.3f}, {HOME_EE_POS[1]:.3f}, {HOME_EE_POS[2]:.3f})")

                    if init_pose_only:
                        print(" -> 초기 자세 복귀만 요청됨 - 스캔 이동 없이 종료")
                        publish_progress("RETURN_HOME_COMPLETE", 100.0)
                        publish_status("SUCCESS")
                        phase = "IDLE"
                    else:
                        print(f" -> 2) 초기 위치 기준 Z={SCAN_APPROACH_Z}m 상승 시작 (Target: {SCAN_POS[0]:.3f}, {SCAN_POS[1]:.3f}, {SCAN_POS[2]:.3f})")
                        phase = "SCAN_APPROACH"
                    step_count = 0
                elif step_count > 300:
                    print(f"\n[ERROR] INIT_POSE 관절 정렬 실패 (joint_err={joint_err:.4f}rad) - 실패 처리")
                    publish_status("FAILURE:INIT_POSE_TIMEOUT")
                    phase = "IDLE"

            # [STEP 0-B] 배터리 스캔 고도 상승 (Z = 0.7m)
            elif phase == "SCAN_APPROACH":
                publish_progress("SCAN_NAV", 50.0)
                # world 절대 방향을 새로 계산하지 않고, INIT_POSE 완료 시점에 저장해둔
                # 실제 EE 자세(scan_hold_quat)를 그대로 유지한 채 Z만 올린다.
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                actions = arm_controller.forward(
                    target_end_effector_position=SCAN_POS,
                    target_end_effector_orientation=scan_hold_quat
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                xy_err = math.hypot(SCAN_POS[0] - cur_pos[0], SCAN_POS[1] - cur_pos[1])
                z_err = abs(SCAN_POS[2] - cur_pos[2])

                if step_count % 30 == 0:
                    print(f"[SCAN_NAV] Step={step_count} | EE=({cur_pos[0]:.4f}, {cur_pos[1]:.4f}, {cur_pos[2]:.4f}) | "
                          f"Target=({SCAN_POS[0]:.4f}, {SCAN_POS[1]:.4f}, {SCAN_POS[2]:.4f}) | "
                          f"XY err={xy_err*1000:.1f}mm | Z err={z_err*1000:.1f}mm")

                # Z만 올리는 단계라 Z 도착 여부를 기준으로 판정하고, XY는 크게 안 벗어났는지만 확인한다
                # (기존엔 XYZ 통합 거리(PICK_TOLERANCE_STRICT)로만 판정해서, Z는 도착했는데 XY가
                # 몇 mm~cm 밀리면 통과를 못 했다 - MAX_STUCK_STEPS가 사실상 무한대라 완화 조건도 안 먹었음).
                if z_err < 0.005 and xy_err < 0.02:
                    print(f"[OK] 배터리 스캔 위치 도착 완료! ({cur_pos[0]:.3f}, {cur_pos[1]:.3f}, {cur_pos[2]:.3f})")
                    publish_progress("SCAN_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"
                elif step_count > 600:
                    print(f"\n[ERROR] SCAN_APPROACH Timeout | XY err={xy_err*1000:.1f}mm, Z err={z_err*1000:.1f}mm")
                    publish_status("FAILURE:SCAN_APPROACH_TIMEOUT")
                    phase = "IDLE"

            # [STEP 0.5] 방향 정렬 + 버스바 스캔 고도 상승 (제자리에서 quat_busbar로 회전, XY는 그대로)
            elif phase == "SCAN_BUSBAR_LIFT":
                publish_progress("SCAN_BUSBAR_LIFT", 30.0)
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                actions = arm_controller.forward(
                    target_end_effector_position=BUSBAR_SCAN_LIFT_POS,
                    target_end_effector_orientation=quat_busbar
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                err = math.dist(cur_pos, tuple(BUSBAR_SCAN_LIFT_POS))
                if err < 0.02 or step_count > 300:
                    print("[OK] 방향 정렬 및 고도 상승 완료 -> 버스바 스캔 위치로 수평 이동")
                    phase = "SCAN_BUSBAR_APPROACH"
                    step_count = 0

            # [STEP 1] 버스바 스캔 위치 이동
            elif phase == "SCAN_BUSBAR_APPROACH":
                publish_progress("SCAN_BUSBAR_NAV", 50.0)
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                actions = arm_controller.forward(
                    target_end_effector_position=BUSBAR_SCAN_POS,
                    target_end_effector_orientation=quat_busbar
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                xy_err = math.hypot(BUSBAR_SCAN_POS[0] - cur_pos[0], BUSBAR_SCAN_POS[1] - cur_pos[1])
                z_err = abs(BUSBAR_SCAN_POS[2] - cur_pos[2])

                if step_count % 30 == 0:
                    print(f"[SCAN_BUSBAR_NAV] Step={step_count} | EE=({cur_pos[0]:.4f}, {cur_pos[1]:.4f}, {cur_pos[2]:.4f}) | "
                          f"Target=({BUSBAR_SCAN_POS[0]:.4f}, {BUSBAR_SCAN_POS[1]:.4f}, {BUSBAR_SCAN_POS[2]:.4f}) | "
                          f"XY err={xy_err*1000:.1f}mm | Z err={z_err*1000:.1f}mm")

                # 기존엔 XYZ 통합 거리(PICK_TOLERANCE_STRICT=10mm)로만 판정해서, MAX_STUCK_STEPS가
                # 사실상 무한대라 완화 조건도 안 먹혔다 (SCAN_APPROACH와 동일한 버그).
                if xy_err < 0.02 and z_err < 0.02:
                    print(f"[OK] 버스바 스캔 위치 도착 완료! ({cur_pos[0]:.3f}, {cur_pos[1]:.3f}, {cur_pos[2]:.3f})")
                    publish_progress("SCAN_BUSBAR_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"
                elif step_count > 600:
                    print(f"\n[ERROR] SCAN_BUSBAR_APPROACH Timeout | XY err={xy_err*1000:.1f}mm, Z err={z_err*1000:.1f}mm")
                    publish_status("FAILURE:SCAN_BUSBAR_TIMEOUT")
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

            # [4단계] 물리 파지 + FixedJoint 고정 - 버스바의 물리(중력/충돌)는 계속
            # 켜둔 채 그리퍼로 실제로 닫아서 무는데, AMR이 set_world_pose로 매 틱
            # 순간이동하는(최고속에서 틱당 최대 5mm) 반면 버스바 두께는 3mm뿐이라
            # 마찰만으로는 그 이동을 따라가지 못하고 미끄러져 빠진다(실측 확인됨).
            # 그래서 손가락이 다 닫히고 살짝 안정화된 뒤 attach_busbar_to_gripper로
            # 그리퍼(link_6)와 버스바 사이에 실제 FixedJoint를 만들어 완전히 고정한다.
            elif phase == "BUSBAR_GRASP":
                publish_progress("GRASPING", 75.0)

                actions = arm_controller.forward(
                    target_end_effector_position=BUSBAR_PICK_POS,
                    target_end_effector_orientation=quat_busbar
                )
                robot.apply_action(actions)

                grasp_timer += 1
                ramp_frac = min(grasp_timer / GRIP_CLOSE_RAMP_STEPS, 1.0)
                grip_target = ramp_frac * GRIPPER_CLOSE
                robot.gripper.apply_action(ArticulationAction(joint_positions=grip_target))

                if grasp_timer >= GRIP_CLOSE_RAMP_STEPS + GRIP_SETTLE_STEPS:
                    joint = attach_busbar_to_gripper(
                        stage, ee_path, robot, active_busbar_body_path
                    )
                    if joint is None:
                        print(f"\n[ERROR] 버스바 파지 실패 "
                              f"(prim 경로 확인 필요: {active_busbar_body_path})")
                        publish_status("FAILURE:BUSBAR_ATTACH_FAILED")
                        phase = "IDLE"
                    else:
                        busbar_grasped = True
                        busbar_attach_pos = world_xf(
                            stage, active_busbar_body_path
                        ).ExtractTranslation()
                        busbar_attach_z = float(busbar_attach_pos[2])
                        print("[OK] 3. 버스바 물리 파지 + FixedJoint 고정 완료 -> 4. 안전 고도 상승")
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


                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(BUSBAR_LIFT_MOVE_POS))

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    busbar_now_pos = world_xf(
                        stage, active_busbar_body_path
                    ).ExtractTranslation()
                    busbar_lift_rise = (
                        float(busbar_now_pos[2]) - busbar_attach_z
                        if busbar_attach_z is not None
                        else 0.0
                    )

                    if busbar_lift_rise < BUSBAR_MIN_LIFT_RISE:
                        print(f"\n[ERROR] 버스바 미파지: EE만 상승하고 "
                              f"{active_busbar_body_path}는 "
                              f"{busbar_lift_rise*1000:.1f}mm만 상승했습니다.")
                        detach_busbar_from_gripper(
                            stage, active_busbar_body_path
                        )
                        busbar_grasped = False
                        publish_status("FAILURE:BUSBAR_NOT_LIFTED")
                        phase = "IDLE"
                    else:
                        print(f"\n★ [PICK_BUSBAR SUCCESS] 실제 버스바 "
                              f"{busbar_lift_rise*1000:.1f}mm 상승 확인!")
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

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                dist_err = math.dist(cur_pos, tuple(target_mid_pos))

                if cur_pos[2] <= BUSBAR_RELEASE_Z or dist_err < INSERT_TOLERANCE_STRICT:
                    # 실제 물리 그립 상태라 여기서 busbar 위치를 강제로 순간이동(snap)
                    # 시키면 안 된다 - 그리퍼가 여전히 물고 있는 채로 물체만 텔레포트하면
                    # 다음 physics step에서 손가락과 겹침이 발생해 튕겨나갈 수 있다.
                    # 착좌 정확도는 이제 IK가 내려간 실제 위치(target_mid_pos)에 의존한다.
                    print(f"\n[OK] 버스바 안착 체결 완료 (EE Z: {cur_pos[2]:.4f}m)!")

                    phase = "BUSBAR_RELEASE_AND_RETRACT"
                    step_count = 0

            # [9단계] 버스바 해제 및 안전 상공 이탈
            elif phase == "BUSBAR_RELEASE_AND_RETRACT":
                publish_progress("BUSBAR_RETRACT", 95.0)

                if busbar_grasped:
                    # 그리퍼를 열기 전에 FixedJoint부터 풀어야 한다 - 안 그러면 조인트가
                    # 버스바를 계속 붙잡고 있어서 그리퍼가 열려도 안 떨어진다.
                    detach_busbar_from_gripper(stage, active_busbar_body_path)
                    busbar_grasped = False

                retract_pos = np.array([target_mid_pos[0], target_mid_pos[1], BATTERY_CENTER_Z])
                actions = arm_controller.forward(
                    target_end_effector_position=retract_pos,
                    target_end_effector_orientation=euler_to_quaternion_wxyz(0.0, 3.1415, 0.0)
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

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
            # [9.5단계] 방향 정렬 + 스캔 고도 상승 (제자리에서 quat_nut로 회전, XY는 그대로)
            elif phase == "NUT_SCAN_LIFT":
                publish_progress("NUT_SCAN_LIFT", 30.0)
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                actions = arm_controller.forward(target_end_effector_position=NUT_SCAN_LIFT_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                err = math.dist(cur_pos, tuple(NUT_SCAN_LIFT_POS))
                if err < 0.02 or step_count > 300:
                    print(f"[OK] 방향 정렬 및 고도 상승 완료 -> 너트 스캔 위치로 수평 이동")
                    phase = "NUT_SCAN_APPROACH"
                    step_count = 0

            # [10단계] 너트 스캔 위치 이동
            elif phase == "NUT_SCAN_APPROACH":
                publish_progress("NUT_SCAN_NAV", 50.0)
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                actions = arm_controller.forward(target_end_effector_position=NUT_SCAN_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                xy_err = math.hypot(NUT_SCAN_POS[0] - cur_pos[0], NUT_SCAN_POS[1] - cur_pos[1])
                z_err = abs(NUT_SCAN_POS[2] - cur_pos[2])

                if step_count % 30 == 0:
                    print(f"[NUT_SCAN_NAV] Step={step_count} | EE=({cur_pos[0]:.4f}, {cur_pos[1]:.4f}, {cur_pos[2]:.4f}) | "
                          f"Target=({NUT_SCAN_POS[0]:.4f}, {NUT_SCAN_POS[1]:.4f}, {NUT_SCAN_POS[2]:.4f}) | "
                          f"XY err={xy_err*1000:.1f}mm | Z err={z_err*1000:.1f}mm")

                if xy_err < 0.02 and z_err < 0.02:
                    print(f"[OK] 너트 스캔 위치 도착 완료! ({cur_pos[0]:.3f}, {cur_pos[1]:.3f}, {cur_pos[2]:.3f})")
                    publish_progress("NUT_SCAN_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"
                elif step_count > 600:
                    print(f"\n[ERROR] NUT_SCAN_APPROACH Timeout | XY err={xy_err*1000:.1f}mm, Z err={z_err*1000:.1f}mm")
                    publish_status("FAILURE:NUT_SCAN_TIMEOUT")
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
                    # 성공했던 66f1825 방식: 너트를 AMR에 고정해 둔 채 손가락부터
                    # 닫는다. 열린 상태에서 조인트를 먼저 풀면 너트가 움직여 파지
                    # 중심에서 벗어날 수 있다.
                    print(f"[OK] {nut_label} 하강 완료! -> 물리 파지(Gripper Close) 시작")
                    phase = "NUT_GRASP"
                    grasp_timer = 0

            # [13단계] battery4_main과 동일한 순수 접촉/마찰 기반 너트 파지.
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
                    # 바로 다음 태스크로 넘기지 않고, 현재 XY를 고정한 채 peg 길이보다
                    # 충분히 위로 먼저 수직 이탈한다. 이 구간에서도 그리퍼는 계속 닫는다.
                    ee_grasp_pos = np.asarray(
                        world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation(),
                        dtype=float,
                    )
                    clear_z = min(NUT_APPROACH_Z, ee_grasp_pos[2] + NUT_PEG_CLEARANCE_Z)
                    nut_peg_clear_pos = np.array([ee_grasp_pos[0], ee_grasp_pos[1], clear_z])
                    nut_peg_clear_hold = 0
                    print(f"[OK] {nut_label} 물리 파지 완료! -> XY 고정 수직 이탈 "
                          f"{NUT_PEG_CLEARANCE_Z*1000:.0f}mm 시작 (Target Z={clear_z:.4f})")
                    phase = "NUT_LIFT_CLEAR_PEG"
                    step_count = 0

            # [13.5단계] 공급대 peg에서 완전히 빠질 때까지 XY 고정 수직 상승
            elif phase == "NUT_LIFT_CLEAR_PEG":
                publish_progress("NUT_LIFT_CLEAR_PEG", 75.0)
                actions = arm_controller.forward(
                    target_end_effector_position=nut_peg_clear_pos,
                    target_end_effector_orientation=quat_nut,
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                cur_pos = np.asarray(
                    world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation(),
                    dtype=float,
                )
                z_err = abs(cur_pos[2] - nut_peg_clear_pos[2])
                xy_err = float(np.linalg.norm(cur_pos[:2] - nut_peg_clear_pos[:2]))

                # 목표에 실제 도착한 상태가 연속으로 유지되어야 다음 단계로 넘어간다.
                if z_err < NUT_PEG_CLEAR_TOLERANCE and xy_err < PICK_TOLERANCE_STRICT:
                    nut_peg_clear_hold += 1
                else:
                    nut_peg_clear_hold = 0

                if nut_peg_clear_hold >= NUT_PEG_CLEAR_HOLD_STEPS:
                    print(f"[OK] peg 수직 이탈 완료 (Z={cur_pos[2]:.4f}) -> "
                          f"안전 상공({NUT_APPROACH_Z}m) 상승")
                    phase = "NUT_LIFT"
                    step_count = 0

            # [14단계] peg 이탈 완료 후 안전 상공 상승
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

                    # [진단] 너트가 실제로 EE 근처까지 따라 올라왔는지 실측 위치로 확인
                    nut_xf_cur = resolve_nut_assets(nut_index, nut1_xform, nut2_xform, extra_nut_xforms)[0]
                    nut_now_pos, _ = nut_xf_cur.get_world_pose() if nut_xf_cur else (None, None)
                    ee_now_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                    print(f"[진단] 너트 실측 위치={tuple(round(v,4) for v in nut_now_pos) if nut_now_pos is not None else None} "
                          f"| EE 위치={tuple(round(v,4) for v in ee_now_pos)}")

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
                            # 체결이 끝난 너트에 PhysX 해석을 계속 적용하면 볼트와의 미세
                            # 관통을 매 프레임 밀어내면서 떨림/충돌 반력이 생긴다. 현재
                            # 체결 위치를 유지한 채 해당 너트의 rigid body와 collision을 끈다.
                            seated_nut_root, _ = nut_paths_for_index(nut_index)
                            disable_physics_recursively(stage, seated_nut_root)
                            robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
                            print(f"\n[OK] 너트 {nut_index}번 체결 완료! -> "
                                  f"물리/충돌 비활성화({seated_nut_root}) -> "
                                  "꼬인 방향 유지한 채 수직 상승 시작")
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

                all_joints = robot.get_joint_positions()
                cur_arm_joints = all_joints[arm_dof_indices]

                max_step = ARM_INIT_MAX_JOINT_SPEED * PHYSICS_DT
                delta = TARGET_INIT_JOINTS - cur_arm_joints
                ramped_target = cur_arm_joints + np.clip(delta, -max_step, max_step)

                robot.apply_action(
                    ArticulationAction(
                        joint_positions=ramped_target,
                        joint_indices=arm_dof_indices
                    )
                )
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

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

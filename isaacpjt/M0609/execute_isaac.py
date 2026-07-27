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
8. PICK_NUT1 / PICK_NUT2   : 비전으로 검출된 너트 좌표로 접근, 물리 파지(Gripper Close) 및 상승
9. ASSEMBLE_NUT1 / ASSEMBLE_NUT2 : 실시간 배터리 중심 좌표(target_mid_pos) 기준 사전 계산된
   상대 오프셋을 적용해 산출한 볼트 목표 좌표로 이동, 착좌 및 Screwing 체결(Regrasp 포함)
"""

import os
import sys
import math
import numpy as np
from pathlib import Path

# 1. Isaac SimulationApp 초기화
from isaacsim import SimulationApp

_HEADLESS = (
    os.environ.get("AMR_HEADLESS") == "1"
    or "--headless" in sys.argv[1:]
)
_DEBUG_NO_TIMEOUTS = (
    os.environ.get("AMR_DEBUG_NO_TIMEOUTS") == "1"
    or "--debug-no-timeouts" in sys.argv[1:]
)
simulation_app = SimulationApp({"headless": _HEADLESS})

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

sys.stdout.reconfigure(line_buffering=True)

# 2. USD 및 Isaac Core Imports
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.core.utils.types import ArticulationAction

# 3. ROS 2 Imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float32

# RMPFlow Controller 경로 설정
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "rmpflow"))
from m0609_rmpflow_controller import RMPFlowController

_DRIVE_DIR = (
    _THIS_DIR.parents[1] / "src" / "amr_node" / "scrips_drive"
)
sys.path.insert(0, str(_DRIVE_DIR))
from drive_controller import PoseState
from vision_goal import (
    VisionUnavailable,
    compute_busbar_pick_targets,
    compute_busbar_scan_xy,
    select_nearest_busbar_path,
)
from integrated_drive_bridge import (
    IntegratedDriveConfig,
    IntegratedPhysicsDriveBridge,
)
from workcell_transform import rotate_busbar_workcell

# ══════════════════════════════════════════════════════════════════════════
#  [A] 설정 및 파라미터
# ══════════════════════════════════════════════════════════════════════════
USD_PATH = os.environ.get(
    "INTEGRATED_USD_PATH",
    str(_THIS_DIR.parents[1] / "src/Collected_Busbar_amr/Busbar.usd"),
)

# 고정 카메라 영상은 실제 Camera prim에서 렌더링하고, TF만 그 아래의 ROS optical
# frame에서 발행해야 한다. optical Xform을 RenderProduct의 cameraPrim으로 지정하면
# 렌즈 속성을 찾지 못한 Isaac이 다른 Camera로 fallback하여 영상과 TF가 어긋난다.
FIXED_CAMERA_RENDER_PRODUCTS = {
    "/World/Graph/camera_graph/RenderProduct_busbar": "/World/Camera_busbar",
    "/World/Graph/camera_graph/RenderProduct_bolt": "/World/Camera_bolt",
}
FIXED_CAMERA_OPTICAL_FRAMES = (
    "/World/Camera_busbar/busbar_cam_optical_frame",
    "/World/Camera_bolt/bolt_cam_optical_frame",
)
TF_PUBLISH_NODE = "/World/ActionGraph/ros2_publish_transform_tree"

NOVA_CARTER_ROOT = "/World/Nova_Carter/chassis_link"
M0609_PATH       = "/World/m0609"
EE_LINK_NAME     = "link_6"
GRIPPER_JOINTS   = ["finger_joint", "right_inner_knuckle_joint"]

BUSBAR_PRIM_CANDIDATES = (
    ("/World/Z_busbar3", "/World/Z_busbar3/Mesh"),
    ("/World/Z_busbar3_01", "/World/Z_busbar3_01/Mesh"),
    ("/World/Z_busbar3_02", "/World/Z_busbar3_02/Mesh"),
)
BUSBAR_PRIM_MAX_ASSOCIATION_M = 0.30

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
# 너트 스캔 위치: 버스바 체결 완료 시점의 EE 위치 기준 상대 이동 (X, Y) + 고정 고도 (Z)
NUT_SCAN_OFFSET_X = -0.5
NUT_SCAN_OFFSET_Y = -0.3
NUT_SCAN_Z        = 0.9

NUT_HEIGHT         = 0.0095
NUT_GRASP_Z_LOCAL  = NUT_HEIGHT + 0.023
NUT_SUPPLY_TABLE_Z = 0.72                                   # 너트 공급대 높이
NUT_PICK_Z         = NUT_SUPPLY_TABLE_Z - (NUT_GRASP_Z_LOCAL - 0.0395)
NUT_APPROACH_Z     = 0.8                                     # 너트 파지 상공 고도
BOLT_APPROACH_Z    = 0.6                                     # 볼트 체결 상공 고도

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
MAX_STUCK_STEPS          = 1800      # Phase 타임아웃 기준
SCAN_ORIENTATION_TOLERANCE_RAD = np.radians(5.0)
SCAN_BUSBAR_TIMEOUT_STEPS = 1200     # 20초: ArmNode 30초 timeout 전에 정지
PHYSICS_DT               = 1.0 / 60.0

# 고정 busbar camera가 선택한 실제 world 좌표를 중심으로 손목 카메라가
# 제한 범위 안에서 반복 탐색한다. 절대 좌표나 과거 station 좌표는 없다.
_BUSBAR_SCAN_SEARCH_STEP_M = float(
    os.environ.get("BUSBAR_SCAN_SEARCH_STEP_M", "0.05")
)
if (
    not math.isfinite(_BUSBAR_SCAN_SEARCH_STEP_M)
    or _BUSBAR_SCAN_SEARCH_STEP_M <= 0.0
    or _BUSBAR_SCAN_SEARCH_STEP_M > 0.10
):
    raise ValueError(
        "BUSBAR_SCAN_SEARCH_STEP_M은 0보다 크고 0.10m 이하여야 합니다"
    )

URDF_PATH        = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
ROBOT_DESC_PATH  = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
RMPFLOW_CFG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")


def configure_fixed_camera_bridges(stage):
    """현재 USD 카메라 자세를 유지한 채 RenderProduct와 optical TF 관계만 보정한다."""
    for render_product_path, camera_path in FIXED_CAMERA_RENDER_PRODUCTS.items():
        render_product = stage.GetPrimAtPath(render_product_path)
        camera = stage.GetPrimAtPath(camera_path)
        if not render_product.IsValid() or not camera.IsValid():
            raise RuntimeError(
                f"고정 카메라 prim 누락: render_product={render_product_path}, "
                f"camera={camera_path}"
            )

        camera_rel = render_product.GetRelationship("inputs:cameraPrim")
        expected = [Sdf.Path(camera_path)]
        if list(camera_rel.GetTargets()) != expected:
            camera_rel.SetTargets(expected)
            print(f"[camera] {render_product_path} -> {camera_path} 보정")

    tf_node = stage.GetPrimAtPath(TF_PUBLISH_NODE)
    if not tf_node.IsValid():
        raise RuntimeError(f"TF publish node 누락: {TF_PUBLISH_NODE}")

    for optical_path in FIXED_CAMERA_OPTICAL_FRAMES:
        if not stage.GetPrimAtPath(optical_path).IsValid():
            raise RuntimeError(f"고정 카메라 optical frame 누락: {optical_path}")

    target_rel = tf_node.GetRelationship("inputs:targetPrims")
    fixed_related = set(FIXED_CAMERA_RENDER_PRODUCTS.values()) | set(FIXED_CAMERA_OPTICAL_FRAMES)
    targets = [
        target for target in target_rel.GetTargets()
        if str(target) not in fixed_related
    ]
    targets.extend(Sdf.Path(path) for path in FIXED_CAMERA_OPTICAL_FRAMES)
    if list(target_rel.GetTargets()) != targets:
        target_rel.SetTargets(targets)
        print("[camera] busbar/bolt optical TF target 보정")


def configure_ros_domain(stage):
    """USD ActionGraph의 ROS2Context를 현재 ROS_DOMAIN_ID에 맞춘다."""
    domain_id = int(os.environ.get("ROS_DOMAIN_ID", "0"))
    if not 0 <= domain_id <= 232:
        raise RuntimeError("ROS_DOMAIN_ID는 0~232 범위여야 합니다")
    found = 0
    for prim in stage.Traverse():
        node_type = prim.GetAttribute("node:type")
        if (
            not node_type.IsValid()
            or node_type.Get()
            != "isaacsim.ros2.bridge.ROS2Context"
        ):
            continue
        domain_attr = prim.GetAttribute("inputs:domain_id")
        env_attr = prim.GetAttribute("inputs:useDomainIDEnvVar")
        if not domain_attr.IsValid() or not env_attr.IsValid():
            raise RuntimeError(
                f"ROS2Context 속성 누락: {prim.GetPath()}"
            )
        domain_attr.Set(domain_id)
        env_attr.Set(True)
        found += 1
    if found == 0:
        raise RuntimeError("USD에서 ROS2Context를 찾지 못했습니다")
    print(f"[ROS2] USD Context {found}개 domain={domain_id} 적용")


# ══════════════════════════════════════════════════════════════════════════
#  [B] 비전 브릿지 및 유틸리티 함수
# ══════════════════════════════════════════════════════════════════════════
class Execute_Isaac_Busar(Node):
    """ArmNode / BehaviorNode / VisionNode 통신 브릿지"""
    def __init__(self, stage):
        super().__init__("execute_isaac_busar")
        self._stage = stage
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

    def _on_target_pose(self, msg: PoseStamped):
        self.latest_target_pose = msg

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


def quaternion_angular_error_wxyz(current, target):
    """부호가 같은 회전을 뜻하는 q/-q를 고려한 최소 회전 오차."""
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    current_norm = np.linalg.norm(current)
    target_norm = np.linalg.norm(target)
    if current_norm <= 0.0 or target_norm <= 0.0:
        return float("inf")
    dot = abs(float(np.dot(
        current / current_norm,
        target / target_norm,
    )))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


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


AMR_DRIVE_SETTINGS = {
    "/World/Nova_Carter/joint_caster_base": (100.0, 10.0, float("inf")),
    "/World/Nova_Carter/joint_swing_left": (0.0, 1.0e-5, float("inf")),
    "/World/Nova_Carter/joint_swing_right": (0.0, 1.0e-5, float("inf")),
    "/World/Nova_Carter/joint_wheel_left": (0.0, 1.0e6, 200.0),
    "/World/Nova_Carter/joint_wheel_right": (0.0, 1.0e6, 200.0),
}


def prepare_amr_physics(stage):
    """저장된 고정 drive를 실제 wheel/caster 물리값으로 복원한다."""
    for path, (stiffness, damping, max_force) in AMR_DRIVE_SETTINGS.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"AMR drive joint 누락: {path}")
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            raise RuntimeError(f"AMR angular DriveAPI 누락: {path}")
        drive.GetStiffnessAttr().Set(stiffness)
        drive.GetDampingAttr().Set(damping)
        drive.GetMaxForceAttr().Set(max_force)
        drive.GetTargetPositionAttr().Set(0.0)
        drive.GetTargetVelocityAttr().Set(0.0)


def measure_amr_pose(robot):
    position, orientation = robot.get_world_pose()
    linear_velocity = robot.get_linear_velocity()
    angular_velocity = robot.get_angular_velocity()
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
        raise RuntimeError("AMR pose에 NaN/inf가 있습니다")
    w, x, y, z = (float(value) for value in orientation)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1.0e-9:
        raise RuntimeError("AMR quaternion이 유효하지 않습니다")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return PoseState(
        x=float(position[0]),
        y=float(position[1]),
        yaw=math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        ),
        z=float(position[2]),
        up_z=1.0 - 2.0 * (x * x + y * y),
        qx=x,
        qy=y,
        qz=z,
        qw=w,
        linear_speed=float(np.linalg.norm(linear_velocity)),
        angular_speed=float(np.linalg.norm(angular_velocity)),
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
    approach, pick, lift = compute_busbar_pick_targets(
        target_xy=(float(pos.x), float(pos.y)),
        pick_z=_POS_GRAB_PICK,
        approach_z=BUSBAR_APPROACH_Z,
    )
    BUSBAR_APPROACH_POS = np.asarray(approach)
    BUSBAR_PICK_POS = np.asarray(pick)
    # 파지 뒤에는 근거 없는 world +Y 이동 없이 동일 XY에서 수직 상승한다.
    BUSBAR_LIFT_MOVE_POS = np.asarray(lift)

    print(f"[Dynamic Target Set] Approach Pos (Z={BUSBAR_APPROACH_Z:.2f}m): ({BUSBAR_APPROACH_POS[0]:.4f}, {BUSBAR_APPROACH_POS[1]:.4f}, {BUSBAR_APPROACH_POS[2]:.4f})")
    print(f"[Dynamic Target Set] Pick Pos     (Z={BUSBAR_PICK_POS[2]:.4f}m): ({BUSBAR_PICK_POS[0]:.4f}, {BUSBAR_PICK_POS[1]:.4f}, {BUSBAR_PICK_POS[2]:.4f})")
    print(f"[Dynamic Target Set] Vertical Lift                         : ({BUSBAR_LIFT_MOVE_POS[0]:.4f}, {BUSBAR_LIFT_MOVE_POS[1]:.4f}, {BUSBAR_LIFT_MOVE_POS[2]:.4f})")


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
    configure_fixed_camera_bridges(stage)
    configure_ros_domain(stage)
    workcell_rotation = rotate_busbar_workcell(stage, 0.0)
    prepare_amr_physics(stage)

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=PHYSICS_DT,
        rendering_dt=PHYSICS_DT,
    )

    busbar_assets = {}
    for index, (root_path, mesh_path) in enumerate(
        BUSBAR_PRIM_CANDIDATES
    ):
        if (
            not stage.GetPrimAtPath(root_path).IsValid()
            or not stage.GetPrimAtPath(mesh_path).IsValid()
        ):
            continue
        xform = SingleXFormPrim(
            mesh_path,
            name=f"busbar_mesh_{index}",
        )
        initial_position, initial_orientation = xform.get_world_pose()
        busbar_assets[mesh_path] = {
            "root_path": root_path,
            "xform": xform,
            "initial_position": np.asarray(initial_position).copy(),
            "initial_orientation": np.asarray(initial_orientation).copy(),
        }
    if not busbar_assets:
        raise RuntimeError(
            "현재 USD에서 파지 가능한 busbar prim을 찾지 못했습니다: "
            f"{BUSBAR_PRIM_CANDIDATES}"
        )
    busbar_root_path = None
    busbar_xform = None

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
    isaac_node = Execute_Isaac_Busar(stage)

    left_wheel_index = robot.get_dof_index("joint_wheel_left")
    right_wheel_index = robot.get_dof_index("joint_wheel_right")
    if (
        left_wheel_index is None
        or right_wheel_index is None
        or int(left_wheel_index) < 0
        or int(right_wheel_index) < 0
    ):
        raise RuntimeError("통합 articulation에서 wheel DOF를 찾지 못했습니다")
    wheel_indices = np.asarray(
        [int(left_wheel_index), int(right_wheel_index)],
        dtype=np.int64,
    )

    def apply_amr_wheels(left_radps, right_radps):
        robot.apply_action(
            ArticulationAction(
                joint_velocities=np.asarray(
                    [left_radps, right_radps],
                    dtype=np.float32,
                ),
                joint_indices=wheel_indices,
            )
        )

    drive_arrived = [False]
    drive_node = IntegratedPhysicsDriveBridge(
        measure_pose=lambda: measure_amr_pose(robot),
        apply_wheels=apply_amr_wheels,
        config=IntegratedDriveConfig(
            physics_hz=1.0 / PHYSICS_DT,
            debug_no_timeouts=_DEBUG_NO_TIMEOUTS,
            busbar_workcell_yaw_deg=workcell_rotation.yaw_deg,
            busbar_workcell_pivot_xyz=workcell_rotation.pivot_world,
            busbar_approach_yaw_rad=(
                workcell_rotation.busbar_approach_yaw_rad
            ),
        ),
        on_arrived=lambda: drive_arrived.__setitem__(0, True),
    )

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

    print(
        "\nIsaac Sim 준비 완료 - 물리 AMR 주행과 Arm 명령을 대기합니다."
    )

    step_count = 0
    grasp_timer = 0
    was_playing = False
    phase = "IDLE"
    busbar_start_grasp_pos = None
    descend_target_z = None
    target_mid_pos = None
    target_fine_yaw_rad = 0.0

    # ── 너트 조립(Nut Assembly) 상태 변수 (신규 추가) ──
    nut_index      = 0        # 1: 너트 1번, 2: 너트 2번 (NUT_*/MOVE_TO_BOLT_NUT 공용 Phase에서 참조)
    NUT_SCAN_POS   = None      # 너트 스캔 위치 (최초 1회 계산 후 재사용)
    nut_pick_pos   = None      # 비전으로 산출된 현재 너트의 물리 파지 좌표
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
        rclpy.spin_once(isaac_node, timeout_sec=0.0)
        rclpy.spin_once(drive_node, timeout_sec=0.0)
        drive_node.control_step()
        playing = world.is_playing()

        # 1. Play / Stop 상태 보정
        if playing and not was_playing:
            world.reset()
            for asset in busbar_assets.values():
                enable_physics_recursively(stage, asset["root_path"])
                asset["xform"].set_world_pose(
                    position=asset["initial_position"],
                    orientation=asset["initial_orientation"],
                )
            enable_physics_recursively(stage, NUT1_ROOT_PATH)
            enable_physics_recursively(stage, NUT2_ROOT_PATH)
            busbar_root_path = None
            busbar_xform = None
            if nut1_xform and init_nut1_pos is not None:
                nut1_xform.set_world_pose(position=init_nut1_pos, orientation=init_nut1_quat)
            if nut2_xform and init_nut2_pos is not None:
                nut2_xform.set_world_pose(position=init_nut2_pos, orientation=init_nut2_quat)

            step_count = 0
            grasp_timer = 0
            target_fine_yaw_rad = 0.0

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

        # 물리 주행 완료 후 RMPFlow가 사용하는 mobile base world pose를 갱신한다.
        if playing and phase == "IDLE" and drive_arrived[0]:
            drive_arrived[0] = False
            base_link_xf = world_xf(stage, f"{M0609_PATH}/base_link")
            base_pos = base_link_xf.ExtractTranslation()
            base_quat = base_link_xf.ExtractRotationQuat()
            arm_controller._motion_policy.set_robot_base_pose(
                robot_position=np.array([
                    base_pos[0], base_pos[1], base_pos[2]]),
                robot_orientation=np.array([
                    base_quat.GetReal(),
                    *[float(x) for x in base_quat.GetImaginary()],
                ]),
            )

        # 2. ArmNode 명령 분기 처리
        if (
            playing
            and not drive_node.driving
            and isaac_node.requested_task
        ):
            task = isaac_node.requested_task
            isaac_node.requested_task = None

            if task == "STOW_ARM":
                phase = "STOW_ARM"
                step_count = 0
                print(
                    "\n>>> [STOW_ARM] AMR 주행용 초기 관절 자세 복귀 시작"
                )

            elif task == "SCAN_BATTERY":
                phase = "INIT_POSE"
                step_count = 0
                print(f"\n>>> [{task}] 배터리 스캔 요청 수신 -> 1) 초기 관절 정렬 시작")

            elif task.startswith("SCAN_BUSBAR"):
                try:
                    task_name, separator, attempt_text = task.partition(":")
                    if task_name != "SCAN_BUSBAR":
                        raise ValueError(f"지원하지 않는 scan task: {task}")
                    attempt = int(attempt_text) if separator else 0
                    if attempt < 0:
                        raise ValueError("scan attempt는 0 이상이어야 합니다")
                    selected_pose = isaac_node.latest_target_pose
                    if selected_pose is None:
                        raise ValueError(
                            "relay가 선택한 busbar target pose가 없습니다"
                        )
                    if selected_pose.header.frame_id != "world":
                        raise ValueError(
                            "busbar target pose는 world frame이어야 합니다"
                        )
                    pos = selected_pose.pose.position
                    values = (float(pos.x), float(pos.y), float(pos.z))
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError(
                            "busbar target pose에 NaN/inf가 있습니다"
                        )
                    scan_x, scan_y = compute_busbar_scan_xy(
                        target_xy=(values[0], values[1]),
                        approach_yaw_rad=(
                            workcell_rotation.busbar_approach_yaw_rad
                        ),
                        attempt_index=attempt,
                        step_m=_BUSBAR_SCAN_SEARCH_STEP_M,
                    )
                    BUSBAR_SCAN_POS = np.array([
                        scan_x,
                        scan_y,
                        BUSBAR_SCAN_Z,
                    ])
                except (TypeError, ValueError) as exc:
                    print(f"\n[ERROR] [{task}] {exc}")
                    publish_status("FAILURE:INVALID_BUSBAR_SCAN_TARGET")
                    phase = "IDLE"
                else:
                    phase = "SCAN_BUSBAR_APPROACH"
                    step_count = 0
                    print(
                        f"\n>>> [{task}] 선택 busbar 중심 손목 스캔 시작 "
                        f"(Target: X={BUSBAR_SCAN_POS[0]:.3f}, "
                        f"Y={BUSBAR_SCAN_POS[1]:.3f}, "
                        f"Z={BUSBAR_SCAN_POS[2]:.3f})"
                    )

            elif task == "PICK_BUSBAR":
                target_pose = isaac_node.latest_target_pose
                if target_pose is None:
                    print(f"\n[ERROR] [{task}] wrist target pose가 없습니다")
                    publish_status("FAILURE:NO_BUSBAR_PICK_TARGET")
                    phase = "IDLE"
                else:
                    target = target_pose.pose.position
                    live_candidates = []
                    for mesh_path, asset in busbar_assets.items():
                        asset_position, _ = asset["xform"].get_world_pose()
                        live_candidates.append((
                            mesh_path,
                            float(asset_position[0]),
                            float(asset_position[1]),
                        ))
                    try:
                        selected_mesh_path = select_nearest_busbar_path(
                            target_xy=(float(target.x), float(target.y)),
                            candidates=live_candidates,
                            max_distance_m=BUSBAR_PRIM_MAX_ASSOCIATION_M,
                        )
                    except (TypeError, ValueError, VisionUnavailable) as exc:
                        print(
                            f"\n[ERROR] [{task}] 비전 좌표와 USD busbar "
                            f"prim 연관 실패: {exc}"
                        )
                        publish_status(
                            "FAILURE:BUSBAR_PRIM_ASSOCIATION"
                        )
                        phase = "IDLE"
                    else:
                        selected_asset = busbar_assets[selected_mesh_path]
                        busbar_root_path = selected_asset["root_path"]
                        busbar_xform = selected_asset["xform"]
                        update_target_positions(target_pose)
                        phase = "BUSBAR_APPROACH"
                        step_count = 0
                        print(
                            f"\n>>> [{task}] 비전 target에 최근접한 "
                            f"{selected_mesh_path} 파지 선택"
                        )
                        print(
                            f">>> [{task}] 버스바 상공 접근"
                            f"(Z={BUSBAR_APPROACH_Z:.3f}m) 시작"
                        )

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

                # 이전 arm target을 error-fix 증분으로 오인하지 않도록 먼저 비운다.
                isaac_node.latest_target_pose = None
                isaac_node.alignment_success = False
                start_cmd = String()
                start_cmd.data = "START_ERRORFIX_CORRECTION"
                isaac_node.pub_errorfix_command.publish(start_cmd)

                # FINE_ALIGNMENT 진입 시 실시간 EE 위치 기준으로 Target 초기화
                cur_ee = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                target_fine_pos = np.array([cur_ee[0], cur_ee[1], BATTERY_CENTER_Z])
                target_fine_yaw_rad = 0.0
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

            # ── [신규] 너트 1/2번 스캔 -> 파지 -> 체결 (Nut Assembly) ──
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
                if isaac_node.latest_target_pose is not None:
                    pos = isaac_node.latest_target_pose.pose.position
                    nut_pick_pos = np.array([pos.x, pos.y, NUT_PICK_Z])
                    nut_approach_pos = np.array([pos.x, pos.y, NUT_APPROACH_Z])
                    phase = "NUT_APPROACH"
                    step_count = 0
                    print(f"\n>>> [{task}] 너트 {nut_index}번 상공 접근 시작 (Target: X={nut_approach_pos[0]:.3f}, Y={nut_approach_pos[1]:.3f})")
                else:
                    print(f"\n[ERROR] [{task}] 수신된 너트 Target Pose가 없습니다.")
                    publish_status("FAILURE:NO_TARGET_POSE")

            elif task in ("ASSEMBLE_NUT1", "ASSEMBLE_NUT2"):
                nut_index = 1 if task == "ASSEMBLE_NUT1" else 2
                if target_mid_pos is not None:
                    bolt_target_pos = compute_bolt_target_pos(nut_index, target_mid_pos)
                    bolt_touch_pos = np.array([
                        bolt_target_pos[0], bolt_target_pos[1],
                        bolt_target_pos[2] + EE_OFFSET[2] + NUT_GRASP_Z_LOCAL
                    ])
                    phase = "MOVE_TO_BOLT_NUT"
                    step_count = 0
                    print(f"\n>>> [{task}] 너트 {nut_index}번 -> 볼트 {nut_index}번 체결 시작 (Target: X={bolt_target_pos[0]:.4f}, Y={bolt_target_pos[1]:.4f})")
                else:
                    print(f"\n[ERROR] [{task}] 배터리 중심 좌표(target_mid_pos)가 없습니다. 먼저 ASSEMBLE_BUSBAR가 수행되어야 합니다.")
                    publish_status("FAILURE:NO_BATTERY_CENTER")

        # 3. FSM 제어 루프
        if playing and phase != "IDLE" and phase != "DONE":

            # [SAFE] AMR 주행 전 팔을 차체 안쪽 초기 관절 자세로 복귀
            if phase == "STOW_ARM":
                publish_progress("STOW_ARM", 50.0)
                arm_joint_names = [
                    "joint_1", "joint_2", "joint_3",
                    "joint_4", "joint_5", "joint_6",
                ]
                arm_dof_indices = [
                    robot.get_dof_index(name)
                    for name in arm_joint_names
                ]
                robot.apply_action(
                    ArticulationAction(
                        joint_positions=TARGET_INIT_JOINTS,
                        joint_indices=arm_dof_indices,
                    )
                )
                robot.gripper.apply_action(
                    ArticulationAction(joint_positions=GRIPPER_OPEN)
                )
                all_joints = robot.get_joint_positions()
                cur_arm_joints = all_joints[arm_dof_indices]
                joint_err = np.linalg.norm(
                    cur_arm_joints - TARGET_INIT_JOINTS
                )
                if joint_err < JOINT_TOLERANCE:
                    print(
                        "[OK] 팔 안전 자세 복귀 완료 "
                        f"(joint_err={joint_err:.4f}rad)"
                    )
                    publish_progress("STOW_ARM_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"
                elif (
                    not _DEBUG_NO_TIMEOUTS
                    and step_count > MAX_STUCK_STEPS
                ):
                    print(
                        "[ERROR] 팔 안전 자세 복귀 실패 "
                        f"(joint_err={joint_err:.4f}rad)"
                    )
                    publish_status("FAILURE:STOW_ARM")
                    phase = "IDLE"

            # [STEP 0-A] 초기 관절 자세 정렬
            elif phase == "INIT_POSE":
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
                _, cur_quat = robot.end_effector.get_world_pose()
                orientation_err = quaternion_angular_error_wxyz(
                    cur_quat,
                    quat_busbar,
                )

                if step_count % 120 == 0:
                    print(
                        "[SCAN_BUSBAR] "
                        f"position_err={current_err:.4f}m, "
                        f"orientation_err="
                        f"{math.degrees(orientation_err):.1f}deg"
                    )

                if (
                    current_err < PICK_TOLERANCE_STRICT
                    and orientation_err < SCAN_ORIENTATION_TOLERANCE_RAD
                ):
                    print(f"[OK] 버스바 스캔 위치 도착 완료! ({cur_pos[0]:.3f}, {cur_pos[1]:.3f}, {cur_pos[2]:.3f})")
                    publish_progress("SCAN_BUSBAR_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"
                elif (
                    not _DEBUG_NO_TIMEOUTS
                    and step_count > SCAN_BUSBAR_TIMEOUT_STEPS
                ):
                    print(
                        "[ERROR] 버스바 스캔 자세 수렴 실패: "
                        f"position_err={current_err:.4f}m, "
                        f"orientation_err="
                        f"{math.degrees(orientation_err):.1f}deg"
                    )
                    publish_status("FAILURE:SCAN_BUSBAR_IK")
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

                if step_count % 120 == 0:
                    print(
                        "[PICK_BUSBAR APPROACH] "
                        f"position_err={current_err:.4f}m"
                    )

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
                    if busbar_xform is None or busbar_root_path is None:
                        print(
                            "[ERROR] 선택된 busbar prim이 없어 "
                            "가짜 파지로 진행하지 않습니다"
                        )
                        publish_status(
                            "FAILURE:NO_SELECTED_BUSBAR_PRIM"
                        )
                        phase = "IDLE"
                        continue
                    disable_physics_recursively(stage, busbar_root_path)
                    real_pos, _ = busbar_xform.get_world_pose()
                    busbar_start_grasp_pos = np.array(real_pos)

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

                if step_count % 120 == 0:
                    print(
                        "[PICK_BUSBAR LIFT] "
                        f"position_err={current_err:.4f}m"
                    )

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
                publish_progress("FINE_ALIGNMENT_TRACKING", 85.0)

                cur_ee_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()

                if isaac_node.latest_target_pose is not None:
                    correction = isaac_node.latest_target_pose.pose
                    offset = correction.position
                    if abs(offset.x) <= 0.0025 and abs(offset.y) <= 0.0025:
                        target_fine_pos[0] += offset.x
                        target_fine_pos[1] += offset.y
                        target_fine_pos[2] = BATTERY_CENTER_Z

                        quaternion_norm = math.hypot(
                            correction.orientation.z,
                            correction.orientation.w,
                        )
                        yaw_delta = 0.0
                        if quaternion_norm > 0.0:
                            yaw_delta = 2.0 * math.atan2(
                                correction.orientation.z,
                                correction.orientation.w,
                            )
                            if abs(yaw_delta) <= math.radians(1.0):
                                target_fine_yaw_rad += yaw_delta
                                target_fine_yaw_rad = (
                                    target_fine_yaw_rad + math.pi
                                ) % (2.0 * math.pi) - math.pi

                        print(
                            "\n[FINE_ALIGNMENT] Vision Offset Received "
                            f"-> dx: {offset.x:+.4f}m, "
                            f"dy: {offset.y:+.4f}m, "
                            f"dyaw: {math.degrees(yaw_delta):+.3f}deg"
                        )
                        print(
                            "               └─ New Target "
                            f"X: {target_fine_pos[0]:.4f}, "
                            f"Y: {target_fine_pos[1]:.4f}, "
                            f"Z: {target_fine_pos[2]:.4f}, "
                            f"yaw: "
                            f"{math.degrees(target_fine_yaw_rad):+.3f}deg"
                        )

                    isaac_node.latest_target_pose = None

                sys.stdout.write(
                    f"\r\033[K[FINE_ALIGNMENT Loop] Cur EE: ({cur_ee_pos[0]:.4f}, {cur_ee_pos[1]:.4f}) "
                    f"-> Target: ({target_fine_pos[0]:.4f}, {target_fine_pos[1]:.4f})"
                )
                sys.stdout.flush()

                actions = arm_controller.forward(
                    target_end_effector_position=target_fine_pos,
                    target_end_effector_orientation=euler_to_quaternion_wxyz(
                        0.0,
                        3.1415,
                        target_fine_yaw_rad,
                    )
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
                    target_end_effector_orientation=euler_to_quaternion_wxyz(
                        0.0,
                        3.1415,
                        target_fine_yaw_rad,
                    )
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))

                glue_busbar_to_ee(robot, busbar_xform, busbar_start_grasp_pos, blend=1.0)

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                dist_err = math.dist(cur_pos, tuple(target_mid_pos))

                if cur_pos[2] <= BUSBAR_RELEASE_Z or dist_err < INSERT_TOLERANCE_STRICT:
                    if busbar_xform is not None:
                        _, placed_orientation = busbar_xform.get_world_pose()
                        busbar_xform.set_world_pose(
                            position=target_mid_pos,
                            orientation=placed_orientation,
                        )
                    
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
                    target_end_effector_orientation=euler_to_quaternion_wxyz(
                        0.0,
                        3.1415,
                        target_fine_yaw_rad,
                    )
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
    drive_node.stop()
    if 'world' in locals() and world is not None:
        world.clear_instance()
    omni.usd.get_context().close_stage()
    drive_node.destroy_node()
    isaac_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()

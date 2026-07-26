"""
01_pick_and_lift.py / 10_busbar_assembly.py (Full Busbar Assembly + Physical Nut Screwing Version)
─ AMR 베이스 잠금
─ [재시작 초기화 보정] Stop 후 Play 시 너트 및 버스바 위치 초기화 강제 적용
─ [버스바 공정 활성화] 버스바 픽앤플레이스/체결 -> 너트 1번 체결 -> 너트 2번 체결 전체 시퀀스 통합
─ [너트 물리 유지] 너트 파지 시 실제 물리 파지 적용
─ [수정 완료] Screwing Regrasp 시 상공 상승(+0.05m) 후 역회전/하강 적용
─ [수정 완료] 너트 1, 2번 체결 후 상공(Z=0.8m)에서 6번 조인트를 정확히 0도가 되도록 Unwind 적용
─ [추가 완료] TCP Z 높이 0.37m 이하 조건 추가: 토크 및 Z축 Sticking 감지 시 Screwing 조기 종료 및 그리퍼 해제
─ [수정 완료] Screwing Regrasp 하강/재파지 시 Z축 높이를 미세(3mm) 상승 적용
─ [기능 추가] 버스바/볼트/너트 물리 마찰력(Static/Dynamic Friction, Restitution) 설정 적용
─ [오류 수정] SimulationApp 구동 후 다중 sys.path 등록으로 ModuleNotFoundError 완벽 예방
"""

import os
import sys
import math
import gc
import csv
import time
from pathlib import Path

from isaacsim import SimulationApp

# Headless 모드 설정 (환경변수 AMR_HEADLESS=1)
_HEADLESS = os.environ.get("AMR_HEADLESS") == "1"
simulation_app = SimulationApp({"headless": _HEADLESS})

# 최신 isaacsim 패키지 및 기존 omni 패키지 호환성 지원
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.ros2.bridge")
except Exception:
    try:
        from omni.isaac.core.utils.extensions import enable_extension
        enable_extension("omni.isaac.ros2_bridge")
    except Exception as e:
        print(f"[Warning] ROS2 Bridge 활성화 도중 예외 발생 (무시 가능): {e}")

simulation_app.update()
sys.stdout.reconfigure(line_buffering=True)

# ══════════════════════════════════════════════════════════════════════════
# ★ 다중 경로 자동 탐색 (Isaac Sim 초기화 시 경로 리셋 및 위치 불일치 예방) ★
# ══════════════════════════════════════════════════════════════════════════
_THIS_DIR = Path(__file__).resolve().parent
candidate_paths = [
    _THIS_DIR,
    _THIS_DIR / "rmpflow",
    Path("/home/rokey/junhyeok_version/isaacpjt/M0609"),
    Path("/home/rokey/junhyeok_version/isaacpjt/M0609/rmpflow"),
    Path("/home/rokey/isaac_nut/isaacpjt/M0609/rmpflow"),
    Path("/home/rokey/EV_combine/src/rmpflow"),
]

for p in candidate_paths:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from m0609_rmpflow_controller import RMPFlowController  # noqa: E402
except ModuleNotFoundError:
    try:
        from rmpflow.m0609_rmpflow_controller import RMPFlowController  # noqa: E402
    except ModuleNotFoundError as e:
        print(f"\n[ERROR] RMPFlowController 모듈을 찾지 못했습니다!")
        print(f"탐색 경로: {[str(cp) for cp in candidate_paths]}")
        raise e

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, Gf, Sdf
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.core.utils.types import ArticulationAction

# ══════════════════════════════════════════════════════════════════════════
#  [A] 설정 및 파라미터
# ══════════════════════════════════════════════════════════════════════════
USD_PATH = "/home/rokey/EV_combine/src/Collected_Busbar/Busbar.usd"

NOVA_CARTER_ROOT = "/World/Nova_Carter/chassis_link"
M0609_PATH       = "/World/m0609"
EE_LINK_NAME     = "link_6"
GRIPPER_JOINTS   = ["finger_joint", "right_inner_knuckle_joint"]

# 버스바 Prim 경로
BUSBAR_ROOT_PATH      = "/World/busbar"
BUSBAR_POLYSHAPE_PATH = "/World/busbar/geo/PolyShape"

# 너트 Prim 경로
NUT1_ROOT_PATH      = "/World/nut1"
NUT2_ROOT_PATH      = "/World/nut2"
NUT1_POLYSHAPE_PATH = "/World/nut1/geo/PolyShape"
NUT2_POLYSHAPE_PATH = "/World/nut2/geo/PolyShape"

# 볼트 Prim 경로
BOLT1_ROOT_PATH     = "/World/bolt1"
BOLT2_ROOT_PATH     = "/World/bolt2"

# ★ [물리 마찰력 설정 파라미터] ★
BOLT_NUT_STATIC_FRICTION  = 0.4   # 정적 마찰 계수
BOLT_NUT_DYNAMIC_FRICTION = 0.3   # 동적 마찰 계수
BOLT_NUT_RESTITUTION       = 0.0   # 반발 계수

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

NUT_HEIGHT = 0.0095
NUT_GRASP_Z_LOCAL = NUT_HEIGHT + 0.023

# ★ 버스바 및 체결 중심 파라미터 ★
_POS_GRAB_PICK          = np.array([0.5136, 0.7299, 0.455])
BUSBAR_APPROACH_POS     = _POS_GRAB_PICK + np.array([0.0, 0.0, 0.145])
BUSBAR_PICK_POS         = _POS_GRAB_PICK.copy()
BUSBAR_LIFT_MOVE_POS    = _POS_GRAB_PICK + np.array([0.0, 0.1, 0.145])

target_mid_pos          = np.array([1.1606, 0.1836, 0.0693])
TARGET_DESTINATION_POS  = np.array([target_mid_pos[0], target_mid_pos[1], 0.6])
TARGET_INSERT_POS       = np.array([target_mid_pos[0], target_mid_pos[1], target_mid_pos[2]])

# ★ ArmNode 허용 오차 조건 ★
PICK_TOLERANCE_STRICT   = 0.01     # Pick 단계: 0.01m (10mm)
INSERT_TOLERANCE_STRICT = 0.001    # Insert 단계: 0.001m (1mm)
BUSBAR_RELEASE_Z        = 0.37     # 그리퍼 해제 임계 높이
INSERT_SPEED            = 0.0005   # Step당 수직 하강 거리

PICK_TOLERANCE_LOOSE_VAL = 0.015
MAX_STUCK_STEPS          = 60

# ★ 비전 정렬 보정(error_fix_depth.py의 /busbar_alignment_error) 유효 시간 ★
BUSBAR_ALIGNMENT_MAX_AGE = 1.0   # 초 단위, 이보다 오래된 보정값은 사용하지 않고 기본 좌표로 폴백

# ★ 볼트 좌표 ★
BOLT1_POS = np.array([1.0576, 0.3653, 0.152])
BOLT2_POS = np.array([1.2636, 0.0019, 0.152])

# ★ 너트 1번 좌표 (nut1 -> bolt1) ★
NUT_APPROACH_Z    = 0.8
NUT1_PICK_POS     = np.array([0.5746, -0.1008, 0.72 - (NUT_GRASP_Z_LOCAL - 0.0395)])
NUT1_APPROACH_POS = np.array([NUT1_PICK_POS[0], NUT1_PICK_POS[1], NUT_APPROACH_Z])
BOLT1_APPROACH_POS = np.array([BOLT1_POS[0], BOLT1_POS[1], 0.6])
BOLT1_TOUCH_POS    = np.array([BOLT1_POS[0], BOLT1_POS[1], BOLT1_POS[2] + EE_OFFSET[2] + NUT_GRASP_Z_LOCAL])

# ★ 너트 2번 좌표 (nut2 -> bolt2) ★
NUT2_PICK_POS     = np.array([0.6643, -0.1031, 0.72 - (NUT_GRASP_Z_LOCAL - 0.0395)])
NUT2_APPROACH_POS = np.array([NUT2_PICK_POS[0], NUT2_PICK_POS[1], NUT_APPROACH_Z])
BOLT2_APPROACH_POS = np.array([BOLT2_POS[0], BOLT2_POS[1], 0.6])
BOLT2_TOUCH_POS    = np.array([BOLT2_POS[0], BOLT2_POS[1], BOLT2_POS[2] + EE_OFFSET[2] + NUT_GRASP_Z_LOCAL])

# 체결(SCREW) 파라미터
ENGAGE_LEN        = 0.0125    # 체결 깊이 (12.5mm)
SCREW_TURNS_DEG   = 350.0     # 1회전당 350도
REGRASP_CYCLES    = 1         # 총 2회전
SCREW_OMEGA_DEG_S = 120.0     # 초당 120도 회전
PHYSICS_DT        = 1.0 / 60.0
REGRASP_LIFT_HEIGHT = 0.05    # Regrasp 시 수직 상승 높이 (5cm)
REGRASP_Z_OFFSET    = 0.005   # Regrasp 하강/재파지 시 너트를 잡는 위치 보정 높이 (3mm 상승)

TOTAL_REV   = (SCREW_TURNS_DEG / 360.0) * (1 + REGRASP_CYCLES)
NUT_PITCH_M = ENGAGE_LEN / TOTAL_REV

# ★ 완착/토크 감지 파라미터 ★
TORQUE_THRESHOLD      = 45.0   # 6번 조인트 반력 임계값 (Nm)
STUCK_Z_DELTA_THRESH  = 0.0001  # Z축 하강 멈춤 판정 기준 (0.1mm)
STUCK_STEP_LIMIT      = 12      # Z축 변화 없이 토크 지속되는 Step 수
TCP_FORCE_CHECK_Z     = 0.378   # TCP(EE) 높이가 0.37m 이하일 때만 힘/토크 감지 활성화

URDF_PATH        = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
ROBOT_DESC_PATH  = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
RMPFLOW_CFG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")

# ★ Screwing 체결 로그 저장 경로 (후처리 시각화용, plot_screw_log.py 참고) ★
LOG_DIR       = _THIS_DIR / "logs"
SCREW_LOG_CSV = LOG_DIR / "screw_log.csv"
SCREW_LOG_FIELDS = ["nut_id", "step", "pass_idx", "theta_deg", "total_deg", "depth_mm", "tcp_z", "torque_nm", "stuck_counter", "seated"]


# ══════════════════════════════════════════════════════════════════════════
#  [B] 헬퍼 및 Kinematic Pose-Glue / Screwing / 마찰력 생성 함수
# ══════════════════════════════════════════════════════════════════════════
def set_physx_scene_limits(stage):
    scene_prim = stage.GetPrimAtPath("/World/PhysicsScene")
    if not scene_prim.IsValid():
        scene_prim = stage.DefinePrim("/World/PhysicsScene", "PhysicsScene")

    PhysxSchema.PhysxSceneAPI.Apply(scene_prim)

    attr_max_patches = scene_prim.GetAttribute("physxScene:maxFrictionPatches")
    if not attr_max_patches.IsValid():
        attr_max_patches = scene_prim.CreateAttribute("physxScene:maxFrictionPatches", Sdf.ValueTypeNames.Int)
    attr_max_patches.Set(128)

    attr_gpu_found = scene_prim.GetAttribute("physxScene:gpuFoundLostPairsCapacity")
    if not attr_gpu_found.IsValid():
        attr_gpu_found = scene_prim.CreateAttribute("physxScene:gpuFoundLostPairsCapacity", Sdf.ValueTypeNames.Int)
    attr_gpu_found.Set(2 ** 21)

    attr_gpu_total = scene_prim.GetAttribute("physxScene:gpuTotalAggregatePairsCapacity")
    if not attr_gpu_total.IsValid():
        attr_gpu_total = scene_prim.CreateAttribute("physxScene:gpuTotalAggregatePairsCapacity", Sdf.ValueTypeNames.Int)
    attr_gpu_total.Set(2 ** 21)

    attr_time_steps = scene_prim.GetAttribute("physxScene:timeStepsPerSecond")
    if not attr_time_steps.IsValid():
        attr_time_steps = scene_prim.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.Int)
    attr_time_steps.Set(int(1.0 / PHYSICS_DT))

    print("[INFO] PhysX Scene 연산 한계 확장 완료 (MaxFrictionPatches: 128, GPU Buffers Expanded)")


def set_friction_material(stage, material_path, target_prim_path, static_friction, dynamic_friction, restitution):
    target_prim = stage.GetPrimAtPath(target_prim_path)
    if not target_prim.IsValid():
        return None

    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        material_prim = stage.DefinePrim(material_path, "Material")

    phys_mat_api = UsdPhysics.MaterialAPI.Apply(material_prim)
    phys_mat_api.CreateStaticFrictionAttr(static_friction)
    phys_mat_api.CreateDynamicFrictionAttr(dynamic_friction)
    phys_mat_api.CreateRestitutionAttr(restitution)

    physx_mat_api = PhysxSchema.PhysxMaterialAPI.Apply(material_prim)
    physx_mat_api.CreateFrictionCombineModeAttr("average")
    physx_mat_api.CreateRestitutionCombineModeAttr("average")

    shade_material = UsdShade.Material(material_prim)
    for prim in Usd.PrimRange(target_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
            binding_api.Bind(shade_material, UsdShade.Tokens.strongerThanDescendants, "physics")

    print(f"[INFO] 마찰 재질 적용 완료 -> Prim: {target_prim_path} | Static: {static_friction}, Dynamic: {dynamic_friction}")
    return material_prim


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
            col_api = UsdPhysics.CollisionAPI(prim)
            col_api.GetCollisionEnabledAttr().Set(False)
        
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb_api = UsdPhysics.RigidBodyAPI(prim)
            rb_api.GetRigidBodyEnabledAttr().Set(False)


def enable_physics_recursively(stage, prim_path):
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        return

    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            col_api = UsdPhysics.CollisionAPI(prim)
            col_api.GetCollisionEnabledAttr().Set(True)
        
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb_api = UsdPhysics.RigidBodyAPI(prim)
            rb_api.GetRigidBodyEnabledAttr().Set(True)


def yaw_rotated_quat(base_wxyz, delta_deg):
    base_q = Gf.Quatd(float(base_wxyz[0]), Gf.Vec3d(float(base_wxyz[1]), float(base_wxyz[2]), float(base_wxyz[3])))
    base_rot = Gf.Rotation(base_q)
    extra_rot = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), float(delta_deg))
    combined = extra_rot * base_rot
    q = combined.GetQuat()
    return np.array([q.GetReal(), *q.GetImaginary()])


def world_xf(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim 없음: {path}")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)


def find_prim_path(stage, root_path, name):
    root = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def set_all_drives(stage, root_path, stiffness=1.0e8, damping=1.0e4, max_force=1.0e8):
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        for dt in ("angular", "linear"):
            drive = UsdPhysics.DriveAPI.Get(prim, dt)
            if drive:
                drive.GetStiffnessAttr().Set(stiffness)
                drive.GetDampingAttr().Set(damping)
                drive.GetMaxForceAttr().Set(max_force)


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


def initialize_robot(robot, world):
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )


class BusbarAlignmentListener(Node):
    """error_fix_depth.py가 퍼블리시하는 /busbar_alignment_error(Twist)를 non-blocking으로 구독.
    linear.xy = world dx,dy[m] (busbar 구멍 - 기준 볼트), angular.z = dTheta[rad].
    """
    def __init__(self):
        super().__init__("assembly_alignment_listener")
        self.dx = 0.0
        self.dy = 0.0
        self.dtheta = 0.0
        self.last_stamp = 0.0
        self.create_subscription(Twist, "/busbar_alignment_error", self._callback, 10)

    def _callback(self, msg):
        self.dx = msg.linear.x
        self.dy = msg.linear.y
        self.dtheta = msg.angular.z
        self.last_stamp = time.time()

    def get_fresh_xy_correction(self, max_age_sec=BUSBAR_ALIGNMENT_MAX_AGE):
        if self.last_stamp <= 0.0 or (time.time() - self.last_stamp) > max_age_sec:
            return None
        return self.dx, self.dy


def glue_busbar_to_ee(robot, busbar_xform, rest_pick_pos, blend):
    if busbar_xform is None or rest_pick_pos is None:
        return

    ee_pos, ee_quat = robot.end_effector.get_world_pose()
    grasp_point_pos = np.asarray(ee_pos) - EE_OFFSET
    target_pos = grasp_point_pos - np.array([0.0, 0.0, BUSBAR_GRASP_Z_LOCAL])

    busbar_pos = rest_pick_pos + blend * (target_pos - rest_pick_pos)
    busbar_xform.set_world_pose(position=busbar_pos, orientation=np.asarray(ee_quat))


# ══════════════════════════════════════════════════════════════════════════
#  [C] 메인
# ══════════════════════════════════════════════════════════════════════════
def main():
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

    set_physx_scene_limits(stage)
    lock_amr_base(stage, NOVA_CARTER_ROOT)

    mat_path = "/World/Looks/BoltNutMaterial"
    # 버스바(BUSBAR_ROOT_PATH) 포함 마찰 재질 적용
    for target_path in [BUSBAR_ROOT_PATH, NUT1_ROOT_PATH, NUT2_ROOT_PATH, BOLT1_ROOT_PATH, BOLT2_ROOT_PATH]:
        set_friction_material(
            stage=stage,
            material_path=mat_path,
            target_prim_path=target_path,
            static_friction=BOLT_NUT_STATIC_FRICTION,
            dynamic_friction=BOLT_NUT_DYNAMIC_FRICTION,
            restitution=BOLT_NUT_RESTITUTION,
        )

    busbar_xform = SingleXFormPrim(BUSBAR_POLYSHAPE_PATH, name="busbar_poly") if stage.GetPrimAtPath(BUSBAR_POLYSHAPE_PATH).IsValid() else None
    nut1_xform   = SingleXFormPrim(NUT1_POLYSHAPE_PATH, name="nut1_poly") if stage.GetPrimAtPath(NUT1_POLYSHAPE_PATH).IsValid() else None
    nut2_xform   = SingleXFormPrim(NUT2_POLYSHAPE_PATH, name="nut2_poly") if stage.GetPrimAtPath(NUT2_POLYSHAPE_PATH).IsValid() else None

    # 초기 원본 Pose 저장
    init_busbar_pos, init_busbar_quat = busbar_xform.get_world_pose() if busbar_xform else (None, None)
    init_nut1_pos, init_nut1_quat     = nut1_xform.get_world_pose() if nut1_xform else (None, None)
    init_nut2_pos, init_nut2_quat     = nut2_xform.get_world_pose() if nut2_xform else (None, None)

    set_all_drives(stage, M0609_PATH)
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
    initialize_robot(robot, world)

    default_joint_positions = np.array([-0.5, -0.4, 1.4, 0.0, 0.6, 0.0, 0.0, 0.0])
    try:
        robot.set_joint_positions(default_joint_positions)
    except Exception:
        pass

    for _ in range(30):
        world.step(render=True)

    quat_busbar = euler_to_quaternion_wxyz(0.0, 3.1415, 1.5708)
    quat_busbar_0deg = euler_to_quaternion_wxyz(0.0, 3.1415, 0.0)

    rot_nut = R.from_euler('xyz', [0, 180, 0], degrees=True)
    q_nut   = rot_nut.as_quat()
    quat_nut = np.array([q_nut[3], q_nut[0], q_nut[1], q_nut[2]])

    arm_controller = RMPFlowController(
        name="m0609_hardcoded_controller",
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

    # ★ error_fix_depth.py가 퍼블리시하는 /busbar_alignment_error 구독용 non-blocking 리스너 ★
    if not rclpy.ok():
        rclpy.init()
    alignment_listener = BusbarAlignmentListener()

    print("[대기] Isaac Sim UI에서 Play 버튼을 누르면 시퀀스를 시작합니다.")

    step_count = 0
    grasp_timer = 0
    was_playing = False

    # ★ [수정 완료] 버스바 장착 공정(BUSBAR_APPROACH)부터 시작되도록 복원 ★
    phase = "BUSBAR_APPROACH"
    current_err = 0.0

    busbar_start_grasp_pos = None
    descend_target_z       = None
    busbar_insert_xy       = TARGET_INSERT_POS[:2].copy()  # 비전 보정 적용 전 기본값 (하강 시점에 갱신됨)

    screw_sub = "rotate"
    screw_pass_idx = 0
    screw_pass_theta = 0.0
    screw_seat_pos = None
    screw_seat_quat = None
    screw_seat_ee_pos = None
    screw_start_quat = None
    screw_pass_end_pos = None
    screw_release_step = 0
    screw_regrasp_step = 0
    screw_unwind_deg = 0.0

    # 완착 감지용 모니터링 변수
    prev_ee_z = 0.0
    stuck_counter = 0

    # ★ 체결(Screwing) 로그 버퍼 (torque-angle 등 후처리 시각화용) ★
    screw_records = []

    while simulation_app.is_running():
        world.step(render=True)
        rclpy.spin_once(alignment_listener, timeout_sec=0.0)
        playing = world.is_playing()

        # Stop 후 Play 재시작 시
        if playing and not was_playing:
            world.reset()
            initialize_robot(robot, world)
            
            enable_physics_recursively(stage, BUSBAR_ROOT_PATH)
            enable_physics_recursively(stage, NUT1_ROOT_PATH)
            enable_physics_recursively(stage, NUT2_ROOT_PATH)

            if busbar_xform and init_busbar_pos is not None:
                busbar_xform.set_world_pose(position=init_busbar_pos, orientation=init_busbar_quat)
            if nut1_xform and init_nut1_pos is not None:
                nut1_xform.set_world_pose(position=init_nut1_pos, orientation=init_nut1_quat)
            if nut2_xform and init_nut2_pos is not None:
                nut2_xform.set_world_pose(position=init_nut2_pos, orientation=init_nut2_quat)

            try:
                robot.set_joint_positions(default_joint_positions)
            except Exception:
                pass
            step_count = 0
            grasp_timer = 0
            # 재시작 시에도 버스바 장착 공정부터 다시 시작
            phase = "BUSBAR_APPROACH"
            current_err = 0.0
            busbar_start_grasp_pos = None
            descend_target_z       = None
            busbar_insert_xy       = TARGET_INSERT_POS[:2].copy()

            screw_sub = "rotate"
            screw_pass_idx = 0
            screw_pass_theta = 0.0
            stuck_counter = 0
            screw_records = []
            print(f"\n[Play] 시퀀스 재시작 (모든 객체 포즈 및 오차 조건 초기화 완료)")

        if playing and phase != "DONE":

            # ════════════════════════════════════════════════════════════════
            # [1] 버스바 픽앤플레이스 + 장착 시퀀스 (활성화 완료)
            # ════════════════════════════════════════════════════════════════
            if phase == "BUSBAR_APPROACH":
                actions = arm_controller.forward(target_end_effector_position=BUSBAR_APPROACH_POS, target_end_effector_orientation=quat_busbar)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(BUSBAR_APPROACH_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 1. 버스바 상공 접근 완료! -> 2. 파지점 하강 시작")
                    phase = "BUSBAR_DESCEND"
                    step_count = 0
            
            elif phase == "BUSBAR_DESCEND":
                actions = arm_controller.forward(target_end_effector_position=BUSBAR_PICK_POS, target_end_effector_orientation=quat_busbar)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(BUSBAR_PICK_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 2. 버스바 파지점 하강 완료! -> 그리퍼 닫기(Kinematic 파지)")
                    phase = "BUSBAR_GRASP"
                    grasp_timer = 0
            
            elif phase == "BUSBAR_GRASP":
                if grasp_timer == 0:
                    disable_physics_recursively(stage, BUSBAR_ROOT_PATH)
                    if busbar_xform is not None:
                        real_pos, _ = busbar_xform.get_world_pose()
                        busbar_start_grasp_pos = np.array(real_pos)
                    else:
                        busbar_start_grasp_pos = BUSBAR_PICK_POS
            
                actions = arm_controller.forward(target_end_effector_position=BUSBAR_PICK_POS, target_end_effector_orientation=quat_busbar)
                robot.apply_action(actions)
                
                grasp_timer += 1
                ramp_frac = min(grasp_timer / GRIP_CLOSE_RAMP_STEPS, 1.0)
                grip_target = ramp_frac * GRIPPER_CLOSE
                robot.gripper.apply_action(ArticulationAction(joint_positions=grip_target))
            
                glue_busbar_to_ee(robot, busbar_xform, busbar_start_grasp_pos, blend=ramp_frac)
            
                if grasp_timer >= 50:
                    print(f"[OK] 그리퍼 닫기 완료! -> 3. 버스바 상승 및 이동 시작")
                    phase = "BUSBAR_LIFT"
                    step_count = 0
            
            elif phase == "BUSBAR_LIFT":
                actions = arm_controller.forward(target_end_effector_position=BUSBAR_LIFT_MOVE_POS, target_end_effector_orientation=quat_busbar)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))
                
                glue_busbar_to_ee(robot, busbar_xform, busbar_start_grasp_pos, blend=1.0)
            
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(BUSBAR_LIFT_MOVE_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 3. 버스바 상승 이동 완료! -> [INSERT] 1. 체결위치 상공 접근 (그리퍼 0도 회전)")
                    phase = "MOVE_TO_BOLT_APPROACH"
                    step_count = 0
            
            elif phase == "MOVE_TO_BOLT_APPROACH":
                actions = arm_controller.forward(target_end_effector_position=TARGET_DESTINATION_POS, target_end_effector_orientation=quat_busbar_0deg)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))
            
                glue_busbar_to_ee(robot, busbar_xform, busbar_start_grasp_pos, blend=1.0)
            
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(TARGET_DESTINATION_POS))
                if current_err < INSERT_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    correction = alignment_listener.get_fresh_xy_correction()
                    if correction is not None:
                        corr_dx, corr_dy = correction
                        busbar_insert_xy = np.array([TARGET_INSERT_POS[0] - corr_dx, TARGET_INSERT_POS[1] - corr_dy])
                        print(f"[VISION] 정렬 보정 적용 -> dx={corr_dx*1000:+.1f}mm, dy={corr_dy*1000:+.1f}mm")
                    else:
                        busbar_insert_xy = TARGET_INSERT_POS[:2].copy()
                        print(f"[VISION] 유효한 보정값 없음(수신 안 됨/오래됨) -> 기본 좌표로 진행")

                    print(f"[OK] 1. 체결위치 상공 접근 완료! -> 2. 점진적 하강(-{INSERT_SPEED}m/step) 시작")
                    phase = "BUSBAR_DESCEND_TO_BOLT"
                    step_count = 0
                    descend_target_z = TARGET_DESTINATION_POS[2]

            elif phase == "BUSBAR_DESCEND_TO_BOLT":
                descend_target_z = max(descend_target_z - INSERT_SPEED, TARGET_INSERT_POS[2])
                step_target_pos = np.array([busbar_insert_xy[0], busbar_insert_xy[1], descend_target_z])

                actions = arm_controller.forward(target_end_effector_position=step_target_pos, target_end_effector_orientation=quat_busbar_0deg)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))

                glue_busbar_to_ee(robot, busbar_xform, busbar_start_grasp_pos, blend=1.0)

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                corrected_insert_pos = np.array([busbar_insert_xy[0], busbar_insert_xy[1], TARGET_INSERT_POS[2]])
                dist_err = math.dist(cur_pos, tuple(corrected_insert_pos))

                if cur_pos[2] <= BUSBAR_RELEASE_Z or dist_err < INSERT_TOLERANCE_STRICT:
                    if busbar_xform is not None:
                        busbar_xform.set_world_pose(position=target_mid_pos, orientation=BUSBAR_REST_ORIENTATION)
                    print(f"[OK] 2. 버스바 체결 완료 (EE Z: {cur_pos[2]:.4f}m)! -> 그리퍼 열기 및 상공 이탈")
                    phase = "BUSBAR_RELEASE_AND_RETRACT"
                    step_count = 0
            
            elif phase == "BUSBAR_RELEASE_AND_RETRACT":
                actions = arm_controller.forward(target_end_effector_position=TARGET_DESTINATION_POS, target_end_effector_orientation=quat_busbar_0deg)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
            
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(TARGET_DESTINATION_POS))
                if current_err < INSERT_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 3. 버스바 상공 이탈 완료! -> 너트 1번 체결 공정 진입\n")
                    phase = "NUT1_APPROACH"
                    step_count = 0

            # ════════════════════════════════════════════════════════════════
            # [2] 너트 1번 물리 파지 및 상승 (nut1 -> bolt1)
            # ════════════════════════════════════════════════════════════════
            elif phase == "NUT1_APPROACH":
                actions = arm_controller.forward(target_end_effector_position=NUT1_APPROACH_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(NUT1_APPROACH_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 너트 1번 상공 도착! -> 하강 시작")
                    phase = "NUT1_DESCEND"
                    step_count = 0

            elif phase == "NUT1_DESCEND":
                actions = arm_controller.forward(target_end_effector_position=NUT1_PICK_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(NUT1_PICK_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 너트 1번 하강 완료! -> 물리 파지(Gripper Close) 시작")
                    phase = "NUT1_GRASP"
                    grasp_timer = 0

            elif phase == "NUT1_GRASP":
                actions = arm_controller.forward(target_end_effector_position=NUT1_PICK_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                
                grasp_timer += 1
                ramp_frac = min(grasp_timer / GRIP_CLOSE_RAMP_STEPS, 1.0)
                grip_target = ramp_frac * GRIPPER_CLOSE_NUT
                robot.gripper.apply_action(ArticulationAction(joint_positions=grip_target))

                if grasp_timer >= 50:
                    print(f"[OK] 너트 1번 물리 파지 완료! -> 상공({NUT_APPROACH_Z}m)으로 상승")
                    phase = "NUT1_LIFT"
                    step_count = 0

            elif phase == "NUT1_LIFT":
                actions = arm_controller.forward(target_end_effector_position=NUT1_APPROACH_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(NUT1_APPROACH_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 너트 1번 상승 완료! -> 볼트 1번 상공({BOLT1_APPROACH_POS})으로 이동 시작")
                    phase = "MOVE_TO_BOLT1"
                    step_count = 0

            # ════════════════════════════════════════════════════════════════
            # [3] 볼트 1번 상공 이동 후 착좌 하강
            # ════════════════════════════════════════════════════════════════
            elif phase == "MOVE_TO_BOLT1":
                actions = arm_controller.forward(target_end_effector_position=BOLT1_APPROACH_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(BOLT1_APPROACH_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 볼트 1번 상공 도착! -> 착좌 하강 시작")
                    phase = "NUT1_DESCEND_TO_BOLT1"
                    step_count = 0
                    descend_target_z = BOLT1_APPROACH_POS[2]

            elif phase == "NUT1_DESCEND_TO_BOLT1":
                descend_target_z = max(descend_target_z - INSERT_SPEED, BOLT1_TOUCH_POS[2])
                step_target_pos = np.array([BOLT1_TOUCH_POS[0], BOLT1_TOUCH_POS[1], descend_target_z])

                actions = arm_controller.forward(target_end_effector_position=step_target_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                
                if abs(cur_pos[2] - BOLT1_TOUCH_POS[2]) < PICK_TOLERANCE_LOOSE_VAL or descend_target_z <= BOLT1_TOUCH_POS[2]:
                    ee_now_pos, ee_now_quat = robot.end_effector.get_world_pose()
                    screw_start_quat = np.asarray(ee_now_quat).copy()
                    screw_seat_ee_pos = np.asarray(ee_now_pos).copy()
                    
                    if nut1_xform is not None:
                        real_nut_pos, _ = nut1_xform.get_world_pose()
                        screw_seat_pos = np.array([BOLT1_POS[0], BOLT1_POS[1], real_nut_pos[2]])
                    else:
                        screw_seat_pos = np.array([BOLT1_POS[0], BOLT1_POS[1], BOLT1_POS[2]])
                        
                    screw_seat_quat = quat_nut.copy()

                    screw_sub = "rotate"
                    screw_pass_idx = 0
                    screw_pass_theta = 0.0
                    stuck_counter = 0
                    prev_ee_z = ee_now_pos[2]

                    print(f"[OK] 볼트 1번 착좌 완료 (너트 Z={screw_seat_pos[2]:.4f}m)! -> Screwing 시작")
                    phase = "NUT1_SCREW"
                    step_count = 0

            # ════════════════════════════════════════════════════════════════
            # [4] Screwing (너트 1번 -> 볼트 1번 체결)
            # ════════════════════════════════════════════════════════════════
            elif phase == "NUT1_SCREW":
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

                    screw_records.append({
                        "nut_id": 1,
                        "step": step_count,
                        "pass_idx": screw_pass_idx,
                        "theta_deg": screw_pass_theta,
                        "total_deg": total_deg,
                        "depth_mm": depth_m * 1000.0,
                        "tcp_z": float(cur_ee_pos[2]),
                        "torque_nm": float(curr_torque),
                        "stuck_counter": stuck_counter,
                        "seated": bool(is_seated_by_torque),
                    })

                    if step_count % 20 == 0:
                        print(f"  [NUT1 SCREW] Pass {screw_pass_idx+1}/{1+REGRASP_CYCLES} | Theta: {screw_pass_theta:.1f}° | 깊이: {depth_m*1000:.2f}mm / 목표 {ENGAGE_LEN*1000:.1f}mm | TCP Z: {cur_ee_pos[2]:.4f}m | 토크: {curr_torque:.1f}Nm")

                    if pass_done or is_seated_by_torque:
                        if is_seated_by_torque:
                            print(f"  [체결 감지] 너트 1번 완착(Seating) 감지! (TCP Z: {cur_ee_pos[2]:.4f}m <= 0.378m, 토크: {curr_torque:.1f}Nm) -> Screwing 조기 종료 및 그리퍼 해제")

                        if depth_m >= ENGAGE_LEN or screw_pass_idx >= REGRASP_CYCLES or is_seated_by_torque:
                            robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
                            print(f"\n[OK] 너트 1번 체결 완료! -> 꼬인 방향 유지한 채 수직 상승 시작")
                            phase = "NUT1_RETRACT_LIFT"
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

            # ════════════════════════════════════════════════════════════════
            # [5] 너트 1번 체결 후: 수직 상승 -> 상공에서 되감기 -> 정렬
            # ════════════════════════════════════════════════════════════════
            elif phase == "NUT1_RETRACT_LIFT":
                ee_now_pos, _ = robot.end_effector.get_world_pose()
                lift_target_pos = np.array([1.0033894, 0.1387623, 0.8001476])

                last_screw_quat = yaw_rotated_quat(screw_start_quat, screw_pass_theta)
                actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=last_screw_quat)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(lift_target_pos))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 수직 이탈 완료! -> 상공 안전 지대에서 6번 조인트 0도로 되감기(Unwind) 시작")
                    phase = "NUT1_RETRACT_UNWIND"
                    screw_unwind_deg = screw_pass_theta
                    step_count = 0

            elif phase == "NUT1_RETRACT_UNWIND":
                screw_unwind_deg = max(screw_unwind_deg - SCREW_OMEGA_DEG_S * PHYSICS_DT, 0.0)
                ee_now_pos, _ = robot.end_effector.get_world_pose()
                lift_target_pos = np.array([ee_now_pos[0], ee_now_pos[1], NUT_APPROACH_Z])

                target_quat = yaw_rotated_quat(screw_start_quat, screw_unwind_deg)
                actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=target_quat)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                if screw_unwind_deg <= 0.0:
                    print(f"[OK] 6번 조인트 0도 원점 되감기 완료! -> 기본 방향(quat_nut) 정렬")
                    phase = "NUT1_RETRACT_ROTATE"
                    step_count = 0

            elif phase == "NUT1_RETRACT_ROTATE":
                ee_now_pos, _ = robot.end_effector.get_world_pose()
                lift_target_pos = np.array([ee_now_pos[0], ee_now_pos[1], NUT_APPROACH_Z])

                actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(lift_target_pos))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 너트 1번 완전히 이탈 성공! -> [너트 2번 체결 공정 시작]\n")
                    phase = "NUT2_APPROACH"
                    step_count = 0

            # ════════════════════════════════════════════════════════════════
            # [6] 너트 2번 물리 파지 및 상승 (nut2 -> bolt2)
            # ════════════════════════════════════════════════════════════════
            elif phase == "NUT2_APPROACH":
                actions = arm_controller.forward(target_end_effector_position=NUT2_APPROACH_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(NUT2_APPROACH_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 너트 2번 상공 도착! -> 하강 시작")
                    phase = "NUT2_DESCEND"
                    step_count = 0

            elif phase == "NUT2_DESCEND":
                actions = arm_controller.forward(target_end_effector_position=NUT2_PICK_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(NUT2_PICK_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 너트 2번 하강 완료! -> 물리 파지(Gripper Close) 시작")
                    phase = "NUT2_GRASP"
                    grasp_timer = 0

            elif phase == "NUT2_GRASP":
                actions = arm_controller.forward(target_end_effector_position=NUT2_PICK_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                
                grasp_timer += 1
                ramp_frac = min(grasp_timer / GRIP_CLOSE_RAMP_STEPS, 1.0)
                grip_target = ramp_frac * GRIPPER_CLOSE_NUT
                robot.gripper.apply_action(ArticulationAction(joint_positions=grip_target))

                if grasp_timer >= 50:
                    print(f"[OK] 너트 2번 물리 파지 완료! -> 상공({NUT_APPROACH_Z}m)으로 상승")
                    phase = "NUT2_LIFT"
                    step_count = 0

            elif phase == "NUT2_LIFT":
                actions = arm_controller.forward(target_end_effector_position=NUT2_APPROACH_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(NUT2_APPROACH_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 너트 2번 상승 완료! -> 볼트 2번 상공({BOLT2_APPROACH_POS})으로 이동 시작")
                    phase = "MOVE_TO_BOLT2"
                    step_count = 0

            # ════════════════════════════════════════════════════════════════
            # [7] 볼트 2번 상공 이동 후 착좌 하강
            # ════════════════════════════════════════════════════════════════
            elif phase == "MOVE_TO_BOLT2":
                actions = arm_controller.forward(target_end_effector_position=BOLT2_APPROACH_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(BOLT2_APPROACH_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 볼트 2번 상공 도착! -> 착좌 하강 시작")
                    phase = "NUT2_DESCEND_TO_BOLT2"
                    step_count = 0
                    descend_target_z = BOLT2_APPROACH_POS[2]

            elif phase == "NUT2_DESCEND_TO_BOLT2":
                descend_target_z = max(descend_target_z - INSERT_SPEED, BOLT2_TOUCH_POS[2])
                step_target_pos = np.array([BOLT2_TOUCH_POS[0], BOLT2_TOUCH_POS[1], descend_target_z])

                actions = arm_controller.forward(target_end_effector_position=step_target_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                
                if abs(cur_pos[2] - BOLT2_TOUCH_POS[2]) < PICK_TOLERANCE_LOOSE_VAL or descend_target_z <= BOLT2_TOUCH_POS[2]:
                    ee_now_pos, ee_now_quat = robot.end_effector.get_world_pose()
                    screw_start_quat = np.asarray(ee_now_quat).copy()
                    screw_seat_ee_pos = np.asarray(ee_now_pos).copy()
                    
                    if nut2_xform is not None:
                        real_nut_pos, _ = nut2_xform.get_world_pose()
                        screw_seat_pos = np.array([BOLT2_POS[0], BOLT2_POS[1], real_nut_pos[2]])
                    else:
                        screw_seat_pos = np.array([BOLT2_POS[0], BOLT2_POS[1], BOLT2_POS[2]])
                        
                    screw_seat_quat = quat_nut.copy()

                    screw_sub = "rotate"
                    screw_pass_idx = 0
                    screw_pass_theta = 0.0
                    stuck_counter = 0
                    prev_ee_z = ee_now_pos[2]

                    print(f"[OK] 볼트 2번 착좌 완료 (너트 Z={screw_seat_pos[2]:.4f}m)! -> Screwing 시작")
                    phase = "NUT2_SCREW"
                    step_count = 0

            # ════════════════════════════════════════════════════════════════
            # [8] Screwing (너트 2번 -> 볼트 2번 체결)
            # ════════════════════════════════════════════════════════════════
            elif phase == "NUT2_SCREW":
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

                    screw_records.append({
                        "nut_id": 2,
                        "step": step_count,
                        "pass_idx": screw_pass_idx,
                        "theta_deg": screw_pass_theta,
                        "total_deg": total_deg,
                        "depth_mm": depth_m * 1000.0,
                        "tcp_z": float(cur_ee_pos[2]),
                        "torque_nm": float(curr_torque),
                        "stuck_counter": stuck_counter,
                        "seated": bool(is_seated_by_torque),
                    })

                    if step_count % 20 == 0:
                        print(f"  [NUT2 SCREW] Pass {screw_pass_idx+1}/{1+REGRASP_CYCLES} | Theta: {screw_pass_theta:.1f}° | 깊이: {depth_m*1000:.2f}mm / 목표 {ENGAGE_LEN*1000:.1f}mm | TCP Z: {cur_ee_pos[2]:.4f}m | 토크: {curr_torque:.1f}Nm")

                    if pass_done or is_seated_by_torque:
                        if is_seated_by_torque:
                            print(f"  [체결 감지] 너트 2번 완착(Seating) 감지! (TCP Z: {cur_ee_pos[2]:.4f}m <= 0.378m, 토크: {curr_torque:.1f}Nm) -> Screwing 조기 종료 및 그리퍼 해제")

                        if depth_m >= ENGAGE_LEN or screw_pass_idx >= REGRASP_CYCLES or is_seated_by_torque:
                            robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
                            print(f"\n[OK] 너트 2번 볼트 2번 체결 완료! -> 꼬인 방향 유지한 채 수직 상승 시작")
                            phase = "NUT2_RETRACT_LIFT"
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

            # ════════════════════════════════════════════════════════════════
            # [9] 너트 2번 체결 후: 수직 상승 -> 상공에서 되감기 -> 정렬
            # ════════════════════════════════════════════════════════════════
            elif phase == "NUT2_RETRACT_LIFT":
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
                    phase = "NUT2_RETRACT_UNWIND"
                    screw_unwind_deg = screw_pass_theta
                    step_count = 0

            elif phase == "NUT2_RETRACT_UNWIND":
                screw_unwind_deg = max(screw_unwind_deg - SCREW_OMEGA_DEG_S * PHYSICS_DT, 0.0)
                ee_now_pos, _ = robot.end_effector.get_world_pose()
                lift_target_pos = np.array([ee_now_pos[0], ee_now_pos[1], NUT_APPROACH_Z])

                target_quat = yaw_rotated_quat(screw_start_quat, screw_unwind_deg)
                actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=target_quat)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                if screw_unwind_deg <= 0.0:
                    print(f"[OK] 6번 조인트 0도 원점 되감기 완료! -> 기본 방향(quat_nut) 정렬")
                    phase = "NUT2_RETRACT_ROTATE"
                    step_count = 0

            elif phase == "NUT2_RETRACT_ROTATE":
                ee_now_pos, _ = robot.end_effector.get_world_pose()
                lift_target_pos = np.array([ee_now_pos[0], ee_now_pos[1], NUT_APPROACH_Z])

                actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(lift_target_pos))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"\n[전체 시퀀스 최종 성공] 버스바 장착 + 너트 1, 2번 체결 및 로봇 후퇴 완료!")
                    phase = "DONE"

            # 실시간 로그 출력
            if step_count % 30 == 0 and not phase.endswith("GRASP"):
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                print(f"  [{phase}] Step {step_count:03d} | EE: {np.round(cur_pos, 4)} | Err: {current_err*1000:6.2f} mm")

            step_count += 1

        was_playing = playing

    if screw_records:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(SCREW_LOG_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SCREW_LOG_FIELDS)
            writer.writeheader()
            writer.writerows(screw_records)
        print(f"[INFO] 체결(Screwing) 로그 저장 완료 -> {SCREW_LOG_CSV} ({len(screw_records)} rows)")

    alignment_listener.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    if 'world' in locals() and world is not None:
        world.clear_instance()
    omni.usd.get_context().close_stage()
    gc.collect()

    simulation_app.close()


if __name__ == "__main__":
    main()
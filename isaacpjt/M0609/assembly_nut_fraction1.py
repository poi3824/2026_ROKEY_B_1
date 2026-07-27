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

# ROS2 Bridge 활성화 (execute_isaac.py / assembly_vision.py와 동일 패턴 -
# "isaacsim.ros2.bridge"는 import는 성공하지만 enable_extension이 조용히 no-op되어
# rclpy가 로드되지 않는 환경이 확인됨. 실제로 rclpy를 사용하려면 이 방식이어야 함)
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()
sys.stdout.reconfigure(line_buffering=True)

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

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
BOLT_NUT_STATIC_FRICTION  = 1.0   # 정적 마찰 계수
BOLT_NUT_DYNAMIC_FRICTION = 1.0   # 동적 마찰 계수
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
SCREW_TURNS_DEG   = 350.0     # 1패스당 350도
SCREW_OMEGA_DEG_S = 240.0     # 초당 240도 회전 (실제 피치 적용으로 늘어난 회전수 보상, 2배 가속)
PHYSICS_DT        = 1.0 / 60.0
REGRASP_LIFT_HEIGHT = 0.05    # Regrasp 시 수직 상승 높이 (5cm)
REGRASP_Z_OFFSET    = 0.005   # Regrasp 하강/재파지 시 너트를 잡는 위치 보정 높이 (3mm 상승)

# ★ 나사산 피치를 bolt.usd 메쉬에서 직접 실측(고정각도 슬라이스, root-to-root 간격)해서
#   사용한다 - 기존엔 ENGAGE_LEN을 임의 회전수(2회전)로 나눠 역산한 가짜 피치였고,
#   실제 나사산 형상/간섭과 무관했다(그래서 체결 중 반력 토크가 거의 0이었음).
#   bolt.usd 자체(mm 단위) 실측 피치=0.9066mm/rev, 메이저지름=6.35mm(1/4인치) ->
#   1/4인치-28 UNF 이론 피치(25.4/28=0.9071mm)와 거의 정확히 일치.
#   다만 World0123.usd에서 nut1/nut2는 이 원본 자산 대비 1.78배로 스케일돼 배치돼
#   있어(실측), 실제 체결 대상 크기에 맞춰 그 배율을 반영한다.
_BOLT_NATIVE_PITCH_M = (25.4 / 28.0) / 1000.0   # 1/4인치-28 UNF 이론 피치(m)
_NUT_WORLD_SCALE     = 1.78                     # World0123.usd nut1/nut2 xformOp:scale 실측값
NUT_PITCH_M = _BOLT_NATIVE_PITCH_M * _NUT_WORLD_SCALE   # ≈ 1.6147mm/rev

# 실제 피치로 ENGAGE_LEN까지 도달하는 데 필요한 회전수를 역산해 REGRASP_CYCLES를 정한다
# (기존엔 회전수를 고정하고 피치를 거꾸로 맞췄지만, 이제 피치가 실측 고정값이므로 반대 방향).
# +1패스는 부동소수 오차/여유 마진 - depth_m >= ENGAGE_LEN 도달 시 그 전에 조기 종료된다.
_TOTAL_REV_NEEDED = ENGAGE_LEN / NUT_PITCH_M                              # ≈ 7.74회전
REGRASP_CYCLES    = math.ceil(_TOTAL_REV_NEEDED * 360.0 / SCREW_TURNS_DEG)  # ≈ 8 (총 9패스)

# ★ 완착/토크 감지 파라미터 ★
TORQUE_THRESHOLD      = 45.0   # 6번 조인트 반력 임계값 (Nm)
# STUCK_Z_DELTA_THRESH: 실제 나사산 피치 적용 후 회전당 Z 하강량이 훨씬 작아져서
# (기존 가짜 피치 6.43mm/rev -> 실측 1.61mm/rev) 고정값(0.1mm)을 쓰면 정상 진행 중
# 스텝당 하강량(≈0.018mm)마저 이 값보다 작아 매 스텝 stuck_counter가 증가하고, 결국
# 목표 깊이(12.5mm)의 절반(6.22mm)도 못 가서 조기 완착으로 오검출됨(실측 확인).
# SCREW_OMEGA_DEG_S/NUT_PITCH_M이 나중에 또 바뀌어도 항상 유효하도록, 정상 진행 시
# 스텝당 예상 하강량의 30%를 기준으로 동적 계산한다(진짜 정체는 이보다 훨씬 작음).
_NOMINAL_Z_STEP_M     = (SCREW_OMEGA_DEG_S * PHYSICS_DT / 360.0) * NUT_PITCH_M
STUCK_Z_DELTA_THRESH  = 0.3 * _NOMINAL_Z_STEP_M  # ≈ 5.4um
STUCK_STEP_LIMIT      = 12      # Z축 변화 없이 토크 지속되는 Step 수
TCP_FORCE_CHECK_Z     = 0.378   # TCP(EE) 높이가 0.37m 이하일 때만 힘/토크 감지 활성화

URDF_PATH        = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
ROBOT_DESC_PATH  = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
RMPFLOW_CFG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")

# ★ Screwing 체결 로그 저장 경로 (후처리 시각화용, plot_screw_log.py 참고) ★
LOG_DIR       = _THIS_DIR / "logs"
SCREW_LOG_CSV = LOG_DIR / "screw_log.csv"
SCREW_LOG_FIELDS = ["nut_id", "step", "pass_idx", "theta_deg", "total_deg", "depth_mm", "tcp_z", "torque_nm", "stuck_counter", "seated"]

# ★ YawAligner(PI) 실시간 모니터링 로그 저장 경로 (plot_yaw_align_log.py 참고) ★
YAW_ALIGN_LOG_CSV = LOG_DIR / "yaw_align_log.csv"
YAW_ALIGN_LOG_FIELDS = ["t_s", "step", "phase", "error_deg", "correction_deg", "dx_mm", "dy_mm", "saturated"]


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

class YawAligner:
    """버스바 yaw 정렬 오차(dTheta)를 0으로 미는 증분형(incremental) PI 보정기.

    이 문제는 "고정된 기하학적 오정렬(D) - 지금까지 명령한 누적 보정량(C)" = 잔여오차
    구조라, 매 스텝 오차를 새로 계산해서 명령을 통째로 덮어쓰는 표준 위치형(positional)
    PID는 맞지 않는다 - 시뮬레이션해보면 위치형 P만으로는 0으로 수렴하지 않고, D항은
    물리스텝 dt(1/60s)에서 게인이 과도하게 커져 ±한계각에 튕기며 진동한다(RMPFlow
    전달함수를 정식으로 동정하지 않고 D항을 썼을 때 실측됨).
    대신 증분형(velocity-form) PI를 쓴다: 매 "새" 비전 샘플마다
        C += Ki * error + Kp * (error - 직전 error)

    ★ 게인은 감이 아니라 RMPFlow 실제 플랜트(목표 yaw 커맨드 -> EE 실제 yaw)의 스텝응답을
    직접 측정해서 설계했다 (run_yaw_step_response_test/YAW_SYSID=1, logs/yaw_step_response.csv).
    실측 결과: 오버슈트 없는 순수 1차계, 시간상수 τ≈0.283s, DC게인 K≈1.0.

    ⚠ 주의: 이 τ=0.283s는 "커맨드→EE 실제 자세" 만 측정한 값이고, 실제 폐루프에는
    로봇 이동→버스바 회전→카메라 촬영→OpenCV 처리(error_fix_depth1.py)→ROS 퍼블리시/
    구독까지의 비전 왕복 지연시간(θ)이 추가로 들어간다. 이 θ를 측정하지 않고 θ=0으로
    가정한 채 λ=τ/3(Kp=3.0)까지 대역폭을 밀어붙였더니 실측(2026-07-26)에서 진동 심화 +
    정상상태 미수렴이 실제로 관측됨 - 미지의 지연시간이 있는 루프에서 λ를 과도하게
    작게 잡으면 전형적으로 나타나는 증상. θ를 정식으로 재측정하기 전까지는 안전 마진이
    큰 λ=τ(Kp=1.0)로 되돌린다.

    SIMC 튜닝법(Skogestad, FOPDT): Kp = τ/(K·(λ+θ)), τI = min(τ, 4(λ+θ)).
    λ=τ (θ=0 가정 시 가장 보수적인 선택, τ/1 ≥ τ/4라 τI=τ 그대로 적용):
        Kp = τ/(K·λ) = τ/(K·τ) = 1/K = 1.0
        Ki(연속) = Kp/τI = Kp/τ ≈ 3.534,  Ki(코드, 스텝당) = Ki(연속) × Δt(1/60s) ≈ 0.0589
    로 산출 - 재시뮬레이션(실측 τ 기반 1차 플랜트 폐루프, θ=0 가정)에서 D=-1.86deg 기준
    약 1.6초 안에 0.05deg 이하로 단조 수렴(오버슈트 없음)하는 것을 확인. 실제 θ>0이므로
    이보다 다소 느리거나 약간의 오버슈트가 있을 수 있음 - 그래도 여전히 진동/발산하면
    θ를 실측해서 다시 설계해야 한다.
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


def glue_busbar_to_ee(robot, busbar_xform, rest_pick_pos, blend):
    if busbar_xform is None or rest_pick_pos is None:
        return

    ee_pos, ee_quat = robot.end_effector.get_world_pose()
    grasp_point_pos = np.asarray(ee_pos) - EE_OFFSET
    target_pos = grasp_point_pos - np.array([0.0, 0.0, BUSBAR_GRASP_Z_LOCAL])

    busbar_pos = rest_pick_pos + blend * (target_pos - rest_pick_pos)
    busbar_xform.set_world_pose(position=busbar_pos, orientation=np.asarray(ee_quat))


def run_yaw_step_response_test(world, robot, arm_controller, hold_pos, quat_base, step_deg,
                                hold_steps, log_steps, csv_path):
    """YawAligner 게인을 감으로 잡지 않고, RMPFlow 실제 플랜트(목표 yaw 커맨드 -> EE 실제
    yaw)의 스텝응답을 직접 측정해서 CSV로 남긴다 - 여기서 얻은 시간상수/게인으로 PI를
    설계한다(YAW_SYSID=1 환경변수로만 실행되는 진단 전용 경로, 본 조립 시퀀스와 무관)."""
    rows = []
    # 1) 기준 자세로 정지 유지하며 정착 (스텝 인가 전 초기 과도응답 제거)
    for _ in range(hold_steps):
        actions = arm_controller.forward(target_end_effector_position=hold_pos, target_end_effector_orientation=quat_base)
        robot.apply_action(actions)
        world.step(render=False)

    step_quat = yaw_rotated_quat(quat_base, step_deg)
    r_base = R.from_quat([quat_base[1], quat_base[2], quat_base[3], quat_base[0]])

    # 2) step_deg 만큼 목표 yaw를 계단형으로 바꾸고 EE 실제 yaw 응답을 매 스텝 로깅
    for i in range(log_steps):
        actions = arm_controller.forward(target_end_effector_position=hold_pos, target_end_effector_orientation=step_quat)
        robot.apply_action(actions)
        world.step(render=False)

        _, actual_quat = robot.end_effector.get_world_pose()
        r_actual = R.from_quat([actual_quat[1], actual_quat[2], actual_quat[3], actual_quat[0]])
        r_rel = r_actual * r_base.inv()
        yaw_deg = math.degrees(r_rel.as_rotvec()[2])
        rows.append({"step": i, "t_s": i * PHYSICS_DT, "target_deg": step_deg, "actual_deg": yaw_deg})

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "t_s", "target_deg", "actual_deg"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SYSID] yaw step-response 저장 완료 -> {csv_path} ({len(rows)} rows)")


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

    if os.environ.get("YAW_SYSID") == "1":
        print("[SYSID] Yaw 스텝응답 측정 모드 - 본 조립 시퀀스는 건너뜀")
        run_yaw_step_response_test(
            world, robot, arm_controller,
            hold_pos=TARGET_DESTINATION_POS, quat_base=quat_busbar_0deg,
            step_deg=5.0, hold_steps=120, log_steps=180,
            csv_path=LOG_DIR / "yaw_step_response.csv",
        )
        simulation_app.close()
        return

    # ★ error_fix_depth.py가 퍼블리시하는 /busbar_alignment_error 구독용 non-blocking 리스너 ★
    if not rclpy.ok():
        rclpy.init()
    alignment_listener = BusbarAlignmentListener()
    # λ=τ 보수적 SIMC 설계값 (YawAligner 클래스 문서 참고 - 비전 왕복 지연시간을
    # 아직 실측 못 해서 안전 마진을 크게 둠) - Kp=1.0, Ki=0.0589(스텝당).
    # 출력은 ±15deg로 클램프해 급격한 점프 방지.
    yaw_aligner = YawAligner(kp=1.0, ki=0.0589, out_limit_deg=15.0)
    last_dtheta_stamp = 0.0
    yaw_align_start_time = None  # yaw_aligner.reset() 시점에 찍는 기준 시각 (t_s 축 기준)

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

    # ★ YawAligner(PI) 실시간 모니터링 로그 버퍼 (error/correction 수렴 확인용) ★
    yaw_align_records = []

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
            yaw_align_records = []
            yaw_align_start_time = None
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
                if step_count == 0:
                    yaw_aligner.reset()
                    last_dtheta_stamp = 0.0
                    yaw_align_start_time = time.time()
                if alignment_listener.last_stamp > last_dtheta_stamp and \
                        (time.time() - alignment_listener.last_stamp) <= BUSBAR_ALIGNMENT_MAX_AGE:
                    last_dtheta_stamp = alignment_listener.last_stamp
                    _raw_error_deg = math.degrees(alignment_listener.dtheta)
                    yaw_aligner.update(_raw_error_deg)
                    # XY도 yaw와 같은 프레임에서 같이 갱신 - 버스바 구멍은 회전축에서
                    # 떨어져 있어서, yaw가 계속 바뀌는 동안 XY를 한 번만 캡처해서 고정하면
                    # 그 뒤 yaw 보정 때문에 구멍 위치가 다시 어긋난다(실측 확인됨).
                    busbar_insert_xy = np.array([
                        TARGET_INSERT_POS[0] - alignment_listener.dx,
                        TARGET_INSERT_POS[1] - alignment_listener.dy,
                    ])
                    yaw_align_records.append({
                        "t_s": time.time() - yaw_align_start_time,
                        "step": step_count,
                        "phase": phase,
                        "error_deg": _raw_error_deg,
                        "correction_deg": yaw_aligner.correction_deg,
                        "dx_mm": alignment_listener.dx * 1000.0,
                        "dy_mm": alignment_listener.dy * 1000.0,
                        "saturated": abs(yaw_aligner.correction_deg) >= yaw_aligner.out_limit_deg,
                    })
                quat_busbar_yawfix = yaw_rotated_quat(quat_busbar_0deg, yaw_aligner.correction_deg)

                actions = arm_controller.forward(target_end_effector_position=TARGET_DESTINATION_POS, target_end_effector_orientation=quat_busbar_yawfix)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))

                glue_busbar_to_ee(robot, busbar_xform, busbar_start_grasp_pos, blend=1.0)

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(TARGET_DESTINATION_POS))
                if current_err < INSERT_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 1. 체결위치 상공 접근 완료! -> 2. 점진적 하강(-{INSERT_SPEED}m/step) 시작")
                    phase = "BUSBAR_DESCEND_TO_BOLT"
                    step_count = 0
                    descend_target_z = TARGET_DESTINATION_POS[2]

            elif phase == "BUSBAR_DESCEND_TO_BOLT":
                descend_target_z = max(descend_target_z - INSERT_SPEED, TARGET_INSERT_POS[2])
                step_target_pos = np.array([busbar_insert_xy[0], busbar_insert_xy[1], descend_target_z])

                if alignment_listener.last_stamp > last_dtheta_stamp and \
                        (time.time() - alignment_listener.last_stamp) <= BUSBAR_ALIGNMENT_MAX_AGE:
                    last_dtheta_stamp = alignment_listener.last_stamp
                    _raw_error_deg = math.degrees(alignment_listener.dtheta)
                    yaw_aligner.update(_raw_error_deg)
                    # 하강 중에도 XY를 계속 실시간으로 추적(위와 동일한 이유)
                    busbar_insert_xy = np.array([
                        TARGET_INSERT_POS[0] - alignment_listener.dx,
                        TARGET_INSERT_POS[1] - alignment_listener.dy,
                    ])
                    yaw_align_records.append({
                        "t_s": time.time() - yaw_align_start_time,
                        "step": step_count,
                        "phase": phase,
                        "error_deg": _raw_error_deg,
                        "correction_deg": yaw_aligner.correction_deg,
                        "dx_mm": alignment_listener.dx * 1000.0,
                        "dy_mm": alignment_listener.dy * 1000.0,
                        "saturated": abs(yaw_aligner.correction_deg) >= yaw_aligner.out_limit_deg,
                    })
                quat_busbar_yawfix = yaw_rotated_quat(quat_busbar_0deg, yaw_aligner.correction_deg)

                actions = arm_controller.forward(target_end_effector_position=step_target_pos, target_end_effector_orientation=quat_busbar_yawfix)
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
                    if nut1_xform is not None:
                        _dbg_pos, _ = nut1_xform.get_world_pose()
                        print(f"[DEBUG-PHASE0] NUT1_DESCEND 종료(파지 직전, baseline) nut1 world pos={_dbg_pos}")
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
                    if nut1_xform is not None:
                        _dbg_pos, _ = nut1_xform.get_world_pose()
                        print(f"[DEBUG-PHASE0] NUT1_GRASP 종료 시점 nut1 world pos={_dbg_pos}, EE pos={NUT1_PICK_POS}")
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
                    if nut1_xform is not None:
                        _dbg_pos, _ = nut1_xform.get_world_pose()
                        print(f"[DEBUG-PHASE0] NUT1_LIFT 종료 시점 nut1 world pos={_dbg_pos}, EE pos={tuple(cur_pos)} (target {NUT1_APPROACH_POS})")
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
                    if nut1_xform is not None:
                        _dbg_pos, _ = nut1_xform.get_world_pose()
                        print(f"[DEBUG-PHASE0] MOVE_TO_BOLT1 종료 시점 nut1 world pos={_dbg_pos}, EE pos={tuple(cur_pos)} (target {BOLT1_APPROACH_POS})")
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

    if yaw_align_records:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(YAW_ALIGN_LOG_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=YAW_ALIGN_LOG_FIELDS)
            writer.writeheader()
            writer.writerows(yaw_align_records)
        print(f"[INFO] YawAligner(PI) 모니터링 로그 저장 완료 -> {YAW_ALIGN_LOG_CSV} ({len(yaw_align_records)} rows)")

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
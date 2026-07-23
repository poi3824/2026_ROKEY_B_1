"""
11_nut_screw_kinematic.py ─ Kinematic Pose-Glue Nut Screwing (10번/GitHub 이식)

10번(10_thread_controller.py)의 free_nut/screw_nut 두 개 + HANDOFF 스왑 + PhysX
아티큘레이션(PhysxMimicJointAPI) 구조는 그 복잡도 자체가 GPU 크래시("Nested
articulation roots", "artiSolveInternal...fail to launch kernel")의 근본 원인이었다.

이 파일은:
  ① 파지 "로직"만 10번에서 재사용한다(PickPlaceController 이벤트 타이밍, 그리퍼
     닫힘 램프) — 물리적 마찰로 붙잡는 게 아니라, 그립이 시작되는 순간부터 매
     스텝 "그립점 = EE − EE_OFFSET"(PickPlaceController.forward() 가 스스로
     쓰는 공식을 그대로 역산) 으로 너트를 그리퍼에 붙인다(kinematic pose-glue).
     너트는 CollisionAPI 자체가 없다 — 접촉이 필요 없으니 충돌 폭발/튐이
     구조적으로 불가능하다.
  ② 볼트/너트 자산은 GitHub Tech-Multiverse/omniverse-nut-and-bolt-digital-twin
     저장소의 hex_bolt/{bolt,nut}.usd (nut_bolt_assets/ 에 복사) 를 사용한다.
  ③ 체결(SCREW) 도 같은 pose-glue 방식의 연장이다 — PhysX 조인트(RackAndPinion/
     Mimic)로 회전↔하강을 실제로 물리 커플링하는 대신(이 Isaac Sim 빌드에서
     PhysxPhysicsRackAndPinionJoint 가 실측 검증 결과 회전은 추적되지만 프리즘
     쪽에 어떤 힘도 전달하지 않는 것으로 확인됨 — 사실상 동작하지 않음), "손목
     회전량 → 나사 피치로 환산한 하강 깊이"를 Python 에서 직접 계산해 너트의
     kinematic pose(위치+자세)를 매 스텝 갱신한다. 그리퍼-너트 접촉은 끝까지
     한 번도 켜지 않는다(사용자 제안: "접촉 꺼두고 회전속도만 맞추면 시각적으로
     자연스럽다" — 이 파일의 핵심 설계 원칙).
  ④ 체결 표현은 래칫: 그리퍼가 닫힌 채 SCREW_TURNS_DEG(파라미터, 기본 270°)
     회전 → 그립 풀고 손목만 -SCREW_TURNS_DEG 되감기 → 재파지 → 다시 회전.
     REGRASP_CYCLES 회 반복(총 패스 = 1+REGRASP_CYCLES). 너트 자체의 누적
     회전/깊이는 release/unwind 동안 갱신을 멈춰(게이팅) 그대로 유지된다.

이 설계에서 볼트·너트는 물리 강체가 전혀 아니다(RigidBodyAPI/CollisionAPI 없음,
순수 참조 지오메트리) — 로봇 팔/그리퍼만 실제 PhysX 시뮬레이션 대상이고, 너트는
100% 우리가 계산한 kinematic pose 로 매 스텝 재배치된다. 10번을 괴롭혔던 SDF/
meshSimplification/contact-offset/아티큘레이션 튜닝이 전부 불필요해진다.

실행:
  /home/rokey/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh \
      /home/rokey/cobot3_ws/isaacpjt/M0609/11_nut_screw_kinematic.py
  → 뷰포트에서 Play. 환경변수 BOLT_HEADLESS=1 이면 헤드리스 자동 실행 후 종료.
"""

import os
from isaacsim import SimulationApp

_HEADLESS = os.environ.get("BOLT_HEADLESS") == "1"
simulation_app = SimulationApp({"headless": _HEADLESS})

import sys
from pathlib import Path
import numpy as np
import carb
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf

from isaacsim.core.api import World
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.prims import SingleGeometryPrim, SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "rmpflow"))
from m0609_pick_place_controller import PickPlaceController  # noqa: E402
from m0609_rmpflow_controller import RMPFlowController  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
#  [A] 로봇 상수 (10번과 동일 — 검증된 물리 설정 재사용)
# ══════════════════════════════════════════════════════════════════════════
ROBOT_USD_PATH   = str(_THIS_DIR / "Collected_m0609_camera/m0609_camera.usd")
ROBOT_URDF_PATH  = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
ROBOT_DESC_PATH  = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
RMPFLOW_CFG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")
ROBOT_PRIM_PATH  = "/World/m0609"
EE_LINK_NAME     = "link_6"
GRIPPER_JOINTS   = ["finger_joint", "right_inner_knuckle_joint"]
GRIPPER_MECH_JOINTS = {
    "finger_joint", "right_inner_knuckle_joint",
    "left_inner_knuckle_to_finger_joint", "right_inner_knuckle_to_finger_joint",
    "left_inner_finger_joint", "right_inner_finger_joint",
}

DRIVE_STIFFNESS, DRIVE_DAMPING, DRIVE_MAX_FORCE = 1.0e8, 1.0e6, 1.0e8

# 그리퍼 기구부(4절 링크) 드라이브 — 10번에서 실측 검증된 값 그대로 재사용
# (닫힘 접촉 충격 방지 램프, "낮은 강성 위치 서보" 전략의 결론값). 너트에 실제로
# 닿지 않으므로 접촉 관련 튜닝은 필요 없지만, 그리퍼 자체(손가락 4절 링크)가
# 내부적으로 안정적으로 움직이려면 이 드라이브 설정이 여전히 필요하다.
GRIPPER_DRIVE_STIFFNESS = 45.0 / 57.29578
GRIPPER_DRIVE_DAMPING = 5.0 / 57.29578
GRIPPER_HOLD_STIFFNESS = 180.0 / 57.29578
GRIPPER_HOLD_DAMPING = 35.0 / 57.29578
GRIPPER_HOLD_RAMP_STEPS = 50
GRIP_CLOSE_POSITION = 0.96
GRIP_CLOSE_RAMP_STEPS = 75

# ALIGN(pick + 볼트 위 정렬) 단계 이벤트 타이밍 — 10번과 동일(그리퍼 닫힘 램프가
# event3 안에서 끝나도록 튜닝된 값). event7(open) 진입은 가로채 SEAT 로 전환한다.
#   ⚠ 실측(2026-07-22): event6(=하강, "lower to place")이 우리 PLACE_POS 높이까지
#   완전히 수렴하기 전에 event7 로 넘어가 gap 이 몇 mm 남는 현상이 있음(SEAT
#   판정 로그로 확인) — kinematic 스냅이라 결과(최종 체결 깊이/정렬)에는 영향
#   없지만(순간 반영이라 물리적 위험 없음) 시각적으로 약간의 점프가 보일 수
#   있다. dt 값을 줄이는 방향(0.0008)으로 시도했으나 오히려 더 못 미침(gap
#   증가, 실측 확인) — 10번에서 검증된 0.003 이 이 구간에서는 더 낫다.
#   PLACE_POS 자체를 낮추거나 event6 스텝수를 다른 방식으로 늘리는 재튜닝은
#   추후 GUI 로 진행할 것(기능적으로는 이미 100% 체결 성공, 순수 시각 폴리시).
EVENTS_DT = [0.011, 0.006, 0.05, 1.0 / 150, 0.01, 0.013, 0.003, 1.0, 0.011, 0.08]

PHYSICS_DT = 1.0 / 60.0
ART_POS_ITERS, ART_VEL_ITERS = 192, 1


# ══════════════════════════════════════════════════════════════════════════
#  [B] 볼트/너트 자산 (GitHub Tech-Multiverse/omniverse-nut-and-bolt-digital-twin)
#  — 순수 시각 지오메트리로만 쓴다. RigidBodyAPI/CollisionAPI 를 아예 부여하지
#  않는다(체결 동력은 전적으로 Python 이 계산한 kinematic pose 가 담당하므로
#  물리 강체일 필요가 없다 — 10번을 괴롭힌 SDF/contact-offset 튜닝이 전부 불필요).
# ══════════════════════════════════════════════════════════════════════════
BOLT_USD_PATH = str(_THIS_DIR / "nut_bolt_assets/hex_bolt/bolt.usd")
NUT_USD_PATH  = str(_THIS_DIR / "nut_bolt_assets/hex_bolt/nut.usd")

# 실측(add_reference_to_stage, 스케일 보정 없이 참조 시 — 2026-07-22 이 프로젝트의
# Isaac Sim 에서 헤드리스로 직접 측정. GitHub repo 의 static_bolt.usda 는 참조 후
# 별도 ×0.001 스케일을 얹지만, 이 환경에서는 그 보정 없이 이미 mm 단위로 타당한
# 크기가 나옴 — 메트릭 어셈블러가 자동 보정한 것으로 추정. 실측값을 그대로 신뢰):
#   bolt: X ±7.33mm, Y ±6.35mm, Z 0→30.16mm(로컬 원점=바닥, 볼트 끝=30.16mm)
#   nut : X ±7.33mm, Y ±6.35mm, Z 0→4.76mm (로컬 원점=바닥면, 두께 4.76mm)
RAW_BOLT_HEIGHT = 0.030163
RAW_NUT_HEIGHT  = 0.0047625
RAW_HALF_X, RAW_HALF_Y = 0.00733, 0.00635

# RG2 그리퍼 패드(원기둥형, 지름 ~40mm)에 비해 원본 크기가 매우 작아(10번이 겪은
# M8 스케일 미스매치와 유사하거나 더 작음) 그리퍼 안에서 비례가 안 맞아 보인다 —
# 접촉이 필요 없는 구조라 물리적으로는 원본 크기도 문제없지만, 시각적 비례를
# 맞추기 위해 균일 확대한다. 필요 없으면 1.0 으로.
NUT_BOLT_VISUAL_SCALE = 2.0

BOLT_HEIGHT = RAW_BOLT_HEIGHT * NUT_BOLT_VISUAL_SCALE
NUT_HEIGHT  = RAW_NUT_HEIGHT * NUT_BOLT_VISUAL_SCALE
NUT_MASS_UNUSED = None  # 너트는 강체가 아니므로 질량 개념 자체가 없음(참고용 주석)

BOLT_XY     = np.array([0.45, -0.20])   # 볼트 고정 위치
BOLT_BASE_Z = 0.0                       # 볼트 로컬 원점 = 바닥(실측과 일치)
BOLT_TIP_Z  = BOLT_BASE_Z + BOLT_HEIGHT

NUT_PICK_XY = np.array([0.45, 0.20])    # 너트 초기 대기 위치
# 너트 로컬 원점 = 바닥면(실측) → 오프셋 0. 10번의 Factory 너트(원점≠바닥)와 다름.
NUT_ORIGIN_TO_BOTTOM = 0.0

PEDESTAL_HEIGHT = 0.02   # 순수 시각용 — 물리 콜리전이 없으므로 아무 값이나 안전
PEDESTAL_SIZE   = 0.03

SCREW_HOVER_CLEAR = 0.001   # ALIGN 목표: 너트 바닥이 볼트 끝보다 이만큼 위

# ══════════════════════════════════════════════════════════════════════════
#  체결(SCREW) 파라미터 — 전부 Python 이 직접 계산(조인트 없음)
# ══════════════════════════════════════════════════════════════════════════
ENGAGE_LEN = 0.020    # 목표 체결 깊이(볼트 축 방향으로 너트가 내려가는 총 거리)

# 파지/체결 판정 임계값
ENGAGE_XY_TOL_M = 0.006
ENGAGE_TILT_DEG = 8.0
ENGAGE_GAP_M    = 0.003

# ── 사용자 요청: 회전각/재파지 횟수는 나중에 조정 가능하도록 파라미터화 ──
SCREW_TURNS_DEG   = 270.0   # 패스당 회전각(도) — 그리퍼가 닫힌 채 도는 각도
REGRASP_CYCLES    = 2       # 재파지(unwind→regrasp) 횟수 — 총 패스 수 = 1+이값
SCREW_OMEGA_DEG_S = 60.0    # 너트/손목 각속도(도/초) — 둘 다 이 값으로 구동되어
                             #   항상 시각적으로 정확히 일치한다(같은 누적 변수 사용).
#   ⚠ 실측(2026-07-22, GUI 육안 확인): +1.0(그리퍼가 +270 돌며 조임)으로 하니
#   그리퍼와 너트가 반대로 도는 것처럼 보임 — 그리퍼가 -270 돌면서 너트를 잡고
#   돌려야 올바르게 보임(사용자 확인). depth_m 계산은 SCREW_DIRECTION 부호와
#   무관하게 항상 누적 증가하므로(아래 rotate 서브상태 참고) 이 부호는 순수
#   시각적 회전 감각만 바꾼다 — 하강 자체는 영향 없음.
SCREW_DIRECTION   = 1.0    # 조임 방향 부호(그리퍼 회전 감각). 반대로 보이면 1.0.

TOTAL_REV = (SCREW_TURNS_DEG / 360.0) * (1 + REGRASP_CYCLES)
NUT_PITCH_M = ENGAGE_LEN / TOTAL_REV   # 시각용 피치 — 목표 깊이에 정확히 도달하도록 역산

HOME_LIFT_Z = 0.30
#   ⚠ 실측(2026-07-22): PickPlaceController 의 호버 높이(end_effector_initial_height,
#   10번 관례상 0.30 재사용)와 우리 PICK_POS.z(≈25mm, 자산이 아주 작고 받침대가
#   낮아 발생)의 차이가 커서(≈275mm) event1(하강)/event4(들어올림)/event5~6(전이)
#   구간마다 팔 전체가 크게 급강하/급상승하는 것으로 GUI 확인됨("픽할 때도
#   파지할 때도 z 로 너무 많이 내려간다" — 사용자 확인). 볼트 높이(60.3mm)는
#   여유있게 넘으면서 낙차만 줄이도록 낮춘다.
PICK_HOVER_HEIGHT = 0.15

# 파지점(그립점) — 너트 로컬 원점(=바닥) 기준 오프셋. 얇은 너트라 중앙 근처.
NUT_GRASP_Z_LOCAL = NUT_HEIGHT + 0.03

NUT_REST_ROOT_Z = PEDESTAL_HEIGHT   # 너트 원점=바닥이므로 받침대 위에 바로 얹힘
PICK_POS = np.array([NUT_PICK_XY[0], NUT_PICK_XY[1], NUT_REST_ROOT_Z + NUT_GRASP_Z_LOCAL])

NUT_ALIGN_ROOT_Z = BOLT_TIP_Z + SCREW_HOVER_CLEAR   # 너트 원점=바닥이므로 오프셋 불필요
PLACE_POS = np.array([BOLT_XY[0], BOLT_XY[1], NUT_ALIGN_ROOT_Z + NUT_GRASP_Z_LOCAL])

EE_OFFSET = np.array([0.0, 0.0, 0.185])   # 10번 실측치 재사용(같은 로봇/컨트롤러)
NUT_REST_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])   # 받침대 위 원래(업라이트) 자세

MAX_HEADLESS_STEPS = 6000


# ══════════════════════════════════════════════════════════════════════════
#  [C] 헬퍼
# ══════════════════════════════════════════════════════════════════════════
def find_prim_path(root_path, name):
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def set_all_drives(root_path):
    """팔 관절 드라이브(고강성). 그리퍼 기구부는 제외."""
    stage = omni.usd.get_context().get_stage()
    n = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if prim.GetName() in GRIPPER_MECH_JOINTS:
            continue
        for dt in ("angular", "linear"):
            drive = UsdPhysics.DriveAPI.Get(prim, dt)
            if drive:
                drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
                drive.GetDampingAttr().Set(DRIVE_DAMPING)
                drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)
                n += 1
    print(f"  [OK] 팔 드라이브 {n}개 설정 (stiffness={DRIVE_STIFFNESS:.0e})")


_GRIPPER_DRIVE_ATTR_CACHE = []


def set_gripper_drives(root_path, stiffness=GRIPPER_DRIVE_STIFFNESS, damping=GRIPPER_DRIVE_DAMPING,
                        label="설정", verbose=True):
    """그리퍼 기구부(4절 링크) 드라이브 강성/댐핑 설정(10번과 동일 — attr 캐싱으로
       Play 도중 재저작 없이 Set() 만 호출, GPU 캐시 충돌 방지)."""
    global _GRIPPER_DRIVE_ATTR_CACHE
    if not _GRIPPER_DRIVE_ATTR_CACHE:
        stage = omni.usd.get_context().get_stage()
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
            if prim.GetName() not in GRIPPER_MECH_JOINTS:
                continue
            for dt in ("angular", "linear"):
                drive = UsdPhysics.DriveAPI.Get(prim, dt)
                if drive:
                    _GRIPPER_DRIVE_ATTR_CACHE.append((drive.GetStiffnessAttr(), drive.GetDampingAttr()))
    for stiff_attr, damp_attr in _GRIPPER_DRIVE_ATTR_CACHE:
        stiff_attr.Set(stiffness)
        damp_attr.Set(damping)
    if verbose:
        print(f"  [OK] 그리퍼 드라이브 {len(_GRIPPER_DRIVE_ATTR_CACHE)}개 {label} 설정 (stiffness={stiffness:.1f})")


def solver_iters_only(root_path):
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    art_prim = None
    for p in Usd.PrimRange(root):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_prim = p
            break
    art_prim = art_prim or root
    ap = PhysxSchema.PhysxArticulationAPI.Apply(art_prim)
    ap.CreateSolverPositionIterationCountAttr(ART_POS_ITERS)
    ap.CreateSolverVelocityIterationCountAttr(ART_VEL_ITERS)
    ap.CreateEnabledSelfCollisionsAttr(False)
    print(f"  [OK] 솔버 반복수 상향: pos={ART_POS_ITERS} vel={ART_VEL_ITERS}, selfCol=off")


def tune_physics_scene():
    stage = omni.usd.get_context().get_stage()
    scene_prim = None
    for p in Usd.PrimRange(stage.GetPrimAtPath("/")):
        if p.HasAPI(UsdPhysics.Scene) or p.GetTypeName() == "PhysicsScene":
            scene_prim = p
            break
    if scene_prim is None:
        print("  [경고] PhysicsScene prim을 못 찾음 — 씬 레벨 튜닝 생략")
        return
    ps = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    ps.CreateSolverTypeAttr("TGS")
    ps.CreateMaxPositionIterationCountAttr(ART_POS_ITERS)
    ps.CreateMaxVelocityIterationCountAttr(ART_VEL_ITERS)
    ps.CreateBounceThresholdAttr(0.2)
    ps.CreateEnableGPUDynamicsAttr(True)
    ps.CreateBroadphaseTypeAttr("GPU")
    print(f"  [OK] PhysicsScene 튜닝: solverType=TGS maxPosIter={ART_POS_ITERS} GPU dynamics on")


def initialize_robot(robot, world):
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )


def _deinstance(prim_path):
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    for _ in range(6):
        changed = False
        for p in Usd.PrimRange(root):
            if p.IsInstanceable():
                p.SetInstanceable(False)
                changed = True
        if not changed:
            break


def axis_tilt_deg(quat_wxyz):
    w, x, y, z = [float(v) for v in quat_wxyz]
    rot = Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    local_z = rot.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
    cos_a = float(np.clip(local_z[2], -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def yaw_rotated_quat(base_wxyz, delta_deg):
    base_q = Gf.Quatd(float(base_wxyz[0]),
                       Gf.Vec3d(float(base_wxyz[1]), float(base_wxyz[2]), float(base_wxyz[3])))
    base_rot = Gf.Rotation(base_q)
    extra_rot = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), float(delta_deg))
    combined = extra_rot * base_rot
    q = combined.GetQuat()
    return np.array([q.GetReal(), *q.GetImaginary()])


def measured_yaw_delta_deg(base_wxyz, now_wxyz):
    """base_wxyz → now_wxyz 로 가는 동안 world +Z 축 기준으로 실제 얼마나
       돌았는지(부호 있는 도) — "우리가 명령한 목표각"이 아니라 RMPFlow IK 가
       실제로 도달한 EE 자세를 그대로 측정한 값이다.
       ⚠ 실측(2026-07-22): target_quat(우리가 계산한 목표)와 nut_quat 을
       각각 SCREW_DIRECTION 으로 독립 계산했더니, 부호를 뒤집어도(-1↔1) GUI
       상 그리퍼-너트 회전 방향 불일치가 그대로였음(사용자 2회 확인) — 우리
       목표각의 부호가 실제 IK 결과의 시각적 회전 감각과 일치한다는 보장이
       없다는 뜻. 그래서 "계산"을 그만두고 "측정"으로 바꾼다: 너트 자세를
       그리퍼의 실측 회전량에서 직접 뽑아 쓰면, 우리 목표식 부호가 뭐든 항상
       실제 그리퍼 움직임과 일치할 수밖에 없다."""
    base_q = Gf.Quatd(float(base_wxyz[0]),
                       Gf.Vec3d(float(base_wxyz[1]), float(base_wxyz[2]), float(base_wxyz[3])))
    now_q = Gf.Quatd(float(now_wxyz[0]),
                      Gf.Vec3d(float(now_wxyz[1]), float(now_wxyz[2]), float(now_wxyz[3])))
    delta_q = now_q * base_q.GetInverse()   # now = delta * base (world frame)
    w = delta_q.GetReal()
    v = delta_q.GetImaginary()
    return float(np.degrees(2.0 * np.arctan2(v[2], w)))


# ── 볼트/너트 참조 (순수 지오메트리 — 물리 API 없음) ────────────────────────
def reference_visual_only(usd_path, root_path, position_xyz, scale):
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, root_path)
    add_reference_to_stage(usd_path, root_path + "/geo")
    xf = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(root_path))
    xf.SetTranslate(Gf.Vec3d(float(position_xyz[0]), float(position_xyz[1]), float(position_xyz[2])))
    xf.SetScale(Gf.Vec3f(float(scale), float(scale), float(scale)))
    _deinstance(root_path)
    return SingleXFormPrim(root_path, name=root_path.split("/")[-1])


def build_pedestal_visual(xy):
    """순수 시각용 받침대(콜리전 없음) — 너트를 바닥이 아니라 살짝 띄워서 보여주는 용도."""
    stage = omni.usd.get_context().get_stage()
    cx, cy = float(xy[0]), float(xy[1])
    cube = UsdGeom.Cube.Define(stage, "/World/pedestal")
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.XformCommonAPI(cube.GetPrim())
    xf.SetTranslate(Gf.Vec3d(cx, cy, PEDESTAL_HEIGHT / 2.0))
    xf.SetScale(Gf.Vec3f(PEDESTAL_SIZE, PEDESTAL_SIZE, PEDESTAL_HEIGHT))
    cube.CreateDisplayColorAttr().Set([Gf.Vec3f(0.4, 0.4, 0.42)])
    print(f"  [OK] 받침대(시각용, 콜리전 없음) @ ({cx},{cy})")


# ══════════════════════════════════════════════════════════════════════════
#  [D] 메인
# ══════════════════════════════════════════════════════════════════════════
def main():
    carb.settings.get_settings().set_int("/persistent/physics/visualizationDisplayColliders", 2)

    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()

    print("\n[1] 로봇 USD 로드")
    stage = omni.usd.get_context().get_stage()
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    world_prim.GetReferences().AddReference(ROBOT_USD_PATH)
    for _ in range(15):
        simulation_app.update()
    print("  [OK] robot")

    print("\n[2] 물리 설정")
    set_all_drives(ROBOT_PRIM_PATH)
    set_gripper_drives(ROBOT_PRIM_PATH)
    solver_iters_only(ROBOT_PRIM_PATH)
    tune_physics_scene()

    print("\n[3] 볼트/너트 자산 로드 (GitHub repo, 순수 시각 — 물리 API 없음)")
    build_pedestal_visual(NUT_PICK_XY)
    bolt_xform = reference_visual_only(
        BOLT_USD_PATH, "/World/bolt",
        (float(BOLT_XY[0]), float(BOLT_XY[1]), float(BOLT_BASE_Z)),
        NUT_BOLT_VISUAL_SCALE,
    )
    nut_xform = reference_visual_only(
        NUT_USD_PATH, "/World/nut",
        (float(NUT_PICK_XY[0]), float(NUT_PICK_XY[1]), float(NUT_REST_ROOT_Z)),
        NUT_BOLT_VISUAL_SCALE,
    )
    print(f"  [OK] 볼트 @ {tuple(np.round(BOLT_XY, 3))} 높이={BOLT_HEIGHT*1000:.1f}mm"
          f"  너트 @ {tuple(np.round(NUT_PICK_XY, 3))} 두께={NUT_HEIGHT*1000:.1f}mm"
          f" (scale={NUT_BOLT_VISUAL_SCALE})")

    print("\n[4] 로봇 등록")
    ee_path = find_prim_path(ROBOT_PRIM_PATH, EE_LINK_NAME)
    if ee_path is None:
        raise RuntimeError(f"'{EE_LINK_NAME}' 링크를 찾을 수 없음")
    gripper = ParallelGripper(
        end_effector_prim_path=ee_path, joint_prim_names=GRIPPER_JOINTS,
        joint_opened_positions=np.array([0.0, 0.0]),
        joint_closed_positions=np.array([GRIP_CLOSE_POSITION, GRIP_CLOSE_POSITION]),
        action_deltas=None,
    )
    robot = world.scene.add(SingleManipulator(
        prim_path=ROBOT_PRIM_PATH, name="m0609_robot",
        end_effector_prim_path=ee_path, gripper=gripper,
    ))
    print(f"  [OK] SingleManipulator, EE={ee_path}")

    world.reset()
    initialize_robot(robot, world)
    for _ in range(30):
        world.step(render=True)

    gripper_dof_indices = [robot.dof_names.index(n) for n in GRIPPER_JOINTS]
    n_dof = len(robot.dof_names)
    print(f"  [OK] 그리퍼 관절 인덱스={gripper_dof_indices} (전체 관절수={n_dof})")

    print("\n[5] 컨트롤러 생성 (PICK_CARRY=PickPlaceController, SCREW=RMPFlowController)")
    align_controller = PickPlaceController(
        name="m0609_nut_screw_controller",
        gripper=robot.gripper, robot_articulation=robot,
        end_effector_initial_height=PICK_HOVER_HEIGHT, events_dt=EVENTS_DT,
        urdf_path=ROBOT_URDF_PATH, robot_description_path=ROBOT_DESC_PATH,
        rmpflow_config_path=RMPFLOW_CFG_PATH, end_effector_frame_name=EE_LINK_NAME,
    )
    screw_controller = RMPFlowController(
        name="m0609_screw_cspace_controller", robot_articulation=robot,
        urdf_path=ROBOT_URDF_PATH, robot_description_path=ROBOT_DESC_PATH,
        rmpflow_config_path=RMPFLOW_CFG_PATH, end_effector_frame_name=EE_LINK_NAME,
    )
    print(f"  [OK] PICK {np.round(PICK_POS,3)}  ALIGN(볼트 위) {np.round(PLACE_POS,3)}")
    print(f"       체결목표: 패스당 {SCREW_TURNS_DEG:.0f}° x {1+REGRASP_CYCLES}패스 = {TOTAL_REV:.2f}rev,"
          f" pitch={NUT_PITCH_M*1000:.2f}mm/rev(시각용), ENGAGE_LEN={ENGAGE_LEN*1000:.1f}mm"
          f" (kinematic pose-glue — 조인트 없음)")

    # ── 상태머신: PICK_CARRY(pick+정렬, kinematic pose-glue) → SEAT(1스텝 스냅)
    #    → SCREW(래칫, 순수 Python 계산) → SETTLE → JUDGE → HOME ──
    phase = {"name": "PICK_CARRY", "reported": False}

    def reset_cycle():
        stage = omni.usd.get_context().get_stage()
        nut_xform.set_world_pose(
            position=np.array([float(NUT_PICK_XY[0]), float(NUT_PICK_XY[1]), float(NUT_REST_ROOT_Z)]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        world.reset()
        initialize_robot(robot, world)
        for _ in range(30):
            world.step(render=True)
        set_gripper_drives(ROBOT_PRIM_PATH, GRIPPER_DRIVE_STIFFNESS, GRIPPER_DRIVE_DAMPING, label="재설정")
        align_controller.reset()
        screw_controller.reset()
        phase.clear()
        phase.update(name="PICK_CARRY", reported=False)

    def step_cycle():
        """한 물리 스텝만큼 상태머신을 전진시킨다:
           PICK_CARRY → SEAT → SCREW → SETTLE → JUDGE → HOME."""

        def apply_grip_hold():
            grip_action = ArticulationAction(
                joint_positions=np.array([GRIP_CLOSE_POSITION, GRIP_CLOSE_POSITION]),
                joint_indices=np.array(gripper_dof_indices),
            )
            robot.apply_action(grip_action)

        def glue_nut_to_ee(blend):
            """PICK_CARRY 중: 너트를 그리퍼가 잡고 있는 실제 지점에 붙인다.
               ⚠ 실측(2026-07-22, GUI 육안 확인): 이전엔 그립 완료 시점의 "너트
               실제 pose"(그때까지 원래 받침대 위치에 가만히 있던, EE 와 무관한
               값)를 EE 기준 상대 pose 로 캡처해서 재생했는데 — 두 물체가 애초에
               서로 아무 관계가 없었으니 그 결과가 우연에 맡겨진 값이었고, 회전
               성분까지 섞여 들어가 너트가 그리퍼 안쪽으로 파고드는 것처럼 보이고
               (사용자 확인) 이 값이 SEAT 시점 seat_quat 으로 그대로 전달돼
               SCREW 회전 동기화까지 깨졌었다(seat_quat 자체가 이미 이상한
               회전이었으므로 SCREW_DIRECTION 부호를 뒤집어도 고쳐지지 않았던
               이유). PickPlaceController 자신이 쓰는 공식(position_target =
               grasp_point + EE_OFFSET, forward() 참고)을 그대로 역산해 "그립점
               = EE - EE_OFFSET" 으로 직접 계산 — 캡처가 아니라 항상 참인 관계식
               이라 결과가 우연에 좌우되지 않는다. 자세는 그리퍼가 PICK_CARRY
               내내 고정 방향([0,pi,0])만 쓰므로(위치만 보간) 회전시킬 필요가
               없어 원래 놓여있던(=업라이트) 자세를 그대로 유지한다.
               ⚠ "그립점"(위 공식 결과)은 PICK_POS/PLACE_POS 관례상 너트 원점
               (=바닥면) 기준 NUT_GRASP_Z_LOCAL 만큼 위 지점이다 — nut_xform 의
               position 은 원점(바닥면) 기준이므로 그만큼 다시 빼줘야 실제
               너트가 그립점에 놓인다(빠뜨렸다가 gap 이 NUT_GRASP_Z_LOCAL 만큼
               어긋나는 것으로 실측 확인).
               ⚠ 실측(2026-07-22, GUI 육안 확인): event3(그립 닫기) 진입 첫
               프레임에 위 공식으로 곧장 스냅했더니, 손가락은 아직 안 닫혔는데
               너트만 받침대 위치→그립점으로 순간이동(텔레포트)해 "잡히기 직전에
               공중으로 붕 뜬다"는 인상을 줬다(사용자 확인) — 받침대든 트레이든
               "원래 놓여있던 자리"가 계산된 그립점과 다르면 항상 재발하는
               구조적 문제였다. 손가락 닫힘 램프(ramp_frac, GRIP_CLOSE_RAMP_STEPS)
               와 같은 진행률을 blend 로 받아, "원래 놓여있던 자리" → "그립점"을
               같은 속도로 선형 보간해 손가락이 닫히는 것과 같은 리듬으로 자연스럽게
               끌려오게 한다(ramp_frac=1 이 되면 사실상 glue_nut_to_ee(1.0) 과 동일)."""
            ee_pos, _ = robot.end_effector.get_world_pose()
            grasp_point_pos = np.asarray(ee_pos) - EE_OFFSET
            target_pos = grasp_point_pos - np.array([0.0, 0.0, NUT_GRASP_Z_LOCAL])
            rest_pos = np.array([float(NUT_PICK_XY[0]), float(NUT_PICK_XY[1]), float(NUT_REST_ROOT_Z)])
            nut_pos = rest_pos + blend * (target_pos - rest_pos)
            nut_xform.set_world_pose(position=nut_pos, orientation=NUT_REST_ORIENTATION)

        if phase["name"] == "PICK_CARRY":
            phase["_step"] = phase.get("_step", 0) + 1
            ev = align_controller.get_current_event()
            if ev != phase.get("_last_ev", -1) or phase["_step"] % 10 == 0:
                eep_dbg, _ = robot.end_effector.get_world_pose()
                print(f"  [DBG-PICK] step={phase['_step']} ev={ev} EE_z={float(np.asarray(eep_dbg)[2])*1000:.1f}mm"
                      f" (PICK_POS.z={PICK_POS[2]*1000:.1f}mm pedestal_top={PEDESTAL_HEIGHT*1000:.1f}mm)")
                phase["_last_ev"] = ev

            if ev >= 7:
                # PickPlaceController 가 여기서 그리퍼를 열려 함 — 가로채서 SEAT 로 전환.
                sp, sq = robot.end_effector.get_world_pose()
                phase["start_pos"] = np.asarray(sp).copy()
                phase["start_quat"] = np.asarray(sq).copy()

                nut_pos, nut_quat = nut_xform.get_world_pose()
                nut_pos = np.asarray(nut_pos); nut_quat = np.asarray(nut_quat)
                xy_err = float(np.linalg.norm(nut_pos[:2] - BOLT_XY))
                gap_m = BOLT_TIP_Z - float(nut_pos[2])
                tilt = axis_tilt_deg(nut_quat)
                engaged_ok = (xy_err <= ENGAGE_XY_TOL_M) and (tilt <= ENGAGE_TILT_DEG) and (gap_m <= ENGAGE_GAP_M)
                print(f"  [SEAT 판정] xy오차={xy_err*1000:.2f}mm(허용{ENGAGE_XY_TOL_M*1000:.1f})"
                      f" 기울기={tilt:.1f}deg(허용{ENGAGE_TILT_DEG:.1f}) gap={gap_m*1000:.2f}mm"
                      f"(허용{ENGAGE_GAP_M*1000:.1f}) → ok={engaged_ok}")

                # 너트를 볼트축 위 정확한 착좌 pose 로 스냅(kinematic 이라 반력 없음).
                seat_pos = np.array([float(BOLT_XY[0]), float(BOLT_XY[1]), float(BOLT_TIP_Z)])
                seat_quat = nut_quat.copy()
                nut_xform.set_world_pose(position=seat_pos, orientation=seat_quat)

                phase["seat_pos"] = seat_pos
                phase["seat_quat"] = seat_quat
                phase["seat_ee_pos"] = phase["start_pos"].copy()
                phase["pass_theta_deg"] = 0.0
                phase["pass_idx"] = 0
                phase["nut_visual_deg"] = 0.0
                phase["_prev_measured_deg"] = 0.0   # rotate 진입 시 매번 재초기화(아래 참고)
                phase["name"] = "SCREW"
                phase["screw_sub"] = "rotate"
                print(f"  [PICK_CARRY 완료] EE={np.round(phase['start_pos'],3)} → SCREW 시작")
                return

            action = align_controller.forward(
                picking_position=PICK_POS, placing_position=PLACE_POS,
                current_joint_positions=robot.get_joint_positions(),
                end_effector_offset=EE_OFFSET,
            )
            robot.apply_action(action)

            if ev >= 3:
                if ev == 3:
                    phase["_close_ramp_step"] = phase.get("_close_ramp_step", 0) + 1
                    ramp_frac = min(phase["_close_ramp_step"] / GRIP_CLOSE_RAMP_STEPS, 1.0)
                    grip_target = ramp_frac * GRIP_CLOSE_POSITION
                else:
                    ramp_frac = 1.0
                    grip_target = GRIP_CLOSE_POSITION
                grip_action = ArticulationAction(
                    joint_positions=np.array([grip_target, grip_target]),
                    joint_indices=np.array(gripper_dof_indices),
                )
                robot.apply_action(grip_action)

                if ramp_frac >= 1.0 and not phase.get("_hold_ramp_done"):
                    phase["_hold_ramp_step"] = phase.get("_hold_ramp_step", 0) + 1
                    hf = min(phase["_hold_ramp_step"] / GRIPPER_HOLD_RAMP_STEPS, 1.0)
                    cur_stiff = GRIPPER_DRIVE_STIFFNESS + hf * (GRIPPER_HOLD_STIFFNESS - GRIPPER_DRIVE_STIFFNESS)
                    cur_damp = GRIPPER_DRIVE_DAMPING + hf * (GRIPPER_HOLD_DAMPING - GRIPPER_DRIVE_DAMPING)
                    set_gripper_drives(ROBOT_PRIM_PATH, cur_stiff, cur_damp, label="꽉잡기램프", verbose=(hf >= 1.0))
                    if hf >= 1.0:
                        phase["_hold_ramp_done"] = True

                # ev>=3(그립 닫기 시작)부터는 매 스텝 "그립점 = EE - EE_OFFSET" 로
                # 너트를 붙인다 — 캡처가 아니라 항상 성립하는 관계식이라 그립이
                # 덜 닫힌 중간 프레임에도 안전하다. ramp_frac(손가락 닫힘 진행률)을
                # 그대로 blend 로 넘겨 "원래 자리→그립점" 이동도 손가락 닫힘과
                # 같은 리듬으로 진행되게 한다(순간 텔레포트 방지).
                if not phase.get("_glue_started"):
                    phase["_glue_started"] = True
                    print(f"  [GRASP] pose-glue 시작 (ev={ev}, step={phase['_step']})")
                glue_nut_to_ee(ramp_frac)

        elif phase["name"] == "SCREW":
            phase["_screw_step"] = phase.get("_screw_step", 0) + 1
            sub = phase.get("screw_sub", "rotate")

            if sub == "rotate":
                # ⚠ 실측(2026-07-22, GUI 2회 확인): 너트/손목을 각각 독립적으로
                #   "우리가 계산한 목표각"으로 구동했더니 SCREW_DIRECTION 부호를
                #   뒤집어도 그리퍼-너트 회전 방향 불일치가 그대로였다 — 우리
                #   목표식의 부호가 RMPFlow IK 가 실제로 만들어내는 시각적 회전
                #   감각과 일치한다는 보장이 없었다는 뜻. 그래서 너트 자세는
                #   "계산"이 아니라 "측정"으로 바꾼다: 매 스텝 그리퍼가 실제로
                #   도달한 자세에서 직선(월드 Z) 회전량을 그대로 뽑아 누적한다
                #   (nut_visual_deg) — 우리 목표식 부호가 뭐든 항상 실제 그리퍼
                #   움직임과 시각적으로 일치할 수밖에 없다. 깊이(depth_m)는
                #   그리퍼 실측과 무관하게 우리가 직접 램프한 값(pass_theta_deg)
                #   그대로 써서 체결 진행 자체는 안정적으로 유지한다.
                ee_now_pos, ee_now_quat = robot.end_effector.get_world_pose()
                measured_deg = measured_yaw_delta_deg(phase["start_quat"], np.asarray(ee_now_quat))
                phase["nut_visual_deg"] += measured_deg - phase["_prev_measured_deg"]
                phase["_prev_measured_deg"] = measured_deg

                phase["pass_theta_deg"] = min(
                    phase["pass_theta_deg"] + SCREW_OMEGA_DEG_S * PHYSICS_DT, SCREW_TURNS_DEG)
                pass_done = phase["pass_theta_deg"] >= SCREW_TURNS_DEG

                total_deg = phase["pass_idx"] * SCREW_TURNS_DEG + phase["pass_theta_deg"]
                depth_m = min((total_deg / 360.0) * NUT_PITCH_M, ENGAGE_LEN)

                # 너트: 볼트축 xy 고정, z 는 깊이만큼 하강, 자세는 그리퍼 실측 회전 그대로.
                nut_pos = phase["seat_pos"].copy()
                nut_pos[2] = phase["seat_pos"][2] - depth_m
                nut_quat = yaw_rotated_quat(phase["seat_quat"], phase["nut_visual_deg"])
                nut_xform.set_world_pose(position=nut_pos, orientation=nut_quat)

                target_pos = phase["seat_ee_pos"].copy()
                target_pos[2] = phase["seat_ee_pos"][2] - depth_m
                target_quat = yaw_rotated_quat(phase["start_quat"], SCREW_DIRECTION * phase["pass_theta_deg"])
                action = screw_controller.forward(
                    target_end_effector_position=target_pos,
                    target_end_effector_orientation=target_quat,
                )
                robot.apply_action(action)
                apply_grip_hold()

                if phase["_screw_step"] % 30 == 0:
                    print(f"  [DBG-SCREW] pass={phase['pass_idx']} theta={phase['pass_theta_deg']:.0f}"
                          f" nut_visual={phase['nut_visual_deg']:.0f} measured={measured_deg:.0f}"
                          f" depth={depth_m*1000:.2f}/{ENGAGE_LEN*1000:.1f}mm")

                if pass_done:
                    if depth_m >= ENGAGE_LEN or phase["pass_idx"] >= REGRASP_CYCLES:
                        phase["name"] = "SETTLE"
                        phase["settle_steps"] = 0
                        print(f"  [SCREW 전체 완료] 총 {phase['pass_idx']+1}패스,"
                              f" 체결깊이={depth_m*1000:.2f}mm(목표 {ENGAGE_LEN*1000:.1f}mm)")
                    else:
                        phase["pass_end_pos"] = target_pos.copy()
                        phase["pass_end_depth"] = depth_m
                        phase["screw_sub"] = "release"
                        phase["_release_step"] = 0
                        print(f"  [SCREW pass {phase['pass_idx']} 완료] 깊이={depth_m*1000:.2f}mm"
                              f" → 그리퍼 릴리즈 → 언와인드 시작")
                return

            if sub == "release":
                phase["_release_step"] = phase.get("_release_step", 0) + 1
                rf = min(phase["_release_step"] / GRIP_CLOSE_RAMP_STEPS, 1.0)
                release_target = (1.0 - rf) * GRIP_CLOSE_POSITION
                grip_action = ArticulationAction(
                    joint_positions=np.array([release_target, release_target]),
                    joint_indices=np.array(gripper_dof_indices),
                )
                robot.apply_action(grip_action)
                hold_quat = yaw_rotated_quat(phase["start_quat"], SCREW_DIRECTION * SCREW_TURNS_DEG)
                action = screw_controller.forward(
                    target_end_effector_position=phase["pass_end_pos"],
                    target_end_effector_orientation=hold_quat,
                )
                robot.apply_action(action)
                # 너트는 게이팅(고정) — pose 갱신 없음.
                if rf >= 1.0:
                    phase["screw_sub"] = "unwind"
                    phase["wrist_unwind_deg"] = SCREW_TURNS_DEG
                return

            if sub == "unwind":
                phase["wrist_unwind_deg"] = max(
                    phase["wrist_unwind_deg"] - SCREW_OMEGA_DEG_S * PHYSICS_DT, 0.0)
                unwind_done = phase["wrist_unwind_deg"] <= 0.0
                target_quat = yaw_rotated_quat(phase["start_quat"], SCREW_DIRECTION * phase["wrist_unwind_deg"])
                action = screw_controller.forward(
                    target_end_effector_position=phase["pass_end_pos"],
                    target_end_effector_orientation=target_quat,
                )
                robot.apply_action(action)
                grip_action = ArticulationAction(
                    joint_positions=np.array([0.0, 0.0]),
                    joint_indices=np.array(gripper_dof_indices),
                )
                robot.apply_action(grip_action)
                # 너트는 계속 게이팅(고정).
                if unwind_done:
                    phase["screw_sub"] = "regrasp"
                    phase["_regrasp_step"] = 0
                    print(f"  [SCREW pass {phase['pass_idx']} 언와인드 완료] 재파지 시작")
                return

            if sub == "regrasp":
                phase["_regrasp_step"] = phase.get("_regrasp_step", 0) + 1
                rf = min(phase["_regrasp_step"] / GRIP_CLOSE_RAMP_STEPS, 1.0)
                grip_target = rf * GRIP_CLOSE_POSITION
                grip_action = ArticulationAction(
                    joint_positions=np.array([grip_target, grip_target]),
                    joint_indices=np.array(gripper_dof_indices),
                )
                robot.apply_action(grip_action)
                action = screw_controller.forward(
                    target_end_effector_position=phase["pass_end_pos"],
                    target_end_effector_orientation=phase["start_quat"],
                )
                robot.apply_action(action)
                if rf >= 1.0:
                    phase["pass_idx"] += 1
                    phase["pass_theta_deg"] = 0.0
                    phase["seat_ee_pos"] = phase["pass_end_pos"].copy()
                    phase["screw_sub"] = "rotate"
                    # 다음 rotate 진입 시 델타 계산의 기준점을 "지금 실제 측정값"
                    # 으로 다시 잡는다 — 안 하면 이전 패스 끝(≈SCREW_TURNS_DEG)의
                    # 낡은 값과 비교돼 재회전 첫 스텝에서 nut_visual_deg 가 잘못
                    # 크게 튄다.
                    ee_now_pos, ee_now_quat = robot.end_effector.get_world_pose()
                    phase["_prev_measured_deg"] = measured_yaw_delta_deg(
                        phase["start_quat"], np.asarray(ee_now_quat))
                    print(f"  [SCREW pass {phase['pass_idx']} 재파지 완료] 재회전 시작")
                return

        elif phase["name"] == "SETTLE":
            apply_grip_hold()
            phase["settle_steps"] = phase.get("settle_steps", 0) + 1
            if phase["settle_steps"] >= 20:
                phase["name"] = "JUDGE"

        elif phase["name"] == "JUDGE" and not phase["reported"]:
            pos, quat = nut_xform.get_world_pose()
            pos = np.asarray(pos); quat = np.asarray(quat)
            depth_mm = (float(phase["seat_pos"][2]) - float(pos[2])) * 1000.0
            xy_err = float(np.linalg.norm(pos[:2] - BOLT_XY))
            tilt = axis_tilt_deg(quat)
            success = (depth_mm >= 0.6 * ENGAGE_LEN * 1000.0) and (xy_err < 0.005) and (tilt < 5.0)
            print("\n" + "=" * 60)
            print(f"[결과] 너트 최종 위치 = {np.round(pos, 4)}")
            print(f"       패스 수 = {phase['pass_idx']+1} (상한 {1+REGRASP_CYCLES})")
            print(f"       체결 깊이 = {depth_mm:.2f}mm / 목표 {ENGAGE_LEN*1000:.1f}mm"
                  f" ({100.0*depth_mm/(ENGAGE_LEN*1000.0):.0f}%)")
            print(f"       볼트 xy 오차 = {xy_err*1000:.1f}mm,  기울기 = {tilt:.1f}도"
                  f" (kinematic 이라 항상 0에 가까워야 정상)")
            print(f"       체결 성공 = {success}")
            print("=" * 60 + "\n")
            phase["reported"] = True
            eep, _ = robot.end_effector.get_world_pose()
            phase["home_base_pos"] = np.asarray(eep).copy()
            phase["name"] = "HOME"

        elif phase["name"] == "HOME":
            phase["_home_step"] = phase.get("_home_step", 0) + 1
            if not phase.get("_home_release_done"):
                rf = min(phase["_home_step"] / GRIP_CLOSE_RAMP_STEPS, 1.0)
                release_target = (1.0 - rf) * GRIP_CLOSE_POSITION
                grip_action = ArticulationAction(
                    joint_positions=np.array([release_target, release_target]),
                    joint_indices=np.array(gripper_dof_indices),
                )
                robot.apply_action(grip_action)
                action = screw_controller.forward(
                    target_end_effector_position=phase["home_base_pos"],
                    target_end_effector_orientation=phase["start_quat"],
                )
                robot.apply_action(action)
                if rf >= 1.0:
                    phase["_home_release_done"] = True
                    phase["_home_step"] = 0
                return
            target_pos = phase["home_base_pos"].copy()
            target_pos[2] = HOME_LIFT_Z
            action = screw_controller.forward(
                target_end_effector_position=target_pos,
                target_end_effector_orientation=phase["start_quat"],
            )
            robot.apply_action(action)
            if phase["_home_step"] >= 150 and not phase.get("_home_reported"):
                phase["_home_reported"] = True
                print("  [HOME] 후퇴 완료 — 실험 종료")

    # ── 실행: GUI(Play 엣지) / 헤드리스(자동) ──────────────────────────────
    if _HEADLESS:
        print("\n[6] 헤드리스 자동 실행\n")
        reset_cycle()
        for _ in range(MAX_HEADLESS_STEPS):
            step_cycle()
            world.step(render=False)
            if phase.get("_home_reported"):
                break
        simulation_app.close()
        return

    print("\n[6] 준비 완료 — 뷰포트에서 Play 를 누르면 파지+체결 실험을 실행합니다.")
    print("     (Stop 후 다시 Play 하면 처음부터 재실행)\n")
    was_playing = False
    while simulation_app.is_running():
        world.step(render=True)
        playing = world.is_playing()

        if playing and not was_playing:
            print("[Play] 파지+체결 실험 시작")
            reset_cycle()

        if playing:
            step_cycle()

        was_playing = playing

    simulation_app.close()


if __name__ == "__main__":
    main()

"""
7_connector_insertion_real.py ─ 실제 Molex Mega-Fit STEP 커넥터로 peg-in-hole (v3)

부품(convert_step.py 로 STEP→USD 변환된 실물):
  - 소켓 = Receptacle Housing 200456  (connector_assets/2004563216.usd) : 고정, 캐비티 +Z(위)
  - peg  = Plug Free-hang 213815      (connector_assets/2138150106.usd) : 로봇이 잡아 삽입

물리: 리셉터클=정적 삼각메시 콜라이더(오목 캐비티 정확), 플러그=동적 convexDecomposition.
모션/재시작/성공판정은 6_connector_insertion.py(v2)의 검증된 구조 재사용
  (PickPlaceController + Play 엣지 루프).

⚠ 아래 [B] 배치/방향/파지 상수는 렌더 실측 기반 "최선 추정"값 — Play 로 확인 후 미세 튜닝 필요.

실행:
  /home/rokey/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh \
      /home/rokey/cobot3_ws/isaacpjt/M0609/7_connector_insertion_real.py
"""

import os
from isaacsim import SimulationApp

# SEVEN_HEADLESS=1 → 헤드리스, SEVEN_SELFTEST=1 → 자동 Play 후 파지-안정 진단 로그
_HEADLESS = os.environ.get("SEVEN_HEADLESS") == "1"
_SELFTEST = os.environ.get("SEVEN_SELFTEST") == "1"
simulation_app = SimulationApp({"headless": _HEADLESS or _SELFTEST})

import sys
from pathlib import Path
import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.prims import SingleGeometryPrim, SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "rmpflow"))
from m0609_pick_place_controller import PickPlaceController  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
#  [A] 로봇 상수 (6_connector_insertion.py 와 동일)
# ══════════════════════════════════════════════════════════════════════════
ROBOT_USD_PATH   = str(_THIS_DIR / "Collected_m0609_camera/m0609_camera.usd")
ROBOT_URDF_PATH  = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
ROBOT_DESC_PATH  = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
RMPFLOW_CFG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")
ROBOT_PRIM_PATH  = "/World/m0609"
EE_LINK_NAME     = "link_6"
GRIPPER_JOINTS   = ["finger_joint", "right_inner_knuckle_joint"]
FINGER_LINKS     = ["left_inner_finger", "right_inner_finger"]
# 팔 드라이브: 4/5_pick_place 와 동일한 강한 추종(하강·파지에 필요). 접촉 안정은 아래
#   convexHull + 솔버반복 + 속도상한 + 그리퍼 힘제한이 담당(팔을 약하게 하면 못 내려감).
DRIVE_STIFFNESS, DRIVE_DAMPING, DRIVE_MAX_FORCE = 1.0e8, 1.0e6, 1.0e8
# 파지 방식: 얇은 강체를 스티프한 손가락으로 "짜내면" 접촉 솔버가 폭발(휘날림·비산)한다.
#   → 손가락↔플러그 충돌을 필터링하고, 파지 순간 플러그를 그리퍼에 fixed joint(용접)로
#     단단히 고정한다(=firm control). 그립 close 값은 시각적 연출용(0.72 로 손가락이 플러그를 감쌈).
GRIP_OPEN, GRIP_CLOSE, GRIP_DELTA = [0.0, 0.0], [0.72, 0.72], [-0.5, -0.5]
GRASP_JOINT = "/World/grasp_joint"      # 파지 시 활성화되는 용접 조인트
# ⚠ 시도했다가 되돌림(2026-07-19): 파지 조인트를 D6(스프링 드라이브)로 바꿔 삽입 구간에서만
#   순응(compliant)하게 풀어보았으나, 강성을 어떻게 조정해도(약하게/세게, 접촉직전만 전환 등)
#   전부 완전 실패(21~30mm 이탈, 90° 전복)로 악화됨 — FixedJoint(강체 weld) 가 유일하게
#   안정적으로 동작(2.1mm 삽입/12.4° 기울기). 원인은 아티큘레이션 링크(EE)↔동적 바디(플러그)
#   조인트에 스프링 드라이브를 걸면 접촉 시 자세를 못 붙잡고 그대로 넘어가는 것으로 추정 —
#   XY 서치(spiral) 등 grasp 조인트를 안 건드리는 방식으로 다시 시도 필요.
#   event4(lift) 는 절대 빠르게 하지 않는다 — 타이트한 지그에서 빼내는 구간이라 빠르게 하면
#   플러그가 벽에 긁히며 충격 폭주(휘날림)가 재발한다(실측: ev4 0.02→0.03 만으로 jv 93 rad/s
#   재발). 대신 파지 "전" 순수 대기(ev2, 관성 정착만 하는 구간)를 크게 줄이고, 접촉이 없는
#   순수 이동 구간(ev0,1,5,6,8)만 소폭 가속해 체감 속도를 개선한다.
EVENTS_DT = [0.011, 0.006, 0.05, 0.1, 0.02, 0.013, 0.003, 1.0, 0.011, 0.08]

# 물리 안정화: 링크/부품 속도 상한(휘날림·비산 직접 차단), 아티큘레이션 솔버 반복수
MAX_LIN_VEL = 3.0                       # m/s
MAX_ANG_VEL = 20.0                      # rad/s
ART_POS_ITERS, ART_VEL_ITERS = 32, 4
# ⚠ RMPFlow 컨트롤러가 내부적으로 1/60 로 적분 → World dt 도 1/60 로 맞춰야 모션 타이밍이 맞음
#   (1/240 로 하면 팔이 1/4 속도라 파지 지점까지 못 내려감). 접촉 안정은 속도상한/솔버/유한힘이 담당.
PHYSICS_DT = 1.0 / 60.0


# ══════════════════════════════════════════════════════════════════════════
#  [B] 실물 커넥터 배치/방향/파지  ── ⚠ Play 확인 후 튜닝 대상
# ══════════════════════════════════════════════════════════════════════════
RECEP_USD = str(_THIS_DIR / "connector_assets/2004563216.usd")   # 200456 리셉터클(소켓)
PLUG_USD  = str(_THIS_DIR / "connector_assets/2138150106.usd")   # 213815 플러그(peg)
PART_SCALE = 1.0                       # 실측 ~37mm, 실제 스케일 유지 (그리퍼 85mm 파지 가능)
# ⚠ 시도했다가 되돌림(2026-07-19): 리셉터클만 8% 확대해 캐비티에 여유(clearance)를 줘봤으나
#   삽입깊이·기울기 개선 거의 없음(1.3mm/13.9° — 기존 2.1mm/12.4°와 사실상 동일). 접촉 시
#   기울기(~12°)로 인한 횡방향 오차가 이 정도 여유보다 커서(12mm 하강 시 tan(12°)*12mm≈2.5mm)
#   소폭 확대로는 못 이김 — 실제 치수 유지가 이 파일의 목적이기도 해 되돌림.
RECEP_SCALE = 1.0

# 소켓: 캐비티가 +Z(위)로 열려 있음(렌더 확인). 바닥에 안착, 고정.
RECEP_XY    = np.array([0.45, -0.21])  # 원래 -0.20 에서 -Y로 5mm 이동(삽입 정렬 미세조정)
RECEP_Z     = 0.016                     # bbox min_z=-0.0156 → 바닥에 닿게
RECEP_EULER = (0.0, 0.0, 0.0)           # 그대로 두면 캐비티 +Z
RECEP_MATE_TOP_Z = RECEP_Z + 0.011      # 캐비티 입구(대략 상단) 월드 z

# 플러그: mating면(+Y, 핀 6개)이 아래(-Z)를 향하도록 X축 -90° 회전.
#   회전 후: x폭37 · y두께10.6 · z높이32, mating(핀) 아래·wire end 위.
#   실측(Play): 마주보는 면은 맞으나 핀 배열(키)이 소켓과 180° 어긋남 → 하강 후(월드 Z축
#   기준, 로컬 X회전과 무관하게 바깥쪽에서 적용) Z축 180° 추가 회전으로 키 정렬.
PLUG_XY    = np.array([0.45, 0.20])
PLUG_Z     = 0.003                      # 지그(nest) 안, 바닥 근처에 안착
PLUG_EULER = (-90.0, 0.0, 180.0)        # +Y(mating) → -Z(down), 키 정렬 180° 보정
PLUG_MASS  = 0.06                       # 과도한 그립에 튕겨나가지 않게 상향

# 지그(nest): 플러그를 세워 고정(넘어짐 방지). 플러그 하단이 웰에 들어가고 상단이 돌출.
#   플러그 실제 bbox 로부터 자동 크기/중심 산출(런타임). 벽만 4개(바닥 없음, 그라운드가 바닥).
NEST_MARGIN_X = 0.006                   # 내부폭 여유(플러그 x폭 + 이만큼) — 원래값 유지
                                         #   (실측: 여유를 줄여도 기울기 개선 효과는 작고,
                                         #    오히려 들어올릴 때 벽과 마찰이 커져 충격 유발됨).
NEST_MARGIN_Y = 0.004                   # 내부폭 여유(플러그 y두께 + 이만큼) — 원래값 유지
NEST_WALL     = 0.006
NEST_HEIGHT   = 0.012                   # 지지 구간을 넓혀 지렛대 효과(기울어짐) 축소.
                                         #   ⚠ 실측: 벽을 너무 높이거나(0.018) 파지점 여유를
                                         #   좁히면(GRASP_BELOW_TOP 근접) 들어올릴 때 플러그가
                                         #   벽에 긁혀 충격이 폭주함(jv 급등) — 아래 GRASP_BELOW_TOP
                                         #   과 함께 벽 상단 위 여유(clearance)를 충분히 확보.

# 파지/삽입 (런타임 bbox 로 자동 산출 — 하드코딩 오프셋 제거).
EE_OFFSET       = np.array([0.0, 0.0, 0.20])
GRASP_BELOW_TOP = 0.006                 # 파지점을 지그 벽 상단보다 충분히 위(≈14mm 여유)로 잡아
                                         #   들어올릴 때 플러그가 벽과 긁히지 않게 함(기존
                                         #   0.008보단 낮춰, 기울기가 더 작은 지점을 잡음).
INSERT_DEPTH    = 0.012                 # 캐비티 입구보다 이만큼 더 내림(기존 0.005→0.012:
                                         #   SELFTEST 실측 — 5mm 는 캐비티 가장자리에 살짝 걸치는
                                         #   깊이라 접촉 시 옆으로 미끄러져 90°로 넘어짐(소켓거리
                                         #   21mm). 더 깊이 내려 확실히 벽 안쪽으로 걸리게 함).
# 리셉터클 캐비티 실제 중심 vs bbox 전체 중심(rctr) 오차 — 메시 정점을 상단면 기준
# 밴드로 슬라이싱해 실측(probe_cavity.py, 2026-07-19): 캐비티 중심이 bbox 중심보다
# -Y 로 약 2mm 치우쳐 있음(하우징 형상이 Y축 비대칭). PLACE 목표에 보정.
RECEP_CAVITY_XY_OFFSET = np.array([-0.0001, -0.0020])

# ⚠ 나선 탐색(spiral search) 시도했다가 되돌림(2026-07-19): event6(삽입 하강)에서 목표 XY 를
#   나선으로 흔들며 구멍을 찾게 해봤으나, 팔이 매우 뻣뻣해서(1e8) 이미 입구 아래로 박힌 상태에서
#   흔들면 그대로 긁으며 지렛대처럼 튕겨나감 — 반경3mm/4회전(71mm,90°), 반경1.5mm/2회전+게이팅
#   (27mm,97.7°) 둘 다 기존 순수 강체하강(2mm,12.4°)보다 악화. 파지부와 마찬가지로, 팔이
#   완전히 강체·비컴플라이언트인 한 접촉 중 목표를 흔드는 어떤 방식도 위험함 — 실제 힘 피드백
#   기반 탐색이나 팔 강성 자체를 낮추는 근본적 재설계가 필요.

PEG_TYPE_ID, SOCKET_EXPECT_TYPE = 1, 1


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
    # 4/5_pick_place 와 동일: 모든 관절 균일 드라이브. 그리퍼 4절링크 균형을 깨면(비대칭 힘)
    #   닫힘 시 링크가 서로 싸워 폭발하므로, 힘을 관절별로 다르게 주지 않는다.
    stage = omni.usd.get_context().get_stage()
    n = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        for dt in ("angular", "linear"):
            drive = UsdPhysics.DriveAPI.Get(prim, dt)
            if drive:
                drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
                drive.GetDampingAttr().Set(DRIVE_DAMPING)
                drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)
                n += 1
    print(f"  [OK] 드라이브 {n}개 균일 설정 (stiffness={DRIVE_STIFFNESS:.0e})")


def solver_iters_only(root_path):
    """접촉 안정을 위해 아티큘레이션 솔버 반복수만 상향(그 외 변경 없음)."""
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
    # 손가락이 플러그 없이 완전히 닫힐 때 서로 자기충돌해 튀는 것 방지
    ap.CreateEnabledSelfCollisionsAttr(False)
    print(f"  [OK] 솔버 반복수 상향: pos={ART_POS_ITERS} vel={ART_VEL_ITERS}, selfCol=off")


def filter_pair(prim_path_a, prim_path_b):
    """두 prim(및 하위) 간 충돌 필터 — 손가락↔플러그 짜냄 폭발 방지."""
    stage = omni.usd.get_context().get_stage()
    a = stage.GetPrimAtPath(prim_path_a)
    UsdPhysics.FilteredPairsAPI.Apply(a).CreateFilteredPairsRel().AddTarget(prim_path_b)


def make_grasp_joint(body0_path, body1_path):
    """비활성 fixed joint 를 미리 생성(파지 순간 활성화)."""
    stage = omni.usd.get_context().get_stage()
    j = UsdPhysics.FixedJoint.Define(stage, GRASP_JOINT)
    j.CreateBody0Rel().SetTargets([body0_path])
    j.CreateBody1Rel().SetTargets([body1_path])
    j.CreateJointEnabledAttr(False)


def _quatf(q):
    return Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def canonical_plug_quat():
    """PLUG_EULER 로부터 산출한 '곧게 선' 기준 쿼터니언(w,x,y,z) — 런타임 자세와 무관한 고정값.
       파지(용접) 순간 이 값으로 자세를 스냅해 이후 이송·삽입 내내 플러그가 곧게 유지되게 한다.
    """
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    return euler_angles_to_quat(np.radians(np.array(PLUG_EULER, dtype=float)))


def quat_angle_deg(qa, qb):
    """두 쿼터니언(w,x,y,z) 사이의 회전각(도) — 기울기(자세오차) 측정용."""
    qa = np.asarray(qa, dtype=float); qb = np.asarray(qb, dtype=float)
    dot = float(np.clip(abs(np.dot(qa, qb)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def engage_grasp(p0, q0, p1, q1):
    """현재 상대 포즈로 용접 조인트 활성화(플러그를 그리퍼에 단단히 고정).
       q1 은 호출부에서 측정치 대신 canonical_plug_quat() 을 넘겨 자세를 정렬한다(위치 p1 은 측정치 유지).
    """
    stage = omni.usd.get_context().get_stage()
    m0 = Gf.Matrix4d(); m0.SetRotateOnly(Gf.Quatd(_quatf(q0))); m0.SetTranslateOnly(Gf.Vec3d(float(p0[0]), float(p0[1]), float(p0[2])))
    m1 = Gf.Matrix4d(); m1.SetRotateOnly(Gf.Quatd(_quatf(q1))); m1.SetTranslateOnly(Gf.Vec3d(float(p1[0]), float(p1[1]), float(p1[2])))
    rel = m1 * m0.GetInverse()
    t = rel.ExtractTranslation(); r = rel.ExtractRotationQuat()
    j = UsdPhysics.FixedJoint.Get(stage, GRASP_JOINT)
    j.CreateLocalPos0Attr(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
    j.CreateLocalRot0Attr(Gf.Quatf(float(r.GetReal()), *[float(x) for x in r.GetImaginary()]))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.GetJointEnabledAttr().Set(True)


def release_grasp():
    stage = omni.usd.get_context().get_stage()
    j = UsdPhysics.FixedJoint.Get(stage, GRASP_JOINT)
    if j:
        j.GetJointEnabledAttr().Set(False)


def reference_part(root_path, usd_path, pos, euler_deg, scale):
    """부모 Xform 패턴: 깨끗한 부모(root_path)를 만들고 reference는 자식(/geo)에.
       변환 USD의 defaultPrim 이 instanceable 이라, reference 보유 prim 에 직접 transform 을
       주면 먹지 않는다 → 내가 소유한 부모 Xform 에 transform 을 걸어 확실히 적용.
    """
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, root_path)                 # 깨끗한 부모 (reference 없음)
    add_reference_to_stage(usd_path, root_path + "/geo")   # reference 는 자식에
    xf = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(root_path))
    xf.SetTranslate(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    xf.SetRotate(Gf.Vec3f(*[float(e) for e in euler_deg]))
    xf.SetScale(Gf.Vec3f(float(scale), float(scale), float(scale)))


def _meshes_under(root_path):
    """instanceable(중첩 포함) 해제 후 하위 Mesh prim 목록."""
    stage = omni.usd.get_context().get_stage()
    for _ in range(6):                       # 중첩 instance 를 반복적으로 해제
        changed = False
        for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
            if p.IsInstanceable():
                p.SetInstanceable(False)
                changed = True
        if not changed:
            break
    return [p for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)) if p.GetTypeName() == "Mesh"]


def apply_physics(root_path, dynamic, approximation, mass=None, material=None):
    """임포트 메시에 물리 부여.
       dynamic=True  → 루트에 RigidBody/Mass, 메시는 convexDecomposition 등 동적 콜라이더.
       dynamic=False → 정적 콜라이더(approximation='none' = 정확한 삼각메시, 오목 캐비티 OK).
    """
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    meshes = _meshes_under(root_path)
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(root)
        m = UsdPhysics.MassAPI.Apply(root)
        if mass is not None:
            m.CreateMassAttr(float(mass))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)   # 속도 상한(비산 방지)
        rb.CreateMaxLinearVelocityAttr(MAX_LIN_VEL)
        rb.CreateMaxAngularVelocityAttr(MAX_ANG_VEL)
    for mesh in meshes:
        UsdPhysics.CollisionAPI.Apply(mesh)
        mca = UsdPhysics.MeshCollisionAPI.Apply(mesh)
        mca.CreateApproximationAttr().Set(approximation)
        PhysxSchema.PhysxCollisionAPI.Apply(mesh)
        if material is not None:
            SingleGeometryPrim(prim_path=str(mesh.GetPath()),
                               name=f"{mesh.GetName()}_g").apply_physics_material(material)
    print(f"  [OK] 물리 부여 {root_path}: dynamic={dynamic} approx={approximation} meshes={len(meshes)}")


def initialize_robot(robot, world):
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )


def world_bbox(prim_path):
    """prim 의 월드 AABB (min, max, center). 물리 이전(authored) 포즈 기준."""
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                             [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    mn = np.array([mn[0], mn[1], mn[2]])
    mx = np.array([mx[0], mx[1], mx[2]])
    return mn, mx, (mn + mx) / 2.0


def build_nest(world, center_xy, inner_x, inner_y, material=None):
    """플러그를 세워 고정하는 사각 웰(벽 4개, 그라운드가 바닥). 정적."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    ax, ay, t, h = inner_x / 2, inner_y / 2, NEST_WALL, NEST_HEIGHT
    cz = h / 2
    ox, oy = 2 * (ax + t), 2 * (ay + t)
    for nm, pos, scale in [
        ("xp", [cx + ax + t / 2, cy, cz], [t, oy, h]),
        ("xn", [cx - ax - t / 2, cy, cz], [t, oy, h]),
        ("yp", [cx, cy + ay + t / 2, cz], [ox, t, h]),
        ("yn", [cx, cy - ay - t / 2, cz], [ox, t, h]),
    ]:
        c = world.scene.add(FixedCuboid(
            prim_path=f"/World/nest/{nm}", name=f"nest_{nm}",
            position=np.array(pos), scale=np.array(scale),
            color=np.array([0.35, 0.35, 0.4]),
        ))
        if material is not None:
            c.apply_physics_material(material)
        # 지그↔로봇 충돌 필터: 그리퍼가 지그 벽에 부딪혀 팔이 폭발하는 것 방지
        #   (지그는 플러그만 붙잡고, 로봇에는 투명)
        wall = omni.usd.get_context().get_stage().GetPrimAtPath(f"/World/nest/{nm}")
        UsdPhysics.FilteredPairsAPI.Apply(wall).CreateFilteredPairsRel().AddTarget(ROBOT_PRIM_PATH)
    print(f"  [OK] 지그(nest) @ ({cx:.3f},{cy:.3f}) inner {inner_x*1000:.0f}x{inner_y*1000:.0f}mm h{h*1000:.0f}mm (로봇충돌 필터)")


def check_correct_part(part_type_id, socket_expected_type):
    return part_type_id == socket_expected_type


# ══════════════════════════════════════════════════════════════════════════
#  [D] 메인
# ══════════════════════════════════════════════════════════════════════════
def main():
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
    # 솔버 반복수만 강화(접촉 안정). 속도상한/자기충돌 변경은 아티큘레이션을 되레 불안정하게
    #   만들 수 있어 제외 — 4/5_pick_place 의 검증된 물리에 최소 변경만.
    solver_iters_only(ROBOT_PRIM_PATH)

    print("\n[3] 실물 커넥터 로드")
    part_mat = PhysicsMaterial(
        prim_path="/World/Physics_Materials/part_mat",
        static_friction=1.6, dynamic_friction=1.2, restitution=0.0,
    )
    # 소켓(리셉터클): 고정, 정확 삼각메시(오목 캐비티). RECEP_SCALE 로 캐비티에 여유(clearance) 부여.
    reference_part("/World/receptacle", RECEP_USD,
                   [RECEP_XY[0], RECEP_XY[1], RECEP_Z], RECEP_EULER, RECEP_SCALE)
    apply_physics("/World/receptacle", dynamic=False, approximation="none", material=part_mat)

    # peg(플러그): 동적 강체, convexHull(단일 볼록 → 스파이크 접촉 제거로 안정)
    reference_part("/World/connector", PLUG_USD,
                   [PLUG_XY[0], PLUG_XY[1], PLUG_Z], PLUG_EULER, PART_SCALE)
    apply_physics("/World/connector", dynamic=True, approximation="convexHull",
                  mass=PLUG_MASS, material=part_mat)
    plug = SingleRigidPrim("/World/connector", name="plug")
    # 손가락↔플러그 충돌 필터(짜냄 폭발 방지). 플러그↔리셉터클/그라운드 충돌은 유지.
    filter_pair("/World/connector", ROBOT_PRIM_PATH)

    # 지그(nest): 플러그 실제 bbox 로 크기/중심 자동 산출 → 플러그를 세워 고정(넘어짐 방지)
    pmn, pmx, pctr = world_bbox("/World/connector")
    build_nest(world, pctr[:2],
               inner_x=(pmx[0] - pmn[0]) + NEST_MARGIN_X,
               inner_y=(pmx[1] - pmn[1]) + NEST_MARGIN_Y, material=part_mat)

    # 자기보정 PICK/PLACE (authored bbox 기준 — 원점↔형상 오프셋 자동 보정, 치우침 해결)
    rmn, rmx, rctr = world_bbox("/World/receptacle")
    pick_pos  = np.array([pctr[0], pctr[1], pmx[2] - GRASP_BELOW_TOP])
    grasp_to_mating = float(pick_pos[2] - pmn[2])
    recep_center_xy = rctr[:2].copy() + RECEP_CAVITY_XY_OFFSET
    place_pos = np.array([recep_center_xy[0], recep_center_xy[1], rmx[2] + grasp_to_mating - INSERT_DEPTH])

    print("\n[4] 로봇 등록")
    ee_path = find_prim_path(ROBOT_PRIM_PATH, EE_LINK_NAME)
    if ee_path is None:
        raise RuntimeError(f"'{EE_LINK_NAME}' 링크를 찾을 수 없음")
    gripper = ParallelGripper(
        end_effector_prim_path=ee_path, joint_prim_names=GRIPPER_JOINTS,
        joint_opened_positions=np.array(GRIP_OPEN),
        joint_closed_positions=np.array(GRIP_CLOSE),
        action_deltas=np.array(GRIP_DELTA),
    )
    robot = world.scene.add(SingleManipulator(
        prim_path=ROBOT_PRIM_PATH, name="m0609_robot",
        end_effector_prim_path=ee_path, gripper=gripper,
    ))
    finger_mat = PhysicsMaterial(
        prim_path="/World/Physics_Materials/finger_mat",
        static_friction=1.8, dynamic_friction=1.4, restitution=0.0,
    )
    for ln in FINGER_LINKS:
        lp = find_prim_path(ROBOT_PRIM_PATH, ln)
        if lp:
            SingleGeometryPrim(prim_path=lp, name=f"{ln}_geom").apply_physics_material(finger_mat)
    print(f"  [OK] SingleManipulator, EE={ee_path}")

    make_grasp_joint(ee_path, "/World/connector")   # 파지용 용접 조인트(비활성 상태로 생성)

    world.reset()
    initialize_robot(robot, world)
    for _ in range(30):
        world.step(render=True)

    print("\n[5] PickPlaceController 생성")
    controller = PickPlaceController(
        name="m0609_real_connector_controller",
        gripper=robot.gripper, robot_articulation=robot,
        end_effector_initial_height=0.30, events_dt=EVENTS_DT,
        urdf_path=ROBOT_URDF_PATH, robot_description_path=ROBOT_DESC_PATH,
        rmpflow_config_path=RMPFLOW_CFG_PATH, end_effector_frame_name=EE_LINK_NAME,
    )
    print(f"  [OK] 컨트롤러 준비 — PICK {np.round(pick_pos,3)}  PLACE {np.round(place_pos,3)}")
    if not check_correct_part(PEG_TYPE_ID, SOCKET_EXPECT_TYPE):
        print("  [WRONG_PART] 잘못된 커넥터 — (복구 로직은 후속)")

    # ── 헤드리스 자동 진단(SEVEN_SELFTEST=1): 파지 중 휘날림/비산 여부 로그 후 종료 ──
    if _SELFTEST:
        world.reset()
        initialize_robot(robot, world)
        # 지그 안에서 중력으로 정착할 시간(타워진 지그 벽 안에서 안정화) — reset 직후 몇 프레임 물리만 진행.
        for _ in range(30):
            world.step(render=False)
        controller.reset()
        world.play()
        canon_q = canonical_plug_quat()
        plug0 = np.asarray(plug.get_world_pose()[0])
        tilt_in_nest = quat_angle_deg(np.asarray(plug.get_world_pose()[1]), canon_q)
        max_jv, max_z, max_drift, min_ee_z = 0.0, 0.0, 0.0, 9.9
        ee_at_pick, attached = None, False
        spike_info, max_jv_info = None, None
        tilt_pre_grasp, tilt_post_snap, tilt_apex, steps_since_attach = None, None, None, -1
        place_pos_run = place_pos.copy()   # 파지 순간 xy 드리프트로 보정될 이번 run 의 실제 place 목표
        grasp_xy_drift = None
        step = 0
        for step in range(3000):
            ev = controller.get_current_event() if hasattr(controller, "get_current_event") else -1
            # 파지 용접: 그리퍼가 닫히고 팔이 정지한 event3(하강 완료·close hold)에서 고정
            #   → 그 다음 event4(lift)에서 이미 붙어 있어 스냅 충격 없음. open(event7)에서 해제.
            #   q1 자리에 측정치 대신 canon_q(캐노니컬 업라이트)를 넘겨 파지 순간 자세를 정렬.
            if 3 <= ev < 7 and not attached:
                eep_, eeq_ = robot.end_effector.get_world_pose()
                pp_, pq_ = plug.get_world_pose()
                tilt_pre_grasp = quat_angle_deg(np.asarray(pq_), canon_q)
                engage_grasp(eep_, eeq_, pp_, canon_q)
                attached = True
                steps_since_attach = 0
                # 컨트롤러는 pick_pos→place_pos 사이 '가상 목표점'만 보간하고 실제 플러그
                #   위치는 추적하지 않는다. 지그 정착/서보 오차로 실제 파지 위치(pp_)가
                #   nominal pick_pos 와 어긋난 만큼, place 목표도 같은 델타로 보정해 상쇄한다.
                grasp_xy_drift = np.asarray(pp_[:2]) - pick_pos[:2]
                place_pos_run = place_pos.copy()
                place_pos_run[:2] = recep_center_xy - grasp_xy_drift
            elif ev >= 7 and attached:
                release_grasp()
                attached = False
            if attached and steps_since_attach >= 0:
                steps_since_attach += 1
                if steps_since_attach == 10:   # 용접 직후(주변 접촉 전) 자세 스냅 확인용
                    tilt_post_snap = quat_angle_deg(np.asarray(plug.get_world_pose()[1]), canon_q)
                if ev == 5:   # xy 이동(장애물 없는 구간) — 삽입 시도 전 기울기 참고치
                    tilt_apex = quat_angle_deg(np.asarray(plug.get_world_pose()[1]), canon_q)
            if not controller.is_done():
                act = controller.forward(
                    picking_position=pick_pos, placing_position=place_pos_run,
                    current_joint_positions=robot.get_joint_positions(),
                    end_effector_offset=EE_OFFSET)
                robot.apply_action(act)
            world.step(render=False)
            jv = robot.get_joint_velocities()
            if jv is not None:
                mjv = float(np.max(np.abs(np.asarray(jv))))
                if mjv > max_jv:
                    max_jv = mjv
                    max_jv_info = (step, ev)   # 최대 jv 가 어느 event 에서 나오는지(진단용)
                if mjv > 50.0 and spike_info is None:   # 첫 스파이크 맥락 기록
                    ep = np.asarray(robot.end_effector.get_world_pose()[0])
                    pgp = np.asarray(plug.get_world_pose()[0])
                    spike_info = (step, ev, round(float(ep[2]), 3), round(float(pgp[2]), 3), attached)
            eep = np.asarray(robot.end_effector.get_world_pose()[0])
            min_ee_z = min(min_ee_z, float(eep[2]))
            if ev in (1, 2, 3) and ee_at_pick is None:   # 하강~그립 이벤트 근처의 EE 위치
                ee_at_pick = eep.copy()
            pp = np.asarray(plug.get_world_pose()[0])
            max_z = max(max_z, float(pp[2]))
            max_drift = max(max_drift, float(np.linalg.norm(pp[:2] - plug0[:2])))
            if controller.is_done():
                break
        pp, pq_final = plug.get_world_pose()
        pp = np.asarray(pp)
        tilt_final = quat_angle_deg(np.asarray(pq_final), canon_q)
        eep = np.asarray(robot.end_effector.get_world_pose()[0])
        gj = robot.gripper.get_joint_positions()
        sock_err = float(np.linalg.norm(pp[:2] - recep_center_xy))
        print("\n" + "=" * 60)
        print(f"[SELFTEST] steps={step} max|joint_vel|={max_jv:.2f} rad/s @ (step,event)={max_jv_info}")
        _gxy = grasp_xy_drift * 1000 if grasp_xy_drift is not None else None
        print(f"[SELFTEST] 파지시 xy드리프트(mm)={np.round(_gxy,1) if _gxy is not None else None}  "
              f"보정된 PLACE={np.round(place_pos_run,3)}")
        print(f"[SELFTEST] PICK 목표={np.round(pick_pos,3)}  EE(그립근처)={np.round(ee_at_pick,3) if ee_at_pick is not None else None}")
        print(f"[SELFTEST] EE 최저z={min_ee_z*1000:.0f}mm  EE 최종={np.round(eep,3)}")
        print(f"[SELFTEST] gripper 최종관절={np.round(np.asarray(gj),3)} (open=0, close=0.72)")
        print(f"[SELFTEST] plug 최고z={max_z*1000:.0f}mm  최종={np.round(pp,3)}  소켓거리={sock_err*1000:.0f}mm")
        print(f"[SELFTEST] 첫 스파이크(step,event,ee_z,plug_z,attached)={spike_info}")
        _tpg = tilt_pre_grasp if tilt_pre_grasp is not None else -1.0
        _tps = tilt_post_snap if tilt_post_snap is not None else -1.0
        _tap = tilt_apex if tilt_apex is not None else -1.0
        print(f"[SELFTEST] 기울기(도) 지그안={tilt_in_nest:.1f}  파지직전(정렬전측정)={_tpg:.1f}  "
              f"용접직후(10step)={_tps:.1f}  이송중(ev5)={_tap:.1f}  배치시(최종,삽입접촉후)={tilt_final:.1f}")
        # '정렬됨' 은 이번 수정 범위(지그→파지→이송)의 성공 기준: ev5(이송, 장애물 없음) 시점 기울기.
        #   최종(tilt_final)은 소켓 삽입 '접촉' 이후 값 — 완전 착좌는 후속 힘제어 과제라 여기 포함 안 함.
        print(f"[SELFTEST] 휘날림없음={max_jv<25.0}  들어올림={max_z>0.05}  소켓이송={sock_err<0.05}  정렬됨(이송중)={_tap<5.0}")
        print("=" * 60 + "\n")
        simulation_app.close()
        return

    print("\n[6] 준비 완료 — Play 를 누르면 삽입. Stop→Play 재실행.\n")
    canon_q = canonical_plug_quat()
    was_playing, reported, attached = False, False, False
    place_pos_run = place_pos.copy()   # 파지 순간 xy 드리프트로 보정될 이번 run 의 실제 place 목표
    while simulation_app.is_running():
        world.step(render=True)
        playing = world.is_playing()

        if playing and not was_playing:
            print("[Play] 삽입 시퀀스 시작")
            world.reset()
            initialize_robot(robot, world)
            # 지그 안에서 중력으로 정착할 시간(타워진 지그 벽 안에서 안정화)
            for _ in range(30):
                world.step(render=True)
            controller.reset()
            release_grasp()          # 재시작 시 용접 해제
            reported, attached = False, False
            place_pos_run = place_pos.copy()

        if playing and not controller.is_done():
            ev = controller.get_current_event()
            # 파지 용접: event3(정지·close hold)에서 고정, open(event7)에서 해제 → 스냅/휘날림 없음
            #   q1 자리에 측정치 대신 canon_q(캐노니컬 업라이트)를 넘겨 파지 순간 자세를 정렬.
            if 3 <= ev < 7 and not attached:
                eep_, eeq_ = robot.end_effector.get_world_pose()
                pp_, pq_ = plug.get_world_pose()
                engage_grasp(eep_, eeq_, pp_, canon_q)
                attached = True
                # 실제 파지 위치의 nominal pick_pos 대비 xy 오차만큼 place 목표를 보정(상쇄)
                grasp_xy_drift = np.asarray(pp_[:2]) - pick_pos[:2]
                place_pos_run = place_pos.copy()
                place_pos_run[:2] = recep_center_xy - grasp_xy_drift
                print(f"  [파지] xy드리프트={np.round(grasp_xy_drift*1000,1)}mm  보정된 PLACE={np.round(place_pos_run,3)}")
            elif ev >= 7 and attached:
                release_grasp()
                attached = False
            action = controller.forward(
                picking_position=pick_pos, placing_position=place_pos_run,
                current_joint_positions=robot.get_joint_positions(),
                end_effector_offset=EE_OFFSET,
            )
            robot.apply_action(action)
        elif playing and controller.is_done() and not reported:
            for _ in range(20):
                world.step(render=True)
            pos, quat = plug.get_world_pose()
            pos = np.asarray(pos)
            xy_err = float(np.linalg.norm(pos[:2] - recep_center_xy))
            tilt = quat_angle_deg(np.asarray(quat), canon_q)
            print("\n" + "=" * 60)
            print(f"[결과] 플러그 최종 위치 = {np.round(pos, 4)}")
            print(f"       소켓 xy 오차 = {xy_err*1000:.1f}mm,  z = {pos[2]*1000:.1f}mm,  기울기 = {tilt:.1f}도")
            print(f"       (완전 착좌 판정은 실물 공차상 후속 힘제어 필요)")
            print("=" * 60 + "\n")
            reported = True

        was_playing = playing

    simulation_app.close()


if __name__ == "__main__":
    main()

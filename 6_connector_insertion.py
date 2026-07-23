"""
6_connector_insertion.py ─ 전장 커넥터 삽입(peg-in-hole) standalone 베이스라인 (v2)

v1 실행 영상에서 확인된 두 버그를 수정:
  ① 재시작 안 됨 → Play 엣지(is_playing) 감지로 Stop→Play 마다 처음부터 재실행.
  ② 파지 실패   → 직접 IK 폐기, 4/5_pick_place에서 검증된 RMPFlow PickPlaceController 재사용.

삽입 = "place into pocket". 커넥터를 상단 근처에서 잡아 소켓 포켓에 내려놓으면 삽입이 된다
(손가락이 포켓 벽 위로 뜨도록 파지 높이/포켓 깊이를 맞춰둠).

WRONG_PART/오삽입/복구는 check_correct_part() 확장 지점만 남겨둔다.

실행:
  /home/rokey/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh \
      /home/rokey/cobot3_ws/isaacpjt/M0609/6_connector_insertion.py
  → 뷰포트에서 Play 를 누르면 실행. Stop 후 다시 Play 하면 재실행.
"""

# ── 0. Isaac Sim 런치 (다른 isaacsim import보다 먼저) ─────────────────────
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import sys
from pathlib import Path
import numpy as np
import omni.client
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator

_THIS_DIR = Path(__file__).resolve().parent
# rmpflow 인프라 폴더를 sys.path에 등록해야 컨트롤러 내부 import가 동작 (4_pick_place와 동일)
sys.path.insert(0, str(_THIS_DIR / "rmpflow"))
from m0609_pick_place_controller import PickPlaceController  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
#  A. 상수
# ══════════════════════════════════════════════════════════════════════════
# 로봇 자산 — 기존 프로젝트 파일에서 "경로만" 참고
ROBOT_USD_PATH   = str(_THIS_DIR / "Collected_m0609_camera/m0609_camera.usd")
ROBOT_URDF_PATH  = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
ROBOT_DESC_PATH  = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
RMPFLOW_CFG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")
ROBOT_PRIM_PATH  = "/World/m0609"
EE_LINK_NAME     = "link_6"
GRIPPER_JOINTS   = ["finger_joint", "right_inner_knuckle_joint"]
FINGER_LINKS     = ["left_inner_finger", "right_inner_finger"]

# 관절 드라이브 — 위치 명령 추종 보장 (4_pick_place와 동일 사상)
DRIVE_STIFFNESS, DRIVE_DAMPING, DRIVE_MAX_FORCE = 1e8, 1e6, 1e8

# 그리퍼 — 커넥터(지름 36mm)를 확실히 물도록 close 값을 큐브(0.5)보다 상향
GRIP_OPEN   = [0.0, 0.0]
GRIP_CLOSE  = [0.6, 0.6]
GRIP_DELTA  = [-0.5, -0.5]

# Pick&Place 10단계 타이밍 (4_pick_place 검증값)
EVENTS_DT = [0.008, 0.005, 0.02, 0.1, 0.0025, 0.01, 0.0025, 1.0, 0.008, 0.08]

# ── 커넥터(peg) : 절차적 그립 몸체(박스, 실제 물리) + Molex 실제 커넥터 시각 캡 ──
#   Molex 정품은 22×2.7×5mm 납작한 시각 전용 메시(충돌/물리 없음)라 직접 파지 불가.
#   → 신뢰성 있는 절차적 박스가 파지·삽입을 담당하고, Molex 메시는 상단에 외형만 얹는다.
MOLEX_CAP_REL   = "/Isaac/Samples/Rigging/Jetbot/Jetbot_Base/parts/molex_connector_JFb.usd"
CAP_SCALE       = 1.8                                   # 원본 확대(시각 전용)
CAP_MESH_OFFSET = np.array([0.0, -0.00615, -0.0129])   # 원본 메시 중심 오프셋(실측)
PEG_SIZE  = np.array([0.036, 0.036, 0.070])            # 커넥터 하우징 박스 L×W×H
PEG_POS   = np.array([0.45, 0.20, PEG_SIZE[2] / 2.0])  # 바닥에 세워둠 (center z=0.035)
PEG_TYPE_ID = 1                     # WRONG_PART 확장용

# ── 소켓(hole) : 포켓형(바닥판+4벽) ──
SOCKET_XY     = np.array([0.45, -0.20])
SOCKET_HALF   = 0.024               # 내부 반폭 (peg 반경 0.018 → 반경 여유 6mm)
SOCKET_WALL   = 0.008
SOCKET_BASE   = 0.010               # 바닥판 두께 = 포켓 바닥(peg 착지면) z
SOCKET_WALL_H = 0.050
SOCKET_EXPECT_TYPE = 1

# ── 삽입 기하 (파지점을 커넥터 상단 근처로 잡아 손가락이 벽 위로 뜨게 함) ──
GRIP_ABOVE_CENTER = 0.025           # 파지점 = peg 중심 위 25mm (상단 근처)
FLOOR_Z          = SOCKET_BASE
SEATED_CENTER_Z  = FLOOR_Z + PEG_SIZE[2] / 2.0         # 착지 시 peg 중심 z (=0.045)
PICK_POS  = np.array([PEG_POS[0],   PEG_POS[1],   PEG_POS[2] + GRIP_ABOVE_CENTER])
PLACE_POS = np.array([SOCKET_XY[0], SOCKET_XY[1], SEATED_CENTER_Z + GRIP_ABOVE_CENTER])
EE_OFFSET = np.array([0.0, 0.0, 0.20])                 # 접근 높이


# ══════════════════════════════════════════════════════════════════════════
#  B. 헬퍼
# ══════════════════════════════════════════════════════════════════════════
def find_prim_path(root_path: str, name: str):
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def set_all_drives(root_path: str):
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
    print(f"  [OK] 드라이브 {n}개 설정")


def nucleus_exists(url: str) -> bool:
    if not url:
        return False
    try:
        res, _ = omni.client.stat(url)
        return res == omni.client.Result.OK
    except Exception:
        return False


def build_socket(world: World):
    """포켓형 소켓: 바닥판 1 + 벽 4 (모두 정적 FixedCuboid)."""
    sx, sy = float(SOCKET_XY[0]), float(SOCKET_XY[1])
    a, t, h = SOCKET_HALF, SOCKET_WALL, SOCKET_WALL_H
    wall_cz = SOCKET_BASE + h / 2.0
    outer = 2.0 * (a + t)
    world.scene.add(FixedCuboid(
        prim_path="/World/socket/base", name="socket_base",
        position=np.array([sx, sy, SOCKET_BASE / 2.0]),
        scale=np.array([outer, outer, SOCKET_BASE]),
        color=np.array([0.25, 0.25, 0.28]),
    ))
    for nm, pos, scale in [
        ("wx_p", [sx + a + t / 2, sy, wall_cz], [t, outer, h]),
        ("wx_n", [sx - a - t / 2, sy, wall_cz], [t, outer, h]),
        ("wy_p", [sx, sy + a + t / 2, wall_cz], [2 * a, t, h]),
        ("wy_n", [sx, sy - a - t / 2, wall_cz], [2 * a, t, h]),
    ]:
        world.scene.add(FixedCuboid(
            prim_path=f"/World/socket/{nm}", name=f"socket_{nm}",
            position=np.array(pos), scale=np.array(scale),
            color=np.array([0.35, 0.35, 0.40]),
        ))
    print(f"  [OK] 소켓(포켓) @ ({sx}, {sy}), 내부반폭 {a*1000:.0f}mm, 벽높이 {h*1000:.0f}mm")


def _attach_molex_cap(cap_path: str):
    """Molex 실제 커넥터 메시를 몸체 상단에 시각(충돌 없음)으로 정렬 배치."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(cap_path)
    xf = UsdGeom.XformCommonAPI(prim)
    xf.SetScale(Gf.Vec3f(CAP_SCALE, CAP_SCALE, CAP_SCALE))
    # 원본 메시 중심을 박스 상단면(local z=+PEG_SIZE[2]/2)으로 이동
    top = float(PEG_SIZE[2] / 2.0)
    t = -CAP_MESH_OFFSET * CAP_SCALE + np.array([0.0, 0.0, top])
    xf.SetTranslate(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))


def create_connector(world: World, material: PhysicsMaterial):
    """절차적 박스(실제 파지/삽입) + Molex 실제 커넥터 시각 캡. (pose 핸들, 상태) 반환."""
    peg = world.scene.add(DynamicCuboid(
        prim_path="/World/connector", name="connector",
        position=PEG_POS.copy(), scale=PEG_SIZE.copy(),
        color=np.array([0.12, 0.12, 0.16]), mass=0.05, physics_material=material,
    ))
    molex_url = (get_assets_root_path() or "") + MOLEX_CAP_REL
    if nucleus_exists(molex_url):
        add_reference_to_stage(molex_url, "/World/connector/cap")
        _attach_molex_cap("/World/connector/cap")
        print("  [OK] 커넥터 = 절차적 박스 + Molex 실제 커넥터 외형 캡")
        return peg, "box+molex_cap"
    print("  [OK] 커넥터 = 절차적 박스 (Molex 캡 미접속 — 네트워크/경로 확인)")
    return peg, "box_only"


def initialize_robot(robot, world):
    """로봇 + 그리퍼 초기화 (4_pick_place와 동일)."""
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )


def check_correct_part(part_type_id: int, socket_expected_type: int) -> bool:
    """WRONG_PART 확장 지점 — 지금은 종류 일치만 (베이스라인은 통과 가정)."""
    return part_type_id == socket_expected_type


# ══════════════════════════════════════════════════════════════════════════
#  C. 메인
# ══════════════════════════════════════════════════════════════════════════
def main():
    # ── C-1. World + 로봇 USD 로드 (/World 에 reference — 4_pick_place와 동일) ──
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    print("\n[1] 로봇 USD 로드")
    stage = omni.usd.get_context().get_stage()
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    world_prim.GetReferences().AddReference(ROBOT_USD_PATH)
    for _ in range(15):
        simulation_app.update()
    print(f"  [OK] {ROBOT_USD_PATH}")

    # ── C-2. 물리 설정 ────────────────────────────────────────────────────
    print("\n[2] 물리 설정")
    set_all_drives(ROBOT_PRIM_PATH)

    # ── C-3. 장면: 소켓 + 커넥터 + 마찰 재질 ──────────────────────────────
    print("\n[3] 장면 구성")
    build_socket(world)
    grip_mat = PhysicsMaterial(
        prim_path="/World/Physics_Materials/grip_mat",
        static_friction=1.8, dynamic_friction=1.4, restitution=0.0,
    )
    peg, peg_status = create_connector(world, grip_mat)

    # ── C-4. 로봇 등록 (ParallelGripper + SingleManipulator) ──────────────
    print("\n[4] 로봇 등록")
    ee_path = find_prim_path(ROBOT_PRIM_PATH, EE_LINK_NAME)
    if ee_path is None:
        raise RuntimeError(f"'{EE_LINK_NAME}' 링크를 찾을 수 없음")
    gripper = ParallelGripper(
        end_effector_prim_path=ee_path,
        joint_prim_names=GRIPPER_JOINTS,
        joint_opened_positions=np.array(GRIP_OPEN),
        joint_closed_positions=np.array(GRIP_CLOSE),
        action_deltas=np.array(GRIP_DELTA),
    )
    robot = world.scene.add(SingleManipulator(
        prim_path=ROBOT_PRIM_PATH, name="m0609_robot",
        end_effector_prim_path=ee_path, gripper=gripper,
    ))
    print(f"  [OK] SingleManipulator @ {ROBOT_PRIM_PATH}, EE={ee_path}")

    # 손가락 마찰(파지 안정화)
    finger_mat = PhysicsMaterial(
        prim_path="/World/Physics_Materials/finger_mat",
        static_friction=1.8, dynamic_friction=1.4, restitution=0.0,
    )
    for ln in FINGER_LINKS:
        lp = find_prim_path(ROBOT_PRIM_PATH, ln)
        if lp:
            SingleGeometryPrim(prim_path=lp, name=f"{ln}_geom").apply_physics_material(finger_mat)

    # ── C-5. reset + 초기화 + 컨트롤러 생성 ───────────────────────────────
    world.reset()
    initialize_robot(robot, world)
    for _ in range(30):
        world.step(render=True)

    print("\n[5] PickPlaceController 생성")
    controller = PickPlaceController(
        name="m0609_connector_controller",
        gripper=robot.gripper,
        robot_articulation=robot,
        end_effector_initial_height=0.30,
        events_dt=EVENTS_DT,
        urdf_path=ROBOT_URDF_PATH,
        robot_description_path=ROBOT_DESC_PATH,
        rmpflow_config_path=RMPFLOW_CFG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
    )
    print("  [OK] 컨트롤러 준비 완료")
    print(f"      PICK  @ {np.round(PICK_POS, 3)}")
    print(f"      PLACE @ {np.round(PLACE_POS, 3)} (소켓 삽입 지점)")

    if not check_correct_part(PEG_TYPE_ID, SOCKET_EXPECT_TYPE):
        print("  [WRONG_PART] 잘못된 커넥터 — (복구 로직은 후속 단계)")

    # ── C-6. Play 엣지 재실행 루프 (★ 재시작 버그 수정) ───────────────────
    print("\n[6] 준비 완료 — 뷰포트에서 Play 를 누르면 삽입을 실행합니다.")
    print("     (Stop 후 다시 Play 하면 처음부터 재실행)\n")
    was_playing = False
    reported = False
    while simulation_app.is_running():
        world.step(render=True)
        playing = world.is_playing()

        # Play 엣지 → 리셋 후 처음부터
        if playing and not was_playing:
            print("[Play] 삽입 시퀀스 시작")
            world.reset()
            initialize_robot(robot, world)
            controller.reset()
            reported = False

        if playing and not controller.is_done():
            action = controller.forward(
                picking_position=PICK_POS,
                placing_position=PLACE_POS,
                current_joint_positions=robot.get_joint_positions(),
                end_effector_offset=EE_OFFSET,
            )
            robot.apply_action(action)

        elif playing and controller.is_done() and not reported:
            # ── 삽입 성공 판정 ──
            for _ in range(20):
                world.step(render=True)
            pos, _ = peg.get_world_pose()
            pos = np.asarray(pos)
            xy_err = float(np.linalg.norm(pos[:2] - SOCKET_XY))
            z_err = float(abs(pos[2] - SEATED_CENTER_Z))
            success = (xy_err < 0.020) and (z_err < 0.020)
            print("\n" + "=" * 60)
            print(f"[결과] 커넥터 최종 위치 = {np.round(pos, 4)}  (peg={peg_status})")
            print(f"       xy 오차={xy_err*1000:.1f}mm  z 오차={z_err*1000:.1f}mm")
            print(f"       삽입 성공 = {success}")
            print("=" * 60 + "\n")
            reported = True

        was_playing = playing

    simulation_app.close()


if __name__ == "__main__":
    main()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

from pathlib import Path
import sys
import time
import random
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32


import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator

_THIS_DIR = Path(__file__).resolve().parent

# rmpflow 인프라 폴더 경로 등록 (인프라 파일 내부 import가 그대로 동작)
RMPFLOW_DIR = str(_THIS_DIR / "rmpflow")
if RMPFLOW_DIR not in sys.path:
    sys.path.insert(0, RMPFLOW_DIR)
 
from m0609_pick_place_controller import PickPlaceController

# ╔══════════════════════════════════════════════════════════════╗
# ║  A. Task 파라미터 (이전 장과 동일)                              ║
# ╚══════════════════════════════════════════════════════════════╝
USD_PATH        = str(_THIS_DIR / "Collected_m0609_camera/m0609_camera.usd")
ROBOT_PRIM_PATH = "/World/m0609"
EE_LINK_NAME    = "link_6"
GRIPPER_JOINTS  = ["finger_joint", "right_inner_knuckle_joint"]

DRIVE_STIFFNESS = 1e8
DRIVE_DAMPING   = 1e4
DRIVE_MAX_FORCE = 1e8

GRIPPER_OPEN    = [0.0, 0.0]
GRIPPER_CLOSE   = [0.5, 0.5]
GRIPPER_DELTA   = [-0.5, -0.5]

FINGER_STATIC   = 1.8
FINGER_DYNAMIC  = 1.4
CUBE_STATIC     = 1.2
CUBE_DYNAMIC    = 1.0


# ╔══════════════════════════════════════════════════════════════╗
# ║  B. Controller 파라미터 (★ 이번 장에서 새로 추가)               ║
# ╚══════════════════════════════════════════════════════════════╝

# ── B-1. 인프라 파일 경로 (RMPFlow가 참조) ────────────────────
M0609_URDF_PATH           = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
M0609_DESCRIPTION_PATH    = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
M0609_RMPFLOW_CONFIG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")

# ── B-2. Pick & Place 동작 파라미터 ───────────────────────────
CUBE_INIT_POS = np.array([0.30, 0.4, 0.0515 / 2.0])   # 큐브 초기 위치
GOAL_POS_GREEN = np.array([0.55, -0.35, 0.0])
GOAL_POS_BLUE = np.array([-0.55, 0.35, 0.0])             # 목표 위치
GOAL_POS = GOAL_POS_BLUE
EE_OFFSET     = np.array([0.0, 0.0, 0.2])               # 접근 높이
CUBE_COLOR    = np.array([0.0, 0.0, 1.0])

# ── B-3. 10단계 타이밍 (작을수록 빠름) ────────────────────────
EVENTS_DT = [
    0.008,   # 0. 접근 이동
    0.005,   # 1. 하강
    0.02,    # 2. 그리퍼 닫기 대기
    0.1,     # 3. 그리퍼 닫힘 유지
    0.0025,  # 4. 들어올리기
    0.01,    # 5. Place 위치로 이동
    0.0025,  # 6. 하강
    1,       # 7. 그리퍼 열기 대기
    0.008,   # 8. 상승
    0.08,    # 9. 복귀
]


# ============================================================
# 유틸 (이전 장과 동일)
# ============================================================
def find_prim_path_by_name(root_path: str, name: str):
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None

def initialize_robot(robot, world):
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )
    robot.set_joint_positions(np.zeros(robot.num_dof))

# ============================================================
# Task — 이전 장에서 완성한 M0609Task (변경 없음)
# ============================================================
class M0609Task(BaseTask):

    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._task_achieved = False

        if not rclpy.ok():
            rclpy.init()
        self.ros_node = Node("isaac_subscriber")

        self.color_sub = self.ros_node.create_subscription(
            Int32,
            "/color_id",
            self.color_callback,
            10
        )
    
    def color_callback(self, msg):
        global GOAL_POS
        color_data = msg.data
        if color_data == 1:
            GOAL_POS = GOAL_POS_BLUE
        else:
            GOAL_POS = GOAL_POS_GREEN
        
    
    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._load_usd()
        self._discover_links()
        self._setup_physics()
        self._register_robot(scene)
        self._create_scene(scene)
        print("\n  [완료] 씬 구성 성공!\n")

    def _load_usd(self):
        print("\n" + "=" * 60)
        print("[1.LOAD] USD 로드")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()
        world_prim = stage.GetPrimAtPath("/World")
        if not world_prim.IsValid():
            world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        world_prim.GetReferences().AddReference(USD_PATH)
        for _ in range(15):
            simulation_app.update()
        print(f"  [OK] {USD_PATH}")

    def _discover_links(self):
        print("\n" + "=" * 60)
        print("[2.DISCOVER] 링크 경로 탐색")
        print("=" * 60)
        self._ee_path = find_prim_path_by_name(ROBOT_PRIM_PATH, EE_LINK_NAME)
        if self._ee_path is None:
            raise RuntimeError(f"'{EE_LINK_NAME}' not found")
        print(f"  EE ({EE_LINK_NAME}) = {self._ee_path}")
        for jn in GRIPPER_JOINTS:
            print(f"  {jn:<35} = {find_prim_path_by_name(ROBOT_PRIM_PATH, jn)}")


    def _setup_physics(self):
        print("\n" + "=" * 60)
        print("[3.PHYSICS] 물리 설정")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()

        drive_count = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PRIM_PATH)):
            for dt in ["angular", "linear"]:
                drive = UsdPhysics.DriveAPI.Get(prim, dt)
                if drive:
                    drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
                    drive.GetDampingAttr().Set(DRIVE_DAMPING)
                    drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)
                    drive_count += 1
        print(f"  [OK] drive updated: {drive_count}")

    def _register_robot(self, scene):
        print("\n" + "=" * 60)
        print("[4.REGISTER] 로봇 등록")
        print("=" * 60)
        gripper = ParallelGripper(
            end_effector_prim_path=self._ee_path,
            joint_prim_names=GRIPPER_JOINTS,
            joint_opened_positions=np.array(GRIPPER_OPEN),
            joint_closed_positions=np.array(GRIPPER_CLOSE),
            action_deltas=np.array(GRIPPER_DELTA),
        )
        self._robot = scene.add(
            SingleManipulator(
                prim_path=ROBOT_PRIM_PATH,
                name="m0609_robot",
                end_effector_prim_path=self._ee_path,
                gripper=gripper,
            )
        )
        print(f"  [OK] SingleManipulator: {ROBOT_PRIM_PATH}")

    def _create_scene(self, scene):
        print("\n" + "=" * 60)
        print("[5.SCENE] 작업 환경 구성")
        print("=" * 60)
        cube_material = PhysicsMaterial(
            prim_path="/World/Physics_Materials/cube_material",
            static_friction=CUBE_STATIC,
            dynamic_friction=CUBE_DYNAMIC,
            restitution=0.0,
        )

        self._cube = scene.add(
            DynamicCuboid(
                prim_path="/World/target_cube",
                name="target_cube",
                position=CUBE_INIT_POS,
                scale=np.array([0.05, 0.05, 0.05]),
                color=CUBE_COLOR,
                mass=0.05,
                physics_material=cube_material,
            )
        )

        print(f"  [OK] cube @ {CUBE_INIT_POS}")

        scene.add(
            VisualCuboid(
                prim_path="/World/goal_marker_green",
                name="goal_marker_green",
                position=GOAL_POS_GREEN,
                scale=np.array([0.06, 0.06, 0.001]),
                color=np.array([0.0, 1.0, 0.0]),
            )
        )

        scene.add(
            VisualCuboid(
                prim_path="/World/goal_marker_blue",
                name="goal_marker_blue",
                position=GOAL_POS_BLUE,
                scale=np.array([0.06, 0.06, 0.001]),
                color=np.array([0.0, 0.0, 1.0]),
            )
        )

        print(f"  [OK] goal @ {GOAL_POS_BLUE}, {GOAL_POS_GREEN}")
        
        finger_material = PhysicsMaterial(
            prim_path="/World/Physics_Materials/finger_material",
            static_friction=FINGER_STATIC,
            dynamic_friction=FINGER_DYNAMIC,
            restitution=0.0,
        )
        for link_name in ["left_inner_finger", "right_inner_finger"]:
            link_path = find_prim_path_by_name(ROBOT_PRIM_PATH, link_name)
            if link_path:
                SingleGeometryPrim(
                    prim_path=link_path,
                    name=f"{link_name}_geom",
                ).apply_physics_material(finger_material)
                print(f"  [OK] friction: {link_path}")

    def get_observations(self):
        cube_pos, _ = self._cube.get_world_pose()

        return {
            self._robot.name: {
                "joint_positions": self._robot.get_joint_positions(),
            },
            self._cube.name: {
                "position": cube_pos,
                "goal_position": GOAL_POS,
            },
        }

    def pre_step(self, control_index, simulation_time):
        cube_pos, _ = self._cube.get_world_pose()

        if not self._task_achieved and np.mean(np.abs(GOAL_POS - cube_pos)) < 0.02:
            self._task_achieved = True

    def post_reset(self):

        rand_x = random.uniform(-0.2, 0.2)
        rand_y = random.uniform(-0.2, 0.2)

        new_random_pos = np.array([
            CUBE_INIT_POS[0] + rand_x,
            CUBE_INIT_POS[1] + rand_y,
            CUBE_INIT_POS[2]
        ])

        self._cube.set_world_pose(position=new_random_pos)
        self._cube.set_linear_velocity(np.zeros(3))
        self._cube.set_angular_velocity(np.zeros(3))

        rand_int = random.randint(1, 2)
        if rand_int == 1:
            # 파란색
            self._cube.get_applied_visual_material().set_color(np.array([0.0, 0.0, 1.0]))
        else:
            # 초록색 (RGB: 0, 1, 0)
            self._cube.get_applied_visual_material().set_color(np.array([0.0, 1.0, 0.0]))
        
        self._robot.gripper.set_joint_positions(
            self._robot.gripper.joint_opened_positions
        )
        
        self._task_achieved = False


# ╔══════════════════════════════════════════════════════════════╗
# ║  C. 메인 — Controller 생성 및 실행 (★ 이번 장 핵심)           ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    # ── C-1. World + Task (이전 장과 동일) ────────────────────
    my_world = World(stage_units_in_meters=1.0)
    task = M0609Task(name="m0609_task")
    my_world.add_task(task)
    my_world.reset()

    robot = my_world.scene.get_object("m0609_robot")
    initialize_robot(robot, my_world)

    # 홈 포지션 안정화 대기
    for _ in range(30):
        my_world.step(render=True)

    # ── C-2. Controller 생성 (initialize 이후에만 가능) ───────
    print("\n" + "=" * 60)
    print("[C-2] PickPlaceController 생성")
    print("=" * 60)
    print(f"  URDF        = {M0609_URDF_PATH}")
    print(f"  description = {M0609_DESCRIPTION_PATH}")
    print(f"  rmpflow     = {M0609_RMPFLOW_CONFIG_PATH}")
    print(f"  events_dt   = {EVENTS_DT}")
    print(f"  EE frame    = {EE_LINK_NAME}")

    controller = PickPlaceController(
        name="m0609_pick_place_controller",
        gripper=robot.gripper,
        robot_articulation=robot,
        end_effector_initial_height=0.30,
        events_dt=EVENTS_DT,
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
    )
    print("  [OK] Controller 생성 완료")

    # ── C-3. 초기 상태 진단 ───────────────────────────────────
    ee_pos, _ = robot.end_effector.get_world_pose()
    print(f"\n  EE 초기 위치 = {ee_pos}")
    print(f"  큐브 위치    = {CUBE_INIT_POS}")
    # print(f"  목표 위치    = {GOAL_POS}")

    # ── C-4. Controller 실행 루프 ─────────────────────────────
    print("\n[Pick & Place 시작]\n")
    was_playing = False
    task_done = False

    while simulation_app.is_running():
        my_world.step(render=True)
        time.sleep(0.01)
        is_playing = my_world.is_playing()

        rclpy.spin_once(task.ros_node, timeout_sec=0.0)

        # Play 시작 감지 → 리셋
        if is_playing and not was_playing:
            my_world.reset()
            initialize_robot(robot, my_world)
            controller.reset()
            task_done = False

        # 매 스텝 제어
        if is_playing and not task_done:

            # (1) 관측 데이터 수집
            obs = task.get_observations()
            cube_position  = obs["target_cube"]["position"]
            current_joints = obs["m0609_robot"]["joint_positions"]

            # (2) Controller에 목표 전달 → 관절 명령 생성
            actions = controller.forward(
                picking_position=cube_position,
                placing_position=GOAL_POS,
                current_joint_positions=current_joints,
                end_effector_offset=EE_OFFSET,
            )

            # (3) 로봇에 적용
            robot.apply_action(actions)

            # (4) 완료 확인
            if controller.is_done():
                print("[완료] Pick & Place 성공!")
                task_done = True
                my_world.pause()

            # 디버그 출력
            event = controller.get_current_event()
            ee_pos, _ = robot.end_effector.get_world_pose()
            print(f"  [event={event}] cube_z={cube_position[2]:.4f}  ee_z={ee_pos[2]:.4f}")

        was_playing = is_playing

    simulation_app.close()


if __name__ == "__main__":
    main()
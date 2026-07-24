import os
import sys
import time
import threading
from pathlib import Path

from isaacsim import SimulationApp

_HEADLESS = os.environ.get("AMR_HEADLESS") == "1"
simulation_app = SimulationApp({"headless": _HEADLESS})

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

import numpy as np
import omni.usd
from pxr import Usd, UsdPhysics, UsdGeom, Gf

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Int32, String

from isaacsim.core.api import World
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.core.utils.types import ArticulationAction

_THIS_DIR = Path(__file__).resolve().parent

# RMPFlow 모듈 경로 등록
sys.path.insert(0, str(_THIS_DIR / "rmpflow"))
from m0609_rmpflow_controller import RMPFlowController

# USD 및 URDF / RMPFlow 설정 파일 경로 (Collected_World_0123은 isaacpjt/M0609가 아니라
# 저장소 루트의 src/ 밑에 있다)
USD_PATH = str(_THIS_DIR.parent.parent / "src/Collected_World_0123/World0123.usd")
URDF_PATH = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
ROBOT_DESC_PATH = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
RMPFLOW_CFG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")

NOVA_CARTER_ROOT = "/World/Nova_Carter/chassis_link"
M0609_PATH       = "/World/m0609"
M0609_BASE_PATH  = "/World/m0609/base_link"
EE_LINK_NAME     = "link_6"
GRIPPER_JOINTS   = ["finger_joint", "right_inner_knuckle_joint"]

# 그리퍼 위치 파라미터
GRIPPER_OPEN  = np.array([0.0, 0.0])
GRIPPER_CLOSE = np.array([0.96, 0.96])
GRIPPER_DELTA = np.array([-0.5, -0.5])

# ★ 너트 파지: 그리퍼 핑거 마찰만으로는 안정적으로 붙잡지 못해(8_bolt_nut_screw.py에서
# 이미 검증된 문제) FixedJoint로 물리적으로 용접한다. arm_node의 NUT_GRASP에서 그리퍼를
# 닫은 직후 결속(engage), FASTEN 재생 궤적이 그리퍼를 여는 프레임에서 해제(release)한다.
# RigidBodyAPI/CollisionAPI는 nut1/nut2 자체가 아니라 geo/PolyShape 메시에 붙어 있다
# (World0123.usd 실측 확인됨) -- FixedJoint body1은 반드시 실제 RigidBody 프림을 가리켜야 한다.
NUT_PRIM_PATHS = {"nut1": "/World/nut1/geo/PolyShape", "nut2": "/World/nut2/geo/PolyShape"}
# 너트별로 body0/body1을 "생성 시점에 고정"해둔 별도 조인트를 쓴다. 하나의 조인트를
# 매번 engage_grasp()에서 body0Rel/body1Rel로 재타겟하면(이전 버전) PhysX가 Play 이후
# 조인트 토폴로지 변경을 제대로 재쿠킹하지 못해 "disjointed body transforms" 경고와
# 함께 팔이 튕겨나가는(snap) 현상이 재현됐다 -- body 타겟은 절대 런타임에 바꾸지 않고,
# enabled 플래그와 로컬 오프셋(순수 attribute, 토폴로지 아님)만 매번 갱신한다
# (8_bolt_nut_screw.py에서 검증된 패턴과 동일).
GRASP_JOINT_PATHS = {
    "nut1": "/World/_nut_grasp_joint_nut1",
    "nut2": "/World/_nut_grasp_joint_nut2",
}


def _upright_quat(quat):
    """더 이상 보정하지 않고 그대로 반환한다. World0123.usd의 nut1/nut2
    geo/PolyShape 자체 orient를 고쳐서(2026-07-24) local Z가 이제 정말로 위를
    향하므로, 여기서 identity를 강제하면 오히려 라이브 실제 자세(이제는 이미
    올바름, 다만 identity와 정확히 같지는 않고 Z축 기준 180도 yaw 차이가 있을 수
    있음)와 어긋나는 새 mismatch를 만들어 FixedJoint가 "disjointed"로 판단해
    다시 스냅(180도 튐)을 일으킨다 -- 실측으로 확인됨. 씬이 이미 정정됐으니
    engage_grasp()는 라이브 자세를 그대로 믿는다(8_bolt_nut_screw.py의 원래
    방식과 동일)."""
    return quat


def make_grasp_joints(stage, ee_path):
    for nut_id, joint_path in GRASP_JOINT_PATHS.items():
        j = UsdPhysics.FixedJoint.Define(stage, joint_path)
        j.CreateBody0Rel().SetTargets([ee_path])
        j.CreateBody1Rel().SetTargets([NUT_PRIM_PATHS[nut_id]])
        j.CreateJointEnabledAttr(False)


def engage_grasp(stage, ee_path, nut_id):
    """ee_path(그리퍼)-nut_id(너트)의 현재 상대 자세를 그대로 고정해 결속을 활성화한다."""
    joint_path = GRASP_JOINT_PATHS.get(nut_id)
    if joint_path is None:
        return False

    b0_xf = get_world_transform(stage, ee_path)
    b1_xf = get_world_transform(stage, NUT_PRIM_PATHS[nut_id])
    b0_pos, b0_quat = b0_xf.ExtractTranslation(), b0_xf.ExtractRotationQuat()
    b1_pos, b1_quat_raw = b1_xf.ExtractTranslation(), b1_xf.ExtractRotationQuat()
    b1_quat = _upright_quat(b1_quat_raw)

    # 진단용: 보정 로직이 실제로 그 순간 라이브 자세에 대해 뭘 하고 있는지 확인.
    import math
    raw_z = Gf.Rotation(b1_quat_raw).TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
    fixed_z = Gf.Rotation(b1_quat).TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
    raw_tilt = math.degrees(math.acos(max(-1.0, min(1.0, raw_z[2]))))
    fixed_tilt = math.degrees(math.acos(max(-1.0, min(1.0, fixed_z[2]))))
    print(
        f"[GRASP] {nut_id} 결속 시점 raw local_z={tuple(round(v,3) for v in raw_z)} "
        f"tilt={raw_tilt:.1f}deg -> 보정후 local_z={tuple(round(v,3) for v in fixed_z)} "
        f"tilt={fixed_tilt:.1f}deg  |  raw_quat_wxyz={tuple(round(v,4) for v in (b1_quat_raw.GetReal(), *b1_quat_raw.GetImaginary()))}"
    )

    m0 = Gf.Matrix4d(); m0.SetRotateOnly(b0_quat); m0.SetTranslateOnly(b0_pos)
    m1 = Gf.Matrix4d(); m1.SetRotateOnly(b1_quat); m1.SetTranslateOnly(b1_pos)
    rel = m1 * m0.GetInverse()
    t = rel.ExtractTranslation()
    r = rel.ExtractRotationQuat()

    j = UsdPhysics.FixedJoint.Get(stage, joint_path)
    j.CreateLocalPos0Attr(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
    j.CreateLocalRot0Attr(Gf.Quatf(float(r.GetReal()), *[float(x) for x in r.GetImaginary()]))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.GetJointEnabledAttr().Set(True)
    return True


def release_grasp(stage):
    for joint_path in GRASP_JOINT_PATHS.values():
        j = UsdPhysics.FixedJoint.Get(stage, joint_path)
        if j:
            j.GetJointEnabledAttr().Set(False)


class IsaacRmpflowNode(Node):
    """목표 Pose 수신, 그리퍼 명령 수신 및 현재 엔드이펙터 Pose 피드백 퍼블리시 노드"""

    def __init__(self):
        super().__init__('isaac_rmpflow_node')
        
        self.target_position = None
        self.target_orientation = None
        self.latest_gripper_cmd = 0  # 초기 상태: 0 (열림)
        # arm_node의 nut_fasten(/arm/joint_command 직접 관절 재생)이 팔을 쥐고 있는 동안은
        # False로 내려와 RMPFlow가 마지막 target_pose를 계속 재적용해 충돌하는 것을 막는다.
        self.rmpflow_enabled = True

        # 1. 목표 Pose 서브스크라이버
        self.subscription = self.create_subscription(
            PoseStamped,
            '/arm/target_pose',
            self.target_pose_callback,
            10
        )

        # 2. 그리퍼 명령 서브스크라이버 (0: 열림, 1: 닫힘)
        self.gripper_sub = self.create_subscription(
            Int32,
            '/arm/gripper_command',
            self.gripper_command_callback,
            10
        )

        # 3. RMPFlow 활성/비활성 서브스크라이버
        self.rmpflow_enable_sub = self.create_subscription(
            Bool,
            '/arm/rmpflow_enable',
            self.rmpflow_enable_callback,
            10
        )

        # 4. 너트 결속(FixedJoint) 요청 서브스크라이버. data=nut_id 결속, data=""(빈 문자열) 해제.
        # None="대기 중인 요청 없음" 센티널이라 매 프레임 재적용하지 않고 딱 한 번만 처리한다.
        self.nut_grasp_request = None
        self.nut_grasp_attach_sub = self.create_subscription(
            String,
            '/arm/nut_grasp_attach',
            self.nut_grasp_attach_callback,
            10
        )

        # 3. 현재 Pose 퍼블리셔 (Feedback 전용)
        self.current_pose_pub = self.create_publisher(
            PoseStamped,
            '/arm/current_pose',
            10
        )
        
        self.get_logger().info("Isaac Sim RMPFlow Node 준비 완료")

    def target_pose_callback(self, msg: PoseStamped):
        self.target_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        # RMPFlowController Quaternion 순서: [w, x, y, z]
        self.target_orientation = np.array([
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ])

    def gripper_command_callback(self, msg: Int32):
        """0: 열림(Open), 1: 닫힘(Close) 수신"""
        self.latest_gripper_cmd = msg.data

    def rmpflow_enable_callback(self, msg: Bool):
        self.rmpflow_enabled = msg.data

    def nut_grasp_attach_callback(self, msg: String):
        self.nut_grasp_request = msg.data

    def publish_current_ee_pose(self, stage, ee_prim_path):
        """현재 link_6의 World Pose를 ROS 2 토픽으로 퍼블리시"""
        if not ee_prim_path:
            return

        prim = stage.GetPrimAtPath(ee_prim_path)
        if not prim.IsValid():
            return

        ee_xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        pos = ee_xf.ExtractTranslation()
        quat = ee_xf.ExtractRotationQuat()

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        
        msg.pose.orientation.w = float(quat.GetReal())
        msg.pose.orientation.x = float(quat.GetImaginary()[0])
        msg.pose.orientation.y = float(quat.GetImaginary()[1])
        msg.pose.orientation.z = float(quat.GetImaginary()[2])

        self.current_pose_pub.publish(msg)


def set_all_drives(stage, root_path, stiffness=1.0e8, damping=1.0e4, max_force=1.0e8):
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        for dt in ("angular", "linear"):
            drive = UsdPhysics.DriveAPI.Get(prim, dt)
            if drive:
                drive.GetStiffnessAttr().Set(stiffness)
                drive.GetDampingAttr().Set(damping)
                drive.GetMaxForceAttr().Set(max_force)


def lock_amr_base(stage, amr_root_path):
    """★ [핵심 해결 1] AMR 차체 및 바퀴 관절을 완전 잠금(Break Lock)하여 미끄러짐/기울어짐 방지"""
    amr_prim = stage.GetPrimAtPath(amr_root_path).GetParent()
    if not amr_prim.IsValid():
        amr_prim = stage.GetPrimAtPath(amr_root_path)

    locked_count = 0
    for prim in Usd.PrimRange(amr_prim):
        for dt in ("angular", "linear"):
            drive = UsdPhysics.DriveAPI.Get(prim, dt)
            if drive:
                drive.GetStiffnessAttr().Set(1.0e9)
                drive.GetDampingAttr().Set(1.0e6)
                drive.GetMaxForceAttr().Set(1.0e9)
                if drive.GetTargetVelocityAttr():
                    drive.GetTargetVelocityAttr().Set(0.0)
                locked_count += 1
    print(f"AMR 베이스 및 바퀴 관절 드라이브 잠금 완료 (잠긴 Drive 수: {locked_count})")


def find_prim_path(stage, root_path, name):
    root = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def get_world_transform(stage, path):
    prim = stage.GetPrimAtPath(path)
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)


def main():
    rclpy.init()

    ctx = omni.usd.get_context()
    ctx.open_stage(USD_PATH)
    for _ in range(15):
        simulation_app.update()

    stage = ctx.get_stage()
    world = World(stage_units_in_meters=1.0)

    # ★ AMR 차체 및 바퀴 고정 적용
    lock_amr_base(stage, NOVA_CARTER_ROOT)

    # 로봇팔 관절 Physics Drive 적용
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
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )

    make_grasp_joints(stage, ee_path)

    # RMPFlow Controller 생성
    arm_controller = RMPFlowController(
        name="m0609_external_pose_controller",
        robot_articulation=robot,
        urdf_path=URDF_PATH,
        robot_description_path=ROBOT_DESC_PATH,
        rmpflow_config_path=RMPFLOW_CFG_PATH,
        end_effector_frame_name="link_6",
    )

    # ROS 2 스핀 스레드 실행
    sub_node = IsaacRmpflowNode()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(sub_node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    print("\n[Isaac Sim] RMPFlow 실행 준비 완료. Play 버튼을 눌러 동작을 시작하세요.")

    # Simulation Loop
    was_playing = False
    while simulation_app.is_running():
        world.step(render=True)

        playing = world.is_playing()

        if playing and not was_playing:
            # ★ Stop 후 다시 Play를 누른 시점. Stop을 누르면 물리 상태(로봇
            # 관절 등)는 USD 초기 자세로 리셋되지만, RMPFlow 컨트롤러 내부
            # 상태와 ROS2 쪽에 남아있는 마지막 target_pose는 리셋되지 않는다.
            # 이 상태에서 곧바로 arm_controller.forward()가 그 오래된(멀리
            # 떨어진) 목표를 계속 쓰면, 이제 막 초기 자세로 리셋된 로봇이
            # 한 프레임 만에 그 큰 오차를 따라잡으려다 튕겨나간다.
            # -> world/robot을 다시 초기화하고, RMPFlow도 리셋하고, 마지막
            #    target을 지워서 새 명령이 올 때까지 움직이지 않게 한다.
            world.reset()
            robot.initialize()
            robot.gripper.initialize(
                physics_sim_view=world.physics_sim_view,
                articulation_apply_action_func=robot.apply_action,
                get_joint_positions_func=robot.get_joint_positions,
                set_joint_positions_func=robot.set_joint_positions,
                dof_names=robot.dof_names,
            )
            arm_controller.reset()
            sub_node.target_position = None
            sub_node.target_orientation = None
            release_grasp(stage)
            sub_node.nut_grasp_request = None
            print("재시작 감지 -> world/robot/RMPFlow 재초기화, 이전 목표 좌표/너트 결속 초기화")

        was_playing = playing

        if playing:
            # ★ [핵심 해결 2] 매 프레임마다 RMPFlow 베이스 Pose를 실제 base_link World 좌표로 실시간 갱신!
            base_xf = get_world_transform(stage, M0609_BASE_PATH)
            base_position = base_xf.ExtractTranslation()
            base_quaternion = base_xf.ExtractRotationQuat()
            arm_controller._motion_policy.set_robot_base_pose(
                robot_position=np.array([base_position[0], base_position[1], base_position[2]]),
                robot_orientation=np.array([
                    base_quaternion.GetReal(),
                    *[float(val) for val in base_quaternion.GetImaginary()]
                ]),
            )

            # A. 현재 엔드이펙터 Pose 지속 퍼블리시
            sub_node.publish_current_ee_pose(stage, ee_path)

            # A-2. 너트 결속/해제 요청 처리 (rmpflow_enabled 여부와 무관하게 항상 처리 -
            # 결속은 RMPFlow 구간(NUT_GRASP)에서, 해제는 joint 재생 구간(FASTEN)에서 온다)
            if sub_node.nut_grasp_request is not None:
                req = sub_node.nut_grasp_request
                sub_node.nut_grasp_request = None
                if req:
                    if engage_grasp(stage, ee_path, req):
                        print(f"[GRASP] {req} FixedJoint 결속 활성화")
                    else:
                        print(f"[GRASP] 알 수 없는 nut_id={req!r}, 결속 요청 무시")
                else:
                    release_grasp(stage)
                    print("[GRASP] FixedJoint 결속 해제")

            # B. 목표 Pose 수신 시 RMPFlow 연산 및 명령 적용
            # (nut_fasten이 /arm/joint_command로 직접 관절을 제어하는 동안은 건너뛴다 -
            #  안 그러면 여기서 마지막 target_pose를 계속 재적용해 재생 궤적과 충돌한다)
            if sub_node.rmpflow_enabled:
                if sub_node.target_position is not None and sub_node.target_orientation is not None:
                    action = arm_controller.forward(
                        target_end_effector_position=sub_node.target_position,
                        target_end_effector_orientation=sub_node.target_orientation,
                    )
                    robot.apply_action(action)

                # C. 그리퍼 상태 제어 (GRIPPER_JOINTS도 nut_fasten 재생 관절 목록에 포함되므로 동일하게 건너뜀)
                if sub_node.latest_gripper_cmd == 1:
                    robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))
                else:
                    robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

    sub_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == '__main__':
    main()
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
from pxr import Usd, UsdPhysics, UsdGeom

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32

from isaacsim.core.api import World
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.core.utils.types import ArticulationAction

_THIS_DIR = Path(__file__).resolve().parent

# RMPFlow 모듈 경로 등록
sys.path.insert(0, str(_THIS_DIR / "rmpflow"))
from m0609_rmpflow_controller import RMPFlowController

# USD 및 URDF / RMPFlow 설정 파일 경로
USD_PATH = str(_THIS_DIR / "Collected_World0_123/World0123.usd")
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
GRIPPER_CLOSE = np.array([0.8, 0.8])
GRIPPER_DELTA = np.array([-0.5, -0.5])


class IsaacRmpflowNode(Node):
    """목표 Pose 수신, 그리퍼 명령 수신 및 현재 엔드이펙터 Pose 피드백 퍼블리시 노드"""

    def __init__(self):
        super().__init__('isaac_rmpflow_node')
        
        self.target_position = None
        self.target_orientation = None
        self.latest_gripper_cmd = 0  # 초기 상태: 0 (열림)

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
            print("재시작 감지 -> world/robot/RMPFlow 재초기화, 이전 목표 좌표 초기화")

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

            # B. 목표 Pose 수신 시 RMPFlow 연산 및 명령 적용
            if sub_node.target_position is not None and sub_node.target_orientation is not None:
                action = arm_controller.forward(
                    target_end_effector_position=sub_node.target_position,
                    target_end_effector_orientation=sub_node.target_orientation,
                )
                robot.apply_action(action)

            # C. 그리퍼 상태 제어
            if sub_node.latest_gripper_cmd == 1:
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))
            else:
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

    sub_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == '__main__':
    main()
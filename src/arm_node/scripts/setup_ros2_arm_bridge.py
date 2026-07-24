"""setup_ros2_arm_bridge.py -- World0123.usd에 arm_node <-> Isaac Sim 실시간 연동용
OmniGraph 노드를 /World/ActionGraph에 추가하고 스테이지에 저장한다 (1회성 스크립트).

기존 /World/ActionGraph에는 TF 퍼블리시(ros2_publish_transform_tree)만 있었고,
README에 명시된 아래 두 연동이 빠져 있었다:
  Isaac Sim --/joint_states--> arm_node
  Isaac Sim <--/arm/joint_command-- arm_node

추가하는 노드:
  ros2_subscribe_arm_joint_command (ROS2SubscribeJointState, topic=/arm/joint_command)
    -> isaac_articulation_controller (IsaacArticulationController, targetPrim=chassis_link)
  ros2_publish_joint_states (ROS2PublishJointState, topic=/joint_states, targetPrim=chassis_link)

targetPrim은 /World/Nova_Carter/chassis_link -- World0123.usd에서 m0609이 FixedJoint로
Nova_Carter에 용접돼 있어 실제 PhysX 아티큘레이션 루트가 거기 있기 때문
(record_nut_fasten_trajectory.py에서 확인한 것과 동일한 이유).

실행:
  /home/rokey/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh \
      /home/rokey/EV_combine/src/arm_node/scripts/setup_ros2_arm_bridge.py
"""
import os

_HEADLESS = os.environ.get("BOLT_HEADLESS", "1") == "1"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": _HEADLESS})

import omni.usd
import omni.kit.app
import omni.graph.core as og

WORLD_USD = "/home/rokey/EV_combine/src/Collected_World_0123/World0123.usd"
ARTICULATION_ROOT_PATH = "/World/Nova_Carter/chassis_link"
GRAPH_PATH = "/World/ActionGraph"

# 기본 앱 설정(isaacsim.exp.base.python.kit)엔 ros2 bridge가 자동으로 안 켜져 있어서
# ROS2SubscribeJointState/ROS2PublishJointState 노드 타입이 등록 안 된 상태다.
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.core.nodes", True)
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
for _ in range(10):
    simulation_app.update()

context = omni.usd.get_context()
context.open_stage(WORLD_USD)
for _ in range(20):
    simulation_app.update()

keys = og.Controller.Keys
og.Controller.edit(
    GRAPH_PATH,
    {
        keys.CREATE_NODES: [
            ("ros2_subscribe_arm_joint_command", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            ("isaac_articulation_controller", "isaacsim.core.nodes.IsaacArticulationController"),
            ("ros2_publish_joint_states", "isaacsim.ros2.bridge.ROS2PublishJointState"),
        ],
        keys.SET_VALUES: [
            ("ros2_subscribe_arm_joint_command.inputs:topicName", "/arm/joint_command"),
            ("isaac_articulation_controller.inputs:targetPrim", ARTICULATION_ROOT_PATH),
            ("ros2_publish_joint_states.inputs:topicName", "/joint_states"),
            ("ros2_publish_joint_states.inputs:targetPrim", ARTICULATION_ROOT_PATH),
        ],
        keys.CONNECT: [
            (f"{GRAPH_PATH}/on_playback_tick.outputs:tick", "ros2_subscribe_arm_joint_command.inputs:execIn"),
            ("ros2_subscribe_arm_joint_command.outputs:execOut", "isaac_articulation_controller.inputs:execIn"),
            ("ros2_subscribe_arm_joint_command.outputs:jointNames", "isaac_articulation_controller.inputs:jointNames"),
            ("ros2_subscribe_arm_joint_command.outputs:positionCommand", "isaac_articulation_controller.inputs:positionCommand"),
            (f"{GRAPH_PATH}/on_playback_tick.outputs:tick", "ros2_publish_joint_states.inputs:execIn"),
            (f"{GRAPH_PATH}/isaac_read_simulation_time.outputs:simulationTime", "ros2_publish_joint_states.inputs:timeStamp"),
        ],
    },
)

for _ in range(5):
    simulation_app.update()

stage = context.get_stage()
stage.GetRootLayer().Save()

_log_f = open(os.path.join(os.path.dirname(__file__), "setup_ros2_arm_bridge_result.txt"), "w")
_log_f.write(f"[저장 완료] {WORLD_USD}\n")
_log_f.close()

simulation_app.close()

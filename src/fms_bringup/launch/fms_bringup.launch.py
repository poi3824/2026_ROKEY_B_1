"""FMS 연동 ROS2 시스템 아키텍처 전체 노드 launch.

fleet_manager_node, behavior_node, amr_node, arm_node, perception_node를 함께 기동한다.
Isaac Sim은 별도 환경에서 /camera/color, /camera/depth, /joint_states 등을 퍼블리시해야 한다.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='fleet_manager_node',
            executable='fleet_manager_node',
            name='fleet_manager_node',
            output='screen',
        ),
        Node(
            package='behavior_node',
            executable='behavior_node',
            name='behavior_node',
            output='screen',
        ),
        Node(
            package='amr_node',
            executable='amr_node',
            name='amr_node',
            output='screen',
        ),
        Node(
            package='arm_node',
            executable='arm_node',
            name='arm_node',
            output='screen',
        ),
        Node(
            package='perception_node',
            executable='perception_node',
            name='perception_node',
            output='screen',
        ),
    ])

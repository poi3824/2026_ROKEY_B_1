"""FMS 연동 ROS2 시스템 아키텍처 전체 노드 launch.

fleet_manager_node, behavior_node, amr_node, arm_node, perception_node, error_fix_node를
함께 기동한다. Isaac Sim은 별도 환경에서 /camera/color, /camera/depth, /joint_states 등을
퍼블리시해야 한다.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'start_fleet_manager',
            default_value='false',
            description='자동 작업 발행 Fleet Manager 실행 여부',
        ),
        Node(
            package='fleet_manager_node',
            executable='fleet_manager_node',
            name='fleet_manager_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('start_fleet_manager')),
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
        Node(
            package='error_fix',
            executable='error_fix_node',
            name='error_fix_node',
            output='screen',
        ),
    ])

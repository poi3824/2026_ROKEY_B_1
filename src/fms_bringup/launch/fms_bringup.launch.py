"""FMS 연동 ROS2 시스템 아키텍처 전체 노드 launch.

fleet_manager_node, behavior_node, amr_node, arm_node, perception_node, error_fix_node를
함께 기동한다. Isaac Sim은 별도 환경에서 /camera/color, /camera/depth, /joint_states 등을
퍼블리시해야 한다.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _node(package, executable, *, condition=None):
    """Create a node with consistent, launch-friendly console logging."""
    return Node(
        package=package,
        executable=executable,
        name=executable,
        output='screen',
        emulate_tty=True,
        ros_arguments=['--log-level', LaunchConfiguration('log_level')],
        condition=condition,
    )


def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_COLORIZED_OUTPUT', '1'),
        SetEnvironmentVariable(
            'RCUTILS_CONSOLE_OUTPUT_FORMAT',
            '[{time}] [{severity}] [{name}] {message}',
        ),
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),
        DeclareLaunchArgument(
            'start_fleet_manager',
            default_value='false',
            description='자동 작업 발행 Fleet Manager 실행 여부',
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            choices=['debug', 'info', 'warn', 'error', 'fatal'],
            description='모든 ROS 노드의 콘솔 로그 레벨',
        ),
        _node(
            'fleet_manager_node',
            'fleet_manager_node',
            condition=IfCondition(LaunchConfiguration('start_fleet_manager')),
        ),
        _node('behavior_node', 'behavior_node'),
        _node('amr_node', 'amr_node'),
        _node('arm_node', 'arm_node'),
        _node('perception_node', 'perception_node'),
        _node('error_fix', 'error_fix_node'),
    ])

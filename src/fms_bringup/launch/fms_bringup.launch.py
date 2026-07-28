"""
FMS 연동 ROS 2 시스템의 전체 노드를 기동한다.

fleet_manager_node, behavior_node, amr_node, arm_node, 카메라별 perception_node,
error_fix_node를 함께 기동한다.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

from perception_node.multi_camera_config import (
    CAMERA_CONFIGS,
    DEFAULT_MODEL_FILENAME,
    FALLBACK_MODEL_FILENAME,
    perception_remappings,
)


def _perception_nodes(model_path):
    """Create one strictly isolated perception process for each camera source."""
    return [
        Node(
            package='perception_node',
            executable='perception_node',
            name=config.node_name,
            output='screen',
            parameters=[{
                'rgb_topic': config.rgb_topic,
                'depth_topic': config.depth_topic,
                'camera_info_topic': config.camera_info_topic,
                'camera_frame_override': config.optical_frame,
                'debug_image_topic': (
                    f'/{config.namespace}/perception/debug_image'
                ),
                'model_path': model_path,
                'max_rgb_depth_skew_sec': 0.1,
                'max_input_receipt_age_sec': 1.0,
                'require_unique_camera_source': True,
                'required_target_frames': 3,
                'use_bolt_roi': config.use_bolt_roi,
            }],
            remappings=perception_remappings(config),
        )
        for config in CAMERA_CONFIGS
    ]


def generate_launch_description():
    model_path = LaunchConfiguration('perception_model_path')
    nodes = [
        DeclareLaunchArgument(
            'perception_model_path',
            default_value=PathJoinSubstitution([
                FindPackageShare('perception_node'),
                'models',
                DEFAULT_MODEL_FILENAME,
            ]),
            description=(
                'YOLO model shared by all camera perception instances; '
                f'set this to {FALLBACK_MODEL_FILENAME} for the v1 fallback'
            ),
        ),
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
    ]
    nodes.extend(_perception_nodes(model_path))
    nodes.append(
        Node(
            package='error_fix',
            executable='error_fix_node',
            name='error_fix_node',
            output='screen',
        )
    )
    return LaunchDescription(nodes)

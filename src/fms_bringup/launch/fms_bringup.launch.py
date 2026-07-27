"""FMS 연동 ROS2 시스템 아키텍처 전체 노드 launch.

fleet_manager_node, behavior_node, amr_node, arm_node, 카메라별 perception_node,
error_fix_node를 함께 기동한다.
Isaac Sim은 별도 환경에서 손목·버스바·볼트 카메라와 /joint_states 등을 퍼블리시해야 한다.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def _perception_remappings(camera_namespace, keep_global=()):
    """perception_node의 절대 토픽/서비스 이름을 카메라별로 격리한다."""
    names = (
        '/perception/detections_3d',
        '/perception/get_grasp_pose',
        '/perception/get_bolt_pair',
        '/vision/busbar_grasp',
        '/vision/nut_pose',
        '/vision/busbar_grasp_pose',
        '/vision/bolt_a_pose',
        '/vision/bolt_b_pose',
    )
    return [
        (name, name if name in keep_global else f'/{camera_namespace}{name}')
        for name in names
    ]


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
            name='wrist_perception_node',
            output='screen',
            parameters=[{
                'rgb_topic': '/rgb',
                'depth_topic': '/depth',
                'camera_info_topic': '/camera_info',
                'camera_frame_override': 'camera_color_optical_frame',
                'debug_image_topic': '/wrist/perception/debug_image',
            }],
            remappings=_perception_remappings(
                'wrist',
                keep_global=('/perception/get_grasp_pose', '/vision/nut_pose'),
            ),
        ),
        Node(
            package='perception_node',
            executable='perception_node',
            name='busbar_perception_node',
            output='screen',
            parameters=[{
                'rgb_topic': '/busbar_cam/rgb',
                'depth_topic': '/busbar_cam/depth',
                'camera_info_topic': '/busbar_cam/camera_info',
                'camera_frame_override': 'busbar_cam_optical_frame',
                'debug_image_topic': '/busbar_cam/perception/debug_image',
            }],
            remappings=_perception_remappings(
                'busbar_cam',
                keep_global=('/vision/busbar_grasp',),
            ),
        ),
        Node(
            package='perception_node',
            executable='perception_node',
            name='bolt_perception_node',
            output='screen',
            parameters=[{
                'rgb_topic': '/bolt_cam/rgb',
                'depth_topic': '/bolt_cam/depth',
                'camera_info_topic': '/bolt_cam/camera_info',
                'camera_frame_override': 'bolt_cam_optical_frame',
                'debug_image_topic': '/bolt_cam/perception/debug_image',
            }],
            remappings=_perception_remappings('bolt_cam'),
        ),
        Node(
            package='error_fix',
            executable='error_fix_node',
            name='error_fix_node',
            output='screen',
        ),
    ])

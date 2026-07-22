"""perception_node를 rosbag 재생만으로 단독 테스트하기 위한 launch.

이 bag(rosbag2_2026_07_22-12_27_13)에는 tf가 녹화되어 있지 않으므로, world 좌표
변환이 동작하도록 world -> sim_camera identity static transform을 임시로 함께
띄운다.

TODO: 실제 로봇/Isaac Sim 연동 시에는 이 identity static_transform_publisher를
빼고, 실제 카메라 extrinsic(로봇 URDF/robot_state_publisher 또는 캘리브레이션
결과)에서 나온 tf를 사용해야 한다.

사용법:
    ros2 launch perception_node perception_bag_test.launch.py
    # 별도 터미널에서
    ros2 bag play <bag 경로>/rosbag2_2026_07_22-12_27_13
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world_to_sim_camera_placeholder_tf',
            arguments=['--frame-id', 'world', '--child-frame-id', 'sim_camera'],
            output='screen',
        ),
        Node(
            package='perception_node',
            executable='perception_node',
            name='perception_node',
            output='screen',
        ),
    ])

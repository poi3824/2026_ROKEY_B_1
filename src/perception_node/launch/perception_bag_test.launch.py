"""perception_node를 rosbag 재생만으로 단독 테스트하기 위한 launch.

bag에 따라 tf 유무가 다르다:
- rosbag2_2026_07_22-12_27_13, rosbag2_bolt: tf 없음. 이미지 frame_id도 sim_camera.
  -> publish_placeholder_tf:=true 로 identity static tf(world -> sim_camera)를
     띄워야 world 좌표 변환이 동작한다.
- rosbag2_busbar: 실제 tf 포함(world -> camera_color_optical_frame, 팔 움직임 반영).
  단, 이미지 헤더 frame_id는 여전히 sim_camera라서 tf 트리의 프레임 이름과
  다르다 -> camera_frame_override:=camera_color_optical_frame 으로 tf 조회에
  쓸 프레임 이름을 맞춰줘야 한다. 이 경우 placeholder tf는 끄고(기본값) bag이
  퍼블리시하는 실제 tf를 그대로 사용한다.

사용법:
    # tf 포함 bag (기본값)
    ros2 launch perception_node perception_bag_test.launch.py
    ros2 bag play <bag 경로>/rosbag2_busbar

    # tf 없는 bag
    ros2 launch perception_node perception_bag_test.launch.py \
        publish_placeholder_tf:=true camera_frame_override:=sim_camera
    ros2 bag play <bag 경로>/rosbag2_2026_07_22-12_27_13
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera_frame_override_arg = DeclareLaunchArgument(
        'camera_frame_override', default_value='camera_color_optical_frame')
    publish_placeholder_tf_arg = DeclareLaunchArgument(
        'publish_placeholder_tf', default_value='false')

    camera_frame_override = LaunchConfiguration('camera_frame_override')
    publish_placeholder_tf = LaunchConfiguration('publish_placeholder_tf')

    return LaunchDescription([
        camera_frame_override_arg,
        publish_placeholder_tf_arg,
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world_to_camera_placeholder_tf',
            arguments=['--frame-id', 'world', '--child-frame-id', camera_frame_override],
            output='screen',
            condition=IfCondition(publish_placeholder_tf),
        ),
        Node(
            package='perception_node',
            executable='perception_node',
            name='perception_node',
            output='screen',
            parameters=[{'camera_frame_override': camera_frame_override}],
        ),
    ])

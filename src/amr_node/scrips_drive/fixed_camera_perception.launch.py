"""Busbar.usd 고정 카메라용 perception 두 인스턴스를 완전히 분리해 실행한다."""

from launch import LaunchDescription
from launch_ros.actions import Node


COMMON_REMAPPINGS = (
    "/perception/get_grasp_pose",
    "/perception/get_bolt_pair",
    "/perception/detections_3d",
    "/vision/busbar_grasp",
    "/vision/nut_pose",
    "/vision/busbar_grasp_pose",
    "/vision/bolt_a_pose",
    "/vision/bolt_b_pose",
)


def _perception_node(camera_name, optical_frame, use_bolt_roi):
    prefix = f"/{camera_name}"
    remappings = [
        (name, f"{prefix}{name}")
        for name in COMMON_REMAPPINGS
    ]
    return Node(
        package="perception_node",
        executable="perception_node",
        name=f"perception_{camera_name}",
        output="screen",
        parameters=[
            {
                "rgb_topic": f"{prefix}/rgb",
                "depth_topic": f"{prefix}/depth",
                "camera_info_topic": f"{prefix}/camera_info",
                "camera_frame_override": optical_frame,
                "world_frame": "world",
                "debug_image_topic": f"{prefix}/perception/debug_image",
                "use_bolt_roi": use_bolt_roi,
                "use_busbar_pnp": False,
                "require_unique_camera_source": True,
                "max_rgb_depth_skew_sec": 0.1,
                "max_input_receipt_age_sec": 1.0,
            }
        ],
        remappings=remappings,
    )


def generate_launch_description():
    return LaunchDescription(
        [
            _perception_node(
                "busbar_cam",
                "busbar_cam_optical_frame",
                False,
            ),
            _perception_node(
                "bolt_cam",
                "bolt_cam_optical_frame",
                True,
            ),
        ]
    )

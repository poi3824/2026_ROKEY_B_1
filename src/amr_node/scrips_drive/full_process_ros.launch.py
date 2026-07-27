"""기존 bringup을 수정하지 않고 비전 물리 주행+Arm 전체 공정 노드를 실행한다."""

import sys
from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


DRIVE_DIR = Path(__file__).resolve().parent


def _wrist_remappings():
    names = (
        "/perception/detections_3d",
        "/perception/get_grasp_pose",
        "/perception/get_bolt_pair",
        "/vision/busbar_grasp",
        "/vision/nut_pose",
        "/vision/busbar_grasp_pose",
        "/vision/bolt_a_pose",
        "/vision/bolt_b_pose",
    )
    keep_global = {
        "/perception/get_grasp_pose",
        "/vision/nut_pose",
    }
    return [
        (
            name,
            name if name in keep_global else f"/wrist{name}",
        )
        for name in names
    ]


def generate_launch_description():
    fixed_perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(DRIVE_DIR / "fixed_camera_perception.launch.py")
        )
    )
    wrist_perception = Node(
        package="perception_node",
        executable="perception_node",
        name="perception_wrist",
        output="screen",
        parameters=[
            {
                "rgb_topic": "/rgb",
                "depth_topic": "/depth",
                "camera_info_topic": "/camera_info",
                "camera_frame_override": "camera_color_optical_frame",
                "debug_image_topic": "/wrist/perception/debug_image",
                "use_bolt_roi": False,
                # 손목 근접 시 planar PnP의 반대 해가 선택되어 world Z가
                # 음수가 되는 실측 사례가 있어 depth 역투영을 사용한다.
                "use_busbar_pnp": False,
                "require_unique_camera_source": True,
            }
        ],
        remappings=_wrist_remappings(),
    )
    arm = Node(
        package="arm_node",
        executable="arm_node",
        name="arm_node",
        output="screen",
        parameters=[
            {
                "use_amr_selected_battery_midpoint": True,
                "debug_no_timeouts": True,
                "wrist_busbar_required_samples": 2,
                "wrist_busbar_observation_sec": 2.0,
                "wrist_busbar_max_target_distance_m": 0.10,
                "wrist_busbar_max_target_z_distance_m": 0.10,
                "wrist_busbar_max_sample_shift_m": 0.05,
            }
        ],
    )
    error_fix = Node(
        package="error_fix",
        executable="error_fix_node",
        name="error_fix_node",
        output="screen",
    )
    relay = ExecuteProcess(
        cmd=[
            sys.executable,
            str(DRIVE_DIR / "amr_node_drive.py"),
            "--ros-args",
            "-p",
            "vision_standoff_m:=0.90",
            "-p",
            "battery_standoff_m:=0.75",
            "-p",
            "battery_preapproach_extra_m:=0.80",
            "-p",
            "battery_initial_pose_tolerance_m:=0.05",
            "-p",
            "battery_initial_yaw_tolerance_deg:=5.0",
            "-p",
            "vision_max_target_drift_m:=0.005",
            "-p",
            "vision_enforce_active_drift:=false",
            "-p",
            "busbar_preapproach_extra_m:=0.80",
            "-p",
            "active_vision_timeout_sec:=6.0",
            "-p",
            "busbar_scan_yaw_offset_deg:=180.0",
            "-p",
            "battery_bolt_pair_spacing_m:=0.331",
            "-p",
            "battery_bolt_pair_spacing_tolerance_m:=0.080",
            "-p",
            "debug_no_timeouts:=true",
        ],
        output="screen",
    )
    return LaunchDescription(
        [
            fixed_perception,
            wrist_perception,
            arm,
            error_fix,
            relay,
        ]
    )

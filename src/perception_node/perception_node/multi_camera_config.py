"""Shared launch contract for the wrist, busbar, and bolt perception nodes."""

from dataclasses import dataclass


DEFAULT_MODEL_FILENAME = 'keypoints_busbar6pt_v3.pt'
FALLBACK_MODEL_FILENAME = 'keypoints_busbar6pt_v1.pt'


@dataclass(frozen=True)
class CameraPerceptionConfig:
    """ROS graph settings for one camera-specific perception instance."""

    namespace: str
    node_name: str
    rgb_topic: str
    depth_topic: str
    camera_info_topic: str
    optical_frame: str
    keep_global_outputs: tuple[str, ...] = ()
    use_bolt_roi: bool = False


CAMERA_CONFIGS = (
    CameraPerceptionConfig(
        namespace='wrist',
        node_name='wrist_perception_node',
        rgb_topic='/rgb',
        depth_topic='/depth',
        camera_info_topic='/camera_info',
        optical_frame='camera_color_optical_frame',
        keep_global_outputs=('/vision/nut_pose',),
    ),
    CameraPerceptionConfig(
        namespace='busbar_cam',
        node_name='busbar_perception_node',
        rgb_topic='/busbar_cam/rgb',
        depth_topic='/busbar_cam/depth',
        camera_info_topic='/busbar_cam/camera_info',
        optical_frame='busbar_cam_optical_frame',
        keep_global_outputs=('/vision/busbar_grasp',),
    ),
    CameraPerceptionConfig(
        namespace='bolt_cam',
        node_name='bolt_perception_node',
        rgb_topic='/bolt_cam/rgb',
        depth_topic='/bolt_cam/depth',
        camera_info_topic='/bolt_cam/camera_info',
        optical_frame='bolt_cam_optical_frame',
        use_bolt_roi=True,
    ),
)

PERCEPTION_GRAPH_NAMES = (
    '/perception/detections_3d',
    '/perception/get_grasp_pose',
    '/perception/get_bolt_pair',
    '/perception/reset_cache',
    '/perception/reset_ack',
    '/vision/busbar_grasp',
    '/vision/nut_pose',
)


def perception_remappings(config: CameraPerceptionConfig) -> list[tuple[str, str]]:
    """Return isolated graph names while preserving only intentional legacy outputs."""
    return [
        (
            name,
            name
            if name in config.keep_global_outputs
            else f'/{config.namespace}{name}',
        )
        for name in PERCEPTION_GRAPH_NAMES
    ]

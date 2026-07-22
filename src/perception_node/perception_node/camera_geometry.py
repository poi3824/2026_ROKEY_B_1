"""픽셀 좌표 + depth -> 카메라 좌표계 3D 좌표 역투영, 그리고 world 좌표계로의 tf 변환.

sensor_msgs/CameraInfo에서 얻은 실제 intrinsic(K, D)을 사용한다
(perception 브랜치 프로토타입의 D455 스펙 추정치 대신).

transform_pixel_to_world()는 perception_node.py(실시간 노드)와
scripts/inspect_bag_frame.py(bag 프레임 1개만 오프라인으로 확인하는 디버그 도구)
양쪽에서 공유한다. tf_buffer를 실시간 TransformListener로 채우든, bag에서 읽은
tf를 직접 set_transform으로 채우든 동일하게 동작한다.

tf 조회는 이미지 header.stamp가 아니라 "가장 최근에 들어온 tf"(rclpy.time.Time(),
즉 시각 0 = latest)를 사용한다. rosbag2_busbar 등 이 프로젝트의 시뮬레이션 녹화본은
/rgb·/depth와 /tf의 header.stamp가 서로 다른(전혀 겹치지 않는) 시간 구간을 쓰는
경우가 있어서, "정확히 이미지가 찍힌 그 시각"의 tf를 요구하면 항상
extrapolation 실패가 난다. 카메라가 매 순간 크게 움직이지 않는 스캔 상황이라면
최신 tf를 그대로 써도 정확도 손실이 실질적으로 무시할 만하다.
"""
import numpy as np
from geometry_msgs.msg import PointStamped
from image_geometry import PinholeCameraModel
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from tf2_geometry_msgs import do_transform_point
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer


def make_camera_model(camera_info: CameraInfo) -> PinholeCameraModel:
    model = PinholeCameraModel()
    model.fromCameraInfo(camera_info)
    return model


def pixel_to_camera_point(model: PinholeCameraModel, u: float, v: float, depth: float) -> tuple[float, float, float]:
    """(u, v) 픽셀 + depth(광학축 방향 거리, m) -> 카메라 프레임 3D 좌표 (x, y, z)."""
    rect_u, rect_v = model.rectifyPoint((u, v))
    ray_x, ray_y, ray_z = model.projectPixelTo3dRay((rect_u, rect_v))
    scale = depth / ray_z
    return ray_x * scale, ray_y * scale, ray_z * scale


def sample_depth(depth_image: np.ndarray, u: float, v: float):
    """(u, v) 픽셀의 depth(m)를 반환. 이미지 범위 밖이거나 값이 무효하면 None."""
    row, col = int(round(v)), int(round(u))
    if not (0 <= row < depth_image.shape[0] and 0 <= col < depth_image.shape[1]):
        return None
    depth_value = float(depth_image[row, col])
    if not np.isfinite(depth_value) or depth_value <= 0.0:
        return None
    return depth_value


def transform_pixel_to_world(model: PinholeCameraModel, depth_image: np.ndarray, pixel_uv,
                              tf_buffer: Buffer, world_frame: str, camera_frame_id: str, stamp,
                              timeout_sec: float = 0.2, on_tf_error=None):
    """픽셀 -> (카메라 좌표, world 좌표, 실패 사유) 로 변환.

    camera_point / world_point: (x, y, z) 튜플, 실패 시 None.
    status: "" (성공) / "no depth" (depth 범위 밖 또는 무효) / "tf fail" (tf 조회 실패).
    tf 실패의 상세 예외 메시지는 status에 넣지 않고(오버레이 등에 짧게 쓰기 위함)
    on_tf_error(ex) 콜백으로 필요할 때만 전달한다.
    """
    u, v = pixel_uv
    depth_value = sample_depth(depth_image, u, v)
    if depth_value is None:
        return None, None, 'no depth'

    camera_point = pixel_to_camera_point(model, u, v, depth_value)

    point_camera = PointStamped()
    point_camera.header.stamp = stamp
    point_camera.header.frame_id = camera_frame_id
    point_camera.point.x, point_camera.point.y, point_camera.point.z = camera_point

    try:
        transform = tf_buffer.lookup_transform(
            world_frame, camera_frame_id, Time(),  # Time() = 시각 0 = "가장 최근 tf"
            timeout=Duration(seconds=timeout_sec))
    except TransformException as ex:
        if on_tf_error is not None:
            on_tf_error(ex)
        return camera_point, None, 'tf fail'

    point_world = do_transform_point(point_camera, transform)
    world_point = (point_world.point.x, point_world.point.y, point_world.point.z)
    return camera_point, world_point, ''

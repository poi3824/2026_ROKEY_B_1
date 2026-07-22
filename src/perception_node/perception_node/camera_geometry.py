"""픽셀 좌표 + depth -> 카메라 좌표계 3D 좌표 역투영.

sensor_msgs/CameraInfo에서 얻은 실제 intrinsic(K, D)을 사용한다
(perception 브랜치 프로토타입의 D455 스펙 추정치 대신).
"""
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo


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

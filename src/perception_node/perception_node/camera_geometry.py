"""
픽셀 좌표와 depth를 카메라 3D 좌표로 역투영한 뒤 world tf로 변환한다.

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
import math

import cv2
import numpy as np
from geometry_msgs.msg import PointStamped
from image_geometry import PinholeCameraModel
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from tf2_geometry_msgs import do_transform_point
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer

from perception_node.detector import BUSBAR_KEYPOINT_ORDER


def make_camera_model(camera_info: CameraInfo) -> PinholeCameraModel:
    model = PinholeCameraModel()
    model.fromCameraInfo(camera_info)
    return model


def pixel_to_camera_point(
    model: PinholeCameraModel,
    u: float,
    v: float,
    depth: float,
) -> tuple[float, float, float]:
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


def camera_point_to_world(
    camera_point,
    tf_buffer: Buffer,
    world_frame: str,
    camera_frame_id: str,
    stamp,
    timeout_sec: float = 0.2,
    on_tf_error=None,
):
    """Transform a camera-frame point into world coordinates, or return None."""
    point_camera = PointStamped()
    point_camera.header.stamp = stamp
    point_camera.header.frame_id = camera_frame_id
    point_camera.point.x, point_camera.point.y, point_camera.point.z = camera_point

    try:
        transform = tf_buffer.lookup_transform(
            world_frame,
            camera_frame_id,
            Time(),
            timeout=Duration(seconds=timeout_sec),
        )
    except TransformException as ex:
        if on_tf_error is not None:
            on_tf_error(ex)
        return None

    point_world = do_transform_point(point_camera, transform)
    return point_world.point.x, point_world.point.y, point_world.point.z


def transform_pixel_to_world(
    model: PinholeCameraModel,
    depth_image: np.ndarray,
    pixel_uv,
    tf_buffer: Buffer,
    world_frame: str,
    camera_frame_id: str,
    stamp,
    timeout_sec: float = 0.2,
    on_tf_error=None,
):
    """
    픽셀을 카메라 좌표, world 좌표, 실패 사유로 변환한다.

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

    world_point = camera_point_to_world(
        camera_point,
        tf_buffer,
        world_frame,
        camera_frame_id,
        stamp,
        timeout_sec,
        on_tf_error,
    )
    if world_point is None:
        return camera_point, None, 'tf fail'
    return camera_point, world_point, ''


# ---------------------------------------------------------------------------
# Busbar 6-keypoint PnP path.
#
# The object points come from /World/Z_busbar3/Mesh. The authored prim scale
# is 0.75, so applying it here converts the mesh-local metre coordinates into
# the actual simulated dimensions used by the v1/v3 training labels.
BUSBAR_WORLD_SCALE = 0.75
_BUSBAR_LOCAL_KEYPOINTS_M = {
    "hole_A": (-0.241638, -0.138881, 0.003000),
    "hole_B": (0.241638, 0.138881, 0.003000),
    "elbow_A": (-0.211638, -0.030000, 0.003000),
    "elbow_B": (0.211638, 0.030000, 0.003000),
    "tip_A": (-0.271638, -0.168881, 0.003000),
    "tip_B": (0.271638, 0.168881, 0.003000),
}
BUSBAR_OBJECT_POINTS_M = np.array(
    [_BUSBAR_LOCAL_KEYPOINTS_M[name] for name in BUSBAR_KEYPOINT_ORDER],
    dtype=np.float64,
) * BUSBAR_WORLD_SCALE


def solve_busbar_pnp(model: PinholeCameraModel, keypoints_px: np.ndarray):
    """
    Solve a planar busbar camera pose from the six v1/v3 landmarks.

    Returns ``(rvec, tvec, reprojection_error_px)`` or None for malformed,
    degenerate, or behind-camera solutions.
    """
    image_points = np.asarray(keypoints_px, dtype=np.float64)
    if (
        image_points.shape != (len(BUSBAR_KEYPOINT_ORDER), 2)
        or not np.all(np.isfinite(image_points))
        or np.ptp(image_points[:, 0]) < 1.0
        or np.ptp(image_points[:, 1]) < 1.0
    ):
        return None

    camera_matrix = np.asarray(model.intrinsicMatrix(), dtype=np.float64)
    distortion = np.asarray(model.distortionCoeffs(), dtype=np.float64)
    try:
        ok, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            BUSBAR_OBJECT_POINTS_M.reshape(-1, 1, 3),
            image_points.reshape(-1, 1, 2),
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return None
    if not ok or not rvecs or errors is None:
        return None

    candidates = [
        index
        for index, tvec in enumerate(tvecs)
        if (
            np.all(np.isfinite(tvec))
            and float(np.asarray(tvec).reshape(3)[2]) > 0.0
        )
    ]
    if not candidates:
        return None
    best = min(
        candidates,
        key=lambda index: float(np.asarray(errors[index]).reshape(-1)[0]),
    )
    return (
        rvecs[best],
        tvecs[best],
        float(np.asarray(errors[best]).reshape(-1)[0]),
    )


def busbar_pnp_world_pose(
    model: PinholeCameraModel,
    keypoints_px: np.ndarray,
    tf_buffer: Buffer,
    world_frame: str,
    camera_frame_id: str,
    stamp,
    timeout_sec: float = 0.2,
    on_tf_error=None,
):
    """Return the two hole world points, midpoint, and yaw for a busbar."""
    solved = solve_busbar_pnp(model, keypoints_px)
    if solved is None:
        return {'status': 'pnp fail'}
    rvec, tvec, reprojection_error = solved
    rotation, _ = cv2.Rodrigues(rvec)

    hole_world = {}
    for name in ("hole_A", "hole_B"):
        local_point = (
            np.asarray(_BUSBAR_LOCAL_KEYPOINTS_M[name], dtype=np.float64)
            * BUSBAR_WORLD_SCALE
        )
        camera_point = (
            rotation @ local_point.reshape(3, 1) + tvec
        ).reshape(3)
        world_point = camera_point_to_world(
            tuple(camera_point),
            tf_buffer,
            world_frame,
            camera_frame_id,
            stamp,
            timeout_sec,
            on_tf_error,
        )
        if world_point is None:
            return {'status': 'tf fail'}
        hole_world[name] = world_point

    ax, ay, _ = hole_world["hole_A"]
    bx, by, _ = hole_world["hole_B"]
    return {
        'hole_world': hole_world,
        'mid_xy': ((ax + bx) / 2.0, (ay + by) / 2.0),
        'yaw_rad': math.atan2(by - ay, bx - ax),
        'reprojection_error_px': reprojection_error,
        'status': '',
    }


def dual_path_discrepancy_m(pnp_xy, depth_xy) -> float:
    """Return planar disagreement between PnP and depth estimates in metres."""
    return math.hypot(pnp_xy[0] - depth_xy[0], pnp_xy[1] - depth_xy[1])


def select_busbar_world_point(
    depth_world,
    pnp_world,
    max_z_disagreement_m: float = 0.10,
    prefer_depth: bool = False,
):
    """
    Select the safer of the depth and six-keypoint PnP busbar estimates.

    Landmark-centroid depth is more accurate when it lands on the busbar.
    Wrist-camera views can instead sample the floor through the open centre;
    in that case the large Z disagreement makes the geometry-only PnP result
    the safer estimate.

    Returns ``(world_point, source, z_disagreement_m)``. ``source`` is
    ``"depth"``, ``"pnp"``, or ``"none"``.
    """

    def _valid_point(point):
        if point is None:
            return False
        values = np.asarray(point, dtype=float)
        return values.shape == (3,) and np.all(np.isfinite(values))

    depth_valid = _valid_point(depth_world)
    pnp_valid = _valid_point(pnp_world)
    if depth_valid and pnp_valid:
        z_disagreement = abs(float(depth_world[2]) - float(pnp_world[2]))
        if prefer_depth:
            return tuple(depth_world), 'depth', z_disagreement
        if z_disagreement <= max_z_disagreement_m:
            return tuple(depth_world), 'depth', z_disagreement
        return tuple(pnp_world), 'pnp', z_disagreement
    if depth_valid:
        return tuple(depth_world), 'depth', None
    if pnp_valid:
        return tuple(pnp_world), 'pnp', None
    return None, 'none', None

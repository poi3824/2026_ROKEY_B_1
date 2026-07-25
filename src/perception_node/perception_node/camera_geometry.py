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


def camera_point_to_world(camera_point, tf_buffer: Buffer, world_frame: str, camera_frame_id: str,
                           stamp, timeout_sec: float = 0.2, on_tf_error=None):
    """카메라 프레임 3D 좌표 -> world 좌표. 실패 시 None.

    tf 실패의 상세 예외 메시지는 반환값에 넣지 않고(오버레이 등에 짧게 쓰기 위함)
    on_tf_error(ex) 콜백으로 필요할 때만 전달한다. depth 기반 경로(transform_pixel_to_world)와
    PnP 기반 경로(busbar_pnp_world_pose) 양쪽이 이 tf 조회 로직을 공유한다.
    """
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
        return None

    point_world = do_transform_point(point_camera, transform)
    return (point_world.point.x, point_world.point.y, point_world.point.z)


def transform_pixel_to_world(model: PinholeCameraModel, depth_image: np.ndarray, pixel_uv,
                              tf_buffer: Buffer, world_frame: str, camera_frame_id: str, stamp,
                              timeout_sec: float = 0.2, on_tf_error=None):
    """픽셀 -> (카메라 좌표, world 좌표, 실패 사유) 로 변환.

    camera_point / world_point: (x, y, z) 튜플, 실패 시 None.
    status: "" (성공) / "no depth" (depth 범위 밖 또는 무효) / "tf fail" (tf 조회 실패).
    """
    u, v = pixel_uv
    depth_value = sample_depth(depth_image, u, v)
    if depth_value is None:
        return None, None, 'no depth'

    camera_point = pixel_to_camera_point(model, u, v, depth_value)
    world_point = camera_point_to_world(
        camera_point, tf_buffer, world_frame, camera_frame_id, stamp, timeout_sec, on_tf_error)
    if world_point is None:
        return camera_point, None, 'tf fail'
    return camera_point, world_point, ''


# ---------------------------------------------------------------------------
# Stage3: busbar 6-keypoint PnP 경로 (경로 A). depth 기반 단일픽셀 역투영(경로 B, 위)과
# 이중경로로 교차검증한다(perception_node.py의 _resolve_busbar_pose).
#
# object point: /World/Z_busbar3/Mesh(World0123.usd)의 로컬 정점을 직접 파싱해 원피팅/코너
# 검출로 뽑은 값(vision_pipeline_summary.md 2.1절과 동일 절차, isaacpjt/M0609/
# generate_realsense_dataset_kp.py의 extract_busbar_keypoints를 로컬 좌표에 적용).
# 2026-07-25 실측: hole 반경=10.000mm, hole-hole 거리=557.4111mm — 문서 수치와 일치.
# world 배치 스케일 0.75는 /World/Z_busbar3 자신의 xformOp:scale에 직접 authored돼 있음이
# 확인됨(vision_pipeline_summary.md 2.2절) — 순수 등방 스케일이라 object point에 그대로
# 곱해 실제 미터 단위로 맞추면 solvePnP의 tvec이 곧바로 실측 세계 단위가 된다.
BUSBAR_WORLD_SCALE = 0.75
_BUSBAR_LOCAL_KEYPOINTS_M = {
    "hole_A": (-0.241638, -0.138881, 0.003000),
    "hole_B": (0.241638, 0.138881, 0.003000),
    "elbow_A": (-0.211638, -0.030000, 0.003000),
    "elbow_B": (0.211638, 0.030000, 0.003000),
    "tip_A": (-0.271638, -0.168881, 0.003000),
    "tip_B": (0.271638, 0.168881, 0.003000),
}
# 6점 전부 로컬 z가 동일(평면형 상단면) -> 평면 물체 전용 IPPE가 적합(일반 반복 PnP는
# 평면 입력에서 해가 불안정할 수 있음). solvePnPGeneric이 최대 2개 해를 내므로
# reprojection error가 작은 쪽을 채택한다.
BUSBAR_OBJECT_POINTS_M = np.array(
    [_BUSBAR_LOCAL_KEYPOINTS_M[name] for name in BUSBAR_KEYPOINT_ORDER], dtype=np.float64
) * BUSBAR_WORLD_SCALE


def solve_busbar_pnp(model: PinholeCameraModel, keypoints_px: np.ndarray):
    """busbar 6-keypoint 픽셀 -> (rvec, tvec, reprojection_error_px) 또는 실패 시 None.

    keypoints_px: shape (6,2), BUSBAR_KEYPOINT_ORDER 순서(detector.YoloPoseDetector 출력 그대로).
    """
    image_points = np.asarray(keypoints_px, dtype=np.float64).reshape(6, 1, 2)
    K = model.intrinsicMatrix()
    D = model.distortionCoeffs()

    ok, rvecs, tvecs, errors = cv2.solvePnPGeneric(
        BUSBAR_OBJECT_POINTS_M.reshape(6, 1, 3), image_points, K, D,
        flags=cv2.SOLVEPNP_IPPE)
    if not ok or not rvecs:
        return None

    best = int(np.argmin(errors))
    return rvecs[best], tvecs[best], float(errors[best][0])


def busbar_pnp_world_pose(model: PinholeCameraModel, keypoints_px: np.ndarray, tf_buffer: Buffer,
                           world_frame: str, camera_frame_id: str, stamp,
                           timeout_sec: float = 0.2, on_tf_error=None):
    """busbar 6-keypoint -> world 기준 hole_A/hole_B 3D 좌표 + yaw.

    반환: dict(hole_world={name: (x,y,z)}, mid_xy=(x,y), yaw_rad=float,
    reprojection_error_px=float, status=str). 실패 시 status에 'pnp fail'/'tf fail'.
    yaw는 회전행렬 성분 대신 world 변환 후의 hole_A->hole_B 벡터에서 atan2로 직접
    구한다 — 오일러 각 규약 모호성을 피하고, arm_node가 실제로 쓰는 단일 yaw 스칼라
    (top-down 접근이라 azimuth만 유의미)와 바로 대응된다.
    """
    solved = solve_busbar_pnp(model, keypoints_px)
    if solved is None:
        return {'status': 'pnp fail'}
    rvec, tvec, reproj_err = solved
    R, _ = cv2.Rodrigues(rvec)

    hole_world = {}
    for name in ("hole_A", "hole_B"):
        local_pt = np.array(_BUSBAR_LOCAL_KEYPOINTS_M[name], dtype=np.float64) * BUSBAR_WORLD_SCALE
        cam_pt = (R @ local_pt.reshape(3, 1) + tvec).reshape(3)
        world_pt = camera_point_to_world(
            tuple(cam_pt), tf_buffer, world_frame, camera_frame_id, stamp, timeout_sec, on_tf_error)
        if world_pt is None:
            return {'status': 'tf fail'}
        hole_world[name] = world_pt

    ax, ay, _ = hole_world["hole_A"]
    bx, by, _ = hole_world["hole_B"]
    mid_xy = ((ax + bx) / 2.0, (ay + by) / 2.0)
    yaw_rad = math.atan2(by - ay, bx - ax)

    return {
        'hole_world': hole_world,
        'mid_xy': mid_xy,
        'yaw_rad': yaw_rad,
        'reprojection_error_px': reproj_err,
        'status': '',
    }


def dual_path_discrepancy_m(pnp_xy, depth_xy) -> float:
    """경로 A(PnP)와 경로 B(depth) mid_xy 간 평면 유클리드 거리(m).

    vision_pipeline_summary.md Stage3 게이트(2mm, PhysX contactOffset 실측과 일치 —
    2.2절 참고)와 비교해 perception_node.py가 재검출/경고 여부를 판단하는 데 쓴다.
    """
    return math.hypot(pnp_xy[0] - depth_xy[0], pnp_xy[1] - depth_xy[1])

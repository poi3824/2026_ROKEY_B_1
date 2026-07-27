"""perception_node
실행 계층 · YOLO 검출(bolt/nut/busbar) -> world 좌표 변환 -> 토픽 발행.

SUB <rgb_topic>          (sensor_msgs/Image, rgb8)
SUB <depth_topic>        (sensor_msgs/Image, 32FC1, m)
SUB <camera_info_topic>  (sensor_msgs/CameraInfo)

PUB /perception/detections_3d  (vision_msgs/Detection3DArray, <world_frame> 기준)
PUB <debug_image_topic>         (sensor_msgs/Image, bgr8) — rqt_image_view 등으로
                                 keypoints/bbox/좌표(픽셀, 카메라 프레임, world 프레임) 및
                                 BOLT ROI 영역을 확인하기 위한 디버그 오버레이.
                                 publish_debug_image로 끌 수 있음.
SRV /perception/get_grasp_pose (fms_interfaces/GetGraspPose) — 라벨별 최신 검출(world 좌표,
                                 최근 SMOOTHING_WINDOW_SEC 이내 표본의 평균값)을 캐시에서
                                 그대로 반환. 호출 시점에 재추론하지 않으므로, 호출 전에
                                 타이머 주기(detection_period_sec)만큼은 대상이 카메라 시야에
                                 안정적으로 들어와 있어야 한다. grasp_query_max_age_sec보다
                                 캐시가 오래됐거나 해당 라벨 검출이 없으면 found=false.
SRV /perception/get_bolt_pair  (fms_interfaces/GetBoltPair) — 'bolt' 라벨 중 이번 tick 상위
                                 2개를 이전 tick 위치와 최근접 매칭해 A/B 각각 별도로
                                 롤링 평균한 값을 반환. 버스바가 다리를 걸치는 볼트 2개의
                                 실측 위치가 필요한 busbar_insert INSERT 단계에서 쓴다.
PUB /vision/busbar_grasp        (fms_interfaces/BusbarGrasp) — 'busbar' 라벨이 이번 tick에
                                 검출됐을 때만 (롤링 평균) world 좌표로 발행. behavior_node가
                                 구독해 busbar_insert 액션 goal.target_pose로 그대로 실어 보낸다.
PUB /vision/nut_pose            (fms_interfaces/NutPose) — 'nut' 라벨 검출 시 (롤링 평균) 발행.
                                 id는 항상 0(현재 파이프라인은 라벨당 최고 점수 1개만
                                 추적하므로 여러 너트를 구분하지 못함).

라벨별 검출은 단발성 YOLO+depth 한 tick 값을 그대로 쓰지 않는다 — 최근
SMOOTHING_WINDOW_SEC(기본 2초) 안의 표본을 모아 평균/표준편차를 계산하고, 매 tick
터미널에 raw/평균/표준편차/표본수를 로그로 남긴다(표준편차가 크면 warn). 캐시·토픽·
서비스 응답은 전부 이 평균값을 쓴다 — 단일 tick 노이즈에 팔이 흔들리는 것을 줄이기 위함.
'bolt'는 위 이유로 개별 라벨 캐시(get_grasp_pose)에는 넣지 않고, 대신 2개를 쌍으로
추적하는 get_bolt_pair 전용 경로로만 나간다. StudPose(fms_interfaces/msg)는 주석상
YOLO가 아닌 Hough Circle 기반으로 설계돼 있어 이 노드의 YOLO 파이프라인과는 별개
구현이 필요 — 여기서는 다루지 않는다.

카메라 프레임 -> world_frame 변환은 tf2 lookupTransform으로 조회한다. 이 노드는
카메라가 어디에 있는지 알지 못하며, world_frame -> 카메라 frame_id로의 tf가 어디선가
(로봇 URDF/robot_state_publisher, 또는 캘리브레이션용 static_transform_publisher)
발행되고 있어야 world 좌표가 채워진다. tf가 없으면 해당 검출은 skip하고 경고 로그만
남긴다.

tf 조회에 사용할 프레임 이름은 camera_info.header.frame_id를 기본으로 쓰지만,
camera_frame_override 파라미터가 설정되어 있으면 그 값을 대신 쓴다. 녹화 장비/센서
드라이버가 이미지 헤더에 넣는 frame_id와 실제 tf 트리에 있는 프레임 이름이 다른
경우(예: 이미지는 sim_camera, tf는 camera_color_optical_frame)를 위한 것.
"""
import collections
import math
import os
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from fms_interfaces.msg import BusbarGrasp, NutPose
from fms_interfaces.srv import GetBoltPair, GetGraspPose
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from perception_node import overlay
from perception_node.camera_geometry import (
    busbar_pnp_world_pose, dual_path_discrepancy_m, make_camera_model, transform_pixel_to_world)
from perception_node.detector import BUSBAR_KEYPOINT_ORDER, YoloPoseDetector

DEFAULT_MODEL_PATH = os.path.join(
    get_package_share_directory('perception_node'), 'models', 'keypoints_busbar6pt_v3.pt')

BUSBAR_LABEL = 'busbar'
NUT_LABEL = 'nut'
BOLT_LABEL = 'bolt'

# 롤링 평균/오차 분포 확인용 시간 창. 이 안에 든 표본만 평균·표준편차 계산에 쓴다.
SMOOTHING_WINDOW_SEC = 2.0
SMOOTHING_MIN_SAMPLES = 2
SMOOTHING_STD_WARN_MM = 5.0

# Stage3 이중경로(PnP vs depth) 불일치 게이트. vision_pipeline_summary.md 2.2절 실측:
# 이 값은 볼트 콜리전의 실제 PhysX contactOffset(2.0mm)과 일치 — 근거 있는 값.
BUSBAR_DUAL_PATH_GATE_M = 0.002


class PerceptionNode(Node):

    def __init__(self):
        super().__init__('perception_node')

        self.declare_parameter('rgb_topic', '/rgb')
        self.declare_parameter('depth_topic', '/depth')
        self.declare_parameter('camera_info_topic', '/camera_info')
        self.declare_parameter('model_path', DEFAULT_MODEL_PATH)
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('camera_frame_override', 'camera_color_optical_frame')
        self.declare_parameter('conf_threshold', 0.6)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('detection_period_sec', 0.5)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_image_topic', '/perception/debug_image')
        self.declare_parameter('grasp_query_max_age_sec', 5.0)
        self.declare_parameter('max_rgb_depth_skew_sec', 0.1)
        self.declare_parameter('max_input_receipt_age_sec', 1.0)
        self.declare_parameter('require_unique_camera_source', False)
        self.declare_parameter('use_busbar_pnp', True)

        # 🔥 bolt 라벨 스캔 시에만 적용할 ROI 영역 파라미터 (640x640 기준)
        self.declare_parameter('use_bolt_roi', True)
        self.declare_parameter('bolt_roi_x_min', 150)
        self.declare_parameter('bolt_roi_x_max', 490)
        self.declare_parameter('bolt_roi_y_min', 150)
        self.declare_parameter('bolt_roi_y_max', 450)
        self.declare_parameter('bolt_roi_reference_width', 640)
        self.declare_parameter('bolt_roi_reference_height', 640)

        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        model_path = self.get_parameter('model_path').value
        self._world_frame = self.get_parameter('world_frame').value
        self._camera_frame_override = self.get_parameter('camera_frame_override').value
        self._conf_threshold = self.get_parameter('conf_threshold').value
        self._iou_threshold = self.get_parameter('iou_threshold').value
        detection_period_sec = self.get_parameter('detection_period_sec').value
        publish_debug_image = self.get_parameter('publish_debug_image').value
        debug_image_topic = self.get_parameter('debug_image_topic').value
        self._grasp_query_max_age_sec = self.get_parameter('grasp_query_max_age_sec').value
        self._max_rgb_depth_skew_sec = float(
            self.get_parameter('max_rgb_depth_skew_sec').value)
        self._max_input_receipt_age_sec = float(
            self.get_parameter('max_input_receipt_age_sec').value)
        self._require_unique_camera_source = bool(
            self.get_parameter('require_unique_camera_source').value)
        self._use_busbar_pnp = bool(
            self.get_parameter('use_busbar_pnp').value)
        self._rgb_topic = str(rgb_topic)
        self._depth_topic = str(depth_topic)
        self._camera_info_topic = str(camera_info_topic)

        if (
            not math.isfinite(self._max_rgb_depth_skew_sec)
            or self._max_rgb_depth_skew_sec <= 0.0
            or not math.isfinite(self._max_input_receipt_age_sec)
            or self._max_input_receipt_age_sec <= 0.0
        ):
            raise ValueError('RGB/depth skew와 input age 제한은 유한한 양수여야 합니다')

        self._bridge = CvBridge()
        self._detector = YoloPoseDetector(model_path)

        self._latest_rgb = None
        self._latest_rgb_header = None
        self._latest_rgb_received_wall = 0.0
        self._latest_depth = None
        self._latest_depth_header = None
        self._latest_depth_received_wall = 0.0
        self._last_processed_rgb_stamp_ns = None
        self._camera_model = None
        self._camera_frame_id = None

        self._latest_by_label = {}
        self._recent_detections = collections.defaultdict(list)
        self._recent_bolt_a = []
        self._recent_bolt_b = []
        self._latest_bolt_pair = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._rgb_sub = self.create_subscription(
            Image, rgb_topic, self._on_rgb, sensor_qos)
        self._depth_sub = self.create_subscription(
            Image, depth_topic, self._on_depth, sensor_qos)
        self._camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self._on_camera_info, sensor_qos)

        self._detections_pub = self.create_publisher(
            Detection3DArray, '/perception/detections_3d', 10)

        self._debug_image_pub = None
        if publish_debug_image:
            self._debug_image_pub = self.create_publisher(Image, debug_image_topic, 10)

        self._grasp_pose_srv = self.create_service(
            GetGraspPose, '/perception/get_grasp_pose', self._handle_get_grasp_pose)
        self._bolt_pair_srv = self.create_service(
            GetBoltPair, '/perception/get_bolt_pair', self._handle_get_bolt_pair)

        self._busbar_grasp_pub = self.create_publisher(BusbarGrasp, '/vision/busbar_grasp', 10)
        self._nut_pose_pub = self.create_publisher(NutPose, '/vision/nut_pose', 10)

        # Isaac Sim 내장 rclpy(kit python, 3.11)는 시스템 콜론 워크스페이스에서 빌드한
        # fms_interfaces(시스템 python 3.10용으로 컴파일됨)를 import할 수 없다 - Python
        # 버전이 달라 typesupport .so가 ABI 호환이 안 됨(source setup.bash로도 해결 안 됨,
        # 내장 rclpy는 PYTHONPATH를 안 본다). isaacpjt/M0609/assembly_nut_physics.py처럼
        # Isaac Sim 프로세스 안에서 직접 구독해야 하는 곳을 위해, 표준 geometry_msgs만
        # 쓰는 병행 토픽을 같이 발행한다(execute_isaac.py가 이미 PoseStamped/Bool/Int32/
        # String을 문제없이 쓰는 걸로 표준 메시지는 내장 rclpy에도 있음이 확인됨).
        self._busbar_grasp_pose_pub = self.create_publisher(
            PoseStamped, '/vision/busbar_grasp_pose', 10)
        self._bolt_a_pose_pub = self.create_publisher(PoseStamped, '/vision/bolt_a_pose', 10)
        self._bolt_b_pose_pub = self.create_publisher(PoseStamped, '/vision/bolt_b_pose', 10)

        self._timer = self.create_timer(detection_period_sec, self._detect_and_publish)

        self.get_logger().info(
            f'perception_node started (model={model_path}, world_frame={self._world_frame})')

    def _on_rgb(self, msg: Image):
        self._latest_rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self._latest_rgb_header = msg.header
        self._latest_rgb_received_wall = time.monotonic()

    def _on_depth(self, msg: Image):
        self._latest_depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self._latest_depth_header = msg.header
        self._latest_depth_received_wall = time.monotonic()

    def _on_camera_info(self, msg: CameraInfo):
        self._camera_model = make_camera_model(msg)
        self._camera_frame_id = self._camera_frame_override or msg.header.frame_id

    @staticmethod
    def _stamp_ns(stamp):
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)
        if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
            raise ValueError(f'invalid ROS stamp: {sec}.{nanosec:09d}')
        return sec * 1_000_000_000 + nanosec

    def _detect_and_publish(self):
        if (
            self._latest_rgb is None
            or self._latest_rgb_header is None
            or self._latest_depth is None
            or self._latest_depth_header is None
            or self._camera_model is None
        ):
            return

        if self._require_unique_camera_source:
            source_counts = {
                self._rgb_topic: self.count_publishers(self._rgb_topic),
                self._depth_topic: self.count_publishers(self._depth_topic),
                self._camera_info_topic: self.count_publishers(
                    self._camera_info_topic),
            }
            invalid_sources = [
                f'{topic}={count}'
                for topic, count in source_counts.items()
                if count != 1
            ]
            if invalid_sources:
                self.get_logger().error(
                    'camera source publisher 수가 정확히 1이 아니어서 검출 중단: '
                    + ', '.join(invalid_sources),
                    throttle_duration_sec=3.0)
                return

        try:
            rgb_stamp_ns = self._stamp_ns(self._latest_rgb_header.stamp)
            depth_stamp_ns = self._stamp_ns(self._latest_depth_header.stamp)
        except ValueError as exc:
            self.get_logger().error(
                f'잘못된 camera source timestamp로 검출 중단: {exc}',
                throttle_duration_sec=3.0)
            return

        if self._last_processed_rgb_stamp_ns is not None:
            if rgb_stamp_ns == self._last_processed_rgb_stamp_ns:
                return
            if rgb_stamp_ns < self._last_processed_rgb_stamp_ns:
                self.get_logger().warn(
                    'RGB source timestamp가 역행해 현재 frame을 버리고 '
                    '새 시퀀스를 기다립니다',
                    throttle_duration_sec=3.0)
                self._last_processed_rgb_stamp_ns = None
                return

        now_wall = time.monotonic()
        rgb_age = now_wall - self._latest_rgb_received_wall
        depth_age = now_wall - self._latest_depth_received_wall
        if (
            rgb_age > self._max_input_receipt_age_sec
            or depth_age > self._max_input_receipt_age_sec
        ):
            self.get_logger().warn(
                f'camera input receipt age 초과: rgb={rgb_age:.3f}s, '
                f'depth={depth_age:.3f}s',
                throttle_duration_sec=3.0)
            return

        source_skew_sec = abs(rgb_stamp_ns - depth_stamp_ns) / 1.0e9
        if source_skew_sec > self._max_rgb_depth_skew_sec:
            self.get_logger().warn(
                f'RGB/depth source stamp 차이 {source_skew_sec:.3f}s가 '
                f'제한 {self._max_rgb_depth_skew_sec:.3f}s를 초과해 검출 중단',
                throttle_duration_sec=3.0)
            return

        rgb = self._latest_rgb
        depth = self._latest_depth
        header = self._latest_rgb_header
        self._last_processed_rgb_stamp_ns = rgb_stamp_ns

        array_msg = Detection3DArray()
        array_msg.header.stamp = header.stamp
        array_msg.header.frame_id = self._world_frame

        debug_image = rgb.copy() if self._debug_image_pub is not None else None
        best_this_tick = {}  # label -> (score, world_point), 이번 tick 안에서만 비교
        bolt_this_tick = []  # 'bolt'는 tick당 여러 개(보통 2개)일 수 있어 따로 전부 모음
        busbar_yaw_this_tick = None  # best_this_tick[busbar]가 갱신될 때만 같이 갱신

        # ROI 파라미터 로드
        use_bolt_roi = self.get_parameter('use_bolt_roi').value
        reference_width = int(
            self.get_parameter('bolt_roi_reference_width').value)
        reference_height = int(
            self.get_parameter('bolt_roi_reference_height').value)
        if reference_width <= 0 or reference_height <= 0:
            self.get_logger().error(
                'bolt ROI reference 크기는 양수여야 합니다',
                throttle_duration_sec=3.0)
            return
        image_height, image_width = rgb.shape[:2]
        scale_x = image_width / reference_width
        scale_y = image_height / reference_height
        bolt_x_min = round(
            self.get_parameter('bolt_roi_x_min').value * scale_x)
        bolt_x_max = round(
            self.get_parameter('bolt_roi_x_max').value * scale_x)
        bolt_y_min = round(
            self.get_parameter('bolt_roi_y_min').value * scale_y)
        bolt_y_max = round(
            self.get_parameter('bolt_roi_y_max').value * scale_y)

        # 🔥 rqt_image_view 시각화를 위한 ROI 박스 그리기
        if debug_image is not None and use_bolt_roi:
            cv2.rectangle(
                debug_image,
                (bolt_x_min, bolt_y_min),
                (bolt_x_max, bolt_y_max),
                (255, 0, 0),
                2
            )
            cv2.putText(
                debug_image,
                'BOLT ROI',
                (bolt_x_min + 5, bolt_y_min + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        for det in self._detector.detect(rgb, self._conf_threshold, self._iou_threshold):
            label = det['label']

            # 🔥 'bolt' 라벨인 경우에만 ROI 범위 검사 수행
            if use_bolt_roi and label == BOLT_LABEL:
                px_x, px_y = det['pixel']
                if not (bolt_x_min <= px_x <= bolt_x_max and bolt_y_min <= px_y <= bolt_y_max):
                    continue  # 배터리 스캔 영역 밖의 볼트는 스킵

            if label == BUSBAR_LABEL:
                world_point, yaw_rad, status = self._resolve_busbar_pose(det, depth, header.stamp)
                camera_point = None  # PnP 경로는 단일 카메라 포인트 개념이 없어 오버레이만 생략
            else:
                camera_point, world_point, status = self._transform_pixel(
                    det['pixel'], depth, header.stamp)
                yaw_rad = None

            if debug_image is not None:
                overlay.draw_detection(debug_image, det, camera_point, world_point, status)

            if world_point is not None:
                array_msg.detections.append(
                    self._make_detection3d(det, world_point, header.stamp))
                score = det['score']
                if label not in best_this_tick or score > best_this_tick[label][0]:
                    best_this_tick[label] = (score, world_point)
                    if label == BUSBAR_LABEL:
                        busbar_yaw_this_tick = yaw_rad
                if label == BOLT_LABEL:
                    bolt_this_tick.append((score, world_point))

        now = self.get_clock().now()
        for label, (score, world_point) in best_this_tick.items():
            if label == BOLT_LABEL:
                continue
            mean_point, _n = self._update_smoothed(label, world_point, now)
            self._latest_by_label[label] = (score, mean_point, now)
            yaw_rad = busbar_yaw_this_tick if label == BUSBAR_LABEL else None
            self._publish_vision_topic(label, mean_point, header.stamp, yaw_rad)

        self._update_bolt_pair(bolt_this_tick, now)

        self._detections_pub.publish(array_msg)

        if debug_image is not None:
            debug_msg = self._bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            debug_msg.header = header
            self._debug_image_pub.publish(debug_msg)

    def _publish_vision_topic(self, label, world_point, stamp, yaw_rad=None):
        """behavior_node/arm_node가 구독하는 /vision/* 토픽에 이번 tick 최고점 검출을
        실어 보낸다. busbar이고 yaw_rad가 주어지면(Stage3 PnP 성공) Z축 기준 실제
        orientation을 채운다 — top-down 접근이라 azimuth(yaw)만 의미 있고 roll/pitch는
        다루지 않는다. bolt/nut은 keypoint가 위치(top_center) 하나뿐이라 orientation
        정보가 없으므로 항상 identity. yaw_rad는 위치(mean_point)와 달리 롤링 평균 없이
        이번 tick 값을 그대로 쓴다(Phase3 스코프 — 오차가 크면 후속에서 스칼라 평균 추가)."""
        wx, wy, wz = world_point
        if label == BUSBAR_LABEL:
            msg = BusbarGrasp()
            msg.pose.header.stamp = stamp
            msg.pose.header.frame_id = self._world_frame
            msg.pose.pose.position.x = wx
            msg.pose.pose.position.y = wy
            msg.pose.pose.position.z = wz
            if yaw_rad is not None:
                msg.pose.pose.orientation.z = math.sin(yaw_rad / 2.0)
                msg.pose.pose.orientation.w = math.cos(yaw_rad / 2.0)
            else:
                msg.pose.pose.orientation.w = 1.0
            self._busbar_grasp_pub.publish(msg)
            self._busbar_grasp_pose_pub.publish(msg.pose)
        elif label == NUT_LABEL:
            msg = NutPose()
            msg.id = 0
            msg.pose.header.stamp = stamp
            msg.pose.header.frame_id = self._world_frame
            msg.pose.pose.position.x = wx
            msg.pose.pose.position.y = wy
            msg.pose.pose.position.z = wz
            msg.pose.pose.orientation.w = 1.0
            self._nut_pose_pub.publish(msg)

    def _smooth_bucket(self, bucket: list, world_point, now) -> tuple:
        bucket.append((np.asarray(world_point, dtype=float), now))
        cutoff = now - Duration(seconds=SMOOTHING_WINDOW_SEC)
        while bucket and bucket[0][1] < cutoff:
            bucket.pop(0)

        positions = np.array([p for p, _ in bucket])
        mean = positions.mean(axis=0)
        std = positions.std(axis=0) if len(bucket) > 1 else np.zeros(3)
        return mean, std, len(bucket)

    def _update_smoothed(self, label, world_point, now) -> tuple:
        """라벨 하나의 롤링 평균/표준편차를 갱신한다. 반환값(mean_point, sample_count)을
        캐시/토픽 발행에 그대로 쓴다 — 단발성 YOLO+depth tick 값을 곧바로 신뢰하지
        않기 위함. 이 값 자체가 "검출 결과"이자 다른 노드가 실제로 받게 될 값이므로
        로그는 매 tick(기본 0.5초)마다가 아니라 throttle로 주기적으로만 남긴다 —
        detection_period_sec가 짧아도 터미널이 같은 정보로 도배되지 않게."""
        bucket = self._recent_detections[label]
        mean, std, n = self._smooth_bucket(bucket, world_point, now)
        std_mm = std * 1000.0

        self.get_logger().info(
            f"[{label}] raw=({world_point[0]:.4f},{world_point[1]:.4f},{world_point[2]:.4f}) "
            f"mean=({mean[0]:.4f},{mean[1]:.4f},{mean[2]:.4f}) "
            f"std=({std_mm[0]:.1f},{std_mm[1]:.1f},{std_mm[2]:.1f})mm n={n}",
            throttle_duration_sec=3.0)
        if std_mm.max() > SMOOTHING_STD_WARN_MM:
            self.get_logger().warn(
                f"[{label}] 검출 흔들림 큼 (표준편차 최대 {std_mm.max():.1f}mm > "
                f"{SMOOTHING_STD_WARN_MM:.0f}mm) — 조명/거리/각도 또는 conf_threshold 확인 필요",
                throttle_duration_sec=3.0
            )
        return tuple(mean), n

    def _update_bolt_pair(self, bolt_candidates, now):
        if len(bolt_candidates) < 2:
            return

        # 점수 기준 정렬에서 화면 중앙(카메라 축) 근처 우선 정렬로 보정
        bolt_candidates = sorted(
            bolt_candidates,
            key=lambda c: np.hypot(c[1][0], c[1][1])
        )[:2]

        points = [np.asarray(wp, dtype=float) for _, wp in bolt_candidates]

        prev_a = self._recent_bolt_a[-1][0] if self._recent_bolt_a else None
        prev_b = self._recent_bolt_b[-1][0] if self._recent_bolt_b else None
        if prev_a is not None and prev_b is not None:
            d_same = np.linalg.norm(points[0] - prev_a) + np.linalg.norm(points[1] - prev_b)
            d_cross = np.linalg.norm(points[0] - prev_b) + np.linalg.norm(points[1] - prev_a)
            if d_cross < d_same:
                points = [points[1], points[0]]

        mean_a, std_a, n_a = self._smooth_bucket(self._recent_bolt_a, points[0], now)
        mean_b, std_b, n_b = self._smooth_bucket(self._recent_bolt_b, points[1], now)
        mid_xy = ((mean_a[0] + mean_b[0]) / 2, (mean_a[1] + mean_b[1]) / 2)

        # get_bolt_pair 서비스가 그대로 반환하는 값이라 "검출 결과" 로그이지만, 이것도
        # 매 tick 찍으면 스팸이므로 위 _update_smoothed와 동일하게 throttle.
        self.get_logger().info(
            f"[bolt-pair] A mean=({mean_a[0]:.4f},{mean_a[1]:.4f},{mean_a[2]:.4f}) "
            f"std_max={std_a.max()*1000:.1f}mm n={n_a} | "
            f"B mean=({mean_b[0]:.4f},{mean_b[1]:.4f},{mean_b[2]:.4f}) "
            f"std_max={std_b.max()*1000:.1f}mm n={n_b} | mid_xy=({mid_xy[0]:.4f},{mid_xy[1]:.4f})",
            throttle_duration_sec=3.0
        )
        self._latest_bolt_pair = (tuple(mean_a), tuple(mean_b), now, min(n_a, n_b))
        # fms_interfaces 없이도 쓸 수 있는 표준 geometry_msgs 병행 발행(위 publisher
        # 생성부 주석 참고) - assembly_nut_physics.py 같은 Isaac Sim 내장 rclpy
        # 프로세스가 이 두 토픽을 직접 구독한다.
        self._bolt_a_pose_pub.publish(self._make_pose_stamped(mean_a, now))
        self._bolt_b_pose_pub.publish(self._make_pose_stamped(mean_b, now))

    def _make_pose_stamped(self, world_point, stamp) -> PoseStamped:
        wx, wy, wz = world_point
        pose = PoseStamped()
        pose.header.frame_id = self._world_frame
        pose.header.stamp = stamp.to_msg()
        pose.pose.position.x = float(wx)
        pose.pose.position.y = float(wy)
        pose.pose.position.z = float(wz)
        pose.pose.orientation.w = 1.0
        return pose

    def _log_tf_error(self, ex):
        self.get_logger().warn(
            f'{self._world_frame} <- {self._camera_frame_id} tf 조회 실패, 검출 skip: {ex}',
            throttle_duration_sec=5.0)

    def _transform_pixel(self, pixel_uv, depth_image, stamp):
        return transform_pixel_to_world(
            self._camera_model, depth_image, pixel_uv, self._tf_buffer,
            self._world_frame, self._camera_frame_id, stamp, on_tf_error=self._log_tf_error)

    def _resolve_busbar_pose(self, det, depth_image, stamp):
        """busbar 검출 1건 -> (world_point, yaw_rad, status).

        Stage3 이중경로: (A) 6-keypoint PnP(camera_geometry.busbar_pnp_world_pose),
        (B) hole_A/hole_B 픽셀 각각의 depth 역투영. (A)가 성공하면 그 mid_xy/yaw를
        쓰고 (B)와 비교해 BUSBAR_DUAL_PATH_GATE_M을 넘으면 경고 로그를 남긴다 —
        실제 접근 중단은 이번 스코프 밖(arm_node/behavior_node의 후속 과제)이라
        지금은 안전장치로 로그만 남긴다. (A)가 실패하면(가림/반사 등) 기존 방식대로
        6-keypoint 중심 픽셀의 depth 역투영으로 폴백한다(orientation 없이).
        """
        if not self._use_busbar_pnp:
            _, world_point, status = self._transform_pixel(
                det['pixel'], depth_image, stamp)
            return world_point, None, status

        pnp = busbar_pnp_world_pose(
            self._camera_model, det['keypoints_px'], self._tf_buffer,
            self._world_frame, self._camera_frame_id, stamp, on_tf_error=self._log_tf_error)

        if pnp['status'] == '':
            hole_a_px = tuple(det['keypoints_px'][BUSBAR_KEYPOINT_ORDER.index('hole_A')])
            hole_b_px = tuple(det['keypoints_px'][BUSBAR_KEYPOINT_ORDER.index('hole_B')])
            _, hole_a_depth_world, _status_a = self._transform_pixel(hole_a_px, depth_image, stamp)
            _, hole_b_depth_world, _status_b = self._transform_pixel(hole_b_px, depth_image, stamp)
            if hole_a_depth_world is not None and hole_b_depth_world is not None:
                depth_mid_xy = (
                    (hole_a_depth_world[0] + hole_b_depth_world[0]) / 2.0,
                    (hole_a_depth_world[1] + hole_b_depth_world[1]) / 2.0,
                )
                discrepancy_m = dual_path_discrepancy_m(pnp['mid_xy'], depth_mid_xy)
                msg = (f"[busbar] PnP-depth 이중경로 불일치={discrepancy_m*1000:.2f}mm "
                       f"(게이트 {BUSBAR_DUAL_PATH_GATE_M*1000:.0f}mm), "
                       f"reproj_err={pnp['reprojection_error_px']:.2f}px")
                # 같은 호출 지점(소스 라인)에서 throttle_duration_sec를 쓰면 rclpy가 내부적으로
                # 호출 지점별 상태를 캐싱하는데, 그 상태에 severity도 같이 묶여있어 매번
                # warn/info를 오갈 경우 "Logger severity cannot be changed between calls"로
                # 죽는다(실측, 2026-07-25). 그래서 호출 지점을 아예 분리해야 한다.
                if discrepancy_m > BUSBAR_DUAL_PATH_GATE_M:
                    self.get_logger().warn(msg, throttle_duration_sec=3.0)
                else:
                    self.get_logger().info(msg, throttle_duration_sec=3.0)

            za = pnp['hole_world']['hole_A'][2]
            zb = pnp['hole_world']['hole_B'][2]
            world_point = (pnp['mid_xy'][0], pnp['mid_xy'][1], (za + zb) / 2.0)
            return world_point, pnp['yaw_rad'], ''

        _, world_point, status = self._transform_pixel(det['pixel'], depth_image, stamp)
        return world_point, None, status

    def _make_detection3d(self, det, world_point, stamp) -> Detection3D:
        detection = Detection3D()
        detection.header.stamp = stamp
        detection.header.frame_id = self._world_frame

        wx, wy, wz = world_point
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = det['label']
        hypothesis.hypothesis.score = det['score']
        hypothesis.pose.pose.position.x = wx
        hypothesis.pose.pose.position.y = wy
        hypothesis.pose.pose.position.z = wz
        hypothesis.pose.pose.orientation.w = 1.0
        detection.results.append(hypothesis)

        detection.bbox.center.position.x = wx
        detection.bbox.center.position.y = wy
        detection.bbox.center.position.z = wz
        detection.bbox.center.orientation.w = 1.0
        x0, y0, x1, y1 = det['bbox_px']
        detection.bbox.size.x = float(x1 - x0)
        detection.bbox.size.y = float(y1 - y0)
        detection.bbox.size.z = 0.0

        return detection

    def _handle_get_grasp_pose(self, request, response):
        cached = self._latest_by_label.get(request.label)
        if cached is None:
            response.found = False
            response.message = f"'{request.label}' 라벨 검출 캐시 없음"
            return response

        _score, world_point, stamp = cached
        age_sec = (self.get_clock().now() - stamp).nanoseconds / 1e9
        if age_sec > self._grasp_query_max_age_sec:
            response.found = False
            response.message = f"'{request.label}' 캐시가 {age_sec:.1f}s 전 데이터로 오래됨"
            return response

        response.found = True
        response.pose = self._make_pose_stamped(world_point, stamp)
        response.message = f"'{request.label}' 검출 좌표 반환 ({age_sec:.2f}s 전)"
        return response

    def _handle_get_bolt_pair(self, request, response):
        if self._latest_bolt_pair is None:
            response.found = False
            response.message = "'bolt' 쌍 검출 캐시 없음"
            return response

        mean_a, mean_b, stamp, sample_count = self._latest_bolt_pair
        age_sec = (self.get_clock().now() - stamp).nanoseconds / 1e9
        if age_sec > self._grasp_query_max_age_sec:
            response.found = False
            response.message = f"'bolt' 쌍 캐시가 {age_sec:.1f}s 전 데이터로 오래됨"
            return response
        if sample_count < SMOOTHING_MIN_SAMPLES:
            response.found = False
            response.message = f"'bolt' 쌍 표본 수 부족 (n={sample_count} < {SMOOTHING_MIN_SAMPLES})"
            return response

        response.found = True
        response.pose_a = self._make_pose_stamped(mean_a, stamp)
        response.pose_b = self._make_pose_stamped(mean_b, stamp)
        response.message = f"'bolt' 쌍 반환 ({age_sec:.2f}s 전, n={sample_count})"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            try:
                node.destroy_node()
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()

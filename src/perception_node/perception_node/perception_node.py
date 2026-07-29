"""
YOLO로 bolt, nut, busbar를 검출해 world 좌표와 ROS 인터페이스로 제공한다.

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
SRV /perception/get_bolt_pair  (fms_interfaces/GetBoltPair) — confidence 0.6 이상 'bolt'
                                 검출의 모든 조합 중 실제 배터리팩 볼트 간격과 맞고, 쌍의
                                 ordered bolt_2가 화면 중앙에 가장 가까운 한 쌍을 고른다
                                 (고정캠도 현재 station bolt_2 위로 이동). 쌍 중점 거리는
                                 동률 보정에 쓰며,
                                 이전 tick 위치와 최근접 매칭해 A/B 각각 별도로 롤링 평균한
                                 값을 반환한다.
PUB /vision/busbar_grasp        (fms_interfaces/BusbarGrasp) — 'busbar' 라벨이 이번 tick에
                                 검출됐을 때만 (롤링 평균) world 좌표로 발행. behavior_node가
                                 구독해 busbar_insert 액션 goal.target_pose로 그대로 실어 보낸다.
PUB /vision/nut_pose            (fms_interfaces/NutPose) — 'nut' 라벨 검출 시 (롤링 평균) 발행.
                                 id는 항상 0. reset 뒤 첫 표본은 화면 중앙 우선으로
                                 선택하고, 이후 표본은 world 좌표 anchor에 고정해 이웃
                                 너트와 섞이지 않도록 추적한다.

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
import threading
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from fms_interfaces.msg import BusbarGrasp, NutPose
from fms_interfaces.srv import GetBoltPair, GetGraspPose
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup,
)
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Empty
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from perception_node import overlay
from perception_node.camera_geometry import (
    busbar_pnp_world_pose,
    make_camera_model,
    select_busbar_world_point,
    transform_pixel_to_world,
)
from perception_node.detector import (
    BUSBAR_KEYPOINT_ORDER,
    MIN_BUSBAR_KEYPOINT_CONFIDENCE,
    YoloPoseDetector,
    busbar_keypoints_are_reliable,
)
from perception_node.multi_camera_config import DEFAULT_MODEL_FILENAME
from perception_node.perception_policy import (
    BoltPairAssociationGate,
    CameraFrameGate,
    DEFAULT_BOLT_PAIR_DX_M,
    DEFAULT_BOLT_PAIR_DY_M,
    DEFAULT_BOLT_PAIR_MAX_Z_DELTA_M,
    DEFAULT_BOLT_PAIR_X_TOLERANCE_M,
    DEFAULT_BOLT_PAIR_Y_TOLERANCE_M,
    SingleTargetAssociationGate,
    SourceTimestampEpochTracker,
    TargetFrameBarrier,
    TimestampedFrame,
    advance_source_boundary,
    frame_receipt_error,
    scale_roi,
    select_central_bolt_pair,
    select_central_candidates,
    select_newest_synchronized_pair,
    select_sticky_single_target,
    stamp_to_nanoseconds,
)

DEFAULT_MODEL_PATH = os.path.join(
    get_package_share_directory('perception_node'), 'models', DEFAULT_MODEL_FILENAME)

BUSBAR_LABEL = 'busbar'
NUT_LABEL = 'nut'
BOLT_LABEL = 'bolt'

# 롤링 평균/오차 분포 확인용 시간 창. 이 안에 든 표본만 평균·표준편차 계산에 쓴다.
SMOOTHING_WINDOW_SEC = 2.0
SMOOTHING_MIN_SAMPLES = 2
SMOOTHING_STD_WARN_MM = 5.0
FRAME_PAIR_QUEUE_SIZE = 16


class PerceptionNode(Node):

    def __init__(self):
        super().__init__('perception_node')
        # Keep sensor delivery and reset/service control schedulable while a
        # GPU inference is running. The timer gets an exclusive group plus a
        # defensive lock so detector calls can never overlap.
        self._sensor_callback_group = ReentrantCallbackGroup()
        self._control_callback_group = ReentrantCallbackGroup()
        self._inference_callback_group = MutuallyExclusiveCallbackGroup()
        self._inference_lock = threading.Lock()

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
        # Isaac RGB/depth delivery can be offset by a little over one render
        # tick.  Keep a bounded 200 ms pair window while the independent
        # freshness, monotonic-stamp, and single-publisher gates stay strict.
        self.declare_parameter('max_rgb_depth_skew_sec', 0.2)
        self.declare_parameter('max_input_receipt_age_sec', 1.0)
        self.declare_parameter('require_unique_camera_source', True)
        self.declare_parameter('use_busbar_pnp', True)
        self.declare_parameter('prefer_busbar_depth', False)
        self.declare_parameter('max_busbar_pnp_reprojection_px', 20.0)
        self.declare_parameter('required_target_frames', 3)
        # A camera configured as on-demand keeps its subscriptions/services
        # alive, but does not run YOLO until reset_cache or an explicit pose
        # service request opens one capture.  The wrist camera uses this in
        # the fixed-camera-first assembly workflow.
        self.declare_parameter('on_demand_inference', False)
        # 0 preserves the checkpoint's default.  The bolt-camera launch
        # contract selects 640 explicitly; wrist/busbar remain unchanged.
        self.declare_parameter('inference_image_size', 0)

        # 🔥 bolt 라벨 스캔 시에만 적용할 ROI 영역 파라미터 (640x640 기준)
        self.declare_parameter('use_bolt_roi', True)
        self.declare_parameter('bolt_roi_x_min', 150)
        self.declare_parameter('bolt_roi_x_max', 560)
        self.declare_parameter('bolt_roi_y_min', 180)
        self.declare_parameter('bolt_roi_y_max', 450)
        self.declare_parameter('bolt_roi_reference_width', 640)
        self.declare_parameter('bolt_roi_reference_height', 640)
        self.declare_parameter(
            'bolt_pair_expected_dx_m',
            DEFAULT_BOLT_PAIR_DX_M,
        )
        self.declare_parameter(
            'bolt_pair_expected_dy_m',
            DEFAULT_BOLT_PAIR_DY_M,
        )
        self.declare_parameter(
            'bolt_pair_x_tolerance_m',
            DEFAULT_BOLT_PAIR_X_TOLERANCE_M,
        )
        self.declare_parameter(
            'bolt_pair_y_tolerance_m',
            DEFAULT_BOLT_PAIR_Y_TOLERANCE_M,
        )
        self.declare_parameter(
            'bolt_pair_max_z_delta_m',
            DEFAULT_BOLT_PAIR_MAX_Z_DELTA_M,
        )

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
        max_rgb_depth_skew_sec = float(
            self.get_parameter('max_rgb_depth_skew_sec').value)
        max_input_receipt_age_sec = float(
            self.get_parameter('max_input_receipt_age_sec').value)
        self._max_rgb_depth_skew_sec = max_rgb_depth_skew_sec
        self._max_input_receipt_age_sec = max_input_receipt_age_sec
        self._require_unique_camera_source = bool(
            self.get_parameter('require_unique_camera_source').value)
        self._use_busbar_pnp = bool(
            self.get_parameter('use_busbar_pnp').value)
        self._prefer_busbar_depth = bool(
            self.get_parameter('prefer_busbar_depth').value)
        self._max_busbar_pnp_reprojection_px = float(
            self.get_parameter(
                'max_busbar_pnp_reprojection_px').value)
        required_target_frames = self.get_parameter('required_target_frames').value
        self._on_demand_inference = bool(
            self.get_parameter('on_demand_inference').value)
        inference_image_size = int(
            self.get_parameter('inference_image_size').value)
        self._bolt_pair_expected_delta_xy = (
            float(self.get_parameter('bolt_pair_expected_dx_m').value),
            float(self.get_parameter('bolt_pair_expected_dy_m').value),
        )
        self._bolt_pair_x_tolerance_m = float(
            self.get_parameter('bolt_pair_x_tolerance_m').value)
        self._bolt_pair_y_tolerance_m = float(
            self.get_parameter('bolt_pair_y_tolerance_m').value)
        self._bolt_pair_max_z_delta_m = float(
            self.get_parameter('bolt_pair_max_z_delta_m').value)
        self._rgb_topic = str(rgb_topic)
        self._depth_topic = str(depth_topic)
        self._camera_info_topic = str(camera_info_topic)

        if (
            not math.isfinite(self._grasp_query_max_age_sec)
            or self._grasp_query_max_age_sec <= 0.0
        ):
            raise ValueError('grasp_query_max_age_sec must be a finite positive number')
        if (
            not math.isfinite(self._max_busbar_pnp_reprojection_px)
            or self._max_busbar_pnp_reprojection_px <= 0.0
        ):
            raise ValueError(
                'max_busbar_pnp_reprojection_px must be finite and positive')
        if inference_image_size != 0 and inference_image_size < 32:
            raise ValueError(
                'inference_image_size must be 0 or an integer >= 32')
        expected_dx, expected_dy = self._bolt_pair_expected_delta_xy
        if (
            not math.isfinite(expected_dx)
            or not math.isfinite(expected_dy)
            or math.hypot(expected_dx, expected_dy) <= 0.0
        ):
            raise ValueError(
                'bolt_pair_expected_dx_m/dy_m must form a finite non-zero vector')
        if (
            not math.isfinite(self._bolt_pair_x_tolerance_m)
            or self._bolt_pair_x_tolerance_m <= 0.0
        ):
            raise ValueError(
                'bolt_pair_x_tolerance_m must be finite and positive')
        if (
            not math.isfinite(self._bolt_pair_y_tolerance_m)
            or self._bolt_pair_y_tolerance_m <= 0.0
        ):
            raise ValueError(
                'bolt_pair_y_tolerance_m must be finite and positive')
        if (
            not math.isfinite(self._bolt_pair_max_z_delta_m)
            or self._bolt_pair_max_z_delta_m < 0.0
        ):
            raise ValueError(
                'bolt_pair_max_z_delta_m must be finite and non-negative')

        self._bridge = CvBridge()
        self._detector = YoloPoseDetector(
            model_path,
            inference_image_size=(
                inference_image_size if inference_image_size else None
            ),
        )
        self._frame_gate = CameraFrameGate(
            max_skew_sec=max_rgb_depth_skew_sec,
            max_receipt_age_sec=max_input_receipt_age_sec,
        )
        self._source_epoch_tracker = SourceTimestampEpochTracker(
            max_pair_skew_sec=max_rgb_depth_skew_sec,
        )
        self._target_barrier = TargetFrameBarrier(required_target_frames)
        self._bolt_pair_association = BoltPairAssociationGate()
        self._nut_association = SingleTargetAssociationGate()
        # Keep reset, inference commit, and service cache reads atomic under
        # the multi-threaded executor. Generation fences also prevent work
        # started before reset from repopulating a cleared cache.
        self._state_lock = threading.RLock()

        # RGB and depth callbacks are independent.  Keep a small bounded
        # history so inference selects an actual timestamp-matched pair
        # instead of accidentally combining each topic's unrelated latest
        # frame after a slow GPU inference callback.
        self._rgb_frames = collections.deque(maxlen=FRAME_PAIR_QUEUE_SIZE)
        self._depth_frames = collections.deque(maxlen=FRAME_PAIR_QUEUE_SIZE)
        self._last_observed_rgb_stamp_ns = None
        self._last_observed_depth_stamp_ns = None
        self._reset_monotonic_cutoff = float('-inf')
        self._camera_model = None
        self._camera_frame_id = None

        self._latest_by_label = {}
        self._latest_target_received_at = {}
        self._target_present_latest = {}
        self._recent_detections = collections.defaultdict(list)
        self._recent_bolt_a = []
        self._recent_bolt_b = []
        self._latest_bolt_pair = None
        # A bolt pair is an explicit, short-lived query path.  Keeping it
        # dormant until GetBoltPair is requested prevents the three camera
        # instances from continuously trying to associate every incidental
        # bolt in the scene (and from emitting misleading same-pack warnings)
        # while the fixed-camera assembly flow is using its own prim-based
        # bolt reference.  The epoch fences an inference that started before
        # a request/reset/completed response from mutating the new capture.
        self._bolt_pair_tracking_active = False
        self._bolt_pair_capture_epoch = 0
        # Continuous cameras preserve their existing behavior.  An on-demand
        # camera starts idle and gets a new epoch for each reset/request so an
        # inference already in flight cannot repopulate an ended capture.
        self._demand_capture_active = not self._on_demand_inference
        self._demand_capture_epoch = 0
        self._consecutive_stale_inferences = 0
        self._total_stale_inferences = 0
        self._last_inference_latency_sec = None
        self._last_stale_inference_reason = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=FRAME_PAIR_QUEUE_SIZE,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._rgb_sub = self.create_subscription(
            Image,
            rgb_topic,
            self._on_rgb,
            sensor_qos,
            callback_group=self._sensor_callback_group,
        )
        self._depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self._on_depth,
            sensor_qos,
            callback_group=self._sensor_callback_group,
        )
        self._camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self._on_camera_info,
            sensor_qos,
            callback_group=self._sensor_callback_group,
        )

        self._detections_pub = self.create_publisher(
            Detection3DArray, '/perception/detections_3d', 10)

        self._debug_image_pub = None
        if publish_debug_image:
            self._debug_image_pub = self.create_publisher(Image, debug_image_topic, 10)

        self._grasp_pose_srv = self.create_service(
            GetGraspPose,
            '/perception/get_grasp_pose',
            self._handle_get_grasp_pose,
            callback_group=self._control_callback_group,
        )
        self._bolt_pair_srv = self.create_service(
            GetBoltPair,
            '/perception/get_bolt_pair',
            self._handle_get_bolt_pair,
            callback_group=self._control_callback_group,
        )
        self._reset_sub = self.create_subscription(
            Empty,
            '/perception/reset_cache',
            self._on_reset_cache,
            10,
            callback_group=self._control_callback_group,
        )
        self._reset_ack_pub = self.create_publisher(
            Empty, '/perception/reset_ack', 10)

        self._busbar_grasp_pub = self.create_publisher(BusbarGrasp, '/vision/busbar_grasp', 10)
        self._nut_pose_pub = self.create_publisher(NutPose, '/vision/nut_pose', 10)

        self._timer = self.create_timer(
            detection_period_sec,
            self._detect_and_publish,
            callback_group=self._inference_callback_group,
        )

        self.get_logger().info(
            f'perception_node started (model={model_path}, world_frame={self._world_frame}, '
            f'required_target_frames={self._target_barrier.required_frames}, '
            f"inference_mode={'on_demand' if self._on_demand_inference else 'continuous'})")

    def _on_rgb(self, msg: Image):
        received_at = time.monotonic()
        with self._state_lock:
            if received_at <= self._reset_monotonic_cutoff:
                return
            callback_generation = self._target_barrier.generation
        try:
            stamp_ns = stamp_to_nanoseconds(msg.header.stamp)
        except ValueError as exc:
            self.get_logger().error(
                f'invalid RGB source timestamp: {exc}',
                throttle_duration_sec=3.0)
            return
        with self._state_lock:
            if self._source_epoch_tracker.stamp_is_quarantined(
                'RGB',
                stamp_ns,
            ):
                self.get_logger().warn(
                    'RGB callback from delayed old source epoch quarantined',
                    throttle_duration_sec=2.0,
                )
                return
        rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self._state_lock:
            if (
                received_at <= self._reset_monotonic_cutoff
                or not self._target_barrier.is_current(
                    callback_generation
                )
            ):
                return
            self._rgb_frames.append(TimestampedFrame(
                stamp_ns=stamp_ns,
                received_at=received_at,
                payload=(rgb, msg.header),
            ))
            self._last_observed_rgb_stamp_ns = advance_source_boundary(
                self._last_observed_rgb_stamp_ns,
                stamp_ns,
            )

    def _on_depth(self, msg: Image):
        received_at = time.monotonic()
        with self._state_lock:
            if received_at <= self._reset_monotonic_cutoff:
                return
            callback_generation = self._target_barrier.generation
        try:
            stamp_ns = stamp_to_nanoseconds(msg.header.stamp)
        except ValueError as exc:
            self.get_logger().error(
                f'invalid depth source timestamp: {exc}',
                throttle_duration_sec=3.0)
            return
        with self._state_lock:
            if self._source_epoch_tracker.stamp_is_quarantined(
                'depth',
                stamp_ns,
            ):
                self.get_logger().warn(
                    'depth callback from delayed old source epoch quarantined',
                    throttle_duration_sec=2.0,
                )
                return
        depth = self._bridge.imgmsg_to_cv2(
            msg, desired_encoding='passthrough')
        with self._state_lock:
            if (
                received_at <= self._reset_monotonic_cutoff
                or not self._target_barrier.is_current(
                    callback_generation
                )
            ):
                return
            self._depth_frames.append(TimestampedFrame(
                stamp_ns=stamp_ns,
                received_at=received_at,
                payload=(depth, msg.header),
            ))
            self._last_observed_depth_stamp_ns = advance_source_boundary(
                self._last_observed_depth_stamp_ns,
                stamp_ns,
            )

    def _on_camera_info(self, msg: CameraInfo):
        camera_model = make_camera_model(msg)
        camera_frame_id = (
            self._camera_frame_override or msg.header.frame_id)
        with self._state_lock:
            self._camera_model = camera_model
            self._camera_frame_id = camera_frame_id

    def _on_reset_cache(self, _msg: Empty):
        """Discard pre-move images and detections while retaining anti-replay stamps."""
        with self._state_lock:
            # Callbacks capture their local receipt time at entry. This cutoff
            # rejects a callback that started before reset but decoded only
            # after the reset lock became available.
            self._reset_monotonic_cutoff = time.monotonic()
            self._rgb_frames.clear()
            self._depth_frames.clear()
            self._latest_by_label.clear()
            self._latest_target_received_at.clear()
            self._target_present_latest.clear()
            self._recent_detections.clear()
            self._recent_bolt_a.clear()
            self._recent_bolt_b.clear()
            self._latest_bolt_pair = None
            PerceptionNode._cancel_bolt_pair_tracking_locked(self)
            self._bolt_pair_association.reset()
            self._nut_association.reset()
            self._target_barrier.reset()
            tracker_seeded = self._source_epoch_tracker.seed_reset_boundaries(
                self._last_observed_rgb_stamp_ns,
                self._last_observed_depth_stamp_ns,
            )
            self._frame_gate.reset(
                preserve_source_sequence=True,
                rgb_boundary_ns=self._last_observed_rgb_stamp_ns,
                depth_boundary_ns=self._last_observed_depth_stamp_ns,
            )
            self._consecutive_stale_inferences = 0
            self._last_inference_latency_sec = None
            self._last_stale_inference_reason = None
            # A reset is the normal explicit boundary before legacy wrist
            # busbar/nut scans.  It both clears old observations and starts
            # one fresh on-demand capture for that scan.
            PerceptionNode._arm_on_demand_inference_locked(
                self,
                restart=True,
            )
            generation = self._target_barrier.generation
        self._reset_ack_pub.publish(Empty())
        self.get_logger().info(
            'perception cache reset '
            f'(generation={generation}, '
            f'epoch_tracker_seeded={tracker_seeded}); '
            'waiting for three new target frames')

    def _arm_on_demand_inference_locked(self, *, restart: bool = False) -> bool:
        """Open one on-demand detector capture without changing ROS names.

        Only the wrist configuration opts into this behavior.  A normal
        camera remains continuous, while a reset or first explicit service
        request wakes an idle camera.  ``restart`` creates a new fence for a
        reset even if a prior request was still being processed.
        """
        if not getattr(self, '_on_demand_inference', False):
            return False
        if (
            getattr(self, '_demand_capture_active', False)
            and not restart
        ):
            return False
        self._demand_capture_active = True
        self._demand_capture_epoch = (
            int(getattr(self, '_demand_capture_epoch', 0)) + 1
        )
        return True

    def _finish_on_demand_inference_locked(self) -> None:
        """Return a completed on-demand capture to silent idle state."""
        if not getattr(self, '_on_demand_inference', False):
            return
        self._demand_capture_active = False
        # Fence detector work that started before the caller received its
        # stable snapshot.  The next reset/request opens a new epoch.
        self._demand_capture_epoch = (
            int(getattr(self, '_demand_capture_epoch', 0)) + 1
        )

    def _hard_reset_source_epoch_locked(self, epoch: int) -> None:
        """Atomically discard every boundary tied to a restarted source clock."""
        self._reset_monotonic_cutoff = time.monotonic()
        self._rgb_frames.clear()
        self._depth_frames.clear()
        self._last_observed_rgb_stamp_ns = None
        self._last_observed_depth_stamp_ns = None
        self._latest_by_label.clear()
        self._latest_target_received_at.clear()
        self._target_present_latest.clear()
        self._recent_detections.clear()
        self._recent_bolt_a.clear()
        self._recent_bolt_b.clear()
        self._latest_bolt_pair = None
        PerceptionNode._cancel_bolt_pair_tracking_locked(self)
        self._bolt_pair_association.reset()
        self._nut_association.reset()
        self._target_barrier.reset()
        self._frame_gate.reset(preserve_source_sequence=False)
        self._consecutive_stale_inferences = 0
        self._last_inference_latency_sec = None
        self._last_stale_inference_reason = None
        self.get_logger().warn(
            'camera source timestamp epoch rollback confirmed; '
            f'hard-resetting perception state (source_epoch={epoch}, '
            f'generation={self._target_barrier.generation})'
        )

    def _detect_and_publish(self):
        """Run at most one detector call while control callbacks stay live."""
        # Avoid even an "inference still active" timer warning once a wrist
        # request has completed.  _detect_and_publish_once repeats this gate
        # after acquiring the inference lock to cover a concurrent reset.
        with self._state_lock:
            if (
                getattr(self, '_on_demand_inference', False)
                and not getattr(self, '_demand_capture_active', False)
            ):
                return
        if not self._inference_lock.acquire(blocking=False):
            self.get_logger().warn(
                'previous perception inference still active; timer tick skipped',
                throttle_duration_sec=3.0,
            )
            return
        try:
            self._detect_and_publish_once()
        finally:
            self._inference_lock.release()

    def _detect_and_publish_once(self):
        # Do this before publisher counting, source diagnostics, and GPU work.
        # Subscriptions remain connected while idle, preserving all existing
        # topic/service names but keeping the fixed-camera-only flow quiet.
        with self._state_lock:
            if (
                getattr(self, '_on_demand_inference', False)
                and not getattr(self, '_demand_capture_active', False)
            ):
                return
            demand_capture_epoch = int(
                getattr(self, '_demand_capture_epoch', 0))

        if self._require_unique_camera_source:
            publisher_counts = {
                self._rgb_topic: self.count_publishers(self._rgb_topic),
                self._depth_topic: self.count_publishers(self._depth_topic),
                self._camera_info_topic: self.count_publishers(
                    self._camera_info_topic),
            }
        else:
            publisher_counts = {'uniqueness-check-disabled': 1}

        with self._state_lock:
            if (
                not self._rgb_frames
                or not self._depth_frames
                or self._camera_model is None
            ):
                return

            inference_generation = self._target_barrier.generation
            bolt_pair_tracking_active = self._bolt_pair_tracking_active
            bolt_pair_capture_epoch = self._bolt_pair_capture_epoch
            association_anchor = (
                self._bolt_pair_association.anchor
                if bolt_pair_tracking_active
                else None
            )
            nut_anchor = self._nut_association.anchor
            camera_model = self._camera_model
            camera_frame_id = self._camera_frame_id
            now_monotonic = time.monotonic()
            pair = select_newest_synchronized_pair(
                self._rgb_frames,
                self._depth_frames,
                max_skew_sec=self._max_rgb_depth_skew_sec,
                max_receipt_age_sec=self._max_input_receipt_age_sec,
                now=now_monotonic,
                rgb_after_ns=self._frame_gate.last_rgb_stamp_ns,
                depth_after_ns=self._frame_gate.last_depth_stamp_ns,
            )

            # A real simulator/source restart rolls timestamps back below the
            # normal anti-replay boundary. Search a coherent fresh pair again
            # without that boundary solely for the two-pair epoch detector.
            epoch_pair = pair
            if epoch_pair is None:
                epoch_pair = select_newest_synchronized_pair(
                    self._rgb_frames,
                    self._depth_frames,
                    max_skew_sec=self._max_rgb_depth_skew_sec,
                    max_receipt_age_sec=(
                        self._max_input_receipt_age_sec
                    ),
                    now=now_monotonic,
                )

            epoch_rejection_reason = None
            if epoch_pair is not None:
                epoch_result = self._source_epoch_tracker.observe_pair(
                    epoch_pair.rgb.stamp_ns,
                    epoch_pair.depth.stamp_ns,
                )
                if epoch_result.rebase:
                    self._hard_reset_source_epoch_locked(
                        epoch_result.epoch
                    )
                    inference_generation = (
                        self._target_barrier.generation
                    )
                    association_anchor = None
                    pair = epoch_pair
                elif epoch_result.accept_pair:
                    # Normally ``pair`` already points here. This also heals a
                    # frame-gate boundary that diverged before the tracker was
                    # introduced.
                    pair = epoch_pair
                else:
                    pair = None
                    if (
                        epoch_result.pending
                        or epoch_result.quarantined
                    ):
                        epoch_rejection_reason = epoch_result.reason

            # If no queued pair meets freshness/skew/epoch constraints,
            # evaluate one pair only for actionable diagnostics. A diagnostic
            # pair never advances CameraFrameGate's accepted sequence.
            diagnostic_only = pair is None
            if diagnostic_only:
                if epoch_pair is not None:
                    rgb_frame = epoch_pair.rgb
                    depth_frame = epoch_pair.depth
                else:
                    rgb_frame = self._rgb_frames[-1]
                    depth_frame = self._depth_frames[-1]
            else:
                rgb_frame = pair.rgb
                depth_frame = pair.depth

            evaluation_gate = self._frame_gate
            if diagnostic_only:
                # Preserve actionable rejection messages without allowing an
                # inspection-only pair to mutate the live anti-replay sequence.
                evaluation_gate = CameraFrameGate(
                    max_skew_sec=self._max_rgb_depth_skew_sec,
                    max_receipt_age_sec=self._max_input_receipt_age_sec,
                )
                evaluation_gate.reset(
                    rgb_boundary_ns=self._frame_gate.last_rgb_stamp_ns,
                    depth_boundary_ns=self._frame_gate.last_depth_stamp_ns,
                )

            gate_result = evaluation_gate.evaluate(
                rgb_stamp_ns=rgb_frame.stamp_ns,
                depth_stamp_ns=depth_frame.stamp_ns,
                rgb_received_at=rgb_frame.received_at,
                depth_received_at=depth_frame.received_at,
                now=now_monotonic,
                publisher_counts=publisher_counts,
            )
            if (
                not diagnostic_only
                and gate_result.accepted
            ):
                self._last_observed_rgb_stamp_ns = advance_source_boundary(
                    self._last_observed_rgb_stamp_ns,
                    rgb_frame.stamp_ns,
                )
                self._last_observed_depth_stamp_ns = advance_source_boundary(
                    self._last_observed_depth_stamp_ns,
                    depth_frame.stamp_ns,
                )
        if epoch_rejection_reason is not None:
            self.get_logger().warn(
                'camera source epoch candidate rejected: '
                f'{epoch_rejection_reason}',
                throttle_duration_sec=2.0,
            )
            return
        if diagnostic_only or not gate_result.accepted:
            if 'publisher count' in gate_result.reason:
                self.get_logger().error(
                    f'camera input rejected: {gate_result.reason}',
                    throttle_duration_sec=3.0)
            elif 'duplicate' not in gate_result.reason:
                self.get_logger().warn(
                    f'camera input rejected: {gate_result.reason}',
                    throttle_duration_sec=3.0)
            return

        rgb, header = rgb_frame.payload
        depth, _depth_header = depth_frame.payload
        rgb_stamp_ns = rgb_frame.stamp_ns
        inference_started_at = time.monotonic()

        array_msg = Detection3DArray()
        array_msg.header.stamp = header.stamp
        array_msg.header.frame_id = self._world_frame

        debug_image = rgb.copy() if self._debug_image_pub is not None else None
        best_this_tick = {}
        resolved_by_label = collections.defaultdict(list)

        # ROI 파라미터 로드
        use_bolt_roi = self.get_parameter('use_bolt_roi').value
        image_height, image_width = rgb.shape[:2]
        try:
            bolt_x_min, bolt_x_max, bolt_y_min, bolt_y_max = scale_roi(
                image_size=(image_width, image_height),
                reference_size=(
                    int(self.get_parameter('bolt_roi_reference_width').value),
                    int(self.get_parameter('bolt_roi_reference_height').value),
                ),
                reference_bounds=(
                    int(self.get_parameter('bolt_roi_x_min').value),
                    int(self.get_parameter('bolt_roi_x_max').value),
                    int(self.get_parameter('bolt_roi_y_min').value),
                    int(self.get_parameter('bolt_roi_y_max').value),
                ),
            )
        except ValueError as exc:
            self.get_logger().error(
                f'invalid bolt ROI: {exc}',
                throttle_duration_sec=3.0)
            return

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
            px_x, px_y = det['pixel']

            # 🔥 'bolt' 라벨인 경우에만 ROI 범위 검사 수행
            if use_bolt_roi and label == BOLT_LABEL:
                if not (bolt_x_min <= px_x <= bolt_x_max and bolt_y_min <= px_y <= bolt_y_max):
                    continue  # 배터리 스캔 영역 밖의 볼트는 스킵

            if label == BUSBAR_LABEL and self._use_busbar_pnp:
                camera_point, world_point, status = (
                    self._resolve_busbar_pose(
                        det,
                        depth,
                        header.stamp,
                        camera_model=camera_model,
                        camera_frame_id=camera_frame_id,
                    )
                )
            else:
                camera_point, world_point, status = self._transform_pixel(
                    det['pixel'],
                    depth,
                    header.stamp,
                    camera_model=camera_model,
                    camera_frame_id=camera_frame_id,
                )

            if debug_image is not None:
                overlay.draw_detection(debug_image, det, camera_point, world_point, status)

            if world_point is not None:
                array_msg.detections.append(
                    self._make_detection3d(det, world_point, header.stamp))
                resolved_by_label[label].append({
                    'score': float(det['score']),
                    'pixel': (px_x, px_y),
                    'world_point': world_point,
                })
            elif label == BOLT_LABEL:
                self.get_logger().warn(
                    f"[bolt] YOLO 검출됐지만 world 좌표 변환 실패 pixel=({px_x},{px_y}), status={status}",
                    throttle_duration_sec=2.0)

        for label, candidates in resolved_by_label.items():
            if label == BUSBAR_LABEL:
                selected = select_central_candidates(
                    candidates, (image_width, image_height), limit=1)
            elif label == NUT_LABEL:
                nut_candidate = select_sticky_single_target(
                    candidates,
                    (image_width, image_height),
                    anchor_point=nut_anchor,
                    max_anchor_distance_m=(
                        self._nut_association.max_distance_m
                    ),
                )
                selected = (
                    [nut_candidate]
                    if nut_candidate is not None
                    else []
                )
            elif label == BOLT_LABEL:
                continue
            else:
                selected = sorted(
                    candidates, key=lambda candidate: candidate['score'], reverse=True)[:1]
            if label != BOLT_LABEL and selected:
                candidate = selected[0]
                best_this_tick[label] = (
                    candidate['score'],
                    candidate['world_point'],
                )

        # Bolt detections still appear in the generic Detection3D output, but
        # pair ranking/association is deliberately demand-driven.  The direct
        # fixed-camera flow does not call GetBoltPair, so it must not keep
        # selecting unrelated packs or producing a periodic warning.
        bolt_pair = None
        if bolt_pair_tracking_active:
            bolt_pair = select_central_bolt_pair(
                resolved_by_label.get(BOLT_LABEL, ()),
                (image_width, image_height),
                expected_delta_xy=self._bolt_pair_expected_delta_xy,
                x_tolerance_m=self._bolt_pair_x_tolerance_m,
                y_tolerance_m=self._bolt_pair_y_tolerance_m,
                max_z_delta_m=self._bolt_pair_max_z_delta_m,
                anchor_pair=association_anchor,
                max_anchor_endpoint_distance_m=(
                    self._bolt_pair_association.max_endpoint_distance_m
                ),
            )

        with self._state_lock:
            if (
                getattr(self, '_on_demand_inference', False)
                and (
                    not getattr(self, '_demand_capture_active', False)
                    or int(getattr(self, '_demand_capture_epoch', 0))
                    != demand_capture_epoch
                )
            ):
                # A service completed or reset ended this request while YOLO
                # was running.  Never publish/caches its stale result.
                return
            if not self._target_barrier.is_current(inference_generation):
                self.get_logger().info(
                    'discarding inference completed across perception reset '
                    f'(stale generation={inference_generation}, '
                    f'current={self._target_barrier.generation})')
                return

            # Inference may take longer than the input-age contract. Recheck
            # the original callback receipt times immediately before mutating
            # any target cache or barrier.
            receipt_error = frame_receipt_error(
                rgb_received_at=rgb_frame.received_at,
                depth_received_at=depth_frame.received_at,
                now=time.monotonic(),
                max_receipt_age_sec=self._max_input_receipt_age_sec,
            )
            if receipt_error is not None:
                inference_latency_sec = (
                    time.monotonic() - inference_started_at
                )
                self._record_stale_inference(
                    receipt_error,
                    inference_latency_sec,
                )
                self.get_logger().warn(
                    'discarding inference with stale input at commit: '
                    f'{receipt_error}; '
                    f'latency={inference_latency_sec:.3f}s, '
                    f'consecutive='
                    f'{self._consecutive_stale_inferences}, '
                    f'total={self._total_stale_inferences}',
                    throttle_duration_sec=2.0,
                )
                return

            self._record_successful_inference(
                time.monotonic() - inference_started_at
            )
            now = self.get_clock().now()
            # Presence represents the newest accepted inference only. Cached
            # smoothing/barrier state can survive a short detector flicker,
            # but a service must not return that old pose as currently seen.
            bolt_pair_capture_current = (
                bolt_pair_tracking_active
                and self._bolt_pair_tracking_active
                and self._bolt_pair_capture_epoch
                == bolt_pair_capture_epoch
            )
            if bolt_pair_capture_current:
                self._target_present_latest[BOLT_LABEL] = False
            self._target_present_latest[NUT_LABEL] = False
            nut_committed = False
            for label, (score, world_point) in best_this_tick.items():
                if label == NUT_LABEL:
                    associated, distance_m = (
                        self._nut_association.record(world_point)
                    )
                    if not associated:
                        self.get_logger().warn(
                            '[nut] candidate rejected by sticky world '
                            f'anchor (distance={distance_m * 1000:.1f}mm)',
                            throttle_duration_sec=1.0,
                        )
                        continue
                    nut_committed = True
                mean_point, _n = self._update_smoothed(
                    label, world_point, now)
                self._latest_by_label[label] = (
                    inference_generation,
                    score,
                    mean_point,
                    now,
                )
                self._record_target_frame(
                    label,
                    rgb_stamp_ns,
                    inference_generation,
                    rgb_frame.received_at,
                    depth_frame.received_at,
                )
                self._publish_vision_topic(
                    label, mean_point, header.stamp)
                if label == NUT_LABEL:
                    self._target_present_latest[NUT_LABEL] = True

            if (
                not nut_committed
                and self._nut_association.active
            ):
                if self._nut_association.record_miss():
                    self.get_logger().warn(
                        '[nut] anchored target missing for '
                        f'{self._nut_association.miss_limit} '
                        'consecutive frames -> anchor/smoothing/'
                        '3-frame barrier reset',
                        throttle_duration_sec=1.0,
                    )
                    self._clear_nut_track(
                        reset_association=False
                    )
                else:
                    self.get_logger().warn(
                        '[nut] retaining sticky anchor through detector '
                        f'flicker (miss '
                        f'{self._nut_association.consecutive_misses}/'
                        f'{self._nut_association.miss_limit})',
                        throttle_duration_sec=1.0,
                    )

            if bolt_pair_capture_current:
                if bolt_pair is None:
                    if self._bolt_pair_association.active:
                        if self._bolt_pair_association.record_miss():
                            self.get_logger().warn(
                                '[bolt-pair] anchored pair missing for '
                                f'{self._bolt_pair_association.miss_limit} '
                                'consecutive frames -> anchor/smoothing/'
                                '3-frame barrier reset',
                                throttle_duration_sec=1.0,
                            )
                            self._clear_bolt_track(
                                reset_association=False)
                        else:
                            self.get_logger().warn(
                                '[bolt-pair] ignoring non-anchor candidates '
                                f'(miss '
                                f'{self._bolt_pair_association.consecutive_misses}/'
                                f'{self._bolt_pair_association.miss_limit})',
                                throttle_duration_sec=1.0,
                            )
                    if len(resolved_by_label.get(BOLT_LABEL, ())) >= 2:
                        self.get_logger().warn(
                            '[bolt-pair] same-pack geometry와 맞는 볼트쌍 없음',
                            throttle_duration_sec=2.0,
                        )
                elif self._update_bolt_pair([
                    (
                        candidate['score'],
                        candidate['world_point'],
                        candidate['pixel'],
                    )
                    for candidate in bolt_pair
                ], now, inference_generation):
                    self._target_present_latest[BOLT_LABEL] = True
                    self._record_target_frame(
                        BOLT_LABEL,
                        rgb_stamp_ns,
                        inference_generation,
                        rgb_frame.received_at,
                        depth_frame.received_at,
                    )

            self._detections_pub.publish(array_msg)

            if debug_image is not None:
                debug_msg = self._bridge.cv2_to_imgmsg(
                    debug_image, encoding='bgr8')
                debug_msg.header = header
                self._debug_image_pub.publish(debug_msg)

    def _record_stale_inference(
        self,
        reason: str,
        latency_sec: float,
    ) -> None:
        """Record a detector result rejected by the one-second input contract."""
        self._consecutive_stale_inferences += 1
        self._total_stale_inferences += 1
        self._last_inference_latency_sec = float(latency_sec)
        self._last_stale_inference_reason = str(reason)

    def _record_successful_inference(self, latency_sec: float) -> None:
        """Clear the consecutive-stale run after a committed inference."""
        self._consecutive_stale_inferences = 0
        self._last_inference_latency_sec = float(latency_sec)
        self._last_stale_inference_reason = None

    def _inference_diagnostic_suffix(self) -> str:
        """Return compact service-visible inference freshness diagnostics."""
        consecutive = getattr(
            self,
            '_consecutive_stale_inferences',
            0,
        )
        if consecutive <= 0:
            return ''
        total = getattr(self, '_total_stale_inferences', consecutive)
        latency = getattr(self, '_last_inference_latency_sec', None)
        reason = getattr(self, '_last_stale_inference_reason', None)
        latency_text = (
            f'{float(latency):.3f}s'
            if latency is not None
            else 'unknown'
        )
        return (
            ' [stale_inference '
            f'consecutive={consecutive} total={total} '
            f'latency={latency_text} reason={reason}]'
        )

    def _record_target_frame(
        self,
        label: str,
        stamp_ns: int,
        generation: int,
        rgb_received_at: float,
        depth_received_at: float,
    ):
        if not self._target_barrier.is_current(generation):
            return False
        # Store the real callback receipt timestamps. Inference completion
        # time would falsely rejuvenate a slow or stalled input pair.
        self._latest_target_received_at[label] = (
            float(rgb_received_at),
            float(depth_received_at),
        )
        ready = self._target_barrier.record(
            label,
            stamp_ns,
            generation=generation,
        )
        if not ready:
            self.get_logger().info(
                f"[{label}] post-reset frame barrier "
                f"{self._target_barrier.count(label)}/"
                f"{self._target_barrier.required_frames}")
        return True

    def _target_input_error(self, label: str):
        if self._require_unique_camera_source:
            counts = {
                self._rgb_topic: self.count_publishers(self._rgb_topic),
                self._depth_topic: self.count_publishers(self._depth_topic),
                self._camera_info_topic: self.count_publishers(
                    self._camera_info_topic),
            }
            invalid = [
                f'{topic}={count}'
                for topic, count in counts.items()
                if count != 1
            ]
            if invalid:
                return 'camera publisher 수가 정확히 1이 아님: ' + ', '.join(invalid)

        received_times = self._latest_target_received_at.get(label)
        if received_times is None:
            return f"'{label}' 유효 target frame receipt 없음"
        rgb_received_at, depth_received_at = received_times
        receipt_error = frame_receipt_error(
            rgb_received_at=rgb_received_at,
            depth_received_at=depth_received_at,
            now=time.monotonic(),
            max_receipt_age_sec=self._max_input_receipt_age_sec,
        )
        if receipt_error is not None:
            return (
                f"'{label}' target frame receipt age 초과 "
                f"({receipt_error})")
        return None

    def _publish_vision_topic(self, label, world_point, stamp):
        wx, wy, wz = world_point
        if label == BUSBAR_LABEL:
            msg = BusbarGrasp()
            msg.pose.header.stamp = stamp
            msg.pose.header.frame_id = self._world_frame
            msg.pose.pose.position.x = wx
            msg.pose.pose.position.y = wy
            msg.pose.pose.position.z = wz
            msg.pose.pose.orientation.w = 1.0
            self._busbar_grasp_pub.publish(msg)
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
        bucket = self._recent_detections[label]
        mean, std, n = self._smooth_bucket(bucket, world_point, now)
        std_mm = std * 1000.0

        self.get_logger().info(
            f"[{label}] raw=({world_point[0]:.4f},{world_point[1]:.4f},{world_point[2]:.4f}) "
            f"mean=({mean[0]:.4f},{mean[1]:.4f},{mean[2]:.4f}) "
            f"std=({std_mm[0]:.1f},{std_mm[1]:.1f},{std_mm[2]:.1f})mm n={n}"
        )
        if std_mm.max() > SMOOTHING_STD_WARN_MM:
            self.get_logger().warn(
                f"[{label}] 검출 흔들림 큼 (표준편차 최대 {std_mm.max():.1f}mm > "
                f"{SMOOTHING_STD_WARN_MM:.0f}mm) — 조명/거리/각도 또는 conf_threshold 확인 필요"
            )
        return tuple(mean), n

    def _update_bolt_pair(self, bolt_candidates, now, generation):
        if not self._target_barrier.is_current(generation):
            return False
        if len(bolt_candidates) != 2:
            if bolt_candidates:
                self.get_logger().warn(
                    f"[bolt-pair] 유효한 same-pack 볼트 수 오류: "
                    f"{len(bolt_candidates)}/2",
                    throttle_duration_sec=2.0)
            return False

        raw_points = [
            np.asarray(wp, dtype=float)
            for _, wp, _px in bolt_candidates
        ]
        ordered_points, restarted, endpoint_distance = (
            self._bolt_pair_association.record(raw_points)
        )
        points = [
            np.asarray(point, dtype=float)
            for point in ordered_points
        ]
        if restarted:
            self.get_logger().warn(
                '[bolt-pair] 다른 same-pack 후보로 전환 감지 '
                f'(anchor endpoint 이동={endpoint_distance * 1000:.1f}mm > '
                f'{self._bolt_pair_association.max_endpoint_distance_m * 1000:.0f}mm)'
                ' -> smoothing/3-frame barrier 재시작',
                throttle_duration_sec=1.0,
            )
            self._clear_bolt_track(reset_association=False)

        mean_a, std_a, n_a = self._smooth_bucket(self._recent_bolt_a, points[0], now)
        mean_b, std_b, n_b = self._smooth_bucket(self._recent_bolt_b, points[1], now)
        mid_xy = ((mean_a[0] + mean_b[0]) / 2, (mean_a[1] + mean_b[1]) / 2)

        self.get_logger().info(
            f"[bolt-pair] A mean=({mean_a[0]:.4f},{mean_a[1]:.4f},{mean_a[2]:.4f}) "
            f"std_max={std_a.max()*1000:.1f}mm n={n_a} | "
            f"B mean=({mean_b[0]:.4f},{mean_b[1]:.4f},{mean_b[2]:.4f}) "
            f"std_max={std_b.max()*1000:.1f}mm n={n_b} | mid_xy=({mid_xy[0]:.4f},{mid_xy[1]:.4f})"
        )
        self._latest_bolt_pair = (
            generation,
            tuple(mean_a),
            tuple(mean_b),
            now,
            min(n_a, n_b),
        )
        return True

    def _clear_bolt_track(self, *, reset_association: bool) -> None:
        """Discard bolt observations and optionally their physical-pair anchor."""
        self._recent_bolt_a.clear()
        self._recent_bolt_b.clear()
        self._latest_bolt_pair = None
        self._latest_target_received_at.pop(BOLT_LABEL, None)
        self._target_present_latest[BOLT_LABEL] = False
        self._target_barrier.reset_target(BOLT_LABEL)
        if reset_association:
            self._bolt_pair_association.reset()

    def _arm_bolt_pair_tracking_locked(self) -> None:
        """Start one fresh GetBoltPair capture, if no caller owns one yet."""
        if self._bolt_pair_tracking_active:
            return
        self._clear_bolt_track(reset_association=True)
        self._bolt_pair_tracking_active = True
        self._bolt_pair_capture_epoch += 1

    def _cancel_bolt_pair_tracking_locked(self) -> None:
        """Fence in-flight pair work after reset or a completed response."""
        self._bolt_pair_tracking_active = False
        self._bolt_pair_capture_epoch = (
            int(getattr(self, '_bolt_pair_capture_epoch', 0)) + 1
        )

    def _clear_nut_track(self, *, reset_association: bool) -> None:
        """Discard nut observations and optionally their physical anchor."""
        self._recent_detections.pop(NUT_LABEL, None)
        self._latest_by_label.pop(NUT_LABEL, None)
        self._latest_target_received_at.pop(NUT_LABEL, None)
        self._target_present_latest[NUT_LABEL] = False
        self._target_barrier.reset_target(NUT_LABEL)
        if reset_association:
            self._nut_association.reset()

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

    def _log_tf_error(self, ex, camera_frame_id=None):
        frame_id = camera_frame_id or self._camera_frame_id
        self.get_logger().warn(
            f'{self._world_frame} <- {frame_id} '
            f'tf 조회 실패, 검출 skip: {ex}',
            throttle_duration_sec=5.0)

    def _transform_pixel(
        self,
        pixel_uv,
        depth_image,
        stamp,
        *,
        camera_model=None,
        camera_frame_id=None,
    ):
        camera_model = camera_model or self._camera_model
        camera_frame_id = camera_frame_id or self._camera_frame_id
        return transform_pixel_to_world(
            camera_model, depth_image, pixel_uv, self._tf_buffer,
            self._world_frame, camera_frame_id, stamp,
            on_tf_error=lambda exc: self._log_tf_error(
                exc,
                camera_frame_id,
            ))

    def _resolve_busbar_pose(
        self,
        det,
        depth_image,
        stamp,
        *,
        camera_model=None,
        camera_frame_id=None,
    ):
        """
        Resolve a v1/v3 busbar using centroid depth plus planar PnP.

        Six-landmark centroid depth is the most accurate path when it lands on
        the busbar. If a wrist view samples the floor through the open centre,
        its Z differs sharply from the geometry-only PnP estimate and PnP wins.
        """
        if camera_model is None and camera_frame_id is None:
            # Preserve the narrow test/extension hook that overrides the
            # three-argument transform helper.
            camera_model = self._camera_model
            camera_frame_id = getattr(
                self,
                '_camera_frame_id',
                None,
            )
            depth_camera, depth_world, depth_status = (
                self._transform_pixel(
                    det['pixel'],
                    depth_image,
                    stamp,
                )
            )
        else:
            camera_model = camera_model or self._camera_model
            camera_frame_id = (
                camera_frame_id or self._camera_frame_id
            )
            depth_camera, depth_world, depth_status = (
                self._transform_pixel(
                    det['pixel'],
                    depth_image,
                    stamp,
                    camera_model=camera_model,
                    camera_frame_id=camera_frame_id,
                )
            )
        pnp_world = None
        pnp_status = 'unavailable'
        reprojection_error = None
        keypoints = np.asarray(det.get('keypoints_px', ()), dtype=float)
        keypoints_reliable = busbar_keypoints_are_reliable(
            keypoints,
            det.get('keypoints_conf', ()),
            (
                int(camera_model.width),
                int(camera_model.height),
            ),
        )
        if not keypoints_reliable:
            pnp_status = (
                "landmark confidence/in-frame gate failed "
                f"(min={MIN_BUSBAR_KEYPOINT_CONFIDENCE:.2f})"
            )
        elif keypoints.shape == (len(BUSBAR_KEYPOINT_ORDER), 2):
            pnp = busbar_pnp_world_pose(
                camera_model,
                keypoints,
                self._tf_buffer,
                self._world_frame,
                camera_frame_id,
                stamp,
                on_tf_error=lambda exc: self._log_tf_error(
                    exc,
                    camera_frame_id,
                ),
            )
            pnp_status = pnp['status']
            if pnp['status'] == '':
                reprojection_error = pnp['reprojection_error_px']
                if reprojection_error > (
                    self._max_busbar_pnp_reprojection_px
                ):
                    pnp_status = (
                        f'reprojection {reprojection_error:.2f}px exceeds '
                        f'{self._max_busbar_pnp_reprojection_px:.2f}px'
                    )
                    self.get_logger().warn(
                        f'[busbar] PnP rejected: {pnp_status}',
                        throttle_duration_sec=3.0,
                    )
                    pnp = None
            if pnp is not None and pnp_status == '':
                hole_a = pnp['hole_world']['hole_A']
                hole_b = pnp['hole_world']['hole_B']
                pnp_world = (
                    pnp['mid_xy'][0],
                    pnp['mid_xy'][1],
                    (hole_a[2] + hole_b[2]) / 2.0,
                )

        world_point, source, z_disagreement = select_busbar_world_point(
            depth_world,
            pnp_world,
            prefer_depth=self._prefer_busbar_depth,
        )
        if world_point is None:
            return depth_camera, None, (
                f'depth={depth_status or "ok"}, pnp={pnp_status}'
            )

        details = []
        if reprojection_error is not None:
            details.append(f'pnp_reprojection={reprojection_error:.2f}px')
        if z_disagreement is not None:
            details.append(f'z_delta={z_disagreement:.3f}m')
        self.get_logger().info(
            f'[busbar] pose_source={source}'
            + (f' ({", ".join(details)})' if details else ''),
            throttle_duration_sec=3.0,
        )
        camera_point = depth_camera if source == 'depth' else None
        return camera_point, world_point, ''

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
        with self._state_lock:
            # A direct legacy service call is also an explicit demand signal.
            # ArmNode normally resets first, but this keeps callers that only
            # use GetGraspPose compatible with the dormant wrist instance.
            PerceptionNode._arm_on_demand_inference_locked(self)
            response = self._handle_get_grasp_pose_locked(
                request, response)
            if response.found:
                PerceptionNode._finish_on_demand_inference_locked(self)
            return self._tag_service_generation(response)

    def _tag_service_generation(self, response):
        """Expose reset generation through an existing compatible field."""
        diagnostic_fn = getattr(
            self,
            '_inference_diagnostic_suffix',
            None,
        )
        diagnostic = (
            diagnostic_fn()
            if callable(diagnostic_fn)
            else ''
        )
        response.message = (
            f"[generation={self._target_barrier.generation}] "
            f"{response.message}"
            f"{diagnostic}"
        )
        return response

    def _handle_get_grasp_pose_locked(self, request, response):
        if not self._target_barrier.ready(request.label):
            response.found = False
            response.message = (
                f"'{request.label}' 새 target frame 준비 중 "
                f"(n={self._target_barrier.count(request.label)} < "
                f"{self._target_barrier.required_frames})")
            return response

        if (
            request.label == NUT_LABEL
            and not getattr(
                self,
                '_target_present_latest',
                {},
            ).get(NUT_LABEL, True)
        ):
            response.found = False
            response.message = (
                "'nut' 최신 accepted inference에 anchored target이 "
                "보이지 않음; flicker miss policy 유지 중"
            )
            return response

        input_error = self._target_input_error(request.label)
        if input_error is not None:
            response.found = False
            response.message = input_error
            return response

        cached = self._latest_by_label.get(request.label)
        if cached is None:
            response.found = False
            response.message = f"'{request.label}' 라벨 검출 캐시 없음"
            return response

        generation, _score, world_point, stamp = cached
        if not self._target_barrier.is_current(generation):
            response.found = False
            response.message = (
                f"'{request.label}' 캐시가 이전 reset 세대 데이터임")
            return response

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
        with self._state_lock:
            # Keep the public service usable on any configured instance even
            # if that instance happens to use on-demand inference.
            PerceptionNode._arm_on_demand_inference_locked(self)
            self._arm_bolt_pair_tracking_locked()
            response = self._handle_get_bolt_pair_locked(
                request, response)
            if response.found:
                # The caller received the stable snapshot.  Return to idle so
                # nearby battery packs cannot churn pair tracking/logs during
                # the subsequent arm motion.  A later service call re-arms a
                # fresh three-frame capture.
                self._cancel_bolt_pair_tracking_locked()
                PerceptionNode._finish_on_demand_inference_locked(self)
            return self._tag_service_generation(response)

    def _handle_get_bolt_pair_locked(self, request, response):
        if not self._target_barrier.ready(BOLT_LABEL):
            response.found = False
            response.message = (
                "'bolt' 새 target frame 준비 중 "
                f"(n={self._target_barrier.count(BOLT_LABEL)} < "
                f"{self._target_barrier.required_frames})")
            return response

        if not getattr(
            self,
            '_target_present_latest',
            {},
        ).get(BOLT_LABEL, True):
            response.found = False
            response.message = (
                "'bolt' 최신 accepted inference에 anchored pair가 "
                "보이지 않음; flicker miss policy 유지 중"
            )
            return response

        input_error = self._target_input_error(BOLT_LABEL)
        if input_error is not None:
            response.found = False
            response.message = input_error
            return response

        if self._latest_bolt_pair is None:
            response.found = False
            response.message = "'bolt' 쌍 검출 캐시 없음"
            return response

        (
            generation,
            mean_a,
            mean_b,
            stamp,
            sample_count,
        ) = self._latest_bolt_pair
        if not self._target_barrier.is_current(generation):
            response.found = False
            response.message = "'bolt' 쌍 캐시가 이전 reset 세대 데이터임"
            return response

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
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.remove_node(node)
        executor.shutdown()
        if rclpy.ok():
            try:
                node.destroy_node()
                rclpy.shutdown()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()

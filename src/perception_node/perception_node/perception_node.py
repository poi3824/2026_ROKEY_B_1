"""perception_node
실행 계층 · YOLO 검출(bolt/nut/busbar) -> world 좌표 변환 -> 토픽 발행.

SUB <rgb_topic>          (sensor_msgs/Image, rgb8)
SUB <depth_topic>        (sensor_msgs/Image, 32FC1, m)
SUB <camera_info_topic>  (sensor_msgs/CameraInfo)

PUB /perception/detections_3d  (vision_msgs/Detection3DArray, <world_frame> 기준)
PUB <debug_image_topic>         (sensor_msgs/Image, bgr8) — rqt_image_view 등으로
                                 마스크/bbox/좌표(픽셀, 카메라 프레임, world 프레임)를
                                 확인하기 위한 디버그 오버레이. publish_debug_image로 끌 수 있음.

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
import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from perception_node import overlay
from perception_node.camera_geometry import make_camera_model, transform_pixel_to_world
from perception_node.detector import YoloSegDetector

DEFAULT_MODEL_PATH = os.path.join(
    get_package_share_directory('perception_node'), 'models', 'best.pt')


class PerceptionNode(Node):

    def __init__(self):
        super().__init__('perception_node')

        self.declare_parameter('rgb_topic', '/rgb')
        self.declare_parameter('depth_topic', '/depth')
        self.declare_parameter('camera_info_topic', '/camera_info')
        self.declare_parameter('model_path', DEFAULT_MODEL_PATH)
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('camera_frame_override', '')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('detection_period_sec', 0.5)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_image_topic', '/perception/debug_image')

        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        model_path = self.get_parameter('model_path').value
        self._world_frame = self.get_parameter('world_frame').value
        self._camera_frame_override = self.get_parameter('camera_frame_override').value
        self._conf_threshold = self.get_parameter('conf_threshold').value
        detection_period_sec = self.get_parameter('detection_period_sec').value
        publish_debug_image = self.get_parameter('publish_debug_image').value
        debug_image_topic = self.get_parameter('debug_image_topic').value

        self._bridge = CvBridge()
        self._detector = YoloSegDetector(model_path)

        self._latest_rgb = None
        self._latest_rgb_header = None
        self._latest_depth = None
        self._camera_model = None
        self._camera_frame_id = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._rgb_sub = self.create_subscription(Image, rgb_topic, self._on_rgb, 10)
        self._depth_sub = self.create_subscription(Image, depth_topic, self._on_depth, 10)
        self._camera_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self._on_camera_info, 10)

        self._detections_pub = self.create_publisher(
            Detection3DArray, '/perception/detections_3d', 10)

        self._debug_image_pub = None
        if publish_debug_image:
            self._debug_image_pub = self.create_publisher(Image, debug_image_topic, 10)

        self._timer = self.create_timer(detection_period_sec, self._detect_and_publish)

        self.get_logger().info(
            f'perception_node started (model={model_path}, world_frame={self._world_frame})')

    def _on_rgb(self, msg: Image):
        self._latest_rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self._latest_rgb_header = msg.header

    def _on_depth(self, msg: Image):
        self._latest_depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _on_camera_info(self, msg: CameraInfo):
        self._camera_model = make_camera_model(msg)
        self._camera_frame_id = self._camera_frame_override or msg.header.frame_id

    def _detect_and_publish(self):
        if self._latest_rgb is None or self._latest_depth is None or self._camera_model is None:
            return

        rgb = self._latest_rgb
        depth = self._latest_depth
        header = self._latest_rgb_header

        array_msg = Detection3DArray()
        array_msg.header.stamp = header.stamp
        array_msg.header.frame_id = self._world_frame

        debug_image = rgb.copy() if self._debug_image_pub is not None else None

        for det in self._detector.detect(rgb, self._conf_threshold):
            camera_point, world_point, status = self._transform_pixel(
                det['pixel'], depth, header.stamp)

            if debug_image is not None:
                overlay.draw_detection(debug_image, det, camera_point, world_point, status)

            if world_point is not None:
                array_msg.detections.append(
                    self._make_detection3d(det, world_point, header.stamp))

        self._detections_pub.publish(array_msg)

        if debug_image is not None:
            debug_msg = self._bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            debug_msg.header = header
            self._debug_image_pub.publish(debug_msg)

    def _transform_pixel(self, pixel_uv, depth_image, stamp):
        def log_tf_error(ex):
            self.get_logger().warn(
                f'{self._world_frame} <- {self._camera_frame_id} tf 조회 실패, 검출 skip: {ex}',
                throttle_duration_sec=5.0)

        return transform_pixel_to_world(
            self._camera_model, depth_image, pixel_uv, self._tf_buffer,
            self._world_frame, self._camera_frame_id, stamp, on_tf_error=log_tf_error)

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
        # 2D bbox 픽셀 크기를 그대로 xy 크기로 근사한 값일 뿐, 실제 3D extent가 아님.
        x0, y0, x1, y1 = det['bbox_px']
        detection.bbox.size.x = float(x1 - x0)
        detection.bbox.size.y = float(y1 - y0)
        detection.bbox.size.z = 0.0

        return detection


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

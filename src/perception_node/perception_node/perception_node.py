"""perception_node
실행 계층 · YOLO 검출(bolt/nut/busbar) -> world 좌표 변환 -> 토픽 발행.

SUB <rgb_topic>          (sensor_msgs/Image, rgb8)
SUB <depth_topic>        (sensor_msgs/Image, 32FC1, m)
SUB <camera_info_topic>  (sensor_msgs/CameraInfo)

PUB /perception/detections_3d  (vision_msgs/Detection3DArray, <world_frame> 기준)

카메라 프레임 -> world_frame 변환은 tf2 lookupTransform으로 조회한다. 이 노드는
카메라가 어디에 있는지 알지 못하며, world_frame -> 카메라 frame_id로의 tf가 어디선가
(로봇 URDF/robot_state_publisher, 또는 캘리브레이션용 static_transform_publisher)
발행되고 있어야 world 좌표가 채워진다. tf가 없으면 해당 검출은 skip하고 경고 로그만
남긴다.
"""
import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_point
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from perception_node.camera_geometry import make_camera_model, pixel_to_camera_point
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
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('detection_period_sec', 0.5)

        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        model_path = self.get_parameter('model_path').value
        self._world_frame = self.get_parameter('world_frame').value
        self._conf_threshold = self.get_parameter('conf_threshold').value
        detection_period_sec = self.get_parameter('detection_period_sec').value

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
        self._camera_frame_id = msg.header.frame_id

    def _detect_and_publish(self):
        if self._latest_rgb is None or self._latest_depth is None or self._camera_model is None:
            return

        rgb = self._latest_rgb
        depth = self._latest_depth
        header = self._latest_rgb_header

        array_msg = Detection3DArray()
        array_msg.header.stamp = header.stamp
        array_msg.header.frame_id = self._world_frame

        for det in self._detector.detect(rgb, self._conf_threshold):
            point_world = self._to_world_point(det['pixel'], depth, header.stamp)
            if point_world is None:
                continue
            array_msg.detections.append(
                self._make_detection3d(det, point_world, header.stamp))

        self._detections_pub.publish(array_msg)

    def _to_world_point(self, pixel_uv, depth_image, stamp):
        u, v = pixel_uv
        row, col = int(round(v)), int(round(u))
        if not (0 <= row < depth_image.shape[0] and 0 <= col < depth_image.shape[1]):
            return None

        depth_value = float(depth_image[row, col])
        if not np.isfinite(depth_value) or depth_value <= 0.0:
            return None

        cx, cy, cz = pixel_to_camera_point(self._camera_model, u, v, depth_value)

        point_camera = PointStamped()
        point_camera.header.stamp = stamp
        point_camera.header.frame_id = self._camera_frame_id
        point_camera.point.x = cx
        point_camera.point.y = cy
        point_camera.point.z = cz

        try:
            transform = self._tf_buffer.lookup_transform(
                self._world_frame, self._camera_frame_id, Time.from_msg(stamp),
                timeout=Duration(seconds=0.2))
        except TransformException as ex:
            self.get_logger().warn(
                f'{self._world_frame} <- {self._camera_frame_id} tf 조회 실패, 검출 skip: {ex}',
                throttle_duration_sec=5.0)
            return None

        return do_transform_point(point_camera, transform)

    def _make_detection3d(self, det, point_world, stamp) -> Detection3D:
        detection = Detection3D()
        detection.header.stamp = stamp
        detection.header.frame_id = self._world_frame

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = det['label']
        hypothesis.hypothesis.score = det['score']
        hypothesis.pose.pose.position = point_world.point
        hypothesis.pose.pose.orientation.w = 1.0
        detection.results.append(hypothesis)

        detection.bbox.center.position = point_world.point
        detection.bbox.center.orientation.w = 1.0
        # 2D bbox 픽셀 크기를 그대로 xy 크기로 근사한 값일 뿐, 실제 3D extent가 아님.
        w_px, h_px = det['bbox_size_px']
        detection.bbox.size.x = float(w_px)
        detection.bbox.size.y = float(h_px)
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

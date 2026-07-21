"""perception_node
실행 계층 · D455 시뮬 센서 인식 (Hough Circle 스터드 검출, YOLO 버스바/너트 검출).

SUB /camera/color   (sensor_msgs/Image)  <- Isaac Sim D455
SUB /camera/depth   (sensor_msgs/Image)  <- Isaac Sim D455

PUB /vision/stud_pose      (fms_interfaces/StudPose)
PUB /vision/busbar_grasp   (fms_interfaces/BusbarGrasp)
PUB /vision/nut_pose       (fms_interfaces/NutPose)
"""
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from fms_interfaces.msg import StudPose, BusbarGrasp, NutPose

DETECTION_PERIOD_SEC = 0.5


class PerceptionNode(Node):

    def __init__(self):
        super().__init__('perception_node')

        self._bridge = CvBridge()
        self._latest_color = None
        self._latest_depth = None

        self._color_sub = self.create_subscription(
            Image, '/camera/color', self._on_color, 10)
        self._depth_sub = self.create_subscription(
            Image, '/camera/depth', self._on_depth, 10)

        self._stud_pub = self.create_publisher(StudPose, '/vision/stud_pose', 10)
        self._busbar_grasp_pub = self.create_publisher(BusbarGrasp, '/vision/busbar_grasp', 10)
        self._nut_pub = self.create_publisher(NutPose, '/vision/nut_pose', 10)

        self._timer = self.create_timer(DETECTION_PERIOD_SEC, self._detect_and_publish)

        self.get_logger().info('perception_node started')

    def _on_color(self, msg: Image):
        self._latest_color = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _on_depth(self, msg: Image):
        self._latest_depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _detect_and_publish(self):
        if self._latest_color is None:
            return

        for center in self._detect_studs_hough_circle(self._latest_color):
            self._publish_stud_pose(center)

        grasp_point = self._detect_busbar_grasp_yolo(self._latest_color)
        if grasp_point is not None:
            self._publish_busbar_grasp(grasp_point)

        for nut_id, center in self._detect_nuts_yolo(self._latest_color):
            self._publish_nut_pose(nut_id, center)

    # --- Hough Circle 기반 스터드 위치 검출 ------------------------------------
    def _detect_studs_hough_circle(self, color_image):
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=100, param2=30, minRadius=5, maxRadius=30)

        if circles is None:
            return []
        return [(x, y) for x, y, _r in np.round(circles[0, :]).astype(int)]

    # --- YOLO 기반 버스바 파지점 / 너트 위치 검출 --------------------------------
    def _detect_busbar_grasp_yolo(self, color_image):
        # TODO: 학습된 YOLO 모델로 버스바 파지점(bounding box 중심) 추론.
        return None

    def _detect_nuts_yolo(self, color_image):
        # TODO: 학습된 YOLO 모델로 너트 bounding box 목록 추론.
        return []

    # --- 픽셀 좌표 -> 3D pose 변환 -------------------------------------------
    def _pixel_to_pose(self, pixel_xy):
        # TODO: depth 이미지 + 카메라 intrinsic으로 실제 3D 좌표 역투영.
        pose = self._make_pose_stamped(0.0, 0.0, 0.0)
        return pose

    def _make_pose_stamped(self, x, y, z):
        from geometry_msgs.msg import PoseStamped
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'camera_color_optical_frame'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0
        return pose

    def _publish_stud_pose(self, pixel_xy):
        msg = StudPose()
        msg.id = 0
        msg.pose = self._pixel_to_pose(pixel_xy)
        self._stud_pub.publish(msg)

    def _publish_busbar_grasp(self, pixel_xy):
        msg = BusbarGrasp()
        msg.pose = self._pixel_to_pose(pixel_xy)
        self._busbar_grasp_pub.publish(msg)

    def _publish_nut_pose(self, nut_id, pixel_xy):
        msg = NutPose()
        msg.id = nut_id
        msg.pose = self._pixel_to_pose(pixel_xy)
        self._nut_pub.publish(msg)


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

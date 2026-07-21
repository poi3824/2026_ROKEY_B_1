"""D455 카메라의 RGB+depth 이미지를 구독해서 단자 홀의 3D 위치+각도를
계산하고 /vision/terminal_pose로 발행하는 ROS2 노드.

TODO: 아래는 실제 정보가 없어서 임시로 잡은 값. 우분투에서 확인 후 교체 필요.
- IMAGE_TOPIC / DEPTH_TOPIC: 실제 Isaac Sim 카메라 토픽 이름 확인 필요
  (`ros2 topic list`로 확인)
- depth 이미지 단위: pixel_to_world()는 "미터 단위 실수"를 가정하는데,
  실제 depth 토픽이 "밀리미터 단위 정수(16bit)"로 올 수 있음 -> 그 경우
  image_callback에서 depth_image를 미터로 변환(예: /1000.0)해야 함
- 각도->쿼터니언 변환: 카메라가 회전 없이 정확히 아래를 본다는 가정 하에,
  이미지에서 구한 각도를 월드 yaw로 그대로 사용함 (depth_estimation.py의
  CAMERA_WORLD_POSITION 가정과 동일한 전제)
"""
import math

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, Quaternion
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image

from depth_estimation import pixel_to_world_from_depth_image
from pose_estimation import detect_terminal_holes, draw_debug, pair_terminal_holes

IMAGE_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth/image_raw"
DEBUG_IMAGE_TOPIC = "/vision/debug_image"
POSE_TOPIC = "/vision/terminal_pose"


def _yaw_deg_to_quaternion(yaw_deg: float) -> Quaternion:
    """평면상 각도(도) -> Z축 회전 쿼터니언. (X,Y축 회전은 없다고 가정)"""
    half = math.radians(yaw_deg) / 2
    q = Quaternion()
    q.z = math.sin(half)
    q.w = math.cos(half)
    return q


class TerminalHoleDetectorNode(Node):
    def __init__(self):
        super().__init__("terminal_hole_detector")
        self.bridge = CvBridge()

        # RGB와 depth는 별개 토픽이라, 같은 순간 찍힌 것끼리 짝지어 받기
        # 위해 동기화(synchronize) 필요.
        rgb_sub = Subscriber(self, Image, IMAGE_TOPIC)
        depth_sub = Subscriber(self, Image, DEPTH_TOPIC)
        self.sync = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.05
        )
        self.sync.registerCallback(self.image_callback)

        self.debug_pub = self.create_publisher(Image, DEBUG_IMAGE_TOPIC, 10)
        self.pose_pub = self.create_publisher(PoseArray, POSE_TOPIC, 10)

    def image_callback(self, rgb_msg: Image, depth_msg: Image):
        image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        holes = detect_terminal_holes(image)
        pairs = pair_terminal_holes(image, holes)

        pose_array = PoseArray()
        pose_array.header = rgb_msg.header
        for p in pairs:
            u, v = int(round(p["mid"][0])), int(round(p["mid"][1]))
            wx, wy, wz = pixel_to_world_from_depth_image(u, v, depth_image)

            pose = Pose()
            pose.position.x = wx
            pose.position.y = wy
            pose.position.z = wz
            pose.orientation = _yaw_deg_to_quaternion(p["angle_deg"])
            pose_array.poses.append(pose)

        self.get_logger().info(
            f"detected {len(holes)} holes, {len(pairs)} pairs "
            f"-> published {len(pose_array.poses)} poses"
        )
        self.pose_pub.publish(pose_array)

        debug = draw_debug(image, holes)
        debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        debug_msg.header = rgb_msg.header
        self.debug_pub.publish(debug_msg)


def main():
    rclpy.init()
    node = TerminalHoleDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

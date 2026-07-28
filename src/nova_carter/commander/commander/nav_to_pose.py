import math
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.utilities import remove_ros_args

# station_1 pose (behavior_node.py의 STATION_POSES와 동일, Isaac Sim World0123_stations.usd 기준).
DEFAULT_POSE = (0.667, -0.038, -1.5708)


def yaw_to_quaternion(yaw: float):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def main(args=None):
    rclpy.init(args=args)

    argv = remove_ros_args(args=sys.argv)[1:]
    x, y, yaw = (float(v) for v in argv[:3]) if len(argv) >= 3 else DEFAULT_POSE

    navigator = BasicNavigator()
    navigator.waitUntilNav2Active()

    goal_pose = PoseStamped()
    goal_pose.header.frame_id = 'map'
    goal_pose.header.stamp = navigator.get_clock().now().to_msg()
    goal_pose.pose.position.x = x
    goal_pose.pose.position.y = y
    qx, qy, qz, qw = yaw_to_quaternion(yaw)
    goal_pose.pose.orientation.x = qx
    goal_pose.pose.orientation.y = qy
    goal_pose.pose.orientation.z = qz
    goal_pose.pose.orientation.w = qw

    navigator.get_logger().info(f'goToPose -> x={x:.3f} y={y:.3f} yaw={yaw:.3f}')
    navigator.goToPose(goal_pose)

    while not navigator.isTaskComplete():
        feedback = navigator.getFeedback()
        if feedback:
            navigator.get_logger().info(
                f'남은 거리: {feedback.distance_remaining:.2f} m')

    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        navigator.get_logger().info('목표 지점 도착')
    elif result == TaskResult.CANCELED:
        navigator.get_logger().warn('이동이 취소됨')
    elif result == TaskResult.FAILED:
        navigator.get_logger().error('이동 실패')

    navigator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

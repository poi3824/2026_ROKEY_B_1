"""amr_node
실행 계층 · 목표 스테이션으로 이동, 도착·이동·오류 상태 보고.

SUB /amr/goal      (fms_interfaces/AmrGoal)     <- behavior_node
PUB /amr/status    (fms_interfaces/AmrStatus)   -> behavior_node

PUB /cmd_vel       (geometry_msgs/Twist)        -> Isaac Sim (이동 명령, differential_drive가 구독)
SUB /odom          (nav_msgs/Odometry)          <- Isaac Sim (실제 위치 피드백)

목표 도달까지 회전->직진->최종정렬 3단계 P제어로 폐루프 주행한다(hardcoded_amr_drive.py에서
검증된 게인/최소속도 바닥값을 그대로 이식). 정지 마찰(stiction) 때문에 명령값이 너무 작으면
바퀴가 안 도는 문제가 있어, 오차가 남아있는데 명령이 min_speed보다 작으면 min_speed로 올린다.
"""
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from fms_interfaces.msg import AmrGoal, AmrStatus

CONTROL_HZ = 20.0

K_LINEAR = 0.6
K_ANGULAR = 1.5

MAX_LINEAR_SPEED = 0.4
MAX_ANGULAR_SPEED = 0.8

# 바퀴 정지 마찰(stiction) 회피용 바닥값 - hardcoded_amr_drive.py에서 실측으로 검증된 값.
MIN_LINEAR_SPEED = 0.12
MIN_ANGULAR_SPEED = 0.25

ROTATE_TO_TARGET_TOLERANCE_RAD = 0.08
XY_TOLERANCE_M = 0.08
FINAL_YAW_TOLERANCE_RAD = 0.08
DRIVE_REALIGN_THRESHOLD_RAD = ROTATE_TO_TARGET_TOLERANCE_RAD * 2.5

GOAL_TIMEOUT_SEC = 30.0  # 이 시간 안에 도착 못 하면 ERROR 처리(안전장치).
REVERSE_SELECTION_MARGIN_RAD = math.radians(20.0)
PROGRESS_LOG_INTERVAL_SEC = 1.0

# BUS1은 실제 파지 위치이므로 일반 이동보다 천천히, 더 작은 위치 오차까지 접근한다.
BUSBAR_XY_TOLERANCE_M = 0.03
BUSBAR_MAX_LINEAR_SPEED = 0.12
BUSBAR_MIN_LINEAR_SPEED = 0.08
PRECISE_POSITION_WAYPOINTS = {
    'busbar_dock_approach',
    'busbar_dock_exit',
    'busbar_table',
    'busbar_2_approach',
    'busbar_table_2',
    'busbar_3_approach',
    'busbar_table_3',
}

# 현재 busbar_table에서는 실제 팔 파지 대신 2초 STUB만 수행한다. 테이블 바로 앞에서
# 최종 yaw를 맞추기 위한 제자리 회전은 차체/팔의 회전 반경으로 테이블을 밀 수 있으므로,
# 위치 도착 즉시 정지하고 접근 방향을 유지한다. 실제 파지를 연결할 때는 테이블에서 떨어진
# 사전 접근점에서 먼저 자세를 맞추는 도킹 경로를 구현한 뒤 이 항목을 제거해야 한다.
POSITION_ONLY_WAYPOINTS = {
    'busbar_table',
    'busbar_table_2',
    'busbar_table_3',
    'busbar_dock_exit',
    'busbar_return_mid',
}

# 장애물 바로 옆에서 목표 방향을 향해 먼저 제자리 회전하거나 후진하지 않고,
# linear.x와 angular.z를 동시에 주어 전진 곡선으로 빠져나가야 하는 안전 경유지.
FORWARD_ARC_WAYPOINTS = {
    'scan_3_exit',
    'scan_4_approach',
    'scan_station_4',
    'scan_station_5',
    'scan_station_6',
    'scan_6_right_exit',
    'scan_6_top_right',
    'busbar_return_approach',
    'busbar_return_mid',
    'station_1_return',
}
ARC_LINEAR_SPEED = 0.12

# IsaacComputeOdometry의 /odom은 아래 월드 HOME에서 (0, 0, 0)으로 시작한다. 목표는
# 사용자가 Isaac Property 패널에서 지정한 월드 좌표로 받으므로, odom 측정값만 월드
# 좌표로 변환한 뒤 폐루프 오차를 계산한다.
HOME_WORLD_X = -1.00983
HOME_WORLD_Y = -0.24722
HOME_WORLD_YAW = -math.pi / 2

PHASE_ROTATE_TO_TARGET = 'ROTATE_TO_TARGET'
PHASE_DRIVE = 'DRIVE'
PHASE_ROTATE_TO_FINAL = 'ROTATE_TO_FINAL'
PHASE_DONE = 'DONE'


def with_min_speed(command: float, min_speed: float) -> float:
    """오차가 남아있는데(0이 아닌데) 명령값 크기가 min_speed보다 작으면 부호는 유지한 채
    min_speed로 올려서, 정지 마찰 때문에 바퀴가 안 도는 것을 방지한다."""
    if command == 0.0:
        return 0.0
    if abs(command) < min_speed:
        return math.copysign(min_speed, command)
    return command


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def odom_pose_to_world(x: float, y: float, yaw: float):
    """HOME 상대 /odom pose를 Isaac Sim 월드 pose로 변환한다."""
    cos_yaw = math.cos(HOME_WORLD_YAW)
    sin_yaw = math.sin(HOME_WORLD_YAW)
    world_x = HOME_WORLD_X + cos_yaw * x - sin_yaw * y
    world_y = HOME_WORLD_Y + sin_yaw * x + cos_yaw * y
    world_yaw = normalize_angle(HOME_WORLD_YAW + yaw)
    return world_x, world_y, world_yaw


def select_drive_direction(target_heading: float, current_yaw: float) -> int:
    """목표까지 전진/후진할 때 필요한 회전 중 더 작은 쪽을 선택한다.

    반환값 +1은 전진, -1은 후진이다. 후진을 선택하면 차체가 바라볼 목표 방향은
    target_heading + pi가 되고 /cmd_vel linear.x는 음수가 된다.
    """
    forward_error = abs(normalize_angle(target_heading - current_yaw))
    reverse_error = abs(normalize_angle(target_heading + math.pi - current_yaw))
    # 두 방향 차이가 작으면 불필요한 후진을 피하고 전진을 유지한다.
    return -1 if reverse_error + REVERSE_SELECTION_MARGIN_RAD < forward_error else 1


def clamp(value: float, max_abs: float) -> float:
    return max(-max_abs, min(max_abs, value))


class AmrNode(Node):

    def __init__(self):
        super().__init__('amr_node')

        self._goal_sub = self.create_subscription(AmrGoal, '/amr/goal', self._on_goal, 10)
        self._status_pub = self.create_publisher(AmrStatus, '/amr/status', 10)
        self._cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._odom_sub = self.create_subscription(Odometry, '/odom', self._on_odom, 10)

        self._current_pose = None  # (x, y, yaw), /odom 수신 전엔 None
        self._goal = None
        self._phase = PHASE_DONE
        self._goal_start_time = None
        self._drive_direction = None  # +1 전진, -1 후진. goal마다 한 번 선택해 유지.
        self._last_progress_log_time = None
        self._last_logged_phase = None

        self._control_timer = self.create_timer(1.0 / CONTROL_HZ, self._control_loop)

        self.get_logger().info(
            '\n' + '=' * 72 +
            '\n[AMR] 주행 제어 노드 시작 — /amr/goal을 기다립니다.' +
            '\n' + '=' * 72)

    def _on_odom(self, msg: Odometry):
        odom_x = msg.pose.pose.position.x
        odom_y = msg.pose.pose.position.y
        odom_yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self._current_pose = odom_pose_to_world(odom_x, odom_y, odom_yaw)

    def _on_goal(self, msg: AmrGoal):
        self.get_logger().info(
            '\n' + '=' * 72 +
            f'\n[새 이동 시작] 목적지: {msg.station_id}' +
            f'\n목표 좌표: X={msg.x:.3f}, Y={msg.y:.3f}, '
            f'방향={math.degrees(msg.theta):.1f}°' +
            '\n' + '=' * 72)
        self._goal = msg
        self._phase = (
            PHASE_DRIVE
            if msg.station_id in FORWARD_ARC_WAYPOINTS
            else PHASE_ROTATE_TO_TARGET
        )
        self._goal_start_time = self.get_clock().now()
        self._drive_direction = None
        self._last_progress_log_time = None
        self._last_logged_phase = None
        self._publish_status(AmrStatus.STATE_MOVING, msg.station_id, '이동 중')

    def _control_loop(self):
        if self._phase == PHASE_DONE or self._goal is None:
            return

        if self._current_pose is None:
            return  # 아직 /odom을 한 번도 못 받음 - Isaac Sim 연동 확인 필요.

        elapsed = (self.get_clock().now() - self._goal_start_time).nanoseconds / 1e9
        if elapsed > GOAL_TIMEOUT_SEC:
            self.get_logger().error(f'{self._goal.station_id} 이동 타임아웃({GOAL_TIMEOUT_SEC}s)')
            self._phase = PHASE_DONE
            self._cmd_vel_pub.publish(Twist())
            self._publish_status(AmrStatus.STATE_ERROR, self._goal.station_id, '이동 타임아웃')
            return

        x, y, yaw = self._current_pose
        dx = self._goal.x - x
        dy = self._goal.y - y
        distance = math.hypot(dx, dy)
        xy_tolerance = (
            BUSBAR_XY_TOLERANCE_M
            if self._goal.station_id in PRECISE_POSITION_WAYPOINTS
            else XY_TOLERANCE_M
        )
        target_heading = math.atan2(dy, dx)

        # 목표가 차체 뒤쪽이면 제자리에서 90~180도 크게 돌지 않고 후진한다. 좁은 ST1/BUS1
        # 작업점에서 차체와 팔이 크게 휘돌며 배터리팩/테이블에 닿는 것을 줄인다.
        if self._drive_direction is None:
            if self._goal.station_id in FORWARD_ARC_WAYPOINTS:
                self._drive_direction = 1
            else:
                self._drive_direction = select_drive_direction(target_heading, yaw)
            direction_name = '전진' if self._drive_direction > 0 else '후진'
            self.get_logger().info(
                f'[주행 방향] {direction_name} 선택 | '
                f'목적지={self._goal.station_id}')

        travel_heading = target_heading
        if self._drive_direction < 0:
            travel_heading = normalize_angle(target_heading + math.pi)
        heading_error = normalize_angle(travel_heading - yaw)

        twist = Twist()

        if self._phase == PHASE_ROTATE_TO_TARGET:
            if distance < xy_tolerance:
                if self._complete_position_only_goal():
                    return
                self._phase = PHASE_ROTATE_TO_FINAL
            elif abs(heading_error) < ROTATE_TO_TARGET_TOLERANCE_RAD:
                self._phase = PHASE_DRIVE
            else:
                cmd = clamp(K_ANGULAR * heading_error, MAX_ANGULAR_SPEED)
                twist.angular.z = with_min_speed(cmd, MIN_ANGULAR_SPEED)

        elif self._phase == PHASE_DRIVE:
            if distance < xy_tolerance:
                if self._complete_position_only_goal():
                    return
                self._phase = PHASE_ROTATE_TO_FINAL
            elif (
                self._goal.station_id in FORWARD_ARC_WAYPOINTS
                and abs(heading_error) > DRIVE_REALIGN_THRESHOLD_RAD
            ):
                # 장애물 옆에서는 선행 제자리 회전 대신 저속 전진과 회전을 동시에 명령한다.
                # 회전 오차가 줄어들면 아래 일반 DRIVE 제어로 자연스럽게 넘어간다.
                twist.linear.x = ARC_LINEAR_SPEED
                cmd = clamp(K_ANGULAR * heading_error, MAX_ANGULAR_SPEED)
                twist.angular.z = with_min_speed(cmd, MIN_ANGULAR_SPEED)
            elif abs(heading_error) > DRIVE_REALIGN_THRESHOLD_RAD:
                self._phase = PHASE_ROTATE_TO_TARGET
            else:
                max_linear = MAX_LINEAR_SPEED
                min_linear = MIN_LINEAR_SPEED
                if self._goal.station_id in PRECISE_POSITION_WAYPOINTS:
                    max_linear = BUSBAR_MAX_LINEAR_SPEED
                    min_linear = BUSBAR_MIN_LINEAR_SPEED
                lin_cmd = clamp(K_LINEAR * distance, max_linear)
                linear_speed = with_min_speed(lin_cmd, min_linear)
                twist.linear.x = self._drive_direction * linear_speed
                ang_cmd = clamp(K_ANGULAR * heading_error, MAX_ANGULAR_SPEED * 0.5)
                twist.angular.z = with_min_speed(ang_cmd, MIN_ANGULAR_SPEED * 0.5)

        elif self._phase == PHASE_ROTATE_TO_FINAL:
            final_error = normalize_angle(self._goal.theta - yaw)
            if abs(final_error) < FINAL_YAW_TOLERANCE_RAD:
                self._phase = PHASE_DONE
                self._cmd_vel_pub.publish(Twist())
                self._publish_status(AmrStatus.STATE_ARRIVED, self._goal.station_id, '도착')
                return
            cmd = clamp(K_ANGULAR * final_error, MAX_ANGULAR_SPEED)
            twist.angular.z = with_min_speed(cmd, MIN_ANGULAR_SPEED)

        self._cmd_vel_pub.publish(twist)
        self._log_drive_progress(x, y, yaw, distance, heading_error, twist)

    def _log_drive_progress(self, x, y, yaw, distance, heading_error, twist):
        """주행 상황을 사람이 바로 읽을 수 있도록 1초 간격으로 요약한다."""
        now = self.get_clock().now()
        phase_changed = self._phase != self._last_logged_phase
        interval_elapsed = (
            self._last_progress_log_time is None or
            (now - self._last_progress_log_time).nanoseconds / 1e9
            >= PROGRESS_LOG_INTERVAL_SEC
        )
        if not phase_changed and not interval_elapsed:
            return

        phase_label = {
            PHASE_ROTATE_TO_TARGET: '목표 방향으로 회전 중',
            PHASE_DRIVE: '목표 위치로 주행 중',
            PHASE_ROTATE_TO_FINAL: '최종 방향 정렬 중',
            PHASE_DONE: '도착',
        }.get(self._phase, self._phase)
        direction = '전진' if self._drive_direction != -1 else '후진'
        self.get_logger().info(
            f'[주행 현황] {phase_label} | {direction} | '
            f'현재=({x:.3f}, {y:.3f}, {math.degrees(yaw):.1f}°) | '
            f'남은 거리={distance:.3f}m | '
            f'방향 오차={math.degrees(heading_error):+.1f}° | '
            f'cmd_vel: 직진={twist.linear.x:+.3f}, 회전={twist.angular.z:+.3f}')
        self._last_progress_log_time = now
        self._last_logged_phase = self._phase

    def _complete_position_only_goal(self) -> bool:
        """위치 전용 waypoint면 최종 제자리 회전 없이 즉시 정지·도착 처리한다."""
        if self._goal.station_id not in POSITION_ONLY_WAYPOINTS:
            return False
        self._phase = PHASE_DONE
        self._cmd_vel_pub.publish(Twist())
        self.get_logger().info(
            f'[정밀 도착] {self._goal.station_id} 위치 도착 | '
            '정지 명령 전송 | 테이블 앞 추가 회전 생략')
        self._publish_status(
            AmrStatus.STATE_ARRIVED, self._goal.station_id, '위치 도착')
        return True

    def _publish_status(self, state: int, station_id: str, message: str):
        status = AmrStatus()
        status.state = state
        status.station_id = station_id
        status.message = message
        self._status_pub.publish(status)
        self.get_logger().info(f'PUB /amr/status -> {station_id} state={state}')


def main(args=None):
    rclpy.init(args=args)
    node = AmrNode()
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

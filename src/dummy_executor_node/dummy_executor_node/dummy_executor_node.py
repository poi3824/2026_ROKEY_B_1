"""dummy_executor_node

GPU/Isaac Sim 없이 behavior_node + fleet_manager_node의 FSM을 검증하기 위한
더미 실행 노드. arm_node · amr_node · perception_node가 실제로 발행할 토픽을
대신 발행하고, 이들이 구독할 커맨드 토픽을 구독해 즉시 응답을 준다.

사용법
------
1) fms_interfaces, fleet_manager_node, behavior_node, 그리고 이 노드가 속한
   패키지(예: dummy_executor_node 패키지)를 colcon build.
2) 터미널 3개에서 각각 실행:
     ros2 run fleet_manager_node fleet_manager_node
     ros2 run behavior_node behavior_node
     ros2 run dummy_executor_node dummy_executor_node
3) behavior_node 로그를 보면서 FSM이
     IDLE -> MOVE_TO_STATION -> WAIT_BUSBAR_VISION -> GRASP_BUSBAR
     -> INSERT_BUSBAR -> WAIT_NUT_VISION -> FASTEN_APPROACH -> FASTEN -> REPORT
   순서로 도는지 확인한다.

실패/복구 로직 테스트
--------------------
아래 FAIL_STAGES 집합에 값을 넣으면 해당 단계에서 일부러 실패 응답을 보낸다.
  - 'GRASP'  : 버스바 파지 실패
  - 'INSERT' : 버스바 삽입 실패
  - 'APPROACH' : 체결 접근 실패
  - 'FASTEN' : 체결 실패
  - 'MOVE'   : AMR 이동 실패 (실제 프로토콜 값은 아니고, 이 더미 노드 내부에서만
               쓰는 테스트용 표시자)
behavior_node의 MAX_RETRY(3회)와 재시도 로직이 정상 동작하는지, 재시도 초과 시
/fleet/report 에 success=False 로 보고되는지까지 확인할 수 있다.
"""
import rclpy
from rclpy.node import Node

from fms_interfaces.msg import (
    AmrGoal, AmrStatus,
    BusbarCommand, BusbarResult,
    FastenCommand, FastenResult,
    StudPose, NutPose, BusbarGrasp,
)

# 테스트하고 싶은 실패 단계를 여기 넣는다. 예: FAIL_STAGES = {'FASTEN'}
FAIL_STAGES: set[str] = {'GRASP'}


class DummyExecutorNode(Node):
    """arm_node · amr_node · perception_node를 대신하는 더미 실행 노드."""

    def __init__(self):
        super().__init__('dummy_executor_node')

        # perception_node 대신: 비전 인식 결과를 주기적으로 흘려보낸다.
        self._busbar_grasp_pub = self.create_publisher(BusbarGrasp, '/vision/busbar_grasp', 10)
        self._stud_pose_pub = self.create_publisher(StudPose, '/vision/stud_pose', 10)
        self._nut_pose_pub = self.create_publisher(NutPose, '/vision/nut_pose', 10)
        self.create_timer(1.0, self._publish_vision)

        # amr_node 대신: 이동 목표를 받아 도착/오류 상태를 보고한다.
        self._amr_status_pub = self.create_publisher(AmrStatus, '/amr/status', 10)
        self.create_subscription(AmrGoal, '/amr/goal', self._on_amr_goal, 10)

        # arm_node 대신: 버스바/체결 커맨드를 받아 결과를 보고한다.
        self._busbar_result_pub = self.create_publisher(BusbarResult, '/busbar/result', 10)
        self._fasten_result_pub = self.create_publisher(FastenResult, '/fasten/result', 10)
        self.create_subscription(BusbarCommand, '/busbar/command', self._on_busbar_command, 10)
        self.create_subscription(FastenCommand, '/fasten/command', self._on_fasten_command, 10)

        self.get_logger().info(
            f'dummy_executor_node started (GPU/Isaac Sim 없이 FSM 검증용) '
            f'FAIL_STAGES={FAIL_STAGES or "없음"}')

    # --- perception_node 대역 ------------------------------------------------
    def _publish_vision(self):
        grasp = BusbarGrasp()
        grasp.pose.pose.position.x, grasp.pose.pose.position.y = 0.10, 0.05
        self._busbar_grasp_pub.publish(grasp)

        stud = StudPose()
        stud.id = 1
        self._stud_pose_pub.publish(stud)

        nut = NutPose()
        nut.id = 1
        self._nut_pose_pub.publish(nut)

    # --- amr_node 대역 --------------------------------------------------------
    def _on_amr_goal(self, msg: AmrGoal):
        fail = 'MOVE' in FAIL_STAGES
        status = AmrStatus()
        status.station_id = msg.station_id
        status.state = AmrStatus.STATE_ERROR if fail else AmrStatus.STATE_ARRIVED
        status.message = '더미 이동 실패 주입' if fail else '더미 도착 완료'
        self._amr_status_pub.publish(status)
        self.get_logger().info(
            f'[dummy amr] /amr/goal <- {msg.station_id} -> state={status.state}')

    # --- arm_node 대역 (버스바) -------------------------------------------------
    def _on_busbar_command(self, msg: BusbarCommand):
        result = BusbarResult()
        result.success = msg.command not in FAIL_STAGES
        result.message = f'{msg.command} 더미 {"성공" if result.success else "실패"}'
        self._busbar_result_pub.publish(result)
        self.get_logger().info(
            f'[dummy arm] /busbar/command <- {msg.command} -> success={result.success}')

    # --- arm_node 대역 (체결) --------------------------------------------------
    def _on_fasten_command(self, msg: FastenCommand):
        result = FastenResult()
        result.success = msg.command not in FAIL_STAGES
        result.torque = 12.5
        result.message = f'{msg.command} 더미 {"성공" if result.success else "실패"}'
        self._fasten_result_pub.publish(result)
        self.get_logger().info(
            f'[dummy arm] /fasten/command <- {msg.command} -> success={result.success}')


def main(args=None):
    rclpy.init(args=args)
    node = DummyExecutorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

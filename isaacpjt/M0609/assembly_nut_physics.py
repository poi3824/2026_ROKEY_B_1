"""
01_pick_and_lift.py / 10_busbar_assembly.py (Physically Active Nut Gripping Version + Vision + 버스바 복원)
─ AMR 베이스 잠금
─ [재시작 초기화 보정] Stop 후 Play 시 너트 및 버스바 위치 초기화 강제 적용
─ [너트 물리 유지] 너트 파지 시 disable_physics_recursively 및 glue_nut_to_ee 미사용 (실제 물리 파지)
─ [버스바 복원] origin/assembly의 원래 버전은 버스바 시퀀스를 우회했으나(재시작 핸들러는
  원래부터 BUSBAR_APPROACH로 리셋했음 - 우회는 초기 phase 한 줄뿐이었다) 여기서는 복원해
  버스바 장착 -> 너트1 체결 -> 너트2 체결 전체 파이프라인을 돈다.
─ [비전 연동, 폴백 없음] rclpy로 별도 프로세스의 perception_node(busbar_cam/bolt_cam 각
  인스턴스)를 표준 geometry_msgs/PoseStamped 토픽으로 구독(fms_interfaces 커스텀
  msg/srv는 Isaac 내장 rclpy와 Python 버전이 달라 import 불가 - 실측 확인) -
  BUSBAR_APPROACH_POS/BUSBAR_PICK_POS는 /vision/busbar_grasp_pose, BOLT1_POS/BOLT2_POS는
  /vision/bolt_a_pose·bolt_b_pose로 대체한다. 하드코딩 값으로 조용히 넘어가지 않는다 -
  검출될 때까지 무기한 대기하고, 검출되면 하드코딩 값과의 차이(mm)만 로그로 남긴다
  (resolve_vision_positions). 같은 프로세스에 YOLO(ultralytics/torch)를 얹지 않는다 -
  yaw_sweep_render.py가 이미 문서화한 대로 Isaac Sim 번들
  파이썬과 의존성 충돌 위험이 있어, execute_isaac.py와 동일하게 rclpy만 이 프로세스에
  두고 실제 추론은 별도 python3 프로세스(perception_node)에 맡긴다.
─ [수정 완료] Screwing Regrasp 시 상공 상승(+0.05m) 후 역회전/하강 적용
─ [수정 완료] 너트 1, 2번 체결 후 상공(Z=0.8m)에서 6번 조인트를 정확히 0도가 되도록 Unwind 적용
─ [추가 완료] TCP Z 높이 0.37m 이하 조건 추가: 토크 및 Z축 Sticking 감지 시 Screwing 조기 종료 및 그리퍼 해제 (튕김 방지)
─ [수정 완료] Screwing Regrasp 하강/재파지 시 Z축 높이를 미세(3mm) 상승 적용
─ [수정 완료] Nut1 체결 후 Retract 시 오리엔테이션 매개변수 오류 수정 및 팔 꼬임 방지 보정
"""

import os
import sys
import math
import gc
import time
from pathlib import Path

from isaacsim import SimulationApp

# Headless 모드 설정 (환경변수 AMR_HEADLESS=1)
_HEADLESS = os.environ.get("AMR_HEADLESS") == "1"
simulation_app = SimulationApp({"headless": _HEADLESS})

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.ros2_bridge")

simulation_app.update()

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf
from scipy.spatial.transform import Rotation as R

from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.core.utils.types import ArticulationAction

# 비전 연동: 같은 프로세스에 YOLO(ultralytics/torch)를 얹지 않는다 - isaacpjt/M0609/
# yaw_sweep_render.py가 이미 문서화한 대로 Isaac Sim 번들 파이썬과 의존성이 충돌할
# 위험이 있다. execute_isaac.py와 동일한 패턴으로 rclpy만 이 프로세스에 두고, 실제
# YOLO 추론은 별도 python3 프로세스(src/perception_node, ultralytics 포함)에 맡긴 뒤
# ROS2 토픽/서비스로 결과만 받는다.
import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
# 주의: fms_interfaces(커스텀 msg/srv)는 여기서 import하지 않는다. Isaac Sim 내장
# rclpy(kit python 3.11)는 시스템 콜론 워크스페이스(python 3.10용으로 빌드됨)의
# 커스텀 패키지를 ABI 불일치로 import할 수 없다(source setup.bash로도 해결 안 됨,
# 내장 rclpy는 PYTHONPATH를 참조하지 않는다 - 실측 확인됨: ModuleNotFoundError).
# 그래서 perception_node가 병행 발행하는 표준 geometry_msgs/PoseStamped 토픽만 쓴다.

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "rmpflow"))
from m0609_rmpflow_controller import RMPFlowController  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════
#  [A] 설정 및 파라미터
# ══════════════════════════════════════════════════════════════════════════
USD_PATH = "/home/rokey/EV_combine/src/Collected_World_0123/World0123.usd"

NOVA_CARTER_ROOT = "/World/Nova_Carter/chassis_link"
M0609_PATH       = "/World/m0609"
EE_LINK_NAME     = "link_6"
GRIPPER_JOINTS   = ["finger_joint", "right_inner_knuckle_joint"]

# 버스바 Prim 경로
BUSBAR_ROOT_PATH      = "/World/busbar"
BUSBAR_POLYSHAPE_PATH = "/World/busbar/geo/PolyShape"

# 너트 Prim 경로
NUT1_ROOT_PATH      = "/World/nut1"
NUT2_ROOT_PATH      = "/World/nut2"
NUT1_POLYSHAPE_PATH = "/World/nut1/geo/PolyShape"
NUT2_POLYSHAPE_PATH = "/World/nut2/geo/PolyShape"

# 그리퍼 파라미터
GRIPPER_OPEN      = np.array([0.0, 0.0])
GRIPPER_CLOSE     = np.array([0.8, 0.8])
GRIPPER_CLOSE_NUT = np.array([0.96, 0.96])
GRIPPER_DELTA     = np.array([-0.5, -0.5])
GRIP_CLOSE_RAMP_STEPS = 50

# Kinematic Pose-Glue 파라미터
EE_OFFSET = np.array([0.0, 0.0, 0.185])
BUSBAR_HEIGHT = 0.003
BUSBAR_GRASP_Z_LOCAL = BUSBAR_HEIGHT + 0.02
BUSBAR_REST_ORIENTATION = np.array([0.5, -0.5, 0.5, 0.5])

NUT_HEIGHT = 0.0095
NUT_GRASP_Z_LOCAL = NUT_HEIGHT + 0.023

# ★ 버스바 및 체결 중심 파라미터 (하드코딩 기본값 - resolve_vision_positions가
# busbar_cam 검출값이 있으면 덮어씀) ★
_POS_GRAB_PICK          = np.array([0.5136, 0.7299, 0.455])
BUSBAR_APPROACH_POS     = _POS_GRAB_PICK + np.array([0.0, 0.0, 0.145])
BUSBAR_PICK_POS         = _POS_GRAB_PICK.copy()
BUSBAR_LIFT_MOVE_POS    = _POS_GRAB_PICK + np.array([0.0, 0.1, 0.145])

target_mid_pos          = np.array([1.1606, 0.1836, 0.0693])
TARGET_DESTINATION_POS  = np.array([target_mid_pos[0], target_mid_pos[1], 0.6])
TARGET_INSERT_POS       = np.array([target_mid_pos[0], target_mid_pos[1], target_mid_pos[2]])

# ★ ArmNode 원본 오차 조건 ★
PICK_TOLERANCE_STRICT   = 0.01     # Pick 단계: 0.01m (10mm)
INSERT_TOLERANCE_STRICT = 0.001    # Insert 단계: 0.001m (1mm)
BUSBAR_RELEASE_Z        = 0.37     # 그리퍼 해제 임계 높이
INSERT_SPEED            = 0.0005   # Step당 수직 하강 거리

PICK_TOLERANCE_LOOSE_VAL = 0.015
MAX_STUCK_STEPS          = 60

# ★ 볼트 좌표 ★
BOLT1_POS = np.array([1.0576, 0.3653, 0.152])
BOLT2_POS = np.array([1.2636, 0.0019, 0.152])

# ★ 너트 1번 좌표 (nut1 -> bolt1) ★
NUT_APPROACH_Z    = 0.8
NUT1_PICK_POS     = np.array([0.5746, -0.1008, 0.72 - (NUT_GRASP_Z_LOCAL - 0.0395)])
NUT1_APPROACH_POS = np.array([NUT1_PICK_POS[0], NUT1_PICK_POS[1], NUT_APPROACH_Z])
BOLT1_APPROACH_POS = np.array([BOLT1_POS[0], BOLT1_POS[1], 0.6])
BOLT1_TOUCH_POS    = np.array([BOLT1_POS[0], BOLT1_POS[1], BOLT1_POS[2] + EE_OFFSET[2] + NUT_GRASP_Z_LOCAL])

# ★ 너트 2번 좌표 (nut2 -> bolt2) ★
NUT2_PICK_POS     = np.array([0.6643, -0.1031, 0.72 - (NUT_GRASP_Z_LOCAL - 0.0395)])
NUT2_APPROACH_POS = np.array([NUT2_PICK_POS[0], NUT2_PICK_POS[1], NUT_APPROACH_Z])
BOLT2_APPROACH_POS = np.array([BOLT2_POS[0], BOLT2_POS[1], 0.6])
BOLT2_TOUCH_POS    = np.array([BOLT2_POS[0], BOLT2_POS[1], BOLT2_POS[2] + EE_OFFSET[2] + NUT_GRASP_Z_LOCAL])

# 체결(SCREW) 파라미터
ENGAGE_LEN        = 0.0125    # 체결 깊이 (12.5mm)
SCREW_TURNS_DEG   = 350.0     # 1회전당 350도
REGRASP_CYCLES    = 1         # 총 2회전
SCREW_OMEGA_DEG_S = 120.0     # 초당 120도 회전
PHYSICS_DT        = 1.0 / 60.0
REGRASP_LIFT_HEIGHT = 0.05    # Regrasp 시 수직 상승 높이 (5cm)
REGRASP_Z_OFFSET    = 0.005   # Regrasp 하강/재파지 시 너트를 잡는 위치 보정 높이 (3mm 상승)

TOTAL_REV   = (SCREW_TURNS_DEG / 360.0) * (1 + REGRASP_CYCLES)
NUT_PITCH_M = ENGAGE_LEN / TOTAL_REV

# ★ 완착/토크 감지 파라미터 ★
TORQUE_THRESHOLD      = 45.0   # 6번 조인트 반력 임계값 (Nm)
STUCK_Z_DELTA_THRESH  = 0.0001  # Z축 하강 멈춤 판정 기준 (0.1mm)
STUCK_STEP_LIMIT      = 12      # Z축 변화 없이 토크 지속되는 Step 수
TCP_FORCE_CHECK_Z     = 0.378   # TCP(EE) 높이가 0.37m 이하일 때만 힘/토크 감지 활성화

URDF_PATH        = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
ROBOT_DESC_PATH  = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
RMPFLOW_CFG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")

# ★ 비전 연동 파라미터 ★ src/perception_node를 busbar_cam/bolt_cam 각각 별도
# 인스턴스로(rgb_topic/depth_topic/camera_info_topic/camera_frame_override 파라미터만
# 다르게, node 이름도 다르게) 띄워둬야 아래 토픽이 채워진다 - 실행 방법은
# reposition_bolt_camera.py 옆의 실행 안내 참고. 하드코딩 좌표는 더 이상 폴백(대체
# 사용)이 아니라 비교 기준으로만 쓴다 - 비전이 응답할 때까지 무기한 대기한다.
# 퍼셉션이 안 떠 있으면 여기서 계속 멈춰 있는 게 의도된 동작(하드코딩으로 몰래
# 넘어가서 비전이 실제로 동작하는지 착각하는 걸 막기 위함). 서비스(get_bolt_pair)
# 대신 표준 geometry_msgs/PoseStamped 토픽만 쓴다 - fms_interfaces는 Isaac 내장
# rclpy에서 import 불가(파일 상단 주석 참고).
# 토픽명이 인스턴스 구분 없이 perception_node.py에 하드코딩돼 있어(퍼블리셔 코드
# 참고), busbar_cam/bolt_cam 두 인스턴스를 네임스페이스 없이 그냥 띄우면 서로 다른
# 카메라의 검출값이 같은 토픽에 뒤섞여 들어온다(실측: rosbag에서 busbar_grasp_pose에
# 두 개의 완전히 다른 좌표가 500ms 주기로 번갈아 찍힘 -> "파지 위치가 무작위" 증상의
# 원인). 그래서 각 perception_node 인스턴스는 -r __ns:=/busbar_cam,
# -r __ns:=/bolt_cam 로 띄우고(실행 방법 문서 참고), 여기서는 그 네임스페이스가
# 붙은 토픽만 구독한다.
BUSBAR_GRASP_POSE_TOPIC = "/busbar_cam/vision/busbar_grasp_pose"
BOLT_A_POSE_TOPIC = "/bolt_cam/vision/bolt_a_pose"
BOLT_B_POSE_TOPIC = "/bolt_cam/vision/bolt_b_pose"
VISION_POLL_LOG_SEC = 3.0   # "아직 대기 중" 로그 출력 간격


class VisionBridge(Node):
    """assembly_nut_physics.py 전용 최소 rclpy 노드. YOLO 추론은 하지 않고,
    별도 프로세스의 perception_node가 이미 계산해 발행하는 world 좌표(표준
    geometry_msgs/PoseStamped)만 구독한다(같은 프로세스에 ultralytics를 얹지 않는
    이유, fms_interfaces를 안 쓰는 이유는 파일 상단 주석 참고)."""

    def __init__(self):
        super().__init__("assembly_vision_bridge")
        self.latest_busbar_xy = None  # (x, y) world, 최신 수신값
        self.latest_bolt_a_xy = None
        self.latest_bolt_b_xy = None
        self._busbar_sub = self.create_subscription(
            PoseStamped, BUSBAR_GRASP_POSE_TOPIC, self._on_busbar_pose, 10)
        self._bolt_a_sub = self.create_subscription(
            PoseStamped, BOLT_A_POSE_TOPIC, self._on_bolt_a_pose, 10)
        self._bolt_b_sub = self.create_subscription(
            PoseStamped, BOLT_B_POSE_TOPIC, self._on_bolt_b_pose, 10)

    def _on_busbar_pose(self, msg: PoseStamped):
        self.latest_busbar_xy = (msg.pose.position.x, msg.pose.position.y)

    def _on_bolt_a_pose(self, msg: PoseStamped):
        self.latest_bolt_a_xy = (msg.pose.position.x, msg.pose.position.y)

    def _on_bolt_b_pose(self, msg: PoseStamped):
        self.latest_bolt_b_xy = (msg.pose.position.x, msg.pose.position.y)


def resolve_vision_positions(vision_node: "VisionBridge"):
    """시작 시 1회, 비전 실측 좌표를 받을 때까지 무기한 대기한 뒤 BUSBAR_*_POS/
    BOLT*_POS 및 파생 좌표를 덮어쓴다. 폴백 없음 - busbar_cam/bolt_cam
    perception_node가 응답하기 전까지는 여기서 멈춰 있는다. 원래 하드코딩 값은
    실제 목표로는 쓰지 않고, 검출값과 비교해 차이(mm)를 로그로만 남긴다."""
    global BUSBAR_APPROACH_POS, BUSBAR_PICK_POS, BUSBAR_LIFT_MOVE_POS
    global BOLT1_POS, BOLT2_POS, BOLT1_APPROACH_POS, BOLT1_TOUCH_POS
    global BOLT2_APPROACH_POS, BOLT2_TOUCH_POS

    hardcoded_busbar_xy = BUSBAR_PICK_POS[:2].copy()
    hardcoded_bolt1_xy = BOLT1_POS[:2].copy()
    hardcoded_bolt2_xy = BOLT2_POS[:2].copy()

    print("[vision] busbar_cam 검출 대기 중 (perception_node busbar_cam 인스턴스 필요)...")
    t0, last_log = time.time(), 0.0
    while vision_node.latest_busbar_xy is None:
        # simulation_app.update()를 안 부르면 Isaac Sim 자체 UI 이벤트 루프가 안 돌아서
        # 뷰포트/조작이 먹통이 된다(대기 중에도 화면이 살아있어야 함) - 실측 확인된 버그.
        simulation_app.update()
        rclpy.spin_once(vision_node, timeout_sec=0.0)
        if time.time() - last_log > VISION_POLL_LOG_SEC:
            last_log = time.time()
            print(f"  -> 아직 미검출... ({time.time() - t0:.0f}s 경과)")
    busbar_xy = vision_node.latest_busbar_xy
    diff_mm = (np.array(busbar_xy) - hardcoded_busbar_xy) * 1000.0
    print(f"[vision] busbar 검출 좌표=({busbar_xy[0]:.4f},{busbar_xy[1]:.4f}) "
          f"| 하드코딩 대비 차이=({diff_mm[0]:+.1f},{diff_mm[1]:+.1f})mm")
    pick_z = BUSBAR_PICK_POS[2]
    new_pick = np.array([busbar_xy[0], busbar_xy[1], pick_z])
    BUSBAR_PICK_POS = new_pick
    BUSBAR_APPROACH_POS = new_pick + np.array([0.0, 0.0, 0.145])
    BUSBAR_LIFT_MOVE_POS = new_pick + np.array([0.0, 0.1, 0.145])

    print("[vision] bolt_cam 볼트쌍 검출 대기 중 (perception_node bolt_cam 인스턴스 필요)...")
    t0, last_log = time.time(), 0.0
    while vision_node.latest_bolt_a_xy is None or vision_node.latest_bolt_b_xy is None:
        simulation_app.update()
        rclpy.spin_once(vision_node, timeout_sec=0.0)
        if time.time() - last_log > VISION_POLL_LOG_SEC:
            last_log = time.time()
            print(f"  -> 아직 미검출... ({time.time() - t0:.0f}s 경과)")

    cand_a = np.array(vision_node.latest_bolt_a_xy)
    cand_b = np.array(vision_node.latest_bolt_b_xy)
    d_same = np.linalg.norm(cand_a - hardcoded_bolt1_xy) + np.linalg.norm(cand_b - hardcoded_bolt2_xy)
    d_cross = np.linalg.norm(cand_a - hardcoded_bolt2_xy) + np.linalg.norm(cand_b - hardcoded_bolt1_xy)
    new_bolt1_xy, new_bolt2_xy = (cand_a, cand_b) if d_same <= d_cross else (cand_b, cand_a)
    diff1_mm = (new_bolt1_xy - hardcoded_bolt1_xy) * 1000.0
    diff2_mm = (new_bolt2_xy - hardcoded_bolt2_xy) * 1000.0
    print(f"[vision] BOLT1 검출=({new_bolt1_xy[0]:.4f},{new_bolt1_xy[1]:.4f}) "
          f"차이=({diff1_mm[0]:+.1f},{diff1_mm[1]:+.1f})mm | "
          f"BOLT2 검출=({new_bolt2_xy[0]:.4f},{new_bolt2_xy[1]:.4f}) "
          f"차이=({diff2_mm[0]:+.1f},{diff2_mm[1]:+.1f})mm")
    BOLT1_POS = np.array([new_bolt1_xy[0], new_bolt1_xy[1], BOLT1_POS[2]])
    BOLT2_POS = np.array([new_bolt2_xy[0], new_bolt2_xy[1], BOLT2_POS[2]])
    BOLT1_APPROACH_POS = np.array([BOLT1_POS[0], BOLT1_POS[1], 0.6])
    BOLT1_TOUCH_POS = np.array(
        [BOLT1_POS[0], BOLT1_POS[1], BOLT1_POS[2] + EE_OFFSET[2] + NUT_GRASP_Z_LOCAL])
    BOLT2_APPROACH_POS = np.array([BOLT2_POS[0], BOLT2_POS[1], 0.6])
    BOLT2_TOUCH_POS = np.array(
        [BOLT2_POS[0], BOLT2_POS[1], BOLT2_POS[2] + EE_OFFSET[2] + NUT_GRASP_Z_LOCAL])


# ══════════════════════════════════════════════════════════════════════════
#  [B] 헬퍼 및 Kinematic Pose-Glue / Screwing 계산 함수
# ══════════════════════════════════════════════════════════════════════════
def euler_to_quaternion_wxyz(roll, pitch, yaw):
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * sp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])


def disable_physics_recursively(stage, prim_path):
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        return

    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            col_api = UsdPhysics.CollisionAPI(prim)
            col_api.GetCollisionEnabledAttr().Set(False)
        
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb_api = UsdPhysics.RigidBodyAPI(prim)
            rb_api.GetRigidBodyEnabledAttr().Set(False)


def enable_physics_recursively(stage, prim_path):
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        return

    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            col_api = UsdPhysics.CollisionAPI(prim)
            col_api.GetCollisionEnabledAttr().Set(True)
        
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb_api = UsdPhysics.RigidBodyAPI(prim)
            rb_api.GetRigidBodyEnabledAttr().Set(True)


def yaw_rotated_quat(base_wxyz, delta_deg):
    base_q = Gf.Quatd(float(base_wxyz[0]), Gf.Vec3d(float(base_wxyz[1]), float(base_wxyz[2]), float(base_wxyz[3])))
    base_rot = Gf.Rotation(base_q)
    extra_rot = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), float(delta_deg))
    combined = extra_rot * base_rot
    q = combined.GetQuat()
    return np.array([q.GetReal(), *q.GetImaginary()])


def world_xf(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim 없음: {path}")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)


def find_prim_path(stage, root_path, name):
    root = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def set_all_drives(stage, root_path, stiffness=1.0e8, damping=1.0e4, max_force=1.0e8):
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        for dt in ("angular", "linear"):
            drive = UsdPhysics.DriveAPI.Get(prim, dt)
            if drive:
                drive.GetStiffnessAttr().Set(stiffness)
                drive.GetDampingAttr().Set(damping)
                drive.GetMaxForceAttr().Set(max_force)


def lock_amr_base(stage, amr_root_path):
    amr_prim = stage.GetPrimAtPath(amr_root_path).GetParent()
    if not amr_prim.IsValid():
        amr_prim = stage.GetPrimAtPath(amr_root_path)

    for prim in Usd.PrimRange(amr_prim):
        for dt in ("angular", "linear"):
            drive = UsdPhysics.DriveAPI.Get(prim, dt)
            if drive:
                drive.GetStiffnessAttr().Set(1.0e9)
                drive.GetDampingAttr().Set(1.0e6)
                drive.GetMaxForceAttr().Set(1.0e9)
                if drive.GetTargetVelocityAttr():
                    drive.GetTargetVelocityAttr().Set(0.0)


def initialize_robot(robot, world):
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )


def glue_busbar_to_ee(robot, busbar_xform, rest_pick_pos, blend):
    if busbar_xform is None or rest_pick_pos is None:
        return

    ee_pos, ee_quat = robot.end_effector.get_world_pose()
    grasp_point_pos = np.asarray(ee_pos) - EE_OFFSET
    target_pos = grasp_point_pos - np.array([0.0, 0.0, BUSBAR_GRASP_Z_LOCAL])

    busbar_pos = rest_pick_pos + blend * (target_pos - rest_pick_pos)
    busbar_xform.set_world_pose(position=busbar_pos, orientation=np.asarray(ee_quat))


# ══════════════════════════════════════════════════════════════════════════
#  [C] 메인
# ══════════════════════════════════════════════════════════════════════════
def main():
    usd_file_path = Path(USD_PATH).resolve()
    if not usd_file_path.is_file():
        raise FileNotFoundError(f"[ERROR] USD 파일을 찾을 수 없습니다: {usd_file_path}")

    ctx = omni.usd.get_context()
    ctx.open_stage(str(usd_file_path))
    for _ in range(15):
        simulation_app.update()
        
    stage = ctx.get_stage()
    if not stage:
        raise RuntimeError(f"[ERROR] Stage를 로드하지 못했습니다: {usd_file_path}")

    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT)

    lock_amr_base(stage, NOVA_CARTER_ROOT)

    busbar_xform = SingleXFormPrim(BUSBAR_POLYSHAPE_PATH, name="busbar_poly") if stage.GetPrimAtPath(BUSBAR_POLYSHAPE_PATH).IsValid() else None
    nut1_xform   = SingleXFormPrim(NUT1_POLYSHAPE_PATH, name="nut1_poly") if stage.GetPrimAtPath(NUT1_POLYSHAPE_PATH).IsValid() else None
    nut2_xform   = SingleXFormPrim(NUT2_POLYSHAPE_PATH, name="nut2_poly") if stage.GetPrimAtPath(NUT2_POLYSHAPE_PATH).IsValid() else None

    # 초기 원본 Pose 저장
    init_busbar_pos, init_busbar_quat = busbar_xform.get_world_pose() if busbar_xform else (None, None)
    init_nut1_pos, init_nut1_quat     = nut1_xform.get_world_pose() if nut1_xform else (None, None)
    init_nut2_pos, init_nut2_quat     = nut2_xform.get_world_pose() if nut2_xform else (None, None)

    set_all_drives(stage, M0609_PATH)
    ee_path = find_prim_path(stage, M0609_PATH, EE_LINK_NAME)
    
    gripper = ParallelGripper(
        end_effector_prim_path=ee_path,
        joint_prim_names=GRIPPER_JOINTS,
        joint_opened_positions=GRIPPER_OPEN,
        joint_closed_positions=GRIPPER_CLOSE,
        action_deltas=GRIPPER_DELTA,
    )

    robot = world.scene.add(SingleManipulator(
        prim_path=NOVA_CARTER_ROOT, 
        name="mobile_manipulator",
        end_effector_prim_path=ee_path,
        gripper=gripper,
    ))

    world.reset()
    initialize_robot(robot, world)

    default_joint_positions = np.array([-0.5, -0.4, 1.4, 0.0, 0.6, 0.0, 0.0, 0.0])
    try:
        robot.set_joint_positions(default_joint_positions)
    except Exception:
        pass

    for _ in range(30):
        world.step(render=True)

    quat_busbar = euler_to_quaternion_wxyz(0.0, 3.1415, 1.5708)
    quat_busbar_0deg = euler_to_quaternion_wxyz(0.0, 3.1415, 0.0)

    rot_nut = R.from_euler('xyz', [0, 180, 0], degrees=True)
    q_nut   = rot_nut.as_quat()
    quat_nut = np.array([q_nut[3], q_nut[0], q_nut[1], q_nut[2]])

    # 비전 연동: busbar_cam/bolt_cam 전용 perception_node 인스턴스를 구독해 시작 시
    # 1회 위치를 실측 좌표로 덮어쓴다. 폴백 없음 - 둘 다 검출될 때까지 여기서 대기한다.
    rclpy.init()
    vision_node = VisionBridge()
    resolve_vision_positions(vision_node)

    arm_controller = RMPFlowController(
        name="m0609_hardcoded_controller",
        robot_articulation=robot,
        urdf_path=URDF_PATH,
        robot_description_path=ROBOT_DESC_PATH,
        rmpflow_config_path=RMPFLOW_CFG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
    )
    
    base_link_xf = world_xf(stage, f"{M0609_PATH}/base_link")
    base_pos = base_link_xf.ExtractTranslation()
    base_quat = base_link_xf.ExtractRotationQuat()
    arm_controller._motion_policy.set_robot_base_pose(
        robot_position=np.array([base_pos[0], base_pos[1], base_pos[2]]),
        robot_orientation=np.array([base_quat.GetReal(), *[float(x) for x in base_quat.GetImaginary()]]),
    )

    print("[대기] Isaac Sim UI에서 Play 버튼을 누르면 시퀀스를 시작합니다.")

    step_count = 0
    grasp_timer = 0
    was_playing = False
    
    # 버스바 장착부터 전체 시퀀스 시작 (원래 이 버전은 너트 물리파지 검증을 위해
    # 버스바를 건너뛰고 NUT1_APPROACH부터 시작했으나, 여기서는 전체 파이프라인을
    # 돌리기 위해 복원한다 - 재시작 핸들러는 원래부터 BUSBAR_APPROACH로 리셋했다).
    phase = "BUSBAR_APPROACH"
    current_err = 0.0

    busbar_start_grasp_pos = None
    descend_target_z       = None
    
    screw_sub = "rotate"
    screw_pass_idx = 0
    screw_pass_theta = 0.0
    screw_seat_pos = None
    screw_seat_quat = None
    screw_seat_ee_pos = None
    screw_start_quat = None
    screw_pass_end_pos = None
    screw_release_step = 0
    screw_regrasp_step = 0
    screw_unwind_deg = 0.0

    # ★ 완착 감지용 모니터링 변수 ★
    prev_ee_z = 0.0
    stuck_counter = 0

    while simulation_app.is_running():
        world.step(render=True)
        rclpy.spin_once(vision_node, timeout_sec=0.0)
        playing = world.is_playing()

        # Stop 후 Play 재시작 시
        if playing and not was_playing:
            world.reset()
            initialize_robot(robot, world)
            
            enable_physics_recursively(stage, BUSBAR_ROOT_PATH)
            enable_physics_recursively(stage, NUT1_ROOT_PATH)
            enable_physics_recursively(stage, NUT2_ROOT_PATH)

            if busbar_xform and init_busbar_pos is not None:
                busbar_xform.set_world_pose(position=init_busbar_pos, orientation=init_busbar_quat)
            if nut1_xform and init_nut1_pos is not None:
                nut1_xform.set_world_pose(position=init_nut1_pos, orientation=init_nut1_quat)
            if nut2_xform and init_nut2_pos is not None:
                nut2_xform.set_world_pose(position=init_nut2_pos, orientation=init_nut2_quat)

            try:
                robot.set_joint_positions(default_joint_positions)
            except Exception:
                pass
            step_count = 0
            grasp_timer = 0
            # ★ 재시작 시에도 너트 1번부터 시작 ★
            phase = "BUSBAR_APPROACH"
            current_err = 0.0
            busbar_start_grasp_pos = None
            descend_target_z       = None
            
            screw_sub = "rotate"
            screw_pass_idx = 0
            screw_pass_theta = 0.0
            stuck_counter = 0
            print(f"\n[Play] 시퀀스 재시작 (모든 객체 포즈 및 오차 조건 초기화 완료)")

        if playing and phase != "DONE":

            # ════════════════════════════════════════════════════════════════
            # [1] 버스바 픽앤플레이스 + 장착 시퀀스
            # ════════════════════════════════════════════════════════════════
            if phase == "BUSBAR_APPROACH":
                actions = arm_controller.forward(target_end_effector_position=BUSBAR_APPROACH_POS, target_end_effector_orientation=quat_busbar)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(BUSBAR_APPROACH_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 1. 버스바 상공 접근 완료! -> 2. 파지점 하강 시작")
                    phase = "BUSBAR_DESCEND"
                    step_count = 0
            
            elif phase == "BUSBAR_DESCEND":
                actions = arm_controller.forward(target_end_effector_position=BUSBAR_PICK_POS, target_end_effector_orientation=quat_busbar)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(BUSBAR_PICK_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 2. 버스바 파지점 하강 완료! -> 그리퍼 닫기(Kinematic 파지)")
                    phase = "BUSBAR_GRASP"
                    grasp_timer = 0
            
            elif phase == "BUSBAR_GRASP":
                if grasp_timer == 0:
                    disable_physics_recursively(stage, BUSBAR_ROOT_PATH)
                    if busbar_xform is not None:
                        real_pos, _ = busbar_xform.get_world_pose()
                        busbar_start_grasp_pos = np.array(real_pos)
                    else:
                        busbar_start_grasp_pos = BUSBAR_PICK_POS
            
                actions = arm_controller.forward(target_end_effector_position=BUSBAR_PICK_POS, target_end_effector_orientation=quat_busbar)
                robot.apply_action(actions)
                
                grasp_timer += 1
                ramp_frac = min(grasp_timer / GRIP_CLOSE_RAMP_STEPS, 1.0)
                grip_target = ramp_frac * GRIPPER_CLOSE
                robot.gripper.apply_action(ArticulationAction(joint_positions=grip_target))
            
                glue_busbar_to_ee(robot, busbar_xform, busbar_start_grasp_pos, blend=ramp_frac)
            
                if grasp_timer >= 50:
                    print(f"[OK] 그리퍼 닫기 완료! -> 3. 버스바 상승 및 이동 시작")
                    phase = "BUSBAR_LIFT"
                    step_count = 0
            
            elif phase == "BUSBAR_LIFT":
                actions = arm_controller.forward(target_end_effector_position=BUSBAR_LIFT_MOVE_POS, target_end_effector_orientation=quat_busbar)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))
                
                glue_busbar_to_ee(robot, busbar_xform, busbar_start_grasp_pos, blend=1.0)
            
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(BUSBAR_LIFT_MOVE_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 3. 버스바 상승 이동 완료! -> [INSERT] 1. 체결위치 상공 접근 (그리퍼 0도 회전)")
                    phase = "MOVE_TO_BOLT_APPROACH"
                    step_count = 0
            
            elif phase == "MOVE_TO_BOLT_APPROACH":
                actions = arm_controller.forward(target_end_effector_position=TARGET_DESTINATION_POS, target_end_effector_orientation=quat_busbar_0deg)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))
            
                glue_busbar_to_ee(robot, busbar_xform, busbar_start_grasp_pos, blend=1.0)
            
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(TARGET_DESTINATION_POS))
                if current_err < INSERT_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 1. 체결위치 상공 접근 완료! -> 2. 점진적 하강(-{INSERT_SPEED}m/step) 시작")
                    phase = "BUSBAR_DESCEND_TO_BOLT"
                    step_count = 0
                    descend_target_z = TARGET_DESTINATION_POS[2]
            
            elif phase == "BUSBAR_DESCEND_TO_BOLT":
                descend_target_z = max(descend_target_z - INSERT_SPEED, TARGET_INSERT_POS[2])
                step_target_pos = np.array([TARGET_INSERT_POS[0], TARGET_INSERT_POS[1], descend_target_z])
            
                actions = arm_controller.forward(target_end_effector_position=step_target_pos, target_end_effector_orientation=quat_busbar_0deg)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))
            
                glue_busbar_to_ee(robot, busbar_xform, busbar_start_grasp_pos, blend=1.0)
            
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                dist_err = math.dist(cur_pos, tuple(TARGET_INSERT_POS))
            
                if cur_pos[2] <= BUSBAR_RELEASE_Z or dist_err < INSERT_TOLERANCE_STRICT:
                    if busbar_xform is not None:
                        busbar_xform.set_world_pose(position=target_mid_pos, orientation=BUSBAR_REST_ORIENTATION)
                    print(f"[OK] 2. 버스바 체결 완료 (EE Z: {cur_pos[2]:.4f}m)! -> 그리퍼 열기 및 상공 이탈")
                    phase = "BUSBAR_RELEASE_AND_RETRACT"
                    step_count = 0
            
            elif phase == "BUSBAR_RELEASE_AND_RETRACT":
                actions = arm_controller.forward(target_end_effector_position=TARGET_DESTINATION_POS, target_end_effector_orientation=quat_busbar_0deg)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
            
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(TARGET_DESTINATION_POS))
                if current_err < INSERT_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 3. 버스바 상공 이탈 완료! -> 너트 1번 체결 공정 진입\n")
                    phase = "NUT1_APPROACH"
                    step_count = 0

            # ════════════════════════════════════════════════════════════════
            # [2] 너트 1번 물리 파지 및 상승 (nut1 -> bolt1)
            # ════════════════════════════════════════════════════════════════
            if phase == "NUT1_APPROACH":
                actions = arm_controller.forward(target_end_effector_position=NUT1_APPROACH_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(NUT1_APPROACH_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 너트 1번 상공 도착! -> 하강 시작")
                    phase = "NUT1_DESCEND"
                    step_count = 0

            elif phase == "NUT1_DESCEND":
                actions = arm_controller.forward(target_end_effector_position=NUT1_PICK_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(NUT1_PICK_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 너트 1번 하강 완료! -> 물리 파지(Gripper Close) 시작")
                    phase = "NUT1_GRASP"
                    grasp_timer = 0

            elif phase == "NUT1_GRASP":
                actions = arm_controller.forward(target_end_effector_position=NUT1_PICK_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                
                grasp_timer += 1
                ramp_frac = min(grasp_timer / GRIP_CLOSE_RAMP_STEPS, 1.0)
                grip_target = ramp_frac * GRIPPER_CLOSE_NUT
                robot.gripper.apply_action(ArticulationAction(joint_positions=grip_target))

                if grasp_timer >= 50:
                    print(f"[OK] 너트 1번 물리 파지 완료! -> 상공({NUT_APPROACH_Z}m)으로 상승")
                    phase = "NUT1_LIFT"
                    step_count = 0

            elif phase == "NUT1_LIFT":
                actions = arm_controller.forward(target_end_effector_position=NUT1_APPROACH_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(NUT1_APPROACH_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 너트 1번 상승 완료! -> 볼트 1번 상공({BOLT1_APPROACH_POS})으로 이동 시작")
                    phase = "MOVE_TO_BOLT1"
                    step_count = 0

            # ════════════════════════════════════════════════════════════════
            # [3] 볼트 1번 상공 이동 후 착좌 하강
            # ════════════════════════════════════════════════════════════════
            elif phase == "MOVE_TO_BOLT1":
                actions = arm_controller.forward(target_end_effector_position=BOLT1_APPROACH_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(BOLT1_APPROACH_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 볼트 1번 상공 도착! -> 착좌 하강 시작")
                    phase = "NUT1_DESCEND_TO_BOLT1"
                    step_count = 0
                    descend_target_z = BOLT1_APPROACH_POS[2]

            elif phase == "NUT1_DESCEND_TO_BOLT1":
                descend_target_z = max(descend_target_z - INSERT_SPEED, BOLT1_TOUCH_POS[2])
                step_target_pos = np.array([BOLT1_TOUCH_POS[0], BOLT1_TOUCH_POS[1], descend_target_z])

                actions = arm_controller.forward(target_end_effector_position=step_target_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                
                if abs(cur_pos[2] - BOLT1_TOUCH_POS[2]) < PICK_TOLERANCE_LOOSE_VAL or descend_target_z <= BOLT1_TOUCH_POS[2]:
                    ee_now_pos, ee_now_quat = robot.end_effector.get_world_pose()
                    screw_start_quat = np.asarray(ee_now_quat).copy()
                    screw_seat_ee_pos = np.asarray(ee_now_pos).copy()
                    
                    if nut1_xform is not None:
                        real_nut_pos, _ = nut1_xform.get_world_pose()
                        screw_seat_pos = np.array([BOLT1_POS[0], BOLT1_POS[1], real_nut_pos[2]])
                    else:
                        screw_seat_pos = np.array([BOLT1_POS[0], BOLT1_POS[1], BOLT1_POS[2]])
                        
                    screw_seat_quat = quat_nut.copy()

                    screw_sub = "rotate"
                    screw_pass_idx = 0
                    screw_pass_theta = 0.0
                    stuck_counter = 0
                    prev_ee_z = ee_now_pos[2]

                    print(f"[OK] 볼트 1번 착좌 완료 (너트 Z={screw_seat_pos[2]:.4f}m)! -> Screwing 시작")
                    phase = "NUT1_SCREW"
                    step_count = 0

            # ════════════════════════════════════════════════════════════════
            # [4] Screwing (너트 1번 -> 볼트 1번 체결)
            # ════════════════════════════════════════════════════════════════
            elif phase == "NUT1_SCREW":
                if screw_sub == "rotate":
                    screw_pass_theta = min(screw_pass_theta + SCREW_OMEGA_DEG_S * PHYSICS_DT, SCREW_TURNS_DEG)
                    pass_done = (screw_pass_theta >= SCREW_TURNS_DEG)

                    total_deg = screw_pass_idx * SCREW_TURNS_DEG + screw_pass_theta
                    depth_m = min((total_deg / 360.0) * NUT_PITCH_M, ENGAGE_LEN)

                    target_pos = screw_seat_ee_pos.copy()
                    target_pos[2] = screw_seat_ee_pos[2] - depth_m
                    target_quat = yaw_rotated_quat(screw_start_quat, screw_pass_theta)

                    actions = arm_controller.forward(target_end_effector_position=target_pos, target_end_effector_orientation=target_quat)
                    robot.apply_action(actions)
                    robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                    # ★ 실시간 토크 감지 및 Z축 정지(Stuck) 모니터링 ★
                    cur_ee_pos, _ = robot.end_effector.get_world_pose()
                    z_movement = abs(prev_ee_z - cur_ee_pos[2])
                    prev_ee_z = cur_ee_pos[2]

                    joint_efforts = robot.get_measured_joint_efforts()
                    curr_torque = abs(joint_efforts[-1]) if joint_efforts is not None and len(joint_efforts) > 0 else 0.0

                    # ★ TCP Z 높이가 0.378m 이하일 때만 힘/토크 감지 적용 ★
                    if cur_ee_pos[2] <= TCP_FORCE_CHECK_Z:
                        if z_movement < STUCK_Z_DELTA_THRESH and depth_m > 0.003:
                            stuck_counter += 1
                        else:
                            stuck_counter = max(0, stuck_counter - 1)

                        is_seated_by_torque = (curr_torque > TORQUE_THRESHOLD) or (stuck_counter >= STUCK_STEP_LIMIT)
                    else:
                        is_seated_by_torque = False

                    if step_count % 20 == 0:
                        print(f"  [NUT1 SCREW] Pass {screw_pass_idx+1}/{1+REGRASP_CYCLES} | Theta: {screw_pass_theta:.1f}° | 깊이: {depth_m*1000:.2f}mm / 목표 {ENGAGE_LEN*1000:.1f}mm | TCP Z: {cur_ee_pos[2]:.4f}m | 토크: {curr_torque:.1f}Nm")

                    if pass_done or is_seated_by_torque:
                        if is_seated_by_torque:
                            print(f"  [체결 감지] 너트 1번 완착(Seating) 감지! (TCP Z: {cur_ee_pos[2]:.4f}m <= 0.378m, 토크: {curr_torque:.1f}Nm) -> Screwing 조기 종료 및 그리퍼 해제")

                        if depth_m >= ENGAGE_LEN or screw_pass_idx >= REGRASP_CYCLES or is_seated_by_torque:
                            robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
                            print(f"\n[OK] 너트 1번 체결 완료! -> 꼬인 방향 유지한 채 수직 상승 시작")
                            phase = "NUT1_RETRACT_LIFT"
                            step_count = 0
                        else:
                            screw_pass_end_pos = target_pos.copy()
                            screw_sub = "release"
                            screw_release_step = 0

                elif screw_sub == "release":
                    screw_release_step += 1
                    rf = min(screw_release_step / GRIP_CLOSE_RAMP_STEPS, 1.0)
                    release_target = (1.0 - rf) * GRIPPER_CLOSE_NUT[0]
                    robot.gripper.apply_action(ArticulationAction(joint_positions=np.array([release_target, release_target])))

                    hold_quat = yaw_rotated_quat(screw_start_quat, screw_pass_theta)
                    actions = arm_controller.forward(target_end_effector_position=screw_pass_end_pos, target_end_effector_orientation=hold_quat)
                    robot.apply_action(actions)

                    if rf >= 1.0:
                        screw_sub = "lift_up"
                        screw_release_step = 0

                elif screw_sub == "lift_up":
                    lift_target_pos = screw_pass_end_pos + np.array([0.0, 0.0, REGRASP_LIFT_HEIGHT])
                    hold_quat = yaw_rotated_quat(screw_start_quat, screw_pass_theta)
                    actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=hold_quat)
                    robot.apply_action(actions)
                    robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                    cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                    if math.dist(cur_pos, tuple(lift_target_pos)) < PICK_TOLERANCE_STRICT or screw_release_step > 40:
                        screw_sub = "unwind"
                        screw_unwind_deg = screw_pass_theta
                    screw_release_step += 1

                elif screw_sub == "unwind":
                    screw_unwind_deg = max(screw_unwind_deg - SCREW_OMEGA_DEG_S * PHYSICS_DT, 0.0)
                    unwind_done = (screw_unwind_deg <= 0.0)

                    lift_target_pos = screw_pass_end_pos + np.array([0.0, 0.0, REGRASP_LIFT_HEIGHT])
                    target_quat = yaw_rotated_quat(screw_start_quat, screw_unwind_deg)
                    actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=target_quat)
                    robot.apply_action(actions)
                    robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                    if unwind_done:
                        screw_sub = "descend_down"
                        screw_release_step = 0

                elif screw_sub == "descend_down":
                    # ★ Regrasp 하강 시 Z축 높이를 미세 상승 보정 (+REGRASP_Z_OFFSET) ★
                    regrasp_descend_target = screw_pass_end_pos + np.array([0.0, 0.0, REGRASP_Z_OFFSET])
                    actions = arm_controller.forward(target_end_effector_position=regrasp_descend_target, target_end_effector_orientation=screw_start_quat)
                    robot.apply_action(actions)
                    robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                    cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                    if math.dist(cur_pos, tuple(regrasp_descend_target)) < PICK_TOLERANCE_STRICT or screw_release_step > 40:
                        screw_sub = "regrasp"
                        screw_regrasp_step = 0
                    screw_release_step += 1

                elif screw_sub == "regrasp":
                    screw_regrasp_step += 1
                    rf = min(screw_regrasp_step / GRIP_CLOSE_RAMP_STEPS, 1.0)
                    grip_target = rf * GRIPPER_CLOSE_NUT[0]
                    robot.gripper.apply_action(ArticulationAction(joint_positions=np.array([grip_target, grip_target])))

                    # ★ Regrasp 재파지 시에도 Z축 높이를 미세 상승 보정 (+REGRASP_Z_OFFSET) ★
                    regrasp_target_pos = screw_pass_end_pos + np.array([0.0, 0.0, REGRASP_Z_OFFSET])
                    actions = arm_controller.forward(target_end_effector_position=regrasp_target_pos, target_end_effector_orientation=screw_start_quat)
                    robot.apply_action(actions)

                    if rf >= 1.0:
                        screw_pass_idx += 1
                        screw_pass_theta = 0.0
                        stuck_counter = 0
                        screw_sub = "rotate"

            # ════════════════════════════════════════════════════════════════
            # [5] 너트 1번 체결 후: 수직 상승 -> 상공에서 되감기 -> 정렬
            # ════════════════════════════════════════════════════════════════
            elif phase == "NUT1_RETRACT_LIFT":
                ee_now_pos, _ = robot.end_effector.get_world_pose()
                lift_target_pos = np.array([1.0033894, 0.1387623, 0.8001476])

                last_screw_quat = yaw_rotated_quat(screw_start_quat, screw_pass_theta)
                actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=last_screw_quat)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(lift_target_pos))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 수직 이탈 완료! -> 상공 안전 지대에서 6번 조인트 0도로 되감기(Unwind) 시작")
                    phase = "NUT1_RETRACT_UNWIND"
                    screw_unwind_deg = screw_pass_theta
                    step_count = 0

            elif phase == "NUT1_RETRACT_UNWIND":
                screw_unwind_deg = max(screw_unwind_deg - SCREW_OMEGA_DEG_S * PHYSICS_DT, 0.0)
                ee_now_pos, _ = robot.end_effector.get_world_pose()
                lift_target_pos = np.array([ee_now_pos[0], ee_now_pos[1], NUT_APPROACH_Z])

                target_quat = yaw_rotated_quat(screw_start_quat, screw_unwind_deg)
                actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=target_quat)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                if screw_unwind_deg <= 0.0:
                    print(f"[OK] 6번 조인트 0도 원점 되감기 완료! -> 기본 방향(quat_nut) 정렬")
                    phase = "NUT1_RETRACT_ROTATE"
                    step_count = 0

            elif phase == "NUT1_RETRACT_ROTATE":
                ee_now_pos, _ = robot.end_effector.get_world_pose()
                lift_target_pos = np.array([ee_now_pos[0], ee_now_pos[1], NUT_APPROACH_Z])

                # [수정 완료] 인자 명칭 오류 수정 (target_end_orientation -> target_end_effector_orientation)
                # quat_nut (너트 1번 접근 때와 동일한 오리엔테이션)으로 완전 정렬
                actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(lift_target_pos))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 너트 1번 완전히 이탈 성공! -> [너트 2번 체결 공정 시작]\n")
                    phase = "NUT2_APPROACH"
                    step_count = 0

            # ════════════════════════════════════════════════════════════════
            # [6] 너트 2번 물리 파지 및 상승 (nut2 -> bolt2)
            # ════════════════════════════════════════════════════════════════
            elif phase == "NUT2_APPROACH":
                actions = arm_controller.forward(target_end_effector_position=NUT2_APPROACH_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(NUT2_APPROACH_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 너트 2번 상공 도착! -> 하강 시작")
                    phase = "NUT2_DESCEND"
                    step_count = 0

            elif phase == "NUT2_DESCEND":
                actions = arm_controller.forward(target_end_effector_position=NUT2_PICK_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(NUT2_PICK_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 너트 2번 하강 완료! -> 물리 파지(Gripper Close) 시작")
                    phase = "NUT2_GRASP"
                    grasp_timer = 0

            elif phase == "NUT2_GRASP":
                actions = arm_controller.forward(target_end_effector_position=NUT2_PICK_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                
                grasp_timer += 1
                ramp_frac = min(grasp_timer / GRIP_CLOSE_RAMP_STEPS, 1.0)
                grip_target = ramp_frac * GRIPPER_CLOSE_NUT
                robot.gripper.apply_action(ArticulationAction(joint_positions=grip_target))

                if grasp_timer >= 50:
                    print(f"[OK] 너트 2번 물리 파지 완료! -> 상공({NUT_APPROACH_Z}m)으로 상승")
                    phase = "NUT2_LIFT"
                    step_count = 0

            elif phase == "NUT2_LIFT":
                actions = arm_controller.forward(target_end_effector_position=NUT2_APPROACH_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(NUT2_APPROACH_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 너트 2번 상승 완료! -> 볼트 2번 상공({BOLT2_APPROACH_POS})으로 이동 시작")
                    phase = "MOVE_TO_BOLT2"
                    step_count = 0

            # ════════════════════════════════════════════════════════════════
            # [7] 볼트 2번 상공 이동 후 착좌 하강
            # ════════════════════════════════════════════════════════════════
            elif phase == "MOVE_TO_BOLT2":
                actions = arm_controller.forward(target_end_effector_position=BOLT2_APPROACH_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(BOLT2_APPROACH_POS))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 볼트 2번 상공 도착! -> 착좌 하강 시작")
                    phase = "NUT2_DESCEND_TO_BOLT2"
                    step_count = 0
                    descend_target_z = BOLT2_APPROACH_POS[2]

            elif phase == "NUT2_DESCEND_TO_BOLT2":
                descend_target_z = max(descend_target_z - INSERT_SPEED, BOLT2_TOUCH_POS[2])
                step_target_pos = np.array([BOLT2_TOUCH_POS[0], BOLT2_TOUCH_POS[1], descend_target_z])

                actions = arm_controller.forward(target_end_effector_position=step_target_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                
                if abs(cur_pos[2] - BOLT2_TOUCH_POS[2]) < PICK_TOLERANCE_LOOSE_VAL or descend_target_z <= BOLT2_TOUCH_POS[2]:
                    ee_now_pos, ee_now_quat = robot.end_effector.get_world_pose()
                    screw_start_quat = np.asarray(ee_now_quat).copy()
                    screw_seat_ee_pos = np.asarray(ee_now_pos).copy()
                    
                    if nut2_xform is not None:
                        real_nut_pos, _ = nut2_xform.get_world_pose()
                        screw_seat_pos = np.array([BOLT2_POS[0], BOLT2_POS[1], real_nut_pos[2]])
                    else:
                        screw_seat_pos = np.array([BOLT2_POS[0], BOLT2_POS[1], BOLT2_POS[2]])
                        
                    screw_seat_quat = quat_nut.copy()

                    screw_sub = "rotate"
                    screw_pass_idx = 0
                    screw_pass_theta = 0.0
                    stuck_counter = 0
                    prev_ee_z = ee_now_pos[2]

                    print(f"[OK] 볼트 2번 착좌 완료 (너트 Z={screw_seat_pos[2]:.4f}m)! -> Screwing 시작")
                    phase = "NUT2_SCREW"
                    step_count = 0

            # ════════════════════════════════════════════════════════════════
            # [8] Screwing (너트 2번 -> 볼트 2번 체결)
            # ════════════════════════════════════════════════════════════════
            elif phase == "NUT2_SCREW":
                if screw_sub == "rotate":
                    screw_pass_theta = min(screw_pass_theta + SCREW_OMEGA_DEG_S * PHYSICS_DT, SCREW_TURNS_DEG)
                    pass_done = (screw_pass_theta >= SCREW_TURNS_DEG)

                    total_deg = screw_pass_idx * SCREW_TURNS_DEG + screw_pass_theta
                    depth_m = min((total_deg / 360.0) * NUT_PITCH_M, ENGAGE_LEN)

                    target_pos = screw_seat_ee_pos.copy()
                    target_pos[2] = screw_seat_ee_pos[2] - depth_m
                    target_quat = yaw_rotated_quat(screw_start_quat, screw_pass_theta)

                    actions = arm_controller.forward(target_end_effector_position=target_pos, target_end_effector_orientation=target_quat)
                    robot.apply_action(actions)
                    robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                    # ★ 실시간 토크 감지 및 Z축 정지(Stuck) 모니터링 ★
                    cur_ee_pos, _ = robot.end_effector.get_world_pose()
                    z_movement = abs(prev_ee_z - cur_ee_pos[2])
                    prev_ee_z = cur_ee_pos[2]

                    joint_efforts = robot.get_measured_joint_efforts()
                    curr_torque = abs(joint_efforts[-1]) if joint_efforts is not None and len(joint_efforts) > 0 else 0.0

                    # ★ TCP Z 높이가 0.378m 이하일 때만 힘/토크 감지 적용 ★
                    if cur_ee_pos[2] <= TCP_FORCE_CHECK_Z:
                        if z_movement < STUCK_Z_DELTA_THRESH and depth_m > 0.003:
                            stuck_counter += 1
                        else:
                            stuck_counter = max(0, stuck_counter - 1)

                        is_seated_by_torque = (curr_torque > TORQUE_THRESHOLD) or (stuck_counter >= STUCK_STEP_LIMIT)
                    else:
                        is_seated_by_torque = False

                    if step_count % 20 == 0:
                        print(f"  [NUT2 SCREW] Pass {screw_pass_idx+1}/{1+REGRASP_CYCLES} | Theta: {screw_pass_theta:.1f}° | 깊이: {depth_m*1000:.2f}mm / 목표 {ENGAGE_LEN*1000:.1f}mm | TCP Z: {cur_ee_pos[2]:.4f}m | 토크: {curr_torque:.1f}Nm")

                    if pass_done or is_seated_by_torque:
                        if is_seated_by_torque:
                            print(f"  [체결 감지] 너트 2번 완착(Seating) 감지! (TCP Z: {cur_ee_pos[2]:.4f}m <= 0.378m, 토크: {curr_torque:.1f}Nm) -> Screwing 조기 종료 및 그리퍼 해제")

                        if depth_m >= ENGAGE_LEN or screw_pass_idx >= REGRASP_CYCLES or is_seated_by_torque:
                            robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
                            print(f"\n[OK] 너트 2번 볼트 2번 체결 완료! -> 꼬인 방향 유지한 채 수직 상승 시작")
                            phase = "NUT2_RETRACT_LIFT"
                            step_count = 0
                        else:
                            screw_pass_end_pos = target_pos.copy()
                            screw_sub = "release"
                            screw_release_step = 0

                elif screw_sub == "release":
                    screw_release_step += 1
                    rf = min(screw_release_step / GRIP_CLOSE_RAMP_STEPS, 1.0)
                    release_target = (1.0 - rf) * GRIPPER_CLOSE_NUT[0]
                    robot.gripper.apply_action(ArticulationAction(joint_positions=np.array([release_target, release_target])))

                    hold_quat = yaw_rotated_quat(screw_start_quat, screw_pass_theta)
                    actions = arm_controller.forward(target_end_effector_position=screw_pass_end_pos, target_end_effector_orientation=hold_quat)
                    robot.apply_action(actions)

                    if rf >= 1.0:
                        screw_sub = "lift_up"
                        screw_release_step = 0

                elif screw_sub == "lift_up":
                    lift_target_pos = screw_pass_end_pos + np.array([0.0, 0.0, REGRASP_LIFT_HEIGHT])
                    hold_quat = yaw_rotated_quat(screw_start_quat, screw_pass_theta)
                    actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=hold_quat)
                    robot.apply_action(actions)
                    robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                    cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                    if math.dist(cur_pos, tuple(lift_target_pos)) < PICK_TOLERANCE_STRICT or screw_release_step > 40:
                        screw_sub = "unwind"
                        screw_unwind_deg = screw_pass_theta
                    screw_release_step += 1

                elif screw_sub == "unwind":
                    screw_unwind_deg = max(screw_unwind_deg - SCREW_OMEGA_DEG_S * PHYSICS_DT, 0.0)
                    unwind_done = (screw_unwind_deg <= 0.0)

                    lift_target_pos = screw_pass_end_pos + np.array([0.0, 0.0, REGRASP_LIFT_HEIGHT])
                    target_quat = yaw_rotated_quat(screw_start_quat, screw_unwind_deg)
                    actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=target_quat)
                    robot.apply_action(actions)
                    robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                    if unwind_done:
                        screw_sub = "descend_down"
                        screw_release_step = 0

                elif screw_sub == "descend_down":
                    # ★ Regrasp 하강 시 Z축 높이를 미세 상승 보정 (+REGRASP_Z_OFFSET) ★
                    regrasp_descend_target = screw_pass_end_pos + np.array([0.0, 0.0, REGRASP_Z_OFFSET])
                    actions = arm_controller.forward(target_end_effector_position=regrasp_descend_target, target_end_effector_orientation=screw_start_quat)
                    robot.apply_action(actions)
                    robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                    cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                    if math.dist(cur_pos, tuple(regrasp_descend_target)) < PICK_TOLERANCE_STRICT or screw_release_step > 40:
                        screw_sub = "regrasp"
                        screw_regrasp_step = 0
                    screw_release_step += 1

                elif screw_sub == "regrasp":
                    screw_regrasp_step += 1
                    rf = min(screw_regrasp_step / GRIP_CLOSE_RAMP_STEPS, 1.0)
                    grip_target = rf * GRIPPER_CLOSE_NUT[0]
                    robot.gripper.apply_action(ArticulationAction(joint_positions=np.array([grip_target, grip_target])))

                    # ★ Regrasp 재파지 시에도 Z축 높이를 미세 상승 보정 (+REGRASP_Z_OFFSET) ★
                    regrasp_target_pos = screw_pass_end_pos + np.array([0.0, 0.0, REGRASP_Z_OFFSET])
                    actions = arm_controller.forward(target_end_effector_position=regrasp_target_pos, target_end_effector_orientation=screw_start_quat)
                    robot.apply_action(actions)

                    if rf >= 1.0:
                        screw_pass_idx += 1
                        screw_pass_theta = 0.0
                        stuck_counter = 0
                        screw_sub = "rotate"

            # ════════════════════════════════════════════════════════════════
            # [9] 너트 2번 체결 후: 수직 상승 -> 상공에서 되감기 -> 정렬
            # ════════════════════════════════════════════════════════════════
            elif phase == "NUT2_RETRACT_LIFT":
                ee_now_pos, _ = robot.end_effector.get_world_pose()
                lift_target_pos = np.array([ee_now_pos[0], ee_now_pos[1], NUT_APPROACH_Z])

                last_screw_quat = yaw_rotated_quat(screw_start_quat, screw_pass_theta)
                actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=last_screw_quat)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(lift_target_pos))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 수직 이탈 완료! -> 상공 안전 지대에서 6번 조인트 0도로 되감기(Unwind) 시작")
                    phase = "NUT2_RETRACT_UNWIND"
                    screw_unwind_deg = screw_pass_theta
                    step_count = 0

            elif phase == "NUT2_RETRACT_UNWIND":
                screw_unwind_deg = max(screw_unwind_deg - SCREW_OMEGA_DEG_S * PHYSICS_DT, 0.0)
                ee_now_pos, _ = robot.end_effector.get_world_pose()
                lift_target_pos = np.array([ee_now_pos[0], ee_now_pos[1], NUT_APPROACH_Z])

                target_quat = yaw_rotated_quat(screw_start_quat, screw_unwind_deg)
                actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=target_quat)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                if screw_unwind_deg <= 0.0:
                    print(f"[OK] 6번 조인트 0도 원점 되감기 완료! -> 기본 방향(quat_nut) 정렬")
                    phase = "NUT2_RETRACT_ROTATE"
                    step_count = 0

            elif phase == "NUT2_RETRACT_ROTATE":
                ee_now_pos, _ = robot.end_effector.get_world_pose()
                lift_target_pos = np.array([ee_now_pos[0], ee_now_pos[1], NUT_APPROACH_Z])

                actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(lift_target_pos))
                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"\n[전체 시퀀스 최종 성공] 너트 1번 체결 + 너트 2번 체결 및 로봇 후퇴 완료!")
                    phase = "DONE"

            # 실시간 로그 출력
            if step_count % 30 == 0 and not phase.endswith("GRASP"):
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                print(f"  [{phase}] Step {step_count:03d} | EE: {np.round(cur_pos, 4)} | Err: {current_err*1000:6.2f} mm")

            step_count += 1

        was_playing = playing

    if 'world' in locals() and world is not None:
        world.clear_instance()
    omni.usd.get_context().close_stage()
    gc.collect()

    vision_node.destroy_node()
    rclpy.shutdown()

    simulation_app.close()


if __name__ == "__main__":
    main()
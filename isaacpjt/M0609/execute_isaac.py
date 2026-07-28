#!/usr/bin/env python3
"""
execute_isaac.py - Busbar Automation Sequence for Isaac Sim
BehaviorNode (FSM) 및 ArmNode, Vision Correction Node 통신 연동 버전

[동작 방식]
0. INIT_POSE            : 관절값 [0, 0, 90, 0, 90, 90] 도 단위로 초기 위치 정렬
1. SCAN_BATTERY         : 볼트캠 이동/reset 후 최신 wrist scan 자세로 이동
2. SCAN_BUSBAR          : 고정캠 이동/reset 및 fixed pose latch 후 wrist scan 자세로 이동
3. PICK_BUSBAR          : Z=0.6m 상공 접근 -> Z=0.455m 파지 위치 하강 -> 물리 파지 및 상승
4. MOVE_BATTERY_CENTER  : ArmNode에서 보낸 배터리 중점 좌표 상공(Z=0.7m)으로 이동
5. FINE_ALIGNMENT       : 비전 노드의 START_ERRORFIX_CORRECTION 트리거 발송 후, 1픽셀 오차 보정 피드백에 맞춰 미세 정렬
6. ASSEMBLE_BUSBAR      : 정렬된 XY 상태를 유지하며 수직 하강 안착, 버스바 고정 해제, 그리퍼 개방 및 상공 이탈
7. SCAN_NUT1 / SCAN_NUT2   : 너트 스캔 위치(버스바 체결 위치 기준 상대 이동)로 이동
8. PICK_NUT1 / PICK_NUT2   : 초기 위치(HOME_EE_POS) 기준 고정 상대좌표로 접근(너트는 Nova
   Carter에 고정되어 비전 없이도 위치가 일정함), 물리 파지(Gripper Close) 및 상승
9. ASSEMBLE_NUT1 / ASSEMBLE_NUT2 : 볼트 1/2번의 실측 월드 좌표(BOLT1/2_WORLD_POS, test_isaac
   씬 고정 배치 기준 하드코딩)로 이동, 착좌 및 Screwing 체결(Regrasp 포함)
10. RETURN_HOME_JOINTS  : 너트 2번 체결 완료 후 초기 관절 각도 [0, 0, 90, 0, 90, 90]로 완전 복귀

진단 모드:
  --preflight-only      : 로봇을 움직이지 않고 USD/카메라/ROS/wheel 계약만 검사
  --smoke-test          : wheel velocity만으로 짧은 전진·후진·회전·도착 시험
"""

import os
import sys
import json
import math
import atexit
import tempfile
import numpy as np
from pathlib import Path

# 1. Isaac SimulationApp 초기화
from isaacsim import SimulationApp

_PREFLIGHT_ONLY = "--preflight-only" in sys.argv[1:]
_SMOKE_TEST = "--smoke-test" in sys.argv[1:]
if _PREFLIGHT_ONLY and _SMOKE_TEST:
    raise ValueError("--preflight-only와 --smoke-test는 동시에 사용할 수 없습니다")
_HEADLESS = (
    os.environ.get("AMR_HEADLESS") == "1"
    or "--headless" in sys.argv[1:]
    or _PREFLIGHT_ONLY
    or _SMOKE_TEST
)
simulation_app = SimulationApp({"headless": _HEADLESS})

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

sys.stdout.reconfigure(line_buffering=True)

# 2. USD 및 Isaac Core Imports
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.core.utils.types import ArticulationAction

# 3. ROS 2 Imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose2D
from std_msgs.msg import String, Float32, Empty

# RMPFlow Controller 경로 설정
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
sys.path.insert(0, str(_THIS_DIR / "rmpflow"))
from m0609_rmpflow_controller import RMPFlowController
from fine_alignment_policy import (
    BoundedPlanarCommandLead,
    FineAlignmentStepGate,
    planar_command_response,
    planar_yaw_delta,
)
from nut_motion_policy import (
    consecutive_pose_settle,
    exact_collision_filter_targets,
    lead_xy_target,
    lead_z_target,
    nut_ee_coupling_error,
    nut_from_ee_offset,
    rate_limited_z_target,
)
from wrist_scan_policy import busbar_wrist_scan_target

_DRIVE_DIR = _REPO_ROOT / "src/amr_node/scrips_drive"
sys.path.insert(0, str(_DRIVE_DIR))
from drive_controller import (
    ControllerConfig,
    Goal2D,
    GoToPoseController,
    PoseState,
    normalize_angle,
)

# ══════════════════════════════════════════════════════════════════════════
#  [A] 설정 및 파라미터
# ══════════════════════════════════════════════════════════════════════════
USD_PATH = str(
    Path(
        os.environ.get(
            "AMR_USD_PATH",
            _REPO_ROOT / "src/Collected_Busbar_fin/Busbar.usd",
        )
    ).expanduser().resolve()
)

# 고정카메라 영상은 실제 Camera prim에서 렌더링하고, TF만 그 아래 ROS optical
# frame에서 발행한다. fin USD의 /World/Graph/camera_graph에 추가된 고정카메라
# RenderProduct는 optical Xform을 Camera로 사용해 잘못된 fallback 영상을 만든다.
# 반면 기존 독립 graph의 RenderProduct는 올바른 Camera prim을 사용하므로, 그
# graph를 전용 namespace/frame으로 재배선하고 잘못된 분기는 실행 전에 끈다.
FIXED_CAMERA_BRIDGES = {
    "/Graph/ROS_Camera_busbar": {
        "camera": "/World/Camera_busbar",
        "namespace": "busbar_cam",
        "optical_frame": "busbar_cam_optical_frame",
    },
    "/Graph/ROS_Camera_bolt": {
        "camera": "/World/Camera_bolt",
        "namespace": "bolt_cam",
        "optical_frame": "bolt_cam_optical_frame",
    },
}
FIXED_CAMERA_OPTICAL_FRAMES = (
    "/World/Camera_busbar/busbar_cam_optical_frame",
    "/World/Camera_bolt/bolt_cam_optical_frame",
)
TF_PUBLISH_NODE = "/World/ActionGraph/ros2_publish_transform_tree"
INVALID_FIXED_CAMERA_BRANCHES = (
    "/World/Graph/camera_graph/RenderProduct_busbar",
    "/World/Graph/camera_graph/CameraInfoPublish_busbar",
    "/World/Graph/camera_graph/RGBPublish_busbar",
    "/World/Graph/camera_graph/DepthPublish_busbar",
    "/World/Graph/camera_graph/RenderProduct_bolt",
    "/World/Graph/camera_graph/CameraInfoPublish_bolt",
    "/World/Graph/camera_graph/RGBPublish_bolt",
    "/World/Graph/camera_graph/DepthPublish_bolt",
)

NOVA_CARTER_ROOT = "/World/Nova_Carter/chassis_link"
M0609_PATH       = "/World/m0609"
EE_LINK_NAME     = "link_6"
BUSBAR_CAMERA_PATH = "/World/Camera_busbar"
BOLT_CAMERA_PATH = "/World/Camera_bolt"
GRIPPER_JOINTS   = ["finger_joint", "right_inner_knuckle_joint"]
GRIPPER_ROOT_PATH = f"{M0609_PATH}/onrobot_rg2ft"

# b83 카메라와 station 3 busbar 사이의 6DoF 상대변환. station 4/5에서도
# 해당 station Mesh의 실제 world pose와 합성하므로 카메라의 상대 시점이 동일하다.
B83_BUSBAR_CAMERA_REL_POSITION = np.array(
    [-0.059450, 0.005578, 1.046259],
    dtype=float,
)
B83_BUSBAR_CAMERA_REL_ORIENTATION = np.array(
    [0.7071068, 0.0, 0.0, 0.7071068],
    dtype=float,
)
B83_EXPECTED_STATION_CAMERA_POSITIONS = {
    3: np.array([0.45413, 2.67599, 1.31549]),
    4: np.array([-0.29177, 2.67599, 1.31549]),
    5: np.array([-1.02960, 2.67599, 1.31549]),
}
B83_EXPECTED_STATION_CAMERA_ORIENTATION = np.array(
    [0.7071068, 0.0, 0.0, 0.7071068],
    dtype=float,
)
B83_CAMERA_PREFLIGHT_TOLERANCE_M = 0.03
B83_CAMERA_PREFLIGHT_TOLERANCE_RAD = math.radians(0.5)

# feature/amr-full-integration에서 실제 주행 검증한 Nova Carter 차동구동 값.
# Collected_Busbar_fin USD의 wheel radius/좌우 joint 간격과도 일치한다.
AMR_LINEAR_SPEED    = 0.35    # m/s
AMR_ANGULAR_SPEED   = 0.75    # rad/s
AMR_POS_TOL         = 0.02    # m
AMR_YAW_TOL         = 0.03    # rad
AMR_WHEEL_RADIUS    = 0.14    # m
AMR_TRACK_WIDTH     = 0.4132  # m
AMR_WHEEL_SIGN      = float(os.environ.get("AMR_WHEEL_SIGN", "1.0"))
AMR_STEER_SIGN      = float(os.environ.get("AMR_STEER_SIGN", "1.0"))
AMR_MAX_WHEEL_RADPS = 10.0
AMR_WHEEL_JOINT_NAMES = ("joint_wheel_left", "joint_wheel_right")
if AMR_WHEEL_SIGN not in (-1.0, 1.0):
    raise ValueError("AMR_WHEEL_SIGN은 -1 또는 1이어야 합니다")
if AMR_STEER_SIGN not in (-1.0, 1.0):
    raise ValueError("AMR_STEER_SIGN은 -1 또는 1이어야 합니다")

STATION_BUSBAR_PATHS = {
    3: ("/World/Z_busbar3", "/World/Z_busbar3/Mesh"),
    4: ("/World/Z_busbar4", "/World/Z_busbar4/Mesh"),
    5: ("/World/Z_busbar5", "/World/Z_busbar5/Mesh"),
}
BUSBAR_TARGET_ASSOCIATION_MAX_M = 0.30

STATION_BOLT_PRIM_PATHS = {
    station: (
        f"/World/battery_pack{station}/_2_8V60Ah_BT/bolt_1",
        f"/World/battery_pack{station}/_2_8V60Ah_BT/bolt_2",
    )
    for station in (3, 4, 5)
}

# 너트 Prim 경로 (체결에 실제로 쓰이는 건 nut1/nut2뿐이지만, 씬에는 여분 너트
# nut3~nut6도 있고 이것들도 전부 AMR에 실린 부품 트레이 위 물체라 이동 시 같이
# 옮겨줘야 한다). 예전엔 nut1_01/nut2_01/02/03이라는 이름이었는데 씬에서 nut3~nut6로
# 순차 개명됨.
NUT1_ROOT_PATH      = "/World/nut1"
NUT2_ROOT_PATH      = "/World/nut2"
NUT1_POLYSHAPE_PATH = "/World/nut1/geo/PolyShape"
NUT2_POLYSHAPE_PATH = "/World/nut2/geo/PolyShape"
NUT1_INTERFERING_PEG_PATH = (
    "/World/Nova_Carter/chassis_link/carter_tray/peg_5"
)
NUT2_INTERFERING_PEG_PATH = (
    "/World/Nova_Carter/chassis_link/carter_tray/peg_4"
)
EXTRA_NUT_ROOT_PATHS = ["/World/nut3", "/World/nut4", "/World/nut5", "/World/nut6"]
EXTRA_NUT_POLYSHAPE_PATHS = [f"{p}/geo/PolyShape" for p in EXTRA_NUT_ROOT_PATHS]
# 각 nut의 SDF hole과 자기 tray peg cylinder가 겹치므로, 실제 rigid body와
# 그 peg 한 쌍만 제외한다. root/cart/robot/finger/다른 peg는 필터링하지 않는다.
NUT_EXACT_PEG_FILTER_SPECS = (
    (
        "NUT1",
        NUT1_ROOT_PATH,
        NUT1_POLYSHAPE_PATH,
        NUT1_INTERFERING_PEG_PATH,
    ),
    (
        "NUT2",
        NUT2_ROOT_PATH,
        NUT2_POLYSHAPE_PATH,
        NUT2_INTERFERING_PEG_PATH,
    ),
    (
        "NUT3",
        EXTRA_NUT_ROOT_PATHS[0],
        EXTRA_NUT_POLYSHAPE_PATHS[0],
        "/World/Nova_Carter/chassis_link/carter_tray/peg_3",
    ),
    (
        "NUT4",
        EXTRA_NUT_ROOT_PATHS[1],
        EXTRA_NUT_POLYSHAPE_PATHS[1],
        "/World/Nova_Carter/chassis_link/carter_tray/peg_1",
    ),
    (
        "NUT5",
        EXTRA_NUT_ROOT_PATHS[2],
        EXTRA_NUT_POLYSHAPE_PATHS[2],
        "/World/Nova_Carter/chassis_link/carter_tray/peg_0",
    ),
    (
        "NUT6",
        EXTRA_NUT_ROOT_PATHS[3],
        EXTRA_NUT_POLYSHAPE_PATHS[3],
        "/World/Nova_Carter/chassis_link/carter_tray/peg_2",
    ),
)

# 그리퍼 파라미터
GRIPPER_OPEN      = np.array([0.0, 0.0])
GRIPPER_CLOSE     = np.array([0.85, 0.85])
# battery4_main에서 실제 너트 파지에 사용한 닫힘 값
GRIPPER_CLOSE_NUT = np.array([0.96, 0.96])
GRIPPER_DELTA     = np.array([-0.5, -0.5])
GRIP_CLOSE_RAMP_STEPS = 50
GRIP_SETTLE_STEPS = 15  # 닫힌 손가락 안에서 tray glue 해제 후 결합 안정화 틱 수

# Kinematic Pose-Glue 파라미터
EE_OFFSET = np.array([0.0, 0.0, 0.185])
BUSBAR_HEIGHT = 0.003
BUSBAR_GRASP_Z_LOCAL = BUSBAR_HEIGHT + 0.02
BUSBAR_REST_ORIENTATION = np.array([0.5, -0.5, 0.5, 0.5])

# 기본 좌표 및 높이 정의
TARGET_INIT_JOINTS   = np.array([0.0, 0.0, np.radians(90.0), 0.0, np.radians(90.0), np.radians(90.0)])

# INIT_POSE/RETURN_HOME_JOINTS 관절 속도 제한 (rad/s) - 목표 각도를 한 틱에 통째로
# 명령하면 오차가 클 때 PD 드라이브가 최대 속도로 꽂아버려 반력으로 AMR까지 흔들리므로,
# 이 속도로 매 틱 목표를 조금씩만 전진시켜 부드럽게 도달하게 한다.
ARM_INIT_MAX_JOINT_SPEED = math.radians(20.0)  # rad/s

_POS_GRAB_PICK       = 0.455
SCAN_APPROACH_Z      = 0.7     # 배터리 스캔 고도
BUSBAR_SCAN_Z        = 0.7     # 버스바 스캔 고도
BUSBAR_APPROACH_Z    = 0.60   # 버스바 파지 접근 고도 (Z = 0.6m)
BATTERY_CENTER_Z     = 0.40   # 배터리 중점 이동 고도 (Z = 0.7m)

# 버스바 체결 및 하강 제어 파라미터
INSERT_SPEED            = 0.0005   # Step당 수직 하강 거리
BUSBAR_RELEASE_Z        = 0.37     # 그리퍼 해제 및 체결 완료 임계 Z 높이
INSERT_TOLERANCE_STRICT = 0.001    # Insert 단계 오차 허용범위 (1mm)

# error_fix의 /target_pose는 이 frame_id에서만 증분 명령으로 해석한다. 한 번의
# 증분을 실제 EE가 추종·정지한 뒤 12틱 확인해야 다음 post-motion 표본을 허용한다.
FINE_ALIGNMENT_DELTA_FRAME = "fine_alignment_delta"
FINE_STEP_POSITION_TOL_M = 0.005
FINE_STEP_ORIENTATION_TOL_RAD = math.radians(3.0)
FINE_STEP_MOTION_TOL_M = 0.0001
FINE_STEP_ORIENTATION_MOTION_TOL_RAD = math.radians(0.005)
FINE_STEP_SETTLE_STEPS = 12
# 80% 명령 응답에서 한 틱 정지 판정 해상도만큼을 관측 margin으로 뺀다.
# error_fix의 0.2 mm 스텝도 0.06 mm 이상의 command-direction 실제 이동을
# 증명해야 하므로 same-frame/no-motion ACK는 허용되지 않는다.
FINE_STEP_POSITION_RESPONSE_NOISE_MARGIN_M = 0.0001
FINE_STEP_ORIENTATION_RESPONSE_NOISE_MARGIN_RAD = math.radians(0.005)
# A settle ACK must prove that the arm executed nearly all of the incremental
# command whenever that command is above the observation noise floor.
FINE_STEP_MIN_RESPONSE_RATIO = 0.80
# During an active step only the RMPFlow controller target can advance along
# the correction direction, by at most 1 mm.  After strict settle, that exact
# effective XY is committed before ACK so the next command cannot jump back.
FINE_CONTROLLER_LEAD_GROWTH_M = 0.00005
FINE_CONTROLLER_MAX_LEAD_M = 0.001

# ══════════════════════════════════════════════════════════════════════════
#  너트 조립(Nut Assembly) 파라미터
# ══════════════════════════════════════════════════════════════════════════
NUT_SCAN_Z        = 0.9
NUT_POSE_POSITION_TOL_M = 0.02
NUT_POSE_ORIENTATION_TOL_RAD = math.radians(3.0)
NUT_POSE_SETTLE_STEPS = 12

# 초기 자세(INIT_POSE 직후 HOME_EE_POS) TCP 좌표 기준, 너트 1~6번까지의 실측 상대
# 오프셋(X,Y만 - Z는 NUT_PICK_Z로 고정). 초기 TCP=(1.0335, 0.1931, 0.8658), 너트1~6
# 실측 월드좌표를 각각 빼서 계산함 - AMR이 어느 배터리 스테이션에 있든, INIT_POSE로
# 되돌아가 새로 구한 HOME_EE_POS에 이 오프셋만 더하면 그 스테이션에서의 정확한 너트
# 위치가 나온다(월드 절대좌표를 하드코딩하면 스테이션이 바뀔 때마다 다 틀어짐).
NUT_OFFSET_FROM_HOME = {
    1: np.array([-0.4587, -0.2957]),
    2: np.array([-0.3683, -0.2961]),
    3: np.array([-0.2781, -0.2949]),
    4: np.array([-0.2781, -0.2093]),
    5: np.array([-0.3692, -0.2107]),
    6: np.array([-0.4589, -0.2083]),
}

NUT_HEIGHT         = 0.0095
NUT_GRASP_Z_LOCAL  = NUT_HEIGHT + 0.023
NUT_SUPPLY_TABLE_Z = 0.72                                   # 너트 공급대 높이
NUT_PICK_Z         = NUT_SUPPLY_TABLE_Z - (NUT_GRASP_Z_LOCAL - 0.0395)
NUT_APPROACH_Z     = 0.9                                     # 너트 파지 상공 고도 - 볼트 펙 위로
                                                              # 완전히 빠져나올 여유를 더 주기 위해
                                                              # 0.8->0.9로 올림(실측: 0.8은 부족해서
                                                              # 들어올리다 펙에 걸려 놓쳤음)
NUT_PEG_CLEARANCE_Z = 0.08                                   # 파지 직후 XY 이동 없이 먼저
                                                              # 수직으로 빠져나올 거리(80mm,
                                                              # 공급대 peg 길이보다 큰 안전값)
NUT_PEG_CLEAR_HOLD_STEPS = 15                                # 이탈 높이 도착 후 파지 안정화
                                                              # 대기(60Hz 기준 0.25초)
NUT_EE_COUPLING_TOLERANCE_M = 0.01                           # 파지 직후 저장한 EE-relative
                                                              # 너트 위치에서 허용할 최대 drift
NUT_EE_SETTLE_MOTION_TOLERANCE_M = 0.001                     # dynamics 전환 뒤 상대 위치가
                                                              # 이 값 이하로 움직여야 안정 tick
NUT_LIFT_COMMAND_MAX_STEP_M = 0.0005                         # friction grasp에 80mm 최종
                                                              # 목표를 한 틱에 주지 않는 Z ramp
NUT_PEG_CLEAR_COMMAND_MARGIN_M = 0.002                       # 실제 nut tracking 오차를 넘기기
                                                              # 위한 command-only Z 여유(2mm)
BOLT_APPROACH_Z    = 0.6                                     # 너트 체결 상공 고도

# ★ 스테이션별 볼트 1/2번 실측 월드 좌표 (test_isaac 씬 고정 배치 기준) ★
# 딕셔너리 키(1,2)는 "이번 스테이션의 몇 번째 볼트인가"이지, 트레이의 물리적 nut_index가
# 아니다(그건 STATION_NUT_INDICES가 따로 정한다).
STATION_BOLT_WORLD_POS = {
    3: {1: np.array([1.0552, 0.3722]), 2: np.array([1.2636, 0.0098])},
    4: {1: np.array([1.0552, -0.2047]), 2: np.array([1.2636, -0.5671])},
    5: {1: np.array([1.0552, -0.7312]), 2: np.array([1.2636, -1.0936])},
}

# 스테이션별로 6슬롯 너트 트레이 중 어느 물리적 너트(1~6)를 쓸지 - 배터리팩4는 3,4번,
# 배터리팩5는 5,6번 너트를 쓴다.
STATION_NUT_INDICES = {
    3: (1, 2),
    4: (3, 4),
    5: (5, 6),
}

# behavior_node.py의 AMR_STATION_POSES와 동일한 좌표 - /amr/goal_pose로 어느 스테이션에
# 와있는지 자동 판별하는 데 쓴다(battery/busbar 두 지점 다 등록해서 어느 태스크
# 시점이든 매칭되게 함).
STATION_AMR_POINTS = {
    3: [(0.6667, -0.0382), (0.5867, 1.9078)],
    4: [(0.6667, -0.6617), (-0.2271, 1.9078)],
    5: [(0.6667, -1.1964), (-0.9586, 1.9078)],
}
BUSBAR_AMR_EXPECTED_YAW = -1.5707

# 너트 체결(Screwing) 파라미터
ENGAGE_LEN           = 0.0125     # 체결 깊이 (12.5mm)
SCREW_TURNS_DEG      = 350.0      # 1회전당 350도
REGRASP_CYCLES       = 1          # 총 2회전 (Regrasp 1회)
SCREW_OMEGA_DEG_S    = 120.0      # 초당 120도 회전
REGRASP_LIFT_HEIGHT  = 0.06       # Regrasp 시 수직 상승 높이 (6cm)
REGRASP_Z_OFFSET     = 0.005      # Regrasp 하강/재파지 시 높이 보정

TOTAL_REV   = (SCREW_TURNS_DEG / 360.0) * (1 + REGRASP_CYCLES)
NUT_PITCH_M = ENGAGE_LEN / TOTAL_REV

TORQUE_THRESHOLD     = 45.0       # 6번 조인트 반력 임계값 (Nm)
STUCK_Z_DELTA_THRESH = 0.0001     # Z축 하강 멈춤 판정 기준 (0.1mm)
STUCK_STEP_LIMIT     = 12         # Z축 변화 없이 토크 지속되는 Step 수
TCP_FORCE_CHECK_Z    = 0.378      # TCP(EE) 높이가 이 값 이하일 때만 힘/토크 감지 활성화

# 위치 저장용 변수
HOME_EE_POS          = None
SCAN_POS             = None
BUSBAR_SCAN_POS      = None
BATTERY_CENTER_POS   = None
target_fine_pos      = None

PICK_TOLERANCE_STRICT    = 0.01     # 10mm
JOINT_TOLERANCE          = 0.02     # 관절 오차 허용범위 (rad)
PICK_TOLERANCE_LOOSE_VAL = 0.015
MAX_STUCK_STEPS          = 1000000  # Phase 타임아웃 기준
PHYSICS_DT               = 1.0 / 60.0
BUSBAR_GRASP_MIN_LIFT_M = 0.05
BUSBAR_LIFT_STABLE_STEPS = 10

URDF_PATH        = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
ROBOT_DESC_PATH  = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
RMPFLOW_CFG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")

# ══════════════════════════════════════════════════════════════════════════
#  [B] 비전 브릿지 및 유틸리티 함수
# ══════════════════════════════════════════════════════════════════════════
def configure_fixed_camera_bridges(stage):
    """고정캠 영상·TF를 전용 토픽으로만 내보내도록 현재 stage를 보정한다.

    이 보정은 OmniGraph가 stage를 처음 보기 전에 runtime overlay에 authoring해야
    한다. stage open 뒤 graph를 비활성화하면 이미 만들어진 synthetic-data
    writer가 남아 계속 잘못된 카메라 영상을 발행할 수 있다.
    """
    for node_path in INVALID_FIXED_CAMERA_BRANCHES:
        node = stage.GetPrimAtPath(node_path)
        if not node.IsValid():
            raise RuntimeError(f"비활성화할 카메라 graph node 누락: {node_path}")
        enabled_attr = node.GetAttribute("inputs:enabled")
        if not enabled_attr.IsValid():
            raise RuntimeError(f"카메라 graph enabled 속성 누락: {node_path}")
        enabled_attr.Set(False)

    for graph_path, config in FIXED_CAMERA_BRIDGES.items():
        graph = stage.GetPrimAtPath(graph_path)
        camera_path = config["camera"]
        camera = stage.GetPrimAtPath(camera_path)
        if not graph.IsValid() or not camera.IsValid():
            raise RuntimeError(
                f"고정 카메라 prim 누락: graph={graph_path}, camera={camera_path}"
            )
        graph.SetActive(True)

        render_product_path = f"{graph_path}/RenderProduct"
        render_product = stage.GetPrimAtPath(render_product_path)
        if not render_product.IsValid():
            raise RuntimeError(f"고정 카메라 RenderProduct 누락: {render_product_path}")
        camera_rel = render_product.GetRelationship("inputs:cameraPrim")
        expected = [Sdf.Path(camera_path)]
        if list(camera_rel.GetTargets()) != expected:
            camera_rel.SetTargets(expected)
        render_product.GetAttribute("inputs:enabled").Set(True)

        publishers = {
            "CameraInfoPublish": "camera_info",
            "RGBPublish": "/rgb",
            "DepthPublish": "/depth",
        }
        for node_name, topic_name in publishers.items():
            node_path = f"{graph_path}/{node_name}"
            node = stage.GetPrimAtPath(node_path)
            if not node.IsValid():
                raise RuntimeError(f"고정 카메라 publisher 누락: {node_path}")
            values = {
                "inputs:enabled": True,
                "inputs:nodeNamespace": config["namespace"],
                "inputs:frameId": config["optical_frame"],
                "inputs:topicName": topic_name,
            }
            for attribute_name, value in values.items():
                attribute = node.GetAttribute(attribute_name)
                if not attribute.IsValid():
                    raise RuntimeError(
                        f"고정 카메라 publisher 속성 누락: "
                        f"{node_path}.{attribute_name}"
                    )
                attribute.Set(value)

        depth_pcl_path = f"{graph_path}/DepthPclPublish"
        depth_pcl = stage.GetPrimAtPath(depth_pcl_path)
        if depth_pcl.IsValid():
            depth_pcl.GetAttribute("inputs:enabled").Set(False)

        print(
            f"[camera] {graph_path} -> /{config['namespace']} "
            f"(frame={config['optical_frame']}, camera={camera_path})"
        )

    tf_node = stage.GetPrimAtPath(TF_PUBLISH_NODE)
    if not tf_node.IsValid():
        raise RuntimeError(f"TF publish node 누락: {TF_PUBLISH_NODE}")

    for optical_path in FIXED_CAMERA_OPTICAL_FRAMES:
        if not stage.GetPrimAtPath(optical_path).IsValid():
            raise RuntimeError(f"고정 카메라 optical frame 누락: {optical_path}")

    target_rel = tf_node.GetRelationship("inputs:targetPrims")
    fixed_related = (
        {config["camera"] for config in FIXED_CAMERA_BRIDGES.values()}
        | set(FIXED_CAMERA_OPTICAL_FRAMES)
    )
    targets = [
        target
        for target in target_rel.GetTargets()
        if str(target) not in fixed_related
    ]
    targets.extend(Sdf.Path(path) for path in FIXED_CAMERA_OPTICAL_FRAMES)
    if list(target_rel.GetTargets()) != targets:
        target_rel.SetTargets(targets)
        print("[camera] busbar/bolt optical TF target 보정")

    print("[camera] 잘못된 /World 고정카메라 RenderProduct 분기 비활성화")


def configure_ros_domain(stage):
    """USD의 모든 ROS2Context를 현재 ``ROS_DOMAIN_ID``에 맞춘다."""
    try:
        domain_id = int(os.environ.get("ROS_DOMAIN_ID", "0"))
    except ValueError as exc:
        raise RuntimeError("ROS_DOMAIN_ID는 정수여야 합니다") from exc
    if not 0 <= domain_id <= 232:
        raise RuntimeError("ROS_DOMAIN_ID는 0~232 범위여야 합니다")

    found = 0
    for prim in stage.Traverse():
        node_type = prim.GetAttribute("node:type")
        if (
            not node_type.IsValid()
            or node_type.Get() != "isaacsim.ros2.bridge.ROS2Context"
        ):
            continue
        domain_attr = prim.GetAttribute("inputs:domain_id")
        env_attr = prim.GetAttribute("inputs:useDomainIDEnvVar")
        if not domain_attr.IsValid() or not env_attr.IsValid():
            raise RuntimeError(f"ROS2Context 속성 누락: {prim.GetPath()}")
        domain_attr.Set(domain_id)
        env_attr.Set(True)
        found += 1

    if found == 0:
        raise RuntimeError("USD에서 ROS2Context를 찾지 못했습니다")
    print(f"[ROS2] USD Context {found}개 domain={domain_id} 적용")


def configure_monotonic_simulation_time(stage):
    """Keep camera source stamps monotonic across Stop→Play restarts."""
    configured = 0
    for prim in stage.Traverse():
        reset_attr = prim.GetAttribute(
            "inputs:resetSimulationTimeOnStop"
        )
        if not reset_attr.IsValid():
            continue
        reset_attr.Set(False)
        configured += 1
    print(
        "[camera] Stop→Play source timestamp reset 비활성 "
        f"({configured} authored nodes)"
    )


def _nut_collision_filter_failures(
    stage,
    *,
    label,
    root_path,
    body_path,
    peg_path,
):
    """Return contract violations for one intentionally exact nut/peg pair."""

    body_path, expected_targets = exact_collision_filter_targets(
        body_path,
        peg_path,
    )
    body_prim = stage.GetPrimAtPath(body_path)
    peg_prim = stage.GetPrimAtPath(expected_targets[0])
    nut_root = stage.GetPrimAtPath(root_path)
    failures = []

    if not body_prim.IsValid():
        failures.append(f"{label} 실제 rigid body prim 누락: {body_path}")
    else:
        if not body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            failures.append(f"{label} RigidBodyAPI 누락: {body_path}")
        if not body_prim.HasAPI(UsdPhysics.CollisionAPI):
            failures.append(f"{label} CollisionAPI 누락: {body_path}")
        actual_targets = tuple(
            str(path)
            for path in UsdPhysics.FilteredPairsAPI(
                body_prim,
            ).GetFilteredPairsRel().GetTargets()
        )
        if actual_targets != expected_targets:
            failures.append(
                f"{label} exact filteredPairs={actual_targets}, "
                f"expected={expected_targets}"
            )

    if not peg_prim.IsValid():
        failures.append(
            f"{label} 간섭 peg prim 누락: {expected_targets[0]}"
        )
    elif not peg_prim.HasAPI(UsdPhysics.CollisionAPI):
        failures.append(
            f"{label} 간섭 peg CollisionAPI 누락: {expected_targets[0]}"
        )

    if not nut_root.IsValid():
        failures.append(f"{label} root prim 누락: {root_path}")
    else:
        legacy_targets = tuple(
            str(path)
            for path in nut_root.GetRelationship(
                "physics:filteredPairs",
            ).GetTargets()
        )
        if legacy_targets:
            failures.append(
                f"{label} legacy broad filteredPairs가 남아 있음: "
                f"{legacy_targets}"
            )

    # Filtering is unilateral in USD Physics.  Catch an unexpected relation
    # anywhere else that targets this nut (or one authored on a descendant),
    # since either would silently broaden this one-pair exception.
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path in {body_path, root_path}:
            continue
        relation = prim.GetRelationship("physics:filteredPairs")
        if not relation.IsValid():
            continue
        targets = tuple(str(path) for path in relation.GetTargets())
        if not targets:
            continue
        authored_on_nut = prim_path.startswith(f"{root_path}/")
        targets_nut = any(
            target == root_path
            or target.startswith(f"{root_path}/")
            for target in targets
        )
        if authored_on_nut or targets_nut:
            failures.append(
                f"예상하지 않은 {label} filteredPairs: "
                f"{prim_path} -> {targets}"
            )

    return failures


def _configure_nut_exact_peg_filter(
    stage,
    *,
    label,
    root_path,
    body_path,
    peg_path,
):
    """Neutralize one legacy broad filter, then author one body/peg pair."""

    body_path, target_paths = exact_collision_filter_targets(
        body_path,
        peg_path,
    )
    body_prim = stage.GetPrimAtPath(body_path)
    peg_prim = stage.GetPrimAtPath(target_paths[0])
    missing = [
        path
        for path, prim in (
            (body_path, body_prim),
            (target_paths[0], peg_prim),
        )
        if not prim.IsValid()
    ]
    if missing:
        raise RuntimeError(
            f"{label} exact collision-filter prim 누락: "
            + ", ".join(missing)
        )
    if not body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"{label} RigidBodyAPI 누락: {body_path}")
    if not body_prim.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError(f"{label} CollisionAPI 누락: {body_path}")
    if not peg_prim.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError(
            f"{label} 간섭 peg CollisionAPI 누락: {target_paths[0]}"
        )

    nut_root = stage.GetPrimAtPath(root_path)
    if not nut_root.IsValid():
        raise RuntimeError(f"{label} root prim 누락: {root_path}")
    legacy_relation = nut_root.GetRelationship("physics:filteredPairs")
    if legacy_relation.IsValid() and not legacy_relation.SetTargets([]):
        raise RuntimeError(
            f"{label} legacy broad filteredPairs neutralize 실패"
        )

    filtered_api = UsdPhysics.FilteredPairsAPI.Apply(body_prim)
    relation = filtered_api.CreateFilteredPairsRel()
    if not relation.SetTargets([Sdf.Path(target_paths[0])]):
        raise RuntimeError(f"{label} exact filteredPairs authoring 실패")

    failures = _nut_collision_filter_failures(
        stage,
        label=label,
        root_path=root_path,
        body_path=body_path,
        peg_path=peg_path,
    )
    if failures:
        details = "\n  - ".join(failures)
        raise RuntimeError(
            f"{label} exact collision-filter 설정 실패:\n  - {details}"
        )
    print(
        f"[{label} PHYSICS] legacy broad filter neutralized; exact pair="
        f"{body_path} -> {target_paths[0]}"
    )


def _all_nut_collision_filter_failures(stage):
    """Return violations for the six intentionally exact own-peg pairs."""

    failures = []
    for label, root_path, body_path, peg_path in NUT_EXACT_PEG_FILTER_SPECS:
        failures.extend(
            _nut_collision_filter_failures(
                stage,
                label=label,
                root_path=root_path,
                body_path=body_path,
                peg_path=peg_path,
            )
        )
    return failures


def configure_exact_nut_peg_filters(stage):
    """Filter each nut body against its own tray peg, and nothing else."""

    for label, root_path, body_path, peg_path in NUT_EXACT_PEG_FILTER_SPECS:
        _configure_nut_exact_peg_filter(
            stage,
            label=label,
            root_path=root_path,
            body_path=body_path,
            peg_path=peg_path,
        )


def create_runtime_stage_overlay(source_path, output_directory):
    """원본 USD를 저장하지 않고, 첫 graph 생성 전 적용할 wrapper USD를 만든다."""
    source_stage = Usd.Stage.Open(str(Path(source_path).resolve()))
    if source_stage is None:
        raise RuntimeError(f"원본 USD stage 열기 실패: {source_path}")

    overlay_path = Path(output_directory) / "runtime_stage.usda"
    layer = Sdf.Layer.CreateNew(str(overlay_path))
    if layer is None:
        raise RuntimeError(f"runtime USD layer 생성 실패: {overlay_path}")
    layer.subLayerPaths = [str(Path(source_path).resolve())]

    stage = Usd.Stage.Open(layer)
    if stage is None:
        raise RuntimeError(f"runtime USD stage 생성 실패: {overlay_path}")
    with Usd.EditContext(stage, layer):
        # Stage 단위/축 같은 메타데이터는 sublayer에서 root layer로 상속되지 않고
        # USD fallback(centimeter, Y-up)을 쓸 수 있으므로 원본 값을 명시 복사한다.
        UsdGeom.SetStageMetersPerUnit(
            stage,
            UsdGeom.GetStageMetersPerUnit(source_stage),
        )
        UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
        stage.SetTimeCodesPerSecond(source_stage.GetTimeCodesPerSecond())
        stage.SetFramesPerSecond(source_stage.GetFramesPerSecond())
        stage.SetStartTimeCode(source_stage.GetStartTimeCode())
        stage.SetEndTimeCode(source_stage.GetEndTimeCode())
        source_default_prim = source_stage.GetDefaultPrim()
        if source_default_prim.IsValid():
            runtime_default_prim = stage.GetPrimAtPath(
                source_default_prim.GetPath(),
            )
            if runtime_default_prim.IsValid():
                stage.SetDefaultPrim(runtime_default_prim)

        configure_ros_domain(stage)
        configure_monotonic_simulation_time(stage)
        configure_fixed_camera_bridges(stage)
        configure_exact_nut_peg_filters(stage)
    if not layer.Save():
        raise RuntimeError(f"runtime USD layer 저장 실패: {overlay_path}")

    print(f"[camera] 원본 비저장 runtime overlay 준비: {overlay_path}")
    return overlay_path


class Execute_Isaac_Busar(Node):
    """ArmNode / BehaviorNode / VisionNode 통신 브릿지"""
    def __init__(self):
        super().__init__("execute_isaac_busar")
        self.latest_target_pose = None
        self.latest_fine_alignment_delta = None
        self.requested_task = None
        self.alignment_success = False
        self.expected_fine_alignment_generation = None

        self.sub_target_pose = self.create_subscription(
            PoseStamped, '/target_pose', self._on_target_pose, 10
        )
        self.sub_task_cmd = self.create_subscription(
            String, '/task_command', self._on_task_command, 10
        )

        # amr_node 연동 (amr_node.py: PUB /amr/goal_pose, SUB /amr/sim_pose, SUB /amr/cancel)
        self.amr_goal_pose = None
        self.amr_cancel_requested = False
        self.sub_amr_goal_pose = self.create_subscription(
            PoseStamped, '/amr/goal_pose', self._on_amr_goal_pose, 10
        )
        self.sub_amr_cancel = self.create_subscription(
            Empty, '/amr/cancel', self._on_amr_cancel, 10
        )
        self.pub_amr_sim_pose = self.create_publisher(Pose2D, '/amr/sim_pose', 10)
        self.pub_amr_drive_state = self.create_publisher(
            String, '/amr/drive_state', 10
        )

        self.pub_phase = self.create_publisher(String, '/isaac_phase', 10)
        self.pub_progress = self.create_publisher(Float32, '/isaac_progress', 10)
        self.pub_status = self.create_publisher(String, '/isaac_status', 10)
        self.pub_errorfix_command = self.create_publisher(String, '/errorfix_command', 10)
        self.pub_busbar_perception_reset = self.create_publisher(
            Empty,
            '/busbar_cam/perception/reset_cache',
            10,
        )
        self.pub_wrist_perception_reset = self.create_publisher(
            Empty,
            '/wrist/perception/reset_cache',
            10,
        )
        self.pub_bolt_perception_reset = self.create_publisher(
            Empty,
            '/bolt_cam/perception/reset_cache',
            10,
        )

    def _on_target_pose(self, msg: PoseStamped):
        if msg.header.frame_id.startswith(
            FINE_ALIGNMENT_DELTA_FRAME
        ):
            self.latest_fine_alignment_delta = msg
        else:
            self.latest_target_pose = msg

    def _on_amr_goal_pose(self, msg: PoseStamped):
        self.amr_goal_pose = msg

    def _on_amr_cancel(self, msg: Empty):
        self.amr_cancel_requested = True

    def publish_amr_drive_state(
        self,
        state,
        message,
        goal_stamp,
        pose=None,
        command=None,
        wheel_speeds=None,
    ):
        """amr_node가 실제 위치/yaw/정지를 확인할 수 있는 상태를 발행한다."""
        if pose is None:
            raise ValueError("/amr/drive_state에는 실제 AMR pose가 필요합니다")
        if wheel_speeds is None:
            wheel_speeds = (0.0, 0.0)
        payload = {
            "version": 1,
            "state": state,
            "goal_stamp": goal_stamp,
            "message": message,
            "pose": {
                "world_x": pose.x,
                "world_y": pose.y,
                "world_yaw": pose.yaw,
                "linear_speed": pose.linear_speed,
                "angular_speed": pose.angular_speed,
            },
            "control": {
                "phase": (
                    command.phase if command is not None
                    else state.lower()
                ),
                "linear": (
                    command.linear if command is not None else 0.0
                ),
                "angular": (
                    command.angular if command is not None else 0.0
                ),
                "distance_error": (
                    command.distance_error
                    if command is not None else None
                ),
                "heading_error": (
                    command.heading_error
                    if command is not None else None
                ),
                "yaw_error": (
                    command.yaw_error if command is not None else None
                ),
                "travel_direction": (
                    command.travel_direction
                    if command is not None else ""
                ),
                "commanded_left_wheel_radps": float(wheel_speeds[0]),
                "commanded_right_wheel_radps": float(wheel_speeds[1]),
            },
        }
        msg = String()
        msg.data = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.pub_amr_drive_state.publish(msg)

    def _on_task_command(self, msg: String):
        self.get_logger().info(f"[Task Command 수신]: {msg.data}")
        if (
            msg.data == "ALIGNMENT_SUCCESS"
            or msg.data.startswith("ALIGNMENT_SUCCESS:")
        ):
            _, separator, generation_text = msg.data.partition(":")
            if not separator:
                self.get_logger().warn(
                    "generation 없는 legacy ALIGNMENT_SUCCESS 무시")
                return
            try:
                generation = int(generation_text)
            except ValueError:
                self.get_logger().warn(
                    f"잘못된 ALIGNMENT_SUCCESS generation 무시: {msg.data}")
                return
            expected_generation = (
                self.expected_fine_alignment_generation
            )
            if generation <= 0 or expected_generation is None:
                self.get_logger().warn(
                    "활성 FINE_ALIGNMENT와 연결되지 않은 완료 신호 무시: "
                    f"{msg.data}")
                return
            if generation != expected_generation:
                self.get_logger().warn(
                    "stale FINE_ALIGNMENT 완료 신호 무시: "
                    f"expected={expected_generation}, received={generation}")
                return
            self.alignment_success = True
            return
        self.requested_task = msg.data


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
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)


def enable_collision_recursively(stage, prim_path):
    """Enable contact geometry without releasing a kinematically glued body."""
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        return
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(
                prim
            ).GetCollisionEnabledAttr().Set(True)


def enable_physics_recursively(stage, prim_path):
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        return
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(True)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(True)


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def _quat_normalize(q):
    quaternion = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if not np.all(np.isfinite(quaternion)) or norm <= 1.0e-9:
        raise ValueError("유효하지 않은 quaternion입니다")
    return quaternion / norm


def _quat_angular_error(q1, q2):
    """Return the shortest 3-D quaternion distance in radians."""
    q1 = _quat_normalize(q1)
    q2 = _quat_normalize(q2)
    quaternion_dot = min(1.0, abs(float(np.dot(q1, q2))))
    return 2.0 * math.acos(quaternion_dot)


def _quat_rotate_vec(q, v):
    q = _quat_normalize(q)
    qv = np.array([0.0, v[0], v[1], v[2]])
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[1:]


def compute_local_offset(parent_pos, parent_quat, child_pos, child_quat):
    """battery4_main 방식: AMR 기준 너트의 로컬 위치/자세를 계산한다."""
    parent_pos = np.asarray(parent_pos, dtype=float)
    parent_quat = _quat_normalize(parent_quat)
    child_pos = np.asarray(child_pos, dtype=float)
    child_quat = _quat_normalize(child_quat)
    parent_inv = _quat_conj(parent_quat)
    return (
        _quat_rotate_vec(parent_inv, child_pos - parent_pos),
        _quat_mul(parent_inv, child_quat),
    )


def compose_world_pose(parent_pos, parent_quat, local_pos, local_quat):
    """battery4_main 방식: AMR 월드 포즈와 로컬 오프셋으로 너트 포즈를 복원한다."""
    parent_pos = np.asarray(parent_pos, dtype=float)
    parent_quat = _quat_normalize(parent_quat)
    local_quat = _quat_normalize(local_quat)
    return (
        parent_pos + _quat_rotate_vec(parent_quat, np.asarray(local_pos, dtype=float)),
        _quat_normalize(_quat_mul(parent_quat, local_quat)),
    )


def busbar_camera_pose_for_asset(busbar_xform):
    """현재 station busbar Mesh에 b83의 camera-to-busbar 6DoF를 합성한다."""
    busbar_position, busbar_orientation = busbar_xform.get_world_pose()
    camera_position, camera_orientation = compose_world_pose(
        busbar_position,
        busbar_orientation,
        B83_BUSBAR_CAMERA_REL_POSITION,
        B83_BUSBAR_CAMERA_REL_ORIENTATION,
    )
    quat_norm = float(np.linalg.norm(camera_orientation))
    if (
        not np.all(np.isfinite(camera_position))
        or not np.all(np.isfinite(camera_orientation))
        or quat_norm <= 1.0e-9
    ):
        raise RuntimeError("busbar camera 합성 pose가 유효하지 않습니다")
    return camera_position, camera_orientation / quat_norm


def bolt_camera_pose_for_target(
    stage,
    bolt_path,
    initial_position,
    initial_orientation,
):
    """한 bolt의 XY로 이동하며 USD의 기존 카메라 Z·자세를 유지한다."""
    bolt_prim = stage.GetPrimAtPath(bolt_path)
    if not bolt_prim.IsValid():
        raise ValueError(f"bolt camera target prim이 없습니다: {bolt_path}")
    bolt_position = world_xf(stage, bolt_path).ExtractTranslation()
    camera_position = np.array(
        [
            float(bolt_position[0]),
            float(bolt_position[1]),
            float(initial_position[2]),
        ],
        dtype=float,
    )
    camera_orientation = np.asarray(initial_orientation, dtype=float).copy()
    if (
        not np.all(np.isfinite(camera_position))
        or not np.all(np.isfinite(camera_orientation))
    ):
        raise RuntimeError("bolt camera target pose가 유효하지 않습니다")
    return camera_position, camera_orientation


def disable_rigidbody_sleep(stage, prim_path):
    """너트처럼 AMR에 용접된 채 오래 정지해 있다가 그리퍼로 옮겨지는 RigidBody는, PhysX가
    "안 움직인다"고 판단해 자동으로 재워버릴 수 있다 - 잠든 채로는 그리퍼가 손가락을
    닫아도 접촉/마찰이 제대로 반영 안 될 수 있어서, sleepThreshold를 0으로 낮춰
    애초에 잠들지 않게 한다."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rb.CreateSleepThresholdAttr().Set(0.0)


def find_rigidbody_ancestor(stage, start_path):
    """start_path부터 위로 올라가며 실제 UsdPhysics.RigidBodyAPI가 적용된 조상 프림을 찾는다."""
    prim = stage.GetPrimAtPath(start_path)
    while prim.IsValid():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(prim.GetPath())
        if prim.GetPath() == Sdf.Path.absoluteRootPath:
            break
        prim = prim.GetParent()
    return None


NUT_AMR_JOINT_NAME = "PhysicsFixedJoint_AMR"


def remove_nut_amr_joint(stage, nut_root_path, joint_name=NUT_AMR_JOINT_NAME):
    """attach_nut_to_amr이 만든 조인트를 완전히 지운다 (비활성화만으로는 부족 - 남은 조인트
    프림이 옛 상대위치를 들고 있다가 다음 attach 때 재사용되면 두 프레임이 어긋난다)."""
    joint_path = f"{nut_root_path}/{joint_name}"
    joint_prim = stage.GetPrimAtPath(joint_path)
    if joint_prim.IsValid():
        UsdPhysics.FixedJoint(joint_prim).GetJointEnabledAttr().Set(False)
        stage.RemovePrim(joint_path)


def attach_nut_to_amr(stage, nut_root_path, nut_rigidbody_path, amr_link_path, nut_xform, robot, joint_name=NUT_AMR_JOINT_NAME):
    """너트를 현재 위치 그대로 AMR 링크에 물리 FixedJoint로 고정한다 (순간이동 없이 그 자리에서 부착).
    world.reset() 직후에만 호출한다 - 이 시점 이후 시뮬레이션 도중에 새로 만드는 조인트는
    PhysX 라이브 반영이 안 되는 문제가 있었다(그리퍼 조인트에서 실측 확인됨)."""
    amr_prim = stage.GetPrimAtPath(amr_link_path)
    nut_prim = stage.GetPrimAtPath(nut_rigidbody_path)
    if not amr_prim.IsValid() or not nut_prim.IsValid():
        return None

    remove_nut_amr_joint(stage, nut_root_path, joint_name)

    amr_pos, amr_quat = robot.get_world_pose()
    amr_pos = np.asarray(amr_pos, dtype=float)
    amr_quat = _quat_normalize(amr_quat)

    nut_pos, nut_quat = nut_xform.get_world_pose()
    nut_pos = np.asarray(nut_pos, dtype=float)
    nut_quat = _quat_normalize(nut_quat)

    rel_pos, rel_quat = compute_local_offset(
        amr_pos,
        amr_quat,
        nut_pos,
        nut_quat,
    )

    # 검증: local0을 다시 world로 복원했을 때 실제 너트 위치와 일치하는지 확인.
    frame0_world_pos = amr_pos + _quat_rotate_vec(amr_quat, rel_pos)
    frame_error = float(np.linalg.norm(frame0_world_pos - nut_pos))
    joint_path = f"{nut_root_path}/{joint_name}"
    if frame_error > 0.001:
        print(f"[ERROR] {joint_path} FixedJoint frame 불일치 {frame_error*1000:.2f}mm -> 생성 취소")
        return None

    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateJointEnabledAttr(False)
    joint.CreateBody0Rel().SetTargets([amr_link_path])
    joint.CreateBody1Rel().SetTargets([nut_rigidbody_path])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in rel_pos]))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(rel_quat[0]), Gf.Vec3f(*[float(v) for v in rel_quat[1:]])))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.GetJointEnabledAttr().Set(True)
    print(f"[NUT JOINT ATTACH] {joint_path} frame_error={frame_error*1000:.3f}mm")
    return joint


def detach_nut_from_amr(stage, nut_root_path, joint_name=NUT_AMR_JOINT_NAME):
    """attach_nut_to_amr으로 만든 조인트를 비활성화해서 너트를 AMR에서 풀어준다 - 그리퍼가
    물리적으로 붙잡아 들어올리기 직전(NUT_GRASP 시작 전)에 호출한다."""
    joint_path = f"{nut_root_path}/{joint_name}"
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid():
        print(f"[WARN] {joint_path} 조인트가 없습니다 (이미 해제됐거나 생성된 적 없음)")
        return False
    enabled_attr = UsdPhysics.FixedJoint(joint_prim).GetJointEnabledAttr()
    if enabled_attr:
        enabled_attr.Set(False)
    print(f"[NUT JOINT DETACH] {joint_path}")
    return True




BUSBAR_GRIP_JOINT_NAME = "PhysicsFixedJoint_Gripper"


def attach_busbar_to_gripper(
    stage,
    gripper_link_path,
    robot,
    busbar_xform,
    busbar_mesh_path,
):
    """버스바를 그리퍼(EE 링크)에 FixedJoint로 고정한다.

    위치를 측정한 Mesh와 FixedJoint Body1을 같은 prim으로 통일해야 부모/자식
    로컬 프레임 차이 때문에 물체가 튀지 않는다.
    """
    gripper_prim = stage.GetPrimAtPath(gripper_link_path)
    busbar_prim = stage.GetPrimAtPath(busbar_mesh_path)
    if not gripper_prim.IsValid() or not busbar_prim.IsValid():
        return None

    ee_pos, ee_quat = robot.end_effector.get_world_pose()
    real_pos, real_quat = busbar_xform.get_world_pose()
    ee_pos = np.asarray(ee_pos, dtype=float)
    ee_quat = _quat_normalize(ee_quat)
    real_pos = np.asarray(real_pos, dtype=float)
    real_quat = _quat_normalize(real_quat)

    rel_pos, rel_quat = compute_local_offset(
        ee_pos,
        ee_quat,
        real_pos,
        real_quat,
    )

    joint_path = f"{busbar_mesh_path}/{BUSBAR_GRIP_JOINT_NAME}"
    joint_prim = stage.GetPrimAtPath(joint_path)
    joint = UsdPhysics.FixedJoint(joint_prim) if joint_prim.IsValid() else UsdPhysics.FixedJoint.Define(stage, joint_path)

    joint.CreateBody0Rel().SetTargets([gripper_link_path])
    joint.CreateBody1Rel().SetTargets([busbar_mesh_path])

    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in rel_pos]))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(float(rel_quat[0]), Gf.Vec3f(*[float(v) for v in rel_quat[1:]])))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateJointEnabledAttr().Set(True)

    print(f"[BUSBAR ATTACH] EE=({ee_pos[0]:.4f},{ee_pos[1]:.4f},{ee_pos[2]:.4f}) "
          f"Busbar=({real_pos[0]:.4f},{real_pos[1]:.4f},{real_pos[2]:.4f}) rel_pos={rel_pos}")
    return joint


def detach_busbar_from_gripper(stage, busbar_mesh_path):
    joint_prim = stage.GetPrimAtPath(
        f"{busbar_mesh_path}/{BUSBAR_GRIP_JOINT_NAME}"
    )
    if joint_prim.IsValid():
        UsdPhysics.FixedJoint(joint_prim).GetJointEnabledAttr().Set(False)



def nut_paths_for_index(nut_index):
    """nut_index(1~6)에 따라 (root_path, polyshape_path) 반환. 3~6번은 EXTRA_NUT_*
    (스테이션4=3,4번 / 스테이션5=5,6번 너트) 목록에서 가져온다."""
    if nut_index == 1:
        return NUT1_ROOT_PATH, NUT1_POLYSHAPE_PATH
    if nut_index == 2:
        return NUT2_ROOT_PATH, NUT2_POLYSHAPE_PATH
    idx = nut_index - 3
    return EXTRA_NUT_ROOT_PATHS[idx], EXTRA_NUT_POLYSHAPE_PATHS[idx]


def world_xf(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim 경로가 올바르지 않습니다: {path}")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)


def find_prim_path(stage, root_path, name):
    root = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def _set_runtime_gains(robot, joint_names, stiffness, damping, max_force):
    """robot._articulation_view(런타임 PhysX 텐서 뷰)에 직접 gain을 적용한다.

    UsdPhysics.DriveAPI의 stiffness/damping 속성을 시뮬레이션 시작 후에 stage에서
    직접 Set()해도 이미 초기화된 ArticulationView는 그 값을 다시 읽어가지 않는다
    (Articulation.set_gains의 save_to_usd=False 기본값이 보여주듯, USD와 런타임
    physics view는 서로 다른 저장소다). 그래서 "강성을 올렸는데도 계속 풀린다"는
    증상은 실제로는 런타임에 전혀 반영이 안 된 것이었다. robot.apply_action이
    쓰는 것과 동일한 _articulation_view.set_gains/set_max_efforts를 써야 실제
    시뮬레이션에 적용된다."""
    if robot is None:
        return
    view = getattr(robot, "_articulation_view", None)
    if view is None:
        return
    # view.set_gains(..., joint_names=...)의 내부 이름->인덱스 변환이 dof_names 순서와
    # 어긋나 있어(IndexError로 실측 확인됨) 범위를 벗어난 인덱스를 만들어냈다. 스크립트
    # 전역에서 이미 잘 쓰이고 있는 robot.get_dof_index()로 직접 인덱스를 구해
    # joint_indices로 넘기면 이 문제를 피할 수 있다.
    joint_indices = [robot.get_dof_index(name) for name in joint_names]
    n = len(joint_indices)
    kps = np.full((1, n), float(stiffness))
    kds = np.full((1, n), float(damping))
    efforts = np.full((1, n), float(max_force))
    view.set_gains(kps=kps, kds=kds, joint_indices=joint_indices, save_to_usd=False)
    view.set_max_efforts(values=efforts, joint_indices=joint_indices)


AMR_DRIVE_SETTINGS = {
    "/World/Nova_Carter/joint_caster_base": (100.0, 10.0, float("inf")),
    "/World/Nova_Carter/joint_swing_left": (0.0, 1.0e-5, float("inf")),
    "/World/Nova_Carter/joint_swing_right": (0.0, 1.0e-5, float("inf")),
    "/World/Nova_Carter/joint_wheel_left": (0.0, 1.0e6, 200.0),
    "/World/Nova_Carter/joint_wheel_right": (0.0, 1.0e6, 200.0),
}


def prepare_amr_physics(stage):
    """잠겨 저장된 Carter drive를 검증된 물리 주행 값으로 복원한다."""
    for path, (stiffness, damping, max_force) in AMR_DRIVE_SETTINGS.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"AMR drive joint 누락: {path}")
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            raise RuntimeError(f"AMR angular DriveAPI 누락: {path}")
        drive.GetStiffnessAttr().Set(stiffness)
        drive.GetDampingAttr().Set(damping)
        drive.GetMaxForceAttr().Set(max_force)
        drive.GetTargetPositionAttr().Set(0.0)
        drive.GetTargetVelocityAttr().Set(0.0)


def _configure_runtime_wheel_drive(robot):
    """초기화된 PhysX articulation에도 wheel velocity-drive gain을 적용한다."""
    indices = [robot.get_dof_index(name) for name in AMR_WHEEL_JOINT_NAMES]
    if any(index is None or int(index) < 0 for index in indices):
        raise RuntimeError(f"AMR wheel DOF 누락: {indices}")
    _set_runtime_gains(
        robot,
        AMR_WHEEL_JOINT_NAMES,
        stiffness=0.0,
        damping=1.0e6,
        max_force=200.0,
    )
    return np.asarray([int(index) for index in indices], dtype=np.int64)


def _configure_runtime_wheel_brake(robot):
    """Hold the current wheel angles while the arm works.

    A zero velocity target alone allowed the wheels to roll slowly under arm
    reaction forces.  Use a finite position drive while stopped, then
    ``unlock_amr_base`` restores the pure velocity drive before every goal.
    """
    indices = [robot.get_dof_index(name) for name in AMR_WHEEL_JOINT_NAMES]
    if any(index is None or int(index) < 0 for index in indices):
        raise RuntimeError(f"AMR wheel DOF 누락: {indices}")
    wheel_indices = np.asarray(
        [int(index) for index in indices],
        dtype=np.int64,
    )
    wheel_positions = np.asarray(
        robot.get_joint_positions()[wheel_indices],
        dtype=np.float32,
    )
    _set_runtime_gains(
        robot,
        AMR_WHEEL_JOINT_NAMES,
        stiffness=1.0e4,
        damping=1.0e4,
        max_force=200.0,
    )
    robot.apply_action(
        ArticulationAction(
            joint_positions=wheel_positions,
            joint_velocities=np.zeros(2, dtype=np.float32),
            joint_indices=wheel_indices,
        )
    )
    return wheel_indices


def _apply_amr_wheel_speeds(robot, wheel_indices, left_radps, right_radps):
    """좌우 실제 wheel joint에 목표 각속도(rad/s)를 전달한다."""
    robot.apply_action(
        ArticulationAction(
            joint_velocities=np.asarray(
                [left_radps, right_radps],
                dtype=np.float32,
            ),
            joint_indices=wheel_indices,
        )
    )


def lock_amr_base(stage, amr_root_path, robot=None):
    """작업 중 wheel 목표속도를 0으로 유지한다.

    예전 1e9 position-lock은 두 번째 주행부터 velocity 명령을 막았으므로 사용하지
    않는다. 물리 drive의 제동(damping/max force)은 그대로 유지한다.
    """
    del amr_root_path
    prepare_amr_physics(stage)
    if robot is not None:
        _configure_runtime_wheel_brake(robot)


def unlock_amr_base(stage, amr_root_path, robot=None):
    """주행 직전 stage와 runtime wheel drive를 대칭적으로 복원한다."""
    del amr_root_path
    prepare_amr_physics(stage)
    if robot is not None:
        _configure_runtime_wheel_drive(robot)


def stiffen_gripper_grip(robot, joint_names=None):
    """물체(버스바 등)를 옮기는 동안 그리퍼 관절 강성을 확 높인다 - AMR 이동 중
    반동(순간이동으로 인한 물리적 충격)에도 그리퍼가 살짝 벌어지지 않도록 하기 위함.
    GRIPPER_OPEN/CLOSE 자체(목표 위치)는 그대로 동작하고, 그 목표를 얼마나 세게
    붙잡을지(강성/최대 힘)만 올라간다.

    반드시 robot._articulation_view.set_gains/set_max_efforts(런타임 physics view)로
    적용해야 한다 - 이전에는 UsdPhysics.DriveAPI를 stage에 직접 Set()했는데, 시뮬레이션이
    이미 시작된 뒤에는 그 변경이 실제 물리 계산에 반영되지 않아 "강성을 올려도 계속
    벌어지는" 문제가 있었다."""
    if joint_names is None:
        joint_names = GRIPPER_JOINTS
    _set_runtime_gains(robot, joint_names, 1.0e9, 1.0e6, 1.0e9)


def quat_wxyz_to_yaw(quat_wxyz):
    """월드 Z축 기준 yaw(rad)만 추출 (AMR은 평면 위를 움직이므로 roll/pitch는 무시)."""
    w, x, y, z = (float(value) for value in quat_wxyz)
    if not all(math.isfinite(value) for value in (w, x, y, z)):
        raise ValueError("orientation quaternion에 NaN 또는 inf가 있습니다")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1.0e-9:
        raise ValueError("orientation quaternion이 0입니다")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def goal_stamp_key(msg):
    """amr_node와 동일한 ``sec:nanosec`` 목표 상관 키."""
    return f"{int(msg.header.stamp.sec)}:{int(msg.header.stamp.nanosec)}"


def pose_stamp_ns(msg):
    """Return a positive integer correlation key from a PoseStamped."""
    stamp_ns = (
        int(msg.header.stamp.sec) * 1_000_000_000
        + int(msg.header.stamp.nanosec)
    )
    if stamp_ns <= 0:
        raise ValueError("PoseStamped source stamp가 0입니다")
    return stamp_ns


def measure_amr_pose(robot):
    """실제 PhysX chassis pose와 속도를 물리 제어기 입력으로 변환한다."""
    position, orientation = robot.get_world_pose()
    linear_velocity = robot.get_linear_velocity()
    angular_velocity = robot.get_angular_velocity()
    values = [
        float(value)
        for value in (
            *position,
            *orientation,
            *linear_velocity,
            *angular_velocity,
        )
    ]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("AMR pose/velocity에 NaN 또는 inf가 있습니다")

    w, x, y, z = (float(value) for value in orientation)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1.0e-9:
        raise RuntimeError("AMR quaternion이 유효하지 않습니다")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return PoseState(
        x=float(position[0]),
        y=float(position[1]),
        yaw=math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        ),
        z=float(position[2]),
        up_z=1.0 - 2.0 * (x * x + y * y),
        qx=x,
        qy=y,
        qz=z,
        qw=w,
        linear_speed=float(np.linalg.norm(linear_velocity)),
        angular_speed=float(np.linalg.norm(angular_velocity)),
    )


def resolve_station_from_amr_xy(x, y, tolerance=0.3):
    """AMR 목표좌표(x,y)를 STATION_AMR_POINTS와 비교해 가장 가까운 스테이션 번호를 찾는다.
    behavior_node.py가 station_id 문자열을 execute_isaac.py에 직접 전달하는 경로가 없어서,
    대신 이미 받고 있는 /amr/goal_pose로 역추정한다. tolerance(m) 밖이면 매칭 실패로 본다."""
    best_station, best_dist = None, None
    for station, points in STATION_AMR_POINTS.items():
        for px, py in points:
            d = math.hypot(x - px, y - py)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_station = station
    if best_dist is not None and best_dist <= tolerance:
        return best_station
    return None


def resolve_busbar_stop_from_amr_xy(x, y, tolerance=AMR_POS_TOL):
    """승인된 AMR position tolerance 안의 canonical busbar stop을 찾는다."""
    best_station, best_dist = None, None
    for station, points in STATION_AMR_POINTS.items():
        busbar_x, busbar_y = points[1]
        distance = math.hypot(x - busbar_x, y - busbar_y)
        if best_dist is None or distance < best_dist:
            best_station, best_dist = station, distance
    if best_dist is not None and best_dist <= tolerance:
        return best_station
    return None


def busbar_station_pose_error(robot, station):
    """Return actual pose plus planar errors from the station busbar stop.

    ``SCAN_BUSBAR`` and ``PICK_BUSBAR`` are only reachable after the AMR has
    arrived at the selected station's busbar stop.  Calling either task from
    the battery stop (or from the initial scene pose) leaves the target
    outside the arm workspace; with the intentionally large arm stuck budget
    that otherwise looks like an infinite wait.
    """
    points = STATION_AMR_POINTS.get(station)
    if points is None:
        raise ValueError(f"unknown station: {station}")
    expected_x, expected_y = points[1]
    pose = measure_amr_pose(robot)
    position_error = math.hypot(
        pose.x - expected_x,
        pose.y - expected_y,
    )
    yaw_error = abs(
        normalize_angle(pose.yaw - BUSBAR_AMR_EXPECTED_YAW)
    )
    return pose, position_error, yaw_error


def yaw_rotated_quat(base_wxyz, delta_deg):
    """base_wxyz 오리엔테이션을 월드 Z축 기준으로 delta_deg 만큼 추가 회전시킨 쿼터니언 반환 (Screwing 회전용)"""
    base_q = Gf.Quatd(float(base_wxyz[0]), Gf.Vec3d(float(base_wxyz[1]), float(base_wxyz[2]), float(base_wxyz[3])))
    base_rot = Gf.Rotation(base_q)
    extra_rot = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), float(delta_deg))
    combined = extra_rot * base_rot
    q = combined.GetQuat()
    return np.array([q.GetReal(), *q.GetImaginary()])


def resolve_nut_assets(nut_index, nut1_xform, nut2_xform, extra_nut_xforms=None):
    """nut_index(1~6)에 따라 대상 너트 Xform과 라벨을 반환. 3~6번은 extra_nut_xforms
    (EXTRA_NUT_ROOT_PATHS와 같은 순서) 목록에서 가져온다 - 안 넘겨주면 라벨만 반환."""
    if nut_index == 1:
        return nut1_xform, "너트 1번"
    if nut_index == 2:
        return nut2_xform, "너트 2번"
    xform = None
    if extra_nut_xforms is not None:
        idx = nut_index - 3
        if 0 <= idx < len(extra_nut_xforms):
            xform = extra_nut_xforms[idx]
    return xform, f"너트 {nut_index}번"


def update_target_positions(target_pose_msg: PoseStamped):
    global BUSBAR_APPROACH_POS, BUSBAR_PICK_POS, BUSBAR_LIFT_MOVE_POS

    pos = target_pose_msg.pose.position
    
    # 1) 실제 파지 좌표 (Z = 0.455m)
    pick_z = _POS_GRAB_PICK
    new_pick = np.array([pos.x, pos.y, pick_z])
    BUSBAR_PICK_POS = new_pick

    # 2) 상공 접근 좌표 (Z = 0.600m 고정)
    BUSBAR_APPROACH_POS = np.array([pos.x, pos.y, BUSBAR_APPROACH_Z])

    # 3) 파지 직후에는 옆으로 이동하지 않고 동일 XY에서 수직 상승한다.
    BUSBAR_LIFT_MOVE_POS = np.array([pos.x, pos.y, BUSBAR_APPROACH_Z])

    print(f"[Dynamic Target Set] Approach Pos (Z={BUSBAR_APPROACH_Z:.2f}m): ({BUSBAR_APPROACH_POS[0]:.4f}, {BUSBAR_APPROACH_POS[1]:.4f}, {BUSBAR_APPROACH_POS[2]:.4f})")
    print(f"[Dynamic Target Set] Pick Pos     (Z={BUSBAR_PICK_POS[2]:.4f}m): ({BUSBAR_PICK_POS[0]:.4f}, {BUSBAR_PICK_POS[1]:.4f}, {BUSBAR_PICK_POS[2]:.4f})")


def _world_pose_for_path(stage, prim_path):
    matrix = world_xf(stage, prim_path)
    position = np.asarray(matrix.ExtractTranslation(), dtype=float)
    quaternion = matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    orientation = np.array(
        [
            float(quaternion.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ],
        dtype=float,
    )
    return position, _quat_normalize(orientation)


def validate_stage_contract(stage):
    """로봇을 움직이지 않고 fin USD의 통합 계약을 검사한다."""
    failures = []

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isclose(meters_per_unit, 1.0, abs_tol=1.0e-9):
        failures.append(f"metersPerUnit={meters_per_unit}, expected=1")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        failures.append(
            f"upAxis={UsdGeom.GetStageUpAxis(stage)}, expected=Z"
        )
    if not any(prim.IsA(UsdPhysics.Scene) for prim in stage.Traverse()):
        failures.append("UsdPhysics.Scene prim 누락")

    failures.extend(_all_nut_collision_filter_failures(stage))

    for graph_path, config in FIXED_CAMERA_BRIDGES.items():
        camera_path = config["camera"]
        render_product_path = f"{graph_path}/RenderProduct"
        graph = stage.GetPrimAtPath(graph_path)
        render_product = stage.GetPrimAtPath(render_product_path)
        camera = stage.GetPrimAtPath(camera_path)
        if not graph.IsValid() or not graph.IsActive():
            failures.append(f"고정 카메라 bridge 비활성/누락: {graph_path}")
        if not camera.IsValid() or not camera.IsA(UsdGeom.Camera):
            failures.append(f"Camera prim 누락/타입 오류: {camera_path}")
        if render_product.GetAttribute("inputs:enabled").Get() is not True:
            failures.append(
                f"고정 카메라 RenderProduct 비활성: {render_product_path}"
            )
        targets = list(
            render_product.GetRelationship("inputs:cameraPrim").GetTargets()
        )
        if targets != [Sdf.Path(camera_path)]:
            failures.append(
                f"{render_product_path} cameraPrim={targets}, "
                f"expected={camera_path}"
            )
        publishers = {
            "CameraInfoPublish": "camera_info",
            "RGBPublish": "/rgb",
            "DepthPublish": "/depth",
        }
        for node_name, topic_name in publishers.items():
            node_path = f"{graph_path}/{node_name}"
            node = stage.GetPrimAtPath(node_path)
            expected_attributes = {
                "inputs:enabled": True,
                "inputs:nodeNamespace": config["namespace"],
                "inputs:frameId": config["optical_frame"],
                "inputs:topicName": topic_name,
            }
            for attribute_name, expected_value in expected_attributes.items():
                actual_value = node.GetAttribute(attribute_name).Get()
                if actual_value != expected_value:
                    failures.append(
                        f"{node_path}.{attribute_name}={actual_value!r}, "
                        f"expected={expected_value!r}"
                    )
        depth_pcl = stage.GetPrimAtPath(f"{graph_path}/DepthPclPublish")
        if depth_pcl.IsValid() and (
            depth_pcl.GetAttribute("inputs:enabled").Get() is not False
        ):
            failures.append(f"불필요한 depth_pcl publisher 활성: {graph_path}")

    for node_path in INVALID_FIXED_CAMERA_BRANCHES:
        node = stage.GetPrimAtPath(node_path)
        if (
            not node.IsValid()
            or node.GetAttribute("inputs:enabled").Get() is not False
        ):
            failures.append(f"잘못된 고정카메라 분기 활성/누락: {node_path}")

    tf_node = stage.GetPrimAtPath(TF_PUBLISH_NODE)
    tf_targets = set(
        str(path)
        for path in tf_node.GetRelationship("inputs:targetPrims").GetTargets()
    )
    for config in FIXED_CAMERA_BRIDGES.values():
        camera_path = config["camera"]
        if camera_path in tf_targets:
            failures.append(f"TF가 Camera prim까지 발행함: {camera_path}")
    for optical_path in FIXED_CAMERA_OPTICAL_FRAMES:
        if optical_path not in tf_targets:
            failures.append(f"optical TF target 누락: {optical_path}")

    expected_domain = int(os.environ.get("ROS_DOMAIN_ID", "0"))
    context_count = 0
    for prim in stage.Traverse():
        node_type = prim.GetAttribute("node:type")
        if (
            not node_type.IsValid()
            or node_type.Get() != "isaacsim.ros2.bridge.ROS2Context"
        ):
            continue
        context_count += 1
        domain = prim.GetAttribute("inputs:domain_id").Get()
        use_env = prim.GetAttribute("inputs:useDomainIDEnvVar").Get()
        if int(domain) != expected_domain or not bool(use_env):
            failures.append(
                f"ROS2Context 불일치: {prim.GetPath()} "
                f"domain={domain}, use_env={use_env}"
            )
    if context_count == 0:
        failures.append("활성 ROS2Context 누락")

    for prim in stage.Traverse():
        reset_attr = prim.GetAttribute(
            "inputs:resetSimulationTimeOnStop"
        )
        if reset_attr.IsValid() and bool(reset_attr.Get()):
            failures.append(
                "Stop→Play camera source stamp reset 활성: "
                f"{prim.GetPath()}"
            )

    station_camera_poses = {}
    for station, (root_path, mesh_path) in STATION_BUSBAR_PATHS.items():
        if (
            not stage.GetPrimAtPath(root_path).IsValid()
            or not stage.GetPrimAtPath(mesh_path).IsValid()
        ):
            failures.append(
                f"station {station} busbar prim 누락: "
                f"{root_path}, {mesh_path}"
            )
            continue
        mesh_position, mesh_orientation = _world_pose_for_path(
            stage,
            mesh_path,
        )
        camera_position, camera_orientation = compose_world_pose(
            mesh_position,
            mesh_orientation,
            B83_BUSBAR_CAMERA_REL_POSITION,
            B83_BUSBAR_CAMERA_REL_ORIENTATION,
        )
        station_camera_poses[station] = (
            camera_position,
            camera_orientation,
        )
        expected = B83_EXPECTED_STATION_CAMERA_POSITIONS[station]
        position_error = float(np.linalg.norm(camera_position - expected))
        if position_error > B83_CAMERA_PREFLIGHT_TOLERANCE_M:
            failures.append(
                f"station {station} busbar camera pose 오차 "
                f"{position_error:.4f}m > "
                f"{B83_CAMERA_PREFLIGHT_TOLERANCE_M:.4f}m; "
                f"actual=({camera_position[0]:.5f}, "
                f"{camera_position[1]:.5f}, {camera_position[2]:.5f}), "
                f"expected=({expected[0]:.5f}, {expected[1]:.5f}, "
                f"{expected[2]:.5f}), mesh=({mesh_position[0]:.5f}, "
                f"{mesh_position[1]:.5f}, {mesh_position[2]:.5f}), "
                f"mesh_quat=({mesh_orientation[0]:.7f}, "
                f"{mesh_orientation[1]:.7f}, "
                f"{mesh_orientation[2]:.7f}, "
                f"{mesh_orientation[3]:.7f})"
            )

        expected_orientation = _quat_normalize(
            B83_EXPECTED_STATION_CAMERA_ORIENTATION
        )
        actual_orientation = _quat_normalize(camera_orientation)
        quaternion_dot = float(
            np.clip(
                abs(np.dot(actual_orientation, expected_orientation)),
                0.0,
                1.0,
            )
        )
        orientation_error = 2.0 * math.acos(quaternion_dot)
        if orientation_error > B83_CAMERA_PREFLIGHT_TOLERANCE_RAD:
            failures.append(
                f"station {station} busbar camera orientation 오차 "
                f"{math.degrees(orientation_error):.4f}deg > "
                f"{math.degrees(B83_CAMERA_PREFLIGHT_TOLERANCE_RAD):.4f}deg; "
                f"actual=({actual_orientation[0]:.7f}, "
                f"{actual_orientation[1]:.7f}, "
                f"{actual_orientation[2]:.7f}, "
                f"{actual_orientation[3]:.7f})"
            )

        bolt_paths = STATION_BOLT_PRIM_PATHS[station]
        for bolt_path in bolt_paths:
            if not stage.GetPrimAtPath(bolt_path).IsValid():
                failures.append(
                    f"station {station} bolt prim 누락: {bolt_path}"
                )

    for joint_path in AMR_DRIVE_SETTINGS:
        joint_prim = stage.GetPrimAtPath(joint_path)
        if not joint_prim.IsValid():
            failures.append(f"AMR wheel/caster DOF 누락: {joint_path}")
            continue
        if not UsdPhysics.DriveAPI.Get(joint_prim, "angular"):
            failures.append(f"AMR angular DriveAPI 누락: {joint_path}")

    if failures:
        details = "\n  - ".join(failures)
        raise RuntimeError(f"preflight 실패:\n  - {details}")

    print(
        "[PREFLIGHT] PASS: camera/render/optical TF/ROS domain/"
        "monotonic source stamp/single-publisher bridge/station prim/"
        "wheel DriveAPI/six exact nut/own-peg collision pairs"
    )
    for station, (position, orientation) in station_camera_poses.items():
        print(
            f"[PREFLIGHT] station {station}: camera position="
            f"({position[0]:.5f}, {position[1]:.5f}, {position[2]:.5f}), "
            f"quat=({orientation[0]:.6f}, {orientation[1]:.6f}, "
            f"{orientation[2]:.6f}, {orientation[3]:.6f}), "
            f"busbar={STATION_BUSBAR_PATHS[station][1]}"
        )


def run_wheel_smoke_test(world, robot, wheel_indices, controller):
    """실제 wheel velocity만 사용해 전진·후진·회전·go-to-pose를 검사한다."""
    class _NoRootTeleportProxy:
        """Delegate articulation access while forbidding root pose writes."""

        def __init__(self, target):
            self._target = target
            self.root_pose_write_attempts = 0

        def __getattr__(self, name):
            return getattr(self._target, name)

        def set_world_pose(self, *args, **kwargs):
            self.root_pose_write_attempts += 1
            raise RuntimeError(
                "smoke-test drive path가 AMR root set_world_pose를 호출했습니다"
            )

    robot = _NoRootTeleportProxy(robot)
    pose_history = []

    def step_with_command(linear, angular, steps):
        left, right = controller.to_wheel_speeds(
            linear,
            angular,
            wheel_sign=AMR_WHEEL_SIGN,
            steer_sign=AMR_STEER_SIGN,
        )
        for _ in range(steps):
            _apply_amr_wheel_speeds(
                robot,
                wheel_indices,
                left,
                right,
            )
            world.step(render=False)
            pose = measure_amr_pose(robot)
            if pose_history:
                jump = math.hypot(
                    pose.x - pose_history[-1].x,
                    pose.y - pose_history[-1].y,
                )
                if jump > 0.10:
                    raise RuntimeError(
                        "smoke-test 중 AMR root pose 불연속 감지 "
                        f"({jump:.3f}m/tick)"
                    )
            pose_history.append(pose)
        _apply_amr_wheel_speeds(robot, wheel_indices, 0.0, 0.0)
        for _ in range(12):
            world.step(render=False)
            pose_history.append(measure_amr_pose(robot))
        return pose_history[-1]

    start = measure_amr_pose(robot)
    pose_history.append(start)
    forward = step_with_command(0.15, 0.0, 90)
    forward_projection = (
        (forward.x - start.x) * math.cos(start.yaw)
        + (forward.y - start.y) * math.sin(start.yaw)
    )
    if forward_projection < 0.02:
        raise RuntimeError(
            "wheel smoke-test 전진 실패: "
            f"projection={forward_projection:.4f}m"
        )

    reverse = step_with_command(-0.15, 0.0, 90)
    reverse_projection = (
        (reverse.x - forward.x) * math.cos(forward.yaw)
        + (reverse.y - forward.y) * math.sin(forward.yaw)
    )
    if reverse_projection > -0.02:
        raise RuntimeError(
            "wheel smoke-test 후진 실패: "
            f"projection={reverse_projection:.4f}m"
        )

    before_turn = reverse
    after_turn = step_with_command(0.0, 0.45, 90)
    turn_delta = math.atan2(
        math.sin(after_turn.yaw - before_turn.yaw),
        math.cos(after_turn.yaw - before_turn.yaw),
    )
    if abs(turn_delta) < 0.10:
        raise RuntimeError(
            f"wheel smoke-test 회전 실패: delta={turn_delta:.4f}rad"
        )

    # 현재 heading 뒤 12cm를 목표로 주어 자동 후진 선택과 최종 허용오차를
    # 실제 PhysX pose에서 함께 확인한다.
    goal = Goal2D(
        x=after_turn.x - 0.12 * math.cos(after_turn.yaw),
        y=after_turn.y - 0.12 * math.sin(after_turn.yaw),
        yaw=after_turn.yaw,
        yaw_required=True,
        allow_reverse=True,
        align_yaw_before_drive=False,
    )
    controller.set_goal(goal, current_pose=after_turn)
    if controller.travel_direction != "reverse":
        raise RuntimeError(
            "wheel smoke-test 자동 후진 선택 실패: "
            f"{controller.travel_direction}"
        )

    command = None
    pose = after_turn
    phase_counts = {}
    tail_samples = []
    for _ in range(2400):
        command = controller.compute(pose, PHYSICS_DT)
        phase_counts[command.phase] = phase_counts.get(command.phase, 0) + 1
        left, right = controller.to_wheel_speeds(
            command.linear,
            command.angular,
            wheel_sign=AMR_WHEEL_SIGN,
            steer_sign=AMR_STEER_SIGN,
        )
        _apply_amr_wheel_speeds(robot, wheel_indices, left, right)
        world.step(render=False)
        next_pose = measure_amr_pose(robot)
        jump = math.hypot(next_pose.x - pose.x, next_pose.y - pose.y)
        if jump > 0.10:
            raise RuntimeError(
                "go-to-pose 중 AMR root pose 불연속 감지 "
                f"({jump:.3f}m/tick)"
            )
        tail_samples.append(
            (
                command.phase,
                command.distance_error,
                abs(command.yaw_error),
                pose.linear_speed,
                pose.angular_speed,
                next_pose.linear_speed,
                next_pose.angular_speed,
            )
        )
        if len(tail_samples) > 12:
            tail_samples.pop(0)
        pose = next_pose
        if command.arrived:
            break
    _apply_amr_wheel_speeds(robot, wheel_indices, 0.0, 0.0)

    if command is None or not command.arrived:
        raise RuntimeError(
            "wheel smoke-test go-to-pose가 2400틱 내 수렴하지 못했습니다: "
            f"phase={getattr(command, 'phase', 'none')}, "
            f"distance={math.hypot(goal.x - pose.x, goal.y - pose.y):.4f}m, "
            f"yaw={abs(normalize_angle(goal.yaw - pose.yaw)):.4f}rad, "
            f"pose=({pose.x:.4f},{pose.y:.4f},{pose.yaw:.4f}), "
            f"phase_counts={phase_counts}, tail={tail_samples}"
        )
    distance_error = math.hypot(goal.x - pose.x, goal.y - pose.y)
    yaw_error = abs(
        math.atan2(
            math.sin(goal.yaw - pose.yaw),
            math.cos(goal.yaw - pose.yaw),
        )
    )
    if distance_error > AMR_POS_TOL or yaw_error > AMR_YAW_TOL:
        raise RuntimeError(
            "wheel smoke-test 최종 pose 허용오차 실패: "
            f"distance={distance_error:.4f}m, yaw={yaw_error:.4f}rad"
        )
    if robot.root_pose_write_attempts != 0:
        raise RuntimeError(
            "wheel smoke-test에서 AMR root pose write가 감지됐습니다"
        )

    print(
        "[SMOKE] PASS: wheel-only forward/reverse/rotate/go-to-pose, "
        f"final distance={distance_error:.4f}m <= {AMR_POS_TOL:.4f}m, "
        f"yaw={yaw_error:.4f}rad <= {AMR_YAW_TOL:.4f}rad, "
        "root set_world_pose calls=0"
    )


# ══════════════════════════════════════════════════════════════════════════
#  [C] 메인 파이프라인
# ══════════════════════════════════════════════════════════════════════════
def main():
    global BUSBAR_APPROACH_POS, BUSBAR_PICK_POS, BUSBAR_LIFT_MOVE_POS, SCAN_POS, HOME_EE_POS, BUSBAR_SCAN_POS, BATTERY_CENTER_POS, target_fine_pos

    usd_file_path = Path(USD_PATH).resolve()
    if not usd_file_path.is_file():
        raise FileNotFoundError(f"[ERROR] USD 파일을 찾을 수 없습니다: {usd_file_path}")

    # OmniGraph는 open_stage 중 graph를 구성할 수 있으므로, stage를 연 다음 session
    # layer를 고치는 방식으로는 이미 생성된 카메라 writer를 확실히 제거할 수 없다.
    # 원본 fin USD를 sublayer로 둔 임시 wrapper를 먼저 만들어 첫 open부터 보정된
    # graph만 보이게 한다. TemporaryDirectory는 main 종료 뒤 자동 정리된다.
    runtime_stage_directory = tempfile.TemporaryDirectory(
        prefix="amr_runtime_usd_",
    )
    runtime_stage_path = create_runtime_stage_overlay(
        usd_file_path,
        runtime_stage_directory.name,
    )

    ctx = omni.usd.get_context()
    opened = ctx.open_stage(str(runtime_stage_path))
    if not opened:
        raise RuntimeError(
            f"[ERROR] runtime Stage open 실패: {runtime_stage_path}"
        )
    stage = ctx.get_stage()
    if not stage:
        raise RuntimeError(
            f"[ERROR] runtime Stage를 로드하지 못했습니다: {runtime_stage_path}"
        )

    for _ in range(15):
        simulation_app.update()

    if _PREFLIGHT_ONLY:
        validate_stage_contract(stage)
        ctx.close_stage()
        simulation_app.close()
        return

    # USD에는 wheel/caster가 1e9 position-drive로 잠겨 저장돼 있다. World/reset보다
    # 먼저 검증된 velocity-drive 값으로 되돌려야 PhysX wheel 명령이 적용된다.
    prepare_amr_physics(stage)
    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT)

    busbar_assets = {}
    for station, (root_path, mesh_path) in STATION_BUSBAR_PATHS.items():
        if (
            not stage.GetPrimAtPath(root_path).IsValid()
            or not stage.GetPrimAtPath(mesh_path).IsValid()
        ):
            raise RuntimeError(
                f"스테이션 {station} 버스바 prim 누락: "
                f"{root_path}, {mesh_path}"
            )
        xform = SingleXFormPrim(
            mesh_path,
            name=f"busbar_station_{station}",
        )
        initial_position, initial_orientation = xform.get_world_pose()
        busbar_assets[station] = {
            "root_path": root_path,
            "mesh_path": mesh_path,
            "xform": xform,
            "initial_position": np.asarray(initial_position).copy(),
            "initial_orientation": np.asarray(initial_orientation).copy(),
        }

    busbar_root_path = None
    busbar_mesh_path = None
    busbar_xform = None

    busbar_camera_xform = SingleXFormPrim(
        BUSBAR_CAMERA_PATH,
        name="busbar_fixed_camera",
    )
    busbar_camera_init_pos, busbar_camera_init_quat = (
        busbar_camera_xform.get_world_pose()
    )
    bolt_camera_xform = SingleXFormPrim(
        BOLT_CAMERA_PATH,
        name="bolt_fixed_camera",
    )
    bolt_camera_init_pos, bolt_camera_init_quat = (
        bolt_camera_xform.get_world_pose()
    )

    nut1_xform = SingleXFormPrim(NUT1_POLYSHAPE_PATH, name="nut1_poly") if stage.GetPrimAtPath(NUT1_POLYSHAPE_PATH).IsValid() else None
    nut2_xform = SingleXFormPrim(NUT2_POLYSHAPE_PATH, name="nut2_poly") if stage.GetPrimAtPath(NUT2_POLYSHAPE_PATH).IsValid() else None
    init_nut1_pos, init_nut1_quat = nut1_xform.get_world_pose() if nut1_xform else (None, None)
    init_nut2_pos, init_nut2_quat = nut2_xform.get_world_pose() if nut2_xform else (None, None)

    # 진단용: NUT1/2_POLYSHAPE_PATH에 실제 RigidBodyAPI가 있는지, 아니면 조상 프림에
    # 있는지 확인 - 다르면 조인트 Body1 대상을 재검토해야 한다.
    print(f"[PHYSICS BODY] AMR  = {find_rigidbody_ancestor(stage, NOVA_CARTER_ROOT)}")
    print(f"[PHYSICS BODY] NUT1 = {find_rigidbody_ancestor(stage, NUT1_POLYSHAPE_PATH)} (used: {NUT1_POLYSHAPE_PATH})")
    print(f"[PHYSICS BODY] NUT2 = {find_rigidbody_ancestor(stage, NUT2_POLYSHAPE_PATH)} (used: {NUT2_POLYSHAPE_PATH})")

    # 너트가 AMR에 용접된 채 오래 정지해 있다가 그리퍼 조인트로 넘어갈 때 잠들어있으면
    # 안 된다 - 애초에 안 잠들도록 설정.
    disable_rigidbody_sleep(stage, NUT1_POLYSHAPE_PATH)
    disable_rigidbody_sleep(stage, NUT2_POLYSHAPE_PATH)
    for extra_poly in EXTRA_NUT_POLYSHAPE_PATHS:
        disable_rigidbody_sleep(stage, extra_poly)

    # 체결에 직접 쓰이진 않지만 AMR 부품 트레이 위 여분 너트들 - 이동 시 nut1/2와 똑같이 같이 옮겨야 한다.
    extra_nut_xforms = [
        SingleXFormPrim(p, name=f"extra_nut_poly_{i}") if stage.GetPrimAtPath(p).IsValid() else None
        for i, p in enumerate(EXTRA_NUT_POLYSHAPE_PATHS)
    ]
    extra_nut_inits = [
        (xf.get_world_pose() if xf else (None, None)) for xf in extra_nut_xforms
    ]

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
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )

    for _ in range(30):
        world.step(render=True)

    wheel_indices = _configure_runtime_wheel_drive(robot)
    lock_amr_base(stage, NOVA_CARTER_ROOT, robot=robot)
    drive_controller = GoToPoseController(
        ControllerConfig(
            position_tolerance=AMR_POS_TOL,
            yaw_tolerance=AMR_YAW_TOL,
            max_linear=AMR_LINEAR_SPEED,
            max_angular=AMR_ANGULAR_SPEED,
            min_angular=0.30,
            wheel_radius=AMR_WHEEL_RADIUS,
            track_width=AMR_TRACK_WIDTH,
            max_wheel_radps=AMR_MAX_WHEEL_RADPS,
            settle_steps=12,
            allow_reverse=True,
        )
    )
    print(
        "[AMR PHYSICS] 실제 wheel velocity 주행 준비: "
        f"indices={wheel_indices.tolist()}, radius={AMR_WHEEL_RADIUS:.4f}m, "
        f"track={AMR_TRACK_WIDTH:.4f}m, "
        f"wheel_sign={AMR_WHEEL_SIGN:+.0f}, "
        f"steer_sign={AMR_STEER_SIGN:+.0f}, "
        "direction=goal-start auto(forward/reverse)"
    )

    runtime_handles = {"node": None, "closed": False}

    def shutdown_runtime():
        """Idempotently stop physical motion and release runtime resources."""
        if runtime_handles["closed"]:
            return
        runtime_handles["closed"] = True

        try:
            _apply_amr_wheel_speeds(
                robot, wheel_indices, 0.0, 0.0
            )
        except Exception as exc:
            print(f"[SHUTDOWN] wheel zero 명령 실패: {exc}")

        node = runtime_handles["node"]
        if node is not None:
            try:
                node.destroy_node()
            except Exception as exc:
                print(f"[SHUTDOWN] ROS node 정리 실패: {exc}")
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as exc:
            print(f"[SHUTDOWN] rclpy 정리 실패: {exc}")
        try:
            world.clear_instance()
        except Exception as exc:
            print(f"[SHUTDOWN] World 정리 실패: {exc}")
        try:
            ctx.close_stage()
        except Exception as exc:
            print(f"[SHUTDOWN] stage 정리 실패: {exc}")
        try:
            simulation_app.close()
        except Exception as exc:
            print(f"[SHUTDOWN] SimulationApp 정리 실패: {exc}")

    # 정상 종료뿐 아니라 KeyboardInterrupt·런타임 예외에서도 wheel 0을 보장한다.
    atexit.register(shutdown_runtime)

    if _SMOKE_TEST:
        # main 진입 직후에는 팔 작업용 wheel position brake가 걸려 있다.
        # smoke는 순수 velocity drive를 검증해야 하므로 먼저 대칭적으로 해제한다.
        unlock_amr_base(stage, NOVA_CARTER_ROOT, robot=robot)
        validate_stage_contract(stage)
        run_wheel_smoke_test(
            world,
            robot,
            wheel_indices,
            drive_controller,
        )
        shutdown_runtime()
        atexit.unregister(shutdown_runtime)
        return

    # battery4_main의 검증된 너트 운반 방식: PICK 전에는 물리를 끄고 AMR 기준 로컬
    # 오프셋으로 매 프레임 포즈를 갱신한다. PICK 명령을 받은 너트만 이 glue에서 풀고
    # 물리를 켜서 그리퍼의 실제 접촉/마찰로 집는다.
    nut_xforms_all = [nut1_xform, nut2_xform, *extra_nut_xforms]
    nut_roots_all = [NUT1_ROOT_PATH, NUT2_ROOT_PATH, *EXTRA_NUT_ROOT_PATHS]
    nut_polys_all = [NUT1_POLYSHAPE_PATH, NUT2_POLYSHAPE_PATH, *EXTRA_NUT_POLYSHAPE_PATHS]
    nut_local_offsets = [None] * len(nut_xforms_all)
    nut_released = [False] * len(nut_xforms_all)
    amr_glue_pos, amr_glue_quat = robot.get_world_pose()
    for i, (nut_xf, nut_root) in enumerate(zip(nut_xforms_all, nut_roots_all)):
        if nut_xf is None:
            continue
        remove_nut_amr_joint(stage, nut_root)
        disable_physics_recursively(stage, nut_root)
        nut_pos, nut_quat = nut_xf.get_world_pose()
        nut_local_offsets[i] = compute_local_offset(
            amr_glue_pos, amr_glue_quat, nut_pos, nut_quat
        )

    quat_busbar = euler_to_quaternion_wxyz(0.0, 3.1415, 1.5708)
    quat_nut = euler_to_quaternion_wxyz(0.0, 3.1415, 0.0)

    rclpy.init()
    isaac_node = Execute_Isaac_Busar()
    runtime_handles["node"] = isaac_node

    arm_controller = RMPFlowController(
        name="m0609_busbar_controller",
        robot_articulation=robot,
        urdf_path=URDF_PATH,
        robot_description_path=ROBOT_DESC_PATH,
        rmpflow_config_path=RMPFLOW_CFG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
    )

    def sync_rmpflow_base_pose():
        """AMR이 이동한 뒤(또는 강제로 멈춘 뒤) RMPFlow가 가정하는 베이스 위치/자세를
        m0609/base_link의 실제 world transform으로 다시 맞춘다. 이걸 안 하면 RMPFlow는
        팔이 예전 베이스 위치에 있는 줄 알고 IK를 풀어서 엉뚱한 자세가 나온다."""
        base_link_xf = world_xf(stage, f"{M0609_PATH}/base_link")
        base_pos = base_link_xf.ExtractTranslation()
        base_quat = base_link_xf.ExtractRotationQuat()
        arm_controller._motion_policy.set_robot_base_pose(
            robot_position=np.array([base_pos[0], base_pos[1], base_pos[2]]),
            robot_orientation=np.array([base_quat.GetReal(), *[float(x) for x in base_quat.GetImaginary()]]),
        )

    def hold_current_arm_pose():
        """현재 팔 6축을 hold하고 RMPFlow의 이전 EE target을 폐기한다."""
        arm_joint_names = (
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        )
        arm_dof_indices = [
            robot.get_dof_index(name) for name in arm_joint_names
        ]
        current_positions = np.asarray(
            robot.get_joint_positions()[arm_dof_indices],
            dtype=float,
        )
        arm_controller.reset()
        sync_rmpflow_base_pose()
        robot.apply_action(
            ArticulationAction(
                joint_positions=current_positions,
                joint_velocities=np.zeros_like(current_positions),
                joint_indices=arm_dof_indices,
            )
        )

    sync_rmpflow_base_pose()

    print("\nIsaac Sim 준비 완료 - BehaviorNode 명령을 대기합니다.")

    step_count = 0
    grasp_timer = 0
    nut_release_timer = 0
    was_playing = False
    playback_started_once = False
    phase = "IDLE"
    init_pose_only = False  # True면 INIT_POSE 완료 후 SCAN_APPROACH로 안 이어지고 바로 종료
    busbar_grasped = False  # 그리퍼-버스바 FixedJoint가 걸려있는 상태인지
    busbar_grasp_start_z = None
    busbar_lift_stable_steps = 0
    descend_target_z = None
    target_mid_pos = None
    target_fine_yaw_rad = 0.0
    fine_alignment_generation = 0
    fine_step_gate = FineAlignmentStepGate(
        position_tolerance_m=FINE_STEP_POSITION_TOL_M,
        orientation_tolerance_rad=FINE_STEP_ORIENTATION_TOL_RAD,
        motion_tolerance_m=FINE_STEP_MOTION_TOL_M,
        orientation_motion_tolerance_rad=(
            FINE_STEP_ORIENTATION_MOTION_TOL_RAD
        ),
        position_response_noise_margin_m=(
            FINE_STEP_POSITION_RESPONSE_NOISE_MARGIN_M
        ),
        orientation_response_noise_margin_rad=(
            FINE_STEP_ORIENTATION_RESPONSE_NOISE_MARGIN_RAD
        ),
        settle_steps=FINE_STEP_SETTLE_STEPS,
    )
    fine_controller_lead = BoundedPlanarCommandLead(
        growth_step_m=FINE_CONTROLLER_LEAD_GROWTH_M,
        max_lead_m=FINE_CONTROLLER_MAX_LEAD_M,
    )
    fine_step_previous_ee_pos = None
    fine_step_previous_ee_orientation = None
    fine_step_start_ee_pos = None
    fine_step_command_delta_xy = None
    fine_step_start_orientation_error_rad = None
    scan_hold_quat = None  # INIT_POSE 완료 시점의 실제 EE 자세 (SCAN_APPROACH가 그대로 유지)

    # ── 너트 조립(Nut Assembly) 상태 변수 ──
    nut_index      = 0        # 전체 장면의 물리 너트 번호(1~6)
    nut_slot       = 0        # 현재 station 안의 bolt/nut 슬롯(1 또는 2)
    NUT_SCAN_POS   = None     # 너트 스캔 위치 (SCAN_NUT1/2마다 해당 너트 오프셋으로 새로 계산)
    NUT_SCAN_LIFT_POS = None  # 방향 정렬용 중간 경유 위치 (현재 XY, 스캔 고도)
    nut_pose_settle_count = 0  # 실제 위치+quaternion 연속 안정화 표본
    BUSBAR_SCAN_LIFT_POS = None  # 버스바 스캔용 방향 정렬 중간 경유 위치 (현재 XY, 스캔 고도)
    nut_pick_pos   = None     # 현재 너트의 물리 파지 좌표
    nut_approach_pos = None   # 현재 너트 파지 상공 접근 좌표
    nut_peg_clear_pos = None  # 파지 직후 peg에서 수직으로 빠져나올 중간 목표
    nut_peg_clear_start_pos = None  # 실제 80mm 이탈을 재는 파지 종료 EE 위치
    nut_peg_clear_start_nut_z = None  # 마찰 파지된 실제 너트의 이탈 기준 높이
    nut_peg_clear_xy_lead = np.zeros(2, dtype=float)
    nut_peg_clear_z_lead = 0.0  # RMPFlow 정상상태 Z 오차를 보상하는 command-only lead
    nut_peg_clear_hold = 0    # 중간 목표 도착 후 안정화 카운터
    nut_pick_active_array_index = None  # PICK 성공 전 cancel/conflict 시 tray glue 복구 대상
    nut_ee_reference_offset = None  # release/settled transport의 nut_position - ee_position
    nut_ee_previous_offset = None  # release settle 중 tick 간 상대 위치 변화 측정
    nut_lift_command_z = None  # peg-clear부터 안전 상공까지 이어지는 bounded Z command
    bolt_target_pos   = None  # 체결 목표 좌표
    bolt_travel_pos   = None  # 볼트 XY의 안전 상공(NUT_APPROACH_Z) 좌표
    bolt_approach_pos = None  # 볼트 바로 위(BOLT_APPROACH_Z) 좌표
    bolt_touch_pos    = None  # 착좌(Screwing 시작) 목표 좌표
    nut_xy_lead       = np.zeros(2, dtype=float)  # RMPFlow 정상상태 XY 오차 보정

    screw_sub          = "rotate"
    screw_pass_idx      = 0
    screw_pass_theta    = 0.0
    screw_seat_ee_pos   = None
    screw_start_quat    = None
    screw_pass_end_pos  = None
    screw_release_step  = 0
    screw_regrasp_step  = 0
    screw_unwind_deg    = 0.0
    prev_ee_z           = 0.0
    stuck_counter       = 0

    # ── AMR 이동 상태 (amr_node <-> /amr/goal_pose, /amr/sim_pose) ──
    amr_moving = False
    amr_target_xy_theta = None
    amr_goal_stamp = ""
    amr_drive_publish_step = 0
    current_station = 3  # /amr/goal_pose로 자동 판별, 기본값은 station 3
    amr_goal_busbar_station = None
    last_arrived_busbar_station = None
    wheels_locked = True  # wheel target 0으로 정지한 상태

    def restore_active_nut_to_tray_glue(reason):
        """Abort an unfinished PICK and restore its original AMR-local slot."""
        nonlocal nut_pick_active_array_index
        nonlocal nut_ee_reference_offset
        nonlocal nut_ee_previous_offset
        nonlocal nut_lift_command_z
        nonlocal nut_release_timer

        array_index = nut_pick_active_array_index
        if array_index is None:
            return False

        nut_xf = nut_xforms_all[array_index]
        local_offset = nut_local_offsets[array_index]
        nut_root = nut_roots_all[array_index]
        if nut_xf is not None and local_offset is not None:
            # Physics must be disabled before moving the prim back to its
            # original tray-relative pose.  The ordinary glue loop owns it
            # again from the next frame onward.
            disable_physics_recursively(stage, nut_root)
            remove_nut_amr_joint(stage, nut_root)
            glue_position, glue_orientation = compose_world_pose(
                *robot.get_world_pose(),
                *local_offset,
            )
            nut_xf.set_world_pose(
                position=glue_position,
                orientation=glue_orientation,
            )
            nut_released[array_index] = False
            print(
                f"[NUT PICK ROLLBACK] 너트 {array_index + 1}번을 "
                f"원래 tray glue 위치로 복구 ({reason})"
            )
        else:
            print(
                f"[ERROR] 너트 {array_index + 1}번 tray glue 복구에 "
                "필요한 Xform/offset이 없습니다"
            )

        robot.gripper.apply_action(
            ArticulationAction(joint_positions=GRIPPER_OPEN)
        )
        nut_pick_active_array_index = None
        nut_ee_reference_offset = None
        nut_ee_previous_offset = None
        nut_lift_command_z = None
        nut_release_timer = 0
        return True

    def publish_status(status_str: str):
        msg = String()
        msg.data = status_str
        isaac_node.pub_status.publish(msg)

    def publish_progress(phase_name: str, pct: float):
        p_msg = String()
        p_msg.data = phase_name
        isaac_node.pub_phase.publish(p_msg)

        prog_msg = Float32()
        prog_msg.data = pct
        isaac_node.pub_progress.publish(prog_msg)

    while simulation_app.is_running():
        world.step(render=True)
        rclpy.spin_once(isaac_node, timeout_sec=0.0)
        playing = world.is_playing()

        # 1. Play / Stop 상태 보정
        if playing and not was_playing:
            is_playback_restart = playback_started_once
            interrupted_phase = (
                phase if phase not in {"IDLE", "DONE"} else None
            )
            restart_amr_goal_stamp = (
                amr_goal_stamp
                if is_playback_restart and amr_moving and amr_goal_stamp
                else ""
            )
            restart_amr_pose = (
                measure_amr_pose(robot)
                if restart_amr_goal_stamp
                else None
            )
            if restart_amr_goal_stamp:
                _apply_amr_wheel_speeds(
                    robot,
                    wheel_indices,
                    0.0,
                    0.0,
                )
            world.reset()
            prepare_amr_physics(stage)
            wheel_indices = _configure_runtime_wheel_drive(robot)
            lock_amr_base(stage, NOVA_CARTER_ROOT, robot=robot)
            drive_controller.clear_goal()
            for asset in busbar_assets.values():
                enable_physics_recursively(stage, asset["root_path"])
                detach_busbar_from_gripper(stage, asset["mesh_path"])
                asset["xform"].set_world_pose(
                    position=asset["initial_position"],
                    orientation=asset["initial_orientation"],
                )
            busbar_root_path = None
            busbar_mesh_path = None
            busbar_xform = None
            busbar_grasped = False
            busbar_grasp_start_z = None
            busbar_lift_stable_steps = 0
            busbar_camera_xform.set_world_pose(
                position=busbar_camera_init_pos,
                orientation=busbar_camera_init_quat,
            )
            bolt_camera_xform.set_world_pose(
                position=bolt_camera_init_pos,
                orientation=bolt_camera_init_quat,
            )
            if nut1_xform and init_nut1_pos is not None:
                nut1_xform.set_world_pose(position=init_nut1_pos, orientation=init_nut1_quat)
            if nut2_xform and init_nut2_pos is not None:
                nut2_xform.set_world_pose(position=init_nut2_pos, orientation=init_nut2_quat)
            for extra_xf, (extra_pos, extra_quat), extra_root, extra_poly in zip(
                extra_nut_xforms, extra_nut_inits, EXTRA_NUT_ROOT_PATHS, EXTRA_NUT_POLYSHAPE_PATHS
            ):
                if extra_xf and extra_pos is not None:
                    extra_xf.set_world_pose(position=extra_pos, orientation=extra_quat)

            # battery4_main과 동일하게 재생 재시작 시 모든 너트를 kinematic glue 상태로
            # 되돌리고 현재 AMR 포즈 기준 로컬 오프셋을 다시 계산한다.
            amr_glue_pos, amr_glue_quat = robot.get_world_pose()
            nut_released = [False] * len(nut_xforms_all)
            for i, (nut_xf, nut_root) in enumerate(zip(nut_xforms_all, nut_roots_all)):
                if nut_xf is None:
                    nut_local_offsets[i] = None
                    continue
                remove_nut_amr_joint(stage, nut_root)
                disable_physics_recursively(stage, nut_root)
                nut_pos, nut_quat = nut_xf.get_world_pose()
                nut_local_offsets[i] = compute_local_offset(
                    amr_glue_pos, amr_glue_quat, nut_pos, nut_quat
                )

            step_count = 0
            grasp_timer = 0
            nut_release_timer = 0
            target_fine_yaw_rad = 0.0
            fine_alignment_generation += 1
            fine_step_gate.reset()
            fine_controller_lead.reset()
            fine_step_previous_ee_pos = None
            fine_step_previous_ee_orientation = None
            fine_step_start_ee_pos = None
            fine_step_command_delta_xy = None
            fine_step_start_orientation_error_rad = None

            nut_index = 0
            nut_slot = 0
            NUT_SCAN_POS = None
            nut_pose_settle_count = 0
            nut_pick_pos = None
            nut_approach_pos = None
            nut_peg_clear_pos = None
            nut_peg_clear_start_pos = None
            nut_peg_clear_start_nut_z = None
            nut_peg_clear_xy_lead = np.zeros(2, dtype=float)
            nut_peg_clear_z_lead = 0.0
            nut_peg_clear_hold = 0
            nut_pick_active_array_index = None
            nut_ee_reference_offset = None
            nut_ee_previous_offset = None
            nut_lift_command_z = None
            bolt_target_pos = None
            bolt_travel_pos = None
            bolt_approach_pos = None
            bolt_touch_pos = None
            nut_xy_lead = np.zeros(2, dtype=float)
            screw_sub = "rotate"
            screw_pass_idx = 0
            screw_pass_theta = 0.0
            stuck_counter = 0

            amr_moving = False
            amr_target_xy_theta = None
            amr_goal_stamp = ""
            amr_drive_publish_step = 0
            amr_goal_busbar_station = None
            last_arrived_busbar_station = None
            wheels_locked = True

            # world.reset() invalidates RMPFlow's cached articulation/base
            # state. 최초 play는 policy만 다시 anchor하고, 실제 playback
            # restart는 저수준 관절 target도 현재 pose hold로 덮어쓴다.
            if is_playback_restart:
                hold_current_arm_pose()
                phase = "IDLE"
                step_count = 0
                isaac_node.requested_task = None
                isaac_node.latest_target_pose = None
                isaac_node.latest_fine_alignment_delta = None
                isaac_node.alignment_success = False
                isaac_node.expected_fine_alignment_generation = None
                isaac_node.pub_wrist_perception_reset.publish(Empty())
                isaac_node.pub_busbar_perception_reset.publish(Empty())
                isaac_node.pub_bolt_perception_reset.publish(Empty())
                reset_cmd = String()
                reset_cmd.data = "RESET_BOLT_DETECTION"
                isaac_node.pub_errorfix_command.publish(reset_cmd)
                if restart_amr_goal_stamp:
                    isaac_node.publish_amr_drive_state(
                        "CANCELED",
                        "playback 재시작으로 실제 wheel 주행을 중단했습니다",
                        restart_amr_goal_stamp,
                        pose=restart_amr_pose,
                    )
                    print(
                        "\n[AMR PHYSICS] playback 재시작으로 goal "
                        f"{restart_amr_goal_stamp} 취소 -> wheel brake"
                    )
                if interrupted_phase is not None:
                    publish_status(
                        "FAILURE:PLAYBACK_RESTART_DURING_"
                        f"{interrupted_phase}"
                    )
                    print(
                        "\n[ARM] playback 재시작으로 "
                        f"{interrupted_phase} phase를 중단하고 팔을 현재 "
                        "자세로 hold합니다"
                    )
                else:
                    # phase가 IDLE이어도 ArmNode가 perception 응답 또는
                    # target/task 토픽 순서를 기다리는 중일 수 있다.
                    publish_status("FAILURE:PLAYBACK_RESTART")
                    print(
                        "\n[ARM] playback 재시작 -> stale target/task/cache "
                        "폐기 및 현재 자세 hold"
                    )
            else:
                arm_controller.reset()
                sync_rmpflow_base_pose()
            playback_started_once = True

        # battery4_main의 너트 AMR glue 추종. PICK된 너트는 nut_released=True가 되어
        # 여기서 제외되고 이후부터 정상 다이나믹 바디로 움직인다.
        if playing:
            amr_glue_pos, amr_glue_quat = robot.get_world_pose()
            for i, nut_xf in enumerate(nut_xforms_all):
                if nut_xf is None or nut_released[i] or nut_local_offsets[i] is None:
                    continue
                glued_pos, glued_quat = compose_world_pose(
                    amr_glue_pos, amr_glue_quat, *nut_local_offsets[i]
                )
                nut_xf.set_world_pose(position=glued_pos, orientation=glued_quat)

        # 1.5. AMR 실제 물리 이동 처리
        #      목표 pose를 차체에 덮어쓰지 않고 좌우 wheel velocity로만 움직인다.
        if playing:
            if isaac_node.amr_cancel_requested:
                isaac_node.amr_cancel_requested = False
                _apply_amr_wheel_speeds(
                    robot, wheel_indices, 0.0, 0.0
                )
                if amr_moving:
                    pose = measure_amr_pose(robot)
                    isaac_node.publish_amr_drive_state(
                        "CANCELED",
                        "취소 요청을 받아 실제 wheel을 정지했습니다",
                        amr_goal_stamp,
                        pose=pose,
                    )
                    print("\n[AMR PHYSICS] 이동 취소 수신 -> wheel 정지")
                drive_controller.clear_goal()
                amr_moving = False
                amr_target_xy_theta = None
                amr_goal_stamp = ""
                amr_goal_busbar_station = None
                last_arrived_busbar_station = None
                if not wheels_locked:
                    lock_amr_base(stage, NOVA_CARTER_ROOT, robot=robot)
                    wheels_locked = True

            # 팔 FSM과 AMR wheel 주행은 동시에 실행하지 않는다. 새 goal 자체는
            # 즉시 소비하되 wheel을 0/brake 상태로 유지하고 명시적으로 거부한다.
            if (
                isaac_node.amr_goal_pose is not None
                and phase not in {"IDLE", "DONE"}
            ):
                rejected_goal = isaac_node.amr_goal_pose
                isaac_node.amr_goal_pose = None
                rejected_stamp = goal_stamp_key(rejected_goal)
                _apply_amr_wheel_speeds(
                    robot, wheel_indices, 0.0, 0.0
                )
                pose = measure_amr_pose(robot)
                if amr_moving and amr_goal_stamp:
                    isaac_node.publish_amr_drive_state(
                        "CANCELED",
                        "팔 작업 중 새 목표가 도착해 이전 wheel 주행을 중단했습니다",
                        amr_goal_stamp,
                        pose=pose,
                    )
                drive_controller.clear_goal()
                amr_moving = False
                amr_target_xy_theta = None
                amr_goal_stamp = ""
                amr_goal_busbar_station = None
                last_arrived_busbar_station = None
                lock_amr_base(stage, NOVA_CARTER_ROOT, robot=robot)
                wheels_locked = True
                sync_rmpflow_base_pose()
                isaac_node.publish_amr_drive_state(
                    "FAILED",
                    f"팔 phase({phase}) 실행 중 AMR 목표를 거부했습니다",
                    rejected_stamp,
                    pose=pose,
                )
                print(
                    f"\n[AMR PHYSICS] 팔 phase '{phase}' 실행 중 새 목표 "
                    f"{rejected_stamp} 거부 -> wheel brake 유지"
                )

            if isaac_node.amr_goal_pose is not None:
                goal_msg = isaac_node.amr_goal_pose
                isaac_node.amr_goal_pose = None
                g_pos = goal_msg.pose.position
                g_ori = goal_msg.pose.orientation
                new_goal_stamp = goal_stamp_key(goal_msg)

                # 새 목표를 해석하기 전에 이전 command를 즉시 끊는다. 유효하지 않은
                # 목표여도 기존 주행이 계속되는 일이 없어야 한다.
                _apply_amr_wheel_speeds(robot, wheel_indices, 0.0, 0.0)
                last_arrived_busbar_station = None
                amr_goal_busbar_station = None
                if amr_moving and amr_goal_stamp:
                    pose = measure_amr_pose(robot)
                    isaac_node.publish_amr_drive_state(
                        "CANCELED",
                        "새 목표가 도착해 이전 wheel 주행을 중단했습니다",
                        amr_goal_stamp,
                        pose=pose,
                    )
                drive_controller.clear_goal()
                amr_moving = False
                amr_target_xy_theta = None
                amr_goal_stamp = ""

                goal_values = (
                    g_pos.x,
                    g_pos.y,
                    g_pos.z,
                    g_ori.w,
                    g_ori.x,
                    g_ori.y,
                    g_ori.z,
                )
                try:
                    if not all(
                        math.isfinite(float(value))
                        for value in goal_values
                    ):
                        raise ValueError("pose에 NaN 또는 inf가 있습니다")
                    g_theta = quat_wxyz_to_yaw(
                        [g_ori.w, g_ori.x, g_ori.y, g_ori.z]
                    )
                except (TypeError, ValueError) as exc:
                    pose = measure_amr_pose(robot)
                    isaac_node.publish_amr_drive_state(
                        "FAILED",
                        f"유효하지 않은 AMR 목표를 거부했습니다: {exc}",
                        new_goal_stamp,
                        pose=pose,
                    )
                    if not wheels_locked:
                        lock_amr_base(
                            stage,
                            NOVA_CARTER_ROOT,
                            robot=robot,
                        )
                        wheels_locked = True
                    print(f"[AMR PHYSICS] 목표 거부: {exc}")
                else:
                    amr_target_xy_theta = (
                        float(g_pos.x),
                        float(g_pos.y),
                        float(g_theta),
                    )
                    amr_goal_stamp = new_goal_stamp
                    amr_moving = True
                    amr_drive_publish_step = 0
                    pose = measure_amr_pose(robot)
                    drive_controller.set_goal(
                        Goal2D(
                            x=float(g_pos.x),
                            y=float(g_pos.y),
                            yaw=float(g_theta),
                            yaw_required=True,
                            allow_reverse=True,
                            align_yaw_before_drive=False,
                        ),
                        current_pose=pose,
                    )
                    if wheels_locked:
                        unlock_amr_base(
                            stage,
                            NOVA_CARTER_ROOT,
                            robot=robot,
                        )
                        wheels_locked = False
                    resolved_station = resolve_station_from_amr_xy(
                        g_pos.x,
                        g_pos.y,
                    )
                    if resolved_station is not None:
                        current_station = resolved_station
                        print(
                            "[STATION] 목표좌표로 스테이션 "
                            f"{current_station}번 인식"
                        )
                    else:
                        print(
                            f"[WARN] 목표좌표({g_pos.x:.4f},"
                            f"{g_pos.y:.4f})가 알려진 스테이션과 "
                            f"안 맞음 -> current_station={current_station} 유지"
                        )
                    amr_goal_busbar_station = (
                        resolve_busbar_stop_from_amr_xy(
                            g_pos.x,
                            g_pos.y,
                        )
                    )
                    isaac_node.publish_amr_drive_state(
                        "ACTIVE",
                        "목표를 수락하고 실제 wheel 주행을 시작합니다",
                        amr_goal_stamp,
                        pose=pose,
                    )
                    print(
                        f"\n>>> [AMR PHYSICS] 이동 목표 수신 "
                        f"(X={g_pos.x:.4f}, Y={g_pos.y:.4f}, "
                        f"Theta={g_theta:.4f}, stamp={amr_goal_stamp}) "
                        f"-> {drive_controller.travel_direction} 방향을 "
                        "이번 goal 동안 고정"
                    )

            pose = measure_amr_pose(robot)

            if amr_moving and amr_target_xy_theta is not None:
                command = drive_controller.compute(pose, PHYSICS_DT)
                left_radps, right_radps = drive_controller.to_wheel_speeds(
                    command.linear,
                    command.angular,
                    wheel_sign=AMR_WHEEL_SIGN,
                    steer_sign=AMR_STEER_SIGN,
                )
                _apply_amr_wheel_speeds(
                    robot,
                    wheel_indices,
                    left_radps,
                    right_radps,
                )
                amr_drive_publish_step += 1

                if command.arrived:
                    _apply_amr_wheel_speeds(
                        robot, wheel_indices, 0.0, 0.0
                    )
                    amr_moving = False
                    amr_target_xy_theta = None
                    drive_controller.clear_goal()
                    isaac_node.publish_amr_drive_state(
                        "ARRIVED",
                        "실제 chassis pose가 위치/yaw 허용오차에 도달하고 정지했습니다",
                        amr_goal_stamp,
                        pose=pose,
                        command=command,
                    )
                    print(
                        "\n[AMR PHYSICS] 실제 chassis 도착·정지 "
                        f"(X={pose.x:.4f}, Y={pose.y:.4f}, "
                        f"Yaw={pose.yaw:.4f})"
                    )
                    lock_amr_base(stage, NOVA_CARTER_ROOT, robot=robot)
                    wheels_locked = True
                    sync_rmpflow_base_pose()
                    last_arrived_busbar_station = (
                        amr_goal_busbar_station
                    )
                    amr_goal_busbar_station = None
                    amr_goal_stamp = ""
                elif amr_drive_publish_step % 60 == 0:
                    isaac_node.publish_amr_drive_state(
                        "DRIVING",
                        command.phase,
                        amr_goal_stamp,
                        pose=pose,
                        command=command,
                        wheel_speeds=(left_radps, right_radps),
                    )
                    print(
                        "[AMR PHYSICS] "
                        f"phase={command.phase}, "
                        f"pose=({pose.x:.3f},{pose.y:.3f},"
                        f"{math.degrees(pose.yaw):.1f}deg), "
                        f"dist={command.distance_error:.3f}m, "
                        f"yaw_err={math.degrees(command.yaw_error):+.1f}deg, "
                        f"wheel=({left_radps:+.2f},"
                        f"{right_radps:+.2f})rad/s"
                    )

            sim_pose_msg = Pose2D()
            sim_pose_msg.x = float(pose.x)
            sim_pose_msg.y = float(pose.y)
            sim_pose_msg.theta = float(pose.yaw)
            isaac_node.pub_amr_sim_pose.publish(sim_pose_msg)

            # 버스바를 실제 물리(마찰) 그립으로 물고 있는 동안은 phase="IDLE" 구간
            # (예: PICK_BUSBAR 완료 후 AMR이 배터리 스테이션으로 이동하는 중)에도
            # 계속 CLOSE 명령을 넣어줘야 한다 - 관절 액션이 안 들어가면 드라이브가
            # 느슨해져 그 사이 손아귀 힘이 빠질 수 있다. 위치는 더 이상 스크립트로
            # 덮어쓰지 않는다 - 그리퍼(팔+AMR과 같이 강체로 움직이는 관절 체인)가
            # 실제로 물고 있는 물리 그립이 알아서 따라간다.
            if busbar_grasped:
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))

        # 2. BehaviorNode / ArmNode 명령 분기 처리
        if playing and isaac_node.requested_task:
            task = isaac_node.requested_task
            isaac_node.requested_task = None

            # 명시적 cancel은 wheel interlock보다 먼저 처리해야 한다. 그렇지 않으면
            # 주행 중 task 거부 경로가 cancel 문자열을 지워 내부 WAIT phase가 남는다.
            if task == "CANCEL_ARM_TASK":
                hold_current_arm_pose()
                restore_active_nut_to_tray_glue("explicit cancel")
                phase = "IDLE"
                step_count = 0
                nut_peg_clear_start_pos = None
                nut_peg_clear_start_nut_z = None
                nut_peg_clear_xy_lead = np.zeros(2, dtype=float)
                nut_peg_clear_z_lead = 0.0
                nut_peg_clear_hold = 0
                nut_pose_settle_count = 0
                target_fine_yaw_rad = 0.0
                fine_step_gate.reset()
                fine_controller_lead.reset()
                fine_step_previous_ee_pos = None
                fine_step_previous_ee_orientation = None
                fine_step_start_ee_pos = None
                fine_step_command_delta_xy = None
                fine_step_start_orientation_error_rad = None
                isaac_node.latest_target_pose = None
                isaac_node.latest_fine_alignment_delta = None
                isaac_node.alignment_success = False
                isaac_node.expected_fine_alignment_generation = None
                reset_cmd = String()
                reset_cmd.data = "RESET_BOLT_DETECTION"
                isaac_node.pub_errorfix_command.publish(reset_cmd)
                print(
                    "\n[ARM] 명시적 Action cancel 수신 -> 현재 관절 hold 및 "
                    "진행 중 phase 중지"
                )
                task = ""

            # 안전장치: 팔 Task는 AMR이 도착해 바퀴가 잠긴 상태에서만 수행되어야 한다.
            if task and not wheels_locked:
                _apply_amr_wheel_speeds(
                    robot, wheel_indices, 0.0, 0.0
                )
                pose = measure_amr_pose(robot)
                print(
                    f"\n[ERROR] [{task}] 실제 wheel 주행 중 팔 명령 수신 "
                    "-> 주행 정지 후 팔 작업 거부"
                )
                isaac_node.publish_amr_drive_state(
                    "FAILED",
                    f"wheel 주행 중 팔 명령({task})이 도착했습니다",
                    amr_goal_stamp,
                    pose=pose,
                )
                drive_controller.clear_goal()
                lock_amr_base(stage, NOVA_CARTER_ROOT, robot=robot)
                wheels_locked = True
                amr_moving = False
                amr_target_xy_theta = None
                amr_goal_stamp = ""
                amr_goal_busbar_station = None
                last_arrived_busbar_station = None
                sync_rmpflow_base_pose()
                publish_status("FAILURE:AMR_STILL_DRIVING")
                if (
                    task == "CONTINUE_BUSBAR_WRIST_SCAN"
                    and phase == "WAIT_BUSBAR_CAMERA_LATCH"
                ):
                    hold_current_arm_pose()
                    phase = "IDLE"
                    step_count = 0
                task = ""

            # ArmNode는 ExecuteArmTask를 직렬화하지만 /task_command 자체는
            # 토픽이므로 중복 publisher나 지연 callback이 active phase를 덮어쓸
            # 수 있다. CONTINUE handshake만 정확한 WAIT phase에서 허용하고,
            # 그 외 경합은 두 task 모두 안전하게 중단해 이전 저수준 관절 target이
            # 계속 실행되지 않도록 현재 관절을 hold한다.
            continue_handshake = (
                task == "CONTINUE_BUSBAR_WRIST_SCAN"
                and phase == "WAIT_BUSBAR_CAMERA_LATCH"
            )
            if (
                task
                and phase not in {"IDLE", "DONE"}
                and not continue_handshake
            ):
                conflicting_phase = phase
                hold_current_arm_pose()
                restore_active_nut_to_tray_glue(
                    f"task conflict: {task}"
                )
                phase = "IDLE"
                step_count = 0
                nut_pose_settle_count = 0
                isaac_node.latest_target_pose = None
                isaac_node.latest_fine_alignment_delta = None
                isaac_node.alignment_success = False
                isaac_node.expected_fine_alignment_generation = None
                target_fine_yaw_rad = 0.0
                fine_step_gate.reset()
                fine_controller_lead.reset()
                fine_step_previous_ee_pos = None
                fine_step_previous_ee_orientation = None
                fine_step_start_ee_pos = None
                fine_step_command_delta_xy = None
                fine_step_start_orientation_error_rad = None
                reset_cmd = String()
                reset_cmd.data = "RESET_BOLT_DETECTION"
                isaac_node.pub_errorfix_command.publish(reset_cmd)
                publish_status("FAILURE:ARM_TASK_CONFLICT")
                print(
                    f"\n[ARM] active phase '{conflicting_phase}' 중 새 task "
                    f"'{task}' 수신 -> 현재 관절 hold 및 task 경합 거부"
                )
                task = ""

            if task == "SCAN_BATTERY":
                # 최신 방식대로 현재 station의 bolt_2 XY로 카메라만 평행
                # 이동하고 기존 Z·자세는 유지한다. 공급대 전체 중점은 인접
                # 배터리팩까지 시야에 넣어 잘못된 bolt쌍을 고를 수 있다.
                # reset 뒤의 3-frame barrier는 perception 인스턴스가 맡는다.
                bolt_paths = STATION_BOLT_PRIM_PATHS.get(current_station)
                if bolt_paths is None or any(
                    not stage.GetPrimAtPath(path).IsValid()
                    for path in bolt_paths
                ):
                    print(
                        f"\n[ERROR] [{task}] 스테이션 {current_station} "
                        f"bolt prim 누락: {bolt_paths}"
                    )
                    publish_status("FAILURE:BOLT_CAMERA_TARGET_NOT_FOUND")
                else:
                    bolt_camera_position, bolt_camera_orientation = (
                        bolt_camera_pose_for_target(
                            stage,
                            bolt_paths[1],
                            bolt_camera_init_pos,
                            bolt_camera_init_quat,
                        )
                    )
                    bolt_camera_xform.set_world_pose(
                        position=bolt_camera_position,
                        orientation=bolt_camera_orientation,
                    )

                    isaac_node.pub_bolt_perception_reset.publish(Empty())
                    reset_cmd = String()
                    reset_cmd.data = "RESET_BOLT_DETECTION"
                    isaac_node.pub_errorfix_command.publish(reset_cmd)
                    phase = "INIT_POSE"
                    init_pose_only = False
                    step_count = 0
                    print(
                        f"\n>>> [{task}] station {current_station} bolt_2 XY로 "
                        "고정 볼트캠 이동/reset 완료 -> 초기 관절 정렬 및 "
                        f"wrist scan 시작 ({bolt_camera_position[0]:.4f}, "
                        f"{bolt_camera_position[1]:.4f})"
                    )

            elif task == "RETURN_HOME":
                # 너트 작업 전처럼 "관절 각도만 초기 자세로 되돌리고 싶을 때" 쓴다.
                # SCAN_BATTERY와 달리 완료 후 SCAN_APPROACH(배터리 스캔 고도 상승)로
                # 이어지지 않고 바로 끝난다.
                phase = "INIT_POSE"
                init_pose_only = True
                step_count = 0
                print(f"\n>>> [{task}] 초기 관절 자세 복귀 시작")

            elif task == "SCAN_BUSBAR":
                # 이전 task나 error_fix가 남긴 PoseStamped를 다음 station 버스바
                # target으로 오인하지 않도록 스캔 세대 시작점에서 비운다.
                isaac_node.latest_target_pose = None
                (
                    busbar_amr_pose,
                    busbar_amr_position_error,
                    busbar_amr_yaw_error,
                ) = busbar_station_pose_error(robot, current_station)
                # 공급대 전체를 한 시야에 넣으면 최고 confidence 후보가 다른
                # station으로 바뀔 수 있다. 단일 논리 카메라를 현재 station의 실제
                # busbar prim 위 고정 시점으로 전환해 후보를 하나로 제한한다.
                asset = busbar_assets.get(current_station)
                if (
                    last_arrived_busbar_station != current_station
                    or busbar_amr_position_error > AMR_POS_TOL
                    or busbar_amr_yaw_error > AMR_YAW_TOL
                ):
                    print(
                        f"\n[ERROR] [{task}] AMR이 station "
                        f"{current_station} 버스바 정차점에 없습니다 | "
                        f"pose=({busbar_amr_pose.x:.4f},"
                        f"{busbar_amr_pose.y:.4f},"
                        f"{busbar_amr_pose.yaw:.4f}) | "
                        f"position error="
                        f"{busbar_amr_position_error * 1000:.1f}mm, "
                        f"yaw error="
                        f"{math.degrees(busbar_amr_yaw_error):.1f}deg, "
                        "last arrived busbar station="
                        f"{last_arrived_busbar_station}"
                    )
                    publish_status(
                        "FAILURE:AMR_NOT_AT_BUSBAR_STATION"
                    )
                    hold_current_arm_pose()
                    phase = "IDLE"
                    step_count = 0
                elif asset is None:
                    print(
                        f"\n[ERROR] [{task}] 스테이션 {current_station} "
                        "버스바 prim 매핑이 없습니다."
                    )
                    publish_status("FAILURE:BUSBAR_CAMERA_TARGET_NOT_FOUND")
                else:
                    arm_controller.reset()
                    sync_rmpflow_base_pose()
                    (
                        busbar_camera_position,
                        busbar_camera_orientation,
                    ) = busbar_camera_pose_for_asset(asset["xform"])
                    busbar_camera_xform.set_world_pose(
                        position=busbar_camera_position,
                        orientation=busbar_camera_orientation,
                    )
                    isaac_node.pub_busbar_perception_reset.publish(Empty())

                    # b83 고정캠은 busbar 바로 위를 내려다보므로 wrist scan 자세로
                    # 먼저 이동하면 팔이 시야를 가려 v3 confidence가 0.6 아래로
                    # 떨어진다. 카메라 이동/reset 뒤 ArmNode가 fixed pose를 latch할
                    # 때까지 여기서 멈추고, 내부 continue 명령을 받은 뒤에만 wrist
                    # scan 자세로 이동한다. 외부 ExecuteArmTask API는 바뀌지 않는다.
                    phase = "WAIT_BUSBAR_CAMERA_LATCH"
                    step_count = 0
                    publish_progress("BUSBAR_CAMERA_READY", 20.0)
                    print(
                        f"\n>>> [{task}] station {current_station} b83 상대 "
                        "6DoF로 고정 버스바캠 이동/reset 완료 -> fixed pose "
                        f"latch 대기 (camera=({busbar_camera_position[0]:.4f}, "
                        f"{busbar_camera_position[1]:.4f}, "
                        f"{busbar_camera_position[2]:.4f}))"
                    )

            elif task == "CONTINUE_BUSBAR_WRIST_SCAN":
                if phase != "WAIT_BUSBAR_CAMERA_LATCH":
                    print(
                        f"\n[ERROR] [{task}] fixed pose latch 대기 상태가 "
                        f"아닙니다 (phase={phase})"
                    )
                    publish_status(
                        "FAILURE:INVALID_BUSBAR_SCAN_CONTINUE"
                    )
                    hold_current_arm_pose()
                    phase = "IDLE"
                    step_count = 0
                else:
                    arm_controller.reset()
                    sync_rmpflow_base_pose()
                    asset = busbar_assets.get(current_station)
                    if asset is None:
                        print(
                            f"\n[ERROR] [{task}] 스테이션 "
                            f"{current_station} 버스바 prim 매핑이 없습니다."
                        )
                        publish_status(
                            "FAILURE:BUSBAR_SCAN_TARGET_NOT_FOUND"
                        )
                        hold_current_arm_pose()
                        phase = "IDLE"
                        step_count = 0
                        continue

                    mesh_center, _ = asset["xform"].get_world_pose()
                    BUSBAR_SCAN_POS = busbar_wrist_scan_target(
                        mesh_center,
                        scan_z=BUSBAR_SCAN_Z,
                    )
                    cur_pos = world_xf(
                        stage,
                        f"{M0609_PATH}/{EE_LINK_NAME}",
                    ).ExtractTranslation()
                    BUSBAR_SCAN_LIFT_POS = np.array(
                        [cur_pos[0], cur_pos[1], BUSBAR_SCAN_Z],
                        dtype=float,
                    )
                    phase = "SCAN_BUSBAR_LIFT"
                    step_count = 0
                    print(
                        "\n>>> [SCAN_BUSBAR] fixed pose latch 확인 -> "
                        "선택 Mesh 기준 가림 회피 wrist scan 자세 이동 시작 "
                        f"(Target: X={BUSBAR_SCAN_POS[0]:.3f}, "
                        f"Y={BUSBAR_SCAN_POS[1]:.3f}, "
                        f"Z={BUSBAR_SCAN_POS[2]:.3f})"
                    )

            elif task == "PICK_BUSBAR":
                (
                    busbar_amr_pose,
                    busbar_amr_position_error,
                    busbar_amr_yaw_error,
                ) = busbar_station_pose_error(robot, current_station)
                if (
                    last_arrived_busbar_station != current_station
                    or busbar_amr_position_error > AMR_POS_TOL
                    or busbar_amr_yaw_error > AMR_YAW_TOL
                ):
                    print(
                        f"\n[ERROR] [{task}] AMR이 station "
                        f"{current_station} 버스바 정차점에 없습니다 | "
                        f"pose=({busbar_amr_pose.x:.4f},"
                        f"{busbar_amr_pose.y:.4f},"
                        f"{busbar_amr_pose.yaw:.4f}) | "
                        f"position error="
                        f"{busbar_amr_position_error * 1000:.1f}mm, "
                        f"yaw error="
                        f"{math.degrees(busbar_amr_yaw_error):.1f}deg, "
                        "last arrived busbar station="
                        f"{last_arrived_busbar_station}"
                    )
                    publish_status(
                        "FAILURE:AMR_NOT_AT_BUSBAR_STATION"
                    )
                    hold_current_arm_pose()
                    isaac_node.latest_target_pose = None
                    phase = "IDLE"
                    step_count = 0
                elif isaac_node.latest_target_pose is not None:
                    arm_controller.reset()
                    sync_rmpflow_base_pose()
                    target_pose = isaac_node.latest_target_pose
                    isaac_node.latest_target_pose = None
                    target = target_pose.pose.position
                    asset = busbar_assets.get(current_station)
                    if asset is None:
                        print(
                            f"\n[ERROR] [{task}] 스테이션 "
                            f"{current_station} 버스바 prim이 없습니다."
                        )
                        publish_status("FAILURE:BUSBAR_PRIM_NOT_FOUND")
                    else:
                        live_position, _ = asset["xform"].get_world_pose()
                        association_error = math.hypot(
                            float(target.x) - float(live_position[0]),
                            float(target.y) - float(live_position[1]),
                        )
                        if (
                            not math.isfinite(association_error)
                            or association_error
                            > BUSBAR_TARGET_ASSOCIATION_MAX_M
                        ):
                            print(
                                f"\n[ERROR] [{task}] 고정캠 target이 "
                                f"station {current_station} busbar와 "
                                f"{association_error:.3f}m 떨어져 있습니다 "
                                f"(허용 {BUSBAR_TARGET_ASSOCIATION_MAX_M:.3f}m)"
                            )
                            publish_status(
                                "FAILURE:BUSBAR_TARGET_STATION_MISMATCH"
                            )
                        else:
                            busbar_root_path = asset["root_path"]
                            busbar_mesh_path = asset["mesh_path"]
                            busbar_xform = asset["xform"]
                            update_target_positions(target_pose)
                            phase = "BUSBAR_APPROACH"
                            step_count = 0
                            print(
                                f"\n>>> [{task}] station "
                                f"{current_station} {busbar_mesh_path} 선택 "
                                f"(비전-prim 오차 "
                                f"{association_error * 1000:.1f}mm) -> "
                                f"상공 접근(Z={BUSBAR_APPROACH_Z:.3f}m)"
                            )
                else:
                    # /target_pose와 /task_command는 서로 다른 ROS 토픽이라 command가
                    # 먼저 도착할 수 있다. 성공·명시적 취소·종료까지 기다린다.
                    phase = "WAIT_BUSBAR_TARGET"
                    step_count = 0
                    print(
                        f"\n>>> [{task}] command가 target pose보다 먼저 도착 "
                        "-> 고정 버스바캠 pose 대기"
                    )

            elif task == "MOVE_BATTERY_CENTER":
                if isaac_node.latest_target_pose is not None:
                    pos = isaac_node.latest_target_pose.pose.position
                    BATTERY_CENTER_POS = np.array([pos.x, pos.y, BATTERY_CENTER_Z])
                    phase = "MOVE_BATTERY_CENTER_APPROACH"
                    step_count = 0
                    print(f"\n>>> [{task}] 배터리 볼트 중점 상공 이동 시작 (Target: X={BATTERY_CENTER_POS[0]:.3f}, Y={BATTERY_CENTER_POS[1]:.3f}, Z={BATTERY_CENTER_POS[2]:.3f})")
                else:
                    print(f"\n[ERROR] [{task}] 수신된 Target Pose가 없습니다.")
                    publish_status("FAILURE:NO_TARGET_POSE")

            elif task == "FINE_ALIGNMENT":
                print(f"\n>>> [{task}] 비전 정밀 오차 보정 준비")
                isaac_node.alignment_success = False
                isaac_node.expected_fine_alignment_generation = None
                bolt_paths = STATION_BOLT_PRIM_PATHS.get(current_station)
                if bolt_paths is None or any(
                    not stage.GetPrimAtPath(path).IsValid()
                    for path in bolt_paths
                ):
                    print(
                        f"[ERROR] [{task}] station {current_station} "
                        "오차보정 기준 bolt prim이 없습니다."
                    )
                    publish_status("FAILURE:ERRORFIX_BOLT_NOT_FOUND")
                else:
                    # error_fix는 중앙의 단일 기준 bolt를 추적하므로 FINE_ALIGNMENT
                    # 동안에는 station bolt_2 XY와 기존 Z·자세를 사용한다.
                    bolt_camera_position, bolt_camera_orientation = (
                        bolt_camera_pose_for_target(
                            stage,
                            bolt_paths[1],
                            bolt_camera_init_pos,
                            bolt_camera_init_quat,
                        )
                    )
                    bolt_camera_xform.set_world_pose(
                        position=bolt_camera_position,
                        orientation=bolt_camera_orientation,
                    )
                    isaac_node.pub_bolt_perception_reset.publish(Empty())

                    # RESET 뒤 detector 자체의 3-frame barrier가 옛 queued frame을
                    # 거부하므로 임의의 wall-clock settle 없이 바로 시작한다.
                    reset_cmd = String()
                    reset_cmd.data = "RESET_BOLT_DETECTION"
                    isaac_node.pub_errorfix_command.publish(reset_cmd)
                    start_cmd = String()
                    fine_alignment_generation += 1
                    start_cmd.data = (
                        "START_ERRORFIX_CORRECTION:"
                        f"{fine_alignment_generation}"
                    )
                    isaac_node.expected_fine_alignment_generation = (
                        fine_alignment_generation
                    )
                    # MOVE_BATTERY_CENTER에서 사용한 absolute /target_pose를
                    # fine-alignment delta로 재해석하지 않는다.
                    isaac_node.latest_target_pose = None
                    isaac_node.latest_fine_alignment_delta = None
                    isaac_node.pub_errorfix_command.publish(start_cmd)

                    print(
                        " -> station bolt_2 기준 고정캠 이동/reset 및 "
                        "START_ERRORFIX_CORRECTION 전송"
                    )

                    cur_ee = world_xf(
                        stage,
                        f"{M0609_PATH}/{EE_LINK_NAME}",
                    ).ExtractTranslation()
                    target_fine_pos = np.array([
                        cur_ee[0],
                        cur_ee[1],
                        BATTERY_CENTER_Z,
                    ])
                    target_fine_yaw_rad = 0.0
                    fine_step_gate.reset()
                    fine_controller_lead.reset()
                    fine_step_previous_ee_pos = None
                    fine_step_previous_ee_orientation = None
                    fine_step_start_ee_pos = None
                    fine_step_command_delta_xy = None
                    fine_step_start_orientation_error_rad = None
                    phase = "FINE_ALIGNMENT"
                    step_count = 0

            elif task == "ASSEMBLE_BUSBAR":
                cur_ee = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                if target_fine_pos is not None:
                    target_mid_pos = np.array([target_fine_pos[0], target_fine_pos[1], 0.0693])
                else:
                    target_mid_pos = np.array([cur_ee[0], cur_ee[1], 0.0693])

                descend_target_z = cur_ee[2]
                phase = "BUSBAR_DESCEND_TO_BOLT"
                step_count = 0
                print(f"\n>>> [{task}] 버스바 수직 하강 안착 시작 (Target Mid Pos: X={target_mid_pos[0]:.4f}, Y={target_mid_pos[1]:.4f})")

            elif task in ("SCAN_NUT1", "SCAN_NUT2"):
                nut_slot = 1 if task == "SCAN_NUT1" else 2
                nut_index = STATION_NUT_INDICES.get(current_station, (1, 2))[nut_slot - 1]
                if HOME_EE_POS is not None:
                    # NUT_OFFSET_FROM_HOME(하드코딩 오프셋) 대신 지금 씬의 너트 실제 world
                    # 위치를 직접 읽는다 - 너트를 옮기거나(PolyShape 물리 위치 정렬 등) 씬을
                    # 재배치할 때마다 오프셋 표를 다시 실측/갱신해야 하는 문제가 있었다.
                    nut_xf, nut_label = resolve_nut_assets(nut_index, nut1_xform, nut2_xform, extra_nut_xforms)
                    if nut_xf is None:
                        print(f"\n[ERROR] [{task}] 너트 {nut_index}번 Xform을 찾을 수 없습니다.")
                        publish_status("FAILURE:NUT_XFORM_NOT_FOUND")
                    else:
                        nut_live_pos, _ = nut_xf.get_world_pose()
                        NUT_SCAN_POS = np.array([
                            float(nut_live_pos[0]),
                            float(nut_live_pos[1]),
                            NUT_SCAN_Z,
                        ])
                        # XY와 orientation을 동시에 크게 바꾸면 RMPFlow가 팔꿈치/손목을 꼬아서
                        # 돌아가는 경로를 잡는 문제가 있었다(실측 확인됨) - 먼저 지금 있는
                        # XY에서 방향만 quat_nut로 맞추고 높이를 스캔 고도로 올린 뒤(NUT_SCAN_LIFT),
                        # 그 다음에 같은 높이/방향을 유지한 채 옆으로만 이동(NUT_SCAN_APPROACH)
                        # 시켜서 회전과 이동을 분리한다.
                        cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                        NUT_SCAN_LIFT_POS = np.array([cur_pos[0], cur_pos[1], NUT_SCAN_Z])
                        nut_pose_settle_count = 0
                        phase = "NUT_SCAN_LIFT"
                        step_count = 0
                        print(f"\n>>> [{task}] 1) 방향 정렬 및 스캔 고도 상승 시작 (Target: X={NUT_SCAN_LIFT_POS[0]:.3f}, Y={NUT_SCAN_LIFT_POS[1]:.3f}, Z={NUT_SCAN_LIFT_POS[2]:.3f})")
                else:
                    print(f"\n[ERROR] [{task}] 초기 위치(HOME_EE_POS)가 없습니다. 먼저 INIT_POSE가 수행되어야 합니다.")
                    publish_status("FAILURE:NO_HOME_POSE")

            elif task in ("PICK_NUT1", "PICK_NUT2"):
                nut_slot = 1 if task == "PICK_NUT1" else 2
                nut_index = STATION_NUT_INDICES.get(current_station, (1, 2))[nut_slot - 1]
                nut_peg_clear_start_pos = None
                nut_peg_clear_start_nut_z = None
                nut_peg_clear_xy_lead = np.zeros(2, dtype=float)
                nut_peg_clear_z_lead = 0.0
                nut_peg_clear_hold = 0
                if HOME_EE_POS is not None:
                    nut_array_index = nut_index - 1
                    # 접근·하강 중에는 원래 tray glue와 physics-disabled 상태를
                    # 그대로 유지한다. 손가락이 완전히 닫힌 뒤에만 NUT_GRASP가
                    # 대상 하나를 dynamics로 넘긴다.
                    nut_xf, nut_label = resolve_nut_assets(nut_index, nut1_xform, nut2_xform, extra_nut_xforms)
                    if nut_xf is None:
                        print(f"\n[ERROR] [{task}] 너트 {nut_index}번 Xform을 찾을 수 없습니다.")
                        publish_status("FAILURE:NUT_XFORM_NOT_FOUND")
                    elif nut_released[nut_array_index]:
                        print(
                            f"\n[ERROR] [{task}] 너트 {nut_index}번은 이미 "
                            "tray glue에서 해제된 상태입니다"
                        )
                        publish_status("FAILURE:NUT_ALREADY_RELEASED")
                    else:
                        nut_live_pos, _ = nut_xf.get_world_pose()
                        pick_x, pick_y = float(nut_live_pos[0]), float(nut_live_pos[1])
                        nut_pick_pos = np.array([pick_x, pick_y, NUT_PICK_Z])
                        nut_approach_pos = np.array([pick_x, pick_y, NUT_APPROACH_Z])
                        nut_pick_active_array_index = nut_array_index
                        nut_ee_reference_offset = None
                        nut_ee_previous_offset = None
                        nut_lift_command_z = None
                        nut_release_timer = 0
                        phase = "NUT_APPROACH"
                        step_count = 0
                        print(f"\n>>> [{task}] 너트 {nut_index}번 상공 접근 시작 (Target: X={nut_approach_pos[0]:.3f}, Y={nut_approach_pos[1]:.3f})")
                else:
                    print(f"\n[ERROR] [{task}] 초기 위치(HOME_EE_POS)가 없습니다. 먼저 INIT_POSE가 수행되어야 합니다.")
                    publish_status("FAILURE:NO_HOME_POSE")

            elif task in ("ASSEMBLE_NUT1", "ASSEMBLE_NUT2"):
                nut_slot = 1 if task == "ASSEMBLE_NUT1" else 2
                nut_index = STATION_NUT_INDICES.get(current_station, (1, 2))[nut_slot - 1]
                bolt_world_xy = STATION_BOLT_WORLD_POS.get(current_station, STATION_BOLT_WORLD_POS[3])[nut_slot]
                bolt_target_pos = np.array([bolt_world_xy[0], bolt_world_xy[1], 0.0])
                bolt_travel_pos = np.array([
                    bolt_target_pos[0],
                    bolt_target_pos[1],
                    NUT_APPROACH_Z,
                ])
                bolt_approach_pos = np.array([
                    bolt_target_pos[0],
                    bolt_target_pos[1],
                    BOLT_APPROACH_Z,
                ])
                bolt_touch_pos = np.array([bolt_target_pos[0], bolt_target_pos[1], 0.3697])
                nut_xy_lead = np.zeros(2, dtype=float)
                phase = "MOVE_TO_BOLT_NUT"
                step_count = 0
                print(f"\n>>> [{task}] 너트 {nut_index}번(스테이션{current_station} 슬롯{nut_slot}) -> 볼트 {nut_slot}번 체결 시작 "
                      f"(Target: X={bolt_target_pos[0]:.4f}, Y={bolt_target_pos[1]:.4f}) "
                      f"| 먼저 Z={NUT_APPROACH_Z:.3f}m에서 수평 이동")

        # 3. FSM 제어 루프
        if playing and phase != "IDLE" and phase != "DONE":

            if phase == "WAIT_BUSBAR_TARGET":
                # ROS 토픽 간 순서 역전은 정상이다. 고정캠 snapshot이 올 때까지
                # 임의 timeout 없이 기다리고, 프로세스 종료 시에는 바깥 loop가 끝난다.
                publish_progress("WAIT_BUSBAR_TARGET", 50.0)
                if isaac_node.latest_target_pose is not None:
                    print(
                        "[OK] 고정 버스바캠 pose 수신 -> PICK_BUSBAR 재개"
                    )
                    phase = "IDLE"
                    isaac_node.requested_task = "PICK_BUSBAR"
                    step_count = 0

            # [STEP 0-A] 초기 관절 자세 정렬
            elif phase == "INIT_POSE":
                publish_progress("INIT_ALIGN", 10.0)

                arm_joint_names = [
                    "joint_1", "joint_2", "joint_3",
                    "joint_4", "joint_5", "joint_6"
                ]
                arm_dof_indices = [robot.get_dof_index(name) for name in arm_joint_names]

                all_joints = robot.get_joint_positions()
                cur_arm_joints = all_joints[arm_dof_indices]

                max_step = ARM_INIT_MAX_JOINT_SPEED * PHYSICS_DT
                delta = TARGET_INIT_JOINTS - cur_arm_joints
                ramped_target = cur_arm_joints + np.clip(delta, -max_step, max_step)

                robot.apply_action(
                    ArticulationAction(
                        joint_positions=ramped_target,
                        joint_indices=arm_dof_indices
                    )
                )
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                joint_err = np.linalg.norm(cur_arm_joints - TARGET_INIT_JOINTS)

                if joint_err < JOINT_TOLERANCE:
                    ee_xf = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}")
                    HOME_EE_POS = np.array(ee_xf.ExtractTranslation(), dtype=float)
                    ee_quat = ee_xf.ExtractRotationQuat()
                    SCAN_POS = np.array([HOME_EE_POS[0], HOME_EE_POS[1], SCAN_APPROACH_Z])
                    # AMR이 어느 방향을 보고 있든, INIT_POSE가 만들어낸 실제 EE 자세를
                    # 그대로 유지한 채 Z만 올린다 (world 절대 방향을 새로 강제하지 않음).
                    scan_hold_quat = np.array(
                        [ee_quat.GetReal(), *[float(v) for v in ee_quat.GetImaginary()]]
                    )

                    print(f"[OK] 1) 관절 정렬 완료! (EE Pos: {HOME_EE_POS[0]:.3f}, {HOME_EE_POS[1]:.3f}, {HOME_EE_POS[2]:.3f})")

                    if init_pose_only:
                        print(" -> 초기 자세 복귀만 요청됨 - 스캔 이동 없이 종료")
                        publish_progress("RETURN_HOME_COMPLETE", 100.0)
                        publish_status("SUCCESS")
                        phase = "IDLE"
                    else:
                        print(f" -> 2) 초기 위치 기준 Z={SCAN_APPROACH_Z}m 상승 시작 (Target: {SCAN_POS[0]:.3f}, {SCAN_POS[1]:.3f}, {SCAN_POS[2]:.3f})")
                        phase = "SCAN_APPROACH"
                    step_count = 0
                elif step_count > MAX_STUCK_STEPS:
                    print(f"\n[ERROR] INIT_POSE 관절 정렬 실패 (joint_err={joint_err:.4f}rad) - 실패 처리")
                    publish_status("FAILURE:INIT_POSE_TIMEOUT")
                    phase = "IDLE"

            # [STEP 0-B] 배터리 스캔 고도 상승 (Z = 0.7m)
            elif phase == "SCAN_APPROACH":
                publish_progress("SCAN_NAV", 50.0)
                # world 절대 방향을 새로 계산하지 않고, INIT_POSE 완료 시점에 저장해둔
                # 실제 EE 자세(scan_hold_quat)를 그대로 유지한 채 Z만 올린다.
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                actions = arm_controller.forward(
                    target_end_effector_position=SCAN_POS,
                    target_end_effector_orientation=scan_hold_quat
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                xy_err = math.hypot(SCAN_POS[0] - cur_pos[0], SCAN_POS[1] - cur_pos[1])
                z_err = abs(SCAN_POS[2] - cur_pos[2])

                if step_count % 30 == 0:
                    print(f"[SCAN_NAV] Step={step_count} | EE=({cur_pos[0]:.4f}, {cur_pos[1]:.4f}, {cur_pos[2]:.4f}) | "
                          f"Target=({SCAN_POS[0]:.4f}, {SCAN_POS[1]:.4f}, {SCAN_POS[2]:.4f}) | "
                          f"XY err={xy_err*1000:.1f}mm | Z err={z_err*1000:.1f}mm")

                # Z만 올리는 단계라 Z 도착 여부를 기준으로 판정하고, XY는 크게 안 벗어났는지만 확인한다
                # (기존엔 XYZ 통합 거리(PICK_TOLERANCE_STRICT)로만 판정해서, Z는 도착했는데 XY가
                # 몇 mm~cm 밀리면 통과를 못 했다 - MAX_STUCK_STEPS가 사실상 무한대라 완화 조건도 안 먹었음).
                if z_err < 0.005 and xy_err < 0.02:
                    print(f"[OK] 배터리 스캔 위치 도착 완료! ({cur_pos[0]:.3f}, {cur_pos[1]:.3f}, {cur_pos[2]:.3f})")
                    publish_progress("SCAN_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"
                elif step_count > MAX_STUCK_STEPS:
                    print(f"\n[ERROR] SCAN_APPROACH Timeout | XY err={xy_err*1000:.1f}mm, Z err={z_err*1000:.1f}mm")
                    publish_status("FAILURE:SCAN_APPROACH_TIMEOUT")
                    phase = "IDLE"

            # [STEP 0.5] 방향 정렬 + 버스바 스캔 고도 상승 (제자리에서 quat_busbar로 회전, XY는 그대로)
            elif phase == "SCAN_BUSBAR_LIFT":
                publish_progress("SCAN_BUSBAR_LIFT", 30.0)
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                actions = arm_controller.forward(
                    target_end_effector_position=BUSBAR_SCAN_LIFT_POS,
                    target_end_effector_orientation=quat_busbar
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                err = math.dist(cur_pos, tuple(BUSBAR_SCAN_LIFT_POS))
                if err < 0.02 or step_count > MAX_STUCK_STEPS:
                    print("[OK] 방향 정렬 및 고도 상승 완료 -> 버스바 스캔 위치로 수평 이동")
                    phase = "SCAN_BUSBAR_APPROACH"
                    step_count = 0

            # [STEP 1] 버스바 스캔 위치 이동
            elif phase == "SCAN_BUSBAR_APPROACH":
                publish_progress("SCAN_BUSBAR_NAV", 50.0)
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                actions = arm_controller.forward(
                    target_end_effector_position=BUSBAR_SCAN_POS,
                    target_end_effector_orientation=quat_busbar
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                xy_err = math.hypot(BUSBAR_SCAN_POS[0] - cur_pos[0], BUSBAR_SCAN_POS[1] - cur_pos[1])
                z_err = abs(BUSBAR_SCAN_POS[2] - cur_pos[2])

                if step_count % 30 == 0:
                    print(f"[SCAN_BUSBAR_NAV] Step={step_count} | EE=({cur_pos[0]:.4f}, {cur_pos[1]:.4f}, {cur_pos[2]:.4f}) | "
                          f"Target=({BUSBAR_SCAN_POS[0]:.4f}, {BUSBAR_SCAN_POS[1]:.4f}, {BUSBAR_SCAN_POS[2]:.4f}) | "
                          f"XY err={xy_err*1000:.1f}mm | Z err={z_err*1000:.1f}mm")

                # 기존엔 XYZ 통합 거리(PICK_TOLERANCE_STRICT=10mm)로만 판정해서, MAX_STUCK_STEPS가
                # 사실상 무한대라 완화 조건도 안 먹혔다 (SCAN_APPROACH와 동일한 버그).
                if xy_err < 0.02 and z_err < 0.02:
                    print(f"[OK] 버스바 스캔 위치 도착 완료! ({cur_pos[0]:.3f}, {cur_pos[1]:.3f}, {cur_pos[2]:.3f})")
                    publish_progress("SCAN_BUSBAR_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"
                elif step_count > MAX_STUCK_STEPS:
                    print(f"\n[ERROR] SCAN_BUSBAR_APPROACH Timeout | XY err={xy_err*1000:.1f}mm, Z err={z_err*1000:.1f}mm")
                    publish_status("FAILURE:SCAN_BUSBAR_TIMEOUT")
                    phase = "IDLE"

            # [2단계] 버스바 상공 위치 접근 (Z = 0.6m)
            elif phase == "BUSBAR_APPROACH":
                publish_progress("APPROACH", 20.0)
                actions = arm_controller.forward(
                    target_end_effector_position=BUSBAR_APPROACH_POS,
                    target_end_effector_orientation=quat_busbar
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(BUSBAR_APPROACH_POS))

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 1. 버스바 상공(Z={BUSBAR_APPROACH_POS[2]:.3f}m) 접근 완료 -> 2. 파지점(Z={BUSBAR_PICK_POS[2]:.3f}m) 수직 하강 시작")
                    phase = "BUSBAR_DESCEND"
                    step_count = 0
                elif step_count > MAX_STUCK_STEPS:
                    print(
                        "\n[ERROR] BUSBAR_APPROACH 수렴 실패 "
                        f"({MAX_STUCK_STEPS} ticks) | "
                        f"EE=({cur_pos[0]:.4f},{cur_pos[1]:.4f},{cur_pos[2]:.4f}) | "
                        f"Target=({BUSBAR_APPROACH_POS[0]:.4f},"
                        f"{BUSBAR_APPROACH_POS[1]:.4f},"
                        f"{BUSBAR_APPROACH_POS[2]:.4f}) | "
                        f"err={current_err * 1000:.1f}mm"
                    )
                    publish_status("FAILURE:BUSBAR_APPROACH_TIMEOUT")
                    phase = "IDLE"
                elif step_count % 60 == 0:
                    print(
                        "[BUSBAR_APPROACH] "
                        f"EE=({cur_pos[0]:.4f},{cur_pos[1]:.4f},{cur_pos[2]:.4f}) | "
                        f"Target=({BUSBAR_APPROACH_POS[0]:.4f},"
                        f"{BUSBAR_APPROACH_POS[1]:.4f},"
                        f"{BUSBAR_APPROACH_POS[2]:.4f}) | "
                        f"err={current_err * 1000:.1f}mm"
                    )

            # [3단계] 파지 위치 수직 하강 (Z = 0.455m)
            elif phase == "BUSBAR_DESCEND":
                publish_progress("DESCEND", 50.0)
                actions = arm_controller.forward(
                    target_end_effector_position=BUSBAR_PICK_POS,
                    target_end_effector_orientation=quat_busbar
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(BUSBAR_PICK_POS))

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print("[OK] 2. 버스바 파지점 도착 -> Kinematic 파지 시작")
                    phase = "BUSBAR_GRASP"
                    grasp_timer = 0
                elif step_count > MAX_STUCK_STEPS:
                    print(
                        "\n[ERROR] BUSBAR_DESCEND 수렴 실패 "
                        f"({MAX_STUCK_STEPS} ticks) | "
                        f"err={current_err * 1000:.1f}mm"
                    )
                    publish_status("FAILURE:BUSBAR_DESCEND_TIMEOUT")
                    phase = "IDLE"
                elif step_count % 60 == 0:
                    print(
                        "[BUSBAR_DESCEND] "
                        f"EE=({cur_pos[0]:.4f},{cur_pos[1]:.4f},{cur_pos[2]:.4f}) | "
                        f"err={current_err * 1000:.1f}mm"
                    )

            # [4단계] 물리 파지 + FixedJoint 고정 - 버스바의 물리(중력/충돌)는 계속
            # 켜둔 채 그리퍼를 0.85까지 닫고, 안정화 뒤 link_6와 버스바 사이에
            # FixedJoint를 만들어 실제 wheel 주행 중에도 파지를 유지한다.
            elif phase == "BUSBAR_GRASP":
                publish_progress("GRASPING", 75.0)

                actions = arm_controller.forward(
                    target_end_effector_position=BUSBAR_PICK_POS,
                    target_end_effector_orientation=quat_busbar
                )
                robot.apply_action(actions)

                grasp_timer += 1
                ramp_frac = min(grasp_timer / GRIP_CLOSE_RAMP_STEPS, 1.0)
                grip_target = ramp_frac * GRIPPER_CLOSE
                robot.gripper.apply_action(ArticulationAction(joint_positions=grip_target))

                if grasp_timer == GRIP_CLOSE_RAMP_STEPS:
                    # 옮기는 동안에도 그리퍼가 계속 꽉 닫힌 것처럼 보이게 강성을 올려둔다
                    # (FixedJoint가 실제 고정을 담당하므로 필수는 아니지만 외관상 유지).
                    stiffen_gripper_grip(robot)

                if grasp_timer >= GRIP_CLOSE_RAMP_STEPS + GRIP_SETTLE_STEPS:
                    joint = attach_busbar_to_gripper(
                        stage,
                        ee_path,
                        robot,
                        busbar_xform,
                        busbar_mesh_path,
                    )
                    if joint is None:
                        print(
                            "\n[ERROR] 버스바 파지 실패 "
                            f"(prim 경로 확인 필요: {busbar_mesh_path})"
                        )
                        publish_status("FAILURE:BUSBAR_ATTACH_FAILED")
                        phase = "IDLE"
                    else:
                        busbar_grasped = True
                        live_busbar_position, _ = busbar_xform.get_world_pose()
                        busbar_grasp_start_z = float(live_busbar_position[2])
                        busbar_lift_stable_steps = 0
                        print("[OK] 3. 버스바 물리 파지 + FixedJoint 고정 완료 -> 4. 안전 고도 상승")
                        phase = "BUSBAR_LIFT"
                        step_count = 0

            # [5단계] 안전 고도 들어올리기 (Z = 0.6m)
            elif phase == "BUSBAR_LIFT":
                publish_progress("LIFTING", 90.0)
                actions = arm_controller.forward(
                    target_end_effector_position=BUSBAR_LIFT_MOVE_POS,
                    target_end_effector_orientation=quat_busbar
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))


                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(BUSBAR_LIFT_MOVE_POS))
                live_busbar_position, _ = busbar_xform.get_world_pose()
                busbar_lift_m = (
                    float(live_busbar_position[2]) - busbar_grasp_start_z
                    if busbar_grasp_start_z is not None
                    else float("-inf")
                )
                if busbar_lift_m >= BUSBAR_GRASP_MIN_LIFT_M:
                    busbar_lift_stable_steps += 1
                else:
                    busbar_lift_stable_steps = 0

                if busbar_lift_stable_steps >= BUSBAR_LIFT_STABLE_STEPS:
                    print(
                        "\n★ [PICK_BUSBAR SUCCESS] 실제 버스바가 "
                        f"{busbar_lift_m * 1000:.1f}mm 상승한 상태를 "
                        f"{busbar_lift_stable_steps}틱 연속 확인했습니다."
                    )
                    publish_progress("COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"
                elif step_count > MAX_STUCK_STEPS:
                    print(
                        "\n[ERROR] BUSBAR_LIFT 수렴 실패 "
                        f"({MAX_STUCK_STEPS} ticks) | "
                        f"EE err={current_err * 1000:.1f}mm, "
                        f"busbar lift={busbar_lift_m * 1000:.1f}mm"
                    )
                    publish_status("FAILURE:BUSBAR_LIFT_TIMEOUT")
                    phase = "IDLE"
                elif step_count % 60 == 0:
                    print(
                        "[BUSBAR_LIFT] "
                        f"EE=({cur_pos[0]:.4f},{cur_pos[1]:.4f},{cur_pos[2]:.4f}) | "
                        f"err={current_err * 1000:.1f}mm | "
                        f"busbar lift={busbar_lift_m * 1000:.1f}mm | "
                        f"stable={busbar_lift_stable_steps}/"
                        f"{BUSBAR_LIFT_STABLE_STEPS}"
                    )

            # [6단계] 배터리 중점 상공으로 이동 (Z = 0.7m, 버스바 고정 상태 유지)
            elif phase == "MOVE_BATTERY_CENTER_APPROACH":
                publish_progress("MOVING_BATTERY_CENTER", 50.0)
                actions = arm_controller.forward(
                    target_end_effector_position=BATTERY_CENTER_POS,
                    target_end_effector_orientation=euler_to_quaternion_wxyz(0.0, 3.1415, 0.0)
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(BATTERY_CENTER_POS))

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    print(f"[OK] 배터리 중점 상공 위치 도착 완료! ({cur_pos[0]:.3f}, {cur_pos[1]:.3f}, {cur_pos[2]:.3f})")
                    publish_progress("MOVE_BATTERY_CENTER_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"

            # [7단계] 비전 보정 노드 피드백 기반 미세 오차 정렬
            elif phase == "FINE_ALIGNMENT":
                publish_progress("FINE_ALIGNMENT_TRACKING", 85.0)

                ee_path = f"{M0609_PATH}/{EE_LINK_NAME}"
                cur_ee_pos, cur_ee_orientation = _world_pose_for_path(
                    stage,
                    ee_path,
                )
                fine_step_started_this_tick = False

                correction_msg = (
                    isaac_node.latest_fine_alignment_delta
                )
                if correction_msg is not None:
                    isaac_node.latest_fine_alignment_delta = None
                    if fine_step_gate.active_stamp_ns is not None:
                        print(
                            "\n[WARN] 이전 FINE_ALIGNMENT delta가 실제 "
                            "settle되기 전에 온 추가 delta를 폐기합니다"
                        )
                    else:
                        correction = correction_msg.pose
                        offset = correction.position
                        try:
                            expected_frame_id = (
                                f"{FINE_ALIGNMENT_DELTA_FRAME}:"
                                f"{fine_alignment_generation}"
                            )
                            if (
                                correction_msg.header.frame_id
                                != expected_frame_id
                            ):
                                raise LookupError(
                                    "stale fine-alignment generation"
                                )
                            stamp_ns = pose_stamp_ns(correction_msg)
                            offset_values = (
                                float(offset.x),
                                float(offset.y),
                                float(offset.z),
                            )
                            if not all(
                                math.isfinite(value)
                                for value in offset_values
                            ):
                                raise ValueError(
                                    "delta position에 NaN 또는 inf가 있습니다"
                                )
                            if (
                                abs(offset.x) > 0.0025
                                or abs(offset.y) > 0.0025
                                or abs(offset.z) > 1.0e-9
                            ):
                                raise ValueError(
                                    "delta position이 safety bound를 "
                                    "벗어났습니다"
                                )
                            yaw_delta = planar_yaw_delta(
                                quaternion_x=correction.orientation.x,
                                quaternion_y=correction.orientation.y,
                                quaternion_z=correction.orientation.z,
                                quaternion_w=correction.orientation.w,
                            )
                        except LookupError as exc:
                            print(
                                "\n[WARN] 현재 task와 다른 "
                                f"FINE_ALIGNMENT delta 폐기: {exc}"
                            )
                        except (TypeError, ValueError) as exc:
                            print(
                                "\n[ERROR] 잘못된 FINE_ALIGNMENT delta를 "
                                f"거부합니다: {exc}"
                            )
                            hold_current_arm_pose()
                            publish_status(
                                "FAILURE:INVALID_FINE_ALIGNMENT_DELTA"
                            )
                            reset_cmd = String()
                            reset_cmd.data = "RESET_BOLT_DETECTION"
                            isaac_node.pub_errorfix_command.publish(
                                reset_cmd
                            )
                            fine_step_gate.reset()
                            fine_controller_lead.reset()
                            fine_step_previous_ee_pos = None
                            fine_step_previous_ee_orientation = None
                            fine_step_start_ee_pos = None
                            fine_step_command_delta_xy = None
                            fine_step_start_orientation_error_rad = None
                            isaac_node.expected_fine_alignment_generation = (
                                None
                            )
                            phase = "IDLE"
                        else:
                            delta_xy = np.array(
                                [offset.x, offset.y],
                                dtype=float,
                            )
                            delta_xy_norm = float(
                                np.linalg.norm(delta_xy)
                            )
                            target_fine_pos[0] += offset.x
                            target_fine_pos[1] += offset.y
                            target_fine_pos[2] = BATTERY_CENTER_Z
                            target_fine_yaw_rad += yaw_delta
                            target_fine_yaw_rad = (
                                target_fine_yaw_rad + math.pi
                            ) % (2.0 * math.pi) - math.pi
                            step_target_orientation = (
                                euler_to_quaternion_wxyz(
                                    0.0,
                                    3.1415,
                                    target_fine_yaw_rad,
                                )
                            )
                            fine_step_gate.begin(
                                stamp_ns,
                                required_position_response_m=(
                                    delta_xy_norm
                                    * FINE_STEP_MIN_RESPONSE_RATIO
                                ),
                                required_orientation_response_rad=(
                                    abs(yaw_delta)
                                    * FINE_STEP_MIN_RESPONSE_RATIO
                                ),
                            )
                            fine_controller_lead.reset()
                            fine_controller_lead.begin(
                                delta_xy[0],
                                delta_xy[1],
                            )
                            fine_step_previous_ee_pos = (
                                cur_ee_pos.copy()
                            )
                            fine_step_previous_ee_orientation = (
                                cur_ee_orientation.copy()
                            )
                            fine_step_start_ee_pos = cur_ee_pos.copy()
                            fine_step_command_delta_xy = delta_xy.copy()
                            fine_step_start_orientation_error_rad = (
                                _quat_angular_error(
                                    cur_ee_orientation,
                                    step_target_orientation,
                                )
                            )
                            fine_step_started_this_tick = True

                            print(
                                "\n[FINE_ALIGNMENT] consensus delta 수신 "
                                f"-> dx={offset.x:+.4f}m, "
                                f"dy={offset.y:+.4f}m, "
                                f"dyaw={math.degrees(yaw_delta):+.3f}deg, "
                                f"stamp={stamp_ns}"
                            )
                            print(
                                "               └─ New Target "
                                f"X={target_fine_pos[0]:.4f}, "
                                f"Y={target_fine_pos[1]:.4f}, "
                                f"Z={target_fine_pos[2]:.4f}, "
                                "yaw="
                                f"{math.degrees(target_fine_yaw_rad):+.3f}deg"
                            )
                            print(
                                "               └─ Controller-only XY lead "
                                f"armed: +{FINE_CONTROLLER_LEAD_GROWTH_M * 1000:.2f}"
                                "mm/tick, "
                                f"cap={FINE_CONTROLLER_MAX_LEAD_M * 1000:.2f}mm; "
                                "logical target/Z/yaw unchanged until settle"
                            )

                if phase == "FINE_ALIGNMENT":
                    target_fine_orientation = (
                        euler_to_quaternion_wxyz(
                            0.0,
                            3.1415,
                            target_fine_yaw_rad,
                        )
                    )
                    position_response_m = 0.0
                    required_position_response_m = 0.0
                    if fine_step_gate.active_stamp_ns is not None:
                        required_position_response_m = (
                            fine_step_gate.required_position_response_m
                        )
                        if (
                            fine_step_start_ee_pos is not None
                            and fine_step_command_delta_xy is not None
                        ):
                            position_response_m = (
                                planar_command_response(
                                    start_x=fine_step_start_ee_pos[0],
                                    start_y=fine_step_start_ee_pos[1],
                                    current_x=cur_ee_pos[0],
                                    current_y=cur_ee_pos[1],
                                    command_x=fine_step_command_delta_xy[0],
                                    command_y=fine_step_command_delta_xy[1],
                                )
                            )

                    lead_status = fine_controller_lead.advance(
                        observed_response_m=position_response_m,
                        required_response_m=(
                            required_position_response_m
                        ),
                    )
                    controller_x, controller_y = (
                        fine_controller_lead.command_xy(
                            target_fine_pos[0],
                            target_fine_pos[1],
                        )
                    )
                    fine_controller_target_pos = target_fine_pos.copy()
                    fine_controller_target_pos[0] = controller_x
                    fine_controller_target_pos[1] = controller_y
                    if lead_status.grew and lead_status.saturated:
                        print(
                            "\n[WARN] FINE controller XY lead가 안전 "
                            f"상한 {FINE_CONTROLLER_MAX_LEAD_M * 1000:.2f}mm"
                            "에 도달했습니다; 실제 80% 응답 전에는 ACK하지 "
                            "않습니다"
                        )

                    sys.stdout.write(
                        "\r\033[K[FINE_ALIGNMENT Loop] "
                        f"Cur EE=({cur_ee_pos[0]:.4f}, "
                        f"{cur_ee_pos[1]:.4f}) -> "
                        f"Logical=({target_fine_pos[0]:.4f}, "
                        f"{target_fine_pos[1]:.4f}), "
                        f"Controller=({fine_controller_target_pos[0]:.4f}, "
                        f"{fine_controller_target_pos[1]:.4f}), "
                        "response="
                        f"{position_response_m * 1000:.3f}/"
                        f"{required_position_response_m * 1000:.3f}mm, "
                        f"lead={lead_status.magnitude_m * 1000:.2f}/"
                        f"{FINE_CONTROLLER_MAX_LEAD_M * 1000:.2f}mm, "
                        f"saturated={lead_status.saturated}, "
                        "pending="
                        f"{fine_step_gate.active_stamp_ns is not None}"
                    )
                    sys.stdout.flush()
                    if (
                        fine_step_gate.active_stamp_ns is not None
                        and step_count % 60 == 0
                    ):
                        print(
                            "\n[FINE_ALIGNMENT RESPONSE] actual="
                            f"{position_response_m * 1000:.3f}mm, "
                            "required="
                            f"{required_position_response_m * 1000:.3f}mm, "
                            f"lead={lead_status.magnitude_m * 1000:.3f}mm/"
                            f"{FINE_CONTROLLER_MAX_LEAD_M * 1000:.3f}mm, "
                            f"saturated={lead_status.saturated}"
                        )

                    actions = arm_controller.forward(
                        target_end_effector_position=(
                            fine_controller_target_pos
                        ),
                        target_end_effector_orientation=(
                            target_fine_orientation
                        ),
                    )
                    robot.apply_action(actions)
                    robot.gripper.apply_action(
                        ArticulationAction(
                            joint_positions=GRIPPER_CLOSE
                        )
                    )

                    if (
                        fine_step_gate.active_stamp_ns is not None
                        and not fine_step_started_this_tick
                    ):
                        previous_position = (
                            fine_step_previous_ee_pos
                            if fine_step_previous_ee_pos is not None
                            else cur_ee_pos
                        )
                        previous_orientation = (
                            fine_step_previous_ee_orientation
                            if fine_step_previous_ee_orientation is not None
                            else cur_ee_orientation
                        )
                        position_error_m = math.dist(
                            cur_ee_pos,
                            target_fine_pos,
                        )
                        orientation_error_rad = _quat_angular_error(
                            cur_ee_orientation,
                            target_fine_orientation,
                        )
                        orientation_response_rad = max(
                            0.0,
                            (
                                fine_step_start_orientation_error_rad
                                - orientation_error_rad
                            )
                            if (
                                fine_step_start_orientation_error_rad
                                is not None
                            )
                            else 0.0,
                        )
                        orientation_motion_rad = (
                            _quat_angular_error(
                                cur_ee_orientation,
                                previous_orientation,
                            )
                        )
                        settle_status = fine_step_gate.update(
                            position_error_m=position_error_m,
                            orientation_error_rad=orientation_error_rad,
                            motion_m=math.dist(
                                cur_ee_pos,
                                previous_position,
                            ),
                            orientation_motion_rad=(
                                orientation_motion_rad
                            ),
                            position_response_m=position_response_m,
                            orientation_response_rad=(
                                orientation_response_rad
                            ),
                        )
                        fine_step_previous_ee_pos = cur_ee_pos.copy()
                        fine_step_previous_ee_orientation = (
                            cur_ee_orientation.copy()
                        )
                        if settle_status.settled:
                            committed_lead = fine_controller_lead.status()
                            committed_x, committed_y = (
                                fine_controller_lead.commit_xy(
                                    target_fine_pos[0],
                                    target_fine_pos[1],
                                )
                            )
                            target_fine_pos[0] = committed_x
                            target_fine_pos[1] = committed_y
                            settled_msg = String()
                            settled_msg.data = (
                                "FINE_ALIGNMENT_STEP_SETTLED:"
                                f"{settle_status.stamp_ns}"
                            )
                            isaac_node.pub_errorfix_command.publish(
                                settled_msg
                            )
                            fine_step_previous_ee_pos = None
                            fine_step_previous_ee_orientation = None
                            fine_step_start_ee_pos = None
                            fine_step_command_delta_xy = None
                            fine_step_start_orientation_error_rad = None
                            print(
                                "\n[FINE_ALIGNMENT] 실제 EE pose/정지 "
                                f"{FINE_STEP_SETTLE_STEPS}틱 확인 -> "
                                "post-motion 3-frame 측정 재개 "
                                f"(stamp={settle_status.stamp_ns}, "
                                "response="
                                f"{position_response_m * 1000:.3f}/"
                                f"{required_position_response_m * 1000:.3f}"
                                "mm, committed lead="
                                f"{committed_lead.magnitude_m * 1000:.3f}mm)"
                            )

                    if (
                        isaac_node.alignment_success
                        and fine_step_gate.active_stamp_ns is None
                    ):
                        print(
                            "\n\n★ [FINE_ALIGNMENT SUCCESS] 미세 오차 "
                            "보정 성공 및 정렬 완료!"
                        )
                        publish_progress(
                            "FINE_ALIGNMENT_COMPLETE",
                            100.0,
                        )
                        publish_status("SUCCESS")
                        fine_step_gate.reset()
                        fine_controller_lead.reset()
                        fine_step_previous_ee_pos = None
                        fine_step_previous_ee_orientation = None
                        fine_step_start_ee_pos = None
                        fine_step_command_delta_xy = None
                        fine_step_start_orientation_error_rad = None
                        isaac_node.expected_fine_alignment_generation = None
                        phase = "IDLE"

            # [8단계] 버스바 수직 하강 체결
            elif phase == "BUSBAR_DESCEND_TO_BOLT":
                publish_progress("BUSBAR_DESCEND_INSERT", 90.0)

                descend_target_z = max(descend_target_z - INSERT_SPEED, target_mid_pos[2])
                step_target_pos = np.array([target_mid_pos[0], target_mid_pos[1], descend_target_z])

                actions = arm_controller.forward(
                    target_end_effector_position=step_target_pos,
                    target_end_effector_orientation=(
                        euler_to_quaternion_wxyz(
                            0.0,
                            3.1415,
                            target_fine_yaw_rad,
                        )
                    ),
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                dist_err = math.dist(cur_pos, tuple(target_mid_pos))

                if cur_pos[2] <= BUSBAR_RELEASE_Z or dist_err < INSERT_TOLERANCE_STRICT:
                    # 실제 물리 그립 상태라 여기서 busbar 위치를 강제로 순간이동(snap)
                    # 시키면 안 된다 - 그리퍼가 여전히 물고 있는 채로 물체만 텔레포트하면
                    # 다음 physics step에서 손가락과 겹침이 발생해 튕겨나갈 수 있다.
                    # 착좌 정확도는 이제 IK가 내려간 실제 위치(target_mid_pos)에 의존한다.
                    print(f"\n[OK] 버스바 안착 체결 완료 (EE Z: {cur_pos[2]:.4f}m)!")

                    phase = "BUSBAR_RELEASE_AND_RETRACT"
                    step_count = 0

            # [9단계] 버스바 해제 및 안전 상공 이탈
            elif phase == "BUSBAR_RELEASE_AND_RETRACT":
                publish_progress("BUSBAR_RETRACT", 95.0)

                if busbar_grasped:
                    # 그리퍼를 열기 전에 FixedJoint부터 풀어야 한다 - 안 그러면 조인트가
                    # 버스바를 계속 붙잡고 있어서 그리퍼가 열려도 안 떨어진다.
                    detach_busbar_from_gripper(stage, busbar_mesh_path)
                    busbar_grasped = False

                retract_pos = np.array([target_mid_pos[0], target_mid_pos[1], BATTERY_CENTER_Z])
                actions = arm_controller.forward(
                    target_end_effector_position=retract_pos,
                    target_end_effector_orientation=(
                        euler_to_quaternion_wxyz(
                            0.0,
                            3.1415,
                            target_fine_yaw_rad,
                        )
                    ),
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(retract_pos))

                if current_err < PICK_TOLERANCE_STRICT or step_count > MAX_STUCK_STEPS:
                    print("\n★ [ASSEMBLE_BUSBAR SUCCESS] 버스바 체결 및 안전 이탈 완료!")
                    publish_progress("ASSEMBLE_BUSBAR_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"

            # ════════════════════════════════════════════════════════════════
            # 너트 조립(Nut Assembly) 공용 Phase
            # ════════════════════════════════════════════════════════════════
            # [9.5단계] 방향 정렬 + 스캔 고도 상승 (제자리에서 quat_nut로 회전, XY는 그대로)
            elif phase == "NUT_SCAN_LIFT":
                publish_progress("NUT_SCAN_LIFT", 30.0)
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                _, cur_quat = robot.end_effector.get_world_pose()
                actions = arm_controller.forward(target_end_effector_position=NUT_SCAN_LIFT_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                position_err = math.dist(
                    cur_pos,
                    tuple(NUT_SCAN_LIFT_POS),
                )
                orientation_err = _quat_angular_error(
                    cur_quat,
                    quat_nut,
                )
                nut_pose_settle_count, pose_settled = (
                    consecutive_pose_settle(
                        position_err,
                        orientation_err,
                        nut_pose_settle_count,
                        position_tolerance_m=NUT_POSE_POSITION_TOL_M,
                        orientation_tolerance_rad=(
                            NUT_POSE_ORIENTATION_TOL_RAD
                        ),
                        required_steps=NUT_POSE_SETTLE_STEPS,
                    )
                )

                if step_count % 30 == 0:
                    print(
                        "[NUT_SCAN_LIFT] "
                        f"position_err={position_err * 1000:.1f}mm, "
                        f"orientation_err="
                        f"{math.degrees(orientation_err):.2f}deg, "
                        f"stable={nut_pose_settle_count}/"
                        f"{NUT_POSE_SETTLE_STEPS}"
                    )

                if pose_settled:
                    print(f"[OK] 방향 정렬 및 고도 상승 완료 -> 너트 스캔 위치로 수평 이동")
                    phase = "NUT_SCAN_APPROACH"
                    step_count = 0

            # [10단계] 너트 스캔 위치 이동
            elif phase == "NUT_SCAN_APPROACH":
                publish_progress("NUT_SCAN_NAV", 50.0)
                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                actions = arm_controller.forward(target_end_effector_position=NUT_SCAN_POS, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                xy_err = math.hypot(NUT_SCAN_POS[0] - cur_pos[0], NUT_SCAN_POS[1] - cur_pos[1])
                z_err = abs(NUT_SCAN_POS[2] - cur_pos[2])

                if step_count % 30 == 0:
                    print(f"[NUT_SCAN_NAV] Step={step_count} | EE=({cur_pos[0]:.4f}, {cur_pos[1]:.4f}, {cur_pos[2]:.4f}) | "
                          f"Target=({NUT_SCAN_POS[0]:.4f}, {NUT_SCAN_POS[1]:.4f}, {NUT_SCAN_POS[2]:.4f}) | "
                          f"XY err={xy_err*1000:.1f}mm | Z err={z_err*1000:.1f}mm")

                if xy_err < 0.02 and z_err < 0.02:
                    print(f"[OK] 너트 스캔 위치 도착 완료! ({cur_pos[0]:.3f}, {cur_pos[1]:.3f}, {cur_pos[2]:.3f})")
                    publish_progress("NUT_SCAN_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"
                elif step_count > MAX_STUCK_STEPS:
                    print(f"\n[ERROR] NUT_SCAN_APPROACH Timeout | XY err={xy_err*1000:.1f}mm, Z err={z_err*1000:.1f}mm")
                    publish_status("FAILURE:NUT_SCAN_TIMEOUT")
                    phase = "IDLE"

            # [11단계] 너트 상공 접근
            elif phase == "NUT_APPROACH":
                publish_progress("NUT_APPROACH", 20.0)
                actions = arm_controller.forward(target_end_effector_position=nut_approach_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(nut_approach_pos))

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    nut_label = resolve_nut_assets(nut_index, nut1_xform, nut2_xform)[1]
                    print(f"[OK] {nut_label} 상공 도착! -> 하강 시작")
                    phase = "NUT_DESCEND"
                    step_count = 0

            # [12단계] 너트 파지점 하강
            elif phase == "NUT_DESCEND":
                publish_progress("NUT_DESCEND", 40.0)
                actions = arm_controller.forward(target_end_effector_position=nut_pick_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                current_err = math.dist(cur_pos, tuple(nut_pick_pos))

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    nut_label = resolve_nut_assets(nut_index, nut1_xform, nut2_xform)[1]
                    # 성공했던 66f1825 방식: 너트를 AMR에 고정해 둔 채 손가락부터
                    # 닫는다. 열린 상태에서 조인트를 먼저 풀면 너트가 움직여 파지
                    # 중심에서 벗어날 수 있다.
                    enable_collision_recursively(
                        stage,
                        nut_roots_all[nut_pick_active_array_index],
                    )
                    print(f"[OK] {nut_label} 하강 완료! -> 물리 파지(Gripper Close) 시작")
                    phase = "NUT_GRASP"
                    grasp_timer = 0

            # [13단계] battery4_main과 동일한 순수 접촉/마찰 기반 너트 파지.
            elif phase == "NUT_GRASP":
                publish_progress("NUT_GRASPING", 60.0)
                actions = arm_controller.forward(target_end_effector_position=nut_pick_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)

                grasp_timer += 1
                ramp_frac = min(grasp_timer / GRIP_CLOSE_RAMP_STEPS, 1.0)
                grip_target = ramp_frac * GRIPPER_CLOSE_NUT
                robot.gripper.apply_action(ArticulationAction(joint_positions=grip_target))

                if grasp_timer >= GRIP_CLOSE_RAMP_STEPS:
                    nut_xf_cur, nut_label = resolve_nut_assets(
                        nut_index,
                        nut1_xform,
                        nut2_xform,
                        extra_nut_xforms,
                    )
                    ee_grasp_pos = np.asarray(
                        world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation(),
                        dtype=float,
                    )
                    clear_z = ee_grasp_pos[2] + NUT_PEG_CLEARANCE_Z
                    command_clear_z = (
                        clear_z + NUT_PEG_CLEAR_COMMAND_MARGIN_M
                    )
                    if nut_xf_cur is None:
                        print(
                            f"[ERROR] {nut_label} 실제 Xform이 없어 "
                            "peg 이탈을 검증할 수 없습니다"
                        )
                        restore_active_nut_to_tray_glue(
                            "missing nut xform after close"
                        )
                        publish_status("FAILURE:NUT_XFORM_NOT_FOUND")
                        phase = "IDLE"
                    elif nut_pick_active_array_index != nut_index - 1:
                        print(
                            f"[ERROR] {nut_label} active PICK 상태가 "
                            "현재 너트와 일치하지 않습니다"
                        )
                        restore_active_nut_to_tray_glue(
                            "active nut mismatch"
                        )
                        publish_status("FAILURE:NUT_PICK_STATE_MISMATCH")
                        phase = "IDLE"
                    elif command_clear_z > NUT_APPROACH_Z:
                        print(
                            f"[ERROR] {nut_label} 80mm 이탈 command "
                            f"Z={command_clear_z:.4f}가 안전 상공 "
                            f"Z={NUT_APPROACH_Z:.4f}를 넘습니다"
                        )
                        restore_active_nut_to_tray_glue(
                            "unsafe peg-clear target"
                        )
                        publish_status("FAILURE:NUT_PEG_CLEAR_TARGET_UNSAFE")
                        phase = "IDLE"
                    elif not nut_released[nut_pick_active_array_index]:
                        # 손가락을 완전히 닫기 전에는 절대 tray glue를 풀지
                        # 않는다. 이 시점부터만 dynamics/contact가 너트를
                        # 소유하며 cancel/conflict는 원래 glue로 되돌린다.
                        nut_start_pos, _ = nut_xf_cur.get_world_pose()
                        nut_start_pos = np.asarray(
                            nut_start_pos,
                            dtype=float,
                        )
                        nut_ee_reference_offset = nut_from_ee_offset(
                            ee_grasp_pos,
                            nut_start_pos,
                        )
                        nut_ee_previous_offset = (
                            nut_ee_reference_offset.copy()
                        )
                        remove_nut_amr_joint(
                            stage,
                            nut_roots_all[
                                nut_pick_active_array_index
                            ],
                        )
                        enable_physics_recursively(
                            stage,
                            nut_roots_all[
                                nut_pick_active_array_index
                            ],
                        )
                        nut_released[nut_pick_active_array_index] = True
                        nut_release_timer = 0
                        print(
                            f"[NUT GRIP RELEASE] {nut_label} 닫힘 완료 "
                            "후 tray glue 해제; EE-relative 결합 안정화 시작"
                        )
                    else:
                        nut_start_pos, _ = nut_xf_cur.get_world_pose()
                        nut_start_pos = np.asarray(
                            nut_start_pos,
                            dtype=float,
                        )
                        current_nut_from_ee = nut_from_ee_offset(
                            ee_grasp_pos,
                            nut_start_pos,
                        )
                        coupling_error = nut_ee_coupling_error(
                            ee_grasp_pos,
                            nut_start_pos,
                            nut_ee_reference_offset,
                        )
                        settle_motion = float(np.linalg.norm(
                            current_nut_from_ee
                            - nut_ee_previous_offset
                        ))
                        if (
                            coupling_error
                            > NUT_EE_COUPLING_TOLERANCE_M
                        ):
                            print(
                                f"[ERROR] {nut_label} tray glue 해제 후 "
                                "EE-relative drift="
                                f"{coupling_error*1000:.1f}mm"
                            )
                            restore_active_nut_to_tray_glue(
                                "grip coupling lost during settle"
                            )
                            publish_status(
                                "FAILURE:NUT_GRIP_COUPLING_LOST"
                            )
                            phase = "IDLE"
                        else:
                            # dynamics 전환 직후 nut가 닫힌 손가락 안에서
                            # self-seat할 수 있다. 최초 release 기준 10mm
                            # 안전 한계는 계속 지키되, 상대 위치가 실제로
                            # 정지한 tick만 연속 안정화로 인정한다.
                            if (
                                settle_motion
                                <= NUT_EE_SETTLE_MOTION_TOLERANCE_M
                            ):
                                nut_release_timer += 1
                            else:
                                nut_release_timer = 0
                            nut_ee_previous_offset = (
                                current_nut_from_ee.copy()
                            )
                            if nut_release_timer >= GRIP_SETTLE_STEPS:
                                # release 순간의 preload 기준이 아니라,
                                # dynamics에서 15틱 정지한 실제 결합 위치를
                                # peg-clear/lift 운반 기준으로 latch한다.
                                nut_ee_reference_offset = (
                                    current_nut_from_ee.copy()
                                )
                                nut_ee_previous_offset = None
                                # 바로 다음 태스크로 넘기지 않고, 현재 XY를
                                # 고정한 채 peg 길이보다 충분히 위로 먼저
                                # 수직 이탈한다.
                                nut_peg_clear_pos = np.array([
                                    ee_grasp_pos[0],
                                    ee_grasp_pos[1],
                                    clear_z,
                                ])
                                nut_peg_clear_start_pos = (
                                    ee_grasp_pos.copy()
                                )
                                nut_peg_clear_start_nut_z = float(
                                    nut_start_pos[2]
                                )
                                nut_lift_command_z = float(
                                    ee_grasp_pos[2]
                                )
                                nut_peg_clear_xy_lead = np.zeros(
                                    2,
                                    dtype=float,
                                )
                                nut_peg_clear_z_lead = 0.0
                                nut_peg_clear_hold = 0
                                print(
                                    f"[OK] {nut_label} 결합 "
                                    f"{GRIP_SETTLE_STEPS}틱 검증 완료! -> "
                                    "release drift="
                                    f"{coupling_error*1000:.1f}mm, "
                                    "settle motion="
                                    f"{settle_motion*1000:.2f}mm; "
                                    "XY 고정 수직 이탈 "
                                    f"{NUT_PEG_CLEARANCE_Z*1000:.0f}mm "
                                    "시작 "
                                    f"(required Z={clear_z:.4f}, command "
                                    f"goal Z={command_clear_z:.4f}, "
                                    f"ramp={NUT_LIFT_COMMAND_MAX_STEP_M*1000:.1f}"
                                    "mm/tick)"
                                )
                                phase = "NUT_LIFT_CLEAR_PEG"
                                step_count = 0

            # [13.5단계] 공급대 peg에서 완전히 빠질 때까지 XY 고정 수직 상승
            elif phase == "NUT_LIFT_CLEAR_PEG":
                publish_progress("NUT_LIFT_CLEAR_PEG", 75.0)
                cur_pos = np.asarray(
                    world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation(),
                    dtype=float,
                )

                # 실제 80mm에 도달한 뒤 15틱을 세는 동안에는 그 순간의 lead를
                # 고정한다. lead가 즉시 감쇠해 다시 80mm 아래로 내려가는 것을
                # 방지하면서도 성공 판정은 command가 아닌 실제 변위로만 수행한다.
                if nut_lift_command_z is None:
                    nut_lift_command_z = float(cur_pos[2])
                if nut_peg_clear_hold == 0:
                    nut_peg_clear_command, nut_peg_clear_xy_lead = lead_xy_target(
                        nut_peg_clear_pos,
                        cur_pos,
                        nut_peg_clear_xy_lead,
                    )
                    nut_peg_clear_tracking_pos = (
                        nut_peg_clear_pos.copy()
                    )
                    nut_peg_clear_tracking_pos[2] += (
                        NUT_PEG_CLEAR_COMMAND_MARGIN_M
                    )
                    z_command, nut_peg_clear_z_lead = lead_z_target(
                        nut_peg_clear_tracking_pos,
                        cur_pos,
                        nut_peg_clear_z_lead,
                    )
                    nut_lift_command_z = rate_limited_z_target(
                        z_command[2],
                        nut_lift_command_z,
                        max_step=NUT_LIFT_COMMAND_MAX_STEP_M,
                    )
                    nut_peg_clear_command[2] = nut_lift_command_z
                else:
                    nut_peg_clear_command = nut_peg_clear_pos.copy()
                    nut_peg_clear_command[:2] += nut_peg_clear_xy_lead
                    nut_peg_clear_command[2] = nut_lift_command_z

                actions = arm_controller.forward(
                    target_end_effector_position=nut_peg_clear_command,
                    target_end_effector_orientation=quat_nut,
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                actual_lift = float(
                    cur_pos[2] - nut_peg_clear_start_pos[2]
                )
                nut_xf_cur = resolve_nut_assets(
                    nut_index,
                    nut1_xform,
                    nut2_xform,
                    extra_nut_xforms,
                )[0]
                xy_err = float(np.linalg.norm(
                    cur_pos[:2] - nut_peg_clear_start_pos[:2]
                ))
                if (
                    nut_xf_cur is None
                    or nut_ee_reference_offset is None
                ):
                    print(
                        "[ERROR] peg 이탈 중 너트 Xform/EE 결합 "
                        "기준이 없습니다"
                    )
                    restore_active_nut_to_tray_glue(
                        "missing peg-clear coupling state"
                    )
                    publish_status(
                        "FAILURE:NUT_GRIP_COUPLING_STATE_MISSING"
                    )
                    phase = "IDLE"
                else:
                    nut_now_pos, _ = nut_xf_cur.get_world_pose()
                    nut_now_pos = np.asarray(
                        nut_now_pos,
                        dtype=float,
                    )
                    actual_nut_lift = float(
                        nut_now_pos[2] - nut_peg_clear_start_nut_z
                    )
                    coupling_error = nut_ee_coupling_error(
                        cur_pos,
                        nut_now_pos,
                        nut_ee_reference_offset,
                    )

                    # 독립적인 EE Z와 nut Z 조건만으로는 같은 물체를
                    # 운반했다는 증거가 아니다. 저장한 EE-relative offset
                    # 역시 유지되어야만 hold tick을 센다.
                    if (
                        coupling_error
                        > NUT_EE_COUPLING_TOLERANCE_M
                    ):
                        print(
                            "[ERROR] peg 이탈 중 너트 결합 상실: "
                            f"relative drift={coupling_error*1000:.1f}mm"
                        )
                        restore_active_nut_to_tray_glue(
                            "coupling lost during peg clear"
                        )
                        publish_status(
                            "FAILURE:NUT_GRIP_COUPLING_LOST"
                        )
                        phase = "IDLE"
                    else:
                        if (
                            actual_lift >= NUT_PEG_CLEARANCE_Z
                            and actual_nut_lift >= NUT_PEG_CLEARANCE_Z
                            and xy_err < PICK_TOLERANCE_STRICT
                        ):
                            nut_peg_clear_hold += 1
                        else:
                            nut_peg_clear_hold = 0

                        if step_count % 120 == 0:
                            print(
                                "[NUT_PEG_CLEAR] "
                                f"ee_lift={actual_lift*1000:.1f}mm, "
                                f"nut_lift={actual_nut_lift*1000:.1f}mm/"
                                f"{NUT_PEG_CLEARANCE_Z*1000:.0f}mm "
                                f"coupling_drift="
                                f"{coupling_error*1000:.1f}mm "
                                f"target_z={nut_peg_clear_pos[2]:.4f} "
                                f"command_z={nut_peg_clear_command[2]:.4f} "
                                f"xy_drift={xy_err*1000:.1f}mm "
                                f"hold={nut_peg_clear_hold}/"
                                f"{NUT_PEG_CLEAR_HOLD_STEPS}"
                            )

                        if nut_peg_clear_hold >= NUT_PEG_CLEAR_HOLD_STEPS:
                            print(
                                "[OK] peg 수직 이탈 완료 "
                                f"(EE 상승={actual_lift*1000:.1f}mm, "
                                "너트 상승="
                                f"{actual_nut_lift*1000:.1f}mm, "
                                "결합 drift="
                                f"{coupling_error*1000:.1f}mm, "
                                f"XY drift={xy_err*1000:.1f}mm) -> "
                                f"안전 상공({NUT_APPROACH_Z}m) 상승"
                            )
                            nut_peg_clear_xy_lead = np.zeros(
                                2,
                                dtype=float,
                            )
                            nut_peg_clear_z_lead = 0.0
                            phase = "NUT_LIFT"
                            step_count = 0

            # [14단계] peg 이탈 완료 후 안전 상공 상승
            elif phase == "NUT_LIFT":
                publish_progress("NUT_LIFTING", 80.0)
                cur_pos = np.asarray(
                    world_xf(
                        stage,
                        f"{M0609_PATH}/{EE_LINK_NAME}",
                    ).ExtractTranslation(),
                    dtype=float,
                )
                if nut_lift_command_z is None:
                    nut_lift_command_z = float(cur_pos[2])
                nut_lift_command_z = rate_limited_z_target(
                    nut_approach_pos[2],
                    nut_lift_command_z,
                    max_step=NUT_LIFT_COMMAND_MAX_STEP_M,
                )
                nut_lift_command = nut_approach_pos.copy()
                nut_lift_command[2] = nut_lift_command_z
                actions = arm_controller.forward(
                    target_end_effector_position=nut_lift_command,
                    target_end_effector_orientation=quat_nut,
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(
                    ArticulationAction(
                        joint_positions=GRIPPER_CLOSE_NUT
                    )
                )

                current_err = math.dist(cur_pos, tuple(nut_approach_pos))
                nut_xf_cur, nut_label = resolve_nut_assets(
                    nut_index,
                    nut1_xform,
                    nut2_xform,
                    extra_nut_xforms,
                )
                if (
                    nut_xf_cur is None
                    or nut_ee_reference_offset is None
                ):
                    print(
                        f"[ERROR] {nut_label} 안전 상공 상승 중 "
                        "Xform/EE 결합 기준이 없습니다"
                    )
                    restore_active_nut_to_tray_glue(
                        "missing lift coupling state"
                    )
                    publish_status(
                        "FAILURE:NUT_GRIP_COUPLING_STATE_MISSING"
                    )
                    phase = "IDLE"
                else:
                    nut_now_pos, _ = nut_xf_cur.get_world_pose()
                    nut_now_pos = np.asarray(
                        nut_now_pos,
                        dtype=float,
                    )
                    coupling_error = nut_ee_coupling_error(
                        cur_pos,
                        nut_now_pos,
                        nut_ee_reference_offset,
                    )
                    if (
                        coupling_error
                        > NUT_EE_COUPLING_TOLERANCE_M
                    ):
                        print(
                            f"[ERROR] {nut_label} 안전 상공 상승 중 "
                            "EE-relative drift="
                            f"{coupling_error*1000:.1f}mm"
                        )
                        restore_active_nut_to_tray_glue(
                            "coupling lost during lift"
                        )
                        publish_status(
                            "FAILURE:NUT_GRIP_COUPLING_LOST"
                        )
                        phase = "IDLE"
                    elif (
                        current_err < PICK_TOLERANCE_STRICT
                        or (
                            current_err
                            < PICK_TOLERANCE_LOOSE_VAL
                            and step_count > MAX_STUCK_STEPS
                        )
                    ):
                        print(
                            f"\n★ [PICK_{nut_label} SUCCESS] 너트 파지 및 "
                            "상승 완수! "
                            f"(결합 drift={coupling_error*1000:.1f}mm)"
                        )
                        print(
                            "[진단] 너트 실측 위치="
                            f"{tuple(round(v,4) for v in nut_now_pos)} "
                            "| EE 위치="
                            f"{tuple(round(v,4) for v in cur_pos)}"
                        )

                        # 이 시점부터는 PICK가 확정되어 후속 ASSEMBLE이
                        # dynamics/contact 상태를 이어받는다. 이후 새 task
                        # conflict가 이 너트를 tray로 되돌리면 안 된다.
                        nut_pick_active_array_index = None
                        nut_ee_reference_offset = None
                        nut_ee_previous_offset = None
                        nut_lift_command_z = None
                        nut_release_timer = 0
                        publish_progress("NUT_PICK_COMPLETE", 100.0)
                        publish_status("SUCCESS")
                        phase = "IDLE"

            # [15단계] 너트 파지 안전 고도에서 볼트 XY로 수평 이동.
            # 큰 XY 이동과 0.9m -> 0.6m 하강을 한 번에 명령하면 RMPFlow가
            # 목표 약 2cm 앞에서 수렴하지 못할 수 있어 두 축 이동을 분리한다.
            elif phase == "MOVE_TO_BOLT_NUT":
                publish_progress("MOVE_TO_BOLT", 20.0)
                cur_pos = np.asarray(
                    world_xf(
                        stage,
                        f"{M0609_PATH}/{EE_LINK_NAME}",
                    ).ExtractTranslation(),
                    dtype=float,
                )
                bolt_travel_command, nut_xy_lead = lead_xy_target(
                    bolt_travel_pos,
                    cur_pos,
                    nut_xy_lead,
                )
                actions = arm_controller.forward(
                    target_end_effector_position=bolt_travel_command,
                    target_end_effector_orientation=quat_nut,
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                current_err = math.dist(cur_pos, tuple(bolt_travel_pos))
                if step_count % 120 == 0:
                    lead_mm = 1000.0 * np.linalg.norm(
                        bolt_travel_command[:2] - bolt_travel_pos[:2]
                    )
                    print(
                        f"[NUT_BOLT_XY] actual=({cur_pos[0]:.4f}, "
                        f"{cur_pos[1]:.4f}, {cur_pos[2]:.4f}) | "
                        f"err={current_err * 1000.0:.1f}mm | "
                        f"command lead={lead_mm:.1f}mm"
                    )

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    nut_label = resolve_nut_assets(nut_index, nut1_xform, nut2_xform)[1]
                    print(f"[OK] 볼트 {nut_slot}번 안전 상공 XY 도착! ({nut_label}) "
                          f"-> Z={BOLT_APPROACH_Z:.3f}m 수직 하강 시작")
                    nut_xy_lead = np.zeros(2, dtype=float)
                    phase = "MOVE_TO_BOLT_NUT_VERTICAL"
                    step_count = 0

            # [15.5단계] 볼트 XY를 고정한 채 체결 접근 고도로 수직 하강
            elif phase == "MOVE_TO_BOLT_NUT_VERTICAL":
                publish_progress("MOVE_TO_BOLT", 30.0)
                cur_pos = np.asarray(
                    world_xf(
                        stage,
                        f"{M0609_PATH}/{EE_LINK_NAME}",
                    ).ExtractTranslation(),
                    dtype=float,
                )
                bolt_approach_command, nut_xy_lead = lead_xy_target(
                    bolt_approach_pos,
                    cur_pos,
                    nut_xy_lead,
                )
                actions = arm_controller.forward(
                    target_end_effector_position=bolt_approach_command,
                    target_end_effector_orientation=quat_nut,
                )
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                current_err = math.dist(cur_pos, tuple(bolt_approach_pos))
                if step_count % 120 == 0:
                    lead_mm = 1000.0 * np.linalg.norm(
                        bolt_approach_command[:2] - bolt_approach_pos[:2]
                    )
                    print(
                        f"[NUT_BOLT_VERTICAL] actual=({cur_pos[0]:.4f}, "
                        f"{cur_pos[1]:.4f}, {cur_pos[2]:.4f}) | "
                        f"err={current_err * 1000.0:.1f}mm | "
                        f"command lead={lead_mm:.1f}mm"
                    )

                if current_err < PICK_TOLERANCE_STRICT or (current_err < PICK_TOLERANCE_LOOSE_VAL and step_count > MAX_STUCK_STEPS):
                    nut_label = resolve_nut_assets(nut_index, nut1_xform, nut2_xform)[1]
                    print(f"[OK] 볼트 {nut_slot}번 체결 상공 도착! ({nut_label}) "
                          "-> 착좌 하강 시작")
                    phase = "NUT_DESCEND_TO_BOLT"
                    step_count = 0
                    descend_target_z = BOLT_APPROACH_Z

            # [16단계] 볼트 착좌 하강
            elif phase == "NUT_DESCEND_TO_BOLT":
                publish_progress("NUT_DESCEND_TO_BOLT", 40.0)
                descend_target_z = max(descend_target_z - INSERT_SPEED, bolt_touch_pos[2])
                step_target_pos = np.array([bolt_touch_pos[0], bolt_touch_pos[1], descend_target_z])

                actions = arm_controller.forward(target_end_effector_position=step_target_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()

                if abs(cur_pos[2] - bolt_touch_pos[2]) < PICK_TOLERANCE_LOOSE_VAL or descend_target_z <= bolt_touch_pos[2]:
                    ee_now_pos, ee_now_quat = robot.end_effector.get_world_pose()
                    screw_start_quat = np.asarray(ee_now_quat).copy()
                    screw_seat_ee_pos = np.asarray(ee_now_pos).copy()

                    screw_sub = "rotate"
                    screw_pass_idx = 0
                    screw_pass_theta = 0.0
                    stuck_counter = 0
                    prev_ee_z = ee_now_pos[2]

                    nut_label = resolve_nut_assets(nut_index, nut1_xform, nut2_xform)[1]
                    print(f"[OK] 볼트 {nut_slot}번 착좌 완료 ({nut_label})! -> Screwing 시작")
                    phase = "NUT_SCREW"
                    step_count = 0

            # [17단계] Screwing
            elif phase == "NUT_SCREW":
                publish_progress("NUT_SCREWING", 70.0)
                if screw_sub == "rotate":
                    screw_pass_theta = min(screw_pass_theta + SCREW_OMEGA_DEG_S * PHYSICS_DT, SCREW_TURNS_DEG)
                    pass_done = (screw_pass_theta >= SCREW_TURNS_DEG)

                    total_deg = screw_pass_idx * SCREW_TURNS_DEG + screw_pass_theta
                    depth_m = min((total_deg / 360.0) * NUT_PITCH_M, ENGAGE_LEN)

                    if screw_pass_idx > 0:
                        regrasp_extra = REGRASP_Z_OFFSET * (1.0 - screw_pass_theta / SCREW_TURNS_DEG)
                    else:
                        regrasp_extra = 0.0

                    target_pos = screw_seat_ee_pos.copy()
                    target_pos[2] = screw_seat_ee_pos[2] - depth_m + regrasp_extra
                    target_quat = yaw_rotated_quat(screw_start_quat, screw_pass_theta)

                    actions = arm_controller.forward(target_end_effector_position=target_pos, target_end_effector_orientation=target_quat)
                    robot.apply_action(actions)
                    robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_CLOSE_NUT))

                    # 실시간 토크 및 Stuck 감지
                    cur_ee_pos, _ = robot.end_effector.get_world_pose()
                    z_movement = abs(prev_ee_z - cur_ee_pos[2])
                    prev_ee_z = cur_ee_pos[2]

                    joint_efforts = robot.get_measured_joint_efforts()
                    curr_torque = abs(joint_efforts[-1]) if joint_efforts is not None and len(joint_efforts) > 0 else 0.0

                    if cur_ee_pos[2] <= TCP_FORCE_CHECK_Z:
                        if z_movement < STUCK_Z_DELTA_THRESH and depth_m > 0.003:
                            stuck_counter += 1
                        else:
                            stuck_counter = max(0, stuck_counter - 1)
                        is_seated_by_torque = (curr_torque > TORQUE_THRESHOLD) or (stuck_counter >= STUCK_STEP_LIMIT)
                    else:
                        is_seated_by_torque = False

                    if step_count % 20 == 0:
                        print(f"  [NUT{nut_index} SCREW] Pass {screw_pass_idx+1}/{1+REGRASP_CYCLES} | Theta: {screw_pass_theta:.1f}° | 깊이: {depth_m*1000:.2f}mm / 목표 {ENGAGE_LEN*1000:.1f}mm | TCP Z: {cur_ee_pos[2]:.4f}m | 토크: {curr_torque:.1f}Nm")

                    if pass_done or is_seated_by_torque:
                        if is_seated_by_torque:
                            print(f"  [체결 감지] 너트 {nut_index}번 완착(Seating) 감지! (TCP Z: {cur_ee_pos[2]:.4f}m, 토크: {curr_torque:.1f}Nm) -> Screwing 조기 종료 및 그리퍼 해제")

                        if depth_m >= ENGAGE_LEN or screw_pass_idx >= REGRASP_CYCLES or is_seated_by_torque:
                            robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))
                            print(f"\n[OK] 너트 {nut_index}번 체결 완료! -> 꼬인 방향 유지한 채 수직 상승 시작")
                            phase = "NUT_RETRACT_LIFT"
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

                    regrasp_target_pos = screw_pass_end_pos + np.array([0.0, 0.0, REGRASP_Z_OFFSET])
                    actions = arm_controller.forward(target_end_effector_position=regrasp_target_pos, target_end_effector_orientation=screw_start_quat)
                    robot.apply_action(actions)

                    if rf >= 1.0:
                        screw_pass_idx += 1
                        screw_pass_theta = 0.0
                        stuck_counter = 0
                        screw_sub = "rotate"

            # [18단계] 체결 후 수직 상승 (꼬인 방향 유지)
            elif phase == "NUT_RETRACT_LIFT":
                publish_progress("NUT_RETRACT_LIFT", 90.0)
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
                    phase = "NUT_RETRACT_UNWIND"
                    screw_unwind_deg = screw_pass_theta
                    step_count = 0

            # [19단계] 상공에서 6번 조인트 0도로 되감기
            elif phase == "NUT_RETRACT_UNWIND":
                publish_progress("NUT_RETRACT_UNWIND", 95.0)
                screw_unwind_deg = max(screw_unwind_deg - SCREW_OMEGA_DEG_S * PHYSICS_DT, 0.0)
                ee_now_pos, _ = robot.end_effector.get_world_pose()
                lift_target_pos = np.array([ee_now_pos[0], ee_now_pos[1], NUT_APPROACH_Z])

                target_quat = yaw_rotated_quat(screw_start_quat, screw_unwind_deg)
                actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=target_quat)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                if screw_unwind_deg <= 0.0:
                    print(
                        "[OK] 6번 조인트 되감기 명령 완료! -> 실제 "
                        "기본 방향(quat_nut) 수렴 대기"
                    )
                    nut_pose_settle_count = 0
                    phase = "NUT_RETRACT_ROTATE"
                    step_count = 0

            # [20단계] 기본 오리엔테이션 정렬
            elif phase == "NUT_RETRACT_ROTATE":
                publish_progress("NUT_RETRACT_ROTATE", 98.0)
                ee_now_pos, ee_now_quat = robot.end_effector.get_world_pose()
                lift_target_pos = np.array([ee_now_pos[0], ee_now_pos[1], NUT_APPROACH_Z])

                actions = arm_controller.forward(target_end_effector_position=lift_target_pos, target_end_effector_orientation=quat_nut)
                robot.apply_action(actions)
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                cur_pos = world_xf(stage, f"{M0609_PATH}/{EE_LINK_NAME}").ExtractTranslation()
                z_err = abs(float(cur_pos[2]) - NUT_APPROACH_Z)
                orientation_err = _quat_angular_error(
                    ee_now_quat,
                    quat_nut,
                )
                nut_pose_settle_count, pose_settled = (
                    consecutive_pose_settle(
                        z_err,
                        orientation_err,
                        nut_pose_settle_count,
                        position_tolerance_m=NUT_POSE_POSITION_TOL_M,
                        orientation_tolerance_rad=(
                            NUT_POSE_ORIENTATION_TOL_RAD
                        ),
                        required_steps=NUT_POSE_SETTLE_STEPS,
                    )
                )

                if step_count % 30 == 0:
                    print(
                        "[NUT_RETRACT_ROTATE] "
                        f"z_err={z_err * 1000:.1f}mm, "
                        f"orientation_err="
                        f"{math.degrees(orientation_err):.2f}deg, "
                        f"stable={nut_pose_settle_count}/"
                        f"{NUT_POSE_SETTLE_STEPS}"
                    )

                if pose_settled:
                    nut_label = resolve_nut_assets(nut_index, nut1_xform, nut2_xform)[1]
                    print(f"\n[OK] {nut_label} 상공 이탈 및 회전 정렬 완료!")
                    
                    # 현재 station의 두 번째 슬롯까지 끝났을 때 초기 관절각으로
                    # 복귀한다. 물리 nut index는 station4/5에서 4/6이므로
                    # nut_index == 2로 판정하면 다음 주행 때 팔이 뻗은 채 남는다.
                    station_nut_pair = STATION_NUT_INDICES.get(
                        current_station,
                        (1, 2),
                    )
                    if nut_index == station_nut_pair[1]:
                        print(
                            " -> 🚀 현재 station 두 번째 너트 체결 완료. "
                            "초기 관절 각도 복귀 시작..."
                        )
                        phase = "RETURN_HOME_JOINTS"
                        step_count = 0
                    else:
                        print(f"★ [ASSEMBLE_{nut_label} SUCCESS] 너트 1번 체결 완료!")
                        publish_progress("ASSEMBLE_NUT_COMPLETE", 100.0)
                        publish_status("SUCCESS")
                        phase = "IDLE"

            # [21단계 - 신규] 너트 2번 완료 후 초기 관절 각도복귀 [0, 0, 90°, 0, 90°, 0]
            elif phase == "RETURN_HOME_JOINTS":
                publish_progress("RETURNING_HOME_JOINTS", 99.0)

                arm_joint_names = [
                    "joint_1", "joint_2", "joint_3",
                    "joint_4", "joint_5", "joint_6"
                ]
                arm_dof_indices = [robot.get_dof_index(name) for name in arm_joint_names]

                all_joints = robot.get_joint_positions()
                cur_arm_joints = all_joints[arm_dof_indices]

                max_step = ARM_INIT_MAX_JOINT_SPEED * PHYSICS_DT
                delta = TARGET_INIT_JOINTS - cur_arm_joints
                ramped_target = cur_arm_joints + np.clip(delta, -max_step, max_step)

                robot.apply_action(
                    ArticulationAction(
                        joint_positions=ramped_target,
                        joint_indices=arm_dof_indices
                    )
                )
                robot.gripper.apply_action(ArticulationAction(joint_positions=GRIPPER_OPEN))

                joint_err = np.linalg.norm(cur_arm_joints - TARGET_INIT_JOINTS)

                if joint_err < JOINT_TOLERANCE:
                    print("\n★ [ALL PROCESS COMPLETE & HOME RETURNED] 모든 너트 체결 및 초기 관절 각도 복귀 완수!")
                    publish_progress("ASSEMBLE_NUT_COMPLETE", 100.0)
                    publish_status("SUCCESS")
                    phase = "IDLE"
                elif step_count > MAX_STUCK_STEPS:
                    print(
                        "\n[ERROR] 초기 관절 자세 복귀 실패 "
                        f"(joint_err={joint_err:.4f}rad)"
                    )
                    publish_status("FAILURE:RETURN_HOME_JOINTS_TIMEOUT")
                    phase = "IDLE"

            # Phase 타임아웃 예외 처리
            if (
                phase not in {
                    "WAIT_BUSBAR_CAMERA_LATCH",
                    "WAIT_BUSBAR_TARGET",
                    "FINE_ALIGNMENT",
                }
                and step_count > MAX_STUCK_STEPS
            ):
                print(f"\n[ERROR] Phase '{phase}' 진행 시간 초과 - 실패 처리")
                publish_status("FAILURE:TIMEOUT")
                phase = "DONE"

            step_count += 1

        was_playing = playing

    shutdown_runtime()
    atexit.unregister(shutdown_runtime)


if __name__ == "__main__":
    main()

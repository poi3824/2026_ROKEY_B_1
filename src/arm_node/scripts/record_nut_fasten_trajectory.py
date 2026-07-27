"""record_nut_fasten_trajectory.py -- World0123.usd 안에서 11번 파일
(isaacpjt/M0609/11_nut_screw_kinematic.py) 방식의 kinematic pose-glue
파지+체결 시퀀스를 실행하면서, 매 스텝 관절값을 기록해 JSON으로 저장한다.

arm_node.py는 실시간 IK 없이 이 기록된 궤적을 그대로 재생(publish)한다
(대상 nut1/peg_0 -> bolt_2 위치가 고정돼 있으므로 오프라인 기록 재생 방식으로 충분).

기록은 두 구간으로 나뉜다:
  APPROACH: 시작 ~ SEAT 판정 직전까지(너트 픽 + 볼트 위 정렬)
  FASTEN  : SEAT ~ SCREW ~ SETTLE ~ JUDGE ~ HOME 복귀까지(체결 + 복귀)
이 구분은 fms_interfaces/FastenCommand.msg 의 APPROACH/FASTEN 명령에 대응한다.

실행:
  /home/rokey/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh \
      /home/rokey/EV_combine/src/arm_node/scripts/record_nut_fasten_trajectory.py
"""
import os
from isaacsim import SimulationApp

_HEADLESS = os.environ.get("BOLT_HEADLESS", "1") == "1"
simulation_app = SimulationApp({"headless": _HEADLESS})

import sys
import json
from pathlib import Path
import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, Gf

from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator

# rmpflow 컨트롤러(PickPlaceController/RMPFlowController)는 아직 src/ 로 옮기지 않고
# 기존 M0609 개발 폴더(오프라인 기록 스크립트 전용 의존성이라 이 경로에 남겨둠)를 그대로 재사용한다.
_M0609_DIR = Path("/home/rokey/EV_combine/isaacpjt/M0609")
sys.path.insert(0, str(_M0609_DIR / "rmpflow"))
from m0609_pick_place_controller import PickPlaceController  # noqa: E402
from m0609_rmpflow_controller import RMPFlowController  # noqa: E402

WORLD_USD = "/home/rokey/EV_combine/src/Collected_World0123/World0123.usd"
OUT_JSON = Path("/home/rokey/EV_combine/src/arm_node/arm_node/data/nut_fasten_trajectory.json")
LOG_PATH = Path("/home/rokey/EV_combine/src/arm_node/scripts/record_result.txt")

_log_f = open(LOG_PATH, "w")


def log(*parts):
    _log_f.write(" ".join(str(p) for p in parts) + "\n")
    _log_f.flush()


# ══════════════════════════════════════════════════════════════════════════
#  [A] 로봇/그리퍼 상수 (11번 파일과 동일 -- 같은 로봇+RG2 조합, 검증된 값 재사용)
# ══════════════════════════════════════════════════════════════════════════
ROBOT_URDF_PATH = str(_M0609_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
ROBOT_DESC_PATH = str(_M0609_DIR / "rmpflow/m0609_description.yaml")
RMPFLOW_CFG_PATH = str(_M0609_DIR / "rmpflow/m0609_rmpflow_common.yaml")
# World0123.usd 에서는 m0609 가 /World/m0609/FixedJoint 로 Nova_Carter/chassis_link 에
# 용접돼 있어 둘이 하나의 PhysX 아티큘레이션이다 (ArticulationRootAPI 도 chassis_link 에
# 있음, m0609 자체엔 없음) -- 11번 파일처럼 m0609 를 단독 아티큘레이션으로 다루면
# physics_sim_view 가 None 이 되어 initialize() 가 깨진다(is_homogeneous AttributeError).
ARM_PRIM_PATH = "/World/m0609"                       # 팔 조인트 드라이브가 실제 위치한 USD 서브트리
ARTICULATION_ROOT_PATH = "/World/Nova_Carter/chassis_link"  # 진짜 PhysX 아티큘레이션 루트(AMR+팔 통합)
EE_LINK_NAME = "link_6"
GRIPPER_JOINTS = ["finger_joint", "right_inner_knuckle_joint"]

PHYSICS_DT = 1.0 / 60.0
ART_POS_ITERS, ART_VEL_ITERS = 192, 1

GRIP_CLOSE_POSITION = 0.96
GRIP_CLOSE_RAMP_STEPS = 75
GRIPPER_DRIVE_STIFFNESS = 45.0 / 57.29578
GRIPPER_DRIVE_DAMPING = 5.0 / 57.29578
GRIPPER_HOLD_STIFFNESS = 180.0 / 57.29578
GRIPPER_HOLD_DAMPING = 35.0 / 57.29578
GRIPPER_HOLD_RAMP_STEPS = 50

EVENTS_DT = [0.011, 0.006, 0.05, 1.0 / 150, 0.01, 0.013, 0.003, 1.0, 0.011, 0.08]
PICK_HOVER_HEIGHT = 0.15
HOME_LIFT_Z = 0.30
EE_OFFSET = np.array([0.0, 0.0, 0.185])
NUT_REST_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])

SCREW_HOVER_CLEAR = 0.001
ENGAGE_LEN = 0.020
ENGAGE_XY_TOL_M = 0.006
ENGAGE_TILT_DEG = 8.0
ENGAGE_GAP_M = 0.003
SCREW_TURNS_DEG = 270.0
REGRASP_CYCLES = 2
SCREW_OMEGA_DEG_S = 60.0
SCREW_DIRECTION = 1.0
TOTAL_REV = (SCREW_TURNS_DEG / 360.0) * (1 + REGRASP_CYCLES)
NUT_PITCH_M = ENGAGE_LEN / TOTAL_REV
NUT_GRASP_Z_LOCAL_OFFSET = 0.03  # nut 로컬원점(바닥)에서 그립점까지 (11번과 동일)

MAX_STEPS = 8000


# ══════════════════════════════════════════════════════════════════════════
#  [B] 헬퍼 (11번 파일과 동일)
# ══════════════════════════════════════════════════════════════════════════
def find_prim_path(root_path, name):
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def set_all_drives(root_path):
    from pxr import UsdPhysics
    stage = omni.usd.get_context().get_stage()
    GRIPPER_MECH_JOINTS = {
        "finger_joint", "right_inner_knuckle_joint",
        "left_inner_knuckle_to_finger_joint", "right_inner_knuckle_to_finger_joint",
        "left_inner_finger_joint", "right_inner_finger_joint",
    }
    n = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if prim.GetName() in GRIPPER_MECH_JOINTS:
            continue
        for dt in ("angular", "linear"):
            drive = UsdPhysics.DriveAPI.Get(prim, dt)
            if drive:
                drive.GetStiffnessAttr().Set(1.0e8)
                drive.GetDampingAttr().Set(1.0e6)
                drive.GetMaxForceAttr().Set(1.0e8)
                n += 1
    log(f"  [OK] 팔 드라이브 {n}개 설정")


_GRIPPER_DRIVE_ATTR_CACHE = []


def set_gripper_drives(root_path, stiffness, damping, label="설정"):
    from pxr import UsdPhysics
    GRIPPER_MECH_JOINTS = {
        "finger_joint", "right_inner_knuckle_joint",
        "left_inner_knuckle_to_finger_joint", "right_inner_knuckle_to_finger_joint",
        "left_inner_finger_joint", "right_inner_finger_joint",
    }
    global _GRIPPER_DRIVE_ATTR_CACHE
    if not _GRIPPER_DRIVE_ATTR_CACHE:
        stage = omni.usd.get_context().get_stage()
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
            if prim.GetName() not in GRIPPER_MECH_JOINTS:
                continue
            for dt in ("angular", "linear"):
                drive = UsdPhysics.DriveAPI.Get(prim, dt)
                if drive:
                    _GRIPPER_DRIVE_ATTR_CACHE.append((drive.GetStiffnessAttr(), drive.GetDampingAttr()))
    for stiff_attr, damp_attr in _GRIPPER_DRIVE_ATTR_CACHE:
        stiff_attr.Set(stiffness)
        damp_attr.Set(damping)


def solver_iters_only(root_path):
    from pxr import UsdPhysics, PhysxSchema
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    art_prim = None
    for p in Usd.PrimRange(root):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_prim = p
            break
    art_prim = art_prim or root
    ap = PhysxSchema.PhysxArticulationAPI.Apply(art_prim)
    ap.CreateSolverPositionIterationCountAttr(ART_POS_ITERS)
    ap.CreateSolverVelocityIterationCountAttr(ART_VEL_ITERS)
    ap.CreateEnabledSelfCollisionsAttr(False)
    log(f"  [OK] 솔버 반복수: pos={ART_POS_ITERS} vel={ART_VEL_ITERS}")


def initialize_robot(robot, world):
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )


def axis_tilt_deg(quat_wxyz):
    w, x, y, z = [float(v) for v in quat_wxyz]
    rot = Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    local_z = rot.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
    cos_a = float(np.clip(local_z[2], -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def yaw_rotated_quat(base_wxyz, delta_deg):
    base_q = Gf.Quatd(float(base_wxyz[0]), Gf.Vec3d(float(base_wxyz[1]), float(base_wxyz[2]), float(base_wxyz[3])))
    base_rot = Gf.Rotation(base_q)
    extra_rot = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), float(delta_deg))
    combined = extra_rot * base_rot
    q = combined.GetQuat()
    return np.array([q.GetReal(), *q.GetImaginary()])


def measured_yaw_delta_deg(base_wxyz, now_wxyz):
    base_q = Gf.Quatd(float(base_wxyz[0]), Gf.Vec3d(float(base_wxyz[1]), float(base_wxyz[2]), float(base_wxyz[3])))
    now_q = Gf.Quatd(float(now_wxyz[0]), Gf.Vec3d(float(now_wxyz[1]), float(now_wxyz[2]), float(now_wxyz[3])))
    delta_q = now_q * base_q.GetInverse()
    w = delta_q.GetReal()
    v = delta_q.GetImaginary()
    return float(np.degrees(2.0 * np.arctan2(v[2], w)))


# ══════════════════════════════════════════════════════════════════════════
#  [C] 메인
# ══════════════════════════════════════════════════════════════════════════
def main():
    context = omni.usd.get_context()
    context.open_stage(WORLD_USD)
    for _ in range(20):
        simulation_app.update()

    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=1.0 / 60.0)

    stage = omni.usd.get_context().get_stage()
    nut_prim = stage.GetPrimAtPath("/World/nut1")
    bolt_prim = stage.GetPrimAtPath("/World/battery_pack3/_2_8V60Ah_BT/bolt_2")

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    nut_bbox = cache.ComputeWorldBound(nut_prim).ComputeAlignedRange()
    bolt_bbox = cache.ComputeWorldBound(bolt_prim).ComputeAlignedRange()

    nut_bottom_z = float(nut_bbox.GetMin()[2])
    nut_cx = float((nut_bbox.GetMin()[0] + nut_bbox.GetMax()[0]) / 2.0)
    nut_cy = float((nut_bbox.GetMin()[1] + nut_bbox.GetMax()[1]) / 2.0)
    nut_height = float(nut_bbox.GetMax()[2] - nut_bbox.GetMin()[2])

    bolt_tip_z = float(bolt_bbox.GetMax()[2])
    bolt_cx = float((bolt_bbox.GetMin()[0] + bolt_bbox.GetMax()[0]) / 2.0)
    bolt_cy = float((bolt_bbox.GetMin()[1] + bolt_bbox.GetMax()[1]) / 2.0)

    NUT_PICK_XY = np.array([nut_cx, nut_cy])
    NUT_REST_ROOT_Z = nut_bottom_z
    BOLT_XY = np.array([bolt_cx, bolt_cy])
    BOLT_TIP_Z = bolt_tip_z
    NUT_GRASP_Z_LOCAL = nut_height + NUT_GRASP_Z_LOCAL_OFFSET

    PICK_POS = np.array([NUT_PICK_XY[0], NUT_PICK_XY[1], NUT_REST_ROOT_Z + NUT_GRASP_Z_LOCAL])
    NUT_ALIGN_ROOT_Z = BOLT_TIP_Z + SCREW_HOVER_CLEAR
    PLACE_POS = np.array([BOLT_XY[0], BOLT_XY[1], NUT_ALIGN_ROOT_Z + NUT_GRASP_Z_LOCAL])

    log(f"[측정] nut1 pick={np.round(PICK_POS,4)} (bottom_z={nut_bottom_z:.4f}, height={nut_height*1000:.2f}mm)")
    log(f"[측정] bolt_2 place={np.round(PLACE_POS,4)} (tip_z={bolt_tip_z:.4f})")

    nut_xform = SingleXFormPrim("/World/nut1", name="nut1")

    log("[1] 물리 설정")
    set_all_drives(ARM_PRIM_PATH)
    set_gripper_drives(ARM_PRIM_PATH, GRIPPER_DRIVE_STIFFNESS, GRIPPER_DRIVE_DAMPING)
    solver_iters_only(ARTICULATION_ROOT_PATH)

    ee_path = find_prim_path(ARM_PRIM_PATH, EE_LINK_NAME)
    if ee_path is None:
        raise RuntimeError(f"'{EE_LINK_NAME}' 링크를 찾을 수 없음")
    gripper = ParallelGripper(
        end_effector_prim_path=ee_path, joint_prim_names=GRIPPER_JOINTS,
        joint_opened_positions=np.array([0.0, 0.0]),
        joint_closed_positions=np.array([GRIP_CLOSE_POSITION, GRIP_CLOSE_POSITION]),
        action_deltas=None,
    )
    robot = world.scene.add(SingleManipulator(
        prim_path=ARTICULATION_ROOT_PATH, name="m0609_robot",
        end_effector_prim_path=ee_path, gripper=gripper,
    ))

    world.reset()
    initialize_robot(robot, world)
    for _ in range(30):
        world.step(render=False)

    gripper_dof_indices = [robot.dof_names.index(n) for n in GRIPPER_JOINTS]
    dof_names = list(robot.dof_names)
    log(f"[2] 로봇 등록 완료. DOF={dof_names}")

    align_controller = PickPlaceController(
        name="rec_pick_place", gripper=robot.gripper, robot_articulation=robot,
        end_effector_initial_height=PICK_HOVER_HEIGHT, events_dt=EVENTS_DT,
        urdf_path=ROBOT_URDF_PATH, robot_description_path=ROBOT_DESC_PATH,
        rmpflow_config_path=RMPFLOW_CFG_PATH, end_effector_frame_name=EE_LINK_NAME,
    )
    screw_controller = RMPFlowController(
        name="rec_screw_cspace", robot_articulation=robot,
        urdf_path=ROBOT_URDF_PATH, robot_description_path=ROBOT_DESC_PATH,
        rmpflow_config_path=RMPFLOW_CFG_PATH, end_effector_frame_name=EE_LINK_NAME,
    )

    trajectory = {"approach": [], "fasten": []}

    def record(segment):
        trajectory[segment].append({
            "t": len(trajectory["approach"]) * PHYSICS_DT if segment == "approach"
                 else len(trajectory["fasten"]) * PHYSICS_DT,
            "positions": [float(v) for v in robot.get_joint_positions()],
        })

    phase = {"name": "PICK_CARRY", "reported": False}

    def glue_nut_to_ee(blend):
        ee_pos, _ = robot.end_effector.get_world_pose()
        grasp_point_pos = np.asarray(ee_pos) - EE_OFFSET
        target_pos = grasp_point_pos - np.array([0.0, 0.0, NUT_GRASP_Z_LOCAL])
        rest_pos = np.array([float(NUT_PICK_XY[0]), float(NUT_PICK_XY[1]), float(NUT_REST_ROOT_Z)])
        nut_pos = rest_pos + blend * (target_pos - rest_pos)
        nut_xform.set_world_pose(position=nut_pos, orientation=NUT_REST_ORIENTATION)

    def apply_grip_hold():
        robot.apply_action(ArticulationAction(
            joint_positions=np.array([GRIP_CLOSE_POSITION, GRIP_CLOSE_POSITION]),
            joint_indices=np.array(gripper_dof_indices)))

    step_count = 0
    while step_count < MAX_STEPS:
        step_count += 1

        if phase["name"] == "PICK_CARRY":
            ev = align_controller.get_current_event()
            if ev >= 7:
                sp, sq = robot.end_effector.get_world_pose()
                phase["start_pos"] = np.asarray(sp).copy()
                phase["start_quat"] = np.asarray(sq).copy()
                nut_pos, nut_quat = nut_xform.get_world_pose()
                seat_pos = np.array([float(BOLT_XY[0]), float(BOLT_XY[1]), float(BOLT_TIP_Z)])
                seat_quat = np.asarray(nut_quat).copy()
                nut_xform.set_world_pose(position=seat_pos, orientation=seat_quat)
                phase["seat_pos"] = seat_pos
                phase["seat_quat"] = seat_quat
                phase["seat_ee_pos"] = phase["start_pos"].copy()
                phase["pass_theta_deg"] = 0.0
                phase["pass_idx"] = 0
                phase["nut_visual_deg"] = 0.0
                phase["_prev_measured_deg"] = 0.0
                phase["name"] = "SCREW"
                phase["screw_sub"] = "rotate"
                log(f"  [PICK_CARRY 완료] step={step_count} EE={np.round(phase['start_pos'],3)}")
                continue

            action = align_controller.forward(
                picking_position=PICK_POS, placing_position=PLACE_POS,
                current_joint_positions=robot.get_joint_positions(),
                end_effector_offset=EE_OFFSET,
            )
            robot.apply_action(action)

            if ev >= 3:
                if ev == 3:
                    phase["_close_ramp_step"] = phase.get("_close_ramp_step", 0) + 1
                    ramp_frac = min(phase["_close_ramp_step"] / GRIP_CLOSE_RAMP_STEPS, 1.0)
                    grip_target = ramp_frac * GRIP_CLOSE_POSITION
                else:
                    ramp_frac = 1.0
                    grip_target = GRIP_CLOSE_POSITION
                robot.apply_action(ArticulationAction(
                    joint_positions=np.array([grip_target, grip_target]),
                    joint_indices=np.array(gripper_dof_indices)))

                if ramp_frac >= 1.0 and not phase.get("_hold_ramp_done"):
                    phase["_hold_ramp_step"] = phase.get("_hold_ramp_step", 0) + 1
                    hf = min(phase["_hold_ramp_step"] / GRIPPER_HOLD_RAMP_STEPS, 1.0)
                    cur_stiff = GRIPPER_DRIVE_STIFFNESS + hf * (GRIPPER_HOLD_STIFFNESS - GRIPPER_DRIVE_STIFFNESS)
                    cur_damp = GRIPPER_DRIVE_DAMPING + hf * (GRIPPER_HOLD_DAMPING - GRIPPER_DRIVE_DAMPING)
                    set_gripper_drives(ARM_PRIM_PATH, cur_stiff, cur_damp)
                    if hf >= 1.0:
                        phase["_hold_ramp_done"] = True

                glue_nut_to_ee(ramp_frac)

            world.step(render=False)
            record("approach")
            continue

        elif phase["name"] == "SCREW":
            sub = phase.get("screw_sub", "rotate")

            if sub == "rotate":
                ee_now_pos, ee_now_quat = robot.end_effector.get_world_pose()
                measured_deg = measured_yaw_delta_deg(phase["start_quat"], np.asarray(ee_now_quat))
                phase["nut_visual_deg"] += measured_deg - phase["_prev_measured_deg"]
                phase["_prev_measured_deg"] = measured_deg

                phase["pass_theta_deg"] = min(phase["pass_theta_deg"] + SCREW_OMEGA_DEG_S * PHYSICS_DT, SCREW_TURNS_DEG)
                pass_done = phase["pass_theta_deg"] >= SCREW_TURNS_DEG

                total_deg = phase["pass_idx"] * SCREW_TURNS_DEG + phase["pass_theta_deg"]
                depth_m = min((total_deg / 360.0) * NUT_PITCH_M, ENGAGE_LEN)

                nut_pos = phase["seat_pos"].copy()
                nut_pos[2] = phase["seat_pos"][2] - depth_m
                nut_quat = yaw_rotated_quat(phase["seat_quat"], phase["nut_visual_deg"])
                nut_xform.set_world_pose(position=nut_pos, orientation=nut_quat)

                target_pos = phase["seat_ee_pos"].copy()
                target_pos[2] = phase["seat_ee_pos"][2] - depth_m
                target_quat = yaw_rotated_quat(phase["start_quat"], SCREW_DIRECTION * phase["pass_theta_deg"])
                action = screw_controller.forward(
                    target_end_effector_position=target_pos, target_end_effector_orientation=target_quat)
                robot.apply_action(action)
                apply_grip_hold()

                world.step(render=False)
                record("fasten")

                if pass_done:
                    if depth_m >= ENGAGE_LEN or phase["pass_idx"] >= REGRASP_CYCLES:
                        phase["name"] = "SETTLE"
                        phase["settle_steps"] = 0
                        log(f"  [SCREW 완료] step={step_count} depth={depth_m*1000:.2f}mm")
                    else:
                        phase["pass_end_pos"] = target_pos.copy()
                        phase["pass_end_depth"] = depth_m
                        phase["screw_sub"] = "release"
                        phase["_release_step"] = 0
                continue

            if sub == "release":
                phase["_release_step"] = phase.get("_release_step", 0) + 1
                rf = min(phase["_release_step"] / GRIP_CLOSE_RAMP_STEPS, 1.0)
                release_target = (1.0 - rf) * GRIP_CLOSE_POSITION
                robot.apply_action(ArticulationAction(
                    joint_positions=np.array([release_target, release_target]),
                    joint_indices=np.array(gripper_dof_indices)))
                hold_quat = yaw_rotated_quat(phase["start_quat"], SCREW_DIRECTION * SCREW_TURNS_DEG)
                action = screw_controller.forward(
                    target_end_effector_position=phase["pass_end_pos"], target_end_effector_orientation=hold_quat)
                robot.apply_action(action)
                world.step(render=False)
                record("fasten")
                if rf >= 1.0:
                    phase["screw_sub"] = "unwind"
                    phase["wrist_unwind_deg"] = SCREW_TURNS_DEG
                continue

            if sub == "unwind":
                phase["wrist_unwind_deg"] = max(phase["wrist_unwind_deg"] - SCREW_OMEGA_DEG_S * PHYSICS_DT, 0.0)
                unwind_done = phase["wrist_unwind_deg"] <= 0.0
                target_quat = yaw_rotated_quat(phase["start_quat"], SCREW_DIRECTION * phase["wrist_unwind_deg"])
                action = screw_controller.forward(
                    target_end_effector_position=phase["pass_end_pos"], target_end_effector_orientation=target_quat)
                robot.apply_action(action)
                robot.apply_action(ArticulationAction(
                    joint_positions=np.array([0.0, 0.0]), joint_indices=np.array(gripper_dof_indices)))
                world.step(render=False)
                record("fasten")
                if unwind_done:
                    phase["screw_sub"] = "regrasp"
                    phase["_regrasp_step"] = 0
                continue

            if sub == "regrasp":
                phase["_regrasp_step"] = phase.get("_regrasp_step", 0) + 1
                rf = min(phase["_regrasp_step"] / GRIP_CLOSE_RAMP_STEPS, 1.0)
                grip_target = rf * GRIP_CLOSE_POSITION
                robot.apply_action(ArticulationAction(
                    joint_positions=np.array([grip_target, grip_target]), joint_indices=np.array(gripper_dof_indices)))
                action = screw_controller.forward(
                    target_end_effector_position=phase["pass_end_pos"], target_end_effector_orientation=phase["start_quat"])
                robot.apply_action(action)
                world.step(render=False)
                record("fasten")
                if rf >= 1.0:
                    phase["pass_idx"] += 1
                    phase["pass_theta_deg"] = 0.0
                    phase["seat_ee_pos"] = phase["pass_end_pos"].copy()
                    phase["screw_sub"] = "rotate"
                    ee_now_pos, ee_now_quat = robot.end_effector.get_world_pose()
                    phase["_prev_measured_deg"] = measured_yaw_delta_deg(phase["start_quat"], np.asarray(ee_now_quat))
                continue

        elif phase["name"] == "SETTLE":
            apply_grip_hold()
            world.step(render=False)
            record("fasten")
            phase["settle_steps"] = phase.get("settle_steps", 0) + 1
            if phase["settle_steps"] >= 20:
                phase["name"] = "JUDGE"
            continue

        elif phase["name"] == "JUDGE" and not phase["reported"]:
            pos, quat = nut_xform.get_world_pose()
            pos = np.asarray(pos); quat = np.asarray(quat)
            depth_mm = (float(phase["seat_pos"][2]) - float(pos[2])) * 1000.0
            xy_err = float(np.linalg.norm(pos[:2] - BOLT_XY))
            tilt = axis_tilt_deg(quat)
            success = (depth_mm >= 0.6 * ENGAGE_LEN * 1000.0) and (xy_err < 0.005) and (tilt < 5.0)
            log(f"[결과] depth={depth_mm:.2f}mm xy_err={xy_err*1000:.1f}mm tilt={tilt:.1f}deg success={success}")
            phase["reported"] = True
            eep, _ = robot.end_effector.get_world_pose()
            phase["home_base_pos"] = np.asarray(eep).copy()
            phase["name"] = "HOME"
            continue

        elif phase["name"] == "HOME":
            phase["_home_step"] = phase.get("_home_step", 0) + 1
            if not phase.get("_home_release_done"):
                rf = min(phase["_home_step"] / GRIP_CLOSE_RAMP_STEPS, 1.0)
                release_target = (1.0 - rf) * GRIP_CLOSE_POSITION
                robot.apply_action(ArticulationAction(
                    joint_positions=np.array([release_target, release_target]), joint_indices=np.array(gripper_dof_indices)))
                action = screw_controller.forward(
                    target_end_effector_position=phase["home_base_pos"], target_end_effector_orientation=phase["start_quat"])
                robot.apply_action(action)
                world.step(render=False)
                record("fasten")
                if rf >= 1.0:
                    phase["_home_release_done"] = True
                    phase["_home_step"] = 0
                continue
            target_pos = phase["home_base_pos"].copy()
            target_pos[2] = HOME_LIFT_Z
            action = screw_controller.forward(
                target_end_effector_position=target_pos, target_end_effector_orientation=phase["start_quat"])
            robot.apply_action(action)
            world.step(render=False)
            record("fasten")
            if phase["_home_step"] >= 150:
                log(f"  [HOME] 복귀 완료. 총 스텝={step_count}")
                break

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "dof_names": dof_names,
            "gripper_dof_indices": gripper_dof_indices,
            "physics_dt": PHYSICS_DT,
            "approach": trajectory["approach"],
            "fasten": trajectory["fasten"],
        }, f)
    log(f"[저장] {OUT_JSON}  approach={len(trajectory['approach'])}스텝 fasten={len(trajectory['fasten'])}스텝")

    _log_f.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

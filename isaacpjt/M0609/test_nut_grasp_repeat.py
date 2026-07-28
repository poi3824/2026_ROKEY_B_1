#!/usr/bin/env python3
"""통합 공정을 건드리지 않는 너트 물리 파지 반복 시험.

execute_isaac_nut_grasp_success.py의 검증된 물리 설정, FixedJoint 처리,
그리퍼 gain 및 RMPFlow 파라미터를 import해서 그대로 사용한다.
배터리/버스바/비전/ROS FSM은 실행하지 않으며, 매 회차 새 USD stage를 열어
다음 순서만 수행한다.

    초기 관절 자세 -> 선택 너트 실제 중심 상공 -> 하강 -> 물리 파지 -> 상승 판정

반복 사이에는 stage 자체를 닫고 다시 열기 때문에 이전 회차의 FixedJoint,
접촉 manifold, 속도 및 런타임 USD 변경이 다음 회차에 남지 않는다.
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수여야 합니다.")
    return parsed


def _nonnegative_float(value):
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("0 이상의 값이어야 합니다.")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collected_Busbar_fin 장면의 너트 물리 파지 반복 시험",
    )
    parser.add_argument(
        "--nut-index",
        type=int,
        choices=range(1, 7),
        default=1,
        metavar="N",
        help="시험할 물리 너트 번호 1~6 (기본값: 1)",
    )
    parser.add_argument(
        "--repeat",
        type=_positive_int,
        default=1,
        metavar="COUNT",
        help="반복 횟수 (기본값: 1; stage 재로딩 검증 전에는 1 권장)",
    )
    parser.add_argument(
        "--hold-sec",
        type=_nonnegative_float,
        default=2.0,
        metavar="SEC",
        help="각 회차 결과 자세 유지 시간 (기본값: 2초)",
    )
    parser.add_argument(
        "--phase-timeout-sec",
        type=_nonnegative_float,
        default=0.0,
        metavar="SEC",
        help="각 이동 단계 제한 시간. 0이면 제한 없음 (기본값: 0)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="뷰포트 없이 실행",
    )
    return parser.parse_args()


ARGS = parse_args()
if ARGS.headless:
    os.environ["AMR_HEADLESS"] = "1"

# Isaac/Kit이 이 시험 파일 전용 인자를 다시 해석하지 않도록 제거한 다음,
# 원본 executor를 라이브러리처럼 import한다. import 시 SimulationApp이 생성된다.
sys.argv = [sys.argv[0]]
import execute_isaac_nut_grasp_success as core  # noqa: E402


class TrialFailure(RuntimeError):
    pass


class SimulationClosed(RuntimeError):
    pass


@dataclass
class TrialRuntime:
    context: object
    stage: object
    world: object
    robot: object
    controller: object
    nut_xform: object
    nut_root_path: str
    nut_body_path: str
    ee_path: str
    quat_nut: np.ndarray


def _step(runtime, render=None):
    if not core.simulation_app.is_running():
        raise SimulationClosed("Isaac Sim 창이 종료되었습니다.")
    runtime.world.step(
        render=(not ARGS.headless) if render is None else bool(render)
    )


def _apply_gripper(robot, target):
    if np.isscalar(target):
        target = np.array([float(target), float(target)])
    robot.gripper.apply_action(
        core.ArticulationAction(
            joint_positions=np.asarray(target, dtype=float),
        )
    )


def _phase_expired(started_at):
    return (
        ARGS.phase_timeout_sec > 0.0
        and time.monotonic() - started_at >= ARGS.phase_timeout_sec
    )


def _ee_position(runtime):
    return np.asarray(
        core.world_xf(
            runtime.stage,
            f"{core.M0609_PATH}/{core.EE_LINK_NAME}",
        ).ExtractTranslation(),
        dtype=float,
    )


def _command_ee(runtime, target_position, target_orientation, gripper_target):
    action = runtime.controller.forward(
        target_end_effector_position=np.asarray(target_position, dtype=float),
        target_end_effector_orientation=np.asarray(
            target_orientation,
            dtype=float,
        ),
    )
    runtime.robot.apply_action(action)
    _apply_gripper(runtime.robot, gripper_target)
    _step(runtime)


def _move_ee(
    runtime,
    target_position,
    target_orientation,
    gripper_target,
    tolerance,
    label,
):
    target_position = np.asarray(target_position, dtype=float)
    started_at = time.monotonic()
    step_count = 0

    while True:
        _command_ee(
            runtime,
            target_position,
            target_orientation,
            gripper_target,
        )
        current = _ee_position(runtime)
        error = float(np.linalg.norm(target_position - current))

        if step_count % 30 == 0:
            print(
                f"[{label}] step={step_count} "
                f"EE=({current[0]:.4f},{current[1]:.4f},{current[2]:.4f}) "
                f"target=({target_position[0]:.4f},"
                f"{target_position[1]:.4f},{target_position[2]:.4f}) "
                f"error={error*1000:.1f}mm"
            )
        if error < tolerance:
            print(f"[{label}] 도착: error={error*1000:.1f}mm")
            return current
        if _phase_expired(started_at):
            raise TrialFailure(
                f"{label}가 {ARGS.phase_timeout_sec:.1f}초 안에 "
                f"{tolerance*1000:.1f}mm로 수렴하지 못했습니다 "
                f"(최종 {error*1000:.1f}mm)."
            )
        step_count += 1


def _initialize_home(runtime):
    arm_joint_names = [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
    ]
    arm_dof_indices = [
        runtime.robot.get_dof_index(name) for name in arm_joint_names
    ]
    started_at = time.monotonic()
    step_count = 0

    while True:
        all_joints = runtime.robot.get_joint_positions()
        current = np.asarray(all_joints[arm_dof_indices], dtype=float)
        max_step = core.ARM_INIT_MAX_JOINT_SPEED * core.PHYSICS_DT
        delta = core.TARGET_INIT_JOINTS - current
        target = current + np.clip(delta, -max_step, max_step)
        runtime.robot.apply_action(
            core.ArticulationAction(
                joint_positions=target,
                joint_indices=arm_dof_indices,
            )
        )
        _apply_gripper(runtime.robot, core.GRIPPER_OPEN)
        _step(runtime)

        joint_error = float(
            np.linalg.norm(current - core.TARGET_INIT_JOINTS)
        )
        if step_count % 30 == 0:
            print(
                f"[INIT_HOME] step={step_count} "
                f"joint_error={joint_error:.4f}rad"
            )
        if joint_error < core.JOINT_TOLERANCE:
            home = _ee_position(runtime)
            print(
                "[INIT_HOME] 완료: "
                f"EE=({home[0]:.4f},{home[1]:.4f},{home[2]:.4f})"
            )
            return home
        if _phase_expired(started_at):
            raise TrialFailure(
                f"INIT_HOME이 {ARGS.phase_timeout_sec:.1f}초 안에 "
                f"수렴하지 못했습니다 (joint_error={joint_error:.4f}rad)."
            )
        step_count += 1


def _sync_rmpflow_base(runtime):
    base_link_xf = core.world_xf(
        runtime.stage,
        f"{core.M0609_PATH}/base_link",
    )
    base_position = base_link_xf.ExtractTranslation()
    base_quaternion = base_link_xf.ExtractRotationQuat()
    runtime.controller._motion_policy.set_robot_base_pose(
        robot_position=np.array(
            [base_position[0], base_position[1], base_position[2]],
            dtype=float,
        ),
        robot_orientation=np.array(
            [
                base_quaternion.GetReal(),
                *[
                    float(value)
                    for value in base_quaternion.GetImaginary()
                ],
            ],
            dtype=float,
        ),
    )


def _build_trial_runtime():
    usd_path = core.Path(core.USD_PATH).resolve()
    if not usd_path.is_file():
        raise TrialFailure(f"USD 파일을 찾지 못했습니다: {usd_path}")

    context = core.omni.usd.get_context()
    context.open_stage(str(usd_path))
    for _ in range(15):
        core.simulation_app.update()

    stage = context.get_stage()
    if not stage:
        raise TrialFailure(f"USD stage 로드 실패: {usd_path}")

    core.configure_gripper_contact_physics(stage)
    core.configure_nut_contact_physics(stage)
    world = core.World(
        stage_units_in_meters=1.0,
        physics_dt=core.PHYSICS_DT,
    )
    core.lock_amr_base(stage, core.NOVA_CARTER_ROOT)

    nut_xforms = []
    for index in range(1, 7):
        _, body_path = core.nut_paths_for_index(index)
        if not stage.GetPrimAtPath(body_path).IsValid():
            raise TrialFailure(f"너트 rigid body 누락: {body_path}")
        nut_xforms.append(
            core.SingleXFormPrim(
                body_path,
                name=f"nut_grasp_test_xform_{index}",
            )
        )

    ee_path = core.find_prim_path(
        stage,
        core.M0609_PATH,
        core.EE_LINK_NAME,
    )
    if not ee_path:
        raise TrialFailure(
            f"End-effector prim 누락: {core.EE_LINK_NAME}"
        )

    gripper = core.ParallelGripper(
        end_effector_prim_path=ee_path,
        joint_prim_names=core.GRIPPER_JOINTS,
        joint_opened_positions=core.GRIPPER_OPEN,
        joint_closed_positions=core.GRIPPER_CLOSE,
        action_deltas=core.GRIPPER_DELTA,
    )
    robot = world.scene.add(
        core.SingleManipulator(
            prim_path=core.NOVA_CARTER_ROOT,
            name="nut_grasp_test_robot",
            end_effector_prim_path=ee_path,
            gripper=gripper,
        )
    )

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
        world.step(render=not ARGS.headless)

    for index, nut_xform in enumerate(nut_xforms, start=1):
        root_path, body_path = core.nut_paths_for_index(index)
        joint = core.attach_nut_to_amr(
            stage,
            root_path,
            body_path,
            core.NOVA_CARTER_ROOT,
            nut_xform,
            robot,
        )
        if joint is None:
            raise TrialFailure(
                f"너트 {index} AMR FixedJoint 생성 실패"
            )

    for _ in range(15):
        _apply_gripper(robot, core.GRIPPER_OPEN)
        world.step(render=not ARGS.headless)

    controller = core.RMPFlowController(
        name="nut_grasp_repeat_controller",
        robot_articulation=robot,
        urdf_path=core.URDF_PATH,
        robot_description_path=core.ROBOT_DESC_PATH,
        rmpflow_config_path=core.RMPFLOW_CFG_PATH,
        end_effector_frame_name=core.EE_LINK_NAME,
    )
    nut_root_path, nut_body_path = core.nut_paths_for_index(
        ARGS.nut_index
    )
    runtime = TrialRuntime(
        context=context,
        stage=stage,
        world=world,
        robot=robot,
        controller=controller,
        nut_xform=nut_xforms[ARGS.nut_index - 1],
        nut_root_path=nut_root_path,
        nut_body_path=nut_body_path,
        ee_path=ee_path,
        quat_nut=core.euler_to_quaternion_wxyz(0.0, 3.1415, 0.0),
    )
    _sync_rmpflow_base(runtime)
    return runtime


def _cleanup_trial(runtime):
    if runtime is None:
        return
    try:
        for index in range(1, 7):
            root_path, body_path = core.nut_paths_for_index(index)
            core.remove_nut_amr_joint(runtime.stage, root_path)
            core.detach_nut_from_gripper(runtime.stage, body_path)
        core.detach_busbar_from_gripper(runtime.stage)
        runtime.world.stop()
    except Exception as exc:
        print(f"[CLEANUP WARN] runtime joint 정리 중: {exc}")
    try:
        runtime.world.clear_instance()
    except Exception as exc:
        print(f"[CLEANUP WARN] World 정리 중: {exc}")
    try:
        runtime.context.close_stage()
        for _ in range(3):
            core.simulation_app.update()
    except Exception as exc:
        print(f"[CLEANUP WARN] Stage 정리 중: {exc}")


def _cleanup_partial_trial():
    """setup 도중 실패해 TrialRuntime을 만들지 못한 경우의 singleton 정리."""
    try:
        world = core.World.instance()
        if world is not None:
            world.stop()
    except Exception as exc:
        print(f"[CLEANUP WARN] 부분 World 정지 중: {exc}")
    try:
        core.World.clear_instance()
    except Exception as exc:
        print(f"[CLEANUP WARN] 부분 World 정리 중: {exc}")
    try:
        context = core.omni.usd.get_context()
        context.close_stage()
        for _ in range(3):
            core.simulation_app.update()
    except Exception as exc:
        print(f"[CLEANUP WARN] 부분 Stage 정리 중: {exc}")


def _hold_result(runtime, success):
    hold_steps = int(round(ARGS.hold_sec / core.PHYSICS_DT))
    target = _ee_position(runtime)
    gripper_target = (
        core.GRIPPER_CLOSE_NUT if success else core.GRIPPER_OPEN
    )
    for _ in range(hold_steps):
        _command_ee(
            runtime,
            target,
            runtime.quat_nut,
            gripper_target,
        )


def _perform_grasp(runtime):
    home_position = _initialize_home(runtime)

    if not core.filter_nut_arm_collision(
        runtime.stage,
        runtime.nut_root_path,
    ):
        raise TrialFailure("너트-손가락 충돌 설정 확인 실패")

    core.set_nut_gripper_gains(
        runtime.robot,
        core.NUT_CLOSE_KP,
        core.NUT_CLOSE_KD,
        label="반복 시험 저강성",
    )

    nut_start_position, _ = runtime.nut_xform.get_world_pose()
    nut_start_position = np.asarray(nut_start_position, dtype=float)
    pick_position = np.array(
        [
            nut_start_position[0],
            nut_start_position[1],
            core.NUT_PICK_Z,
        ],
        dtype=float,
    )
    approach_position = np.array(
        [
            nut_start_position[0],
            nut_start_position[1],
            core.NUT_APPROACH_Z,
        ],
        dtype=float,
    )
    print(
        f"[TARGET] nut={ARGS.nut_index} "
        f"actual=({nut_start_position[0]:.4f},"
        f"{nut_start_position[1]:.4f},{nut_start_position[2]:.4f}) "
        f"pick_z={pick_position[2]:.4f}"
    )

    # 전체 공정의 NUT_SCAN_LIFT와 동일하게, 큰 XY 이동과 손목 방향 변경을
    # 한 번에 시키지 않는다. 현재 XY에서 먼저 quat_nut/안전 고도를 맞춘 뒤
    # 너트 상공으로 수평 이동한다. 비전 스캔 자체는 수행하지 않는다.
    orientation_position = np.array(
        [
            home_position[0],
            home_position[1],
            core.NUT_APPROACH_Z,
        ],
        dtype=float,
    )
    _move_ee(
        runtime,
        orientation_position,
        runtime.quat_nut,
        core.GRIPPER_OPEN,
        0.02,
        "NUT_TEST_ORIENT",
    )
    _move_ee(
        runtime,
        approach_position,
        runtime.quat_nut,
        core.GRIPPER_OPEN,
        core.PICK_TOLERANCE_STRICT,
        "NUT_APPROACH",
    )
    _move_ee(
        runtime,
        pick_position,
        runtime.quat_nut,
        core.GRIPPER_OPEN,
        core.NUT_DESCEND_TOLERANCE_M,
        "NUT_DESCEND",
    )
    core.log_nut_grasp_geometry(
        runtime.stage,
        ARGS.nut_index,
        "반복 시험 하강 완료/열림",
    )

    for step_count in range(1, core.GRIP_CLOSE_RAMP_STEPS + 1):
        fraction = min(
            step_count / core.GRIP_CLOSE_RAMP_STEPS,
            1.0,
        )
        _command_ee(
            runtime,
            pick_position,
            runtime.quat_nut,
            fraction * core.NUT_PRECONTACT_POSITION,
        )

    if not core.enable_nut_arm_collision(
        runtime.stage,
        runtime.nut_root_path,
    ):
        raise TrialFailure("사전 닫기 후 finger collision 확인 실패")

    for _ in range(core.NUT_CONTACT_PRIME_STEPS):
        _command_ee(
            runtime,
            pick_position,
            runtime.quat_nut,
            core.NUT_PRECONTACT_POSITION,
        )

    if not core.detach_nut_from_amr(
        runtime.stage,
        runtime.nut_root_path,
    ):
        raise TrialFailure("AMR-너트 FixedJoint 해제 실패")

    close_ramp_start = core.NUT_AMR_RELEASE_SETTLE_STEPS
    close_ramp_end = (
        close_ramp_start + core.NUT_FINAL_CLOSE_RAMP_STEPS
    )
    gain_ramp_end = close_ramp_end + core.NUT_HOLD_GAIN_RAMP_STEPS

    for step_count in range(1, gain_ramp_end + 1):
        if step_count <= close_ramp_start:
            grip_target = core.NUT_PRECONTACT_POSITION
        elif step_count <= close_ramp_end:
            close_fraction = (
                (step_count - close_ramp_start)
                / core.NUT_FINAL_CLOSE_RAMP_STEPS
            )
            grip_target = (
                core.NUT_PRECONTACT_POSITION
                + close_fraction
                * (
                    core.GRIPPER_CLOSE_NUT[0]
                    - core.NUT_PRECONTACT_POSITION
                )
            )
        else:
            grip_target = core.GRIPPER_CLOSE_NUT[0]

        if step_count > close_ramp_end:
            gain_fraction = min(
                (step_count - close_ramp_end)
                / core.NUT_HOLD_GAIN_RAMP_STEPS,
                1.0,
            )
            stiffness = (
                core.NUT_CLOSE_KP
                + gain_fraction
                * (core.NUT_HOLD_KP - core.NUT_CLOSE_KP)
            )
            damping = (
                core.NUT_CLOSE_KD
                + gain_fraction
                * (core.NUT_HOLD_KD - core.NUT_CLOSE_KD)
            )
            core.set_nut_gripper_gains(
                runtime.robot,
                stiffness,
                damping,
            )

        _command_ee(
            runtime,
            pick_position,
            runtime.quat_nut,
            grip_target,
        )

        if step_count == close_ramp_end:
            core.log_nut_grasp_geometry(
                runtime.stage,
                ARGS.nut_index,
                "반복 시험 0.97rad 명령 완료",
            )
            print(
                "[GRASP] 최종 조임 명령 완료: "
                f"{core.gripper_joint_position_text(runtime.robot)}"
            )

    nut_after_grasp, _ = runtime.nut_xform.get_world_pose()
    nut_after_grasp = np.asarray(nut_after_grasp, dtype=float)
    horizontal_error = float(
        np.linalg.norm(
            nut_after_grasp[:2] - pick_position[:2]
        )
    )
    vertical_drop = float(
        nut_start_position[2] - nut_after_grasp[2]
    )
    print(
        f"[GRASP CHECK] xy_error={horizontal_error*1000:.1f}mm "
        f"z_drop={vertical_drop*1000:.1f}mm"
    )
    if horizontal_error > core.NUT_GRASP_MAX_XY_ERROR:
        raise TrialFailure(
            f"조임 중 너트 수평 이탈 "
            f"{horizontal_error*1000:.1f}mm"
        )
    if vertical_drop > core.NUT_GRASP_MAX_Z_DROP_M:
        raise TrialFailure(
            f"조임 중 너트 낙하 {vertical_drop*1000:.1f}mm"
        )

    for _ in range(core.NUT_GRIP_HOLD_SETTLE_STEPS):
        _command_ee(
            runtime,
            pick_position,
            runtime.quat_nut,
            core.GRIPPER_CLOSE_NUT,
        )

    lift_command_z = float(_ee_position(runtime)[2])
    started_at = time.monotonic()
    step_count = 0
    while True:
        lift_command_z = min(
            lift_command_z
            + core.NUT_LIFT_SPEED_M_S * core.PHYSICS_DT,
            approach_position[2],
        )
        lift_step_position = np.array(
            [
                approach_position[0],
                approach_position[1],
                lift_command_z,
            ],
            dtype=float,
        )
        _command_ee(
            runtime,
            lift_step_position,
            runtime.quat_nut,
            core.GRIPPER_CLOSE_NUT,
        )
        current = _ee_position(runtime)
        error = float(np.linalg.norm(approach_position - current))

        if step_count % 30 == 0:
            current_nut, _ = runtime.nut_xform.get_world_pose()
            current_nut = np.asarray(current_nut, dtype=float)
            print(
                f"[NUT_LIFT] step={step_count} "
                f"EE_z={current[2]:.4f} cmd_z={lift_command_z:.4f} "
                f"nut_delta_z="
                f"{(current_nut[2]-nut_start_position[2])*1000:.1f}mm "
                f"target_error={error*1000:.1f}mm"
            )
        if error < core.PICK_TOLERANCE_STRICT:
            break
        if _phase_expired(started_at):
            raise TrialFailure(
                f"NUT_LIFT가 {ARGS.phase_timeout_sec:.1f}초 안에 "
                f"{core.PICK_TOLERANCE_STRICT*1000:.1f}mm로 "
                f"수렴하지 못했습니다 (최종 {error*1000:.1f}mm)."
            )
        step_count += 1

    final_nut_position, _ = runtime.nut_xform.get_world_pose()
    final_nut_position = np.asarray(final_nut_position, dtype=float)
    lift_delta_z = float(
        final_nut_position[2] - nut_start_position[2]
    )
    print(
        f"[NUT LIFT CHECK] actual_z={final_nut_position[2]:.4f}m "
        f"delta_z={lift_delta_z*1000:.1f}mm"
    )
    if lift_delta_z < core.NUT_LIFT_MIN_DELTA_Z:
        raise TrialFailure(
            f"실제 너트 상승량 {lift_delta_z*1000:.1f}mm가 "
            f"기준 {core.NUT_LIFT_MIN_DELTA_Z*1000:.0f}mm 미만입니다."
        )

    return (
        f"너트 {ARGS.nut_index} 파지 성공, "
        f"상승량 {lift_delta_z*1000:.1f}mm"
    )


def run_trial(trial_number):
    runtime = None
    success = False
    message = ""
    try:
        print()
        print("=" * 72)
        print(
            f"[NUT GRASP TEST] {trial_number}/{ARGS.repeat}회차 시작 "
            f"(nut={ARGS.nut_index})"
        )
        print("=" * 72)
        runtime = _build_trial_runtime()
        message = _perform_grasp(runtime)
        success = True
        print(f"[TRIAL SUCCESS] {message}")
    except SimulationClosed:
        raise
    except Exception as exc:
        message = str(exc)
        print(f"[TRIAL FAILURE] {message}")
    finally:
        if runtime is not None:
            try:
                _hold_result(runtime, success)
            except SimulationClosed:
                pass
            except Exception as exc:
                print(f"[HOLD WARN] 결과 자세 유지 중: {exc}")
            _cleanup_trial(runtime)
        else:
            _cleanup_partial_trial()
    return success, message


def main():
    successes = 0
    failures = 0
    results = []
    print(
        "[NUT GRASP TEST] 전체 공정 우회 모드\n"
        f"  USD     : {core.USD_PATH}\n"
        f"  nut     : {ARGS.nut_index}\n"
        f"  repeat  : {ARGS.repeat}\n"
        f"  timeout : "
        f"{'없음' if ARGS.phase_timeout_sec == 0.0 else f'{ARGS.phase_timeout_sec:.1f}s'}"
    )

    try:
        for trial_number in range(1, ARGS.repeat + 1):
            success, message = run_trial(trial_number)
            results.append((trial_number, success, message))
            if success:
                successes += 1
            else:
                failures += 1
    except SimulationClosed as exc:
        print(f"[STOP] {exc}")
    finally:
        print()
        print("=" * 72)
        print(
            f"[SUMMARY] success={successes}, failure={failures}, "
            f"requested={ARGS.repeat}"
        )
        for trial_number, success, message in results:
            print(
                f"  #{trial_number}: "
                f"{'SUCCESS' if success else 'FAILURE'} - {message}"
            )
        print("=" * 72)
        core.simulation_app.close()

    return 0 if failures == 0 and len(results) == ARGS.repeat else 1


if __name__ == "__main__":
    raise SystemExit(main())

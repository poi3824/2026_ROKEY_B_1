"""execute_isaac_busbar_amr.py
Collected_Busbar3/Busbar.usd 씬에서 Nova Carter AMR 주행만 검증하기 위한 최소 런처.

execute_isaac_stations.py에서 검증된 AMR 주행 인프라 패턴(lock_amr_base/free_caster_wheels/
differential_drive/odometry_tf/clock)만 가져오고, m0609 팔·RMPFlow·그리퍼·라이다·카메라는
이번 라운드 범위가 아니라서 전부 뺐다 (AMR이 실제 좌표로 정확히 이동하는지만 먼저 검증).

사용법:
    AMR_HEADLESS=1 python3 execute_isaac_busbar_amr.py   # 헤드리스
    python3 execute_isaac_busbar_amr.py                  # GUI, Play 자동 시작됨
"""
import os
from pathlib import Path

from isaacsim import SimulationApp

_HEADLESS = os.environ.get("AMR_HEADLESS") == "1"
simulation_app = SimulationApp({"headless": _HEADLESS})

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.ros2_bridge")
simulation_app.update()

import omni.usd
import omni.graph.core as og
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Sdf

import rclpy

from isaacsim.core.api import World

_THIS_DIR = Path(__file__).resolve().parent
USD_PATH = str(_THIS_DIR / "Collected_Busbar3/Busbar.usd")
FACTORY_ASSET_PATH = str(
    _THIS_DIR
    / "factory_assets/industrial_pack/Assets/ArchVis/Industrial/"
      "Buildings/Warehouse/Warehouse01.usd"
)
FACTORY_DRESSING_ENABLED = os.environ.get("AMR_FACTORY_DRESSING", "1") != "0"
# Warehouse01은 centimeters(metersPerUnit=0.01) 기반이다. USD reference를 현재 meter
# stage에 합성할 때 단위가 자동 변환되지 않으므로, 0.005를 적용하면 실제 약
# 37m x 30m x 2.7m 규모가 되어 현재 작업 셀을 여유 있게 감싼다.
FACTORY_DRESSING_SCALE = float(os.environ.get("AMR_FACTORY_SCALE", "0.005"))


def add_factory_dressing(stage):
    """공식 NVIDIA Warehouse01을 시각 장식 전용 레이어로 추가한다.

    기존 Busbar.usd의 물리 환경과 좌표는 유지한다. 참조된 공장 에셋 아래의 CollisionAPI를
    모두 끄므로 기둥/벽/바닥이 AMR 주행을 방해하지 않는다. 크기는 환경변수로 조절 가능하다.
    """
    if not FACTORY_DRESSING_ENABLED:
        print("[Isaac Sim] 공장 장식 비활성화 (AMR_FACTORY_DRESSING=0).")
        return
    if not Path(FACTORY_ASSET_PATH).is_file():
        print(f"[Isaac Sim] 경고: 공장 에셋 파일을 찾지 못해 기본 장면으로 실행: "
              f"{FACTORY_ASSET_PATH}")
        return

    root = stage.DefinePrim("/World/FactoryDressing", "Xform")
    root.GetReferences().AddReference(FACTORY_ASSET_PATH)
    xform = UsdGeom.Xformable(root)
    xform.ClearXformOpOrder()
    # Warehouse 바닥을 작업 기준 Z=0 바로 위에 두어 원본 격자 바닥과 z-fighting 없이
    # 공장 바닥 재질이 보이게 한다. 2mm 차이는 AMR 물리에는 영향을 주지 않는다.
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.002))
    xform.AddScaleOp().Set(Gf.Vec3f(
        FACTORY_DRESSING_SCALE,
        FACTORY_DRESSING_SCALE,
        FACTORY_DRESSING_SCALE,
    ))

    disabled_colliders = 0
    for prim in Usd.PrimRange(root):
        collision = UsdPhysics.CollisionAPI.Get(stage, prim.GetPath())
        if not collision:
            continue
        collision.GetCollisionEnabledAttr().Set(False)
        disabled_colliders += 1

    # 기존 GroundPlane은 물리 충돌을 그대로 담당하되 렌더링만 숨긴다. Warehouse 바닥이
    # 공장 환경의 실제 시각 바닥으로 보이고, 검증된 AMR 접촉 물리는 바뀌지 않는다.
    original_ground = stage.GetPrimAtPath(
        "/World/defaultGroundPlane/GroundPlane")
    if original_ground.IsValid() and original_ground.IsA(UsdGeom.Imageable):
        UsdGeom.Imageable(original_ground).MakeInvisible()

    # Warehouse01의 기둥은 개별 prim이 아니라 재질별 통합 메시(Metal/WhitePaint/
    # GlossyMetal) 여러 개로 나뉘어 있다. 작업 셀 중앙과 배터리 모듈팩 뒤의 일반 기둥,
    # 사다리형 기둥을 모두 없애기 위해 구조용 재질 메시 전체를 숨긴다. 전용 메시로
    # 분리된 바닥·외벽·천장·조명은 그대로 유지된다.
    hidden_factory_prims = []
    for prim_name in (
        "SM_Metal_A1",
        "SM_WhitePaint_A1",
        "SM_WhitePaint_B1",
        "SM_GlossyMetal_A1",
        "SM_GlossyMetal_B1",
        "SM_GlossyMetal_C1",
    ):
        prim = stage.GetPrimAtPath(f"/World/FactoryDressing/{prim_name}")
        if prim.IsValid() and prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden_factory_prims.append(prim_name)

    print(
        f"[Isaac Sim] NVIDIA Warehouse01 공장 장식 적용 완료 "
        f"(scale={FACTORY_DRESSING_SCALE:.3f}, "
        f"비활성화 collider={disabled_colliders}개, "
        f"숨긴 구조물={hidden_factory_prims})."
    )

# NOTE: 이 씬(Collected_Busbar3)은 새로 받은 자산이라 Nova Carter의 정확한 prim 경로를
# Isaac Sim 밖에서 확인할 방법이 없었다(USD crate 바이너리, pxr 파이썬 바인딩이 이 셸엔 없음).
# 기존 씬들(execute_isaac_stations.py 등)과 같은 Nova Carter 자산 관례를 따른다고 가정하고
# 아래 경로를 썼다 - main()의 print_world_children()이 실제 트리를 찍어주니, 처음 실행해서
# "Nova Carter prim을 못 찾음" 경고가 뜨면 그 출력을 보고 이 상수들을 고치면 된다.
NOVA_CARTER_ROOT = "/World/Nova_Carter/chassis_link"
NOVA_CARTER_PARENT = "/World/Nova_Carter"
WHEEL_JOINT_LEFT = "/World/Nova_Carter/joint_wheel_left"
WHEEL_JOINT_RIGHT = "/World/Nova_Carter/joint_wheel_right"

# carter_warehouse_navigation.usd 공식 샘플 값과 동일(execute_isaac_stations.py에서 실측 확인됨).
WHEEL_RADIUS_M = 0.14
WHEEL_TRACK_WIDTH_M = 0.4132


def lock_amr_base(stage, amr_root_path, skip_paths=()):
    """AMR 차체 관절을 완전 잠금한다. skip_paths(구동륜)는 건너뛴다 - 그래야 cmd_vel로
    실제 주행이 가능하다."""
    amr_prim = stage.GetPrimAtPath(amr_root_path).GetParent()
    if not amr_prim.IsValid():
        amr_prim = stage.GetPrimAtPath(amr_root_path)

    locked_count = 0
    for prim in Usd.PrimRange(amr_prim):
        if str(prim.GetPath()) in skip_paths:
            continue
        for dt in ("angular", "linear"):
            drive = UsdPhysics.DriveAPI.Get(prim, dt)
            if drive:
                drive.GetStiffnessAttr().Set(1.0e9)
                drive.GetDampingAttr().Set(1.0e6)
                drive.GetMaxForceAttr().Set(1.0e9)
                if drive.GetTargetVelocityAttr():
                    drive.GetTargetVelocityAttr().Set(0.0)
                locked_count += 1
    print(f"[Isaac Sim] AMR 베이스 관절 드라이브 잠금 완료 (구동륜 제외, 잠긴 Drive 수: {locked_count})")


def free_caster_wheels(stage, amr_root_path):
    """캐스터의 방향전환 스위블 관절만 완전 수동으로 만든다.

    이 에셋에서 역할은 이름과 다르다.
    - joint_swing_left/right: 캐스터 방향전환(스위블)
    - joint_caster_left/right: 캐스터 바퀴 회전(원본부터 Drive API 없음)
    - joint_caster_base: 차체와 캐스터 프레임 사이 ±2도 서스펜션

    이전 구현은 이름에 ``caster``가 포함된 모든 Drive를 damping=200으로 바꿨다. 그 결과
    joint_caster_base 잠금까지 풀렸고, 스위블에는 큰 점성 저항이 남았다. 캐스터 초기 방향에
    따라 좌/우 회전 저항이 달라져 같은 크기의 양·음 angular.z에서 구동륜과 yaw 응답이 크게
    비대칭이 됐다. 스위블은 stiffness/damping/maxForce를 모두 0으로 두고, caster_base는
    lock_amr_base()가 잠근 상태를 유지한다.
    """
    amr_prim = stage.GetPrimAtPath(amr_root_path).GetParent()
    freed = []
    for prim in Usd.PrimRange(amr_prim):
        if prim.GetName() not in ("joint_swing_left", "joint_swing_right"):
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            print(f"[Isaac Sim] 경고: 캐스터 스위블 Drive 없음: {prim.GetPath()}")
            continue
        drive.GetStiffnessAttr().Set(0.0)
        drive.GetDampingAttr().Set(0.0)
        drive.GetMaxForceAttr().Set(0.0)
        drive.GetTargetVelocityAttr().Set(0.0)
        freed.append(str(prim.GetPath()))
    print(f"[Isaac Sim] 캐스터 스위블 완전 자유화: {freed}")


def setup_wheel_velocity_drive(stage, wheel_joint_path, damping=1.0e4, max_force=1.0e6):
    """구동륜은 위치 고정이 아니라 속도 추종 드라이브로 설정한다(stiffness=0, damping만 사용)."""
    prim = stage.GetPrimAtPath(wheel_joint_path)
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if not drive:
        raise RuntimeError(f"구동륜 angular Drive API가 없습니다: {wheel_joint_path}")
    drive.GetStiffnessAttr().Set(0.0)
    drive.GetDampingAttr().Set(damping)
    drive.GetMaxForceAttr().Set(max_force)
    drive.GetTargetVelocityAttr().Set(0.0)
    print(
        f"[Isaac Sim] 구동륜 velocity drive 설정: {wheel_joint_path} "
        f"(damping={damping}, maxForce={max_force})"
    )


def setup_ros_clock():
    """/clock 발행 - use_sim_time 쓰려면 필요."""
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/Graph/ROS_Clock", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            keys.SET_VALUES: [("PublishClock.inputs:topicName", "clock")],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
            ],
        },
    )
    print("[Isaac Sim] /clock 발행 연결 완료.")


def setup_differential_drive_ros2(chassis_prim_path):
    """/cmd_vel -> DifferentialController -> 바퀴 관절 속도."""
    keys = og.Controller.Keys
    graph_path = "/Graph/DifferentialDrive"
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("SubscribeTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                ("BreakLinear", "omni.graph.nodes.BreakVector3"),
                ("BreakAngular", "omni.graph.nodes.BreakVector3"),
                ("DiffController", "isaacsim.robot.wheeled_robots.DifferentialController"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
            ],
            keys.SET_VALUES: [
                ("SubscribeTwist.inputs:topicName", "/cmd_vel"),
                ("DiffController.inputs:wheelDistance", WHEEL_TRACK_WIDTH_M),
                ("DiffController.inputs:wheelRadius", WHEEL_RADIUS_M),
                ("DiffController.inputs:maxLinearSpeed", 1.0),
                ("DiffController.inputs:maxAngularSpeed", 1.2),
                ("ArticulationController.inputs:targetPrim", [chassis_prim_path]),
                ("ArticulationController.inputs:jointNames", ["joint_wheel_left", "joint_wheel_right"]),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "SubscribeTwist.inputs:execIn"),
                ("Context.outputs:context", "SubscribeTwist.inputs:context"),
                ("SubscribeTwist.outputs:execOut", "DiffController.inputs:execIn"),
                ("SubscribeTwist.outputs:linearVelocity", "BreakLinear.inputs:tuple"),
                ("SubscribeTwist.outputs:angularVelocity", "BreakAngular.inputs:tuple"),
                ("BreakLinear.outputs:x", "DiffController.inputs:linearVelocity"),
                ("BreakAngular.outputs:z", "DiffController.inputs:angularVelocity"),
                ("SubscribeTwist.outputs:execOut", "ArticulationController.inputs:execIn"),
                ("DiffController.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
            ],
        },
    )
    print(f"[Isaac Sim] Differential drive 연결 완료: /cmd_vel -> {chassis_prim_path} 바퀴.")


def setup_odometry_tf_ros2(chassis_prim_path, wheel_and_caster_prims):
    """odom 발행(/odom) + odom->base_link 동적 TF + base_link->바퀴/캐스터 정적 TF."""
    keys = og.Controller.Keys
    graph_path = "/Graph/OdometryTF"
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("ComputeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry"),
                ("PublishOdometry", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                ("PublishOdomTF", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                ("PublishSensorsTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ],
            keys.SET_VALUES: [
                ("ComputeOdometry.inputs:chassisPrim", [chassis_prim_path]),
                ("PublishOdometry.inputs:topicName", "odom"),
                ("PublishOdometry.inputs:odomFrameId", "odom"),
                ("PublishOdometry.inputs:chassisFrameId", "base_link"),
                ("PublishOdomTF.inputs:parentFrameId", "odom"),
                ("PublishOdomTF.inputs:childFrameId", "base_link"),
                ("PublishSensorsTF.inputs:parentPrim", [chassis_prim_path]),
                ("PublishSensorsTF.inputs:targetPrims", list(wheel_and_caster_prims)),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "ComputeOdometry.inputs:execIn"),
                ("ComputeOdometry.outputs:execOut", "PublishOdometry.inputs:execIn"),
                ("ComputeOdometry.outputs:position", "PublishOdometry.inputs:position"),
                ("ComputeOdometry.outputs:orientation", "PublishOdometry.inputs:orientation"),
                ("ComputeOdometry.outputs:linearVelocity", "PublishOdometry.inputs:linearVelocity"),
                ("ComputeOdometry.outputs:angularVelocity", "PublishOdometry.inputs:angularVelocity"),
                ("Context.outputs:context", "PublishOdometry.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishOdometry.inputs:timeStamp"),
                ("ComputeOdometry.outputs:execOut", "PublishOdomTF.inputs:execIn"),
                ("ComputeOdometry.outputs:position", "PublishOdomTF.inputs:translation"),
                ("ComputeOdometry.outputs:orientation", "PublishOdomTF.inputs:rotation"),
                ("Context.outputs:context", "PublishOdomTF.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishOdomTF.inputs:timeStamp"),
                ("OnPlaybackTick.outputs:tick", "PublishSensorsTF.inputs:execIn"),
                ("Context.outputs:context", "PublishSensorsTF.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishSensorsTF.inputs:timeStamp"),
            ],
        },
    )
    print(f"[Isaac Sim] Odometry(/odom) + TF 연결 완료 (대상 {len(wheel_and_caster_prims)}개).")


def find_prim_by_name(stage, root_path, name):
    """root_path 아래에서 이름이 정확히 name인 첫 prim의 절대 경로를 찾는다.
    Collected_Busbar3의 Nova Carter는 병합/최적화된 에셋(nova_carter_sim_optimized.usd 등)이라
    기존 씬과 조인트 계층 깊이가 다를 수 있어서, 고정 경로 대신 이름으로 찾는다."""
    root = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def print_world_children(stage, depth=2):
    """Nova Carter 등 실제 prim 경로가 가정과 맞는지 눈으로 확인할 수 있도록, /World 바로
    아래 자식들을 얕은 깊이로 출력한다. NOVA_CARTER_ROOT를 못 찾으면 여기 출력을 보고
    상수를 고치면 된다."""
    world_prim = stage.GetPrimAtPath("/World")
    print("[Isaac Sim] /World 하위 prim (깊이 {}):".format(depth))

    def _walk(prim, level):
        if level > depth:
            return
        print("  " * level + prim.GetPath().pathString)
        for child in prim.GetChildren():
            _walk(child, level + 1)

    for child in world_prim.GetChildren():
        _walk(child, 1)

    carter_prim = stage.GetPrimAtPath(NOVA_CARTER_ROOT)
    if not carter_prim.IsValid():
        print(f"[Isaac Sim] 경고: NOVA_CARTER_ROOT={NOVA_CARTER_ROOT} 를 못 찾음! "
              f"위 트리를 보고 스크립트 상단의 NOVA_CARTER_ROOT/PARENT/WHEEL_JOINT_* 상수를 고쳐야 함.")
    else:
        print(f"[Isaac Sim] NOVA_CARTER_ROOT={NOVA_CARTER_ROOT} 확인됨.")


def main():
    rclpy.init()

    ctx = omni.usd.get_context()
    ctx.open_stage(USD_PATH)
    for _ in range(15):
        simulation_app.update()

    stage = ctx.get_stage()
    # 런타임 전용 /Graph가 파일 저장 시 같이 저장되면 다음 실행에서 "이미 존재함" 에러가
    # 날 수 있어 매번 정리하고 시작한다.
    stage.RemovePrim(Sdf.Path("/Graph"))
    # 원본 Busbar.usd는 수정하지 않고, 공식 NVIDIA 창고 에셋을 런타임 장식으로만 합성한다.
    add_factory_dressing(stage)

    print_world_children(stage)

    world = World(stage_units_in_meters=1.0)

    # 고정 경로(WHEEL_JOINT_LEFT/RIGHT 상수) 대신 이름으로 실제 구동륜 조인트를 찾는다 -
    # 병합된 Nova Carter 에셋은 조인트 계층 깊이가 기존 씬과 다를 수 있어서, 경로가 안
    # 맞으면 lock_amr_base가 구동륜까지 통째로 잠가버리고 setup_wheel_velocity_drive도
    # 존재하지 않는 경로라 조용히 아무 효과 없이 넘어가 버린다(바퀴가 안 움직이는 원인).
    wheel_left_path = find_prim_by_name(stage, NOVA_CARTER_PARENT, "joint_wheel_left")
    wheel_right_path = find_prim_by_name(stage, NOVA_CARTER_PARENT, "joint_wheel_right")
    print(f"[Isaac Sim] joint_wheel_left 실제 경로: {wheel_left_path}")
    print(f"[Isaac Sim] joint_wheel_right 실제 경로: {wheel_right_path}")
    if wheel_left_path is None or wheel_right_path is None:
        print("[Isaac Sim] 경고: 구동륜 조인트를 이름으로 못 찾음 - 아래 프림 트리에서 실제 "
              "이름을 확인해서 find_prim_by_name 검색어를 고쳐야 함.")

    skip_paths = {p for p in (wheel_left_path, wheel_right_path) if p is not None}
    lock_amr_base(stage, NOVA_CARTER_ROOT, skip_paths=skip_paths)
    if wheel_left_path:
        setup_wheel_velocity_drive(stage, wheel_left_path)
    if wheel_right_path:
        setup_wheel_velocity_drive(stage, wheel_right_path)
    free_caster_wheels(stage, NOVA_CARTER_ROOT)

    world.reset()

    world.play()
    print("[Isaac Sim] 자동 재생(Play) 시작됨.")

    setup_ros_clock()
    setup_differential_drive_ros2(NOVA_CARTER_ROOT)
    wheel_and_caster_prims = [
        f"{NOVA_CARTER_PARENT}/wheel_left", f"{NOVA_CARTER_PARENT}/wheel_right",
        f"{NOVA_CARTER_PARENT}/caster_frame_base",
        f"{NOVA_CARTER_PARENT}/caster_swivel_left", f"{NOVA_CARTER_PARENT}/caster_swivel_right",
        f"{NOVA_CARTER_PARENT}/caster_wheel_left", f"{NOVA_CARTER_PARENT}/caster_wheel_right",
    ]
    setup_odometry_tf_ros2(NOVA_CARTER_ROOT, wheel_and_caster_prims)

    print("\n[Isaac Sim] AMR 주행 실행 준비 완료. (자동 Play됨 - 별도 조작 불필요)")

    while simulation_app.is_running():
        world.step(render=True)

    rclpy.shutdown()
    simulation_app.close()


if __name__ == '__main__':
    main()

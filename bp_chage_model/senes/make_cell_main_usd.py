"""
make_cell_main_usd.py
----------------------
구현 순서 5단계: assets/scenes/cell_main.usda 생성.
명세서 3.1절(최종 씬) + 4절(PhysX) + 6절(VariantSet) + 7절(AMR 스테이션) 반영.

*** 이번 단계에서 못 채운 부분 (꼭 확인해줘) ***
1. AMR/M0609 로봇 USD가 없음. 스펙에도 "두산 공식 URDF -> USD 변환본"이라고 돼 있어서
   제가 새로 만들 수 있는 게 아니라, 실제 두산 URDF를 Isaac Sim의 URDF Importer로 변환해서
   assets/robots/m0609.usda, assets/robots/amr_base.usda 로 넣어줘야 함.
   지금은 그 경로를 참조하도록 구조만 만들어놨어 (지금 열면 당연히 reference 에러 남 - 정상임).
   -> ROKEY 과정에서 이미 갖고 있는 M0609 URDF/USD 있으면 그거 경로로 바꿔주면 바로 붙어.
2. WristCam은 스펙상 TCP(로봇 손목)에 매달려야 하는데, 로봇이 없어서 지금은 팩 위 고정 위치에
   임시로 둠. M0609 붙이면 WristCam을 .../TCP 밑으로 재부모화 해줘야 함.
3. PhysxSceneAPI 등 PhysX 전용 속성은 이 샌드박스의 usd-core에 스키마 플러그인이 없어서
   "AddAppliedSchema"로 raw attribute만 수동으로 얹어놨음 (실제 Isaac Sim에서 열면 정상 인식됨,
   제 쪽에서 타입/스키마 정합성 재검증은 못 했다는 점 참고).
4. VariantSet(assemblyState)은 스펙 위치 그대로 /World/Fixture/BatteryPack 에 부여했고,
   bare/partial/complete 전환 시 BatteryPack 안쪽 AssembledBusbars 스코프에 GT용 버스바를
   0/5/11개 채워넣는 걸로 구현함 (AssemblySlots의 terminalA/B 월드좌표를 실제로 읽어서
   버스바를 두 단자 사이에 정렬시킴 - 데이터 기반 자동 배치가 실제로 동작하는지 검증도 겸함).
"""

from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, UsdShade, Sdf, Gf, Kind
import numpy as np

SLOT_TYPES = ["SHORT"] * 5 + ["LONG"] + ["SHORT"] * 5  # 3/4단계와 동일 순서


def _set_xform(prim, translate, quat_wxyz=None, scale=None):
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if quat_wxyz is not None:
        w, x, y, z = quat_wxyz
        op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat)
        op.Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
    if scale is not None:
        xformable.AddScaleOp().Set(Gf.Vec3f(*scale))


def _yaw_quat(deg):
    rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), deg)
    q = rot.GetQuat()
    return (q.GetReal(), *q.GetImaginary())


def _read_terminal_world_pose(battery_pack_path, slot_idx):
    """battery_pack.usda를 직접 열어서 Slot_XX의 terminalA/B MateTarget 월드 위치를 읽는다."""
    stage = Usd.Stage.Open(battery_pack_path)
    slot = stage.GetPrimAtPath(f"/BatteryPack/AssemblySlots/Slot_{slot_idx:02d}")
    rel_a = slot.GetRelationship("slot:terminalA").GetTargets()[0]
    rel_b = slot.GetRelationship("slot:terminalB").GetTargets()[0]
    mt_a = stage.GetPrimAtPath(str(rel_a) + "/MateTarget")
    mt_b = stage.GetPrimAtPath(str(rel_b) + "/MateTarget")
    pa = UsdGeom.Xformable(mt_a).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    pb = UsdGeom.Xformable(mt_b).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    return np.array([pa[0], pa[1], pa[2]]), np.array([pb[0], pb[1], pb[2]])


def _bridge_pose(pa, pb):
    """두 단자 MateTarget 사이를 잇는 버스바 배치 pose 계산: 위치=중점, X축=연결방향, Z축=world up."""
    mid = (pa + pb) / 2.0
    x_axis = pb - pa
    length = np.linalg.norm(x_axis)
    x_axis = x_axis / length if length > 1e-9 else np.array([1.0, 0.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-9:
        y_axis = np.array([0.0, 1.0, 0.0])
    else:
        y_axis = y_axis / y_norm
    z_axis = np.cross(x_axis, y_axis)
    R = Gf.Matrix3d(
        x_axis[0], x_axis[1], x_axis[2],
        y_axis[0], y_axis[1], y_axis[2],
        z_axis[0], z_axis[1], z_axis[2],
    ).GetTranspose()  # 열벡터 기준으로 basis 구성하려면 transpose
    quat = R.ExtractRotation().GetQuat()
    quat_wxyz = (quat.GetReal(), *quat.GetImaginary())
    return tuple(mid), quat_wxyz, float(length)


def make_cell_main(out_path,
                    battery_pack_component_path,  # 실제 디스크 경로 (terminal pose 읽기용)
                    battery_pack_ref="../assemblies/battery_pack.usda",
                    busbar_tray_ref="../assemblies/busbar_tray.usda",
                    short_busbar_ref="../components/busbar_short.usda",
                    long_busbar_ref="../components/busbar_long.usda"):
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("kilogramsPerUnit", 1.0)
    stage.SetTimeCodesPerSecond(60)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # --- PhysicsScene (4.1절) ---
    phys_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    phys_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    phys_scene.CreateGravityMagnitudeAttr(9.81)
    ps_prim = phys_scene.GetPrim()
    ps_prim.AddAppliedSchema("PhysxSceneAPI")
    ps_prim.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.Float).Set(240.0)
    ps_prim.CreateAttribute("physxScene:solverType", Sdf.ValueTypeNames.Token).Set("TGS")
    ps_prim.CreateAttribute("physxScene:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
    ps_prim.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
    ps_prim.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.Token).Set("GPU")

    # --- PhysicsMaterials (4.3절) ---
    mat_scope = UsdGeom.Scope.Define(stage, "/World/PhysicsMaterials")
    cs_mat = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/Copper_Steel")
    cs_api = UsdPhysics.MaterialAPI.Apply(cs_mat.GetPrim())
    cs_api.CreateStaticFrictionAttr(0.55)
    cs_api.CreateDynamicFrictionAttr(0.45)
    cs_api.CreateRestitutionAttr(0.05)

    gp_mat = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/Gripper_Pad")
    gp_api = UsdPhysics.MaterialAPI.Apply(gp_mat.GetPrim())
    gp_api.CreateStaticFrictionAttr(1.2)
    gp_api.CreateDynamicFrictionAttr(1.0)
    gp_api.CreateRestitutionAttr(0.0)

    # --- Environment ---
    env = UsdGeom.Scope.Define(stage, "/World/Environment")
    dome = UsdLux.DomeLight.Define(stage, "/World/Environment/DomeLight")
    dome.CreateIntensityAttr(1000.0)
    key = UsdLux.DistantLight.Define(stage, "/World/Environment/KeyLight")
    key.CreateIntensityAttr(3000.0)
    key.CreateAngleAttr(1.0)
    _set_xform(key.GetPrim(), (0, 0, 2.0), _yaw_quat(0))  # 방향은 필요시 추가 회전으로 조정 요망

    ground = UsdGeom.Mesh.Define(stage, "/World/Environment/GroundPlane")
    gs = 5.0
    ground.CreatePointsAttr([Gf.Vec3f(-gs, -gs, 0), Gf.Vec3f(gs, -gs, 0),
                              Gf.Vec3f(gs, gs, 0), Gf.Vec3f(-gs, gs, 0)])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateExtentAttr([Gf.Vec3f(-gs, -gs, 0), Gf.Vec3f(gs, gs, 0)])
    ground.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(0.2, 0.2, 0.22)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    # --- Fixture: BatteryPack (static) + StationPlan (7장) ---
    fixture = UsdGeom.Xform.Define(stage, "/World/Fixture")
    bp_prim = UsdGeom.Xform.Define(stage, "/World/Fixture/BatteryPack").GetPrim()
    bp_prim.GetReferences().AddReference(battery_pack_ref)
    _set_xform(bp_prim, (0.0, 0.0, 0.0))

    # VariantSet assemblyState (6장) - bare/partial/complete, AssembledBusbars에 GT 버스바 채움
    vset = bp_prim.GetVariantSets().AddVariantSet("assemblyState")
    variant_counts = {"bare": 0, "partial": 5, "complete": 11}
    for variant_name, n_show in variant_counts.items():
        vset.AddVariant(variant_name)
        vset.SetVariantSelection(variant_name)
        with vset.GetVariantEditContext():
            ab_scope = UsdGeom.Scope.Define(stage, "/World/Fixture/BatteryPack/AssembledBusbars")
            for i in range(n_show):
                btype = SLOT_TYPES[i]
                comp_ref = short_busbar_ref if btype == "SHORT" else long_busbar_ref
                pa, pb = _read_terminal_world_pose(battery_pack_component_path, i)
                pos, quat_wxyz, span = _bridge_pose(pa, pb)
                bb_path = f"/World/Fixture/BatteryPack/AssembledBusbars/Busbar_{i:02d}"
                bb_prim = UsdGeom.Xform.Define(stage, bb_path).GetPrim()
                bb_prim.GetReferences().AddReference(comp_ref)
                _set_xform(bb_prim, pos, quat_wxyz)
                slot_prim = stage.GetPrimAtPath(f"/World/Fixture/BatteryPack/AssemblySlots/Slot_{i:02d}")
                if slot_prim.IsValid():
                    slot_prim.GetAttribute("slot:state").Set("fastened")
    vset.SetVariantSelection("bare")  # 기본값은 조립 전(빈 상태)

    # StationPlan (7장) - AMR 정차 스테이션 3개소
    station_scope = UsdGeom.Scope.Define(stage, "/World/Fixture/StationPlan")
    stations = [
        (-0.72, [-1.08, -0.25], [0, 1, 2]),
        (0.00, [-0.4145, 0.4145], [3, 4, 5, 6]),
        (0.72, [0.3055, 1.1345], [7, 8, 9, 10]),
    ]
    for i, (sx, cover_x, slots) in enumerate(stations):
        st_prim = UsdGeom.Xform.Define(stage, f"/World/Fixture/StationPlan/Station_{i}").GetPrim()
        _set_xform(st_prim, (sx, -0.85, 0.0), _yaw_quat(90.0))
        st_prim.CreateAttribute("station:slots", Sdf.ValueTypeNames.IntArray).Set(slots)
        st_prim.CreateAttribute("station:coverX", Sdf.ValueTypeNames.DoubleArray).Set(cover_x)

    # --- BusbarTray: AMR 온보드 탑재 권장(7장) -> AMR 밑에 배치 ---
    amr = UsdGeom.Xform.Define(stage, "/World/AMR")
    _set_xform(amr.GetPrim(), (-0.72, -0.85, 0.0))  # 초기 위치 = Station_0 (임시)
    base_link = UsdGeom.Xform.Define(stage, "/World/AMR/base_link").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base_link)  # 실제 amr_base.usda 붙기 전 임시 RigidBody 마킹

    mount_height = 0.5  # 7장 가정: 베이스 높이 0.5m (6단계 도달성 재계산 대상 파라미터와 동일)
    mount_frame = UsdGeom.Xform.Define(stage, "/World/AMR/MountFrame").GetPrim()
    _set_xform(mount_frame, (0.0, 0.0, mount_height))

    # 실제 M0609+RG2FT+RealSense 에셋 (Collected_m0609_camera) 중 로봇 서브트리만 참조.
    # 파일 전체(m0609_camera.usd)의 defaultPrim은 /World인데 그 안에 GroundPlane/PhysicsScene도
    # 같이 들어있어서 통으로 참조하면 우리 씬이랑 중복됨 -> primPath로 /World/m0609만 콕 집어서 참조.
    m0609_prim = UsdGeom.Xform.Define(stage, "/World/AMR/MountFrame/M0609").GetPrim()
    m0609_prim.GetReferences().AddReference(
        "../robots/Collected_m0609_camera/m0609_camera.usd", "/World/m0609"
    )

    # GraspSensor: 그리퍼 파지 판정 기준점.
    # onrobot_rg2ft 루트 prim 자체의 로컬 원점이 실제 그리퍼 몸체 위치랑 안 맞아서(에셋 자체의
    # 특이한 CAD 원점으로 추정 - gripper_body/fingers는 tool0 근처인데 루트는 0.5m 이상 떨어져있음),
    # 임의 오프셋 대신 실제 두 손가락(inner_finger) 월드 위치의 중점을 직접 계산해서 배치함.
    right_finger = stage.GetPrimAtPath(
        "/World/AMR/MountFrame/M0609/onrobot_rg2ft/right_inner_finger")
    left_finger = stage.GetPrimAtPath(
        "/World/AMR/MountFrame/M0609/onrobot_rg2ft/left_inner_finger")
    gripper_body_prim = stage.GetPrimAtPath(
        "/World/AMR/MountFrame/M0609/onrobot_rg2ft/gripper_body")
    rf_w = UsdGeom.Xformable(right_finger).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    lf_w = UsdGeom.Xformable(left_finger).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    mid_world = Gf.Vec3d((rf_w[0] + lf_w[0]) / 2.0, (rf_w[1] + lf_w[1]) / 2.0, (rf_w[2] + lf_w[2]) / 2.0)
    gbody_to_world = UsdGeom.Xformable(gripper_body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    mid_local = gbody_to_world.GetInverse().Transform(mid_world)  # gripper_body 기준 로컬 좌표로 변환

    grasp_sensor = UsdGeom.Xform.Define(
        stage, "/World/AMR/MountFrame/M0609/onrobot_rg2ft/gripper_body/GraspSensor"
    ).GetPrim()
    _set_xform(grasp_sensor, (mid_local[0], mid_local[1], mid_local[2]))

    onboard_tray = UsdGeom.Xform.Define(stage, "/World/AMR/OnboardTray").GetPrim()
    onboard_tray.GetReferences().AddReference(busbar_tray_ref)
    _set_xform(onboard_tray, (0.2, 0.0, 0.5))  # AMR 위 임시 탑재 위치, 실제 로봇 붙으면 재조정 필요

    # --- Sensors ---
    sensors = UsdGeom.Scope.Define(stage, "/World/Sensors")
    pack_cam = UsdGeom.Camera.Define(stage, "/World/Sensors/PackCam")
    _set_xform(pack_cam.GetPrim(), (0.0, 0.0, 1.2), _yaw_quat(0))  # 팩 상단 고정 뷰, 완전 구현
    pack_cam.CreateFocalLengthAttr(18.0)

    # WristCam: 로봇 에셋에 RealSense D455가 이미 물려있어서(그리퍼 angle_bracket 하위)
    # 별도 placeholder Camera를 새로 안 만듦. 실제 경로:
    #   /World/AMR/MountFrame/M0609/onrobot_rg2ft/angle_bracket/realsense_d455/RSD455/
    #     Camera_OmniVision_OV9782_Color (RGB), Camera_Pseudo_Depth (Depth), Left/Right (스테레오 IR)
    # 이게 로봇을 따라 실제로 움직이는 진짜 카메라라, /World/Sensors 밑에 별도 고정 Xform으로
    # 복제해서 만들면 오히려 "안 움직이는 가짜 WristCam"이 돼서 혼란만 생김 -> 의도적으로 생략.

    # --- Scope_Assembled: 런타임 pick&place 중 재부모화되는 빈 슬롯 (5장 스냅 컨트롤러가 채움) ---
    UsdGeom.Scope.Define(stage, "/World/Scope_Assembled")

    stage.GetRootLayer().Save()
    return stage


if __name__ == "__main__":
    make_cell_main(
        out_path="assets/scenes/cell_main.usda",
        battery_pack_component_path="assets/assemblies/battery_pack.usda",
    )
    print("done")

"""
make_busbar_tray_usd.py
------------------------
구현 순서 4단계: assets/assemblies/busbar_tray.usda 생성.
명세서 3.4절 반영: Jig(static) + Slots(피치 0.06m, 2열) + 버스바 11개(kinematic).

타입 구성(10 SHORT + 1 LONG)은 3단계 AssemblySlots와 동일 순서로 맞춰서 Busbar_XX <-> Slot_XX가
1:1 대응되게 함 (필수는 아니지만 나중에 태스크 플래너 짤 때 편함).
"""

from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf, Kind

SLOT_TYPES = ["SHORT"] * 5 + ["LONG"] + ["SHORT"] * 5  # 3단계 make_slots()와 동일 순서
PITCH = 0.06
N_COLS = 2
BUSBAR_THICKNESS = 0.003


def make_busbar_tray(out_path,
                      short_component="../components/busbar_short.usda",
                      long_component="../components/busbar_long.usda"):
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("kilogramsPerUnit", 1.0)
    stage.SetTimeCodesPerSecond(60)

    tray_path = "/BusbarTray"
    tray = UsdGeom.Xform.Define(stage, tray_path)
    stage.SetDefaultPrim(tray.GetPrim())
    Usd.ModelAPI(tray.GetPrim()).SetKind(Kind.Tokens.assembly)

    n = len(SLOT_TYPES)
    n_rows = (n + N_COLS - 1) // N_COLS

    # --- Jig: 단순 평판 (static collision) - 슬롯 전체를 덮는 크기 + 여유 ---
    margin = 0.03
    jig_x = N_COLS * PITCH + margin
    jig_y = n_rows * PITCH + margin
    jig_thickness = 0.01
    jig = UsdGeom.Cube.Define(stage, f"{tray_path}/Jig")
    jig.CreateSizeAttr(2.0)
    jxf = UsdGeom.Xformable(jig)
    jxf.ClearXformOpOrder()
    cx = (N_COLS - 1) * PITCH / 2.0
    cy = -(n_rows - 1) * PITCH / 2.0
    jxf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, -jig_thickness / 2.0))
    jxf.AddScaleOp().Set(Gf.Vec3f(jig_x / 2.0, jig_y / 2.0, jig_thickness / 2.0))
    UsdPhysics.CollisionAPI.Apply(jig.GetPrim())
    mc = UsdPhysics.MeshCollisionAPI.Apply(jig.GetPrim())
    mc.CreateApproximationAttr(UsdPhysics.Tokens.none)  # static

    # --- Slots + Busbar 레퍼런스 (전부 kinematic, 집기 직전 해제 예정) ---
    slots_scope = UsdGeom.Scope.Define(stage, f"{tray_path}/Slots")
    for i, btype in enumerate(SLOT_TYPES):
        row, col = divmod(i, N_COLS)
        x = col * PITCH
        y = -row * PITCH
        slot_path = f"{tray_path}/Slots/Slot_{i:02d}"
        slot_xform = UsdGeom.Xform.Define(stage, slot_path)
        sxf = UsdGeom.Xformable(slot_xform)
        sxf.ClearXformOpOrder()
        sxf.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))

        busbar_path = f"{slot_path}/Busbar_{i:02d}"
        busbar_prim = UsdGeom.Xform.Define(stage, busbar_path).GetPrim()
        comp = short_component if btype == "SHORT" else long_component
        busbar_prim.GetReferences().AddReference(comp)
        busbar_prim.SetInstanceable(True)

        bxf = UsdGeom.Xformable(busbar_prim)
        bxf.ClearXformOpOrder()
        bxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, BUSBAR_THICKNESS / 2.0))

        # 초기 상태 kinematic (RigidBodyAPI는 레퍼런스로 이미 적용돼있음 -> 속성만 override)
        rb_api = UsdPhysics.RigidBodyAPI(busbar_prim)
        rb_api.CreateKinematicEnabledAttr(True)

        busbar_prim.CreateAttribute("tray:busbarType", Sdf.ValueTypeNames.String).Set(btype)
        busbar_prim.CreateAttribute("tray:slotIndex", Sdf.ValueTypeNames.Int).Set(i)

    stage.GetRootLayer().Save()
    return stage


if __name__ == "__main__":
    make_busbar_tray(out_path="assets/assemblies/busbar_tray.usda")
    print("done")

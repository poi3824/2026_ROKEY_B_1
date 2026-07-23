"""
make_battery_pack_usd.py
-------------------------
구현 순서 3단계: assets/assemblies/battery_pack.usda 생성.
명세서 3.3절(모듈 배치 + AssemblySlots) 반영. 모듈 12개 위치는 좌표식 그대로 루프 생성,
AssemblySlots 11개는 하드코딩 좌표 리스트 대신 "직렬 스네이크 패턴"을 코드로 추론해서 생성.

*** 이번 단계에서 제가 추론/가정한 것 (스펙엔 예시 1개만 있었음, 꼭 확인해줘) ***

1. 극성 배치 수정 (2단계 module_vda355.usda 되돌아가서 고침):
   스펙 예시가 "Slot_00: Module_00/Terminal_N -> Module_01/Terminal_P" 라고 명시함.
   물리적으로 인접한(SHORT 버스바로 이어지는) 단자여야 하므로, Module_00의 +X쪽(오른쪽,
   Module_01과 가까운 쪽) 단자가 반드시 "Terminal_N" 이어야 이 예시가 성립함.
   2단계에서는 제가 임의로 +X=P, -X=N 으로 잡았었는데, 이 예시와 맞지 않아서
   **+X=Terminal_N, -X=Terminal_P 로 뒤집어 재생성**했습니다. (module_vda355.usda 파일 자체를
   이번에 다시 만들었어요 - 아래 산출물에 새 버전 포함)

2. 직렬 연결 순서 = "스네이크(보스트로페돈) 패턴"으로 추론:
   행0: Module_00→01→02→03→04→05 (SHORT 5개, 열 증가 방향)
   행 끝 U턴: Module_05→11 (LONG 1개, 같은 열(col5)의 행0↔행1 수직 연결)
   행1: Module_11→10→09→08→07→06 (SHORT 5개, 열 감소 방향, 역주행)
   총 5+1+5=11개, 스펙의 "총 11개"와 일치. 이 추론이 실제 팩 설계와 다르면
   AssemblySlots 생성 로직(make_slots 함수)만 고치면 되니 alignment만 확인해줘.

3. Housing 형상: 스펙에 정확한 하우징 치수(벽 두께/높이)가 없어서, 지금은 **바닥 슬래브 하나만**
   넣었어 (벽체 생략). 풋프린트는 모듈 배치에서 자동 계산(2.155 x 0.307m, 7장 수치와 일치 확인함).
   실제 하우징 CAD/치수 나오면 교체 필요.
"""

from pxr import Usd, UsdGeom, UsdPhysics, UsdSemantics, Sdf, Gf, Kind

N_ROWS = 2
N_COLS = 6
PITCH_X = 0.360
PITCH_Y = 0.156
MODULE_Z = 0.054  # = module height/2 (2단계에서 확정한 중심원점 컨벤션)
MODULE_LENGTH = 0.355
MODULE_WIDTH = 0.151
MODULE_HEIGHT = 0.108


def module_position(r, c):
    x = -0.900 + PITCH_X * c
    y = -0.078 + PITCH_Y * r
    return (x, y, MODULE_Z)


def module_index(r, c):
    return r * N_COLS + c


def make_slots():
    """직렬 스네이크 패턴으로 AssemblySlots 11개 스펙 생성 (docstring 2번 참고)."""
    slots = []
    seq = 0

    # 행0: 0->1->2->3->4->5 (SHORT)
    for c in range(N_COLS - 1):
        a = module_index(0, c)
        b = module_index(0, c + 1)
        slots.append(dict(seq=seq, busbarType="SHORT",
                           terminalA=(a, "Terminal_N"), terminalB=(b, "Terminal_P")))
        seq += 1

    # U턴: 행0 마지막 열(col5) -> 행1 같은 열(col5) (LONG)
    a = module_index(0, N_COLS - 1)
    b = module_index(1, N_COLS - 1)
    slots.append(dict(seq=seq, busbarType="LONG",
                       terminalA=(a, "Terminal_N"), terminalB=(b, "Terminal_N")))
    seq += 1

    # 행1: 5->4->3->2->1->0 (col 기준 역방향, SHORT)
    for c in range(N_COLS - 1, 0, -1):
        a = module_index(1, c)
        b = module_index(1, c - 1)
        slots.append(dict(seq=seq, busbarType="SHORT",
                           terminalA=(a, "Terminal_P"), terminalB=(b, "Terminal_N")))
        seq += 1

    return slots


def make_battery_pack(out_path, module_component_rel_path="../components/module_vda355.usda"):
    stage = Usd.Stage.CreateNew(out_path)

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("kilogramsPerUnit", 1.0)
    stage.SetTimeCodesPerSecond(60)

    pack_path = "/BatteryPack"
    pack = UsdGeom.Xform.Define(stage, pack_path)
    stage.SetDefaultPrim(pack.GetPrim())
    Usd.ModelAPI(pack.GetPrim()).SetKind(Kind.Tokens.assembly)

    # --- Housing: 단순 바닥 슬래브 (docstring 3번 - 벽체 생략, 플레이스홀더) ---
    half_x = (-0.900 - MODULE_LENGTH / 2.0, 0.900 + MODULE_LENGTH / 2.0)
    half_y = (-0.078 - MODULE_WIDTH / 2.0, 0.078 + MODULE_WIDTH / 2.0)
    margin = 0.02
    floor_thickness = 0.01
    fx0, fx1 = half_x[0] - margin, half_x[1] + margin
    fy0, fy1 = half_y[0] - margin, half_y[1] + margin
    print(f"[info] pack footprint = {fx1-fx0-2*margin:.3f} x {fy1-fy0-2*margin:.3f} m "
          f"(7장 기준값 2.155 x 0.307 과 대조)")

    housing = UsdGeom.Cube.Define(stage, f"{pack_path}/Housing")
    housing.CreateSizeAttr(2.0)
    h_xformable = UsdGeom.Xformable(housing)
    h_xformable.ClearXformOpOrder()
    sx, sy, sz = (fx1 - fx0) / 2.0, (fy1 - fy0) / 2.0, floor_thickness / 2.0
    cx, cy, cz = (fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0, -floor_thickness / 2.0
    h_xformable.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))
    h_xformable.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
    UsdGeom.Imageable(housing).CreatePurposeAttr("default")
    UsdPhysics.CollisionAPI.Apply(housing.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(housing.GetPrim())
    mesh_collision.CreateApproximationAttr(UsdPhysics.Tokens.none)  # static이라 triangle mesh 그대로 허용
    UsdSemantics.LabelsAPI.Apply(housing.GetPrim(), "class").CreateLabelsAttr(["pack_housing"])

    # --- Modules: 12개 자동 배치 (instanceable reference) ---
    modules_scope = UsdGeom.Scope.Define(stage, f"{pack_path}/Modules")
    module_paths = {}
    for r in range(N_ROWS):
        for c in range(N_COLS):
            idx = module_index(r, c)
            name = f"Module_{idx:02d}"
            prim_path = f"{pack_path}/Modules/{name}"
            mod_xform = UsdGeom.Xform.Define(stage, prim_path)
            mod_prim = mod_xform.GetPrim()
            mod_prim.GetReferences().AddReference(module_component_rel_path)
            mod_prim.SetInstanceable(True)
            x, y, z = module_position(r, c)
            mxf = UsdGeom.Xformable(mod_prim)
            mxf.ClearXformOpOrder()
            mxf.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
            module_paths[idx] = prim_path

    # --- AssemblySlots: 11개 메타데이터 자동 생성 (docstring 2번의 make_slots() 로직) ---
    slots_scope = UsdGeom.Scope.Define(stage, f"{pack_path}/AssemblySlots")
    slot_defs = make_slots()
    assert len(slot_defs) == 11, f"슬롯 개수 불일치: {len(slot_defs)} (기대값 11)"

    for i, sd in enumerate(slot_defs):
        slot_path = f"{pack_path}/AssemblySlots/Slot_{i:02d}"
        slot_prim = UsdGeom.Xform.Define(stage, slot_path).GetPrim()

        a_idx, a_term = sd["terminalA"]
        b_idx, b_term = sd["terminalB"]
        term_a_path = f"{module_paths[a_idx]}/Terminals/{a_term}"
        term_b_path = f"{module_paths[b_idx]}/Terminals/{b_term}"

        slot_prim.CreateAttribute("slot:busbarType", Sdf.ValueTypeNames.String).Set(sd["busbarType"])
        slot_prim.CreateAttribute("slot:sequence", Sdf.ValueTypeNames.Int).Set(sd["seq"])
        slot_prim.CreateAttribute("slot:state", Sdf.ValueTypeNames.Token).Set("empty")
        slot_prim.CreateRelationship("slot:terminalA").AddTarget(Sdf.Path(term_a_path))
        slot_prim.CreateRelationship("slot:terminalB").AddTarget(Sdf.Path(term_b_path))

    stage.GetRootLayer().Save()
    return stage


if __name__ == "__main__":
    make_battery_pack(out_path="assets/assemblies/battery_pack.usda")
    print("done")

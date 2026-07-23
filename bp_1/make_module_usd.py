"""
make_module_usd.py
-------------------
구현 순서 2단계: assets/components/module_vda355.usda 생성.
명세서 3.3절(모듈 하위구조) + 8절(프로시저럴 지오메트리: Cube+scale, Cylinder 단자) 반영.

*** 좌표계 관련 판단(꼭 확인해줘) ***
스펙 문서에 모듈 로컬 좌표 원점 컨벤션이 두 군데서 서로 다르게 읽힘:
  (a) battery_pack.usda 배치식: "Module_N 위치 = (..., 0.054)" — 0.054 = 108mm/2.
      이건 모듈 로컬 원점이 "중심"이어야만 말이 되는 값(바닥이 팩 바닥에 닿으려면
      중심높이만큼 띄워야 하므로).
  (b) 단자 로컬 좌표: "로컬 (±0.150, 0, 0.108)" — 0.108 = 모듈 전고(全高) 그대로.
      이건 원점이 "바닥"이어야 말이 되는 값(상면 = 전고).
  두 값이 동시에 맞을 수 없어서(중심원점이면 상면은 0.054여야 함), 저는 (a)를 기준으로
  삼아 "원점 = 중심"으로 통일하고, 단자 Z는 0.108 대신 0.054(=height/2)를 사용했습니다.
  battery_pack.usda(3단계) 만들 때도 이 가정으로 갈 거라 지금 확인해주면 나중에 다시
  뜯어고칠 일이 없어요.
"""

from pxr import Usd, UsdGeom, UsdPhysics, UsdSemantics, Sdf, Gf, Kind


def make_module(out_path,
                 length=0.355, width=0.151, height=0.108,
                 terminal_x_offset=0.150,
                 post_dia=0.014, post_height=0.008):
    stage = Usd.Stage.CreateNew(out_path)

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("kilogramsPerUnit", 1.0)
    stage.SetTimeCodesPerSecond(60)

    module_path = "/Module"
    module = UsdGeom.Xform.Define(stage, module_path)
    stage.SetDefaultPrim(module.GetPrim())
    Usd.ModelAPI(module.GetPrim()).SetKind(Kind.Tokens.component)

    # 7장 목록: module 도 semantics class 부여
    UsdSemantics.LabelsAPI.Apply(module.GetPrim(), "class").CreateLabelsAttr(["module"])

    module.GetPrim().CreateAttribute("module:length", Sdf.ValueTypeNames.Double).Set(length)
    module.GetPrim().CreateAttribute("module:width", Sdf.ValueTypeNames.Double).Set(width)
    module.GetPrim().CreateAttribute("module:height", Sdf.ValueTypeNames.Double).Set(height)

    # --- Geom: 모듈 바디 (시각만, 충돌 없음 - 3.3절 명시) ---
    geom_scope = UsdGeom.Scope.Define(stage, f"{module_path}/Geom")
    body_cube = UsdGeom.Cube.Define(stage, f"{module_path}/Geom/Body")
    body_cube.CreateSizeAttr(2.0)  # -1..1 unit cube, xformOp:scale로 실치수 맞춤
    body_xf = UsdGeom.Xformable(body_cube)
    body_xf.ClearXformOpOrder()
    body_xf.AddScaleOp().Set(Gf.Vec3f(length / 2.0, width / 2.0, height / 2.0))
    UsdGeom.Imageable(body_cube).CreatePurposeAttr("render")
    # 충돌 없음 -> CollisionAPI 의도적으로 미적용 (스펙 3.3: "충돌 없음, 성능")

    # --- Terminals ---
    terminals_scope = UsdGeom.Scope.Define(stage, f"{module_path}/Terminals")

    top_z = height / 2.0  # 원점=중심 컨벤션 (위 docstring 참고)

    def make_terminal(name, x_pos, semantic_class):
        term_path = f"{module_path}/Terminals/{name}"
        term_xf = UsdGeom.Xform.Define(stage, term_path)
        term_xformable = UsdGeom.Xformable(term_xf)
        term_xformable.ClearXformOpOrder()
        term_xformable.AddTranslateOp().Set(Gf.Vec3d(x_pos, 0.0, top_z))

        UsdSemantics.LabelsAPI.Apply(term_xf.GetPrim(), "class").CreateLabelsAttr([semantic_class])

        # Post: UsdGeom.Cylinder, Ø14 x H8mm, 바닥이 단자 원점(모듈 상면)에 닿도록 +H/2 오프셋
        post = UsdGeom.Cylinder.Define(stage, f"{term_path}/Post")
        post.CreateRadiusAttr(post_dia / 2.0)
        post.CreateHeightAttr(post_height)
        post.CreateAxisAttr(UsdGeom.Tokens.z)
        post_xformable = UsdGeom.Xformable(post)
        post_xformable.ClearXformOpOrder()
        post_xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, post_height / 2.0))
        UsdGeom.Imageable(post).CreatePurposeAttr("render")
        UsdPhysics.CollisionAPI.Apply(post.GetPrim())  # static collider (RigidBody 없는 채로 충분)

        # MateTarget: 버스바 Mate가 정합될 목표. post 상단면, identity 회전
        # (+Z = 단자면 위쪽, +X = 모듈 길이방향 = 행(row) 방향과 일치 -> busbar Mate 컨벤션과 정합)
        mate_target = UsdGeom.Xform.Define(stage, f"{term_path}/MateTarget")
        mt_xformable = UsdGeom.Xformable(mate_target)
        mt_xformable.ClearXformOpOrder()
        mt_xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, post_height))

        return term_xf

    # 극성 배정: +X=Terminal_N, -X=Terminal_P
    # (battery_pack.usda 3단계에서 스펙 예시 "Module_00/Terminal_N -> Module_01/Terminal_P"가
    #  물리적으로 인접한 단자끼리 연결되려면 이 배치여야 성립함 - 3단계 스크립트 docstring 참고)
    make_terminal("Terminal_N", terminal_x_offset, "terminal_negative")
    make_terminal("Terminal_P", -terminal_x_offset, "terminal_positive")

    stage.GetRootLayer().Save()
    return stage


if __name__ == "__main__":
    make_module(out_path="assets/components/module_vda355.usda")
    print("done")

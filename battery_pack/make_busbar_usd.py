"""
make_busbar_usd.py
-------------------
명세서 3.2절(Prim 계층) + 8절(프로시저럴 지오메트리)을 실제 USD 파일로 오써링.

구현 순서 1단계: assets/components/busbar_short.usda, busbar_long.usda 생성.

주의 / 확인 필요 사항 (README 참고):
- Grasp_A/B 회전 값은 스펙 문구("TCP의 +Z가 아래를 향하도록")를 최대한 문자 그대로
  구현한 것. 실제 그리퍼 마운트 프레임(두산 TCP set_tcp() 값)과 반드시 대조 확인할 것.
- Semantics는 최신 OpenUSD 네이티브 스키마인 UsdSemantics.LabelsAPI로 구현.
  Isaac Sim 구버전(4.x 이전, 구 isaacsim semantics extension)이면 semantic:Semantics:params:*
  커스텀 attribute 컨벤션으로 별도 변환이 필요할 수 있음 - 사용 중인 Isaac Sim 버전 확인 요망.
"""

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdSemantics, Sdf, Gf, Kind

from busbar_mesh import build_busbar_visual_mesh, build_box_mesh


def _set_xform(prim, translate, quat_wxyz=None):
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    t_op = xformable.AddTranslateOp()
    t_op.Set(Gf.Vec3d(*translate))
    if quat_wxyz is not None:
        w, x, y, z = quat_wxyz
        o_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat)
        o_op.Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))


def _author_mesh(stage, path, points, face_vertex_counts, face_vertex_indices,
                  purpose="default", visible=True):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt_from_points(points))
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts.tolist())
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices.tolist())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    mesh.CreateDoubleSidedAttr(False)
    mesh.CreatePurposeAttr(purpose)
    if not visible:
        UsdGeom.Imageable(mesh).CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    # face-varying normals (faceted look, 각 face의 실제 법선을 그대로 사용)
    normals = []
    idx = face_vertex_indices.reshape(-1, 3)
    pts64 = points.astype(np.float64)
    for a, b, c in idx:
        n = np.cross(pts64[b] - pts64[a], pts64[c] - pts64[a])
        norm = np.linalg.norm(n)
        n = n / norm if norm > 0 else n
        normals.extend([n, n, n])
    mesh.CreateNormalsAttr(
        [Gf.Vec3f(float(n[0]), float(n[1]), float(n[2])) for n in normals]
    )
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)

    # extent
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    mesh.CreateExtentAttr([
        Gf.Vec3f(float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])),
        Gf.Vec3f(float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])),
    ])
    return mesh


def Vt_from_points(points):
    return [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points]


def make_busbar(out_path, variant_name, length, width, thickness,
                 hole_dia, hole_pitch, mass_kg, hole_segments=16):
    stage = Usd.Stage.CreateNew(out_path)

    # --- 1. 전역 설정 (0장) ---
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("kilogramsPerUnit", 1.0)
    stage.SetTimeCodesPerSecond(60)

    busbar_path = "/Busbar"
    busbar = UsdGeom.Xform.Define(stage, busbar_path)
    stage.SetDefaultPrim(busbar.GetPrim())
    Usd.ModelAPI(busbar.GetPrim()).SetKind(Kind.Tokens.component)

    # --- 2. 리지드바디 / 질량 (component 루트에 적용) ---
    UsdPhysics.RigidBodyAPI.Apply(busbar.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(busbar.GetPrim())
    mass_api.CreateMassAttr(float(mass_kg))
    # 관성텐서는 의도적으로 미설정 -> PhysX가 콜리전 형상 기준 자동계산 (스펙 3.2 "관성텐서는 자동계산 허용")

    # --- 3. Semantics (busbar 클래스) ---
    labels_api = UsdSemantics.LabelsAPI.Apply(busbar.GetPrim(), "class")
    labels_api.CreateLabelsAttr(["busbar"])

    # --- 4. Geom: 시각 메쉬(홀 포함) + 충돌 프록시(단순 박스) 분리 (8장) ---
    geom_scope = UsdGeom.Scope.Define(stage, f"{busbar_path}/Geom")

    v_points, v_counts, v_indices, v_subsets = build_busbar_visual_mesh(
        length, width, thickness, hole_dia, hole_pitch, hole_segments
    )
    visual_path = f"{busbar_path}/Geom/Visual"
    visual_mesh = _author_mesh(stage, visual_path, v_points, v_counts, v_indices,
                                purpose="render", visible=True)

    # GeomSubset: CopperFace / CoatedFace (머티리얼 배정용, 상하면=코팅 가정 - 실제 외관에 맞춰 조정 요망)
    for subset_name, face_indices in v_subsets.items():
        UsdGeom.Subset.CreateGeomSubset(
            visual_mesh, subset_name, UsdGeom.Tokens.face, face_indices
        )

    c_points, c_counts, c_indices = build_box_mesh(length, width, thickness)
    collision_path = f"{busbar_path}/Geom/Collision"
    collision_mesh = _author_mesh(stage, collision_path, c_points, c_counts, c_indices,
                                   purpose="guide", visible=False)
    UsdPhysics.CollisionAPI.Apply(collision_mesh.GetPrim())
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(collision_mesh.GetPrim())
    mesh_collision_api.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)

    # --- 5. Frames (3.2절 좌표계 규약) ---
    frames_scope = UsdGeom.Scope.Define(stage, f"{busbar_path}/Frames")

    mate_prim = UsdGeom.Xform.Define(stage, f"{busbar_path}/Frames/Mate").GetPrim()
    _set_xform(mate_prim, (0.0, 0.0, 0.0))  # 원점과 동일, identity

    hole0_prim = UsdGeom.Xform.Define(stage, f"{busbar_path}/Frames/Hole_0").GetPrim()
    _set_xform(hole0_prim, (-hole_pitch / 2.0, 0.0, 0.0))

    hole1_prim = UsdGeom.Xform.Define(stage, f"{busbar_path}/Frames/Hole_1").GetPrim()
    _set_xform(hole1_prim, (hole_pitch / 2.0, 0.0, 0.0))

    # Grasp_A: 상면(+thickness/2)에 위치, TCP의 +Z가 아래(-World Z)를 향하도록 Rx(180deg)
    rot_a = Gf.Rotation(Gf.Vec3d(1, 0, 0), 180.0)
    quat_a = rot_a.GetQuat()
    quat_a_wxyz = (quat_a.GetReal(), *quat_a.GetImaginary())
    grasp_a_prim = UsdGeom.Xform.Define(stage, f"{busbar_path}/Frames/Grasp_A").GetPrim()
    _set_xform(grasp_a_prim, (0.0, 0.0, thickness / 2.0), quat_a_wxyz)

    # Grasp_B: Grasp_A를 (자신의) Z축 기준 180도 추가 회전한 대칭 파지.
    # Rx(180) 이후 로컬 Z축 기준 180 회전 = 월드 기준 Ry(180)과 동일 (X,Y축만 반전, Z축 불변).
    rot_b = Gf.Rotation(Gf.Vec3d(0, 1, 0), 180.0)
    quat_b = rot_b.GetQuat()
    quat_b_wxyz = (quat_b.GetReal(), *quat_b.GetImaginary())
    grasp_b_prim = UsdGeom.Xform.Define(stage, f"{busbar_path}/Frames/Grasp_B").GetPrim()
    _set_xform(grasp_b_prim, (0.0, 0.0, thickness / 2.0), quat_b_wxyz)

    # --- 6. 커스텀 메타데이터 (치수 기록 - 디버깅/검증용, 스펙엔 없지만 실무상 유용) ---
    busbar.GetPrim().CreateAttribute("busbar:length", Sdf.ValueTypeNames.Double).Set(length)
    busbar.GetPrim().CreateAttribute("busbar:width", Sdf.ValueTypeNames.Double).Set(width)
    busbar.GetPrim().CreateAttribute("busbar:thickness", Sdf.ValueTypeNames.Double).Set(thickness)
    busbar.GetPrim().CreateAttribute("busbar:holePitch", Sdf.ValueTypeNames.Double).Set(hole_pitch)
    busbar.GetPrim().CreateAttribute("busbar:holeDiameter", Sdf.ValueTypeNames.Double).Set(hole_dia)
    busbar.GetPrim().CreateAttribute("busbar:variant", Sdf.ValueTypeNames.Token).Set(variant_name)

    stage.GetRootLayer().Save()
    return stage


if __name__ == "__main__":
    make_busbar(
        out_path="assets/components/busbar_short.usda",
        variant_name="SHORT",
        length=0.120, width=0.025, thickness=0.003,
        hole_dia=0.0085, hole_pitch=0.100,
        mass_kg=0.081,
    )
    make_busbar(
        out_path="assets/components/busbar_long.usda",
        variant_name="LONG",
        length=0.190, width=0.025, thickness=0.003,
        hole_dia=0.0085, hole_pitch=0.170,
        mass_kg=0.128,
    )
    print("done")

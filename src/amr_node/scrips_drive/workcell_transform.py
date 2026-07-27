#!/usr/bin/env python3
"""Busbar 작업 셀을 하나의 강체처럼 회전한다."""

import math
from dataclasses import dataclass
from typing import Dict, Tuple

from pxr import Gf, Sdf, Usd, UsdGeom


BUSBAR_TABLE_PATH = "/World/bench_busbar"
BUSBAR_WORKCELL_ROOTS = (
    BUSBAR_TABLE_PATH,
    "/World/Z_busbar3",
    "/World/Z_busbar3_01",
    "/World/Z_busbar3_02",
    "/World/Camera_busbar",
)

# Busbar.usd에 +90도 회전을 영구 반영한 장면의 접근 방향이다.
# target 위치는 perception만 제공하며, 이 값은 작업 셀의 docking normal이다.
BUSBAR_BASE_APPROACH_YAW_DEG = 180.0


@dataclass(frozen=True)
class WorkcellRotation:
    yaw_deg: float
    pivot_world: Tuple[float, float, float]
    busbar_approach_yaw_rad: float
    centers_before: Dict[str, Tuple[float, float, float]]
    centers_after: Dict[str, Tuple[float, float, float]]


def _finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label}는 유한한 값이어야 합니다")
    return value


def _xyz(value) -> Tuple[float, float, float]:
    return tuple(float(value[index]) for index in range(3))


def _world_centers(stage, paths):
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    return {
        path: _xyz(
            cache.GetLocalToWorldTransform(
                stage.GetPrimAtPath(path)
            ).ExtractTranslation()
        )
        for path in paths
    }


def _table_pivot(stage) -> Gf.Vec3d:
    table = stage.GetPrimAtPath(BUSBAR_TABLE_PATH)
    if not table.IsValid() or not table.IsDefined() or not table.IsActive():
        raise RuntimeError(f"버스바 테이블 prim 누락: {BUSBAR_TABLE_PATH}")

    purposes = [
        UsdGeom.Tokens.default_,
        UsdGeom.Tokens.render,
        UsdGeom.Tokens.proxy,
    ]
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        purposes,
        useExtentsHint=True,
    )
    aligned = bbox_cache.ComputeWorldBound(table).ComputeAlignedBox()
    minimum = aligned.GetMin()
    maximum = aligned.GetMax()
    values = [float(minimum[i]) for i in range(3)]
    values.extend(float(maximum[i]) for i in range(3))
    if (
        not all(math.isfinite(value) for value in values)
        or any(minimum[i] > maximum[i] for i in range(3))
    ):
        raise RuntimeError("버스바 테이블 world bounding box를 계산할 수 없습니다")

    # XY 회전 피벗만 필요하다. Z는 0으로 두면 Z 좌표가 그대로 보존된다.
    return Gf.Vec3d(
        0.5 * (minimum[0] + maximum[0]),
        0.5 * (minimum[1] + maximum[1]),
        0.0,
    )


def _group_rotation(pivot: Gf.Vec3d, yaw_deg: float) -> Gf.Matrix4d:
    to_origin = Gf.Matrix4d(1.0)
    to_origin.SetTranslate(Gf.Vec3d(-pivot[0], -pivot[1], 0.0))
    rotation = Gf.Matrix4d(1.0)
    rotation.SetRotate(
        Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), yaw_deg)
    )
    from_origin = Gf.Matrix4d(1.0)
    from_origin.SetTranslate(Gf.Vec3d(pivot[0], pivot[1], 0.0))
    return to_origin * rotation * from_origin


def rotate_busbar_workcell(
    stage: Usd.Stage,
    yaw_deg: float,
    *,
    edit_layer: Sdf.Layer = None,
) -> WorkcellRotation:
    """테이블 중심을 피벗으로 모든 작업 셀 root에 동일한 world 회전을 적용한다.

    bench_busbar의 Cube_01~Cube_06은 자식이므로 별도로 변환하지 않는다. 카메라의
    optical frame도 Camera_busbar의 자식이라 RGB/depth와 TF가 함께 회전한다.
    edit_layer를 생략하면 session layer에만 기록한다. 영구 변환 도구에서는
    명시적으로 root layer를 넘겨 원본 USD에 저장한다.
    """
    if stage is None:
        raise ValueError("USD stage가 필요합니다")
    yaw_deg = _finite(yaw_deg, "workcell yaw")

    for path in BUSBAR_WORKCELL_ROOTS:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsDefined() or not prim.IsActive():
            raise RuntimeError(f"버스바 작업 셀 prim 누락: {path}")
        if not prim.IsLoaded():
            stage.Load(path)

    pivot = _table_pivot(stage)
    before = _world_centers(stage, BUSBAR_WORKCELL_ROOTS)

    target_layer = stage.GetSessionLayer() if edit_layer is None else edit_layer
    if target_layer not in stage.GetLayerStack(includeSessionLayers=True):
        raise ValueError("edit_layer가 stage layer stack에 없습니다")
    stage.SetEditTarget(target_layer)
    if not math.isclose(yaw_deg, 0.0, abs_tol=1.0e-12):
        group = _group_rotation(pivot, yaw_deg)
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        matrices = {}
        for path in BUSBAR_WORKCELL_ROOTS:
            prim = stage.GetPrimAtPath(path)
            world = cache.GetLocalToWorldTransform(prim)
            parent_world = cache.GetLocalToWorldTransform(prim.GetParent())
            matrices[path] = world * group * parent_world.GetInverse()

        for path, local_matrix in matrices.items():
            xformable = UsdGeom.Xformable(stage.GetPrimAtPath(path))
            xformable.ClearXformOpOrder()
            xformable.AddTransformOp(
                UsdGeom.XformOp.PrecisionDouble
            ).Set(local_matrix)

    after = _world_centers(stage, BUSBAR_WORKCELL_ROOTS)
    approach_yaw = math.radians(BUSBAR_BASE_APPROACH_YAW_DEG + yaw_deg)
    approach_yaw = (approach_yaw + math.pi) % (2.0 * math.pi) - math.pi
    return WorkcellRotation(
        yaw_deg=yaw_deg,
        pivot_world=_xyz(pivot),
        busbar_approach_yaw_rad=approach_yaw,
        centers_before=before,
        centers_after=after,
    )

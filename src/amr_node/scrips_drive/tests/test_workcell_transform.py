import math
import sys
from pathlib import Path

from pxr import Gf, Usd, UsdGeom


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from workcell_transform import (  # noqa: E402
    BUSBAR_WORKCELL_ROOTS,
    rotate_busbar_workcell,
)


def _translated_xform(stage, path, xyz):
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    return xform


def _stage_with_workcell():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    table = _translated_xform(
        stage,
        "/World/bench_busbar",
        (0.0, 0.0, 0.0),
    )
    cube = UsdGeom.Cube.Define(
        stage,
        "/World/bench_busbar/Cube_01",
    )
    cube.CreateSizeAttr(2.0)
    for index, path in enumerate(BUSBAR_WORKCELL_ROOTS[1:4], start=1):
        _translated_xform(stage, path, (float(index), 0.0, 0.25))
    camera = UsdGeom.Camera.Define(stage, "/World/Camera_busbar")
    camera.AddTranslateOp().Set(Gf.Vec3d(0.0, 2.0, 1.0))
    assert table.GetPrim().IsValid()
    return stage


def test_workcell_rotates_all_roots_and_table_children_around_one_pivot():
    stage = _stage_with_workcell()
    report = rotate_busbar_workcell(stage, 90.0)

    assert report.pivot_world == (0.0, 0.0, 0.0)
    assert math.isclose(
        report.busbar_approach_yaw_rad,
        -math.pi / 2.0,
    )

    busbar = report.centers_after["/World/Z_busbar3"]
    assert math.isclose(busbar[0], 0.0, abs_tol=1.0e-12)
    assert math.isclose(busbar[1], 1.0, abs_tol=1.0e-12)
    assert math.isclose(busbar[2], 0.25, abs_tol=1.0e-12)

    camera = report.centers_after["/World/Camera_busbar"]
    assert math.isclose(camera[0], -2.0, abs_tol=1.0e-12)
    assert math.isclose(camera[1], 0.0, abs_tol=1.0e-12)

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )
    table_box = bbox_cache.ComputeWorldBound(
        stage.GetPrimAtPath("/World/bench_busbar")
    ).ComputeAlignedBox()
    assert math.isclose(table_box.GetMin()[0], -1.0)
    assert math.isclose(table_box.GetMax()[0], 1.0)


def test_zero_rotation_preserves_positions_and_publishes_baked_normal():
    stage = _stage_with_workcell()
    report = rotate_busbar_workcell(stage, 0.0)

    assert report.centers_after == report.centers_before
    assert math.isclose(
        report.busbar_approach_yaw_rad,
        -math.pi,
    )

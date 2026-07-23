"""
render_parts.py ─ 배치 수정 검증 + 방향 확인.
7_connector_insertion_real.py 와 동일한 부모-Xform 배치/방향으로 부품을 놓고,
월드 포즈를 출력 + 여러 각도 렌더 캡처. 결과 PNG: connector_assets/preview_*.png
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "width": 1024, "height": 1024})

from pathlib import Path
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
import omni.usd
from pxr import Usd, UsdGeom, UsdLux, Gf
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

_DIR = Path(__file__).resolve().parent / "connector_assets"

# 7_connector_insertion_real.py [B] 와 동일
RECEP_USD = str(_DIR / "2004563216.usd")
PLUG_USD  = str(_DIR / "2138150106.usd")
RECEP_XY, RECEP_Z, RECEP_EULER = (0.45, -0.20), 0.016, (0.0, 0.0, 0.0)
PLUG_XY,  PLUG_Z,  PLUG_EULER  = (0.45,  0.20), 0.030, (-90.0, 0.0, 0.0)


def reference_part(root_path, usd_path, pos, euler_deg, scale=1.0):
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, root_path)
    add_reference_to_stage(usd_path, root_path + "/geo")
    xf = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(root_path))
    xf.SetTranslate(Gf.Vec3d(*[float(v) for v in pos]))
    xf.SetRotate(Gf.Vec3f(*[float(e) for e in euler_deg]))
    xf.SetScale(Gf.Vec3f(scale, scale, scale))


def snap(path, eye, target):
    set_camera_view(eye=eye, target=target)
    for _ in range(60):
        simulation_app.update()
    capture_viewport_to_file(get_active_viewport(), str(path))
    for _ in range(40):
        simulation_app.update()
    print(f"  [OK] {path.name}")


def main():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1000.0)

    reference_part("/World/receptacle", RECEP_USD, [RECEP_XY[0], RECEP_XY[1], RECEP_Z], RECEP_EULER)
    reference_part("/World/connector",  PLUG_USD,  [PLUG_XY[0],  PLUG_XY[1],  PLUG_Z],  PLUG_EULER)

    # 플러그 지그(nest) 미리보기 (4벽) — 실제 스크립트와 동일 크기/중심
    c = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    r = c.ComputeWorldBound(stage.GetPrimAtPath("/World/connector")).ComputeAlignedRange()
    pmn, pmx = r.GetMin(), r.GetMax()
    cx, cy = (pmn[0] + pmx[0]) / 2, (pmn[1] + pmx[1]) / 2
    ax, ay, t, h = ((pmx[0]-pmn[0]) + 0.006)/2, ((pmx[1]-pmn[1]) + 0.004)/2, 0.006, 0.014
    ox = 2*(ax+t)
    from isaacsim.core.api.objects import FixedCuboid
    for nm, pos, sc in [("xp",[cx+ax+t/2,cy,h/2],[t,2*(ay+t),h]), ("xn",[cx-ax-t/2,cy,h/2],[t,2*(ay+t),h]),
                        ("yp",[cx,cy+ay+t/2,h/2],[ox,t,h]),       ("yn",[cx,cy-ay-t/2,h/2],[ox,t,h])]:
        world.scene.add(FixedCuboid(prim_path=f"/World/nest/{nm}", name=f"nest_{nm}",
                                    position=np.array([float(pos[0]),float(pos[1]),float(pos[2])]),
                                    scale=np.array(sc), color=np.array([0.35,0.35,0.4])))
    for _ in range(30):
        simulation_app.update()

    # 월드 포즈 확인 (원점이 아니어야 함)
    for label, path in [("RECEPTACLE", "/World/receptacle"), ("PLUG", "/World/connector")]:
        p, _ = SingleGeometryPrim(prim_path=path, name=f"{label}_r").get_world_pose()
        print(f"  [POSE] {label:<11} world = {np.round(np.asarray(p), 4)}")

    tgt = [0.45, 0.0, 0.05]     # 두 부품 중앙
    snap(_DIR / "preview_iso.png",   eye=[0.75, -0.30, 0.35], target=tgt)
    snap(_DIR / "preview_recep.png", eye=[0.45, -0.02, 0.18], target=[0.45, -0.20, 0.02])
    snap(_DIR / "preview_plug.png",  eye=[0.45,  0.42, 0.12], target=[0.45,  0.20, 0.04])

    simulation_app.close()


if __name__ == "__main__":
    main()

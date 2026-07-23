"""
inspect_step_usd.py ─ 변환된 STEP USD(플러그/리셉터클)의 실측 bbox/단위/메시수 출력.
7_connector_insertion_real.py 의 스케일/방향/삽입 기하 상수를 이 수치로 맞춘다.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pathlib import Path
from isaacsim.core.utils.stage import add_reference_to_stage
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

_DIR = Path(__file__).resolve().parent / "connector_assets"
PARTS = {
    "RECEPTACLE(200456)": _DIR / "2004563216.usd",
    "PLUG_FREEHANG(213815)": _DIR / "2138150106.usd",
}


def inspect(label: str, usd_path: Path, idx: int):
    root_path = f"/World/probe_{idx}"
    add_reference_to_stage(str(usd_path), root_path)
    for _ in range(20):
        simulation_app.update()
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(root_path)

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    size = [mx[i] - mn[i] for i in range(3)]
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    n_mesh = sum(1 for p in Usd.PrimRange(prim) if p.GetTypeName() == "Mesh")

    print("\n" + "=" * 66)
    print(f"[{label}]  {usd_path.name}")
    print("=" * 66)
    print(f"  metersPerUnit = {mpu}")
    print(f"  min  = {tuple(round(v,4) for v in mn)}")
    print(f"  max  = {tuple(round(v,4) for v in mx)}")
    print(f"  size = {tuple(round(v,4) for v in size)}  (x,y,z)")
    print(f"  실제 치수(mm) ≈ {tuple(round(v*mpu*1000,2) for v in size)}")
    print(f"  최장축 = {'xyz'[size.index(max(size))]} ({round(max(size)*mpu*1000,2)}mm), 메시수 = {n_mesh}")


def main():
    for i, (label, path) in enumerate(PARTS.items()):
        if path.exists():
            inspect(label, path, i)
        else:
            print(f"[없음] {path}")
    simulation_app.close()


if __name__ == "__main__":
    main()

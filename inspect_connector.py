"""
inspect_connector.py ─ Molex 커넥터 정품 USD의 실측 크기/구조/물리유무를 출력.
이 값으로 6_connector_insertion.py 의 스케일/파지 높이/방향을 정확히 맞춘다.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.nucleus import get_assets_root_path
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

REL = "/Isaac/Samples/Rigging/Jetbot/Jetbot_Base/parts/molex_connector_JFb.usd"


def main():
    root = get_assets_root_path()
    url = root + REL
    print("=" * 70)
    print(f"URL = {url}")
    print("=" * 70)

    add_reference_to_stage(url, "/World/probe")
    for _ in range(30):
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    probe = stage.GetPrimAtPath("/World/probe")

    # 월드 바운딩박스
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bound = cache.ComputeWorldBound(probe)
    rng = bound.ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    size = [mx[i] - mn[i] for i in range(3)]
    print("\n[BBox] (스테이지 기본 단위)")
    print(f"  min  = {tuple(round(v,4) for v in mn)}")
    print(f"  max  = {tuple(round(v,4) for v in mx)}")
    print(f"  size = {tuple(round(v,4) for v in size)}   (x, y, z)")
    print(f"  최대 치수 = {round(max(size),4)}")

    # stage meters-per-unit (단위 확인 — 스케일 계산에 중요)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"\n[Stage] metersPerUnit = {mpu}  → 최대 치수 ≈ {round(max(size)*mpu*1000,1)} mm")

    # 메시/물리 구조
    print("\n[Prim 트리 + 물리 API]")
    n_mesh = 0
    for p in Usd.PrimRange(probe):
        t = p.GetTypeName()
        flags = []
        if UsdPhysics.RigidBodyAPI(p):
            flags.append("Rigid")
        if UsdPhysics.CollisionAPI(p):
            flags.append("Collision")
        if t == "Mesh":
            n_mesh += 1
        depth = str(p.GetPath()).count("/")
        print(f"  {'  '*depth}{p.GetName()} [{t}] {flags}")
    print(f"\n메시 개수 = {n_mesh}")

    simulation_app.close()


if __name__ == "__main__":
    main()

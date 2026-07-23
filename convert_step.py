"""
convert_step.py ─ ~/Downloads 의 Molex Mega-Fit STEP(zip 내부 .stp)를 USD로 변환.

Isaac Sim 5.1 내장 CAD 변환기(omni.kit.converter.cad + HOOPS Exchange)를
omni.kit.asset_converter 파이프라인으로 사용한다. 외부 도구(FreeCAD 등) 불필요.

실행:
  PYTHONUNBUFFERED=1 /home/rokey/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh \
      /home/rokey/cobot3_ws/isaacpjt/M0609/convert_step.py
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import asyncio
import zipfile
from pathlib import Path

from isaacsim.core.utils.extensions import enable_extension

# CAD(STEP) 변환 엔진: HOOPS Exchange 코어를 직접 사용 (asset_converter는 STEP 미지원)
enable_extension("omni.kit.converter.common")
enable_extension("omni.kit.converter.hoops_core")
enable_extension("omni.kit.converter.hoops")
for _ in range(10):
    simulation_app.update()

from omni.kit.converter.hoops_core import get_instance as get_hoops

_THIS_DIR   = Path(__file__).resolve().parent
DOWNLOADS   = Path.home() / "Downloads"
OUT_DIR     = _THIS_DIR / "connector_assets"
STEP_DIR    = OUT_DIR / "step"

# zip → 안의 .stp 이름 (역할 주석)
ZIPS = {
    "187480908-659-2004563216.zip": "2004563216",   # Receptacle Housing (소켓)
    "187480962-659-2138150106.zip": "2138150106",   # Plug Free-hang (peg)
    "187480938-659-2138141106.zip": "2138141106",   # Plug Panel-mount (후속)
    "187465634-659-2280060001.zip": "2280060001",   # CPA Retainer (부속)
}


def unzip_all():
    STEP_DIR.mkdir(parents=True, exist_ok=True)
    stp_paths = {}
    for zip_name, part in ZIPS.items():
        zpath = DOWNLOADS / zip_name
        if not zpath.exists():
            print(f"  [SKIP] zip 없음: {zpath}")
            continue
        with zipfile.ZipFile(zpath) as zf:
            for member in zf.namelist():
                if member.lower().endswith((".stp", ".step")):
                    zf.extract(member, STEP_DIR)
                    stp_paths[part] = STEP_DIR / member
                    print(f"  [OK] unzip {zip_name} → {member}")
    return stp_paths


async def convert_one(in_stp: Path, out_usd: Path):
    # HOOPS Core 직접 호출 (UI 다이얼로그 우회). file_format_args 는 {} = 기본값.
    hoops = get_hoops()
    _, status = await hoops.create_converter_task(str(in_stp), str(out_usd), {})
    if status.error_code != 0:
        print(f"  [FAIL] {in_stp.name} : error_code={status.error_code} {status.error_msg}")
        return False
    size = out_usd.stat().st_size if out_usd.exists() else 0
    print(f"  [OK] 변환 {in_stp.name} → {out_usd.name}  ({size} bytes)")
    return True


def main():
    print("\n[1] STEP 압축 해제")
    stps = unzip_all()
    if not stps:
        print("[오류] .stp 를 찾지 못함")
        simulation_app.close()
        return

    print("\n[2] STEP → USD 변환")
    for part, in_stp in stps.items():
        out_usd = OUT_DIR / f"{part}.usd"
        fut = asyncio.ensure_future(convert_one(in_stp, out_usd))
        while not fut.done():
            simulation_app.update()
        fut.result()

    print("\n[3] 결과")
    for part in stps:
        u = OUT_DIR / f"{part}.usd"
        print(f"  {'있음' if u.exists() else '없음'}  {u}")

    simulation_app.close()


if __name__ == "__main__":
    main()

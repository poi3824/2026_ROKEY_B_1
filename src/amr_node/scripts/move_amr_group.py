"""move_amr_group.py -- Busbar.usd에서 AMR(Nova_Carter) + 매니퓰레이터(m0609) +
너트(nut1/nut2)를 한 그룹으로 묶어 지정한 world-space delta만큼 평행이동시킨다.

실제 물리 주행(cmd_vel/DifferentialController) 연동 전, 하드코딩 좌표로 그룹을
옮겨서 "이동된 위치에서도 조립 로직이 동작하는가"를 먼저 검증하기 위한 스크립트.
동일한 코드를 여러 컴퓨터에서 동일한 Busbar.usd 사본에 대해 반복 실행할 수 있다.

이동 그룹 (모두 /World 바로 밑의 독립 Xform -- USD 계층상 Nova_Carter의 자식이
아니라서 그룹 이동 시 명시적으로 같이 옮겨야 함):
  /World/Nova_Carter   -- chassis_link, carter_tray가 자식이라 자동으로 같이 옮겨감
  /World/m0609          -- FixedJoint로 chassis_link에 물리적으로만 용접, USD 부모는 아님
  /World/nut1, /World/nut2

battery_pack0~5 / Z_busbar1~3 / bench_busbar 등 고정 조립 트레이는 옮기지 않는다
(AMR이 트레이 쪽으로 이동해 오는 구조).

2행 3열 배치, 로봇(X=0.667)과 가까운 쪽이 1행:
  1행 = battery_pack3/4/5 (X=1.333, 로봇과 더 가까움)
  2행 = battery_pack0/1/2 (X=1.677)

측정된 컬럼 간 Y 간격 (참고용 -- AMR의 절대 도달 좌표는 캘리브레이션 전이라
이 스크립트가 임의로 정하지 않는다):
  pack3(-0.369) <-> pack4(-0.946): 0.577
  pack4(-0.946) <-> pack5(-1.473): 0.527
  (2행 pack0/1/2도 거의 동일한 Y: -1.474/-0.941/-0.369)

사용:
  # dry-run (기본값, 파일 저장 안 함) -- 이동 후 예상 좌표만 출력
  python3 move_amr_group.py --dy 0.533

  # 실제 저장 (저장 전 자동으로 .bak_<timestamp> 백업 생성)
  python3 move_amr_group.py --dy 0.533 --save

  # 대상 파일 지정
  python3 move_amr_group.py --dy 0.533 --save --usd /path/to/Busbar.usd

  # GUI에서 4개 prim을 직접 드래그해서 옮긴 뒤 Isaac Sim에서 Ctrl+S로 저장했다면,
  # Property 패널을 하나씩 클릭할 필요 없이 이걸로 현재 좌표 + BASELINE 대비 delta를 한 번에 확인:
  python3 move_amr_group.py --read

  # 눈으로 맞춰서 확정한 프리셋으로 이동 (BASELINE 기준, X는 항상 0으로 고정):
  python3 move_amr_group.py --preset row1 --save
"""
import argparse
import datetime
import os
import shutil
import sys

from pxr import Usd, UsdGeom, Gf, Tf

DEFAULT_USD = "/home/rokey/EV_combine/src/Collected_Busbar_amr/Busbar.usd"

GROUP_PRIMS = [
    "/World/Nova_Carter",
    "/World/m0609",
    "/World/nut1",
    "/World/nut2",
]

# 그룹을 한 번도 옮기지 않은 원본 상태의 translate 값 (--read에서 delta 계산 기준).
BASELINE = {
    "/World/Nova_Carter": Gf.Vec3d(0.6659430917711248, -0.045551821646005186, 2.1389568693498673e-8),
    "/World/m0609": Gf.Vec3d(0.6695317489356534, 0.191648535280057, 0.4316854793539975),
    "/World/nut1": Gf.Vec3d(0.3812361672997574, -0.30126815314508776, 0.9711037512121846),
    "/World/nut2": Gf.Vec3d(0.47100575524334387, -0.3012513976789413, 0.9711038761801655),
}

# BASELINE 기준 delta 프리셋 -- row1(battery_pack3/4/5) 3개 컬럼.
#
# X는 전부 0으로 고정한다: 지금까지 조립(arm reach) 로직이 검증된 유일한 X가
# BASELINE 원본 X이기 때문 -- "트레이 가까이 붙는 라인"을 이거 하나로 통일해서
# 여러 컴퓨터/팀원이 서로 다른 X에서 테스트하며 생기는 혼선을 없앤다. Y만 바꿔서
# 컬럼을 고른다.
#
# col2(-0.5983)는 이번 세션에서 반복 테스트해 애니메이션/물리 양쪽 다 문제없이
# 동작 확인된 값. col1/col3는 실측 컬럼 간격(battery_pack3<->4 0.577m,
# pack4<->5 0.527m, 평균 0.55m)을 그 기준점에 +-로 적용한 값이라 "이 정도 거리에
# 컬럼이 있다"는 추정이지, 어느 컬럼이 정확히 pack3/4/5 중 무엇인지는 팀원이
# 조립 스크립트로 직접 확인해야 한다 (2026-07-27 기준 미검증).
PRESETS = {
    "col1": Gf.Vec3d(0.0, -0.0483, 0.0),   # col2 - 0.55
    "col2": Gf.Vec3d(0.0, -0.5983, 0.0),   # 반복 테스트된 기준점
    "col3": Gf.Vec3d(0.0, -1.1483, 0.0),   # col2 + 0.55
}


def _get_translate_op(stage: Usd.Stage, path: str):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"prim not found: {path}")

    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            return op
    raise RuntimeError(f"{path}: xformOp:translate 없음 -- 스크립트가 가정한 구조와 다름")


def move_group(stage: Usd.Stage, delta: Gf.Vec3d):
    """현재 좌표 기준으로 delta만큼 상대 이동."""
    results = []
    for path in GROUP_PRIMS:
        translate_op = _get_translate_op(stage, path)
        before = translate_op.Get()
        after = Gf.Vec3d(before) + delta
        translate_op.Set(after)
        results.append((path, before, after))
    return results


def apply_preset(stage: Usd.Stage, preset_delta: Gf.Vec3d):
    """BASELINE + preset_delta로 절대 이동 (현재 좌표가 뭐든 상관없이 항상 같은 결과)."""
    results = []
    for path in GROUP_PRIMS:
        translate_op = _get_translate_op(stage, path)
        before = translate_op.Get()
        after = Gf.Vec3d(BASELINE[path]) + preset_delta
        translate_op.Set(after)
        results.append((path, before, after))
    return results


def read_group(stage: Usd.Stage):
    results = []
    for path in GROUP_PRIMS:
        translate_op = _get_translate_op(stage, path)
        current = Gf.Vec3d(translate_op.Get())
        base = BASELINE.get(path)
        delta = current - base if base is not None else None
        results.append((path, current, delta))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--usd", default=DEFAULT_USD, help="대상 Busbar.usd 경로")
    parser.add_argument("--dx", type=float, default=0.0)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.0)
    parser.add_argument("--preset", choices=sorted(PRESETS.keys()),
                         help="BASELINE + 등록된 delta로 절대 이동 (--dx/--dy/--dz와 동시 사용 불가)")
    parser.add_argument("--save", action="store_true", help="지정 안 하면 dry-run(저장 안 함)")
    parser.add_argument("--read", action="store_true",
                         help="이동 없이 현재 좌표 + BASELINE 대비 delta만 출력 (GUI에서 직접 드래그 후 저장한 값 확인용)")
    args = parser.parse_args()

    if not os.path.isfile(args.usd):
        print(f"파일 없음: {args.usd}", file=sys.stderr)
        sys.exit(1)

    try:
        stage = Usd.Stage.Open(args.usd)
    except Tf.ErrorException as e:
        print(f"스테이지를 열 수 없음: {args.usd}\n{e}", file=sys.stderr)
        sys.exit(1)
    if stage is None:
        print(f"스테이지를 열 수 없음: {args.usd}", file=sys.stderr)
        sys.exit(1)

    if args.read:
        try:
            results = read_group(stage)
        except RuntimeError as e:
            print(f"이동 그룹 prim을 찾을 수 없음 -- Busbar.usd가 맞는 파일인지 확인: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"{'prim':35s} {'current translate':35s} delta vs BASELINE")
        deltas = []
        for path, current, delta in results:
            d = tuple(round(v, 4) for v in delta) if delta is not None else "N/A"
            if delta is not None:
                deltas.append(tuple(round(v, 4) for v in delta))
            print(f"  {path:35s} {tuple(round(v, 4) for v in current)!s:35s} {d}")
        if len(set(deltas)) > 1:
            print("\n[경고] 4개 prim의 delta가 서로 다르다 -- 그룹이 같이 안 옮겨졌을 가능성 (축 잠금 풀렸거나 일부만 드래그됨).")
        elif deltas:
            print(f"\n일치 -- 공통 delta = {deltas[0]} (이 값을 --dx/--dy/--dz로 다른 컴퓨터에서 재현하면 됨)")
        return

    manual_delta_given = args.dx != 0.0 or args.dy != 0.0 or args.dz != 0.0
    if args.preset and manual_delta_given:
        print("--preset과 --dx/--dy/--dz는 동시에 못 쓴다.", file=sys.stderr)
        sys.exit(1)
    if not args.preset and not manual_delta_given:
        print("delta가 전부 0이다. --preset 또는 --dx/--dy/--dz 중 하나는 지정해야 한다.", file=sys.stderr)
        sys.exit(1)

    if args.preset:
        preset_delta = PRESETS[args.preset]
        try:
            results = apply_preset(stage, preset_delta)
        except RuntimeError as e:
            print(f"이동 그룹 prim을 찾을 수 없음 -- Busbar.usd가 맞는 파일인지 확인: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"preset = {args.preset} (BASELINE + {tuple(preset_delta)})")
    else:
        delta = Gf.Vec3d(args.dx, args.dy, args.dz)
        try:
            results = move_group(stage, delta)
        except RuntimeError as e:
            print(f"이동 그룹 prim을 찾을 수 없음 -- Busbar.usd가 맞는 파일인지 확인: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"delta = {tuple(delta)}")
    for path, before, after in results:
        print(f"  {path:35s} {tuple(round(v, 4) for v in before)} -> {tuple(round(v, 4) for v in after)}")

    if not args.save:
        print("\n[dry-run] --save를 안 줘서 파일은 저장하지 않았다.")
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{args.usd}.bak_{ts}"
    shutil.copy2(args.usd, backup_path)
    print(f"\n백업 생성: {backup_path}")

    stage.GetRootLayer().Save()
    print(f"저장 완료: {args.usd}")


if __name__ == "__main__":
    main()

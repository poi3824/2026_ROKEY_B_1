"""
probe_assets.py ─ nucleus/SimReady 에 실제 커넥터/조립(peg-in-hole) 정품 에셋이
있는지 경로를 헤드리스로 탐색해서 출력한다.

여기서 나온 .usd 경로를 6_connector_insertion.py 의 CONNECTOR_ASSET_URL /
SOCKET_ASSET_URL 상수에 넣으면 정품 에셋으로 교체된다.

실행:
  /home/rokey/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh \
      /home/rokey/cobot3_ws/isaacpjt/M0609/probe_assets.py
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.client
from isaacsim.core.utils.nucleus import get_assets_root_path

# 관심 키워드 — 커넥터/삽입 조립 관련
KEYWORDS = [
    "connector", "plug", "socket", "receptacle", "terminal",
    "peg", "hole", "insert", "nut", "bolt", "gear", "harness",
]
MAX_DEPTH = 4          # 재귀 깊이 제한 (너무 깊이 들어가면 느림)
MAX_HITS = 400         # 안전 상한


def is_dir(entry) -> bool:
    return bool(entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN)


def crawl(url: str, depth: int, hits: list, seen: set):
    if depth > MAX_DEPTH or len(hits) >= MAX_HITS:
        return
    if url in seen:
        return
    seen.add(url)
    try:
        res, entries = omni.client.list(url)
    except Exception as e:
        print(f"  [list 실패] {url} : {e}")
        return
    if res != omni.client.Result.OK:
        return
    for e in entries:
        name = e.relative_path
        child = url.rstrip("/") + "/" + name
        low = name.lower()
        if low.endswith(".usd") or low.endswith(".usda") or low.endswith(".usdc"):
            if any(k in low for k in KEYWORDS):
                hits.append(child)
                print(f"  [HIT] {child}")
        elif is_dir(e):
            # 디렉터리 이름에 키워드가 있으면 우선 깊이 탐색
            crawl(child, depth + 1, hits, seen)


def main():
    root = get_assets_root_path()
    print("=" * 70)
    print(f"assets_root = {root}")
    print("=" * 70)
    if not root:
        print("[오류] assets_root 를 못 찾음 — 네트워크/nucleus 설정 확인 필요.")
        simulation_app.close()
        return

    # 탐색 시작점 (존재하는 것만 crawl 됨)
    start_dirs = [
        root + "/Isaac/Props",
        root + "/Isaac/Samples/Examples/FrankaNutBolt",
        root + "/Isaac/Samples",
        root + "/Isaac/Props/Factory",
        root + "/NVIDIA/Assets",          # SimReady 계열 (있을 경우)
        root + "/NVIDIA/Assets/SimReady",
    ]

    hits, seen = [], set()
    for d in start_dirs:
        print(f"\n--- crawl: {d} ---")
        crawl(d, 0, hits, seen)

    print("\n" + "=" * 70)
    print(f"총 {len(hits)}개 후보 발견")
    print("=" * 70)
    for h in hits:
        print(h)
    if not hits:
        print("(키워드 매칭 에셋 없음 — 절차적 fallback 사용 권장)")

    simulation_app.close()


if __name__ == "__main__":
    main()

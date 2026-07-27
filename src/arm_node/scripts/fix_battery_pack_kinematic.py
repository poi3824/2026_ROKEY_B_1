"""fix_battery_pack_kinematic.py -- battery_pack3(볼트/너트 체결 대상이 붙어있는 강체)를
kinematic으로 고정해서 물리 드리프트를 막는다 (1회성 스크립트).

record_nut_fasten_trajectory.py가 기록한 bolt_2 좌표는 녹화 시점 스냅샷인데,
battery_pack3에 RigidBodyAPI(kinematicEnabled=False)가 걸려있어 Play를 누르면
중력으로 조금씩 가라앉는다. 그러면 라이브 bolt_2 위치가 녹화된 목표 좌표에서
어긋나 FASTEN 재생이 실제 볼트를 못 맞출 수 있다.

kinematicEnabled=True로 바꾸면 저장된 자세(=녹화 당시 좌표와 동일)에 영구히
고정되어 이 드리프트가 사라진다.

실행:
  /home/rokey/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh \
      /home/rokey/EV_combine/src/arm_node/scripts/fix_battery_pack_kinematic.py
"""
import os

_HEADLESS = os.environ.get("BOLT_HEADLESS", "1") == "1"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": _HEADLESS})

import omni.usd
from pxr import UsdPhysics

WORLD_USD = "/home/rokey/EV_combine/src/Collected_World0123/World0123.usd"
TARGET_PRIM_PATH = "/World/battery_pack3"

context = omni.usd.get_context()
context.open_stage(WORLD_USD)
for _ in range(20):
    simulation_app.update()

stage = context.get_stage()
prim = stage.GetPrimAtPath(TARGET_PRIM_PATH)
rb = UsdPhysics.RigidBodyAPI(prim)
rb.GetKinematicEnabledAttr().Set(True)

for _ in range(5):
    simulation_app.update()

stage.GetRootLayer().Save()

_log_f = open(os.path.join(os.path.dirname(__file__), "fix_battery_pack_kinematic_result.txt"), "w")
_log_f.write(f"[저장 완료] {TARGET_PRIM_PATH} kinematicEnabled=True -> {WORLD_USD}\n")
_log_f.close()

simulation_app.close()

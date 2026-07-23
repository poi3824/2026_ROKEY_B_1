"""
10_thread_controller.py ─ Native Joint Thread Controller (9_nut_grasp_experiment.py 분기)

v1(kinematic thread controller, 2026-07-22 이전)은 Δθ 실측 → nut.set_world_pose() 로 매
스텝 pose 를 직접 덮어쓰는 방식이었는데, engage 순간의 위치 점프가 고정된 볼트를 튕겨내고
(실측), 육각너트(60° 대칭)와 패스당 회전량의 위상이 안 맞아 재파지 때 삐뚤게 물리는
문제(실측)가 있었다. 이 파일은 그 스크립트-레벨 흉내를 걷어내고, 체결 자체를 **PhysX
네이티브 조인트(PhysxMimicJointAPI)** 에 맡긴다 — 참고:
Tech-Multiverse/omniverse-nut-and-bolt-digital-twin 의 static_bolt.usda(고정 볼트 +
PrismaticJoint↔RevoluteJoint mimic 커플링) 패턴을 그대로 이식.

  ① Pick  ② 볼트축 정렬  ③ 착좌     ← ALIGN, PickPlaceController + 실제 마찰 파지(9번과 동일,
                                        free_nut 은 자유 강체, meshSimplification 콜리전)
        ↓  착좌 판정(ENGAGE_XY_TOL_M/TILT/GAP)
  ④ HANDOFF: 그리퍼 열기 → free_nut 격리(kinematic+숨김+콜리전 off, 멀리 이동)
             → screw_nut 표시/콜리전 on → 재파지
  ⑤ SCREW(래칫): 그리퍼 닫힘 상태에서만 손목(joint_6) Δθ 를 NutAssembly 의
     PrismaticJoint 목표위치에 반영 → PhysxMimicJointAPI 가 회전↔하강을 조인트 구속으로
     직접 유지(우리가 계산하지 않음) → 그리퍼 열림(release/unwind) 구간은 목표를 갱신하지
     않아 자동으로 정지(게이팅) → 재파지(regrasp) → 반복.
  ⑥ JUDGE: PrismaticJoint 실제 위치 readback ≥ 목표 깊이(ENGAGE_LEN) → 체결 완료.

PhysX 아티큘레이션 토폴로지는 world.reset() 시점에 고정되어 런타임에 링크를 추가할 수
없다 — "자유롭게 파지·운반되는 강체"와 "볼트축에 조인트로 구속된 아티큘레이션 멤버"를
하나의 너트가 겸할 수 없으므로, 부득이 두 개의 너트(free_nut/screw_nut)와 그 사이의
handoff(swap)로 구성한다(build_nut_assembly() 참고). free_nut 파지는 9번과 동일한 실제
PhysX 마찰 기반(meshSimplification)이고, screw_nut 은 그리퍼로 다시 무는 것은 시각적
동작이며 실제 체결 동력은 전적으로 PrismaticJoint 드라이브 + mimic 조인트가 담당한다.

⚠ gearing 부호/크기, NutSlide↔screw_nut 앵커 오프셋, PrismaticJoint 드라이브 강성은
이론값으로 시작한 것이라 GUI 실측 튜닝이 필요하다(MIMIC_GEARING/PRISM_DRIVE_* 정의부 참고).

실행:
  /home/rokey/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh \
      /home/rokey/cobot3_ws/isaacpjt/M0609/10_thread_controller.py
  → 뷰포트에서 Play 를 누르면 실행. Stop 후 다시 Play 하면 재실행.
  환경변수 BOLT_HEADLESS=1 이면 헤드리스로 자동 실행 후 결과를 출력하고 종료한다.
"""

import os
from isaacsim import SimulationApp

_HEADLESS = os.environ.get("BOLT_HEADLESS") == "1"
simulation_app = SimulationApp({"headless": _HEADLESS})

import sys
from pathlib import Path
import numpy as np
import carb
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.prims import SingleArticulation, SingleGeometryPrim, SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "rmpflow"))
from m0609_pick_place_controller import PickPlaceController  # noqa: E402
from m0609_rmpflow_controller import RMPFlowController  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
#  [A] 로봇 상수 (6/7_connector_insertion*.py 와 동일 — 검증된 물리 설정 재사용)
# ══════════════════════════════════════════════════════════════════════════
ROBOT_USD_PATH   = str(_THIS_DIR / "Collected_m0609_camera/m0609_camera.usd")
ROBOT_URDF_PATH  = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
ROBOT_DESC_PATH  = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
RMPFLOW_CFG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")
ROBOT_PRIM_PATH  = "/World/m0609"
EE_LINK_NAME     = "link_6"
GRIPPER_JOINTS   = ["finger_joint", "right_inner_knuckle_joint"]
FINGER_LINKS     = ["left_inner_finger", "right_inner_finger"]
# RG2 그리퍼 4절 링크 전체(URDF 실측) — finger_joint 를 mimic 하는 나머지 조인트들도 포함해야
#   기구부 내부 힘 균형이 깨지지 않는다(비균일 강성 시 링크끼리 서로 밀어내는 현상 방지).
GRIPPER_MECH_JOINTS = {
    "finger_joint", "right_inner_knuckle_joint",
    "left_inner_knuckle_to_finger_joint", "right_inner_knuckle_to_finger_joint",
    "left_inner_finger_joint", "right_inner_finger_joint",
}

DRIVE_STIFFNESS, DRIVE_DAMPING, DRIVE_MAX_FORCE = 1.0e8, 1.0e6, 1.0e8
# ⚠ 실측(2026-07-20): 위치 목표(각도) 램프 방식으로 닫으면 접촉 순간 위치오차가 갑자기
#   생겨서(목표는 계속 커지는데 물체에 막혀 실제 각도는 못 따라감) 강한 servo 토크가 걸려
#   너트를 쳐서 180도 뒤집고 20cm 튕겨나감(실측 확인). → 그리퍼 기구부는 위치 PD 자체를
#   끄고(stiffness=0, 감쇠만 유지) 대신 **일정한 토크(effort)를 계속 가하는 방식**으로
#   전환한다 — 목표각이 없으니 접촉하면 반력과 자연스럽게 균형 잡히고 멈춤(충격 없음),
#   접촉 전이면 그냥 천천히 닫힘.
#   ⚠ 실측(2026-07-20): 5Nm/댐핑5 → 30Nm/댐핑1 로 6배 올려도 결과가 거의 그대로였음
#   (0.235→0.203rad, 오히려 미세감소) — 토크 크기에 비례하지 않는 저항이라는 뜻.
#   get_gains()/set_gains() 소스(articulation.py:3100대) 확인 결과, PhysX 의 USD
#   DriveAPI 는 angular(revolute) 조인트의 stiffness/damping 을 "도(deg)" 단위로 해석함
#   (rad2deg 변환 로직 존재) — 반면 우리는 "Nm/(rad/s)" 값으로 착각하고 stage 에 직접
#   썼으므로, 실제 저항이 180/π(≈57.3배) 부풀려져 있었던 것. 그만큼 낮춰서 보정한다.
#   ⚠ 실측(2026-07-20): effort(정토크) 제어로 여러 조합(5~45Nm, 댐핑 0.3~5 상당)을 시도했지만
#   전부 실패 — 낮은 토크/댐핑은 접촉 전에 너무 느려 event3 시간 내 목표각을 못 채우고,
#   충분히 빠른 토크는 접촉 순간 URDF 속도한도(2rad/s)로 그대로 부딪혀 너트를 쳐서 180도
#   뒤집고 날려버림. 감속 구간(각도 임계값 넘으면 토크 축소)도 시도했으나 4절 링크의 비선형
#   저항 때문에 타이밍이 지나치게 예민해 접촉 자체를 놓치기 일쑤였음.
#   → **낮은 강성의 위치 서보**로 전환: 목표각을 고정하고 stiffness 를 작게 주면(약한
#   가상 스프링), 오차가 클 때는 그만큼 큰 힘을 내다가 목표에 가까워지거나 접촉저항을
#   만나면 자동으로 힘이 줄어드는 "자기 감속" 특성이 있어 effort 방식보다 훨씬 안전함
#   (8번이 폭발했던 건 stiffness 가 1e8 로 극단적으로 높았기 때문이지, 위치 서보 자체가
#   문제였던게 아님). GRIPPER_DRIVE_STIFFNESS 도 finger_joint 는 deg 단위 보정 필요.
#   ⚠ 실측(2026-07-20): stiffness=15/damping=2, 목표각=1.1 조합 첫 시도로 xy=0mm, tilt=0.0도
#   (완벽) 로 pick→lift→transit→place 까지 성공했으나, place 시점 z 가 목표보다 34mm 낮음
#   — 접촉 후 잔여 파지력(stiffness*(target-current)≈15*0.12≈1.8Nm)이 약해 손 안에서 너트가
#   흘러내린 것으로 추정. 강성/목표각을 올려 파지력을 강화한다.
#   ⚠ 시도했다가 되돌림(2026-07-20): 목표각을 1.16(하드리밋 1.18 바로 아래)까지 올렸더니
#   그리퍼가 받침대를 물어버림(스크린샷 확인, 들어올리지도 못함) — RG2 는 4절 링크 회전
#   손가락이라, 목표각을 한계 가까이 밀어붙일수록 손가락 끝이 오므라들 뿐 아니라 아래로도
#   더 스윙하는 것으로 추정. 목표각은 접촉 문제 없던 1.1로 되돌리고, 파지력 부족은 목표각이
#   아닌 강성(stiffness)만 올려서 해결한다(강성↑ = 같은 각도 오차에서 더 큰 힘, 손가락 끝
#   기하학적 위치는 안 바뀜).
#   ⚠ 시도했다가 되돌림(2026-07-20): 그립점을 챔퍼에서 몸통 중앙(더 넓은 육각 단면, 실측
#   24~27.7mm)으로 옮긴 뒤, 목표각 1.1(챔퍼 기준으로 튜닝된 값)을 그대로 쓰니 lift 중 너트가
#   튕겨나감 — measure_finger_gap.py 매핑(0.90rad→30.6mm, 1.00rad→20.0mm)으로 보면 몸통
#   폭에서는 이미 ~0.91~0.94rad 부근에서 접촉이 시작되는데 목표를 1.1까지 밀어붙이니 잔여
#   오차(stiffness*오차)가 과도하게 커져 메시가 깊이 파고들었다가(interpenetration) lift
#   시작과 함께 그 에너지가 튕겨나가는 반발로 방출된 것으로 추정. 목표각을 실제 접촉각 바로
#   위로 낮추고, 댐핑도 살짝 올려 접촉 시 진동/반발을 억제한다.
#   ⚠ 시도했다가 되돌림(2026-07-20): 목표각을 1.0 으로 낮추고 댐핑도 올렸는데도 여전히
#   튕겨나감 — 진짜 원인은 목표각 크기가 아니라 "점프" 자체였음. event3 진입 즉시
#   GRIP_CLOSE_POSITION 을 통째로 명령하면(당시 current=0) 접촉 전부터 이미 큰 위치오차
#   (stiffness*목표각 전체)로 손가락이 가속되어, 접촉 시점엔 상당한 속도가 붙은 채로 너트를
#   때리는 셈 — 목표각을 아무리 줄여도 "0→목표" 점프인 한 초기 가속은 여전함. 8번
#   (8_bolt_nut_screw.py)이 이미 검증한 해법과 동일하게, 목표를 여러 스텝에 걸쳐 선형으로
#   서서히 올리는 **램프**로 전환한다 — 접촉 전 속도 자체를 억제해 충격을 없앤다.
#   ⚠ 시도했다가 되돌림(2026-07-20): "닫을 땐 약한 강성(45), 든 뒤엔 강한 강성(300)" 으로
#   전환하는 2단계 구조를 시도 — event4 진입 시 강성을 올리는 방식 자체는 맞는 방향이었지만
#   전환 램프 길이(GRIPPER_HOLD_RAMP_STEPS)를 lift(event4, 100스텝짜리 sin 커브) 모션과
#   맞추는 타이밍을 계속 못 맞춤: 60스텝→헐겁게 매달림, 200스텝→더 심하게 처짐, 20스텝(sin
#   커브 완만 구간 안에 맞춤)으로도 여전히 살짝 내려갔다가 밀려남. 전환 자체를 없애고 처음
#   부터 끝까지 **하나의 강성**으로 통일 시도(150) — 그런데 이번엔 반대로 event3(닫기)
#   접촉 순간에 터짐. 45→300으로 즉시 튀는 점프는 없앴지만, 150 자체가 이미 접촉 순간
#   힘(stiffness*잔여오차)을 예전(45) 대비 3배 이상 키워버려서 닫기 쪽이 위험해진 것 —
#   "닫기=약하게 / 든 뒤=세게"는 방향이 맞았고, 문제는 전환을 event4(lift 이미 시작된 뒤)에서
#   했다는 타이밍이었음. → 강성 전환을 event3 "안에서, 팔이 아직 안 움직이는 동안" 끝내는
#   방식으로 재도입한다: 위치 램프(GRIP_CLOSE_RAMP_STEPS)가 다 끝나 잔여오차가 이미 작아진
#   뒤에 강성만 서서히 올리면, 팔이 정지해 있는 상태라 lift 모션과 경합할 타이밍 자체가 없다.
GRIPPER_DRIVE_STIFFNESS = 45.0 / 57.29578   # 접촉/닫기용 저강성(원복)
GRIPPER_DRIVE_DAMPING = 5.0 / 57.29578
#   ⚠ 실측(2026-07-20): explosion 없이 lift 성공했지만 접촉 순간 미세한 튐으로 살짝
#   삐뚤어지게 잡힘 — 파지 강도를 낮추고(250→180) 목표각도 실제 접촉각에 더 가깝게(1.0→0.97,
#   오버트래블 축소로 접촉 시 잔여힘 자체를 줄임 = "잡는 폭을 넓힘") 같이 조정.
GRIPPER_HOLD_STIFFNESS = 180.0 / 57.29578   # event3 안에서 위치 램프 완료 후 도달할 "꽉 잡기" 강성
#   ⚠ 실측(2026-07-21): 램프 속도(GRIPPER_HOLD_RAMP_STEPS)로 처짐/폭발 트레이드오프를 못
#   풀어서(느리면 처짐, 빠르면 explosion) 댐핑을 올려본다 — 댐핑은 속도에 비례해 저항하니
#   램프 속도는 그대로 두고도 처지는/반동하는 진폭 자체를 줄일 수 있음(20→35).
GRIPPER_HOLD_DAMPING = 35.0 / 57.29578
#   ⚠ 실측(2026-07-20): 성공 확인 후 "너무 오래 기다린다"는 피드백 — 점프 방지용 램프가
#   맞지만, 비율 그대로 절반으로 압축해본다(100→50).
#   ⚠ 시도했다가 되돌림(2026-07-21): "잡고 내려갔다 올라가며 튀는" 처짐을 줄이려고 50→18로
#   더 압축했더니 오히려 explosion — 처진 상태(위치오차 커진 상태)에서 강성을 더 빨리
#   올리면 그만큼 큰 교정력이 순간적으로 걸리는 것도 결국 "점프"라 반대 효과였음. 50으로
#   되돌리고, 처짐 자체는 다른 방법(예: GRIPPER_HOLD_DAMPING 조정)으로 접근해야 함.
GRIPPER_HOLD_RAMP_STEPS = 50   # 팔이 정지해 있는 event3 안에서 진행 — lift 와 경합 없음(≈0.8초)
#   ⚠ 2026-07-22: 0.97 은 실제 접촉각(~0.91~0.94) 대비 오버트래블이 남아있어, 손가락이
#   너트 메쉬를 파고드는 정도가 커짐(파지 시 깊게 들어가 메쉬와 충돌) — 접촉각 상단
#   바로 위(0.96, GUI 실측 확정)로 낮춰 파고드는 여유를 줄인다. 파지력 부족은 여전히 stiffness(위
#   GRIPPER_HOLD_STIFFNESS)로 보강하지 목표각으로 보강하지 않는다(목표각을 올리면
#   도로 깊이 파고드는 문제로 돌아감).
GRIP_CLOSE_POSITION = 0.96   # 몸통 폭 기준 실제 접촉각(~0.91~0.94) 바로 위, 오버트래블 최소화
#   ⚠ 실측(2026-07-20): 300스텝(≈5초)은 너무 느리다는 피드백 — 지금까지 터졌던 원인은 항상
#   "0→목표 순간 점프"였지 선형 램프의 기울기 자체는 아니었으므로, 점프만 안 하면 더 가파른
#   램프(짧은 스텝 수)도 안전할 것으로 보고 줄인다.
GRIP_CLOSE_RAMP_STEPS = 75   # 150→75스텝(≈1.25초)으로 절반 단축(성공 확인 후 속도 요청)
GRIP_CLOSE_TARGET = 0.96   # 실측 매핑(measure_finger_gap.py) 상 ≈24mm 간격 — 참고용.

# ALIGN 단계(pick + 볼트 위 정렬) 이벤트 타이밍. event7(open) 진입은 가로채 SCREW 로 전환한다.
#   ⚠ event4(lift) 를 0.02(50step)→0.01(100step)로 늦춤 — M16 폭발이 lift 시작 직후
#   몇 스텝 안에 시작됐음(7_connector_insertion_real.py 의 "event4 는 절대 빠르게 하지
#   않는다" 교훈과 동일 — 지지면에서 떼어내는 구간은 급하게 하면 충격이 폭주함).
#   ⚠ 실측(2026-07-20): event3=40스텝은 턱없이 부족했음 — 그 동안 겨우 0.057→0.126rad밖에
#   못 닫힌 채(거의 열린 상태) event4(lift)로 넘어가버려서, 진짜 닫힘(→1.0rad)이 팔이 이미
#   들어올리기 시작한 뒤에 일어남 — 너트를 감쌀 기회 없이 지나쳐버림(HZ 로그로 확인).
#   event3 를 333스텝(dt=0.003, ≈5.5초)으로 크게 늘려 lift 전에 확실히 다 닫히거나
#   접촉저항으로 멈추게 한다.
#   ⚠ 실측(2026-07-20): 333스텝(2.0Nm-damp/25Nm)으로도 0.785rad(중심간격 66.6mm)까지밖에
#   못 감 — 목표(~1.0rad대, 24mm너트보다 확실히 좁은 gap)까지 평균속도 0.116rad/s 기준
#   추가로 ~600스텝 더 필요. 700스텝(dt≈0.00143)으로 늘렸었음.
#   ⚠ 실측(2026-07-20): 700스텝으로도 event3 종료 시점(0.809rad)엔 아직 접촉 전이었고,
#   실제 접촉은 event4(lift, 팔이 이미 움직이는 중)에서야 일어남 — 우연히 스쳐서 튕겨나감.
#   event3 "안에서" 감속 목표각까지 확실히 도달(팔이 안 움직이는 동안 접촉)하도록 훨씬
#   더 늘린다(감속구간 포함 총 ~1500스텝 필요 추정).
#   ⚠ 수정(2026-07-20): 위 1500스텝(≈25초)은 "점프 후 저강성 스프링이 서서히 수렴하길
#   기다리는" 방식 전제하에 필요했던 시간 — GRIP_CLOSE_RAMP_STEPS 램프로 전환하면서 더 이상
#   불필요. event3 안에서 위치 램프 + 강성 램프(위치 램프 완료 후 시작) + 정착 여유로 재계산.
#   ⚠ 실측(2026-07-20): 파지 성공 확인 후 "절반 정도로 빠르게" 요청 — 위치 램프(150→75) +
#   강성 램프(100→50) 를 절반으로 압축하면서 event3 budget 도 300→150스텝(≈2.5초)으로 맞춤
#   (75+50+25 여유).
EVENTS_DT = [0.011, 0.006, 0.05, 1.0 / 150, 0.01, 0.013, 0.003, 1.0, 0.011, 0.08]

MAX_LIN_VEL = 3.0
MAX_ANG_VEL = 20.0
# 솔버 반복수/접촉 튜닝 — IsaacLab 공식 Factory nut_thread 태스크(정밀조립 벤치마크, 우리와 동일
#   M8 볼트/너트 자산 사용)의 factory_env_cfg.py 값을 그대로 채용:
#   https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py
#   position_iteration=192("interpenetration 방지에 중요"), velocity_iteration=1,
#   contact_offset=5mm/rest_offset=0(mm 단위 나사산 규모에 맞춘 세밀한 접촉),
#   max_depenetration_velocity=5.0(겹침 해소 시 튕겨나감 방지 — v1에서 겪은 문제와 직결).
#   PHYSICS_DT(1/60)는 그대로 유지 — RMPFlow 가 1/60 적분을 전제로 튜닝돼 있어(7_connector_
#   insertion_real.py 문서화된 교훈) IsaacLab 의 1/120 을 따르면 모션 타이밍이 깨짐.
ART_POS_ITERS, ART_VEL_ITERS = 192, 1
# ⚠ 실측(2026-07-20): IsaacLab 값(5mm)을 그대로 쓰면 너트(13~15mm)의 40%에 가까운 크기라
#   콜리전 프록시가 실제 메시보다 훨씬 크게 부풀어 "떠있는" 것처럼 보임(뷰포트 육안 확인).
#   부품 규모(mm 단위)에 맞춰 1mm로 축소.
#   ⚠ 실측(2026-07-20): meshSimplification 전환 후 explosion 은 사라졌지만 접촉 시 살짝
#   "튀는" 잔여 현상 남음 — CONTACT_OFFSET 을 살짝 늘려 더 일찍(덜 파고든 상태에서) 접촉을
#   감지하게 하고, MAX_DEPENETRATION_VEL 을 크게 낮춰 겹침 해소 속도 자체를 제한한다.
CONTACT_OFFSET, REST_OFFSET = 0.002, 0.0    # 1mm→2mm, 더 일찍 접촉 감지
# ⚠ 실측(2026-07-21): pass 2 크래시(z 하강 없이 순수 회전만 할 때 나사산이 제자리에서
#   갈리는 문제) 대응으로 너트 콜리전에 rest offset 을 줘봤으나(0.5mm 는 안 터짐, 1mm 는
#   explosion) 정작 크래시 자체는 못 고쳤고, z 하강을 되살리면서 목적도 사라짐. 그 사이
#   최초 파지(ALIGN) 접촉까지 미세하게 더 튀는(삐뚤어지게 잡힘) 부작용만 남겨서 원래
#   기준값(REST_OFFSET=0.0, 어제부터 안정적으로 검증돼온 값)으로 되돌린다.
NUT_REST_OFFSET = 0.0005   # 되돌림(2026-07-21) — 너트도 REST_OFFSET(0.0)과 동일하게 유지
#   ⚠ 실측(2026-07-20): 0.3 으로 explosion 은 완전히 사라지고 실제로 들어올리는 데 처음
#   성공 — 다만 접촉 순간 미세한 튐이 아직 남아 살짝 삐뚤어지게 잡힘. 더 낮춰본다.
MAX_DEPENETRATION_VEL = 0.1                 # 0.3→0.1, 겹침 해소 속도 상한을 더 낮춤
# ⚠ 시도했다가 되돌림(2026-07-20): 볼트/너트 콜리전 근사를 자산 기본값 "sdf" 대신
#   "none"(볼트, 정적)/"convexDecomposition"(너트, 동적)으로 바꿔봤음 — GPU 코킹 부하를
#   줄이려는 의도였으나, event6(볼트에 접근하는 하강) 에서 접촉 직후 바로 폭발함
#   (tilt 0.2°→68.7° 급변, 실측 확인). SDF 가 나사산처럼 가늘고 복잡한 형상의 정밀 접촉을
#   강건하게 처리하도록 설계된 방식이라(IsaacLab Factory 가 이 시나리오에 SDF 를 쓰는 이유),
#   빼면 오히려 불안정해짐 — sdf 유지가 맞음.
PHYSICS_DT = 1.0 / 60.0


# ══════════════════════════════════════════════════════════════════════════
#  [B] 볼트/너트 정품 자산 + 체결 파라미터
# ══════════════════════════════════════════════════════════════════════════
#   ⚠ loose 공차 버전 유지 — kinematic 전환(④) 이후로는 SDF 저항이 더 이상 문제되지 않지만,
#   ①~③(ALIGN, 실제 SDF 콜리전으로 삽입 입구를 찾아 앉는 구간)에서는 여전히 실물 접촉이라
#   tight 버전은 챔퍼 근처에서 걸릴 여지가 큼 — loose 유지가 안전(Isaac/Props/Factory 에
#   m16_loose 존재 확인, probe_assets.py).
FACTORY_BOLT_REL = "/Isaac/Props/Factory/factory_bolt_m16_loose/factory_bolt_m16_loose.usd"
FACTORY_NUT_REL  = "/Isaac/Props/Factory/factory_nut_m16_loose/factory_nut_m16_loose.usd"
# ⚠ 실측(2026-07-20): M8(13~15mm)은 RG2 그리퍼 손가락 패드(원기둥형, 지름 40mm대)에 비해
#   너무 작아서 아무리 오므려도 패드 사이에 붕 떠있는 것처럼 보임(스크린샷 확인) — 콜리전
#   오프셋 문제가 아니라 그리퍼-부품 스케일 자체의 미스매치였음. M16(볼트 24x24x41mm,
#   너트 24x27.7x13mm — M8의 약 2배)으로 교체해 그리퍼 패드와 비율을 맞춘다.
#   M20(65mm 볼트)은 세로로 너무 길어서 제외.

# 볼트: 바닥에 고정(자산 자체의 root_joint 가 body0=world 로 이미 고정 저작되어 있음).
BOLT_XY     = np.array([0.45, -0.20])   # 볼트 고정 위치
BOLT_BASE_Z = 0.0                       # 참조 위치 = 볼트 로컬 원점(=베이스, 실측과 일치)
BOLT_LEN    = 0.041                     # 실측(M16): 베이스(0) ~ 나사 끝(41mm)
BOLT_TIP_Z  = BOLT_BASE_Z + BOLT_LEN

# 너트: 동적 강체로 참조 + 질량 직접 설정(자산 자체는 mass=0).
NUT_PICK_XY          = np.array([0.45, 0.20])   # 너트 초기 대기 위치
NUT_ORIGIN_TO_BOTTOM = 0.016    # 실측(M16): 너트 로컬 원점 → 바닥면까지 오프셋(16mm)
NUT_HEIGHT           = 0.013    # 실측(M16): 너트 높이(13mm)
NUT_MASS             = 0.015    # 자산 mass=0 → 직접 설정(M16 스틸너트 근사, 15g)

# 받침대(pedestal): 너트를 바닥이 아니라 살짝 띄운 대 위에 둔다.
#   ⚠ 실측(2026-07-20): 너트를 바닥(z=0)에 그대로 두면 파지 하강 시 손가락이 바닥과 충돌해
#   자세가 무너짐(스크린샷으로 확인) — 그리퍼가 내려갈 여유 공간을 만들어준다.
#   ⚠ 실측(2026-07-20): 20mm 로도 손가락 끝이 받침대를 물어버리는 문제 발생(목표각 1.16이
#   주원인으로 추정, 위 GRIP_CLOSE_POSITION 되돌림으로 대부분 해결될 것) — 추가 안전 여유로
#   높이를 늘림. 로봇과는 이미 FilteredPairsAPI 로 충돌 필터링돼 있어(아래 build_pedestal)
#   높여도 다른 부작용 없음.
PEDESTAL_HEIGHT = 0.035   # 35mm(기존 20mm 대비 +15mm 여유)
PEDESTAL_SIZE   = 0.045   # 45x45mm 발판(M16 너트 24x27.7mm 보다 넉넉하게)

# ALIGN 목표(pick + 볼트 위 정렬) — 볼트 끝보다 이만큼 위. SCREW 단계에서 이 지점부터 하강+회전.
SCREW_HOVER_CLEAR = 0.001

# ══════════════════════════════════════════════════════════════════════════
#  체결(SCREW) 파라미터 — Native Joint Thread Controller
#  회전↔하강 커플링은 더 이상 우리가 계산하지 않는다 — NutAssembly(build_nut_assembly())
#  의 PrismaticJoint(하강) + PhysxMimicJointAPI(회전 커플링)가 PhysX 솔버 레벨에서 직접
#  구속한다. 여기 상수들은 그 조인트의 지오메트리/드라이브/게이팅 파라미터다.
# ══════════════════════════════════════════════════════════════════════════
ENGAGE_LEN = 0.020    # 목표 체결 깊이 = PrismaticJoint 구간 길이(20mm)

# thread_engaged 판정(④, HANDOFF 진입 조건) 임계값 — ①②③(ALIGN, PickPlaceController+
#   실제 마찰 콜리전으로 pick+정렬+착좌까지 도달) 이후 이 조건을 만족하면 free_nut→
#   screw_nut 핸드오프로 전환.
ENGAGE_XY_TOL_M = 0.004    # 너트 중심 ↔ 볼트축 xy 오차 허용치
ENGAGE_TILT_DEG = 5.0      # 축 정렬 오차 허용치
ENGAGE_GAP_M    = 0.002    # 너트 바닥면 ↔ 볼트 끝 간 삽입 여유(이 이하면 삽입 완료로 판정)

#   ⚠ 2026-07-22: 9번(0.75=270°)에서 300°(=5/6 rev)로 늘림 — 육각너트는 60°마다 대칭인데
#   270° 는 60°×4.5(정확히 두 면 사이 한가운데)라 재파지 시 반올림으로도 정렬이 안 되는
#   구조적 문제가 있었음(GUI 실측 확인, kinematic 버전 시절). 300°=60°×5 는 매 패스 누적이
#   항상 60°의 정확한 배수가 되어 재파지 때마다 그리퍼가 육각면에 딱 맞게 물린다. joint_6
#   가동범위(±360°) 안이라 이론상 안전하지만, 270°만큼 실측 검증된 값은 아니므로 GUI 로
#   재검증할 것.
SCREW_TURNS = 300.0 / 360.0   # 패스당 목표 회전수(=300°, 60°의 배수라 재파지 정렬이 항상 맞음)
SCREW_OMEGA_DEG_S = 60.0    # 손목 각속도(도/초, 명령용) — 이 속도로 joint_6 가 실제 회전한 만큼만
                            #   (그리퍼가 닫혀있는 동안만) PrismaticJoint 목표를 전진시킨다.
SCREW_DIRECTION = 1.0       # 조임 방향 부호(오른나사 = 하강)

# 손목(joint_6) 가동범위 제한 때문에 SCREW_TURNS 한 패스로는 ENGAGE_LEN 까지 못 돈다 —
#   "래칫(ratchet)" 방식으로 여러 패스를 이어붙인다: 한 패스 다 돌리면 그리퍼를 열고
#   (그립 열림 = 게이팅으로 진행 정지) 손목만 반대로 -SCREW_TURNS*360° 언와인드 →
#   같은 자리에서 재파지 → 재회전. REGRASP_CYCLES 회 반복
#   (= 총 회전 패스 수 = 1(초기) + REGRASP_CYCLES).
REGRASP_CYCLES = 2

# ══════════════════════════════════════════════════════════════════════════
#  NutAssembly — 볼트(고정) + NutSlide(프리즘, 비가시) + screw_nut(리볼루트+mimic)
#  로 구성된 독립 PhysX 아티큘레이션. build_nut_assembly() 참고.
# ══════════════════════════════════════════════════════════════════════════
NUT_ASSEMBLY_PATH = "/World/NutAssembly"
BOLT_PATH      = f"{NUT_ASSEMBLY_PATH}/Bolt"
NUT_SLIDE_PATH = f"{NUT_ASSEMBLY_PATH}/NutSlide"
SCREW_NUT_PATH = f"{NUT_ASSEMBLY_PATH}/screw_nut"
FREE_NUT_PATH  = "/World/free_nut"   # ALIGN 에서 실제 마찰로 파지하는 자유 강체(9번과 동일 역할)

NUT_SLIDE_MASS = 0.05   # 보이지 않는 매개 링크(임의, 가벼움 — 실제 질량감은 screw_nut 이 담당)
# ⚠ 이론값 — 팔/그리퍼처럼 실측 튜닝된 값이 아니다. PrismaticJoint 가 NutSlide(질량
#   NUT_SLIDE_MASS)를 ENGAGE_LEN 구간 안에서 안정적으로 위치서보하도록 우선 넉넉히 잡음.
PRISM_DRIVE_STIFFNESS = 4000.0   # N/m (force 타입 위치 드라이브)
PRISM_DRIVE_DAMPING   = 400.0    # N·s/m
PRISM_DRIVE_MAX_FORCE = 1.0e6

#   ⚠ 실제 M16 피치(2mm/rev)로는 SCREW_TURNS(300°=5/6rev) x (1+REGRASP_CYCLES)=3패스 로
#   2.5회전 = 5mm 밖에 못 감(ENGAGE_LEN=20mm 의 25%). 8/9/구v10번처럼 "시각용 굵은 피치"로
#   키워 정확히 목표 깊이에 도달하게 한다: 20mm / 2.5rev = 8mm/rev.
NUT_PITCH_M = ENGAGE_LEN / (SCREW_TURNS * (1 + REGRASP_CYCLES))   # = 0.008 m/rev(시각용 계수)
# ⚠ mimic 관계식은 PhysX 컨벤션상 "이 조인트 각도 = gearing × 참조조인트 위치 (+offset)".
#   참조(PrismaticJoint) 위치는 미터, 이 조인트(RevoluteJoint)는 라디안이므로
#   gearing 단위는 rad/m = 2π/피치. 부호/스케일은 에셋 축 관례에 따라 달라 이론값으로
#   시작 — repo 예시(-240000)도 자체 에셋 실측치라 그대로 못 씀. GUI 로 재조정할 것
#   (반대로 돌면 이 부호만 뒤집으면 됨 — PrismaticJoint lower/upper 는 건드릴 필요 없음).
MIMIC_GEARING = -(2.0 * np.pi / NUT_PITCH_M) * SCREW_DIRECTION
# 릴리즈/재파지 시 그리퍼 목표각 램프 길이 — 기존 GRIP_CLOSE_RAMP_STEPS(닫기 램프)를 그대로
#   재사용해 점프 없이 열고/닫는다(REGRASP_POS_RAMP_STEPS 라는 별도 상수를 만들지 않음).
# 모든 패스가 끝난 뒤 그리퍼를 놓고 후퇴하는 안전 높이(ALIGN 의 end_effector_initial_height
#   와 동일 관례).
HOME_LIFT_Z = 0.30

# 파지점(그립점) = 너트 로컬 원점 기준 오프셋. PICK/PLACE 둘 다 "그립점" 기준 좌표를 쓴다
#   (6_connector_insertion.py 의 PICK_POS/PLACE_POS 관례와 동일).
#   ⚠ 시도했다가 되돌림(2026-07-20): 그립점을 바닥 쪽(바닥+10mm, 13mm 높이 너트의 상단
#   3mm 지점)으로 올렸다가 받침대 접촉 문제(GRIP_CLOSE_POSITION 1.16)는 해결됐지만, 이번엔
#   그립점이 너트 상단 챔퍼(모따기 경사면)에 걸려 손가락이 평평한 육각 몸통면을 못 물고
#   미끄러짐(스크린샷 확인 — 오래 버텨야 겨우 손가락 뿌리 쪽에 우연히 걸려 딸려 올라옴,
#   재현성 없음). PEDESTAL_HEIGHT 를 35mm 로 이미 올려둔 덕에 받침대 여유는 충분하므로,
#   그립점을 챔퍼 없는 몸통 중앙으로 되돌린다(받침대 상단 기준 6.5mm 위 — 여전히 안전).
#   ⚠ 2026-07-22: 파지 후 lift 중 그리퍼 강성 때문에 살짝 처지는(sag) 현상은 물리적으로
#   정상 — 처진 상태에서 너트/손가락이 받침대 큐브 모서리에 부딪히지 않도록, 그립점을
#   5mm 더 올려 처짐 여유를 확보한다(전에도 이 방식으로 해결한 전례와 동일).
NUT_GRASP_Z_LOCAL = NUT_ORIGIN_TO_BOTTOM + 0.045 + 0.005  # 너트 몸통 중앙 + 처짐 여유(5mm)

# 너트 초기 위치: 받침대 위(바닥면이 받침대 상단에 닿도록)
NUT_REST_ROOT_Z = PEDESTAL_HEIGHT - NUT_ORIGIN_TO_BOTTOM
PICK_POS = np.array([NUT_PICK_XY[0], NUT_PICK_XY[1], NUT_REST_ROOT_Z + NUT_GRASP_Z_LOCAL])

# ALIGN 목표: 너트 바닥면이 볼트 끝보다 SCREW_HOVER_CLEAR 만큼 위에 오도록.
NUT_ALIGN_ROOT_Z = (BOLT_TIP_Z + SCREW_HOVER_CLEAR) - NUT_ORIGIN_TO_BOTTOM
PLACE_POS = np.array([BOLT_XY[0], BOLT_XY[1], NUT_ALIGN_ROOT_Z + NUT_GRASP_Z_LOCAL])
# ⚠ 실측(2026-07-20): 용접 없이 돌려보니 grasp 이벤트 내내 EE 가 z≈0.243 에 머물렀는데,
#   PICK_POS.z(0.030)+EE_OFFSET(0.20)의 이론값은 0.230 — 실제로는 약 13mm 더 높은 곳에서
#   멈춤(이전 8번 실험들에서도 반복 관측된 편차). 그 결과 손가락이 너트(z=0.020~0.033) 위
#   허공에서 닫혀버려 전혀 접촉 못 함. 그 편차만큼 낮춰서 실제로 손가락이 너트 높이까지
#   내려가게 보정한다(용접 버전은 engage_grasp 가 실측치로 알아서 상쇄해줬지만, 이 실험은
#   진짜 접촉이 핵심이라 근사 높이 자체가 정확해야 함).
EE_OFFSET = np.array([0.0, 0.0, 0.185])

# ⚠ 래칫(REGRASP_CYCLES) + HOME 후퇴 단계 추가로 SCREW 쪽 스텝 수가 기존 대비 대폭 늘어남
#   (회전 3패스 x270스텝 + release/unwind/regrasp 왕복 2회 + home 약 225스텝 ≈ +1900) —
#   여유를 두고 상향.
MAX_HEADLESS_STEPS = 8500


# ══════════════════════════════════════════════════════════════════════════
#  [C] 헬퍼 (find_prim_path / set_all_drives / solver_iters_only 는
#            7_connector_insertion_real.py 와 동일 — 검증된 물리 설정 그대로 재사용)
# ══════════════════════════════════════════════════════════════════════════
def find_prim_path(root_path, name):
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def world_bbox_center(prim_path):
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    mn, mx = np.array(rng.GetMin()), np.array(rng.GetMax())
    return (mn + mx) / 2.0


def world_bbox_z(prim_path):
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    return float(rng.GetMin()[2]), float(rng.GetMax()[2])


def set_all_drives(root_path):
    """팔 관절 드라이브(고강성). 그리퍼 기구부는 제외 — set_gripper_drives() 가 따로 담당."""
    stage = omni.usd.get_context().get_stage()
    n = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if prim.GetName() in GRIPPER_MECH_JOINTS:
            continue
        for dt in ("angular", "linear"):
            drive = UsdPhysics.DriveAPI.Get(prim, dt)
            if drive:
                drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
                drive.GetDampingAttr().Set(DRIVE_DAMPING)
                drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)
                n += 1
    print(f"  [OK] 팔 드라이브 {n}개 설정 (stiffness={DRIVE_STIFFNESS:.0e}, 그리퍼 기구부 제외)")


_GRIPPER_DRIVE_ATTR_CACHE = []   # [(stiffness_attr, damping_attr), ...] — 아래 참고


def set_gripper_drives(root_path, stiffness=GRIPPER_DRIVE_STIFFNESS, damping=GRIPPER_DRIVE_DAMPING,
                        label="설정", verbose=True):
    """그리퍼 기구부(4절 링크) 드라이브 강성/댐핑 설정.
       ⚠ "닫을 땐 약하게/든 뒤엔 세게" 2단계 전환을 시도했으나(GRIPPER_DRIVE_STIFFNESS 정의부
       주석 참고) lift 모션과 전환 타이밍을 계속 못 맞춰 폐지 — 지금은 처음부터 끝까지 하나의
       강성(GRIPPER_DRIVE_STIFFNESS)만 쓴다. stiffness/label 인자는 reset_cycle 에서 매
       Play 마다 재설정하는 용도로 남겨둠.
       ⚠ 실측(2026-07-21): 래칫(REGRASP_CYCLES) 도입 후 강성 램프가 패스당 최대 3번 반복되며
       이 함수가 물리 스텝마다(램프 30스텝 x 최대 3회) Usd.PrimRange 전체 순회 +
       DriveAPI.Get() 를 다시 하게 됐는데, 헤드리스 검증에서 매번 정확히 pass 2 재파지
       강성램프 중(세 번째 반복) 'Cannot assign transform to non-root articulation link'
       경고와 함께 GPU 텐서 파이프라인이 멎는 게 100% 재현됨 — 반복적인 USD 속성 재저작이
       PhysX GPU 아티큘레이션 캐시와 충돌하는 것으로 추정. 드라이브 attr 핸들을 최초 1회만
       찾아 캐싱하고, 이후에는 Set() 만 호출해 매 스텝 트리 순회/재저작을 없앤다."""
    global _GRIPPER_DRIVE_ATTR_CACHE
    if not _GRIPPER_DRIVE_ATTR_CACHE:
        stage = omni.usd.get_context().get_stage()
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
            if prim.GetName() not in GRIPPER_MECH_JOINTS:
                continue
            for dt in ("angular", "linear"):
                drive = UsdPhysics.DriveAPI.Get(prim, dt)
                if drive:
                    _GRIPPER_DRIVE_ATTR_CACHE.append((drive.GetStiffnessAttr(), drive.GetDampingAttr()))
    for stiff_attr, damp_attr in _GRIPPER_DRIVE_ATTR_CACHE:
        stiff_attr.Set(stiffness)
        damp_attr.Set(damping)
    if verbose:
        print(f"  [OK] 그리퍼 드라이브 {len(_GRIPPER_DRIVE_ATTR_CACHE)}개 {label} 설정 (stiffness={stiffness:.1f})")


def filter_pair(prim_path_a, prim_path_b):
    stage = omni.usd.get_context().get_stage()
    a = stage.GetPrimAtPath(prim_path_a)
    UsdPhysics.FilteredPairsAPI.Apply(a).CreateFilteredPairsRel().AddTarget(prim_path_b)


def solver_iters_only(root_path):
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    art_prim = None
    for p in Usd.PrimRange(root):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            art_prim = p
            break
    art_prim = art_prim or root
    ap = PhysxSchema.PhysxArticulationAPI.Apply(art_prim)
    ap.CreateSolverPositionIterationCountAttr(ART_POS_ITERS)
    ap.CreateSolverVelocityIterationCountAttr(ART_VEL_ITERS)
    ap.CreateEnabledSelfCollisionsAttr(False)
    print(f"  [OK] 솔버 반복수 상향: pos={ART_POS_ITERS} vel={ART_VEL_ITERS}, selfCol=off")


def tune_physics_scene():
    """PhysicsScene 레벨 튜닝(IsaacLab Factory nut_thread 값 채용) — 씬 전체 솔버/접촉/GPU 설정.
       World() 가 자동 생성한 PhysicsScene prim 을 찾아서 적용(경로를 하드코딩하지 않고 스캔)."""
    stage = omni.usd.get_context().get_stage()
    scene_prim = None
    for p in Usd.PrimRange(stage.GetPrimAtPath("/")):
        if p.HasAPI(UsdPhysics.Scene) or p.GetTypeName() == "PhysicsScene":
            scene_prim = p
            break
    if scene_prim is None:
        print("  [경고] PhysicsScene prim을 못 찾음 — 씬 레벨 튜닝 생략")
        return
    ps = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    ps.CreateSolverTypeAttr("TGS")
    ps.CreateMaxPositionIterationCountAttr(ART_POS_ITERS)
    ps.CreateMaxVelocityIterationCountAttr(ART_VEL_ITERS)
    ps.CreateBounceThresholdAttr(0.2)
    ps.CreateFrictionOffsetThresholdAttr(0.01)
    ps.CreateFrictionCorrelationDistanceAttr(0.00625)
    ps.CreateEnableGPUDynamicsAttr(True)     # SDF 메시 콜라이더(볼트/너트)는 GPU 파이프라인 필요
    ps.CreateBroadphaseTypeAttr("GPU")
    ps.CreateGpuMaxRigidContactCountAttr(2 ** 23)
    ps.CreateGpuMaxRigidPatchCountAttr(2 ** 23)
    ps.CreateGpuCollisionStackSizeAttr(2 ** 28)
    ps.CreateGpuMaxNumPartitionsAttr(1)      # IsaacLab 주석: "안정적 시뮬레이션에 중요"
    print(f"  [OK] PhysicsScene 튜닝: solverType=TGS maxPosIter={ART_POS_ITERS} GPU dynamics on")


def axis_tilt_deg(quat_wxyz):
    """물체 로컬 +Z 축이 world +Z 축과 이루는 각(도). 자기 축(체결) 회전에는 영향받지 않는다
       (회전대칭 물체가 "빙글빙글 돌고 있음"을 "넘어짐"으로 오판하지 않기 위해 사용)."""
    w, x, y, z = [float(v) for v in quat_wxyz]
    rot = Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    local_z = rot.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
    cos_a = float(np.clip(local_z[2], -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def yaw_rotated_quat(base_wxyz, delta_deg):
    """base_wxyz 를 world +Z 축 기준 delta_deg 만큼 추가 회전시킨 쿼터니언(w,x,y,z).
       SCREW phase 목표 자세 계산에 사용(체결 회전 누적, 8_bolt_nut_screw.py 와 동일)."""
    base_q = Gf.Quatd(float(base_wxyz[0]),
                       Gf.Vec3d(float(base_wxyz[1]), float(base_wxyz[2]), float(base_wxyz[3])))
    base_rot = Gf.Rotation(base_q)
    extra_rot = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), float(delta_deg))
    combined = extra_rot * base_rot
    q = combined.GetQuat()
    return np.array([q.GetReal(), *q.GetImaginary()])


def wrap_to_pi(angle_rad):
    """각도를 [-π, π] 로 래핑 — Δθ(스텝당 회전량)는 항상 작은 값(60°/s * dt ≈ 1°/step)이라,
       joint_6 값이 어떤 이유로 랩어라운드돼 있어도 delta 자체에 이 래핑을 적용하면
       ±360° 튐 없이 안전하게 부호/크기를 얻는다."""
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def set_nut_kinematic(nut_prim_path, enabled):
    """HANDOFF(④) 시점에 free_nut 을 kinematic 으로 만들어 파킹(멀리 이동)하는 데 쓴다 —
       kinematic 이면 PhysX 가 힘을 안 가하므로 set_world_pose() 로 옮겨도 다른 물체를
       밀어내는 반발이 없다(collision 도 같이 꺼두는 _set_collision_enabled() 와 병용)."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(nut_prim_path)
    rb = UsdPhysics.RigidBodyAPI.Get(stage, nut_prim_path)
    if not rb:
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateKinematicEnabledAttr(bool(enabled))


_COLLISION_ENABLED_ATTR_CACHE = {}   # root_path -> [collisionEnabled attr, ...] — 아래 참고


def _set_collision_enabled(root_path, enabled):
    """root_path 아래 CollisionAPI 가 적용된 모든 prim 의 physics:collisionEnabled 를
       일괄 토글 — HANDOFF 시 free_nut/screw_nut 을 서로 간섭 없이 격리/노출시키는 데 쓴다.
       ⚠ 실측(2026-07-22): set_gripper_drives() 와 똑같은 이유로, Play 도중(HANDOFF swap)
       이 함수를 Usd.PrimRange 순회 + CreateCollisionEnabledAttr() 로 매번 새로 호출하면
       PhysX GPU 아티큘레이션 캐시와 충돌해 illegal memory access 로 크래시하는 것으로
       추정됨 — attr 핸들을 root_path 별로 최초 1회(설정/리셋 시점, Play 이전이라 안전)만
       찾아 캐싱하고, 이후(HANDOFF swap 등 Play 도중 호출)에는 Set() 만 호출한다."""
    global _COLLISION_ENABLED_ATTR_CACHE
    attrs = _COLLISION_ENABLED_ATTR_CACHE.get(root_path)
    if attrs is None:
        stage = omni.usd.get_context().get_stage()
        attrs = []
        for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
            if UsdPhysics.CollisionAPI(p):
                attrs.append(UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr())
        _COLLISION_ENABLED_ATTR_CACHE[root_path] = attrs
    for attr in attrs:
        attr.Set(bool(enabled))
    return len(attrs)


def _find_rigid_body_path(root_path):
    """root_path 아래에서 RigidBodyAPI 가 이미 적용된 첫 prim 을 찾는다 — Factory 볼트/너트
       자산은 내부 mesh 계층 명명이 고정돼 있지 않을 수 있어(참조만으로 자체 RigidBodyAPI/
       FixedJoint 가 이미 저작돼 있음), 이름 대신 API 존재 여부로 실제 시뮬레이션 바디를
       찾는다(find_prim_path() 의 이름 매칭과 달리 API 기반 탐색)."""
    stage = omni.usd.get_context().get_stage()
    for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if UsdPhysics.RigidBodyAPI(p):
            return str(p.GetPath())
    return None


def _strip_nested_articulation_roots(root_path, keep_path):
    """⚠ 실측(2026-07-22): IsaacLab Factory 프롭(factory_bolt_m16_loose/factory_nut_m16_loose)
       은 자체적으로 PhysicsArticulationRootAPI 를 내장하고 있다 — NutAssembly(keep_path)에
       또 하나의 ArticulationRootAPI 를 씌우면 그 안에 참조된 Bolt/screw_nut 이 "루트 안의
       또 다른 루트"가 되어 "UsdPhysics: Nested articulation roots are not allowed" 로
       world.reset() 마다 파싱이 깨졌다(GPU illegal memory access 로 이어지는 근본 원인으로
       추정). keep_path 하나만 진짜 루트로 남기고, 그 아래 참조된 자산이 내장한
       ArticulationRootAPI 는 전부 제거한다."""
    stage = omni.usd.get_context().get_stage()
    n = 0
    for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if str(p.GetPath()) == keep_path:
            continue
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if p.HasAPI(PhysxSchema.PhysxArticulationAPI):
                p.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            n += 1
    return n


def initialize_robot(robot, world):
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )


# ── 볼트/너트 정품 자산 참조 ─────────────────────────────────────────────────
def _deinstance(prim_path):
    """visuals/collisions 가 instanceable 참조라 기본 PrimRange 순회에 안 잡힘 —
       7_connector_insertion_real.py 의 _meshes_under() 와 동일하게 해제."""
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    for _ in range(6):
        changed = False
        for p in Usd.PrimRange(root):
            if p.IsInstanceable():
                p.SetInstanceable(False)
                changed = True
        if not changed:
            break


def _apply_contact_offsets(root_path, rest_offset=REST_OFFSET):
    """CollisionAPI 적용된 prim 들에 mm 스케일에 맞는 작은 contact/rest offset 적용
       (IsaacLab Factory 값: contact=5mm, rest=0 — 기본값(수cm대)은 우리 규모엔 너무 큼).
       ⚠ typeName=="Mesh" 로 제한하지 않는다 — Factory 볼트/너트 자산은 CollisionAPI 가
       Mesh 에 직접 붙지만, URDF 임포트 로봇(그리퍼 손가락 등)은 감싸는 Xform(node_STL_BINARY_)
       에 CollisionAPI 가 붙고 실제 Mesh 자식은 CollisionAPI 가 없는 다른 저작 방식이라
       Mesh 필터로는 못 찾았음(실측 2026-07-20).
       rest_offset 인자로 호출부별(너트만 더 넉넉하게 등) 오버라이드 가능 — NUT_REST_OFFSET
       정의부 주석 참고."""
    stage = omni.usd.get_context().get_stage()
    n = 0
    for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if UsdPhysics.CollisionAPI(p):
            pc = PhysxSchema.PhysxCollisionAPI.Apply(p)
            pc.CreateContactOffsetAttr(CONTACT_OFFSET)
            pc.CreateRestOffsetAttr(rest_offset)
            n += 1
    return n


def build_pedestal(world, xy, material):
    """너트를 바닥이 아니라 이 위에 올려둔다 — 파지 하강 시 손가락이 바닥과 충돌하지 않도록."""
    cx, cy = float(xy[0]), float(xy[1])
    c = world.scene.add(FixedCuboid(
        prim_path="/World/pedestal", name="pedestal",
        position=np.array([cx, cy, PEDESTAL_HEIGHT / 2.0]),
        scale=np.array([PEDESTAL_SIZE, PEDESTAL_SIZE, PEDESTAL_HEIGHT]),
        color=np.array([0.4, 0.4, 0.42]),
    ))
    c.apply_physics_material(material)
    wall = omni.usd.get_context().get_stage().GetPrimAtPath("/World/pedestal")
    UsdPhysics.FilteredPairsAPI.Apply(wall).CreateFilteredPairsRel().AddTarget(ROBOT_PRIM_PATH)
    print(f"  [OK] 받침대 @ ({cx},{cy}) {PEDESTAL_SIZE*1000:.0f}x{PEDESTAL_SIZE*1000:.0f}mm"
          f" h{PEDESTAL_HEIGHT*1000:.0f}mm (로봇충돌 필터)")


def reference_bolt(bolt_url, material, root_path=BOLT_PATH):
    """볼트: 참조만 하면 자체 root_joint(FixedJoint, body0=world)로 고정됨 — 추가 지그 불필요.
       ⚠ 실측(2026-07-21): 지금까지 볼트에는 물리 재질을 안 걸어서(에셋 기본값 그대로) 너트-
       볼트(나사산) 마찰이 통제가 안 됐음 — material 인자를 받아 명시적으로 적용한다(나사산
       "갈리는" 저항을 낮추기 위한 낮은 마찰 재질, main() 의 bolt_mat 참고).
       root_path 를 인자로 받는다 — NutAssembly(build_nut_assembly()) 하위에 위치시키기
       위함(root_joint 는 에셋 내부에 body0=world 로 이미 저작돼 있어 어느 계층에 둬도
       그대로 world 고정으로 동작)."""
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, root_path)
    add_reference_to_stage(bolt_url, root_path + "/geo")
    xf = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(root_path))
    xf.SetTranslate(Gf.Vec3d(float(BOLT_XY[0]), float(BOLT_XY[1]), float(BOLT_BASE_Z)))
    _deinstance(root_path)
    _apply_contact_offsets(root_path)
    for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if p.GetTypeName() == "Mesh" and UsdPhysics.CollisionAPI(p):
            SingleGeometryPrim(prim_path=str(p.GetPath()),
                                name=p.GetName() + "_g").apply_physics_material(material)
    print(f"  [OK] 볼트(정품자산, 고정) @ ({BOLT_XY[0]},{BOLT_XY[1]},{BOLT_BASE_Z}) 길이 {BOLT_LEN*1000:.0f}mm")


def _set_collision_approximation(root_path, approximation):
    """콜리전 근사 방식을 명시적으로 오버라이드(자산 기본값 "sdf" 대신).
       이 실험에서는 손가락-너트 접촉(간단한 블록 형태)이 핵심이라 8번(볼트-너트 나사산
       접촉)과 달리 convexDecomposition 이 오히려 더 가볍고 안정적일 수 있어 시도해본다."""
    stage = omni.usd.get_context().get_stage()
    n = 0
    for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if UsdPhysics.CollisionAPI(p):
            mca = UsdPhysics.MeshCollisionAPI.Apply(p)
            mca.CreateApproximationAttr().Set(approximation)
            n += 1
    return n


def reference_nut(nut_url, material, root_path=FREE_NUT_PATH, position_xyz=None, name="nut"):
    """너트: 동적 강체로 참조 + 질량 직접 설정(자산 자체 mass=0).
       root_path/position_xyz/name 을 받아 free_nut(ALIGN 에서 실제 마찰로 파지하는 자유
       강체)과 screw_nut(NutAssembly 멤버, build_nut_assembly() 참고) 양쪽에 재사용한다 —
       같은 M16 Factory 너트 자산이라 형상/질량/콜리전 설정이 동일해야 하므로, 위치/이름만
       다르게 호출해 중복을 없앤다."""
    if position_xyz is None:
        position_xyz = (float(NUT_PICK_XY[0]), float(NUT_PICK_XY[1]), float(NUT_REST_ROOT_Z))
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, root_path)
    add_reference_to_stage(nut_url, root_path + "/geo")
    xf = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(root_path))
    xf.SetTranslate(Gf.Vec3d(float(position_xyz[0]), float(position_xyz[1]), float(position_xyz[2])))
    _deinstance(root_path)

    body_path = f"{root_path}/geo/factory_nut"
    body_prim = stage.GetPrimAtPath(body_path)
    if not body_prim.IsValid():
        raise RuntimeError(f"너트 RigidBody prim을 못 찾음: {body_path}")
    mass_api = UsdPhysics.MassAPI.Apply(body_prim)
    mass_api.CreateMassAttr(NUT_MASS)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(body_prim)
    rb.CreateMaxLinearVelocityAttr(MAX_LIN_VEL)
    rb.CreateMaxAngularVelocityAttr(MAX_ANG_VEL)
    rb.CreateMaxDepenetrationVelocityAttr(MAX_DEPENETRATION_VEL)   # 겹침 해소 시 튕겨나감 방지
    rb.CreateSolverPositionIterationCountAttr(ART_POS_ITERS)
    rb.CreateSolverVelocityIterationCountAttr(ART_VEL_ITERS)

    for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if p.GetTypeName() == "Mesh" and UsdPhysics.CollisionAPI(p):
            SingleGeometryPrim(prim_path=str(p.GetPath()),
                                name=p.GetName() + "_g").apply_physics_material(material)
    n_off = _apply_contact_offsets(root_path, rest_offset=NUT_REST_OFFSET)
    # ⚠ 실측(2026-07-20): convexDecomposition 으로는 손가락-너트 접촉이 전혀 감지되지
    #   않았음(그리퍼가 1.18rad=거의 완전폐쇄까지 다 닫혔는데 너트 위치/자세가 한 스텝도
    #   안 변함 — 물리적으로 접촉했다면 있을 수 없는 결과). Factory 너트는 나사산이 있는
    #   오목한 형상이라 convexDecomposition 분해가 제대로 안 됐을 가능성이 큼 — 8번과
    #   동일하게 자산 기본값 SDF 로 되돌렸었음.
    # ⚠ 시도했다가 되돌림(2026-07-20): SDF 로 손가락-너트 사이의 그 수많은 "터짐"이 전부
    #   재현됐는데, GUI 에서 직접 meshSimplification 으로 바꿔보니 즉시 잡힘(실측 확인) —
    #   SDF 가 미세한 관통에도 과도한 반발력을 내는 특성이 진짜 원인이었을 가능성이 큼.
    #   meshSimplification 으로 전환해 파지(손가락-너트)는 해결됐음.
    # ⚠ 실측(2026-07-22): PhysX 는 "meshSimplification 은 동적(dynamic) 바디에 적용될 수
    #   없다"며 매번 convexHull 로 조용히 대체한다는 경고를 낸다 — 이론상으로는 그럼
    #   convexHull 과 다를 게 없어야 하고, convexHull 은 위 문단에서 "손가락-너트 접촉이
    #   전혀 감지 안 됨"이라고 실측됐던 근사라 sdf 로 바꿔봤었다. 그런데 막상 sdf 로
    #   바꾸니 lift(event4→5) 직후 너트가 x=0.45→2.37 로 순간이동하는 explosion 이 그대로
    #   재현됐다(2026-07-22 GUI 실측) — 반대로 meshSimplification(실질 convexHull) 조합은
    #   같은 GUI 에서 explosion 없이 파지·lift 가 됐다. 즉 이론적 경고와 실제 거동이
    #   불일치하는 상황 — **실측을 우선**해 meshSimplification 으로 되돌린다(어차피
    #   convexHull 로 대체되더라도, 그 대체가 일어나는 시점의 다른 조건들과 맞물려 이
    #   조합에서만 안정적이었던 것으로 보임 — 원인 규명보다 재현된 안정 조합 유지가 우선).
    _set_collision_approximation(root_path, "meshSimplification")

    print(f"  [OK] 너트({name}, 정품자산, 동적) @ {tuple(round(v,4) for v in position_xyz)}"
          f" mass={NUT_MASS*1000:.0f}g contactOffset적용={n_off}개")
    return SingleRigidPrim(body_path, name=name)


def build_nut_assembly(bolt_url, nut_url, material):
    """볼트(고정) + NutSlide(보이지 않는 프리즘 매개체) + screw_nut(리볼루트+
       PhysxMimicJointAPI) 로 구성된 독립 PhysX 아티큘레이션을 만든다 — 체결 중 회전↔하강
       커플링을 PhysX 솔버가 조인트 구속으로 직접 유지하게 해서, 이전(Δθ 추정 →
       nut.set_world_pose() 로 매 스텝 덮어쓰던 kinematic 버전)의 engage 순간 위치 점프/
       육각 위상 어긋남/폭발 문제를 구조적으로 없앤다. 참고:
       Tech-Multiverse/omniverse-nut-and-bolt-digital-twin 의 static_bolt.usda(고정 볼트 +
       PhysxMimicJointAPI:rotX) 패턴을 그대로 이식.

       screw_nut 은 처음엔 숨김+콜리전 꺼둔 채로 만들어두고(PhysX 아티큘레이션 토폴로지는
       world.reset() 시점에 고정되어 런타임에 링크를 추가할 수 없으므로, ALIGN 에서 마찰로
       파지할 자유 강체(free_nut)와 겸할 수 없음 — 부득이 2개로 분리), HANDOFF 단계에서
       표시/충돌을 켜고 재파지한다."""
    stage = omni.usd.get_context().get_stage()

    assembly_prim = UsdGeom.Xform.Define(stage, NUT_ASSEMBLY_PATH).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(assembly_prim)
    # ⚠ Bolt/NutSlide/screw_nut 이 서로 부딪히면 조인트 구속과 충돌 반발이 경합해 불안정해질
    #   수 있음(repo 의 static_bolt.usda 도 동일하게 off) — 어차피 셋은 조인트로만 연결되면
    #   충분하고 서로 실제로 맞물릴 필요가 없다.
    PhysxSchema.PhysxArticulationAPI.Apply(assembly_prim).CreateEnabledSelfCollisionsAttr(False)

    reference_bolt(bolt_url, material, root_path=BOLT_PATH)
    bolt_body_path = _find_rigid_body_path(BOLT_PATH)
    if bolt_body_path is None:
        raise RuntimeError(f"볼트 RigidBody prim을 못 찾음: {BOLT_PATH}")

    # ── NutSlide: 보이지 않는 매개 강체 — PrismaticJoint(하강)만 담당 ──
    UsdGeom.Xform.Define(stage, NUT_SLIDE_PATH)
    slide_prim = stage.GetPrimAtPath(NUT_SLIDE_PATH)
    UsdGeom.Imageable(slide_prim).MakeInvisible()
    UsdPhysics.RigidBodyAPI.Apply(slide_prim)
    PhysxSchema.PhysxRigidBodyAPI.Apply(slide_prim)
    UsdPhysics.MassAPI.Apply(slide_prim).CreateMassAttr(NUT_SLIDE_MASS)
    UsdGeom.XformCommonAPI(slide_prim).SetTranslate(
        Gf.Vec3d(float(BOLT_XY[0]), float(BOLT_XY[1]), float(BOLT_TIP_Z)))
    # ⚠ 실측(2026-07-22): NutSlide 를 질량만 있고 콜리전 형상이 전혀 없는 강체로 두었더니
    #   HANDOFF 도 건드리기 전(ALIGN 중)에 GPU 솔버가 900여 스텝 뒤 "artiSolveInternal
    #   TendonAndMimicJointConstraints1T fail to launch kernel!!" 로 죽었다 — 참고한
    #   repo(static_bolt.usda)의 NutSlide 를 다시 보니 실제로는 1mm짜리 작은 Cube 콜리전이
    #   있었는데(collisionEnabled=1) 그걸 빠뜨렸던 것. 실제로 뭔가와 부딪히길 원해서가
    #   아니라, 질량은 있는데 콜리전 형상이 아예 없는 링크를 GPU 브로드페이즈가 안정적으로
    #   못 다루는 것으로 추정 — repo 와 동일하게 작은 콜리전 큐브를 만들어준다(로봇/
    #   free_nut 과는 실제로 부딪힐 필요 없으니 filter_pair 로 배제).
    slide_cube = UsdGeom.Cube.Define(stage, f"{NUT_SLIDE_PATH}/Cube")
    slide_cube.CreateSizeAttr(1.0)
    UsdGeom.XformCommonAPI(slide_cube.GetPrim()).SetScale(Gf.Vec3f(0.001, 0.001, 0.001))
    UsdPhysics.CollisionAPI.Apply(slide_cube.GetPrim()).CreateCollisionEnabledAttr(True)
    filter_pair(NUT_SLIDE_PATH, ROBOT_PRIM_PATH)
    filter_pair(NUT_SLIDE_PATH, f"{FREE_NUT_PATH}/geo/factory_nut")

    prism = UsdPhysics.PrismaticJoint.Define(stage, f"{NUT_SLIDE_PATH}/PrismaticJoint")
    prism.CreateAxisAttr("Z")
    prism.CreateBody0Rel().AddTarget(bolt_body_path)
    prism.CreateBody1Rel().AddTarget(NUT_SLIDE_PATH)
    prism.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(BOLT_TIP_Z)))   # Bolt 로컬 기준 볼트 끝
    prism.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))                  # NutSlide 자기 원점
    prism.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    prism.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    # ⚠ position=0 이 HANDOFF 착좌 지점(볼트 끝), 음수로 갈수록 더 깊이 체결 — 부호가
    #   반대로 보이면 이 limit 대신 MIMIC_GEARING 부호만 뒤집으면 된다(정의부 주석 참고).
    prism.CreateLowerLimitAttr(-float(ENGAGE_LEN))
    prism.CreateUpperLimitAttr(0.0)
    drive = UsdPhysics.DriveAPI.Apply(prism.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateStiffnessAttr(PRISM_DRIVE_STIFFNESS)
    drive.CreateDampingAttr(PRISM_DRIVE_DAMPING)
    drive.CreateMaxForceAttr(PRISM_DRIVE_MAX_FORCE)

    # ── screw_nut: free_nut 과 동일한 Factory 너트 자산 재참조(reference_nut() 재사용) ──
    screw_nut = reference_nut(
        nut_url, material, root_path=SCREW_NUT_PATH,
        position_xyz=(float(BOLT_XY[0]), float(BOLT_XY[1]), float(BOLT_TIP_Z - NUT_ORIGIN_TO_BOTTOM)),
        name="screw_nut",
    )
    screw_nut_body = str(screw_nut.prim_path)
    # ⚠ 필수: screw_nut↔로봇(그리퍼) 콜리전을 필터링한다 — 계획대로 SCREW 단계의 재파지는
    #   순전히 시각적 동작이어야 하는데(체결 동력은 PrismaticJoint+mimic 조인트만 담당),
    #   필터를 안 걸면 그리퍼가 회전하며 문 채로 실제 마찰 토크가 screw_nut 에 전달되고,
    #   mimic 은 양방향 하드 구속(revolute_angle=gearing*prismatic_pos)이라 그 힘이 그대로
    #   PrismaticJoint 목표와 경합 — 우리가 Δθ 로 계산한 목표와 로봇이 실제로 가하는 마찰
    #   회전이 같은 조인트를 서로 다르게 끌어당겨(이중 구동) 매 스텝 불일치가 쌓이다 GPU
    #   솔버가 illegal memory access 로 죽는 것까지 실측 확인됨(2026-07-22). free_nut 과
    #   달리 여기는 필터가 반드시 필요하다.
    filter_pair(screw_nut_body, ROBOT_PRIM_PATH)

    revolute = UsdPhysics.RevoluteJoint.Define(stage, f"{SCREW_NUT_PATH}/RevoluteJoint")
    revolute.CreateAxisAttr("Z")
    revolute.CreateBody0Rel().AddTarget(NUT_SLIDE_PATH)
    revolute.CreateBody1Rel().AddTarget(screw_nut_body)
    revolute.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, -float(NUT_ORIGIN_TO_BOTTOM)))  # NutSlide 기준 너트 원점
    revolute.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))                            # screw_nut 자기 원점
    revolute.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    revolute.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    # ⚠ 실측(2026-07-22): "사실상 무제한"을 흉내내려고 lower/upperLimit 에 ±1e7 이라는
    #   거대한 유한값을 넣었더니, HANDOFF/SCREW 를 건드리기도 전에(ALIGN 도중, 이 조인트를
    #   Python 이 손도 안 댄 상태로) GPU TGS 솔버가 900여 스텝 뒤 "artiSolveInternal
    #   TendonAndMimicJointConstraints1T fail to launch kernel!!" 로 죽었다 — 아티큘레이션은
    #   존재만으로도 매 스텝 계산되므로, 이 거대한 유한 limit 이 여러 스텝 누적되며 수치적으로
    #   불안정해진 것으로 추정. limit 속성 자체를 아예 안 주면 USD Physics 관례상 기본이
    #   무제한(free)이라, 가짜 큰 숫자 대신 limit 을 저작하지 않는다.
    # ⚠ 멀티-apply 스키마 인스턴스명은 "rotX" 고정 — physics:axis="Z" 로 저작해도 PhysX 는
    #   조인트 로컬 프레임을 내부적으로 표준(X)축 관례로 재정렬하므로 mimic 인스턴스명은
    #   axis 토큰과 무관하게 항상 rotX 다(repo 의 static_bolt.usda 도 axis=Z 인데 rotX 사용 —
    #   실제 동작 확인된 예시 그대로 이식한 것이며 우리가 임의로 고른 값이 아니다).
    mimic = PhysxSchema.PhysxMimicJointAPI.Apply(revolute.GetPrim(), "rotX")
    mimic.CreateGearingAttr(float(MIMIC_GEARING))
    mimic.CreateReferenceJointRel().AddTarget(f"{NUT_SLIDE_PATH}/PrismaticJoint")

    # HANDOFF 전까지는 숨김+콜리전 꺼둠(free_nut 파지/이동에 간섭하지 않도록).
    UsdGeom.Imageable(stage.GetPrimAtPath(SCREW_NUT_PATH)).MakeInvisible()
    _set_collision_enabled(SCREW_NUT_PATH, False)

    # ⚠ Factory 볼트/너트 자산은 자체 PhysicsArticulationRootAPI 를 내장하고 있어(IsaacLab
    #   텐서 API 관례), 위에서 NutAssembly 에 씌운 루트와 충돌해 "Nested articulation roots
    #   are not allowed" 로 world.reset() 마다 파싱이 깨졌었다(2026-07-22 GUI 실측 — 이게
    #   이후 GPU illegal memory access 로 이어진 근본 원인으로 추정). NutAssembly 하나만
    #   진짜 루트로 남기고 참조된 자산이 내장한 루트는 제거한다.
    n_stripped = _strip_nested_articulation_roots(NUT_ASSEMBLY_PATH, NUT_ASSEMBLY_PATH)

    print(f"  [OK] NutAssembly: Bolt={bolt_body_path}"
          f" NutSlide(mass={NUT_SLIDE_MASS*1000:.0f}g, 구간 {ENGAGE_LEN*1000:.0f}mm)"
          f" screw_nut(mass={NUT_MASS*1000:.0f}g, gearing={MIMIC_GEARING:.0f})"
          f" 중첩 아티큘레이션 루트 제거={n_stripped}개 — 초기 숨김")
    return screw_nut


# ══════════════════════════════════════════════════════════════════════════
#  [D] 메인
# ══════════════════════════════════════════════════════════════════════════
def main():
    # 콜리전 형상(SDF 등) 뷰포트 시각화 — GUI 메뉴(Window>Physics>Debug)에서 켜는 것과
    # 동일한 carb 설정(0=끔, 2=켬). isaacsim.robot_setup.grasp_editor 의 show_physics_
    # colliders() 참고.
    carb.settings.get_settings().set_int("/persistent/physics/visualizationDisplayColliders", 2)

    world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()

    print("\n[1] 로봇 USD 로드")
    stage = omni.usd.get_context().get_stage()
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    world_prim.GetReferences().AddReference(ROBOT_USD_PATH)
    for _ in range(15):
        simulation_app.update()
    print("  [OK] robot")

    print("\n[2] 물리 설정")
    set_all_drives(ROBOT_PRIM_PATH)
    set_gripper_drives(ROBOT_PRIM_PATH)
    solver_iters_only(ROBOT_PRIM_PATH)
    tune_physics_scene()

    print("\n[3] 볼트/너트 정품자산 로드")
    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("assets_root 를 못 찾음 — 네트워크/nucleus 설정 확인 필요")
    # ⚠ 실측(2026-07-21): IsaacLab Factory 기준값(1.0/1.0)에서 나사산 "갈리는" 저항을
    #   줄이려고 0.3/0.3 으로 낮춤. 지금까지 볼트에는 재질을 아예 안 걸어서 에셋 기본값(통제
    #   불가)이 그대로 쓰이고 있었는데, 너트-볼트 마찰을 실제로 낮추려면 볼트 쪽도 낮춰야
    #   의미가 있으므로 같은 재질을 볼트에도 명시적으로 적용한다.
    part_mat = PhysicsMaterial(
        prim_path="/World/Physics_Materials/part_mat",
        static_friction=0.3, dynamic_friction=0.2, restitution=0.0,
    )
    build_pedestal(world, NUT_PICK_XY, part_mat)
    free_nut = reference_nut(assets_root + FACTORY_NUT_REL, part_mat,
                              root_path=FREE_NUT_PATH, name="free_nut")
    # ⚠ 8번과 달리 여기서는 filter_pair 를 안 건다 — 손가락↔free_nut 이 실제로 부딪히고
    #   마찰로 붙잡는지가 ALIGN(pick+정렬) 단계의 핵심이라 일부러 충돌을 살려둔다.
    screw_nut = build_nut_assembly(assets_root + FACTORY_BOLT_REL, assets_root + FACTORY_NUT_REL, part_mat)
    # ⚠ 실측(2026-07-22): solver_iters_only() 는 지금까지 ROBOT_PRIM_PATH 에만 걸려 있었다 —
    #   NutAssembly(PrismaticJoint+RevoluteJoint+PhysxMimicJointAPI, gearing≈-785) 의
    #   아티큘레이션 루트는 PhysX 기본 솔버 반복수(로봇의 192/1 대비 훨씬 낮음)로 돌고 있었던
    #   것으로 추정 — ALIGN 중(HANDOFF/SCREW 가 이 조인트를 손도 대기 전) step≈900대에서
    #   "artiSolveInternalTendonAndMimicJointConstraints1T fail to launch kernel!!" 로 GPU
    #   솔버가 죽는 게 반복 재현됨(이전에 겪은 NutSlide 콜리전 누락/revolute 거대 limit 건과
    #   같은 증상·같은 커널). 이만큼 기어링비가 큰 mimic 구속은 낮은 반복수로는 수렴이 안 돼
    #   잔차가 누적되다 발산하는 것으로 보임 — 로봇과 동일한 반복수를 걸어 검증한다.
    solver_iters_only(NUT_ASSEMBLY_PATH)

    print("\n[4] 로봇 등록")
    ee_path = find_prim_path(ROBOT_PRIM_PATH, EE_LINK_NAME)
    if ee_path is None:
        raise RuntimeError(f"'{EE_LINK_NAME}' 링크를 찾을 수 없음")
    # ⚠ joint_opened/closed_positions 는 ParallelGripper 생성자가 요구하는 필수값이지만,
    #   실제 닫기는 step_cycle 에서 joint_efforts(토크)로 직접 오버라이드하므로 여기 값은
    #   (연다 액션 등 최소 용도 외에는) 실질적으로 안 쓰인다.
    gripper = ParallelGripper(
        end_effector_prim_path=ee_path, joint_prim_names=GRIPPER_JOINTS,
        joint_opened_positions=np.array([0.0, 0.0]),
        joint_closed_positions=np.array([GRIP_CLOSE_TARGET, GRIP_CLOSE_TARGET]),
        action_deltas=None,
    )
    robot = world.scene.add(SingleManipulator(
        prim_path=ROBOT_PRIM_PATH, name="m0609_robot",
        end_effector_prim_path=ee_path, gripper=gripper,
    ))
    finger_mat = PhysicsMaterial(
        prim_path="/World/Physics_Materials/finger_mat",
        static_friction=1.8, dynamic_friction=1.4, restitution=0.0,
    )
    n_finger_off = 0
    finger_paths = []
    for ln in FINGER_LINKS:
        lp = find_prim_path(ROBOT_PRIM_PATH, ln)
        if lp:
            finger_paths.append(lp)
            SingleGeometryPrim(prim_path=lp, name=f"{ln}_geom").apply_physics_material(finger_mat)
            # ⚠ 실측(2026-07-20): 볼트/너트는 CONTACT_OFFSET 을 줄였지만(1mm) 손가락 패드는
            #   한 번도 안 건드려서 Isaac Sim 기본값(수cm대)이 그대로 남아있었음 — 뷰포트에서
            #   패드 주변에 크게 부풀어 보이는 콜리전 경계의 실제 원인. 동일하게 축소.
            _deinstance(lp)
            n_finger_off += _apply_contact_offsets(lp)
    print(f"  [OK] SingleManipulator, EE={ee_path} (손가락 contactOffset 축소={n_finger_off}개)")

    world.reset()
    initialize_robot(robot, world)
    nut_assembly = SingleArticulation(prim_path=NUT_ASSEMBLY_PATH, name="nut_assembly")
    nut_assembly.initialize(physics_sim_view=world.physics_sim_view)
    for _ in range(30):
        world.step(render=True)

    # NutAssembly 의 두 DOF(PrismaticJoint=하강, RevoluteJoint=mimic 회전) 인덱스.
    #   이름으로 찾는다 — dof_names 는 관례상 조인트 prim 이름과 일치(joint6_idx 와 동일 패턴).
    prism_idx = nut_assembly.dof_names.index("PrismaticJoint")
    revolute_idx = nut_assembly.dof_names.index("RevoluteJoint")
    print(f"  [OK] NutAssembly DOF: prism_idx={prism_idx} revolute_idx={revolute_idx}"
          f" (dof_names={nut_assembly.dof_names})")

    # ⚠ 실측(2026-07-20): GRIPPER_JOINTS 2개(finger_joint, right_inner_knuckle_joint) 모두에
    #   토크를 걸었더니 30Nm(5Nm 대비 6배)로 올려도 거의 안 빨라짐(0.235→0.203rad, 오히려 미세
    #   감소). 원인: onrobot_rg2.urdf 상 right_inner_knuckle_joint 는 실제로
    #   `<mimic joint="finger_joint" multiplier="1"/>` 슬레이브 조인트 — 독립 구동축이 아니라
    #   finger_joint 하나뿐인 진짜 구동축을 그대로 따라가게 강하게 구속돼 있음. 이 슬레이브에
    #   별도 토크를 얹으면 그 힘은 순수 이동에 안 쓰이고 구속조건을 유지하려는 내부 반력에
    #   흡수돼버림(위치제어는 목표각을 똑같이 주면 구속조건과 일치해 문제없었지만, 토크제어는
    #   다름). → 진짜 독립 구동축인 finger_joint 하나에만 토크를 건다.
    GRIP_TORQUE_JOINTS = ["finger_joint"]
    gripper_torque_indices = [robot.dof_names.index(n) for n in GRIP_TORQUE_JOINTS]
    gripper_dof_indices = [robot.dof_names.index(n) for n in GRIPPER_JOINTS]
    n_dof = len(robot.dof_names)
    # ⑤ "손목 회전량 Δθ 실측" 에 쓸 조인트 — EE 월드 쿼터니언에서 yaw 를 뽑는 대신 joint_6
    #   관절각을 직접 읽는다: 6-DOF 자세 전체에서 yaw 성분만 분리하는 계산(짐벌/랩어라운드
    #   위험)이 필요 없고, joint_6 은 URDF 가동범위(±360°대)로 이미 바운드돼 있어 값 자체가
    #   연속적이라 wrap_to_pi 는 안전망으로만 있으면 된다.
    JOINT6_NAME = "joint_6"
    joint6_idx = robot.dof_names.index(JOINT6_NAME)
    print(f"  [OK] 그리퍼 관절 인덱스={gripper_dof_indices}, 토크 인덱스={gripper_torque_indices},"
          f" joint6_idx={joint6_idx} (전체 관절수={n_dof})")

    print("\n[5] 컨트롤러 생성 (ALIGN=PickPlaceController, SCREW=RMPFlowController)")
    align_controller = PickPlaceController(
        name="m0609_nut_grasp_controller",
        gripper=robot.gripper, robot_articulation=robot,
        end_effector_initial_height=0.30, events_dt=EVENTS_DT,
        urdf_path=ROBOT_URDF_PATH, robot_description_path=ROBOT_DESC_PATH,
        rmpflow_config_path=RMPFLOW_CFG_PATH, end_effector_frame_name=EE_LINK_NAME,
    )
    screw_controller = RMPFlowController(
        name="m0609_screw_cspace_controller", robot_articulation=robot,
        urdf_path=ROBOT_URDF_PATH, robot_description_path=ROBOT_DESC_PATH,
        rmpflow_config_path=RMPFLOW_CFG_PATH, end_effector_frame_name=EE_LINK_NAME,
    )
    print(f"  [OK] PICK {np.round(PICK_POS,3)}  ALIGN(볼트 위) {np.round(PLACE_POS,3)}")
    total_turns_cfg = SCREW_TURNS * (1 + REGRASP_CYCLES)
    print(f"       체결목표: 패스당 {SCREW_TURNS*360:.0f}° x {1+REGRASP_CYCLES}패스 = {total_turns_cfg}rev 누적 회전,"
          f" pitch={NUT_PITCH_M*1000:.2f}mm/rev(시각용 계수 — NUT_PITCH_M 정의부 참고),"
          f" ENGAGE_LEN={ENGAGE_LEN*1000:.1f}mm (native joint thread controller)")

    # ── 상태머신: ALIGN(pick+정렬, free_nut 마찰 파지) → HANDOFF(swap→screw_nut 재파지)
    #    → SCREW(래칫, PrismaticJoint+mimic 이 체결 담당) → SETTLE → JUDGE → HOME ──
    phase = {"name": "ALIGN", "reported": False}

    def reset_cycle():
        # ⚠ world.reset() 은 물리 상태만 되돌리고 런타임 USD 속성 편집(kinematic/visibility/
        #   collisionEnabled)이나 kinematic 상태에서의 pose 이동은 되돌리지 않는다 — 이전
        #   Play 의 HANDOFF 로 바뀐 free_nut/screw_nut 상태를 매 Play 마다 명시적으로
        #   원상복구해야 다음 ALIGN(free_nut 동적 강체 pick)이 정상 작동한다.
        stage = omni.usd.get_context().get_stage()
        set_nut_kinematic(str(free_nut.prim_path), False)
        free_nut.set_world_pose(
            position=np.array([float(NUT_PICK_XY[0]), float(NUT_PICK_XY[1]), float(NUT_REST_ROOT_Z)]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        _set_collision_enabled(FREE_NUT_PATH, True)
        UsdGeom.Imageable(stage.GetPrimAtPath(FREE_NUT_PATH)).MakeVisible()
        _set_collision_enabled(SCREW_NUT_PATH, False)
        UsdGeom.Imageable(stage.GetPrimAtPath(SCREW_NUT_PATH)).MakeInvisible()

        world.reset()
        initialize_robot(robot, world)
        nut_assembly.initialize(physics_sim_view=world.physics_sim_view)
        nut_assembly.apply_action(ArticulationAction(
            joint_positions=np.array([0.0]), joint_indices=np.array([prism_idx])))
        for _ in range(30):
            world.step(render=True)
        # ⚠ world.reset() 은 물리 상태만 되돌리고 런타임에 Set() 한 USD 드라이브 속성값은 안
        #   되돌리므로, 매 Play 마다 명시적으로 재설정(2단계 강성 전환은 폐지했지만 안전하게 유지).
        set_gripper_drives(ROBOT_PRIM_PATH, GRIPPER_DRIVE_STIFFNESS, GRIPPER_DRIVE_DAMPING, label="재설정")
        align_controller.reset()
        screw_controller.reset()
        phase.clear()
        phase.update(name="ALIGN", reported=False)
        np0, nq0 = free_nut.get_world_pose()
        print(f"  [DBG] 정착 후(그립 전) free_nut_pos={np.round(np.asarray(np0),3)}"
              f" tilt={axis_tilt_deg(np.asarray(nq0)):.1f}deg")

    def step_cycle():
        """한 물리 스텝만큼 상태머신을 전진시킨다: ALIGN → HANDOFF → SCREW → SETTLE → JUDGE → HOME."""
        def apply_grip_hold():
            """그리퍼를 계속 닫힌 채로 유지. SCREW 단계에서는 순전히 시각적 역할—
               체결 자체(회전↔하강)는 NutAssembly 의 PrismaticJoint+mimic 조인트가 담당하므로
               그립력은 물리적으로 무관하다. ALIGN event3 안에서 이미 위치/강성 램프가
               끝났으므로 목표각만 계속 명령."""
            grip_action = ArticulationAction(
                joint_positions=np.array([GRIP_CLOSE_POSITION, GRIP_CLOSE_POSITION]),
                joint_indices=np.array(gripper_dof_indices),
            )
            robot.apply_action(grip_action)

        if phase["name"] == "ALIGN":
            phase["_step"] = phase.get("_step", 0) + 1
            ev = align_controller.get_current_event()
            if ev != phase.get("_last_ev", -1):
                np_dbg, nq_dbg = free_nut.get_world_pose()
                eep_dbg, _ = robot.end_effector.get_world_pose()
                gj = robot.gripper.get_joint_positions()
                gv = robot.get_joint_velocities()
                ge = robot.get_measured_joint_efforts()
                gae = robot.get_applied_joint_efforts()
                print(f"  [DBG] step={phase['_step']} ev={ev} free_nut={np.round(np.asarray(np_dbg),3)}"
                      f" EE={np.round(np.asarray(eep_dbg),3)}"
                      f" tilt={axis_tilt_deg(np.asarray(nq_dbg)):.1f}deg"
                      f" gripper_joints={np.round(np.asarray(gj),3) if gj is not None else None}"
                      f" finger_vel={gv[gripper_torque_indices[0]]:.4f}"
                      f" finger_meas_eff={ge[gripper_torque_indices[0]]:.4f}"
                      f" finger_applied_eff={gae[gripper_torque_indices[0]] if gae is not None else None}")
                if finger_paths:
                    fz = [world_bbox_z(fp) for fp in finger_paths]
                    nut_z0, nut_z1 = world_bbox_z(str(free_nut.prim_path))
                    fc = [world_bbox_center(fp) for fp in finger_paths]
                    nc = world_bbox_center(str(free_nut.prim_path))
                    print(f"  [HZ] step={phase['_step']} ev={ev} finger_z={np.round(fz,4).tolist()}"
                          f" nut_z=[{nut_z0:.4f},{nut_z1:.4f}]"
                          f" finger_xy={[np.round(c[:2],4).tolist() for c in fc]}"
                          f" nut_xy={np.round(nc[:2],4).tolist()}")
                phase["_last_ev"] = ev

            if ev >= 3 and phase["_step"] % 15 == 0:
                gv = robot.get_joint_velocities()
                gj = robot.gripper.get_joint_positions()
                print(f"  [TRC] step={phase['_step']} ev={ev} finger_pos={gj[0]:.4f}"
                      f" finger_vel={gv[gripper_torque_indices[0]]:.4f}")

            if ev >= 7:
                # ⚠ PickPlaceController 가 여기서 그리퍼를 열려고 함 — 8/9번과 동일하게
                #   가로채서 open 대신 HANDOFF 로 전환한다.
                sp, sq = robot.end_effector.get_world_pose()
                phase["start_pos"] = np.asarray(sp).copy()
                phase["start_quat"] = np.asarray(sq).copy()
                np_, nq_ = free_nut.get_world_pose()
                np_ = np.asarray(np_); nq_ = np.asarray(nq_)

                # ④ thread_engaged 판정 — ①②③(ALIGN)이 이미 free_nut 을 볼트축 위
                #   SCREW_HOVER_CLEAR 만큼의 간격까지 내려놨으므로 이 시점의 실측치가
                #   임계값 이내인지 확인만 하고(설계상 항상 통과해야 정상 — 실패 시 PLACE_POS/
                #   SCREW_HOVER_CLEAR 쪽 기하를 재점검할 신호) HANDOFF 로 넘어간다.
                nut_bottom_z = float(np_[2]) + NUT_ORIGIN_TO_BOTTOM
                gap_m = BOLT_TIP_Z - nut_bottom_z
                xy_err = float(np.linalg.norm(np_[:2] - BOLT_XY))
                tilt = axis_tilt_deg(nq_)
                engaged_ok = (xy_err <= ENGAGE_XY_TOL_M) and (tilt <= ENGAGE_TILT_DEG) and (gap_m <= ENGAGE_GAP_M)
                print(f"  [ENGAGE 판정] xy오차={xy_err*1000:.2f}mm(허용{ENGAGE_XY_TOL_M*1000:.1f})"
                      f" 기울기={tilt:.1f}deg(허용{ENGAGE_TILT_DEG:.1f}) gap={gap_m*1000:.2f}mm"
                      f"(허용{ENGAGE_GAP_M*1000:.1f}) → thread_engaged={engaged_ok}")

                phase["name"] = "HANDOFF"
                phase["handoff_sub"] = "open"
                print(f"  [ALIGN 완료] EE={np.round(phase['start_pos'],3)} → HANDOFF 시작")
                return

            action = align_controller.forward(
                picking_position=PICK_POS, placing_position=PLACE_POS,
                current_joint_positions=robot.get_joint_positions(),
                end_effector_offset=EE_OFFSET,
            )
            robot.apply_action(action)
            if ev >= 3:
                # ⚠ 실측(2026-07-20): effort(정토크) 제어는 여러 조합을 실측해봐도 전부 실패
                #   (자세한 이유는 GRIPPER_DRIVE_STIFFNESS 정의부 주석 참고) — 낮은 강성의
                #   위치 서보로 전환. ArticulationAction 하나에 팔+그리퍼를 같이 욱여넣으면
                #   PickPlaceController 이벤트별 반환 배열 길이가 달라 인덱스 오류가 나므로,
                #   여기서도 그리퍼 전용 ArticulationAction 을 별도로 만들어 분리 적용한다
                #   (joint_indices 로 서브셋 지정 가능 — set_joint_efforts 와 동일한 패턴).
                # ⚠ 시도했다가 되돌림(2026-07-20): event3 진입 즉시 GRIP_CLOSE_POSITION 을
                #   통째로 명령(0→목표 점프)했더니 목표각/댐핑을 아무리 바꿔도 lift 중 계속
                #   튕겨나감 — 접촉 전부터 큰 위치오차로 손가락이 가속되어 접촉 시점에 이미
                #   속도가 붙은 채 부딪히는 게 원인으로 추정(8_bolt_nut_screw.py 가 이미 같은
                #   문제를 램프로 해결한 전례와 동일). event3 동안만 GRIP_CLOSE_RAMP_STEPS 에
                #   걸쳐 0→목표로 선형 램프, 이후(event4+)는 목표 고정.
                if ev == 3:
                    phase["_close_ramp_step"] = phase.get("_close_ramp_step", 0) + 1
                    ramp_frac = min(phase["_close_ramp_step"] / GRIP_CLOSE_RAMP_STEPS, 1.0)
                    grip_target = ramp_frac * GRIP_CLOSE_POSITION
                else:
                    ramp_frac = 1.0
                    grip_target = GRIP_CLOSE_POSITION
                grip_action = ArticulationAction(
                    joint_positions=np.array([grip_target, grip_target]),
                    joint_indices=np.array(gripper_dof_indices),
                )
                robot.apply_action(grip_action)
                # ⚠ 위치 램프가 다 끝나(ramp_frac>=1.0, 잔여오차가 이미 작아진 뒤) 강성만
                #   event3 "안에서" 서서히 올린다 — 팔이 아직 정지해 있는 구간이라 lift(event4)
                #   모션과 타이밍이 경합할 일이 없다(이전엔 이 전환을 event4 진입 시점에 해서
                #   lift 모션과 타이밍이 계속 안 맞았음).
                if ramp_frac >= 1.0 and not phase.get("_hold_ramp_done"):
                    phase["_hold_ramp_step"] = phase.get("_hold_ramp_step", 0) + 1
                    hf = min(phase["_hold_ramp_step"] / GRIPPER_HOLD_RAMP_STEPS, 1.0)
                    cur_stiff = GRIPPER_DRIVE_STIFFNESS + hf * (GRIPPER_HOLD_STIFFNESS - GRIPPER_DRIVE_STIFFNESS)
                    cur_damp = GRIPPER_DRIVE_DAMPING + hf * (GRIPPER_HOLD_DAMPING - GRIPPER_DRIVE_DAMPING)
                    set_gripper_drives(ROBOT_PRIM_PATH, cur_stiff, cur_damp, label="꽉잡기램프", verbose=(hf >= 1.0))
                    if hf >= 1.0:
                        phase["_hold_ramp_done"] = True

        elif phase["name"] == "HANDOFF":
            # ALIGN 종료 위치(start_pos/start_quat)에서 그리퍼만 열고/닫으며 free_nut→
            #   screw_nut 로 손을 바꿔 무는 단계 — 팔은 이동하지 않는다(같은 자리).
            phase["_handoff_step"] = phase.get("_handoff_step", 0) + 1
            sub = phase.get("handoff_sub", "open")

            if sub == "open":
                phase["_open_step"] = phase.get("_open_step", 0) + 1
                rf = min(phase["_open_step"] / GRIP_CLOSE_RAMP_STEPS, 1.0)
                release_target = (1.0 - rf) * GRIP_CLOSE_POSITION
                grip_action = ArticulationAction(
                    joint_positions=np.array([release_target, release_target]),
                    joint_indices=np.array(gripper_dof_indices),
                )
                robot.apply_action(grip_action)
                action = screw_controller.forward(
                    target_end_effector_position=phase["start_pos"],
                    target_end_effector_orientation=phase["start_quat"],
                )
                robot.apply_action(action)
                if rf >= 1.0:
                    phase["handoff_sub"] = "swap"
                return

            if sub == "swap":
                # free_nut 격리(kinematic 전환 + 먼 곳으로 이동 + 숨김) → screw_nut 노출
                #   (보이기만). PhysX 아티큘레이션 토폴로지는 world.reset() 시점에 고정돼
                #   런타임에 링크를 추가할 수 없어(파일 상단 docstring 참고) 하나의 너트로
                #   "자유 파지"와 "조인트 구속 체결"을 겸할 수 없다 — 이 한 스텝에서 손
                #   바꿔치기를 전부 처리한다.
                # ⚠ 실측(2026-07-22): 여기서 screw_nut 의 collisionEnabled 를 Play 도중에
                #   토글(off→on)한 직후 곧바로 nut_assembly 텐서를 건드리니 GPU illegal
                #   memory access 로 크래시했다 — set_gripper_drives() 가 이미 겪은 것과
                #   같은 유형(PhysX GPU 아티큘레이션 캐시 충돌 추정). screw_nut 은 애초에
                #   실제 콜리전이 필요 없다(이동은 PrismaticJoint+mimic 조인트가 전담,
                #   로봇과는 filter_pair, Bolt와는 enabledSelfCollisions=False 로 이미
                #   배제됨) — 그래서 콜리전은 build_nut_assembly() 에서 꺼놓은 채로
                #   영구히 유지하고, 여기서는 더 이상 토글하지 않는다(보이기만 한다).
                #   free_nut 쪽도 같은 이유로 콜리전 토글 없이 kinematic 전환 + 이동만으로
                #   충분히 격리된다.
                stage = omni.usd.get_context().get_stage()
                set_nut_kinematic(str(free_nut.prim_path), True)
                free_nut.set_world_pose(
                    position=np.array([float(NUT_PICK_XY[0]), float(NUT_PICK_XY[1]), -0.5]),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                )
                UsdGeom.Imageable(stage.GetPrimAtPath(FREE_NUT_PATH)).MakeInvisible()

                UsdGeom.Imageable(stage.GetPrimAtPath(SCREW_NUT_PATH)).MakeVisible()

                phase["handoff_sub"] = "close"
                phase["_close_step"] = 0
                print("  [HANDOFF] free_nut 격리 → screw_nut 노출 → 재파지 시작")
                return

            if sub == "close":
                phase["_close_step"] = phase.get("_close_step", 0) + 1
                rf = min(phase["_close_step"] / GRIP_CLOSE_RAMP_STEPS, 1.0)
                grip_target = rf * GRIP_CLOSE_POSITION
                grip_action = ArticulationAction(
                    joint_positions=np.array([grip_target, grip_target]),
                    joint_indices=np.array(gripper_dof_indices),
                )
                robot.apply_action(grip_action)
                action = screw_controller.forward(
                    target_end_effector_position=phase["start_pos"],
                    target_end_effector_orientation=phase["start_quat"],
                )
                robot.apply_action(action)
                if rf >= 1.0:
                    # SCREW 상태 초기화 — PrismaticJoint 목표를 0(핸드오프 착좌 지점)으로
                    #   명시 재확인(reset_cycle 에서도 하지만 안전하게 중복 확인).
                    nut_assembly.apply_action(ArticulationAction(
                        joint_positions=np.array([0.0]), joint_indices=np.array([prism_idx])))
                    phase["prismatic_target_m"] = 0.0
                    phase["prev_joint6"] = float(robot.get_joint_positions()[joint6_idx])
                    phase["theta_deg"] = 0.0
                    phase["wrist_base_deg"] = 0.0
                    phase["pass_idx"] = 0
                    phase["name"] = "SCREW"
                    phase["screw_sub"] = "rotate"
                    print("  [HANDOFF 완료] screw_nut 재파지 → SCREW 시작")
                return

        elif phase["name"] == "SCREW":
            # ⚠ 체결(회전↔하강 커플링)은 더 이상 우리가 계산하지 않는다 — NutAssembly 의
            #   PrismaticJoint(목표위치 드라이브) + PhysxMimicJointAPI 가 조인트 구속으로
            #   직접 담당한다(build_nut_assembly() 참고). 우리는 "그리퍼가 닫혀 있는 동안
            #   (=이 rotate 서브상태)에만" 손목 실측 회전량(Δθ)만큼 PrismaticJoint 목표를
            #   전진시킬 뿐이다 — release/unwind 서브상태에서는 이 갱신 자체를 안 하므로
            #   목표가 그대로 유지돼(force 타입 위치 드라이브가 계속 그 자리를 붙잡음) 너트가
            #   자동으로 멈춘다(그립 게이팅).
            # ⚠ joint_6 가동범위 때문에 SCREW_TURNS(300°) 한 패스로는 ENGAGE_LEN 까지 못
            #   돈다 — 래칫 서브상태머신(rotate→release→unwind→regrasp→rotate...)으로
            #   REGRASP_CYCLES 회 반복.
            phase["_screw_step"] = phase.get("_screw_step", 0) + 1
            sub = phase.get("screw_sub", "rotate")

            if sub == "rotate":
                phase["theta_deg"] += SCREW_OMEGA_DEG_S * PHYSICS_DT
                pass_done = phase["theta_deg"] >= SCREW_TURNS * 360.0
                theta = min(phase["theta_deg"], SCREW_TURNS * 360.0)

                # Δθ = 손목(joint_6) 실측 회전량 → 조임 방향(SCREW_DIRECTION)으로 실제
                #   회전한 양만 PrismaticJoint 목표에 누적(역회전/노이즈는 버림). 그립이
                #   닫혀있는 이 서브상태 안에서만 실행되므로 그 자체가 게이팅이다.
                cur_joint6 = float(robot.get_joint_positions()[joint6_idx])
                d_theta = wrap_to_pi(cur_joint6 - phase["prev_joint6"]) * SCREW_DIRECTION
                phase["prev_joint6"] = cur_joint6
                if d_theta > 0.0:
                    phase["prismatic_target_m"] = max(
                        phase["prismatic_target_m"] - NUT_PITCH_M * d_theta / (2.0 * np.pi),
                        -ENGAGE_LEN,
                    )
                    nut_assembly.apply_action(ArticulationAction(
                        joint_positions=np.array([phase["prismatic_target_m"]]),
                        joint_indices=np.array([prism_idx]),
                    ))

                # EE(로봇)는 최초 파지 위치(start_pos)를 기준으로 PrismaticJoint 의 "실제"
                #   위치(명령값이 아니라 readback — 드라이브 추종 지연이 있어도 항상 시각적
                #   으로 일치)만큼 낮춰 너트가 내려가는 걸 미러링한다.
                prism_pos_m = float(nut_assembly.get_joint_positions()[prism_idx])
                depth_m = -prism_pos_m
                target_pos = phase["start_pos"].copy()
                target_pos[2] = phase["start_pos"][2] + prism_pos_m
                # 방향(orientation)은 최초 파지 자세(start_quat) + wrist_base_deg(패스 기준
                #   각) 기준 — 언와인드로 손목을 매번 그 기준까지 되돌린 뒤 다시 도는 것이라
                #   패스 내에서는 theta 가 누적되지 않는다. wrist_base_deg 자체는 매 패스 끝
                #   (unwind 완료 시점)에 screw_nut 의 실측 회전각(육각 60° 대칭)에 맞춰
                #   갱신된다 — 아래 unwind 블록 참고.
                target_quat = yaw_rotated_quat(phase["start_quat"], phase["wrist_base_deg"] + SCREW_DIRECTION * theta)
                action = screw_controller.forward(
                    target_end_effector_position=target_pos,
                    target_end_effector_orientation=target_quat,
                )
                robot.apply_action(action)
                apply_grip_hold()

                if phase["_screw_step"] % 15 == 0 or phase["_screw_step"] <= 3:
                    eep_dbg, _ = robot.end_effector.get_world_pose()
                    revolute_deg = float(np.degrees(nut_assembly.get_joint_positions()[revolute_idx]))
                    gj = robot.gripper.get_joint_positions()
                    print(f"  [DBG-SCREW] pass={phase['pass_idx']} step={phase['_screw_step']} theta={theta:.0f}"
                          f" depth={depth_m*1000:.2f}mm/{ENGAGE_LEN*1000:.1f}mm joint6={cur_joint6:.3f}rad"
                          f" nut_revolute={revolute_deg:.0f}deg EE_z={float(np.asarray(eep_dbg)[2])*1000:.1f}mm"
                          f" gripper_joints={np.round(np.asarray(gj),3) if gj is not None else None}")

                if pass_done:
                    if depth_m >= ENGAGE_LEN or phase["pass_idx"] >= REGRASP_CYCLES:
                        phase["name"] = "SETTLE"
                        phase["settle_steps"] = 0
                        print(f"  [SCREW 전체 완료] 총 {phase['pass_idx']+1}패스,"
                              f" 체결깊이={depth_m*1000:.2f}mm(목표 {ENGAGE_LEN*1000:.1f}mm) → 정착 대기")
                    else:
                        eep, _ = robot.end_effector.get_world_pose()
                        phase["pass_end_pos"] = np.asarray(eep).copy()
                        phase["screw_sub"] = "release"
                        phase["_release_step"] = 0
                        print(f"  [SCREW pass {phase['pass_idx']} 완료] 깊이={depth_m*1000:.2f}mm"
                              f" → 그리퍼 릴리즈 → 언와인드 시작")
                return

            if sub == "release":
                # 회전 끝난 자세/위치를 유지한 채 그리퍼만 서서히 연다(점프 방지, 닫을 때와
                #   동일한 GRIP_CLOSE_RAMP_STEPS 램프 길이 재사용). 그립이 열리는 순간부터
                #   rotate 블록을 안 지나므로 PrismaticJoint 목표 갱신이 멈춰 너트도 자동으로
                #   정지한다(게이팅).
                phase["_release_step"] = phase.get("_release_step", 0) + 1
                rf = min(phase["_release_step"] / GRIP_CLOSE_RAMP_STEPS, 1.0)
                release_target = (1.0 - rf) * GRIP_CLOSE_POSITION
                grip_action = ArticulationAction(
                    joint_positions=np.array([release_target, release_target]),
                    joint_indices=np.array(gripper_dof_indices),
                )
                robot.apply_action(grip_action)
                hold_quat = yaw_rotated_quat(phase["start_quat"],
                                              phase["wrist_base_deg"] + SCREW_DIRECTION * SCREW_TURNS * 360.0)
                action = screw_controller.forward(
                    target_end_effector_position=phase["pass_end_pos"],
                    target_end_effector_orientation=hold_quat,
                )
                robot.apply_action(action)
                if rf >= 1.0:
                    phase["screw_sub"] = "unwind"
                return

            if sub == "unwind":
                # 그립이 열려 진행이 멈춘 채로, 손목만 -300° 되돌아간다(이번 패스의
                #   wrist_base_deg 기준으로). z 는 pass_end_pos 로 고정(그 사이 하강 없음 —
                #   PrismaticJoint 목표를 안 건드리므로 조인트 구속상 실제로도 안 움직인다).
                phase["theta_deg"] -= SCREW_OMEGA_DEG_S * PHYSICS_DT
                unwind_done = phase["theta_deg"] <= 0.0
                theta = max(phase["theta_deg"], 0.0)
                target_quat = yaw_rotated_quat(phase["start_quat"], phase["wrist_base_deg"] + SCREW_DIRECTION * theta)
                action = screw_controller.forward(
                    target_end_effector_position=phase["pass_end_pos"],
                    target_end_effector_orientation=target_quat,
                )
                robot.apply_action(action)
                grip_action = ArticulationAction(
                    joint_positions=np.array([0.0, 0.0]),
                    joint_indices=np.array(gripper_dof_indices),
                )
                robot.apply_action(grip_action)
                if unwind_done:
                    # ⚠ 육각 너트는 60°마다 대칭 — 다음 패스에서 그리퍼가 다시 닫힐 때 납작한
                    #   면을 물게 하려면, screw_nut 의 "실측"(mimic 조인트 readback) 회전각
                    #   에서 가장 가까운 60°의 배수로 wrist_base_deg 를 재정렬한다.
                    #   SCREW_TURNS 를 60°의 배수(300°)로 잡아둔 덕에 이 재정렬은 사실상 항상
                    #   잔차 0(수치오차 안전망 성격) — kinematic 버전 시절엔 270°(=4x60°+30°)
                    #   라 이 재정렬을 해도 30° 어긋났었다(GUI 실측 확인된 원인).
                    nut_theta_deg = float(np.degrees(nut_assembly.get_joint_positions()[revolute_idx]))
                    phase["wrist_base_deg"] = round(nut_theta_deg / 60.0) * 60.0
                    phase["screw_sub"] = "regrasp"
                    phase["_regrasp_step"] = 0
                    phase["prev_joint6"] = float(robot.get_joint_positions()[joint6_idx])
                    print(f"  [SCREW pass {phase['pass_idx']} 언와인드 완료] 재파지 시작"
                          f" (wrist_base={phase['wrist_base_deg']:.0f}deg 로 재정렬)")
                return

            if sub == "regrasp":
                # 그리퍼만 서서히 재폐쇄 — screw_nut 은 조인트로 구속돼 있어 9번의
                #   regrasp_pos/regrasp_stiff 2단계 강성 램프가 필요 없다(그립은 순전히
                #   시각적 역할). 방향은 위에서 재정렬한 wrist_base_deg 로.
                phase["_regrasp_step"] = phase.get("_regrasp_step", 0) + 1
                rf = min(phase["_regrasp_step"] / GRIP_CLOSE_RAMP_STEPS, 1.0)
                grip_target = rf * GRIP_CLOSE_POSITION
                grip_action = ArticulationAction(
                    joint_positions=np.array([grip_target, grip_target]),
                    joint_indices=np.array(gripper_dof_indices),
                )
                robot.apply_action(grip_action)
                target_quat = yaw_rotated_quat(phase["start_quat"], phase["wrist_base_deg"])
                action = screw_controller.forward(
                    target_end_effector_position=phase["pass_end_pos"],
                    target_end_effector_orientation=target_quat,
                )
                robot.apply_action(action)
                if rf >= 1.0:
                    phase["pass_idx"] += 1
                    phase["theta_deg"] = 0.0
                    phase["screw_sub"] = "rotate"
                    phase["prev_joint6"] = float(robot.get_joint_positions()[joint6_idx])
                    print(f"  [SCREW pass {phase['pass_idx']} 재파지 완료] 재회전 시작")
                return

        elif phase["name"] == "SETTLE":
            apply_grip_hold()
            phase["settle_steps"] = phase.get("settle_steps", 0) + 1
            if phase["settle_steps"] >= 20:
                phase["name"] = "JUDGE"

        elif phase["name"] == "JUDGE" and not phase["reported"]:
            pos, quat = screw_nut.get_world_pose()
            pos = np.asarray(pos); quat = np.asarray(quat)
            prism_pos_m = float(nut_assembly.get_joint_positions()[prism_idx])
            depth_mm = -prism_pos_m * 1000.0
            revolute_deg = float(np.degrees(nut_assembly.get_joint_positions()[revolute_idx]))
            # 체결 깊이 판정 = "볼트 끝(나사 시작점)보다 너트 바닥면이 얼마나 아래로 내려갔는가"(mm)
            #   — PrismaticJoint readback 기준. 조인트로 구속돼 있으므로 너트 실제 world
            #   pose 로 계산한 값과도 사실상 일치해야 정상(어긋나면 조인트 앵커 산식 버그 신호).
            nut_bottom_z = float(pos[2]) + NUT_ORIGIN_TO_BOTTOM
            engagement_mm = (BOLT_TIP_Z - nut_bottom_z) * 1000.0
            xy_err = float(np.linalg.norm(pos[:2] - BOLT_XY))
            tilt = axis_tilt_deg(quat)
            gj = robot.gripper.get_joint_positions()
            success = (engagement_mm >= 3.0) and (xy_err < 0.010) and (tilt < 15.0)
            print("\n" + "=" * 60)
            print(f"[결과] 너트 최종 위치 = {np.round(pos, 4)}")
            print(f"       패스 수 = {phase['pass_idx']+1} (상한 {1+REGRASP_CYCLES})")
            print(f"       체결 깊이(PrismaticJoint readback) = {depth_mm:.2f}mm / 목표"
                  f" {ENGAGE_LEN*1000:.1f}mm ({100.0*depth_mm/(ENGAGE_LEN*1000.0):.0f}%)")
            print(f"       체결 깊이(볼트 끝 기준, 너트 world pose) = {engagement_mm:.2f}mm (위 값과 일치해야 정상)")
            print(f"       screw_nut 누적 회전(mimic readback) = {revolute_deg:.0f}deg")
            print(f"       볼트 xy 오차 = {xy_err*1000:.1f}mm,  기울기 = {tilt:.1f}도"
                  f" (조인트 구속이라 0에 가까워야 정상)")
            print(f"       최종 그리퍼 관절값 = {np.round(np.asarray(gj),3) if gj is not None else None}"
                  f" (그리퍼는 시각적 유지 역할만 — 체결 자체와는 무관)")
            print(f"       체결 성공 = {success}")
            print("=" * 60 + "\n")
            phase["reported"] = True
            eep, _ = robot.end_effector.get_world_pose()
            phase["home_base_pos"] = np.asarray(eep).copy()
            phase["name"] = "HOME"

        elif phase["name"] == "HOME":
            # 모든 패스 종료 후: 그리퍼를 놓고(릴리즈 램프) 제자리에서 안전 높이(HOME_LIFT_Z)
            #   까지 들어올린다. 너트는 이미 SETTLE/JUDGE 로 정착된 상태라 그냥 놓으면 됨
            #   (8번처럼 용접이 없으므로 별도 release_grasp 절차 불필요).
            phase["_home_step"] = phase.get("_home_step", 0) + 1
            if not phase.get("_home_release_done"):
                rf = min(phase["_home_step"] / GRIP_CLOSE_RAMP_STEPS, 1.0)
                release_target = (1.0 - rf) * GRIP_CLOSE_POSITION
                grip_action = ArticulationAction(
                    joint_positions=np.array([release_target, release_target]),
                    joint_indices=np.array(gripper_dof_indices),
                )
                robot.apply_action(grip_action)
                action = screw_controller.forward(
                    target_end_effector_position=phase["home_base_pos"],
                    target_end_effector_orientation=phase["start_quat"],
                )
                robot.apply_action(action)
                if rf >= 1.0:
                    phase["_home_release_done"] = True
                    phase["_home_step"] = 0
                return
            target_pos = phase["home_base_pos"].copy()
            target_pos[2] = HOME_LIFT_Z
            action = screw_controller.forward(
                target_end_effector_position=target_pos,
                target_end_effector_orientation=phase["start_quat"],
            )
            robot.apply_action(action)
            if phase["_home_step"] >= 150 and not phase.get("_home_reported"):
                phase["_home_reported"] = True
                print("  [HOME] 후퇴 완료 — 실험 종료")

    # ── 실행: GUI(Play 엣지) / 헤드리스(자동) ──────────────────────────────
    if _HEADLESS:
        print("\n[6] 헤드리스 자동 실행\n")
        reset_cycle()
        for _ in range(MAX_HEADLESS_STEPS):
            step_cycle()
            world.step(render=False)
            if phase.get("_home_reported"):
                break
        simulation_app.close()
        return

    print("\n[6] 준비 완료 — 뷰포트에서 Play 를 누르면 파지+체결 실험을 실행합니다.")
    print("     (Stop 후 다시 Play 하면 처음부터 재실행)\n")
    was_playing = False
    while simulation_app.is_running():
        world.step(render=True)
        playing = world.is_playing()

        if playing and not was_playing:
            print("[Play] 파지+체결 실험 시작")
            reset_cycle()

        if playing:
            step_cycle()

        was_playing = playing

    simulation_app.close()


if __name__ == "__main__":
    main()

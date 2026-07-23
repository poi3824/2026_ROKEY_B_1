"""
9_nut_grasp_experiment.py ─ "진짜 콜리전(마찰) 기반 파지" + 체결(SCREW) 실험 (8_bolt_nut_screw.py 분기)

8번은 손가락↔너트 충돌을 필터링하고 FixedJoint 용접으로 붙잡는 방식(안정성 우선)인데,
이 파일은 그 필터/용접을 걷어내고 **손가락이 실제로 너트를 부딪혀 마찰로 붙잡는지**를
먼저 검증했고(성공 확인, 2026-07-20), 이제 8번의 SCREW(체결) 단계를 이식해 실제 마찰
그립을 유지한 채로 볼트에 돌려 끼우는 것까지 확장한다. ALIGN(PickPlaceController)에서
event7(그리퍼 open) 진입 직전을 가로채 release 를 막고 SCREW phase 로 전환 — 8번과 동일한
구조지만, 용접이 없으므로 SCREW 도중에도 매 스텝 그립 목표(GRIP_CLOSE_POSITION)를 계속
직접 명령해 마찰만으로 계속 붙잡고 있어야 한다(회전 토크에 마찰이 버티는지가 이번 실험의
핵심 — 8번은 용접이라 애초에 미끄러질 수가 없어서 이 문제 자체가 없었음).

이번 실험에서 8번과 다르게 켜놓은 것:
  - filter_pair(너트, 로봇) 제거 → 손가락-너트가 실제로 충돌/마찰함.
  - 그리퍼 기구부(4절 링크)만 별도 저강성 드라이브 적용 → 접촉 충격을 유연하게 흡수
    (v1 실험 때 균일 고강성으로 시도해 매번 폭발했던 것과의 차이점).
  - GRIP_CLOSE_POSITION 은 event3 안에서 위치 램프 → 강성 램프 순으로 도달(점프 방지,
    자세한 이력은 아래 GRIPPER_DRIVE_STIFFNESS/GRIP_CLOSE_POSITION 정의부 주석 참고).
  - 콜리전 근사는 SDF 대신 meshSimplification(실측으로 explosion 원인이었음을 확인,
    reference_nut() 주석 참고).

joint_6 가동범위 한계로 SCREW_TURNS(270°) 한 패스로는 목표 체결 깊이까지 못 가서, "래칫"
방식으로 확장했다(2026-07-21): 270° 회전 → 그리퍼 릴리즈 → 같은 자리에서 손목만 -270°
언와인드(너트는 볼트에 물려 그대로 있음) → 재파지 → 다시 270° 회전, 이 릴리즈~재회전
사이클을 REGRASP_CYCLES(=2)회 반복(총 회전 패스 = 3) 후 그리퍼를 놓고 HOME_LIFT_Z 높이로
후퇴한다. step_cycle() 의 SCREW 분기가 screw_sub 서브상태(grip_ramp→rotate→release→
unwind→regrasp_pos→regrasp_stiff→...) 로 이를 구현한다.

실행:
  /home/rokey/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh \
      /home/rokey/cobot3_ws/isaacpjt/M0609/9_nut_grasp_experiment.py
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
from isaacsim.core.prims import SingleGeometryPrim, SingleRigidPrim
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
GRIP_CLOSE_POSITION = 0.97   # 몸통 폭 기준 실제 접촉각(~0.91~0.94) 바로 위 — 1.0보다 여유를 더
                            #   둬 오버트래블(잔여힘)을 줄임(파지력은 위 stiffness 로 보강).
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
#   ⚠ 실측(2026-07-21): tight 공차(나사산 간극 거의 없음) 버전을 쓰고 있었는데, SCREW 단계는
#   실제 스레드 조인트가 아니라 회전+하강 커플링 근사(SCREW_DESCENT_PER_TURN_M)라 tight 메시가
#   서로 파고들며 만드는 저항 토크가 비정상적으로 커서 그리퍼 마찰 한계를 넘어 너트가 계속
#   미끄러져 뒤틀렸을 가능성 — 공차가 넉넉한 loose 버전으로 교체해 저항 자체를 줄인다
#   (Isaac/Props/Factory 에 m16_loose 존재 확인, probe_assets.py).
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

# 체결(SCREW) 커플링 — 8_bolt_nut_screw.py 와 동일한 회전각↔하강량 운동학적 근사(PhysX 에
#   나사 조인트가 없어 실제 피치 대신 시각적으로 확인 가능한 가상 계수 사용, 8번 주석 참고).
#   ⚠ 시도했다가 되돌림(2026-07-21): 그리퍼 슬립이 애매해 보인다고 커플링을 아예 껐더니,
#   축 방향 도피 없이 제자리에서 나사산끼리 계속 갈리기만 하는 게 더 부자연스러웠음(육안
#   확인, "갈리는 느낌") — 하강 커플링을 다시 넣는다.
SCREW_DESCENT_PER_TURN_M = 0.002   # 회전 1턴(360°)당 하강량(가상, 시각적 체결감 확보용)
SCREW_TURNS       = 0.75      # 패스당 목표 회전수(8번 실측 — joint_6 가동범위 한계 전 안전 구간)
# ⚠ 실측(2026-07-20): 튕겨나가진 않았지만(SDF 복귀로 해결) 회전 중 마찰 그립이 토크를 못
#   버텨서 손 안에서 너트가 살짝 미끄러져 돌며 비스듬해짐 — 8번은 용접이라 없던 문제.
#   회전 속도를 낮춰 순간 토크 요구량 자체를 줄인다(120→60°/s, 총 소요시간은 늘어남).
SCREW_OMEGA_DEG_S = 60.0      # 각속도(도/초) — 120→60, 순간 토크 요구량 축소
SCREW_DIRECTION   = 1.0       # 조임 방향 부호(8번과 동일)
# SCREW 진입 시 그립을 한 번 더 세게(회전 토크 저항용) — 이미 ALIGN 에서 수렴/정착된
#   상태에서 강성만 더 올리는 거라 위치오차가 작아 점프 위험이 낮음(그래도 짧게는 램프).
#   ⚠ 실측(2026-07-21): 400/30 조합으로도 회전 중 마찰 그립이 토크를 못 버텨 너트가
#   손 안에서 미끄러져 비틀어짐 — 파지 강성/댐핑을 한 단계 더 올린다(400→600, 30→45,
#   비율은 유지). 그래도 재발하면 SCREW_OMEGA_DEG_S(현재 60)를 추가로 낮출 것.
SCREW_GRIP_STIFFNESS = 600.0 / 57.29578
SCREW_GRIP_DAMPING = 45.0 / 57.29578
SCREW_GRIP_RAMP_STEPS = 30

# 손목(joint_6) 가동범위 제한 때문에 SCREW_TURNS(270°) 한 패스로는 목표 체결 깊이까지
#   못 간다 — "래칫(ratchet)" 방식으로 여러 패스를 이어붙인다: 한 패스 다 돌리면 그립을
#   놓고(릴리즈) 손목만 반대로 -270° 언와인드(너트는 볼트에 물려 있으므로 그대로 있음),
#   같은 자리에서 다시 파지한 뒤 +270° 재회전. REGRASP_CYCLES 회 반복(= 총 회전 패스 수는
#   1(초기) + REGRASP_CYCLES).
REGRASP_CYCLES = 2
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
NUT_GRASP_Z_LOCAL = NUT_ORIGIN_TO_BOTTOM + 0.045  # 너트 몸통 중앙(≈6.5mm 위)

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


def reference_bolt(bolt_url, material):
    """볼트: 참조만 하면 자체 root_joint(FixedJoint, body0=world)로 고정됨 — 추가 지그 불필요.
       ⚠ 실측(2026-07-21): 지금까지 볼트에는 물리 재질을 안 걸어서(에셋 기본값 그대로) 너트-
       볼트(나사산) 마찰이 통제가 안 됐음 — material 인자를 받아 명시적으로 적용한다(나사산
       "갈리는" 저항을 낮추기 위한 낮은 마찰 재질, main() 의 bolt_mat 참고)."""
    stage = omni.usd.get_context().get_stage()
    root_path = "/World/bolt"
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


def reference_nut(nut_url, material):
    """너트: 동적 강체로 참조 + 질량 직접 설정(자산 자체 mass=0)."""
    stage = omni.usd.get_context().get_stage()
    root_path = "/World/nut"
    UsdGeom.Xform.Define(stage, root_path)
    add_reference_to_stage(nut_url, root_path + "/geo")
    xf = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(root_path))
    xf.SetTranslate(Gf.Vec3d(float(NUT_PICK_XY[0]), float(NUT_PICK_XY[1]), float(NUT_REST_ROOT_Z)))
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
    # ⚠ 시도했다가 되돌림(2026-07-20): SCREW 단계 추가 후 문제 발견 — meshSimplification 이
    #   너트 안쪽 나사산(암나사) 디테일까지 뭉개버려서, 볼트(여전히 sdf, 정밀 나사산 유지)와
    #   맞물릴 때 뭉개진 거친 폴리곤면에 부딪혀 살짝 튕겨나옴. 손가락 접촉(바깥 육각면)엔
    #   meshSimplification 이 필요하고 볼트 접촉(안쪽 나사산)엔 sdf 가 필요한데, 메쉬 하나에
    #   근사 방식은 하나만 적용 가능해 동시에 둘 다는 안 됨. 그 사이 램프/댐핑/CONTACT_OFFSET/
    #   MAX_DEPENETRATION_VEL 을 많이 다듬어놨으니, SDF로 되돌려도 손가락 쪽 explosion이
    #   재발 안 하는지 테스트한다(재발하면 meshSimplification 으로 다시 돌리고 SCREW 하강
    #   속도를 늦추는 등 다른 방식으로 접근).
    _set_collision_approximation(root_path, "sdf")

    print(f"  [OK] 너트(정품자산, 동적) @ ({NUT_PICK_XY[0]},{NUT_PICK_XY[1]},{NUT_REST_ROOT_Z:.4f})"
          f" mass={NUT_MASS*1000:.0f}g contactOffset적용={n_off}개")
    return SingleRigidPrim(body_path, name="nut")


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
    reference_bolt(assets_root + FACTORY_BOLT_REL, part_mat)
    build_pedestal(world, NUT_PICK_XY, part_mat)
    nut = reference_nut(assets_root + FACTORY_NUT_REL, part_mat)
    # ⚠ 8번과 달리 여기서는 filter_pair 를 안 건다 — 손가락↔너트가 실제로 부딪히고
    #   마찰로 붙잡는지가 이번 실험의 핵심이라 일부러 충돌을 살려둔다.

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
    for _ in range(30):
        world.step(render=True)

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
    print(f"  [OK] 그리퍼 관절 인덱스={gripper_dof_indices}, 토크 인덱스={gripper_torque_indices}"
          f" (전체 관절수={n_dof})")

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
    print(f"       체결목표: 패스당 {SCREW_TURNS}턴 x {1+REGRASP_CYCLES}패스 = {total_turns_cfg}턴 누적 회전,"
          f" 하강계수 {SCREW_DESCENT_PER_TURN_M*1000:.1f}mm/턴,"
          f" 패스당 하강목표 {SCREW_TURNS*SCREW_DESCENT_PER_TURN_M*1000:.2f}mm")

    # ── 상태머신: ALIGN(pick+lift+이동+정렬) → SCREW(하강+회전, 마찰 그립 유지) → SETTLE → JUDGE ──
    phase = {"name": "ALIGN", "reported": False}

    def reset_cycle():
        world.reset()
        initialize_robot(robot, world)
        for _ in range(30):
            world.step(render=True)
        # ⚠ world.reset() 은 물리 상태만 되돌리고 런타임에 Set() 한 USD 드라이브 속성값은 안
        #   되돌리므로, 매 Play 마다 명시적으로 재설정(2단계 강성 전환은 폐지했지만 안전하게 유지).
        set_gripper_drives(ROBOT_PRIM_PATH, GRIPPER_DRIVE_STIFFNESS, GRIPPER_DRIVE_DAMPING, label="재설정")
        align_controller.reset()
        screw_controller.reset()
        phase.clear()
        phase.update(name="ALIGN", reported=False)
        np0, nq0 = nut.get_world_pose()
        print(f"  [DBG] 정착 후(그립 전) nut_pos={np.round(np.asarray(np0),3)}"
              f" tilt={axis_tilt_deg(np.asarray(nq0)):.1f}deg")

    def step_cycle():
        """한 물리 스텝만큼 상태머신을 전진시킨다: ALIGN → SCREW → SETTLE → JUDGE."""
        def apply_grip_hold():
            """SCREW/SETTLE 단계에서 마찰 그립을 계속 유지. ALIGN event3 안에서 이미
               위치/강성 램프가 끝났으므로 여기서는 그냥 목표각을 계속 명령만 한다 —
               용접이 없으므로 매 스텝 이걸 빼먹으면 그립이 그 순간 풀려버린다."""
            grip_action = ArticulationAction(
                joint_positions=np.array([GRIP_CLOSE_POSITION, GRIP_CLOSE_POSITION]),
                joint_indices=np.array(gripper_dof_indices),
            )
            robot.apply_action(grip_action)

        if phase["name"] == "ALIGN":
            phase["_step"] = phase.get("_step", 0) + 1
            ev = align_controller.get_current_event()
            if ev != phase.get("_last_ev", -1):
                np_dbg, nq_dbg = nut.get_world_pose()
                eep_dbg, _ = robot.end_effector.get_world_pose()
                gj = robot.gripper.get_joint_positions()
                gv = robot.get_joint_velocities()
                ge = robot.get_measured_joint_efforts()
                gae = robot.get_applied_joint_efforts()
                print(f"  [DBG] step={phase['_step']} ev={ev} nut={np.round(np.asarray(np_dbg),3)}"
                      f" EE={np.round(np.asarray(eep_dbg),3)}"
                      f" tilt={axis_tilt_deg(np.asarray(nq_dbg)):.1f}deg"
                      f" gripper_joints={np.round(np.asarray(gj),3) if gj is not None else None}"
                      f" finger_vel={gv[gripper_torque_indices[0]]:.4f}"
                      f" finger_meas_eff={ge[gripper_torque_indices[0]]:.4f}"
                      f" finger_applied_eff={gae[gripper_torque_indices[0]] if gae is not None else None}")
                if finger_paths:
                    fz = [world_bbox_z(fp) for fp in finger_paths]
                    nut_z0, nut_z1 = world_bbox_z(str(nut.prim_path))
                    fc = [world_bbox_center(fp) for fp in finger_paths]
                    nc = world_bbox_center(str(nut.prim_path))
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
                # ⚠ PickPlaceController 가 여기서 그리퍼를 열려고 함 — 8_bolt_nut_screw.py 와
                #   동일하게 가로채서 open 대신 SCREW 로 전환한다. 용접이 없으므로 release_grasp
                #   같은 절차는 불필요 — apply_grip_hold() 를 계속 불러서 마찰로만 계속 붙잡는다.
                sp, sq = robot.end_effector.get_world_pose()
                phase["start_pos"] = np.asarray(sp).copy()
                phase["start_quat"] = np.asarray(sq).copy()
                np_, _ = nut.get_world_pose()
                phase["start_nut_z"] = float(np.asarray(np_)[2])
                phase["theta_deg"] = 0.0
                phase["name"] = "SCREW"
                print(f"  [ALIGN 완료] EE={np.round(phase['start_pos'],3)} → SCREW 시작"
                      f" (마찰 그립만으로 유지 — 8번과 달리 용접 없음)")
                apply_grip_hold()
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

        elif phase["name"] == "SCREW":
            # ⚠ 8번은 FixedJoint 용접이라 회전 토크에 미끄러질 수가 없었지만, 여기는 순수
            #   마찰이라 손목을 돌리는 동안 그립이 버티는지가 이 확장의 핵심 검증 포인트.
            # ⚠ joint_6 가동범위 때문에 SCREW_TURNS(270°) 한 패스로는 목표 깊이까지 못 감 —
            #   "래칫" 서브상태머신으로 확장(rotate → release → unwind → regrasp_pos →
            #   regrasp_stiff → rotate ... REGRASP_CYCLES 회 반복). sub 가 없으면 최초
            #   진입이므로 기존 grip_ramp 부터 시작.
            phase["_screw_step"] = phase.get("_screw_step", 0) + 1
            sub = phase.get("screw_sub", "grip_ramp")

            if sub == "grip_ramp":
                # ⚠ lift 때와 같은 교훈 — 강성 전환은 팔(이번엔 회전)이 아직 움직이기 전,
                #   정지한 상태에서 끝내야 타이밍 경합이 없다. 여기서도 EE 를 start_pos/quat
                #   그대로 유지한 채 그립만 먼저 세게 램프하고, 다 끝난 뒤에 회전을 시작한다.
                phase["_screw_grip_ramp_step"] = phase.get("_screw_grip_ramp_step", 0) + 1
                sf = min(phase["_screw_grip_ramp_step"] / SCREW_GRIP_RAMP_STEPS, 1.0)
                cur_stiff = GRIPPER_HOLD_STIFFNESS + sf * (SCREW_GRIP_STIFFNESS - GRIPPER_HOLD_STIFFNESS)
                cur_damp = GRIPPER_HOLD_DAMPING + sf * (SCREW_GRIP_DAMPING - GRIPPER_HOLD_DAMPING)
                set_gripper_drives(ROBOT_PRIM_PATH, cur_stiff, cur_damp, label="체결그립램프", verbose=(sf >= 1.0))
                apply_grip_hold()
                action = screw_controller.forward(
                    target_end_effector_position=phase["start_pos"],
                    target_end_effector_orientation=phase["start_quat"],
                )
                robot.apply_action(action)
                if sf >= 1.0:
                    phase["screw_sub"] = "rotate"
                    phase["pass_idx"] = 0
                    phase["theta_deg"] = 0.0
                    phase["total_theta_deg"] = 0.0
                    phase["pass_base_pos"] = phase["start_pos"].copy()
                return

            if sub == "rotate":
                phase["theta_deg"] += SCREW_OMEGA_DEG_S * PHYSICS_DT
                pass_done = phase["theta_deg"] >= SCREW_TURNS * 360.0
                theta = min(phase["theta_deg"], SCREW_TURNS * 360.0)
                frac = theta / 360.0
                target_pos = phase["pass_base_pos"].copy()
                target_pos[2] = phase["pass_base_pos"][2] - frac * SCREW_DESCENT_PER_TURN_M
                # ⚠ 방향(orientation)은 언제나 최초 파지 자세(start_quat) 기준 — 각 패스는
                #   언와인드로 손목을 매번 그 원위치까지 되돌린 뒤 다시 도는 것이라, 패스별
                #   기준 자세가 누적되지 않는다(z 만 패스마다 더 내려감).
                target_quat = yaw_rotated_quat(phase["start_quat"], SCREW_DIRECTION * theta)
                action = screw_controller.forward(
                    target_end_effector_position=target_pos,
                    target_end_effector_orientation=target_quat,
                )
                robot.apply_action(action)
                apply_grip_hold()
                if phase["_screw_step"] % 15 == 0 or phase["_screw_step"] <= 3:
                    np_dbg, nq_dbg = nut.get_world_pose()
                    eep_dbg, _ = robot.end_effector.get_world_pose()
                    tilt_dbg = axis_tilt_deg(np.asarray(nq_dbg))
                    gj = robot.gripper.get_joint_positions()
                    xy_slip_mm = float(np.linalg.norm(np.asarray(np_dbg)[:2] - np.asarray(eep_dbg)[:2])) * 1000.0
                    print(f"  [DBG-SCREW] pass={phase['pass_idx']} step={phase['_screw_step']} theta={theta:.0f}"
                          f" target_z={target_pos[2]*1000:.1f}mm nut={np.round(np.asarray(np_dbg),3)}"
                          f" tilt={tilt_dbg:.1f} EE_z={float(np.asarray(eep_dbg)[2])*1000:.1f}mm"
                          f" gripper_joints={np.round(np.asarray(gj),3) if gj is not None else None}"
                          f" EE-nut_xy거리={xy_slip_mm:.1f}mm(그립 유지 확인용)")
                if pass_done:
                    phase["total_theta_deg"] += theta
                    if phase["pass_idx"] >= REGRASP_CYCLES:
                        phase["name"] = "SETTLE"
                        phase["settle_steps"] = 0
                        print(f"  [SCREW 전체 완료] 총 {phase['pass_idx']+1}패스,"
                              f" 누적회전={phase['total_theta_deg']:.0f}° → 정착 대기")
                    else:
                        eep, _ = robot.end_effector.get_world_pose()
                        phase["pass_end_pos"] = np.asarray(eep).copy()
                        phase["screw_sub"] = "release"
                        phase["_release_step"] = 0
                        print(f"  [SCREW pass {phase['pass_idx']} 완료] 그리퍼 릴리즈 → 언와인드 시작")
                return

            if sub == "release":
                # 회전 끝난 자세/위치를 유지한 채 그리퍼만 서서히 연다(점프 방지, 닫을 때와
                #   동일한 GRIP_CLOSE_RAMP_STEPS 램프 길이 재사용).
                phase["_release_step"] = phase.get("_release_step", 0) + 1
                rf = min(phase["_release_step"] / GRIP_CLOSE_RAMP_STEPS, 1.0)
                release_target = (1.0 - rf) * GRIP_CLOSE_POSITION
                grip_action = ArticulationAction(
                    joint_positions=np.array([release_target, release_target]),
                    joint_indices=np.array(gripper_dof_indices),
                )
                robot.apply_action(grip_action)
                hold_quat = yaw_rotated_quat(phase["start_quat"], SCREW_DIRECTION * SCREW_TURNS * 360.0)
                action = screw_controller.forward(
                    target_end_effector_position=phase["pass_end_pos"],
                    target_end_effector_orientation=hold_quat,
                )
                robot.apply_action(action)
                if rf >= 1.0:
                    phase["screw_sub"] = "unwind"
                return

            if sub == "unwind":
                # 그리퍼는 열려 있으므로 너트는 그 자리(볼트에 물린 채)에 남고, 손목만
                #   -270° 되돌아간다. z 는 pass_end_pos 로 고정(그 사이 하강 없음).
                phase["theta_deg"] -= SCREW_OMEGA_DEG_S * PHYSICS_DT
                unwind_done = phase["theta_deg"] <= 0.0
                theta = max(phase["theta_deg"], 0.0)
                target_quat = yaw_rotated_quat(phase["start_quat"], SCREW_DIRECTION * theta)
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
                    phase["screw_sub"] = "regrasp_pos"
                    phase["_regrasp_pos_step"] = 0
                    print(f"  [SCREW pass {phase['pass_idx']} 언와인드 완료] 재파지 시작")
                return

            if sub == "regrasp_pos":
                # ALIGN event3 와 동일한 패턴: 위치 램프 먼저(점프 방지), 강성 램프는 그 다음.
                phase["_regrasp_pos_step"] = phase.get("_regrasp_pos_step", 0) + 1
                rf = min(phase["_regrasp_pos_step"] / GRIP_CLOSE_RAMP_STEPS, 1.0)
                grip_target = rf * GRIP_CLOSE_POSITION
                grip_action = ArticulationAction(
                    joint_positions=np.array([grip_target, grip_target]),
                    joint_indices=np.array(gripper_dof_indices),
                )
                robot.apply_action(grip_action)
                action = screw_controller.forward(
                    target_end_effector_position=phase["pass_end_pos"],
                    target_end_effector_orientation=phase["start_quat"],
                )
                robot.apply_action(action)
                if rf >= 1.0:
                    phase["screw_sub"] = "regrasp_stiff"
                    phase["_regrasp_stiff_step"] = 0
                return

            if sub == "regrasp_stiff":
                phase["_regrasp_stiff_step"] = phase.get("_regrasp_stiff_step", 0) + 1
                sf = min(phase["_regrasp_stiff_step"] / SCREW_GRIP_RAMP_STEPS, 1.0)
                cur_stiff = GRIPPER_HOLD_STIFFNESS + sf * (SCREW_GRIP_STIFFNESS - GRIPPER_HOLD_STIFFNESS)
                cur_damp = GRIPPER_HOLD_DAMPING + sf * (SCREW_GRIP_DAMPING - GRIPPER_HOLD_DAMPING)
                set_gripper_drives(ROBOT_PRIM_PATH, cur_stiff, cur_damp,
                                    label=f"재파지램프(pass{phase['pass_idx']+1})", verbose=(sf >= 1.0))
                apply_grip_hold()
                action = screw_controller.forward(
                    target_end_effector_position=phase["pass_end_pos"],
                    target_end_effector_orientation=phase["start_quat"],
                )
                robot.apply_action(action)
                if sf >= 1.0:
                    phase["pass_idx"] += 1
                    phase["theta_deg"] = 0.0
                    phase["pass_base_pos"] = phase["pass_end_pos"].copy()
                    phase["screw_sub"] = "rotate"
                    print(f"  [SCREW pass {phase['pass_idx']} 재파지 완료] 재회전 시작")
                return

        elif phase["name"] == "SETTLE":
            apply_grip_hold()
            phase["settle_steps"] = phase.get("settle_steps", 0) + 1
            if phase["settle_steps"] >= 20:
                phase["name"] = "JUDGE"

        elif phase["name"] == "JUDGE" and not phase["reported"]:
            pos, quat = nut.get_world_pose()
            pos = np.asarray(pos); quat = np.asarray(quat)
            descent_mm = (phase["start_nut_z"] - float(pos[2])) * 1000.0
            # 체결 깊이 판정 = "볼트 끝(나사 시작점)보다 너트 바닥면이 얼마나 아래로 내려갔는가"(mm)
            #   (8_bolt_nut_screw.py 와 동일한 판정 방식 — 명령량 대비가 아니라 절대 깊이로 판단,
            #   실물 나사산 SDF/meshSimplification 접촉이 저항으로 작용해 명령보다 덜 내려가는
            #   게 정상/오히려 바람직한 신호이기 때문).
            nut_bottom_z = float(pos[2]) + NUT_ORIGIN_TO_BOTTOM
            engagement_mm = (BOLT_TIP_Z - nut_bottom_z) * 1000.0
            xy_err = float(np.linalg.norm(pos[:2] - BOLT_XY))
            tilt = axis_tilt_deg(quat)
            gj = robot.gripper.get_joint_positions()
            success = (engagement_mm >= 3.0) and (xy_err < 0.010) and (tilt < 15.0)
            total_turns = 1 + REGRASP_CYCLES
            print("\n" + "=" * 60)
            print(f"[결과] 너트 최종 위치 = {np.round(pos, 4)}")
            print(f"       회전 누적 = {phase['total_theta_deg']:.0f}°"
                  f" (목표 {total_turns}패스 x {SCREW_TURNS*360:.0f}° = {total_turns*SCREW_TURNS*360:.0f}°)")
            print(f"       하강 이동량 = {descent_mm:.2f}mm (명령량 대비 — 접촉저항으로 일부만 진행 가능)")
            print(f"       체결 깊이(볼트 끝 기준) = {engagement_mm:.2f}mm")
            print(f"       볼트 xy 오차 = {xy_err*1000:.1f}mm,  기울기 = {tilt:.1f}도")
            print(f"       최종 그리퍼 관절값 = {np.round(np.asarray(gj),3) if gj is not None else None}"
                  f" (마찰 그립이 회전 내내 버텼는지는 위 [DBG-SCREW] 의 EE-nut_xy거리로 확인)")
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

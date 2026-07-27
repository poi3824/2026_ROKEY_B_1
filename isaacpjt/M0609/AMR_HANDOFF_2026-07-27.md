# AMR 통합 작업 인수인계서 (2026-07-27)

이 문서는 오늘 구현한 Isaac Sim AMR 주행, 공장 배경, 실행 방법과 이후 비전·팔·그리퍼
기능을 연결하는 방법을 정리한 팀 공용 인수인계서다.

## 0. 가장 중요한 현재 상태

> **현재 실제로 동작하는 것은 AMR 이동뿐이다.**
>
> 화면과 로그에 `비전 스캔 중`, `버스바 파지 중`, `버스바·너트 체결 중`이라고 표시되지만,
> 지금은 해당 위치에서 **2초 동안 정지한 뒤 다음 단계로 넘어가는 임시 STUB**이다.
> 카메라 인식, 로봇팔 이동, 그리퍼 파지, 버스바 삽입, 너트 체결이 실행되는 것이 아니다.

흔히 `time.sleep(2)`로 표현하는 임시 대기 동작이지만, 현재 코드는 ROS 콜백 전체를 멈추는
`time.sleep()`을 직접 사용하지 않는다. `behavior_node`의 0.5초 타이머와 deadline을 이용하는
논블로킹 대기 `_enter_stub_wait()`로 같은 효과를 낸다. 따라서 대기 중에도 AMR 상태와 ROS
콜백을 처리할 수 있다.

```text
현재: 목적지 도착 → [2초 정지 STUB] → 다음 목적지
향후: 목적지 도착 → [비전/팔 Action 실행 및 성공 확인] → 다음 목적지
```

## 1. 전체 목표와 오늘 구현 범위

최종 목표:

1. HOME에서 작업 명령을 기다린다.
2. ST1 → ST2 → ST3 → ST4 → ST5 → ST6 순서로 전체 비전 스캔을 먼저 끝낸다.
3. 버스바 테이블에서 스테이션당 버스바 하나를 파지한다.
4. ST1 → ST2 → ST3 → ST4 → ST5 → ST6 순서로 버스바와 너트를 체결한다.
5. 각 스테이션 작업마다 `버스바 테이블 ↔ 해당 스테이션`을 왕복한다.
6. 모든 작업이 끝나면 HOME으로 복귀한다.

오늘 구현된 시연 흐름:

```text
작업 명령 수신
  → ST1 스캔(STUB)
  → ST2 스캔(STUB)
  → ST3 스캔(STUB)
  → 안전 우회 후 ST4 스캔(STUB)
  → ST5 스캔(STUB)
  → ST6 스캔(STUB)
  → 안전 경로로 BUS1 이동
  → BUS1 파지(STUB)
  → ST1 복귀
  → ST1 버스바·너트 체결(STUB)
  → BUS2 이동
  → BUS2 파지(STUB)
  → ST2 이동
  → ST2 버스바·너트 체결(STUB)
  → BUS3 이동
  → BUS3 파지(STUB)
  → ST3 이동
  → ST3 버스바·너트 체결(STUB)
  → 완료 보고
```

아직 구현하지 않은 범위:

- BUS4~BUS6 파지 및 ST4~ST6 체결 왕복
- 최종 HOME 복귀
- 실제 스캔 완료 판정
- 실제 버스바 파지·삽입
- 실제 너트 파지·체결
- 장애물 자동 회피/Nav2/SLAM 연동

## 2. 오늘 해결한 내용

### AMR 주행

- `/amr/goal`의 고정 월드 좌표를 `/odom`과 비교하는 폐루프 주행을 구성했다.
- 주행 출력 토픽을 Isaac Sim이 받는 `/cmd_vel`로 바로잡았다.
- `회전 → 직진 → 최종 자세 정렬` 제어를 기본으로 사용한다.
- 좌표가 임의로 바뀌지 않도록, 시작 odometry를 기준으로 월드 좌표를 복원한다.
- BUS 파지 위치는 일반 지점보다 낮은 속도와 작은 허용 오차로 접근한다.
- 테이블 바로 앞에서는 불필요한 최종 제자리 회전을 생략하는 position-only 도착을 사용한다.
- 경유점별 전진/후진 선택과 곡선 주행 예외를 추가했다.
- 한글 단계 로그, 현재 좌표, 남은 거리, 방향 오차, `/cmd_vel` 값을 출력한다.

### 충돌 방지 동선

- ST3→ST4 직선이 배터리 모듈팩 모서리를 통과하여 하단 안전 경유점을 추가했다.
- ST3에서 조금 더 전진한 후 반시계 방향으로 정렬해 ST4~ST6 라인에 진입한다.
- ST4→ST5→ST6은 같은 방향으로 직진하며 연속 스캔한다.
- ST6 스캔이 끝난 뒤에만 전방 이탈점으로 이동한다.
- ST6→BUS1은 모듈팩을 피해 안전 접근점으로 이동한다.
- BUS1 진입은 테이블 왼쪽 도킹 정렬점에서 자세를 맞춘 뒤 직선으로 진입한다.
- BUS1 파지 후 ST1 복귀는 별도 이탈점과 복귀점을 사용해 모션을 매끄럽게 했다.
- BUS2/BUS3은 새 좌표를 적용하고 평행주차 동작을 제거했다.

### 공장 배경

- NVIDIA Industrial Assets Pack의 `Warehouse01.usd`를 런타임 reference로 적용했다.
- 원본 `Collected_Busbar3/Busbar.usd`는 직접 수정하지 않았다.
- 원본 ground의 시각 표시는 숨기되 물리 바닥은 유지한다.
- AMR과 배터리 모듈팩을 막던 금속·흰색·사다리형 구조물 그룹은 런타임에 숨긴다.
- 공장 배경 collider는 비활성화해 AMR 시연 동선에 영향을 주지 않도록 했다.

## 3. 핵심 파일 구조

```text
/home/rokey/EV_combine
├── isaacpjt/M0609
│   ├── AMR_HANDOFF_2026-07-27.md       ← 이 문서
│   ├── execute_isaac_busbar_amr.py     ← Isaac Sim 씬/AMR ROS 브리지/공장 배경
│   ├── Collected_Busbar3/
│   │   ├── Busbar.usd                  ← 현재 작업 셀 원본 씬
│   │   └── SubUSDs/                    ← 씬 종속 USD
│   └── factory_assets/
│       ├── Industrial_NVD_10012.zip    ← 받은 원본 압축 파일
│       └── industrial_pack/.../
│           └── Warehouse01.usd         ← 현재 사용하는 공장 배경
├── src
│   ├── behavior_node/
│   │   └── behavior_node/behavior_node.py
│   │                                      ← 전체 순서/FSM/좌표/임시 대기
│   ├── amr_node/
│   │   └── amr_node/amr_node.py          ← /odom 기반 이동 제어 및 /cmd_vel
│   ├── arm_node/
│   │   └── arm_node/arm_node.py          ← 실제 파지·삽입·너트 Action 서버 후보
│   ├── perception_node/
│   │   └── perception_node/perception_node.py
│   │                                      ← 비전 검출/서비스/토픽
│   ├── fms_interfaces/
│   │   ├── action/BusbarInsert.action
│   │   ├── action/NutFasten.action
│   │   └── msg/                           ← AMR/FMS/비전 메시지
│   └── fms_bringup/                       ← 전체 노드 launch 패키지
└── AMR_BACKUPS/                           ← 시점별 복구용 프리징 파일
```

각 파일의 책임:

| 파일 | 책임 | 좌표를 바꿀 때 |
|---|---|---|
| `behavior_node.py` | 작업 순서, 상태 전이, waypoint 월드 좌표 | `STATION_POSES` 수정 |
| `amr_node.py` | 목표까지 실제 주행, 속도·허용 오차·방향 결정 | 제어 상수와 waypoint 분류 수정 |
| `execute_isaac_busbar_amr.py` | 씬 로드, 바퀴 제어, odom/TF/clock, 공장 배경 | 씬/에셋 설정만 수정 |
| `arm_node.py` | 팔/그리퍼, 버스바 Action, 너트 Action | 새 씬 기준 팔 목표 보정 |
| `perception_node.py` | 카메라 검출 결과와 grasp/bolt 서비스 | 모델·카메라·좌표 변환 보정 |

## 4. ROS2 데이터 흐름

```text
/fleet/job
    ↓
behavior_node
    ├─ /amr/goal ─────────────→ amr_node
    │                              ├─ /odom 구독
    │                              ├─ /cmd_vel 발행 → Isaac Sim 바퀴
    │                              └─ /amr/status ──→ behavior_node
    │
    ├─ /vision/* ←──────────── perception_node
    ├─ /busbar_insert Action ─→ arm_node
    ├─ /nut_fasten Action ────→ arm_node
    └─ /fleet/report
```

주요 인터페이스:

| 이름 | 타입 | 용도 |
|---|---|---|
| `/fleet/job` | `FleetJob` | 전체 작업 시작 |
| `/fleet/report` | `FleetReport` | 작업 결과 보고 |
| `/amr/goal` | `AmrGoal` | waypoint의 X/Y/theta 전달 |
| `/amr/status` | `AmrStatus` | MOVING/ARRIVED/ERROR |
| `/cmd_vel` | `geometry_msgs/Twist` | AMR 선속도·각속도 |
| `/odom` | `nav_msgs/Odometry` | 현재 AMR pose |
| `/vision/busbar_grasp` | `BusbarGrasp` | 버스바 파지 pose |
| `/vision/stud_pose` | `StudPose` | 스터드 pose |
| `/vision/nut_pose` | `NutPose` | 너트 pose |
| `/busbar_insert` | `BusbarInsert` Action | `GRASP`, `INSERT` |
| `/nut_fasten` | `NutFasten` Action | `NUT_APPROACH`, `NUT_GRASP`, `FASTEN_APPROACH`, `FASTEN` |

## 5. 좌표와 경로 수정 위치

모든 작업 좌표는 다음 딕셔너리에 있다.

```text
src/behavior_node/behavior_node/behavior_node.py
└── STATION_POSES
```

주요 실측 좌표:

| 지점 | X | Y | 방향 |
|---|---:|---:|---:|
| ST1 | 0.66594 | -0.04555 | -90° |
| ST2 | 0.66594 | -0.64732 | -90° |
| ST3 | 0.66594 | -1.17289 | -90° |
| ST4 | 1.98852 | -1.17289 | -90° |
| ST5 | 1.98852 | -0.64732 | -90° |
| ST6 | 1.98852 | -0.11180 | -90° |
| BUS1 | -0.50211 | 2.37274 | -180° |
| BUS2 | 0.48533 | 2.07778 | -90° |
| BUS3 | 1.38165 | 2.07778 | -90° |

주의:

- ST4~ST6의 **스캔 pose**는 진행 방향을 유지하기 위해 `scan_station_4~6`의 +90°를 쓴다.
- 체결하러 갈 때는 `station_4~6`의 -90° pose를 사용해야 한다.
- 좌표만 추가하면 끝나는 것이 아니다. 정밀 접근점은 `amr_node.py`의
  `PRECISE_POSITION_WAYPOINTS`, 최종 회전을 생략할 점은 `POSITION_ONLY_WAYPOINTS`,
  전진 곡선으로 갈 점은 `FORWARD_ARC_WAYPOINTS`에도 ID를 분류해야 한다.
- 장애물 가까이에서 목적지 방향만 바꾸면 제자리 회전 중 차체나 팔이 충돌할 수 있다.
  항상 `안전 접근점 → 자세 정렬 → 직선 도킹` 순서를 권장한다.

## 6. 실행 방법

### 최초 또는 코드 변경 후 빌드

```bash
cd /home/rokey/EV_combine
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select amr_node behavior_node
```

인터페이스나 arm/perception 패키지를 변경했다면 필요한 패키지를 함께 지정한다.

```bash
colcon build --symlink-install \
  --packages-select fms_interfaces amr_node behavior_node arm_node perception_node
```

### 현재 AMR 시연 실행

```
.bashrc에 추가

isaac_ros2_setup() {
    export ROS_DISTRO=humble
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export AMENT_PREFIX_PATH=/opt/ros/humble
    export LD_LIBRARY_PATH=/home/rokey/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/exts/isaacsim.ros2.bridge/humble/lib:$LD_LIBRARY_PATH
}
```

터미널 1 — Isaac Sim:

```bash
source ~/.bashrc
isaac_ros2_setup
isaac_python /home/rokey/EV_combine/isaacpjt/M0609/execute_isaac_busbar_amr.py
```

터미널 2 — AMR 제어:

```bash
source ~/.bashrc
ros2_setup
ros2 run amr_node amr_node
```

터미널 3 — 작업 순서:

```bash
source ~/.bashrc
ros2_setup
ros2 run behavior_node behavior_node
```

터미널 4 — 작업 시작:

```bash
source ~/.bashrc
ros2_setup
ros2 topic pub -1 /fleet/job fms_interfaces/msg/FleetJob \
  "{job_id: 'demo_001', station_id: 'station_1', job_type: 'ASSEMBLE', target: 'busbar_and_nut'}"
```

치명적 주의:

- `isaac_ros2_setup`은 Isaac Sim 터미널에서만 실행한다.
- `ros2_setup`은 일반 ROS2 노드 터미널에서 실행한다.
- 두 setup을 같은 터미널에서 섞으면 Python/rclpy/spdlog 라이브러리 충돌이 날 수 있다.
- 실행 순서는 `Isaac Sim → amr_node → behavior_node → job 발행`을 권장한다.

### 공장 배경 설정

기본은 공장 배경 ON이다.

```bash
# 공장 배경 없이 비교 실행
AMR_FACTORY_DRESSING=0 isaac_python \
  /home/rokey/EV_combine/isaacpjt/M0609/execute_isaac_busbar_amr.py

# 배경 크기 변경 예시
AMR_FACTORY_SCALE=0.005 isaac_python \
  /home/rokey/EV_combine/isaacpjt/M0609/execute_isaac_busbar_amr.py
```

공장 배경의 구조물 숨김은 `execute_isaac_busbar_amr.py`의 `add_factory_dressing()`에서 처리한다.

## 7. 임시 STUB 위치 — 실제 기능이 아닌 부분

설정:

```python
ARM_ACTIONS_ENABLED = False
VISION_ENABLED = False
STUB_WAIT_SEC = 2.0
```

실제 기능 대신 정지 대기하는 함수:

```python
_enter_scan_stub()       # ST1~ST6 카메라 스캔 대신 2초 대기
_enter_grasp_stub()      # BUS1 파지 대신 2초 대기
_enter_fasten_stub()     # ST1 삽입/너트 체결 대신 2초 대기
_enter_grasp2_stub()     # BUS2 파지 대신 2초 대기
_enter_fasten2_stub()    # ST2 삽입/너트 체결 대신 2초 대기
_enter_grasp3_stub()     # BUS3 파지 대신 2초 대기
_enter_fasten3_stub()    # ST3 삽입/너트 체결 대신 2초 대기
```

공통 대기는 `_enter_stub_wait()`이 담당한다. 단순히 시연 대기 시간을 바꾸려면
`STUB_WAIT_SEC`만 수정하면 된다. 하지만 실제 기능을 연결할 때는 이 값을 늘리는 방식으로
해결하면 안 되고, Action 또는 서비스의 **실제 성공 결과를 받은 후** 다음 상태로 전환해야 한다.

## 8. 실제 파지·스캔·체결 기능 연결 방법

### 권장 통합 순서

한꺼번에 전부 켜지 말고 아래 순서로 독립 검증한다.

1. AMR을 정지시킨 상태에서 arm_node가 팔을 제어하는지 확인
2. BUS1에서 `GRASP` Action 단독 시험
3. ST1에서 `INSERT` Action 단독 시험
4. 너트 4단계 Action 단독 시험
5. perception 결과를 화면/로그로만 확인
6. perception pose를 Action goal에 연결
7. BUS1↔ST1 전체 흐름 연결
8. 같은 패턴을 BUS2/ST2, BUS3/ST3에 일반화
9. 마지막으로 BUS4~6과 HOME 복귀 추가

### 8.1 스캔 STUB 교체

현재 `_enter_scan_stub()`은 2초 뒤 무조건 성공으로 간주한다. 다음 구조로 바꾸는 것이 안전하다.

```text
SCAN_REQUEST
  → 카메라 프레임 갱신 확인
  → perception 검출 요청
  → 필요한 객체 개수/신뢰도/pose 유효성 검사
  → 성공: 다음 스테이션
  → 실패: 제한 횟수 재시도 후 RECOVER
```

`perception_node`에는 다음 연결 후보가 있다.

- `/vision/busbar_grasp`
- `/vision/stud_pose`
- `/vision/nut_pose`
- `/perception/get_grasp_pose`
- `/perception/get_bolt_pair`

스테이션마다 “새 프레임에서 나온 결과”인지 확인할 timestamp 또는 scan request ID를 추가하는
것을 권장한다. 이전 스테이션의 마지막 검출값을 새 결과로 오인하면 안 된다.

### 8.2 BUS1 실제 파지 연결

이미 준비된 인터페이스:

```text
behavior_node -- /busbar_insert Action, command=GRASP --> arm_node
```

`behavior_node.py`에는 `_send_busbar_goal('GRASP')`,
`_on_busbar_result()`와 timeout/retry/cancel 코드가 존재한다. 따라서 권장 변경은 다음과 같다.

1. `arm_node.py`의 BUS1 파지 좌표를 `Collected_Busbar3`와 OnRobot RG2 기준으로 재보정한다.
2. arm_node를 단독 실행해 `/busbar_insert`의 `GRASP`를 시험한다.
3. 파지 완료를 육안뿐 아니라 그리퍼 폭/접촉/부착 상태 등으로 판정한다.
4. `GRASP_STUB` 진입 대신 `GRASP_BUSBAR`로 전환하고 `_send_busbar_goal('GRASP')`를 호출한다.
5. Action result가 `success=True`일 때만 BUS1 이탈 상태로 전환한다.

중요: 현재 `ARM_ACTIONS_ENABLED=True` 분기는 초기 코드의 일부만 실제 Action 흐름으로 보내도록
되어 있어, 플래그만 켜면 BUS1~BUS3 전체가 자동 통합되는 구조는 아니다. 각 `GRASP*_STUB`과
`FASTEN*_STUB` 전환을 Action 결과 기반으로 명시적으로 바꾸고 시험해야 한다.

### 8.3 버스바 삽입 연결

버스바 파지와 삽입은 같은 Action을 사용한다.

```text
GRASP 성공
  → AMR이 해당 스테이션으로 이동
  → /busbar_insert command=INSERT
  → INSERT 성공
  → 너트 체결 단계
```

Action goal:

```text
command: GRASP | INSERT
station_id: 대상 스테이션
target_pose: perception에서 얻은 pose
```

현재 `_on_busbar_result()`는 `GRASP_BUSBAR → INSERT_BUSBAR` 흐름을 가지고 있지만, AMR 왕복
사이에 그대로 사용하도록 상태를 분리해야 한다. 즉 파지 성공 직후 INSERT를 보내면 안 되고,
`AMR 이동 완료 → INSERT`가 되도록 구성한다.

### 8.4 너트 체결 연결

권장 순서:

```text
NUT_APPROACH
  → NUT_GRASP
  → FASTEN_APPROACH
  → FASTEN
```

`behavior_node.py`의 `_send_fasten_goal()`과 `_on_fasten_result()`에 위 순서가 구현되어 있고,
`arm_node.py`가 `/nut_fasten` Action 서버를 제공한다.

주의:

- arm_node의 일부 너트 위치와 궤적은 이전 씬 기준 하드코딩/녹화값이다.
- `Collected_Busbar3`에서 너트·볼트·팔 베이스 좌표를 다시 확인해야 한다.
- `FASTEN`은 기록된 관절 궤적을 재생하므로 시작 자세가 맞지 않으면 급격한 관절 이동이 생긴다.
- 궤적 파일이 없을 때 arm_node가 더미 동작을 수행하는 경로가 있으므로, “Action 성공”만으로
  실제 체결이라고 판단하지 말고 궤적 로드 로그와 실제 물체 상태를 함께 확인한다.

### 8.5 상태 구조 권장안

스테이션마다 중복 상태를 계속 만드는 대신 작업 인덱스를 두는 방식이 확장하기 쉽다.

```python
assembly_index = 0
stations = ['station_1', ..., 'station_6']
busbars = ['busbar_table', ..., 'busbar_table_6']
```

```text
MOVE_TO_BUS
→ WAIT_GRASP_VISION
→ GRASP_ACTION
→ MOVE_TO_STATION
→ WAIT_INSERT_VISION
→ INSERT_ACTION
→ NUT_ACTION_SEQUENCE
→ assembly_index += 1
→ 다음 BUS 또는 HOME
```

현재 시연 동선을 보존한 채 별도 브랜치/백업에서 리팩터링하는 것을 권장한다.

## 9. 실제 기능 통합 시 안전 조건

- 팔 동작 전 `/cmd_vel`이 0인지 확인한다.
- AMR 도착 직후 0.5~1초 정도 odom 속도가 안정되는 조건을 검사한다.
- 팔이 안전 자세로 복귀하기 전에는 AMR 이동 goal을 보내지 않는다.
- Action timeout 시 같은 goal을 즉시 중복 발행하지 않는다.
- cancel 결과를 확인한 뒤 재시도한다. 현재 behavior_node에 이 안전 로직이 들어 있다.
- 파지 실패 시 AMR을 바로 출발시키지 않는다.
- perception 결과에 timestamp, frame_id, 유효 범위 검사를 둔다.
- 월드 pose와 로봇 base pose의 좌표계 변환을 명확히 한다.
- 각 스테이션 첫 시험은 속도를 낮추고 한 단계씩 Action을 수동 발행한다.
- 공장 배경은 장식용이며 현재 장애물 회피 센서의 기준 맵이 아니다.

## 10. 디버깅 명령

```bash
# 노드와 토픽
ros2 node list
ros2 topic list

# AMR 목표/상태/속도/위치
ros2 topic echo /amr/goal
ros2 topic echo /amr/status
ros2 topic echo /cmd_vel
ros2 topic echo /odom

# Action 서버 확인
ros2 action list
ros2 action info /busbar_insert
ros2 action info /nut_fasten

# 비전 데이터 확인
ros2 topic echo /vision/busbar_grasp
ros2 topic echo /vision/stud_pose
ros2 topic echo /vision/nut_pose

# 완료 보고 확인
ros2 topic echo /fleet/report
```

AMR이 움직이지 않을 때 확인 순서:

1. Isaac Sim이 Play 상태인지 확인한다.
2. `/clock`과 `/odom`이 계속 갱신되는지 확인한다.
3. `/amr/goal`이 한 번 발행됐는지 확인한다.
4. `/cmd_vel`이 0이 아닌지 확인한다.
5. `/cmd_vel`은 나오는데 차가 안 움직이면 바퀴 articulation/물리 충돌을 확인한다.
6. 목표 방향으로 계속 회전만 하면 현재 yaw, 목표 theta, 회전 부호를 확인한다.
7. 경유점 도착 로그가 있는데 다음 목표가 없으면 behavior_node의 route queue/state를 확인한다.

## 11. 백업과 복구

현재 백업:

```text
AMR_BACKUPS/
├── 1차_AMR_시연동선_프리징_2026-07-27_162035.tar.gz
├── 2차_전체스캔_ST1-ST3_매끄러운복귀_프리징_2026-07-27_183058.tar.gz
└── 3차_공장에셋_적용전_프리징_2026-07-27_183523.tar.gz
```

가장 안정적으로 다듬은 AMR 동선 기준은 2차 백업이다. 공장 에셋 적용 전 상태가 필요하면
3차 백업을 사용한다. 각 압축 파일 옆의 `_복구안내.txt`를 먼저 읽고 복구한다.

중요:

- 복구 전에 현재 파일을 별도 백업한다.
- 압축을 바로 덮어쓰기 전에 `tar -tzf 파일명`으로 내용을 확인한다.
- 원본 `Busbar.usd`와 `SubUSDs`의 상대 경로를 깨뜨리지 않는다.

## 12. 다음 개발자가 바로 할 일

1. 현재 AMR 시연을 한 번 실행해 기준 동선을 확인한다.
2. BUS1에 정차한 상태에서 arm_node의 `GRASP`만 단독 시험한다.
3. `Collected_Busbar3` 기준 파지 좌표와 RG2 그리퍼 동작을 보정한다.
4. `GRASP_STUB` 하나만 실제 Action으로 교체한다.
5. 실패/timeout/cancel 후 AMR이 출발하지 않는지 확인한다.
6. ST1에서 `INSERT`와 너트 4단계를 각각 단독 시험한다.
7. BUS1↔ST1 한 사이클을 완성한 후 BUS2/3에 일반화한다.
8. BUS4~6 좌표와 안전 경유점을 추가한다.
9. 마지막으로 HOME 복귀를 추가한다.

이 순서를 따르면 이미 안정화한 AMR 동선을 유지하면서 실제 파지 기능을 작은 단위로 끼워 넣을
수 있다.

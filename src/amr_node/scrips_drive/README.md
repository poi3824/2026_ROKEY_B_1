# AMR PhysX 비전 좌표 주행

`scrips_drive`는 Nova Carter의 실제 좌우 wheel joint를 PhysX로 구동한다. 주행
목표의 절대 X/Y는 코드, station 표, 기본값 또는 fallback에서 만들지 않는다.
perception이 같은 USD 장면에서 검출한 `world` 좌표만 목표 생성에 사용할 수 있다.

## 입력 정책

런타임에는 좌표가 없는 의미론적 요청만 받는다.

- `battery`: `/bolt_cam/perception/detections_3d`에서 정확히 두 개의 `bolt`를
  이루는 유효 쌍을 고르고 그 순수 기하 중점을 사용한다. 첫 정상 요청에서 선택한
  쌍은 같은 Isaac 세션 동안 후보 식별 기준으로 유지하므로, busbar 작업 후 복귀할
  때 다른 배터리 쌍으로 전환되지 않는다. 실제 목표는 매 요청의 새 검출 좌표로
  계산하며 기억한 중점을 좌표 폴백으로 사용하지 않는다.
- `busbar`: `/busbar_cam/perception/detections_3d`에서 복수 `busbar`가 검출되면
  confidence가 유일하게 가장 높은 한 개를 고른 뒤 연속 검출 안정성을 검사한다.
  최고 confidence가 동점이면 선택하지 않는다. 주행이 시작된 뒤에는 confidence
  순위가 바뀌어 다른 물체로 전환되지 않도록, 최초 좌표와 가장 가까운 후보만
  동일 요청의 대상으로 유지한다. 이 후보가 confidence/drift 검사를 실패하면 다른
  후보로 갈아타지 않고 취소한다.

다음 검사를 모두 통과해야만 목표가 생성된다.

- array와 각 detection의 `frame_id`가 정확히 `world`
- 대상 토픽 publisher가 정확히 하나
- battery는 필요한 후보 수와 일치하며, busbar는 최고 confidence 후보가 하나로
  결정되고 모든 사용 좌표·score가 유한
- 기본 confidence `0.6` 이상
- 기본 3개 이상의 연속 표본
- 각 표본의 원본 camera `header.stamp`가 엄격히 증가(동일 frame 재사용 금지)
- 기본 XY 표준편차 `10 mm` 이하
- 볼트쌍은 중점뿐 아니라 두 볼트 간격의 표준편차도 기본 `10 mm` 이하
- 마지막 표본의 monotonic 수신 age가 기본 `1 s` 이하
- Isaac의 실제 `/amr_physics/world_pose`도 최신 상태
- READY, world pose, 대상 detection과 RGB/depth/camera_info publisher가 각각 정확히 하나

검출 없음, busbar 최고 confidence 동점, battery 후보 수 불일치, 오래된 pose,
불안정한 검출, 잘못된 TF/frame이면 주행하지 않고 `STATE_ERROR`를 발행한다.
이전 좌표, 0 좌표 또는 과거 station 좌표로 대체하지 않는다. 주행이 시작된 뒤에도
perception/world pose가 끊기거나 target이 허용치보다
이동하면 relay lease를 중단하고 `CANCEL`을 보낸다.

perception 노드는 RGB/depth subscriber queue를 depth 1로 제한하고, 새 RGB source
timestamp 하나를 한 번만 처리한다. RGB/depth stamp 차이와 callback 수신 age도
검사하므로 카메라가 멈춘 뒤 캐시된 동일 이미지를 새 표본처럼 재사용하지 않는다.

## 물체 좌표와 chassis 목표

perception 출력은 물체 중심이지 AMR chassis의 docking pose가 아니다. 물체 중심으로
직접 주행하면 충돌하므로 작업 셀마다 고정한 접근 normal의 선상에 목표를 만들고,
검출점 앞에서 지정 stand-off만큼 떨어져 정지한다. battery는 첫 정상 스캔 자세인
Isaac READY baseline과 첫 검증된 비전 target으로 접근 normal과 chassis yaw를 세션당
한 번만 보정한다. 버스바 작업 후 복귀할 때 현재 AMR 위치로 방향을 다시 계산하지
않으므로 같은 배터리 작업면으로 돌아온다. 검출 좌표는 같은 볼트쌍을 연관하는
기준으로만 유지하며, 절대 station 좌표나 주행 fallback으로 저장하지 않는다.

`vision_standoff_m`은 로봇·팔·작업물 기하로 실제 캘리브레이션해야 하는 상대 거리다.
안전하지 않은 임의 기본값은 없으며 relay 시작 시 반드시 명시해야 한다.
`vision_max_target_drift_m`도 비전 측정 오차와 작업 허용치로 정한 값을 반드시
명시해야 한다.

```bash
python3 src/amr_node/scrips_drive/amr_node_drive.py \
  --ros-args \
  -p vision_standoff_m:=0.90 \
  -p battery_standoff_m:=0.75 \
  -p battery_preapproach_extra_m:=0.80 \
  -p battery_initial_pose_tolerance_m:=0.05 \
  -p battery_initial_yaw_tolerance_deg:=5.0 \
  -p vision_max_target_drift_m:=0.005 \
  -p busbar_preapproach_extra_m:=0.80
```

`vision_standoff_m=0.90 m`는 busbar 전용 값으로, `Busbar.usd` 작업 셀을
90도 회전하고 스캔 장착면을 작업대 쪽으로
둔 조건에서 물리 도킹 완료를 확인한 값이다. `0.55 m`에서는 대상과 약 `0.82 m`
떨어진 지점에서 차체가 작업대에 닿아 움직이지 못했다. 다른 USD/로봇 형상에는
그대로 적용하지 말고 다시 측정한다. `0.005 m` drift도 현재 비전 오차와 작업
허용치가 바뀌면 다시 정해야 한다. `battery_standoff_m=0.75 m`는 현재 초기
AMR-배터리 중점 실측 거리 약 `0.766 m`에서 불필요한 이동을 최소화하기 위한
battery 전용 값이다. 첫 비전 target으로 계산한 final goal과 READY baseline의
차이가 `battery_initial_pose_tolerance_m=0.05 m`를 넘으면 잘못된 시작 위치로 보고
방향 보정을 거부한다. 현재 yaw도 READY yaw와
`battery_initial_yaw_tolerance_deg=5 deg` 이내여야 하며, 위치 또는 yaw가
어긋나면 무제한 재시도하지 않고 요청을 실패시킨다. 복귀 때는 final chassis
진행축의 뒤쪽 `0.80 m`에
pre-approach를 만들고, 최초 chassis yaw와 같은 방향으로 직선 진입한다. 작업물
normal 바깥으로 갔다가 final에서 90도 제자리 회전하는 경로를 만들지 않는다.
busbar 충돌 방지 stand-off를 battery에 재사용하지 않는다.

## 같은 USD 장면 요구

drive executor 기본 장면은 다음 파일이다.

```text
src/Collected_Busbar_amr/Busbar.usd
```

`Collected_World_0123/World0123.usd`는 `world`라는 같은 frame 이름을 쓰지만 AMR,
카메라, 작업물의 실제 배치가 다르다. 그 장면에서 얻은 perception 좌표를 drive
executor에 넣으면 안 된다. 반드시 현재 `execute_isaac.py`가 연
`Collected_Busbar_amr/Busbar.usd`의 다음 카메라 토픽을 사용한다.

- `/busbar_cam/{rgb,depth,camera_info}`
- `/bolt_cam/{rgb,depth,camera_info}`
- TF child: `busbar_cam_optical_frame`, `bolt_cam_optical_frame`

### 버스바 작업 셀 90도 회전

`Collected_Busbar_amr/Busbar.usd` 자체에 `bench_busbar`의 world bounding-box
중심을 피벗으로 다음 prim의 +Z축 기준 90도 회전을 영구 반영했다.

- `/World/bench_busbar` — 자식 `Cube_01~Cube_06`도 함께 회전
- `/World/Z_busbar3`, `/World/Z_busbar3_01`, `/World/Z_busbar3_02`
- `/World/Camera_busbar` — 자식 `busbar_cam_optical_frame`도 함께 회전

카메라도 같은 피벗으로 회전했으므로 RGB/depth, optical TF, perception world 좌표가
같은 저장된 장면 변환을 공유한다. executor는 기본값인
`--busbar-workcell-yaw-deg 0`으로 열며 실행 명령에 90을 다시 지정하지 않는다.
해당 옵션은 저장된 자세에서 임시 session-layer 추가 회전이 필요한 경우에만 쓴다.

90도 회전 뒤에는 현재 AMR-검출점 직선을 그대로 쓰면 긴 테이블을 가로지를 수 있다.
executor가 READY에 회전된 작업 셀 접근 normal을 싣고, relay는 busbar에 한해
그 normal과 perception 검출 좌표로 목표를 만든다. 먼저 최종 stand-off보다 기본
`0.80 m` 바깥의 pre-approach에 도착한 다음 직선 docking하므로 테이블 모서리를
향해 바로 들어가지 않는다. 현재 예시의 stand-off `0.90 m`와 합치면 검출점에서
`1.70 m` 떨어진 곳이 회전·정렬 경유점이다. 가까운 `0.20 m` 여유에서는 차체가
테이블에 닿은 상태로 제자리 회전해 바퀴 명령만 나오고 yaw가 멈추는 현상이
재현되어 기본값을 늘렸다. 이 여유 거리는 다음처럼 변경할 수 있다.

물리 제어기의 기본 `--drive-direction auto`는 relay가 후진을 허용한 goal에서
경로가 chassis 뒤쪽에 있으면 후진을 선택한다. 선택한 방향은 goal 도중 바뀌지 않는다.
현재 relay는 busbar pre-approach와 최종 docking에서 후진을 허용한다.
최종 docking에서는 기존 버스바 스캔 자세가 가능한 로봇팔 장착면을 작업대 쪽으로
둔다.
이 장착면은 chassis 진행축의 반대쪽이므로 기본
`busbar_scan_yaw_offset_deg:=180.0`을 적용하고 마지막 짧은 선분의 후진을 허용한다.
90도 회전 작업 셀에서는 접근 normal이 `-180도`, 최종 chassis yaw가 `0도`가 된다.
항상 전진만 허용하려면 executor에 `--drive-direction forward`를 지정할 수 있지만,
이 경우 스캔 장착면 정렬을 위해 불필요한 회전이 다시 생길 수 있다.

```bash
python3 src/amr_node/scrips_drive/amr_node_drive.py \
  --ros-args \
  -p vision_standoff_m:=0.90 \
  -p battery_standoff_m:=0.75 \
  -p battery_preapproach_extra_m:=0.80 \
  -p battery_initial_pose_tolerance_m:=0.05 \
  -p battery_initial_yaw_tolerance_deg:=5.0 \
  -p vision_max_target_drift_m:=0.005 \
  -p busbar_preapproach_extra_m:=0.80 \
  -p busbar_scan_yaw_offset_deg:=180.0 \
  -p active_vision_timeout_sec:=6.0
```

`vision_standoff_m`과 drift 값은 기존과 마찬가지로 실제 캘리브레이션 값이어야 한다.
pre-approach는 안전 경유 거리이며 검출점의 절대 좌표를 대체하는 fallback이 아니다.
`vision_max_age_sec`은 새 목표를 만들 때 쓰는 엄격한 freshness 제한이고,
`active_vision_timeout_sec`은 주행 중 perception frame 누락을 허용하는 시간이다.
후자는 실제 headless 검출 주기 약 `3~3.5초`보다 길어야 하며, 최근 측정값 기준으로
`6.0초`를 사용한다. 새 표본이 수신될 때마다 target drift 검사는 계속 수행한다.

drive용 launch는 `World0123.usd`에서 추출한 물체 모델 좌표를 쓰는 busbar PnP
경로를 끄고 RGB-D+TF 좌표만 사용한다. `setup_fixed_camera_bridges.py`는 현재
`Collected_World_0123/World0123.usd`를 대상으로 하므로 drive 장면 준비에 실행하면
안 된다.

동일 ROS domain에는 이 USD를 연 Isaac executor 하나만 실행한다. 다른 USD나
`fms_bringup.launch.py`의 busbar/bolt perception을 동시에 실행하지 않는다. 동시에
실행하면 source publisher가 중복되어 relay가 주행을 거부한다.

## 파일

- `execute_isaac.py`: Isaac Sim 5.1 PhysX executor
- `drive_controller.py`: ROS/Isaac import가 없는 차동구동 제어기
- `vision_goal.py`: 검출 안정성 gate와 stand-off 목표 계산
- `amr_node_drive.py`: perception과 Isaac JSON protocol을 연결하는 Humble relay
- `fixed_camera_perception.launch.py`: 두 고정 카메라 perception 인스턴스와 명시적 remap
- `full_process_execute_isaac.py`: 물리 AMR와 M0609 FSM을 같은 SimulationApp에서 실행
- `full_process_ros.launch.py`: 고정/손목 perception, relay, ArmNode를 함께 실행
- `vision_drive_demo.py`: 기본값은 전체 공정, `--drive-only`이면 AMR 주행만 시험
- `tests/test_drive_controller.py`: 제어기·비전 gate 단위 테스트

과거 `coordinates.py`, `ping_pong_demo.py`, `LEGACY_STATIONS`, `legacy_input` 경로는
삭제했다. `/amr_physics/goal`의 `AmrGoal.x/y`도 더 이상 구독하지 않는다. Isaac
executor는 `source=perception`, `frame_id=world`인 내부 goal만 수락한다.

## Python 환경

Isaac Sim Python과 ROS 2 Humble Python 버전이 다르므로 터미널을 분리한다.

- Isaac executor: `isaac_ros2` 후 `isaac_python`
- perception/relay/demo: `ros2_set`, `source install/setup.bash`, 시스템 `python3`

## 실행

필요한 perception 모델이 `src/perception_node/models/`에 있는지 확인하고 빌드한다.

```bash
ros2_set
colcon build --symlink-install
source install/setup.bash
```

터미널 1 — 같은 USD/SimulationApp에서 물리 AMR와 M0609 FSM을 실행한다.

기본 디버깅 속도는 `0.35 m/s`, `0.75 rad/s`다. 실제 후진·정차 및 전체 busbar
docking 시험을 통과한 값이며 필요하면 `--max-linear 0.22`로 낮출 수 있다.
작업 셀 목표가 시작 pose에서 약 `2.85 m` 떨어져 있고 회전 중 caster 횡이동이
발생하므로 workspace 안전 반경 기본값은 `3.5 m`다.

```bash
isaac_ros2
isaac_python src/amr_node/scrips_drive/full_process_execute_isaac.py \
  --debug-no-timeouts
```

터미널 2 — 고정/손목 perception, relay, ArmNode를 함께 실행한다.

```bash
ros2_set
source install/setup.bash
ros2 launch src/amr_node/scrips_drive/full_process_ros.launch.py
```

perception 코드의 출력 이름은 `/`로 시작하는 절대 이름이다. 단순 namespace로는 두
인스턴스가 분리되지 않으므로 launch 파일이 모든 detection/service/vision 출력을
명시적으로 `/busbar_cam/...`, `/bolt_cam/...`로 remap한다.

터미널 3 — 좌표를 넣지 않고 전체 공정을 시작한다.

```bash
ros2_set
source install/setup.bash
python3 src/amr_node/scrips_drive/vision_drive_demo.py \
  --debug-no-timeouts
```

위 명령은 기본값으로 배터리 주행/스캔, 팔 안전 자세 복귀, 버스바 주행/손목
스캔/파지, 배터리 복귀/조립, 너트 1·2 체결까지 수행한다. 버스바 스캔은 relay가
주행에 사용한 동일 world 좌표를 쓰며, 손목 카메라의 새 연속 검출이 확인될 때까지
선택 좌표 중심의 제한된 탐색 자세를 반복한다. 탐색은 중앙과 접근 접선 방향
`±0.02 m` 사이에서만 움직이며, 이 scan-only offset은 이후 wrist 카메라가 발행한
PICK 좌표를 덮어쓰지 않는다.

첫 battery 단계는 READY baseline이 이미 정상 배터리 스캔 자세에 있다는 것을
검증하면서 접근 normal과 chassis yaw를 고정한다. 버스바 파지 후 battery로 복귀할
때는 최종 chassis 진행축 뒤쪽의 pre-approach를 거친 뒤 최초와 같은 작업면과
chassis yaw로 직선 도킹한다. 따라서 busbar 위치에서 battery를 향한 방위가
북쪽이어도 북쪽 goal을 새로 만들지 않는다.

버스바 파지는 원래 물리 버전과 동일하게 rigid body/collision을 계속 켠 상태에서
핑거 collider와 마찰로 수행한다. pose-glue, `set_world_pose`, 버스바용
`FixedJoint`는 사용하지 않는다. 파지 후에는 옆으로 이동하지 않고 수직 상승하며,
실제 busbar 상승량이 5 cm 미만이면 성공으로 처리하지 않는다.

`debug_no_timeouts` 옵션은 디버깅에서만 사용한다. goal/stuck/lease와 Action
경과 시간 제한은 끄지만 publisher 유일성, 최초 비전 표본 품질,
workspace·기울기·높이 검사는 유지한다. 전체 공정 launch에서는 검증된 최초 target을
latch하므로 주행 중 5 mm 카메라 노이즈로 취소하지 않는다. 실제 운용에서는 timeout과
active drift 정책을 작업 허용치에 맞춰 다시 켠다. executor를 재시작했다면 이전
simulation timestamp가 TF cache에 남지 않도록 perception과 relay도 순서대로
재시작한다.

AMR 주행만 단발 시험하려면 명시적으로 `--drive-only`를 사용한다.

```bash
python3 src/amr_node/scrips_drive/vision_drive_demo.py \
  --drive-only --targets busbar --cycles 1 --debug-no-timeouts
```

상태 확인:

```bash
ros2 topic echo /busbar_cam/perception/detections_3d
ros2 topic echo /bolt_cam/perception/detections_3d
ros2 topic echo /amr_physics/world_pose
ros2 topic echo /amr_physics/drive_state
ros2 topic echo /amr_physics/status
```

## 실행 전 물리 검사

움직이지 않는 preflight:

```bash
isaac_ros2
isaac_python src/amr_node/scrips_drive/execute_isaac.py \
  --headless --preflight-only
```

짧은 실제 전진·회전 후 reset하는 smoke test:

```bash
isaac_ros2
isaac_python src/amr_node/scrips_drive/execute_isaac.py \
  --headless --smoke-test
```

## 현재 한계

- perception 모델은 `battery1~3`, `busbar1~3` 같은 instance/station ID를 출력하지
  않는다. 따라서 여섯 station 중 하나를 좌표로 선택하는 기능은 제공하지 않는다.
  busbar 최고 confidence 선택은 station 식별이 아니며, 잘못된 대상이 일관되게 더
  높은 confidence를 받으면 구분할 수 없다. 카메라/ROI에는 의도한 대상이 가장
  명확하게 보이고, bolt 카메라에는 의도한 볼트쌍 하나만 보여야 한다.
- 여섯 station 자동 운용에는 perception이 안정적인 `target_id`와 품질 정보를 함께
  발행하는 인터페이스가 추가로 필요하다. battery의 세션 내 최근접 연관은 첫
  요청에서 선택한 같은 볼트쌍을 복귀 때 유지하는 제한된 용도이며 station ID를
  추정하지 않는다. busbar 최근접 연관은 이미 시작된 한 번의 주행 요청 안에서
  confidence 순위 뒤집힘을 막는 용도일 뿐 다음 요청까지 기억하지 않는다.
- perception의 raw `Detection3DArray`에 대해 drive가 별도 연속 표본 gate를 적용한다.
  검출 알고리즘 내부 품질 상태(PnP 성공 여부 등)는 현재 메시지에 없으므로 장기적으로
  `target_id`, covariance, score, sample count, valid source가 포함된 전용 메시지가
  필요하다.
- 현재 구현은 point-to-point 제어이며 Nav2 map/costmap/장애물 우회는 포함하지 않는다.
- battery는 첫 정상 READY baseline과 비전 target으로 고정한 docking normal,
  busbar는 executor가 실제 적용한 작업 셀 회전의 docking normal을 사용한다.
  battery는 chassis 진행축 뒤의 접근 lane, busbar는 작업 셀 바깥 pre-approach를
  거치지만 일반 장애물 우회는 하지 않는다. 각 선분 경로가 비어 있는 장면에서만
  사용한다.
- 전체 공정은 `full_process_execute_isaac.py`가 물리 AMR와 arm state machine을
  같은 live stage에서 실행한다. AMR 전용 `execute_isaac.py`를 동시에 실행하지 않는다.
- `world` 문자열과 publisher 수만으로 서로 다른 USD 장면을 완전히 증명할 수는 없다.
  다른 장면은 별도 `ROS_DOMAIN_ID`로 격리해야 한다.
- 로컬 모델 `keypoints_busbar6pt_v2/v3.pt`, `seg_v3/v4.pt`는 현재 Git untracked다.
  이 컴퓨터에서는 실행할 수 있지만 clean clone 재현을 위해서는 모델 배포 방식이
  별도로 필요하다.

# 2026_ROKEY_B_1

## FMS 연동 ROS2 시스템 아키텍처

`src/`에 FMS 아키텍처 다이어그램을 그대로 옮긴 ROS2 Humble colcon 워크스페이스가 들어있다.
비전 스캔으로 작업을 만들고, 한 스테이션의 이동·조립·너트 체결까지 지휘하는 구조.

### 패키지 구성

| 패키지 | 계층 | 역할 |
|---|---|---|
| `fms_interfaces` | - | 전체 커스텀 msg 정의 (ament_cmake) |
| `fleet_manager_node` | FMS | 남은 작업 판단 → job 생성 → 작업 할당 |
| `behavior_node` | 지휘 | job 해석 → 조립 FSM → 복구 로직 → 체결 시퀀스 |
| `amr_node` | 실행 | 목표 스테이션 이동, 도착/오류 상태 보고 |
| `arm_node` | 실행 | 버스바 파지·삽입, 너트 접근·체결(토크 판정) |
| `perception_node` | 실행 | Hough Circle 스터드 검출, YOLO 버스바/너트 검출 |
| `fms_bringup` | - | 5개 노드를 한 번에 띄우는 launch 패키지 |

### 토픽 인터페이스

```
fleet_manager_node --PUB /fleet/job-->        behavior_node
fleet_manager_node <--SUB /fleet/report--     behavior_node

behavior_node --PUB /amr/goal-->               amr_node
behavior_node <--SUB /amr/status--             amr_node

behavior_node --PUB /busbar/command,/busbar/target--> arm_node
behavior_node --PUB /fasten/command-->                arm_node
behavior_node <--SUB /busbar/result,/fasten/result--  arm_node

perception_node --PUB /vision/stud_pose,/vision/busbar_grasp,/vision/nut_pose--> behavior_node

Isaac Sim --/camera/color,/camera/depth--> perception_node
Isaac Sim --/joint_states--> arm_node <--/arm/joint_command--> Isaac Sim
Isaac Sim --/amr/sim_pose--> amr_node <--/amr/cmd_vel--> Isaac Sim
```

Isaac Sim 씬 에셋은 이 저장소에 포함하지 않는다. 위 시뮬 인터페이스 토픽들을
Isaac Sim 쪽에서 퍼블리시/서브스크라이브하도록 연동하면 된다.

`/joint_states`, `/arm/joint_command`는 World0123.usd의 `/World/ActionGraph`에
OmniGraph 노드(`ROS2SubscribeJointState` → `IsaacArticulationController`,
`ROS2PublishJointState`)로 이미 연동돼 있다 (`scripts/setup_ros2_arm_bridge.py`로
1회 생성, 대상은 실제 PhysX 아티큘레이션 루트인 `/World/Nova_Carter/chassis_link`).
카메라(`/camera/color`, `/camera/depth`)와 TF도 `/World/Graph/camera_graph`,
`/World/ActionGraph`에 동일한 방식으로 이미 연동돼 있다. `/amr/sim_pose`,
`/amr/cmd_vel`(AMR 구동)만 아직 연동 안 됨 — TODO 참고.

### 빌드 및 실행

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

# 전체 노드 기동
ros2 launch fms_bringup fms_bringup.launch.py
```

Isaac Sim GUI에서 실제로 팔이 움직이는 걸 보려면:

```bash
# 1) Isaac Sim에서 World0123.usd를 열고 재생(▶) 버튼을 누른다
~/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/isaac-sim.sh \
    /home/rokey/EV_combine/src/Collected_World0123/World0123.usd

# 2) 다른 터미널에서 arm_node 실행
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 run arm_node arm_node

# 3) 또 다른 터미널에서 체결 명령 발행
ros2 topic pub -1 /fasten/command fms_interfaces/msg/FastenCommand \
    "{command: 'FASTEN', nut_id: 'nut1'}"
```

### 현재 상태 / TODO

각 노드는 pub/sub 인터페이스와 FSM/타이머 골격이 실제로 동작하며, 다음 부분은
Isaac Sim 및 실제 인식·모션 로직 연동이 필요한 TODO로 남겨져 있다.

- `fleet_manager_node`: 비전 스캔 기반 "남은 작업 판단" (현재는 station_1~3 순차 데모)
- `behavior_node`: 복구 로직의 실제 후퇴(retreat) 모션 위임
- `amr_node`: Isaac Sim 실좌표 피드백 기반 도착 판정 (`/amr/sim_pose`)
- `arm_node`: 너트 체결(APPROACH/FASTEN)은 World0123.usd에서 녹화한 관절 궤적
  (`scripts/record_nut_fasten_trajectory.py` → `data/nut_fasten_trajectory.json`)을
  재생하도록 구현됨. 버스바 파지·삽입은 아직 IK/모션 플래닝 TODO, 체결 성공 판정도
  아직 실제 토크 센서가 아닌 시뮬레이션 값
- `perception_node`: YOLO 모델 연동, 픽셀→3D pose 역투영 (depth + intrinsic)

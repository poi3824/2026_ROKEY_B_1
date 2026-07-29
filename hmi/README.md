# EV_combine HMI (모니터링 + 기본 제어)

HMI 브리지 노드가 운영 상태 토픽을 구독하고 제어 토픽을 발행한다. 정확한 FSM 상태와
정렬 오차, 비상정지 연동을 위해 `behavior_node`와 `error_fix`에 표준 메시지 기반의 작은
운영 인터페이스가 추가돼 있다.

## 구조

```
hmi/
├── backend/
│   ├── app.py            # Flask + Flask-SocketIO + rclpy 브릿지 (단일 파일)
│   └── requirements.txt
└── frontend/              # Vite + React 19 대시보드
    └── src/App.jsx
```

## 실행 방법

### 0) Isaac Sim + 실제 조립 노드

기본 월드는 최신 통합 작업 월드인
`/home/rokey/EV_combine/src/Collected_Busbar/Busbar.usd`다.

```bash
cd /home/rokey/EV_combine
source ~/.bashrc
isaac_ros2_setup
isaac_python isaacpjt/M0609/execute_isaac.py
```

다른 월드로 시험할 때는 파일을 수정하지 말고 실행 시 경로만 지정한다.

```bash
ISAAC_WORLD_PATH=/절대/경로/World0123.usd \
isaac_python isaacpjt/M0609/execute_isaac.py
```

Isaac Sim이 뜬 뒤 별도 터미널에서 AMR, 비전, FSM 노드를 실행한다. 특히
`amr_node`가 없으면 FSM이 `MOVE_AMR_BATTERY_SCAN`에서 대기하고 조립을 시작하지 않는다.

HMI에서만 작업을 제어하려면 다음 launch를 사용한다. 기본값으로 자동 작업 발행
`fleet_manager_node`는 실행되지 않으며, 모든 작업은 HMI 요청이 들어올 때까지 대기한다.

```bash
cd /home/rokey/EV_combine
ros2_setup
ros2 launch fms_bringup fms_bringup.launch.py
```

자동 작업 발행이 필요한 별도 시험에서만 명시적으로 활성화한다.

```bash
ros2 launch fms_bringup fms_bringup.launch.py start_fleet_manager:=true
```

### 1) 백엔드

```bash
cd /home/rokey/EV_combine
source /opt/ros/humble/setup.bash
source install/setup.bash        # fms_interfaces 등 커스텀 msg 사용하려면 필수
pip install --user -r hmi/backend/requirements.txt   # 최초 1회
python3 hmi/backend/app.py
```

`http://localhost:5055`에서 뜬다 (REST: `/api/state`, `/api/job`, `/api/cancel`,
`/api/emergency-stop`, `/api/manual-task`, `/api/cameras`, `/api/camera/stream`,
`/api/camera/snapshot`, 실시간: Socket.IO `state` 이벤트).

### 2) 프론트엔드

```bash
cd /home/rokey/EV_combine/hmi/frontend
npm install     # 최초 1회
npm run dev
```

`http://localhost:5173`에서 뜬다. `/api`, `/socket.io` 요청은 `vite.config.js`의 proxy 설정으로
자동으로 백엔드(5055)에 전달된다.

두 프로세스 다 띄운 상태에서 브라우저로 `http://localhost:5173` 접속하면 됨.

## 지금 보여주는 것 (모니터링)

- 실제 FSM 상태(`/behavior/state`)와 현재 task, Isaac Sim phase/progress/status
- 실시간 정렬 오차(`/alignment/error`): dx/dy(px), dTheta(deg), 유효성, 수렴 카운터
- AMR 상태(MOVING/ARRIVED/ERROR) 및 실시간 위치(x, y, theta)
- 최근 Fleet Job / Report
- ROS 작업 카메라 실시간 MJPEG 영상
  - 배터리 3/4 정렬, perception 오버레이, AMR 전·좌·우·후방 선택
  - 영상 확대, 브라우저 전체화면, 현재 프레임 JPEG 저장, FPS/해상도/토픽 표시
- FSM·AMR·비상정지 상태 변화 자동 이벤트 로그

## 지금 되는 것 (기본 제어)

- station 선택 후 `/fleet/job` 수동 발행 (behavior_node가 IDLE 상태여야 실제로 먹힘)
- `/amr/cancel` 발행으로 이동 중인 AMR 취소
- `/emergency_stop` 래치 제어: 작업 시작 차단, 진행 중 AMR/Arm 취소 요청, 정렬 보정 중단
- 전체 조립 공정 실행과 Arm 세부 단계 직접 실행
  - 배터리/버스바 스캔, 버스바 파지, 배터리 중심 이동, 정밀 정렬, 버스바 체결
  - 너트 1·2 각각 스캔, 파지, 체결
  - 세부 단계는 Behavior FSM이 `IDLE`일 때만 실행 가능

스테이션 선택 UI는 1~6을 모두 표시한다. 조립 좌표와 이동식
`/camera_bolt/rgb` 정렬 카메라가 연결된 곳은 `station_1`, `station_2`,
`station_3`다. `station_4`~`station_6`은 좌표가 추가되기 전까지 실행 잠금 상태로
표시된다.

⚠️ `start_fleet_manager:=true`로 실행하면 Fleet Manager가 자체 타이머로 작업을 발행하므로
HMI 수동 작업과 함께 사용하지 않는다.

## 통합 시 유의사항

- HMI 비상정지는 소프트웨어 레벨 정지 요청이다. 설비 인증용 하드웨어 E-stop이나 안전 PLC를
  대체하지 않는다.
- 비상정지 해제 시 Behavior FSM은 `IDLE`로 돌아가며 중단된 작업을 자동 재개하지 않는다.
- `README.md`(레포 루트)에 적힌 `/busbar/command`, `/fasten/command` 등은 죽은 인터페이스다.
  실제로는 `/execute_arm_task`(ExecuteArmTask action) 하나로 behavior_node ↔ arm_node가 통신함
  — 이 HMI도 그 사실을 기준으로 만들어졌다.
- 정식으로 colcon 워크스페이스에 편입하려면 `hmi/backend`를 ROS2 패키지로 감싸거나
  (launch 파일에 `hmi_bridge_node` 추가), 혹은 지금처럼 완전히 별도 프로세스로 계속 둬도 무방함
  — 토픽 인터페이스만 맞으면 어느 쪽이든 동작한다.

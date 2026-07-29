#!/usr/bin/env python3
"""EV_combine HMI 백엔드.

운영 상태 토픽을 구독하고 제어 토픽을 발행하는 "hmi_bridge_node"와 웹 UI를 연결한다.

구독 토픽 (읽기 전용, 모니터링):
  /fleet/job      (fms_interfaces/FleetJob)   - 새 job이 시작될 때
  /fleet/report   (fms_interfaces/FleetReport)- job 종료 결과
  /amr/status     (fms_interfaces/AmrStatus)  - AMR MOVING/ARRIVED/ERROR
  /amr/sim_pose   (geometry_msgs/Pose2D)      - AMR 실시간 (x, y, theta)
  /task_command   (std_msgs/String)           - 현재 실행 중인 task_type (FSM 단계 근사치)
  /behavior/state (std_msgs/String)           - 실제 Behavior FSM 상태
  /alignment/error(std_msgs/String JSON)      - 정렬 오차와 추적 상태
  /emergency_stop (std_msgs/Bool)             - 비상정지 래치 상태
  /isaac_phase    (std_msgs/String)           - Isaac Sim 세부 phase
  /isaac_progress (std_msgs/Float32)          - 진행률(%)
  /isaac_status   (std_msgs/String)           - SUCCESS / FAILURE:reason

발행 토픽 (기본 제어):
  /fleet/job   (fms_interfaces/FleetJob) - 수동으로 station에 ASSEMBLE job 발행
  /amr/cancel  (std_msgs/Empty)          - 이동 중인 AMR 취소
  /emergency_stop (std_msgs/Bool)        - 소프트웨어 비상정지/해제

실행 전 ROS 환경이 source 되어 있어야 한다 (ros2_setup 또는
`source /opt/ros/humble/setup.bash && source install/setup.bash`).
"""
import json
import threading
import time
import uuid

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Pose2D
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Empty, Float32, String

from fms_interfaces.msg import AmrGoal, AmrStatus, FleetJob, FleetReport
from fms_interfaces.action import ExecuteArmTask

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

_state_lock = threading.Lock()
_state = {
    "task_command": None,       # 최근 /task_command (현재 실행 중인 task_type)
    "fsm_state": None,          # 실제 behavior_node ProcessState
    "alignment": None,          # {dx_px, dy_px, dtheta_deg, valid, active, ...}
    "emergency_stop": False,
    "isaac_phase": None,
    "isaac_progress": 0.0,
    "isaac_status": None,
    "amr_status": None,         # {state, station_id, message}
    "amr_pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
    "last_job": None,           # {job_id, station_id, job_type, target}
    "last_report": None,        # {job_id, station_id, success, message}
    "camera": {
        "selected": "battery_4",
        "topic": "/camera_bolt/rgb",
        "online": False,
        "fps": 0.0,
        "width": 0,
        "height": 0,
    },
    "manual_task": {
        "task_type": None,
        "status": "IDLE",
        "message": None,
    },
    "updated_at": time.time(),
}

AMR_STATE_NAMES = {0: "IDLE", 1: "MOVING", 2: "ARRIVED", 3: "ERROR"}
CAMERAS = {
    "battery_4": {"label": "조립 정렬 카메라", "topic": "/camera_bolt/rgb"},
    "perception": {"label": "인식 결과 오버레이", "topic": "/perception/debug_image"},
    "amr_front": {"label": "AMR 전방", "topic": "/front_hawk/rgb"},
    "amr_left": {"label": "AMR 좌측", "topic": "/left_hawk/rgb"},
    "amr_right": {"label": "AMR 우측", "topic": "/right_hawk/rgb"},
    "amr_rear": {"label": "AMR 후방", "topic": "/back_hawk/rgb"},
}
SUPPORTED_STATIONS = {"station_1", "station_2", "station_3"}
AMR_STATION_TARGETS = {
    "station_1": {
        "battery": ("battery3", 0.6667, -0.0382, -1.5707),
        "busbar": ("busbar3", 0.5867, 1.9078, -1.5707),
    },
    "station_2": {
        "battery": ("battery4", 0.6667, -0.6617, -1.5707),
        "busbar": ("busbar4", -0.2271, 1.9078, -1.5707),
    },
    "station_3": {
        "battery": ("battery5", 0.6667, -1.1964, -1.5707),
        "busbar": ("busbar5", -0.9586, 1.9078, -1.5707),
    },
}
MANUAL_ARM_TASKS = {
    "SCAN_BATTERY",
    "SCAN_BUSBAR",
    "PICK_BUSBAR",
    "MOVE_BATTERY_CENTER",
    "FINE_ALIGNMENT",
    "ASSEMBLE_BUSBAR",
    "RETURN_HOME",
    "SCAN_NUT1",
    "PICK_NUT1",
    "ASSEMBLE_NUT1",
    "SCAN_NUT2",
    "PICK_NUT2",
    "ASSEMBLE_NUT2",
}

_camera_condition = threading.Condition()
_camera_jpeg = None
_camera_selected = "battery_4"
_camera_last_frame = 0.0
_camera_frame_count = 0
_camera_fps_started = time.monotonic()
_camera_last_state_push = 0.0
_amr_pose_last_state_push = 0.0
_isaac_progress_last_state_push = 0.0


def _push_update():
    with _state_lock:
        _state["updated_at"] = time.time()
        snapshot = dict(_state)
    socketio.emit("state", snapshot)


class HmiBridgeNode(Node):
    """기존 노드를 건드리지 않고 공개 토픽만 구독/발행하는 별도 브리지 노드."""

    def __init__(self):
        super().__init__("hmi_bridge_node")

        self.create_subscription(String, "/task_command", self._on_task_command, 10)
        self.create_subscription(String, "/behavior/state", self._on_fsm_state, 10)
        self.create_subscription(String, "/alignment/error", self._on_alignment, 10)
        emergency_qos = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            Bool, "/emergency_stop", self._on_emergency_stop, emergency_qos)
        self.create_subscription(String, "/isaac_phase", self._on_isaac_phase, 10)
        self.create_subscription(Float32, "/isaac_progress", self._on_isaac_progress, 10)
        self.create_subscription(String, "/isaac_status", self._on_isaac_status, 10)
        self.create_subscription(AmrStatus, "/amr/status", self._on_amr_status, 10)
        self.create_subscription(Pose2D, "/amr/sim_pose", self._on_amr_sim_pose, 10)
        self.create_subscription(FleetJob, "/fleet/job", self._on_fleet_job, 10)
        self.create_subscription(FleetReport, "/fleet/report", self._on_fleet_report, 10)
        for camera_id, config in CAMERAS.items():
            self.create_subscription(
                Image,
                config["topic"],
                lambda msg, cid=camera_id: self._on_camera_image(cid, msg),
                qos_profile_sensor_data,
            )

        self._job_pub = self.create_publisher(FleetJob, "/fleet/job", 10)
        self._amr_goal_pub = self.create_publisher(AmrGoal, "/amr/goal", 10)
        self._cancel_pub = self.create_publisher(Empty, "/amr/cancel", 10)
        self._station_pub = self.create_publisher(
            String, "/hmi/selected_station", 10)
        self._system_reset_pub = self.create_publisher(
            Empty, "/system/reset", 10)
        self._emergency_stop_pub = self.create_publisher(
            Bool, "/emergency_stop", emergency_qos)
        self._arm_action_client = ActionClient(
            self, ExecuteArmTask, "/execute_arm_task")
        self._manual_goal_handle = None
        self.create_timer(1.0, self._check_camera_health)

        self.get_logger().info("hmi_bridge_node started (monitoring + basic control)")

    def _on_task_command(self, msg: String):
        with _state_lock:
            _state["task_command"] = msg.data
        _push_update()

    def _on_fsm_state(self, msg: String):
        with _state_lock:
            _state["fsm_state"] = msg.data
        _push_update()

    def _on_alignment(self, msg: String):
        try:
            payload = json.loads(msg.data)
            alignment = {
                "dx_px": int(payload["dx_px"]),
                "dy_px": int(payload["dy_px"]),
                "dtheta_deg": float(payload["dtheta_deg"]),
                "valid": bool(payload["valid"]),
                "active": bool(payload["active"]),
                "hold_count": int(payload.get("hold_count", 0)),
                "hold_target": int(payload.get("hold_target", 30)),
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.get_logger().warning(
                f"잘못된 /alignment/error 메시지 무시: {exc}")
            return
        with _state_lock:
            _state["alignment"] = alignment
        _push_update()

    def _on_emergency_stop(self, msg: Bool):
        with _state_lock:
            _state["emergency_stop"] = bool(msg.data)
        _push_update()

    def _on_isaac_phase(self, msg: String):
        with _state_lock:
            _state["isaac_phase"] = msg.data
        _push_update()

    def _on_isaac_progress(self, msg: Float32):
        global _isaac_progress_last_state_push
        with _state_lock:
            _state["isaac_progress"] = float(msg.data)
        now = time.monotonic()
        if now - _isaac_progress_last_state_push >= 0.1:
            _isaac_progress_last_state_push = now
            _push_update()

    def _on_isaac_status(self, msg: String):
        with _state_lock:
            _state["isaac_status"] = msg.data
        _push_update()

    def _on_amr_status(self, msg: AmrStatus):
        with _state_lock:
            _state["amr_status"] = {
                "state": AMR_STATE_NAMES.get(msg.state, str(msg.state)),
                "station_id": msg.station_id,
                "message": msg.message,
            }
        _push_update()

    def _on_amr_sim_pose(self, msg: Pose2D):
        global _amr_pose_last_state_push
        with _state_lock:
            _state["amr_pose"] = {"x": msg.x, "y": msg.y, "theta": msg.theta}
        now = time.monotonic()
        if now - _amr_pose_last_state_push >= 0.1:
            _amr_pose_last_state_push = now
            _push_update()

    def _on_fleet_job(self, msg: FleetJob):
        with _state_lock:
            _state["last_job"] = {
                "job_id": msg.job_id,
                "station_id": msg.station_id,
                "job_type": msg.job_type,
                "target": msg.target,
            }
        _push_update()

    def _on_fleet_report(self, msg: FleetReport):
        with _state_lock:
            _state["last_report"] = {
                "job_id": msg.job_id,
                "station_id": msg.station_id,
                "success": msg.success,
                "message": msg.message,
            }
        _push_update()

    def _on_camera_image(self, camera_id: str, msg: Image):
        """선택된 ROS Image만 JPEG로 변환해 MJPEG 스트림 캐시에 저장한다."""
        global _camera_jpeg, _camera_last_frame, _camera_frame_count
        global _camera_fps_started, _camera_last_state_push
        if camera_id != _camera_selected:
            return

        frame = self._image_to_bgr(msg)
        if frame is None:
            return
        if frame.shape[1] > 1280:
            scale = 1280.0 / frame.shape[1]
            frame = cv2.resize(
                frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        if not ok:
            return

        now = time.monotonic()
        _camera_frame_count += 1
        elapsed = now - _camera_fps_started
        fps = _camera_frame_count / elapsed if elapsed > 0.0 else 0.0
        if elapsed >= 2.0:
            _camera_frame_count = 0
            _camera_fps_started = now

        with _camera_condition:
            _camera_jpeg = encoded.tobytes()
            _camera_last_frame = now
            _camera_condition.notify_all()

        if now - _camera_last_state_push >= 1.0:
            _camera_last_state_push = now
            with _state_lock:
                _state["camera"] = {
                    "selected": camera_id,
                    "topic": CAMERAS[camera_id]["topic"],
                    "online": True,
                    "fps": round(fps, 1),
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                }
            _push_update()

    def _image_to_bgr(self, msg: Image):
        try:
            encoding = msg.encoding.lower()
            channels = {
                "bgr8": 3, "rgb8": 3, "8uc3": 3,
                "bgra8": 4, "rgba8": 4, "mono8": 1, "8uc1": 1,
            }.get(encoding)
            if channels is None:
                self.get_logger().warning(
                    f"지원하지 않는 카메라 인코딩: {msg.encoding}")
                return None
            rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.step)
            frame = rows[:, :msg.width * channels].reshape(
                msg.height, msg.width, channels) if channels > 1 \
                else rows[:, :msg.width]
            if encoding == "rgb8":
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if encoding == "rgba8":
                return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            if encoding == "bgra8":
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if channels == 1:
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            return frame
        except (ValueError, TypeError) as exc:
            self.get_logger().warning(f"카메라 프레임 변환 실패: {exc}")
            return None

    def _check_camera_health(self):
        """프레임이 끊긴 카메라를 오프라인으로 표시한다."""
        if _camera_last_frame == 0.0 \
                or time.monotonic() - _camera_last_frame <= 2.5:
            return
        with _state_lock:
            if not _state["camera"]["online"]:
                return
            _state["camera"] = {
                **_state["camera"],
                "online": False,
                "fps": 0.0,
            }
        _push_update()

    def publish_job(self, station_id: str, job_type: str, target: str) -> str:
        msg = FleetJob()
        msg.job_id = f"hmi_{uuid.uuid4().hex[:8]}"
        msg.station_id = station_id
        msg.job_type = job_type
        msg.target = target
        msg.stamp = self.get_clock().now().to_msg()
        self._job_pub.publish(msg)
        self.get_logger().info(
            f"[HMI] PUB /fleet/job -> {msg.job_id} ({station_id}, {job_type})")
        return msg.job_id

    def publish_cancel(self):
        self._cancel_pub.publish(Empty())
        self.get_logger().info("[HMI] PUB /amr/cancel")

    def publish_amr_move(self, station_id: str, target_kind: str):
        station_id_for_amr, x, y, theta = AMR_STATION_TARGETS[
            station_id][target_kind]
        station_msg = String()
        station_msg.data = station_id
        self._station_pub.publish(station_msg)
        goal = AmrGoal()
        goal.station_id = station_id_for_amr
        goal.x = x
        goal.y = y
        goal.theta = theta
        self._amr_goal_pub.publish(goal)
        self.get_logger().info(
            f"[HMI] AMR 단독 이동 -> {station_id}/{target_kind} "
            f"({x:.4f}, {y:.4f}, {theta:.4f})")

    def publish_system_reset(self):
        self._cancel_pub.publish(Empty())
        estop_release = Bool()
        estop_release.data = False
        self._emergency_stop_pub.publish(estop_release)
        self._system_reset_pub.publish(Empty())
        if self._manual_goal_handle is not None:
            self._manual_goal_handle.cancel_goal_async()
            self._manual_goal_handle = None
        with _state_lock:
            _state["emergency_stop"] = False
            _state["task_command"] = None
            _state["alignment"] = None
            _state["isaac_phase"] = "IDLE"
            _state["isaac_progress"] = 0.0
            _state["isaac_status"] = "RESETTING"
            _state["amr_status"] = {
                "state": "IDLE",
                "station_id": "",
                "message": "시스템 제어 상태 초기화 요청",
            }
            _state["manual_task"] = {
                "task_type": None,
                "status": "IDLE",
                "message": "시스템 제어 상태 초기화 요청",
            }
        _push_update()
        self.get_logger().warning(
            "[HMI] 안전 초기화 발행: AMR 취소 + 비상정지 해제 + /system/reset")

    def publish_emergency_stop(self, enabled: bool):
        msg = Bool()
        msg.data = enabled
        self._emergency_stop_pub.publish(msg)
        if enabled:
            # Behavior 노드가 없더라도 AMR 브리지에는 직접 취소를 전달한다.
            self._cancel_pub.publish(Empty())
            if self._manual_goal_handle is not None:
                self._manual_goal_handle.cancel_goal_async()
                self._manual_goal_handle = None
                self._set_manual_task_result(
                    "CANCELED", "비상정지로 세부 작업 취소 요청")
        with _state_lock:
            _state["emergency_stop"] = enabled
            if not enabled:
                _state["manual_task"] = {
                    "task_type": None,
                    "status": "IDLE",
                    "message": "비상정지 해제 · 보존된 작업 재개 요청",
                }
        _push_update()
        self.get_logger().warning(
            f"[HMI] PUB /emergency_stop -> {enabled}")

    def start_manual_task(self, station_id: str, task_type: str):
        with _state_lock:
            if _state["emergency_stop"]:
                return False, "비상정지 상태에서는 세부 작업을 실행할 수 없습니다"
            if _state["fsm_state"] != "IDLE":
                return False, "Behavior FSM이 IDLE일 때만 세부 작업을 실행할 수 있습니다"
            if _state["manual_task"]["status"] in {"WAITING", "RUNNING"}:
                return False, "이미 실행 중인 세부 작업이 있습니다"
        if task_type not in MANUAL_ARM_TASKS:
            return False, "지원하지 않는 세부 작업입니다"
        if not self._arm_action_client.wait_for_server(timeout_sec=1.0):
            return False, "Arm Action Server가 응답하지 않습니다"

        station_msg = String()
        station_msg.data = station_id
        self._station_pub.publish(station_msg)
        goal = ExecuteArmTask.Goal()
        goal.task_type = task_type
        with _state_lock:
            _state["manual_task"] = {
                "task_type": task_type,
                "status": "WAITING",
                "message": "Arm Action Server 응답 대기",
            }
        _push_update()
        future = self._arm_action_client.send_goal_async(
            goal, feedback_callback=self._on_manual_task_feedback)
        future.add_done_callback(self._on_manual_goal_response)
        return True, "세부 작업 요청을 전송했습니다"

    def _on_manual_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._set_manual_task_result("FAILED", f"Goal 전송 실패: {exc}")
            return
        if not goal_handle.accepted:
            self._set_manual_task_result("REJECTED", "Arm 노드가 작업을 거부했습니다")
            return
        self._manual_goal_handle = goal_handle
        with _state_lock:
            _state["manual_task"]["status"] = "RUNNING"
            _state["manual_task"]["message"] = "작업 실행 중"
        _push_update()
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_manual_task_result)

    def _on_manual_task_feedback(self, feedback_msg):
        feedback = feedback_msg.feedback
        with _state_lock:
            _state["manual_task"]["message"] = (
                f"{feedback.sub_phase} · {feedback.progress_pct:.0f}%")
        _push_update()

    def _on_manual_task_result(self, future):
        self._manual_goal_handle = None
        try:
            result = future.result().result
            self._set_manual_task_result(
                "SUCCESS" if result.success else "FAILED",
                result.message or result.error_code,
            )
        except Exception as exc:
            self._set_manual_task_result("FAILED", f"결과 수신 실패: {exc}")

    def _set_manual_task_result(self, status: str, message: str):
        with _state_lock:
            _state["manual_task"]["status"] = status
            _state["manual_task"]["message"] = message
        _push_update()


_ros_node: HmiBridgeNode = None


def _ros_spin_thread():
    global _ros_node
    rclpy.init()
    _ros_node = HmiBridgeNode()
    try:
        rclpy.spin(_ros_node)
    finally:
        _ros_node.destroy_node()
        rclpy.shutdown()


@app.get("/api/state")
def get_state():
    with _state_lock:
        return jsonify(dict(_state))


@app.post("/api/job")
def post_job():
    if _ros_node is None:
        return jsonify({"error": "ROS 노드가 아직 준비되지 않았습니다"}), 503
    behavior_subscribers = _ros_node.count_subscribers("/fleet/job")
    if behavior_subscribers < 1:
        _ros_node.get_logger().error(
            "[HMI] Job 발행 차단: /fleet/job 구독자가 없습니다 "
            "(fms_bringup/behavior_node 실행 확인)")
        return jsonify({
            "error": (
                "Behavior 노드가 연결되지 않았습니다. "
                "fms_bringup launch 실행 상태를 확인하세요"
            )
        }), 503
    body = request.get_json(silent=True) or {}
    with _state_lock:
        if _state["emergency_stop"]:
            return jsonify({"error": "비상정지를 먼저 해제하세요"}), 409
        if _state["fsm_state"] != "IDLE":
            return jsonify({
                "error": f"Behavior FSM이 IDLE이 아닙니다: {_state['fsm_state']}"
            }), 409
    station_id = body.get("station_id", "station_1")
    if station_id not in SUPPORTED_STATIONS:
        return jsonify({
            "error": f"{station_id}은 현재 이동 좌표가 설정되지 않았습니다"
        }), 400
    job_type = body.get("job_type", "ASSEMBLE")
    target = body.get("target", "busbar_and_nut")
    job_id = _ros_node.publish_job(station_id, job_type, target)
    return jsonify({"job_id": job_id, "station_id": station_id})


@app.post("/api/cancel")
def post_cancel():
    if _ros_node is None:
        return jsonify({"error": "ROS 노드가 아직 준비되지 않았습니다"}), 503
    _ros_node.publish_cancel()
    return jsonify({"ok": True})


@app.post("/api/emergency-stop")
def post_emergency_stop():
    if _ros_node is None:
        return jsonify({"error": "ROS 노드가 아직 준비되지 않았습니다"}), 503
    body = request.get_json(silent=True) or {}
    if not isinstance(body.get("enabled"), bool):
        return jsonify({"error": "enabled는 boolean이어야 합니다"}), 400
    enabled = body["enabled"]
    _ros_node.publish_emergency_stop(enabled)
    return jsonify({"ok": True, "emergency_stop": enabled})


@app.post("/api/manual-task")
def post_manual_task():
    if _ros_node is None:
        return jsonify({"error": "ROS 노드가 아직 준비되지 않았습니다"}), 503
    body = request.get_json(silent=True) or {}
    station_id = body.get("station_id")
    task_type = body.get("task_type")
    if station_id not in SUPPORTED_STATIONS:
        return jsonify({
            "error": f"{station_id}은 현재 작업 좌표가 설정되지 않았습니다"
        }), 400
    ok, message = _ros_node.start_manual_task(station_id, task_type)
    return jsonify({"ok": ok, "message": message}), 202 if ok else 409


@app.post("/api/amr/move")
def post_amr_move():
    if _ros_node is None:
        return jsonify({"error": "ROS 노드가 아직 준비되지 않았습니다"}), 503
    body = request.get_json(silent=True) or {}
    station_id = body.get("station_id")
    target_kind = body.get("target_kind")
    if station_id not in SUPPORTED_STATIONS:
        return jsonify({"error": "지원하지 않는 스테이션입니다"}), 400
    if target_kind not in {"battery", "busbar"}:
        return jsonify({"error": "target_kind는 battery 또는 busbar여야 합니다"}), 400
    with _state_lock:
        if _state["emergency_stop"]:
            return jsonify({"error": "비상정지를 먼저 해제하세요"}), 409
        if _state["fsm_state"] != "IDLE":
            return jsonify({"error": "FSM이 IDLE일 때만 단독 이동할 수 있습니다"}), 409
    _ros_node.publish_amr_move(station_id, target_kind)
    return jsonify({
        "ok": True, "station_id": station_id, "target_kind": target_kind
    })


@app.post("/api/system-reset")
def post_system_reset():
    if _ros_node is None:
        return jsonify({"error": "ROS 노드가 아직 준비되지 않았습니다"}), 503
    _ros_node.publish_system_reset()
    return jsonify({"ok": True})


@app.get("/api/cameras")
def get_cameras():
    return jsonify([
        {"id": camera_id, **config}
        for camera_id, config in CAMERAS.items()
    ])


@app.post("/api/camera/select")
def select_camera():
    global _camera_selected, _camera_jpeg, _camera_last_frame
    global _camera_frame_count, _camera_fps_started
    body = request.get_json(silent=True) or {}
    camera_id = body.get("camera_id")
    if camera_id not in CAMERAS:
        return jsonify({"error": "알 수 없는 camera_id"}), 400
    _camera_selected = camera_id
    with _camera_condition:
        _camera_jpeg = None
        _camera_last_frame = 0.0
    _camera_frame_count = 0
    _camera_fps_started = time.monotonic()
    with _state_lock:
        _state["camera"] = {
            "selected": camera_id,
            "topic": CAMERAS[camera_id]["topic"],
            "online": False,
            "fps": 0.0,
            "width": 0,
            "height": 0,
        }
    _push_update()
    return jsonify({"ok": True, "camera_id": camera_id})


def _mjpeg_frames():
    last_frame = None
    while True:
        with _camera_condition:
            _camera_condition.wait_for(
                lambda: _camera_jpeg is not None
                and _camera_jpeg is not last_frame,
                timeout=2.0,
            )
            frame = _camera_jpeg
        if frame is None or frame is last_frame:
            continue
        last_frame = frame
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-cache\r\n\r\n" + frame + b"\r\n"
        )


@app.get("/api/camera/stream")
def camera_stream():
    return Response(
        _mjpeg_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/camera/snapshot")
def camera_snapshot():
    with _camera_condition:
        frame = _camera_jpeg
    if frame is None:
        return jsonify({"error": "수신된 카메라 프레임이 없습니다"}), 503
    filename = f"{_camera_selected}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    return Response(
        frame,
        mimetype="image/jpeg",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if __name__ == "__main__":
    t = threading.Thread(target=_ros_spin_thread, daemon=True)
    t.start()
    socketio.run(app, host="0.0.0.0", port=5055, debug=False, allow_unsafe_werkzeug=True)

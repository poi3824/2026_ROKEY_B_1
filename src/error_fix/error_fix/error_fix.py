#!/usr/bin/env python3
import sys
import math
import time
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge

from error_fix.alignment_policy import (
    FineAlignmentGate,
    LatestFramePairSynchronizer,
    advance_source_boundary,
)


def select_closest_hole_candidate(candidates, reference):
    """Return the valid hole nearest to the latched optical reference."""
    try:
        reference_x, reference_y = (
            float(reference[0]),
            float(reference[1]),
        )
    except (IndexError, TypeError, ValueError):
        return None
    if not math.isfinite(reference_x) or not math.isfinite(reference_y):
        return None

    valid_candidates = []
    for candidate in candidates:
        try:
            candidate_x, candidate_y, candidate_depth = (
                float(candidate[0]),
                float(candidate[1]),
                float(candidate[2]),
            )
        except (IndexError, TypeError, ValueError):
            continue
        if (
            not math.isfinite(candidate_x)
            or not math.isfinite(candidate_y)
            or not math.isfinite(candidate_depth)
            or candidate_depth <= 0.0
        ):
            continue
        valid_candidates.append(
            (int(round(candidate_x)), int(round(candidate_y)), candidate_depth)
        )

    if not valid_candidates:
        return None
    return min(
        valid_candidates,
        key=lambda item: (
            (item[0] - reference_x) ** 2
            + (item[1] - reference_y) ** 2,
            item[0],
            item[1],
            item[2],
        ),
    )


class BatteryAssemblyVisionNode(Node):
    def __init__(self):
        super().__init__('battery_assembly_vision_node')

        self.bridge = CvBridge()

        # ----------------------------------------------------------------------
        # ROS 2 Subscribers & Publishers
        # ----------------------------------------------------------------------
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            # RGB and depth arrive independently.  A one-message DDS queue
            # drops the only matching counterpart while the other callback is
            # doing Hough/GUI work, which turns otherwise matched source
            # stamps into repeated >100 ms skew rejections.  Keep the same
            # small bounded backlog as the synchronizer; source freshness is
            # still enforced below at <=1 s.
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.rgb_sub = self.create_subscription(
            Image, '/bolt_cam/rgb', self.rgb_callback, image_qos
        )
        self.depth_sub = self.create_subscription(
            Image, '/bolt_cam/depth', self.depth_callback, image_qos
        )

        # 트리거 명령 수신 (Isaac Sim으로부터 오차 보정 시작 신호 구독)
        self.sub_errorfix_cmd = self.create_subscription(
            String, '/errorfix_command', self.task_command_callback, 10
        )

        # Isaac Sim 목표 포즈 퍼블리셔 및 보정 완료 상태 퍼블리셔
        self.pub_target_pose = self.create_publisher(PoseStamped, '/target_pose', 10)
        self.pub_task_cmd = self.create_publisher(String, '/task_command', 10)

        # ----------------------------------------------------------------------
        # 제어 및 오차 보정 파라미터
        # ----------------------------------------------------------------------
        self.POSITION_TOLERANCE_PX = 2
        self.TOLERANCE_DEG = 3
        self.HOLD_FRAMES = 30
        self.ALIGNMENT_DECISION_FRAMES = 3
        self.BOLT_REFERENCE_FRAMES = 3
        self.BOLT_REFERENCE_TOLERANCE_PX = 5.0
        self.FIXED_STEP_M = 0.0002
        self.MAX_VALID_PIXEL_ERR = 400  # 이상치 픽셀 오차 스킵 가드

        # 양쪽 callback이 갱신하는 최신 RGB/depth 후보와 source boundary.
        self._last_observed_rgb_stamp_ns = None
        self._last_observed_depth_stamp_ns = None
        self.fixed_bolt_coords = []
        self._bolt_reference_candidates = []
        self.busbar_hole_coords = []
        self.busbar_hole_depths = []
        self.battery_angle = 0.0
        self.busbar_angle = 0.0

        self.battery_line_data = None
        self.busbar_line_data = None
        self.roi_rect = None

        # 시작 제어 플래그
        self.is_active = False          # 트리거 수신 전까지는 False
        self.bolts_detected = False     # 노드 실행 직후 백그라운드에서 검출 진행
        self.alignment_generation = None
        self.alignment_legacy_session = False
        self._roi_logged = False        # ROI 크기를 한 번만 로그로 찍기 위한 플래그

        # 버스바 미검출 시 Hold 처리용 (픽셀 좌표 차이)
        self.last_valid_dx_px = 0       # hx - bx (양수: 구멍이 오른쪽, 음수: 구멍이 왼쪽)
        self.last_valid_dy_px = 0       # hy - by (양수: 구멍이 아래쪽, 음수: 구멍이 위쪽)
        self.last_valid_dtheta = 0.0
        self.has_valid_tracking = False

        self.alignment_gate = FineAlignmentGate(
            position_tolerance_px=self.POSITION_TOLERANCE_PX,
            angle_tolerance_deg=self.TOLERANCE_DEG,
            hold_frames=self.HOLD_FRAMES,
            correction_step_m=self.FIXED_STEP_M,
            decision_frames=self.ALIGNMENT_DECISION_FRAMES,
        )
        self.frame_pair_sync = LatestFramePairSynchronizer(
            max_skew_sec=0.2,
            max_receipt_age_sec=1.0,
            history_size=10,
        )
        self.hold_count = self.alignment_gate.hold_count

        # 10Hz 오차 보정 제어 루프
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info(
            "🚀 [Vision Node] 실행 완료. "
            "bolt_2 optical-center 정렬 트리거 대기 중...")

    def task_command_callback(self, msg: String):
        """외부 Behavior 또는 Isaac Sim에서 오차 보정 시작 명령을 수신한다."""
        cmd = msg.data
        starts_errorfix = (
            cmd == "START_ERRORFIX_CORRECTION"
            or cmd.startswith("START_ERRORFIX_CORRECTION:")
        )
        if starts_errorfix or cmd in [
            "START_VISION_CORRECTION",
            "MOVE_BATTERY_CENTER_SUCCESS",
        ]:
            generation = None
            if cmd.startswith("START_ERRORFIX_CORRECTION:"):
                generation_text = cmd.partition(":")[2]
                try:
                    generation = int(generation_text)
                except ValueError:
                    self.get_logger().warn(
                        f"잘못된 FINE_ALIGNMENT generation 무시: {cmd}")
                    return
                if generation <= 0:
                    self.get_logger().warn(
                        f"0 이하 FINE_ALIGNMENT generation 무시: {cmd}")
                    return
            self.get_logger().info(
                f"\n>>> [Vision Correction] 오차 보정 시작 신호 수신 ({cmd})!")
            # START 자체도 새 경계를 세운다. RESET과 START 사이에 image
            # callback이 끼어도 그 프레임은 기준 볼트 후보가 될 수 없다.
            self._reset_bolt_detection()
            self.alignment_generation = generation
            self.alignment_legacy_session = generation is None
            self.is_active = True
        elif cmd == "RESET_BOLT_DETECTION":
            self._reset_bolt_detection()
        elif cmd.startswith("FINE_ALIGNMENT_STEP_SETTLED:"):
            stamp_text = cmd.partition(":")[2]
            try:
                stamp_ns = int(stamp_text)
            except ValueError:
                self.get_logger().warn(
                    f"잘못된 FINE_ALIGNMENT settle ack 무시: {cmd}")
                return

            if not self.alignment_gate.acknowledge_settled(stamp_ns):
                self.get_logger().warn(
                    "stale FINE_ALIGNMENT settle ack 무시: "
                    f"{stamp_ns}")
                return

            # Ack 이전에 촬영되었거나 callback queue에 남아 있던 RGB/depth는
            # 다음 correction의 3-frame 합의에 들어갈 수 없다.
            self.frame_pair_sync.reset(
                rgb_boundary_ns=self._last_observed_rgb_stamp_ns,
                depth_boundary_ns=self._last_observed_depth_stamp_ns,
            )
            self.has_valid_tracking = False
            self.get_logger().info(
                "FINE_ALIGNMENT step settle 확인 "
                f"(stamp_ns={stamp_ns}); post-motion 3-frame 대기")

    def _reset_alignment_tracking(self):
        """Clear cached observations and convergence state."""
        self.alignment_gate.reset()
        self.has_valid_tracking = False
        self.last_valid_dx_px = 0
        self.last_valid_dy_px = 0
        self.last_valid_dtheta = 0.0
        self.hold_count = self.alignment_gate.hold_count

    def _reset_bolt_detection(self):
        """Discard observations captured before the bolt camera was repositioned."""
        self.is_active = False
        self.bolts_detected = False
        self.alignment_generation = None
        self.alignment_legacy_session = False
        self.fixed_bolt_coords = []
        self._bolt_reference_candidates = []
        self.busbar_hole_coords = []
        self.busbar_hole_depths = []
        self.battery_line_data = None
        self.busbar_line_data = None
        self.roi_rect = None
        self.frame_pair_sync.reset(
            rgb_boundary_ns=self._last_observed_rgb_stamp_ns,
            depth_boundary_ns=self._last_observed_depth_stamp_ns,
        )
        self._reset_alignment_tracking()
        self._roi_logged = False
        self.get_logger().info(
            ">>> [Vision Node] 카메라 이동 감지 -> 고정 볼트 위치 재탐색 시작")

    def depth_callback(self, msg):
        depth_stamp_ns = self._stamp_ns(msg)
        depth_receipt_ns = time.monotonic_ns()
        self._last_observed_depth_stamp_ns = advance_source_boundary(
            self._last_observed_depth_stamp_ns,
            depth_stamp_ns,
        )
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_frame = np.nan_to_num(
                depth_img,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")
            return
        self.frame_pair_sync.cache_depth(
            depth_frame,
            depth_stamp_ns,
            depth_receipt_ns,
        )

    def rgb_callback(self, msg):
        rgb_stamp_ns = self._stamp_ns(msg)
        rgb_receipt_ns = time.monotonic_ns()
        self._last_observed_rgb_stamp_ns = advance_source_boundary(
            self._last_observed_rgb_stamp_ns,
            rgb_stamp_ns,
        )
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")
            return

        self.frame_pair_sync.cache_rgb(
            frame,
            rgb_stamp_ns,
            rgb_receipt_ns,
        )
        if not self.is_active:
            display_img = frame.copy()
            cv2.putText(
                display_img,
                "Waiting for alignment trigger...",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Battery Assembly Vision - RGB Tracking", display_img)
            cv2.waitKey(1)

    def _try_process_cached_pair(self):
        """Process once when either callback completes one strict frame pair."""
        if not self.is_active:
            return
        pair, pair_reason = self.frame_pair_sync.try_accept(
            now_receipt_ns=time.monotonic_ns(),
        )
        if pair is None:
            if pair_reason not in {'RGB unavailable', 'depth unavailable'}:
                self.get_logger().warn(
                    f'[frame barrier] {pair_reason}',
                    throttle_duration_sec=2.0,
                )
            return
        self._process_accepted_pair(
            pair.rgb.payload,
            pair.depth.payload,
            pair.rgb.stamp_ns,
        )

    def _process_accepted_pair(
        self,
        frame,
        depth_frame,
        rgb_stamp_ns,
    ):
        """Run tracking only for a fresh synchronized pair accepted once."""
        display_img = frame.copy()

        # ----------------------------------------------------------------------
        # 같은 위치의 post-reset bolt가 3 frame 이어져야 latch.
        # ----------------------------------------------------------------------
        if not self.bolts_detected:
            # Executor가 camera origin을 station bolt_2 world XY로 옮기고
            # intrinsic principal point도 정확히 image center이다. 버스바가
            # bolt를 가리는 이 단계에서 Hough로 재검출하면 hole 자체를 bolt로
            # 오인하거나 검출 0개로 영구 대기하므로 optical center를 기준으로
            # 삼는다. fresh/synchronized pair 3개 조건은 그대로 유지한다.
            detected = (
                frame.shape[1] // 2,
                frame.shape[0] // 2,
            )
            self._update_bolt_reference(detected)
            if not self.bolts_detected:
                self._submit_invalid_frame(rgb_stamp_ns)
                cv2.putText(
                    display_img,
                    "Stabilizing post-move bolt reference...",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow(
                    "Battery Assembly Vision - RGB Tracking",
                    display_img,
                )
                cv2.waitKey(1)
                return

        # ----------------------------------------------------------------------
        # 버스바 구멍 추적 및 오차 연산.
        # ----------------------------------------------------------------------
        # 현재 RGB frame에서 hole을 다시 확인하기 전에는 이전 frame의
        # tracking을 절대 재사용하지 않는다. 미검출 frame은 유효 표본이나
        # hold 진행도를 늘리지 않는다.
        self.has_valid_tracking = False
        if self.bolts_detected:
            self.detect_yellow_busbar_holes_depth(
                frame,
                depth_frame,
                min_busbar_area=2000,
            )

            # ROI 박스 및 에지 라인 오버레이
            if self.roi_rect is not None:
                rx, ry, rw, rh = self.roi_rect
                cv2.rectangle(
                    display_img,
                    (rx, ry),
                    (rx + rw, ry + rh),
                    (0, 255, 128),
                    2,
                )

            self.draw_extended_line(
                display_img,
                self.battery_line_data,
                color=(0, 255, 0),
                label=f"Battery Edge ({self.battery_angle:.1f}deg)",
            )
            self.draw_extended_line(
                display_img,
                self.busbar_line_data,
                color=(0, 255, 255),
                label=f"Busbar Edge ({self.busbar_angle:.1f}deg)",
            )

            for bx, by in self.fixed_bolt_coords:
                cv2.circle(display_img, (bx, by), 3, (0, 255, 0), -1)

            if len(self.busbar_hole_coords) > 0:
                bx, by = self.fixed_bolt_coords[0]
                hx, hy = self.busbar_hole_coords[0]
                hole_depth = (
                    self.busbar_hole_depths[0]
                    if len(self.busbar_hole_depths) > 0
                    else 0.0
                )

                dx_px = hx - bx  # 영상 기준 X차이 (양수: 구멍이 오른쪽, 음수: 구멍이 왼쪽)
                dy_px = hy - by  # 영상 기준 Y차이 (양수: 구멍이 아래쪽, 음수: 구멍이 위쪽)
                angle_error = self.battery_angle - self.busbar_angle

                # 이상치 픽셀 오차 스킵 가드
                if (
                    abs(dx_px) > self.MAX_VALID_PIXEL_ERR
                    or abs(dy_px) > self.MAX_VALID_PIXEL_ERR
                    or not math.isfinite(angle_error)
                ):
                    self.get_logger().warn(
                        "⚠️ [비전 노이즈 차단] 비정상적 픽셀 오차 감지! "
                        f"dx:{dx_px}, dy:{dy_px} -> 프레임 스킵")
                    self.has_valid_tracking = False
                else:
                    self.last_valid_dx_px = dx_px
                    self.last_valid_dy_px = dy_px
                    self.last_valid_dtheta = angle_error
                    self.has_valid_tracking = True

                    cv2.circle(display_img, (hx, hy), 5, (0, 0, 255), -1)
                    cv2.line(
                        display_img,
                        (bx, by),
                        (hx, hy),
                        (255, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        display_img,
                        f"E:({dx_px},{dy_px}) Z:{hole_depth:.2f}",
                        (hx + 8, hy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 255),
                        1,
                    )

                    status_line = (
                        f"dx:{dx_px:+2d}px, dy:{dy_px:+2d}px, "
                        f"dTheta:{angle_error:+.2f}deg "
                        f"[{self.hold_count}/{self.HOLD_FRAMES}]")
                    sys.stdout.write(f"\r\033[K[Tracking Status] {status_line}")
                    sys.stdout.flush()

        self.alignment_gate.submit(
            rgb_stamp_ns,
            valid=self.has_valid_tracking,
            dx_px=self.last_valid_dx_px,
            dy_px=self.last_valid_dy_px,
            dtheta_deg=self.last_valid_dtheta,
        )
        self.hold_count = self.alignment_gate.hold_count
        # A continuously ready image subscription can starve the timer in a
        # SingleThreadedExecutor. Consume a decision in the accepted-frame
        # path; consume() clears it, so the timer remains a safe fallback.
        self._consume_alignment_decision()

        cv2.imshow("Battery Assembly Vision - RGB Tracking", display_img)
        cv2.waitKey(1)

    @staticmethod
    def _stamp_ns(msg):
        return (
            int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec)
        )

    def _submit_invalid_frame(self, rgb_stamp_ns):
        self.has_valid_tracking = False
        self.alignment_gate.submit(rgb_stamp_ns, valid=False)
        self.hold_count = self.alignment_gate.hold_count

    def _update_bolt_reference(self, detected):
        if detected is None:
            self.fixed_bolt_coords = []
            self._bolt_reference_candidates = []
            return

        candidate = np.asarray(detected, dtype=float)
        if self._bolt_reference_candidates:
            anchor = np.mean(self._bolt_reference_candidates, axis=0)
            if np.linalg.norm(candidate - anchor) > (
                self.BOLT_REFERENCE_TOLERANCE_PX
            ):
                self._bolt_reference_candidates = []
        self._bolt_reference_candidates.append(candidate)
        self.fixed_bolt_coords = []

        if len(self._bolt_reference_candidates) < self.BOLT_REFERENCE_FRAMES:
            return
        reference = np.mean(self._bolt_reference_candidates, axis=0)
        bolt_x, bolt_y = (int(round(value)) for value in reference)
        self.fixed_bolt_coords = [(bolt_x, bolt_y)]
        self.bolts_detected = True
        self.battery_angle = 0.0
        self.battery_line_data = (
            1.0,
            0.0,
            float(bolt_x),
            float(bolt_y),
        )
        self.get_logger().info(
            "✅ [초기화 완료] post-move bolt_2 optical center "
            f"{self.BOLT_REFERENCE_FRAMES} frame 확인: "
            f"X={bolt_x}, Y={bolt_y}")

    def control_loop(self):
        """Process one strict pair outside the image-subscription callbacks.

        The callbacks only convert/cache frames.  This keeps their QoS queues
        draining while contour/Hough work runs here, so a valid RGB/depth pair
        is not lost merely because the other callback was delayed.  The
        synchronizer still rejects source skew over 0.2 s, stale receipt over
        1 s, and any retrograde source stamp.
        """
        self._try_process_cached_pair()
        self._consume_alignment_decision()

    def _consume_alignment_decision(self):
        """Consume and apply a pending three-frame decision exactly once."""
        if not self.is_active or not self.bolts_detected:
            return

        decision = self.alignment_gate.consume()
        self.hold_count = self.alignment_gate.hold_count
        if decision is None:
            return
        if decision.deferred:
            self.get_logger().info(
                "FINE_ALIGNMENT near-boundary 3-frame jitter 보류; "
                "같은 방향의 새 3-frame 확인 대기",
                throttle_duration_sec=1.0,
            )
            return
        dtheta = decision.dtheta_deg

        # ±2px detector quantization 안의 합의 표본은 이동 없이 hold한다.
        if decision.aligned:
            if decision.success:
                self.get_logger().info(
                    "\n==========================================")
                self.get_logger().info(
                    " ★ [정밀 정렬 완료] "
                    f"±{self.POSITION_TOLERANCE_PX}px 범위의 서로 다른 "
                    f"유효 RGB frame {self.HOLD_FRAMES}개 누적 달성")
                self.get_logger().info(
                    "==========================================")

                msg = String()
                if self.alignment_generation is not None:
                    msg.data = (
                        "ALIGNMENT_SUCCESS:"
                        f"{self.alignment_generation}"
                    )
                elif self.alignment_legacy_session:
                    # Bare START commands remain a legacy-only session.  They
                    # never inherit or reuse a generation from a prior FINE
                    # task, and the current executor intentionally ignores
                    # their unscoped completion.
                    msg.data = "ALIGNMENT_SUCCESS"
                else:
                    self.get_logger().error(
                        "활성 alignment session 식별자가 없어 완료 신호를 "
                        "발행하지 않습니다")
                    self.is_active = False
                    self._reset_alignment_tracking()
                    return
                self.pub_task_cmd.publish(msg)
                self.is_active = False
                self.alignment_generation = None
                self.alignment_legacy_session = False
                self._reset_alignment_tracking()
                return
            return

        if not decision.consensus:
            self.get_logger().warn(
                "FINE_ALIGNMENT 3-frame 합의 실패; correction 생략",
                throttle_duration_sec=2.0,
            )
            return

        # --- dx (위/아래) 설정 ---
        step_x = decision.correction_x_m

        # --- dy (왼쪽/오른쪽) 설정 ---
        step_y = decision.correction_y_m

        # --- 각도 보정 ---
        if dtheta > self.TOLERANCE_DEG:
            step_dtheta = -0.01
        elif dtheta < -self.TOLERANCE_DEG:
            step_dtheta = 0.01
        else:
            step_dtheta = 0.0

        # 메시지 작성 및 퍼블리시
        target_msg = PoseStamped()
        target_msg.header.stamp.sec = int(
            decision.stamp_ns // 1_000_000_000)
        target_msg.header.stamp.nanosec = int(
            decision.stamp_ns % 1_000_000_000)
        target_msg.header.frame_id = "fine_alignment_delta"
        if self.alignment_generation is not None:
            target_msg.header.frame_id += (
                f":{self.alignment_generation}"
            )

        target_msg.pose.position.x = step_x
        target_msg.pose.position.y = step_y
        target_msg.pose.position.z = 0.0

        yaw_rad = math.radians(step_dtheta)
        target_msg.pose.orientation.z = math.sin(yaw_rad / 2.0)
        target_msg.pose.orientation.w = math.cos(yaw_rad / 2.0)

        self.pub_target_pose.publish(target_msg)

    # --------------------------------------------------------------------------
    # 비전 영상 처리 함수
    # --------------------------------------------------------------------------
    def detect_static_bolts_hough(self, frame):
        h, w = frame.shape[:2]
        # FINE_ALIGNMENT가 시작될 때 bolt_camera를 현재 station의 bolt_2 바로
        # 위(world XY)로 옮기고 RESET_BOLT_DETECTION을 먼저 보낸다. 따라서 이
        # 보정 경로의 기준 bolt는 화면 중앙 부근에 있어야 한다. SCAN_BATTERY의
        # pair-midpoint 관측은 FINE_ALIGNMENT reset에서 폐기된다.
        roi_frac_w, roi_frac_h = 0.22, 0.22
        roi_w, roi_h = int(w * roi_frac_w), int(h * roi_frac_h)
        x_start, y_start = int((w - roi_w) / 2), int((h - roi_h) / 2)
        self.roi_rect = (x_start, y_start, roi_w, roi_h)

        if not self._roi_logged:
            self.get_logger().info(
                f"[ROI] {roi_w}x{roi_h}px (면적={roi_w * roi_h}px², "
                f"전체 프레임({w}x{h}) 대비 {roi_frac_w * roi_frac_h * 100:.1f}%)"
            )
            self._roi_logged = True

        # 1. 고정 볼트 ROI 추출
        roi = frame[y_start:y_start + roi_h, x_start:x_start + roi_w]

        # 2. Grayscale 및 CLAHE 명암비 강조
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(gray)

        # 3. Otsu 이진화 및 가우시안 블러
        _, thresh = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blurred_bolt = cv2.GaussianBlur(thresh, (5, 5), 0)

        # 4. Hough Circles 원 검출
        circles = cv2.HoughCircles(
            blurred_bolt, cv2.HOUGH_GRADIENT, dp=1.0,
            minDist=15, param1=50, param2=12, minRadius=2, maxRadius=4
        )

        self.fixed_bolt_coords = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            detected_candidates = []

            for i in circles[0, :]:
                cx, cy, r = int(i[0]), int(i[1]), int(i[2])
                detected_candidates.append((cx + x_start, cy + y_start, cx, cy, r))

            # X 좌표 기준 오름차순 정렬 (가장 왼쪽 원 선택)
            detected_candidates.sort(key=lambda c: c[0])

            best_candidate = detected_candidates[0]
            self.fixed_bolt_coords = [(best_candidate[0], best_candidate[1])]

        # 5. Canny Edge 및 라인 검출
        edges = cv2.Canny(clahe, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=40,
            minLineLength=40,
            maxLineGap=10,
        )

        best_line = None
        max_length = 0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                angle_deg = math.degrees(math.atan2(dy, dx))

                if (
                    -20.0 <= angle_deg <= 20.0
                    or angle_deg >= 160.0
                    or angle_deg <= -160.0
                ):
                    if length > max_length:
                        max_length = length
                        best_line = (x1, y1, x2, y2)

        if best_line is not None:
            x1, y1, x2, y2 = best_line
            gx1, gy1 = x1 + x_start, y1 + y_start
            gx2, gy2 = x2 + x_start, y2 + y_start

            dx_g, dy_g = gx2 - gx1, gy2 - gy1
            if dx_g < 0:
                dx_g, dy_g = -dx_g, -dy_g

            norm = math.hypot(dx_g, dy_g)
            vx, vy = dx_g / norm, dy_g / norm

            self.battery_angle = math.degrees(math.atan2(vy, vx))
            self.battery_line_data = (vx, vy, float(gx1), float(gy1))
        elif self.fixed_bolt_coords:
            bx, by = self.fixed_bolt_coords[0]
            self.battery_angle = 0.0
            self.battery_line_data = (1.0, 0.0, float(bx), float(by))

    def detect_yellow_busbar_holes_depth(self, frame, depth_img, min_busbar_area=2000):
        h, w = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([35, 255, 255])

        raw_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            raw_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        filtered_mask = np.zeros_like(raw_mask)
        main_busbar_cnt = None
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_busbar_area:
                cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
                if area > max_area:
                    max_area = area
                    main_busbar_cnt = cnt

        mask_edges = cv2.Canny(filtered_mask, 50, 150)
        mask_lines = cv2.HoughLinesP(
            mask_edges,
            1,
            np.pi / 180,
            threshold=30,
            minLineLength=20,
            maxLineGap=10,
        )

        bottommost_line = None
        max_y = -1

        if mask_lines is not None:
            for line in mask_lines:
                x1, y1, x2, y2 = line[0]
                dx, dy = x2 - x1, y2 - y1
                angle_deg = math.degrees(math.atan2(dy, dx))

                if (
                    -20.0 <= angle_deg <= 20.0
                    or angle_deg >= 160.0
                    or angle_deg <= -160.0
                ):
                    mid_y = (y1 + y2) / 2.0
                    if mid_y > max_y:
                        max_y = mid_y
                        bottommost_line = (x1, y1, x2, y2)

        if bottommost_line is not None:
            mx1, my1, mx2, my2 = bottommost_line
            mdx, mdy = mx2 - mx1, my2 - my1
            if mdx < 0:
                mdx, mdy = -mdx, -mdy
            norm = math.hypot(mdx, mdy)
            vx, vy = mdx / norm, mdy / norm
            self.busbar_angle = math.degrees(math.atan2(vy, vx))
            cx, cy = (mx1 + mx2) / 2.0, (my1 + my2) / 2.0
            self.busbar_line_data = (vx, vy, float(cx), float(cy))
        elif main_busbar_cnt is not None:
            [vx, vy, cx, cy] = cv2.fitLine(
                main_busbar_cnt,
                cv2.DIST_L2,
                0,
                0.01,
                0.01,
            )
            vx_val, vy_val = vx[0], vy[0]
            if vx_val < 0:
                vx_val, vy_val = -vx_val, -vy_val
            self.busbar_angle = math.degrees(math.atan2(vy_val, vx_val))
            self.busbar_line_data = (vx_val, vy_val, cx[0], cy[0])

        depth_h, depth_w = depth_img.shape[:2]
        depth_img_resized = (
            cv2.resize(
                depth_img,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
            if (depth_h, depth_w) != (h, w)
            else depth_img
        )

        depth_norm = cv2.normalize(
            depth_img_resized,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        depth_blur = cv2.GaussianBlur(depth_norm, (5, 5), 0)

        laplacian = cv2.Laplacian(depth_blur, cv2.CV_8U, ksize=3)
        masked_laplacian = cv2.bitwise_and(laplacian, laplacian, mask=filtered_mask)

        circles = cv2.HoughCircles(
            masked_laplacian, cv2.HOUGH_GRADIENT, dp=1.0,
            minDist=10, param1=30, param2=12, minRadius=3, maxRadius=12
        )

        detected_holes = []
        if circles is not None:
            for circle in np.asarray(circles).reshape(-1, 3):
                if not np.all(np.isfinite(circle)):
                    continue
                cx, cy = (
                    int(round(float(circle[0]))),
                    int(round(float(circle[1]))),
                )
                if not (0 <= cx < w and 0 <= cy < h):
                    continue
                if filtered_mask[cy, cx] == 0:
                    continue
                val = float(depth_img_resized[cy, cx])
                if not math.isfinite(val) or val <= 0.0:
                    continue
                detected_holes.append((cx, cy, val))

        reference = (w // 2, h // 2)
        if self.fixed_bolt_coords:
            fixed_x, fixed_y = self.fixed_bolt_coords[0]
            if math.isfinite(fixed_x) and math.isfinite(fixed_y):
                reference = (fixed_x, fixed_y)
        selected_hole = select_closest_hole_candidate(
            detected_holes,
            reference,
        )
        if selected_hole is None:
            self.busbar_hole_coords = []
            self.busbar_hole_depths = []
        else:
            self.busbar_hole_coords = [
                (selected_hole[0], selected_hole[1])
            ]
            self.busbar_hole_depths = [selected_hole[2]]

    def draw_extended_line(self, img, line_data, color, thickness=2, label=""):
        if line_data is None:
            return
        vx, vy, x0, y0 = line_data
        h, w = img.shape[:2]

        scale = max(w, h) * 2
        p1 = (int(x0 - vx * scale), int(y0 - vy * scale))
        p2 = (int(x0 + vx * scale), int(y0 + vy * scale))

        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
        if label:
            cv2.putText(img, label, (int(x0) - 100, int(y0) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main(args=None):
    rclpy.init(args=args)
    node = BatteryAssemblyVisionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

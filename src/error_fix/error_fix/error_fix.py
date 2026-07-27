#!/usr/bin/env python3
import sys
import math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

# ──────────────────────────────────────────────────────────────────────────
# camera_bolt 고정 World Pose (Busbar.usd -> /World/Camera_bolt prim에서 추출).
# 카메라가 /World에 고정 마운트된 탑다운(수직 아래) 카메라라서 실행 중 갱신 불필요.
# ──────────────────────────────────────────────────────────────────────────
CAMERA_BOLT_WORLD_POS = np.array([1.259848, -0.005358, 1.218396])
CAMERA_BOLT_WORLD_QUAT_WXYZ = np.array([0.7071067811865476, 0.0, 0.0, 0.7071067811865475])
_CAMERA_BOLT_ROT = R.from_quat([
    CAMERA_BOLT_WORLD_QUAT_WXYZ[1], CAMERA_BOLT_WORLD_QUAT_WXYZ[2],
    CAMERA_BOLT_WORLD_QUAT_WXYZ[3], CAMERA_BOLT_WORLD_QUAT_WXYZ[0],
])
# ROS 카메라 광학 프레임(X:right, Y:down, Z:forward) -> USD 카메라 로컬 프레임(X:right, Y:up, Z:backward)
_OPTICAL_TO_USD_CAM = np.diag([1.0, -1.0, -1.0])


def pixel_depth_to_world(u, v, depth_m, fx, fy, cx, cy):
    """픽셀 좌표(u,v) + depth(m)를 camera_bolt 고정 pose 기준 world 좌표(m)로 변환."""
    x_opt = (u - cx) * depth_m / fx
    y_opt = (v - cy) * depth_m / fy
    z_opt = depth_m
    p_cam_usd = _OPTICAL_TO_USD_CAM @ np.array([x_opt, y_opt, z_opt])
    return _CAMERA_BOLT_ROT.apply(p_cam_usd) + CAMERA_BOLT_WORLD_POS


class BatteryAssemblyVisionNode(Node):
    def __init__(self):
        super().__init__('battery_assembly_vision_node')
        
        self.bridge = CvBridge()

        # ----------------------------------------------------------------------
        # ROS 2 Subscribers & Publishers
        # ----------------------------------------------------------------------
        self.rgb_sub = self.create_subscription(
            Image, '/camera_bolt/rgb', self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera_bolt/depth', self.depth_callback, 10
        )
        self.caminfo_sub = self.create_subscription(
            CameraInfo, '/camera_bolt/camera_info', self.caminfo_callback, 10
        )

        # 트리거 명령 수신 (Isaac Sim으로부터 오차 보정 시작 신호 구독)
        self.sub_errorfix_cmd = self.create_subscription(
            String, '/errorfix_command', self.task_command_callback, 10
        )

        # Isaac Sim 목표 포즈 퍼블리셔 및 보정 완료 상태 퍼블리셔
        self.pub_target_pose = self.create_publisher(PoseStamped, '/target_pose', 10)
        self.pub_task_cmd = self.create_publisher(String, '/task_command', 10)
        # 고정 볼트(fixed_bolt_coords)의 절대 world 좌표 1회 발행용 (너트 체결 타겟 계산에 사용,
        # 팔의 kinematic 오차와 무관한 값 - execute_isaac.py가 구독).
        self.pub_bolt_anchor = self.create_publisher(PoseStamped, '/vision/bolt_anchor_pose', 10)

        # ----------------------------------------------------------------------
        # 제어 및 오차 보정 파라미터
        # ----------------------------------------------------------------------
        self.TOLERANCE_DEG = 3        # 목표 각도 정밀도: 0.5도 이하
        self.MAX_VALID_PIXEL_ERR = 200  # 이상치 픽셀 오차 스킵 가드

        # camera_bolt intrinsics (최초 1회 수신 시 캐시, 고정 카메라라 갱신 불필요)
        self.fx = self.fy = self.cx = self.cy = None

        # 고정 볼트 world 좌표 산출용 (bolts_detected 이후 depth 샘플을 모아 평균, 노이즈 감소)
        self._bolt_world_samples = []
        self.BOLT_WORLD_AVG_SAMPLES = 30
        self.bolt_anchor_published = False

        # 최신 Depth 프레임 및 상태 변수
        self.current_depth_frame = None
        self.fixed_bolt_coords = []
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

        # 버스바 미검출 시 Hold 처리용 (픽셀 좌표 차이)
        self.last_valid_dx_px = 0       # hx - bx (양수: 구멍이 오른쪽, 음수: 구멍이 왼쪽)
        self.last_valid_dy_px = 0       # hy - by (양수: 구멍이 아래쪽, 음수: 구멍이 위쪽)
        self.last_valid_dtheta = 0.0
        self.has_valid_tracking = False

        # 연속 30 Step 유지 카운터
        self.hold_count = 0

        # 10Hz 오차 보정 제어 루프
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info("🚀 [Vision Node] 실행 완료. 고정 볼트 탐색을 시작합니다 (트리거 대기 중...)")

    def task_command_callback(self, msg: String):
        """외부(Behavior / Isaac Sim)로부터 오차 보정 시작 명령 수신"""
        cmd = msg.data
        if cmd in ["START_ERRORFIX_CORRECTION", "START_VISION_CORRECTION", "MOVE_BATTERY_CENTER_SUCCESS"]:
            self.get_logger().info(f"\n>>> [Vision Correction] 오차 보정 시작 신호 수신 ({cmd})!")
            self.is_active = True
            self.hold_count = 0  # 새로 시작 시 연속 카운터 초기화

    def caminfo_callback(self, msg: CameraInfo):
        if self.fx is None:
            self.fx, self.fy, self.cx, self.cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
            self.get_logger().info(f"✅ camera_bolt intrinsics 수신: fx={self.fx:.2f}, fy={self.fy:.2f}")

    def _update_bolt_anchor_world(self):
        """고정 볼트(fixed_bolt_coords[0])의 depth를 샘플링해 world 좌표를 평균 내고,
        충분히 모이면 /vision/bolt_anchor_pose로 1회 발행한다."""
        if self.bolt_anchor_published or not self.fixed_bolt_coords:
            return
        if self.fx is None or self.current_depth_frame is None:
            return

        bx, by = self.fixed_bolt_coords[0]
        depth_h, depth_w = self.current_depth_frame.shape[:2]
        if not (0 <= by < depth_h and 0 <= bx < depth_w):
            return
        depth_m = float(self.current_depth_frame[by, bx])
        if depth_m <= 0.0:
            return

        world_pos = pixel_depth_to_world(bx, by, depth_m, self.fx, self.fy, self.cx, self.cy)
        self._bolt_world_samples.append(world_pos)

        if len(self._bolt_world_samples) >= self.BOLT_WORLD_AVG_SAMPLES:
            avg_pos = np.mean(self._bolt_world_samples, axis=0)
            msg = PoseStamped()
            msg.header.frame_id = "world"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in avg_pos)
            msg.pose.orientation.w = 1.0
            self.pub_bolt_anchor.publish(msg)
            self.bolt_anchor_published = True
            self.get_logger().info(f"✅ 고정 볼트 world 좌표 확정(평균 {self.BOLT_WORLD_AVG_SAMPLES}프레임) -> {avg_pos}")

    def depth_callback(self, msg):
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.current_depth_frame = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")

    def rgb_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")
            return

        display_img = frame.copy()

        # ----------------------------------------------------------------------
        # [STEP 1] 고정 볼트 검출 (노드가 켜지자마자 백그라운드에서 항상 수행)
        # ----------------------------------------------------------------------
        if not self.bolts_detected:
            self.detect_static_bolts_hough(frame)
            if len(self.fixed_bolt_coords) > 0:
                self.bolts_detected = True
                self.get_logger().info(f"✅ [초기화 완료] 고정 볼트 위치 선점 완료: X={self.fixed_bolt_coords[0][0]}, Y={self.fixed_bolt_coords[0][1]}")
            else:
                cv2.putText(display_img, "Searching Static Bolts...", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if self.bolts_detected and not self.bolt_anchor_published:
            self._update_bolt_anchor_world()

        # ----------------------------------------------------------------------
        # [STEP 2] 트리거 수신 전 대기 모드 (시각화만 출력)
        # ----------------------------------------------------------------------
        if not self.is_active:
            if self.bolts_detected:
                cv2.putText(display_img, "Static Bolt Ready. Waiting for Trigger...", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                for bx, by in self.fixed_bolt_coords:
                    cv2.circle(display_img, (bx, by), 4, (0, 255, 0), -1)

            cv2.imshow("Battery Assembly Vision - RGB Tracking", display_img)
            cv2.waitKey(1)
            return

        # ----------------------------------------------------------------------
        # [STEP 3] 버스바 구멍 추적 및 오차 연산 (트리거 수신 후에만 작동)
        # ----------------------------------------------------------------------
        if self.bolts_detected and self.current_depth_frame is not None:
            self.detect_yellow_busbar_holes_depth(frame, self.current_depth_frame, min_busbar_area=2000)

            # ROI 박스 및 에지 라인 오버레이
            if self.roi_rect is not None:
                rx, ry, rw, rh = self.roi_rect
                cv2.rectangle(display_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 128), 2)

            self.draw_extended_line(display_img, self.battery_line_data, color=(0, 255, 0), label=f"Battery Edge ({self.battery_angle:.1f}deg)")
            self.draw_extended_line(display_img, self.busbar_line_data, color=(0, 255, 255), label=f"Busbar Edge ({self.busbar_angle:.1f}deg)")

            for bx, by in self.fixed_bolt_coords:
                cv2.circle(display_img, (bx, by), 3, (0, 255, 0), -1)

            if len(self.busbar_hole_coords) > 0:
                bx, by = self.fixed_bolt_coords[0]
                hx, hy = self.busbar_hole_coords[0]
                hole_depth = self.busbar_hole_depths[0] if len(self.busbar_hole_depths) > 0 else 0.0

                dx_px = hx - bx  # 영상 기준 X차이 (양수: 구멍이 오른쪽, 음수: 구멍이 왼쪽)
                dy_px = hy - by  # 영상 기준 Y차이 (양수: 구멍이 아래쪽, 음수: 구멍이 위쪽)
                angle_error = self.battery_angle - self.busbar_angle

                # 이상치 픽셀 오차 스킵 가드
                if abs(dx_px) > self.MAX_VALID_PIXEL_ERR or abs(dy_px) > self.MAX_VALID_PIXEL_ERR:
                    self.get_logger().warn(f"⚠️ [비전 노이즈 차단] 비정상적 픽셀 오차 감지! dx:{dx_px}, dy:{dy_px} -> 프레임 스킵")
                    self.has_valid_tracking = False
                else:
                    self.last_valid_dx_px = dx_px
                    self.last_valid_dy_px = dy_px
                    self.last_valid_dtheta = angle_error
                    self.has_valid_tracking = True

                    cv2.circle(display_img, (hx, hy), 5, (0, 0, 255), -1)
                    cv2.line(display_img, (bx, by), (hx, hy), (255, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(display_img, f"E:({dx_px},{dy_px}) Z:{hole_depth:.2f}", (hx + 8, hy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

                    status_line = f"dx:{dx_px:+2d}px, dy:{dy_px:+2d}px, dTheta:{angle_error:+.2f}deg [{self.hold_count}/30]"
                    sys.stdout.write(f"\r\033[K[Tracking Status] {status_line}")
                    sys.stdout.flush()

        cv2.imshow("Battery Assembly Vision - RGB Tracking", display_img)
        cv2.waitKey(1)

    def control_loop(self):
        """픽셀 좌표(hx, hy)와 볼트 좌표(bx, by)가 완벽히 일치할 때까지 보정하는 제어 루프"""
        if not self.is_active or not self.bolts_detected or not self.has_valid_tracking:
            return

        bx, by = self.fixed_bolt_coords[0]
        hx, hy = self.busbar_hole_coords[0] if len(self.busbar_hole_coords) > 0 else (bx, by)
        dtheta = self.last_valid_dtheta

        # 🎯 [수렴 조건] 픽셀 오차가 0 (hx == bx, hy == by) & 각도 0.5도 이하
        if hx == bx and hy == by and abs(dtheta) <= self.TOLERANCE_DEG:
            self.hold_count += 1
            
            # 연속 30 Step(약 3초) 달성 시 완수 신호 퍼블리시 및 종료
            if self.hold_count >= 30:
                self.get_logger().info(f"\n==========================================")
                self.get_logger().info(f" ★ [정밀 정렬 완료] 픽셀 완전 일치(0px 오차) 30 Step 유지 달성")
                self.get_logger().info(f"==========================================")
                
                msg = String()
                msg.data = "ALIGNMENT_SUCCESS"
                self.pub_task_cmd.publish(msg)
                self.is_active = False
                self.hold_count = 0
                return
        else:
            self.hold_count = 0  # 1픽셀이라도 차이 날 경우 연속 카운터 초기화

        FIXED_STEP = 0.0002  # 0.1mm (0.0001m) 고정 스텝

        # --- dx (위/아래) 설정 ---
        if hy < by:          # 구멍이 위에 있음
            step_x = +FIXED_STEP
        elif hy > by:        # 구멍이 아래쪽에 있음
            step_x = -FIXED_STEP
        else:
            step_x = 0.0

        # --- dy (왼쪽/오른쪽) 설정 ---
        if hx < bx:          # 구멍이 오른쪽에 있음 (로봇 관점 dy 설정)
            step_y = +FIXED_STEP
        elif hx > bx:        # 구멍이 왼쪽에 있음
            step_y = -FIXED_STEP
        else:
            step_y = 0.0

        # --- 각도 보정 ---
        if dtheta > 0:
            step_dtheta = -0.01
        elif dtheta < 0:
            step_dtheta = 0.01
        else:
            step_dtheta = 0.0

        # 메시지 작성 및 퍼블리시
        target_msg = PoseStamped()
        target_msg.header.stamp = self.get_clock().now().to_msg()
        target_msg.header.frame_id = "world"
        
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
        x_start, y_start = int(w * 0.45), int(h * 0.25)
        roi_w, roi_h = int(w * 0.40), int(h * 0.30)
        self.roi_rect = (x_start, y_start, roi_w, roi_h)
        
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
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=40, maxLineGap=10)

        best_line = None
        max_length = 0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                angle_deg = math.degrees(math.atan2(dy, dx))

                if -20.0 <= angle_deg <= 20.0 or angle_deg >= 160.0 or angle_deg <= -160.0:
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

        contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        mask_lines = cv2.HoughLinesP(mask_edges, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=10)

        bottommost_line = None
        max_y = -1

        if mask_lines is not None:
            for line in mask_lines:
                x1, y1, x2, y2 = line[0]
                dx, dy = x2 - x1, y2 - y1
                angle_deg = math.degrees(math.atan2(dy, dx))

                if -20.0 <= angle_deg <= 20.0 or angle_deg >= 160.0 or angle_deg <= -160.0:
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
            [vx, vy, cx, cy] = cv2.fitLine(main_busbar_cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            vx_val, vy_val = vx[0], vy[0]
            if vx_val < 0:
                vx_val, vy_val = -vx_val, -vy_val
            self.busbar_angle = math.degrees(math.atan2(vy_val, vx_val))
            self.busbar_line_data = (vx_val, vy_val, cx[0], cy[0])

        depth_h, depth_w = depth_img.shape[:2]
        depth_img_resized = cv2.resize(depth_img, (w, h), interpolation=cv2.INTER_NEAREST) if (depth_h, depth_w) != (h, w) else depth_img

        depth_norm = cv2.normalize(depth_img_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_blur = cv2.GaussianBlur(depth_norm, (5, 5), 0)

        laplacian = cv2.Laplacian(depth_blur, cv2.CV_8U, ksize=3)
        masked_laplacian = cv2.bitwise_and(laplacian, laplacian, mask=filtered_mask)

        circles = cv2.HoughCircles(
            masked_laplacian, cv2.HOUGH_GRADIENT, dp=1.0,
            minDist=10, param1=30, param2=12, minRadius=3, maxRadius=12
        )

        detected_holes = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cx, cy = int(i[0]), int(i[1])
                if filtered_mask[cy, cx] > 0:
                    val = float(depth_img_resized[cy, cx])
                    detected_holes.append((cx, cy, val))

        detected_holes.sort(key=lambda item: item[0])
        self.busbar_hole_coords = [(h[0], h[1]) for h in detected_holes]
        self.busbar_hole_depths = [h[2] for h in detected_holes]

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
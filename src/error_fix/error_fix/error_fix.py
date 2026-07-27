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
# ★ camera_bolt 고정 World Pose (error_fix_depth1.py에서 Busbar.usd 실측으로
# 캘리브레이션한 값 그대로 재사용) - 픽셀+depth를 실제 world 미터 오차로 정확히
# 환산해서 진짜 Kp/Ki PI를 적용하기 위함 (기존엔 depth를 구독만 하고 안 썼음).
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


class IncrementalPI:
    """증분형(velocity-form) PI - 매 사이클 '이번에 추가로 보정할 양(delta)'을 반환한다.
    delta = Ki*error + Kp*(error - 직전 error). 받는 쪽(execute_isaac.py)이 이 delta를
    누적(+=)하는 구조와 맞물려 설계됨 (assembly_nut_fraction1.py의 YawAligner와 동일 패턴)."""

    def __init__(self, kp, ki, out_limit=None):
        self.kp = kp
        self.ki = ki
        self.out_limit = out_limit
        self.prev_error = None

    def reset(self):
        self.prev_error = None

    def step(self, error):
        delta_error = 0.0 if self.prev_error is None else (error - self.prev_error)
        self.prev_error = error
        delta = self.ki * error + self.kp * delta_error
        if self.out_limit is not None:
            delta = max(-self.out_limit, min(self.out_limit, delta))
        return delta


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

        # ----------------------------------------------------------------------
        # 제어 및 오차 보정 파라미터
        # ----------------------------------------------------------------------
        self.TOLERANCE_DEG = 0.5        # 목표 각도 정밀도: 0.5도 이하
        self.MAX_VALID_PIXEL_ERR = 200  # 이상치 픽셀 오차 스킵 가드 (배터리 중점 스캔 오차로 초기 오프셋이 커도 추적 유지되도록 200->400)

        # 최신 Depth 프레임 및 상태 변수
        self.current_depth_frame = None
        self.current_depth_frame_resized = None  # RGB 해상도에 맞춰 정렬된 depth (world 변환용)
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

        # 버스바 미검출 시 Hold 처리용
        self.last_valid_dtheta = 0.0
        self.has_valid_tracking = False

        # ★ 카메라 intrinsics (camera_info 최초 1회 수신 시 캐시, 고정 카메라라 갱신 불필요)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # ★ world 좌표(m) 기준 실측 오차 - pixel_depth_to_world로 계산됨
        self.last_valid_dx_m = 0.0
        self.last_valid_dy_m = 0.0
        self.has_valid_world_error = False

        # ★ yaw 게인을 낮춰도 정상상태 오실레이션이 남는 문제 대응(2026-07-27):
        # dtheta가 매 프레임 battery_angle-busbar_angle을 raw로 다시 계산해서 노이즈가
        # 큰데, Kp*(error-prev_error)항이 이 노이즈를 그대로 증폭시켜 게인을 낮춰도
        # 노이즈 크기만큼은 계속 진동한다. dtheta 자체를 지수이동평균(EMA)으로 스무딩해서
        # PI에 넣기 전에 노이즈를 줄인다.
        self.dtheta_ema = None
        self.ANGLE_EMA_ALPHA = 0.25

        # ★ 진짜 PI 제어기 (world 미터/도 단위 오차에 비례). 위치는 실측 캘리브레이션이
        # 없어 보수적인 값으로 시작 - 너무 느리거나 진동하면 kp/ki를 조정할 것.
        # 각도는 assembly_nut_fraction1.py에서 RMPFlow 스텝응답 실측으로 설계된 값 재사용.
        # ★ out_limit=0.002(2mm)로 클램프 - execute_isaac.py의 FINE_ALIGNMENT가
        # abs(offset.x) <= 0.0025(2.5mm)를 넘으면 그 스텝을 통째로 버리기 때문에,
        # 그보다 작게 유지해야 큰 오차에서도 스텝이 누락되지 않는다.
        # ★ Kp를 더 낮춤(2026-07-27, 노이즈 증폭 완화): Kp*(error-prev_error)항이 픽셀
        # 노이즈를 그대로 증폭시키는 게 확인돼서 0.15->0.075로 절반 인하. Ki(적분/수렴
        # 속도)는 그대로 유지.
        self.pi_x = IncrementalPI(kp=0.075, ki=0.35, out_limit=0.002)   # m/cycle
        self.pi_y = IncrementalPI(kp=0.075, ki=0.35, out_limit=0.002)   # m/cycle
        # ★ yaw 오실레이션 튜닝(2026-07-27): assembly_nut_fraction1.py의 Kp=1.0/Ki=0.0589는
        # RMPFlow 플랜트(τ≈0.283s)만 실측하고 비전 왕복지연 θ=0으로 가정한 λ=τ 값인데,
        # 여기 dtheta는 그 소스(/busbar_alignment_error, 스무딩됨)가 아니라 매 프레임 Hough
        # 라인 재피팅 raw 값이라 노이즈가 훨씬 크고, θ도 더 길 가능성이 높음 - 원래 문서에
        # "θ=0 가정하고 밀어붙이면 진동"이라 경고된 상황이 재현됨. 대역폭을 절반(λ=2τ)으로
        # 낮췄었으나(Kp=0.5) 스무딩 추가 후에도 노이즈 증폭이 남아 Kp만 추가로 절반 인하.
        # Ki(수렴 속도)는 유지.
        self.pi_yaw = IncrementalPI(kp=0.25, ki=0.0295, out_limit=0.5)  # deg/cycle

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
            self.pi_x.reset()
            self.pi_y.reset()
            self.pi_yaw.reset()
            self.dtheta_ema = None

    def caminfo_callback(self, msg):
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info(
                f"✅ camera_bolt intrinsics 수신: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}"
            )

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
                    if self.dtheta_ema is None:
                        self.dtheta_ema = angle_error
                    else:
                        self.dtheta_ema = (self.ANGLE_EMA_ALPHA * angle_error
                                            + (1.0 - self.ANGLE_EMA_ALPHA) * self.dtheta_ema)
                    self.last_valid_dtheta = self.dtheta_ema
                    self.has_valid_tracking = True

                    # ★ 진짜 PI를 쓰려면 실제 world 오차(m)가 필요 - 픽셀+depth를
                    # camera_bolt 고정 캘리브레이션으로 언프로젝션해서 계산한다.
                    self.has_valid_world_error = False
                    if (self.fx is not None and self.current_depth_frame_resized is not None
                            and hole_depth > 0.0):
                        bolt_depth = float(self.current_depth_frame_resized[by, bx])
                        if bolt_depth > 0.0:
                            bolt_world = pixel_depth_to_world(bx, by, bolt_depth, self.fx, self.fy, self.cx, self.cy)
                            hole_world = pixel_depth_to_world(hx, hy, hole_depth, self.fx, self.fy, self.cx, self.cy)
                            world_err = hole_world - bolt_world
                            self.last_valid_dx_m = float(world_err[0])
                            self.last_valid_dy_m = float(world_err[1])
                            self.has_valid_world_error = True

                    cv2.circle(display_img, (hx, hy), 5, (0, 0, 255), -1)
                    cv2.line(display_img, (bx, by), (hx, hy), (255, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(display_img, f"E:({dx_px},{dy_px}) Z:{hole_depth:.2f}", (hx + 8, hy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

                    world_str = (f"world_dx:{self.last_valid_dx_m*1000:+.1f}mm world_dy:{self.last_valid_dy_m*1000:+.1f}mm"
                                 if self.has_valid_world_error else "world:N/A")
                    status_line = f"dx:{dx_px:+2d}px, dy:{dy_px:+2d}px, dTheta:{angle_error:+.2f}deg, {world_str} [{self.hold_count}/30]"
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

        if self.has_valid_world_error:
            # ★ 진짜 PI - world 미터 오차(hole-bolt)에 비례해서 스텝이 자동으로 커지고
            # (오차 클 때) 작아짐(수렴할 때). error = -world_err로 줘야 hole이 bolt에
            # 도달하는 방향으로 delta가 나온다(부호는 execute_isaac.py가 그대로
            # target_fine_pos += delta 하는 것과 맞물려서 검증 필요 - 반대로 움직이면
            # pi_x/pi_y의 error 부호를 반대로 뒤집을 것).
            step_x = self.pi_x.step(-self.last_valid_dx_m)
            step_y = self.pi_y.step(-self.last_valid_dy_m)
        else:
            # depth/camera_info 아직 준비 안 됐을 때 폴백: 기존 고정 스텝(bang-bang)
            FIXED_STEP = 0.0006
            if hy < by:
                step_x = +FIXED_STEP
            elif hy > by:
                step_x = -FIXED_STEP
            else:
                step_x = 0.0
            if hx < bx:
                step_y = +FIXED_STEP
            elif hx > bx:
                step_y = -FIXED_STEP
            else:
                step_y = 0.0

        # --- 각도 보정: assembly_nut_fraction1.py에서 실측 검증된 Kp/Ki 그대로 재사용 ---
        step_dtheta = self.pi_yaw.step(dtheta)

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
        # RGB 픽셀 좌표계와 정렬된 depth (고정 볼트 depth 샘플링에 재사용 - world 오차 계산용)
        self.current_depth_frame_resized = depth_img_resized

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
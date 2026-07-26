#!/usr/bin/env python3
import sys
import math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from scipy.spatial.transform import Rotation as R

# cv_bridge 대체: Isaac Sim 번들 Python(3.11)과 /opt/ros/humble의 cv_bridge_boost.so(3.10용 컴파일)
# 사이 ABI 불일치로 cv_bridge를 import할 수 없어 sensor_msgs/Image -> numpy 변환을 직접 구현.
_ENCODING_TO_DTYPE_CHANNELS = {
    'mono8': (np.uint8, 1),
    '8UC1': (np.uint8, 1),
    'mono16': (np.uint16, 1),
    '16UC1': (np.uint16, 1),
    '32FC1': (np.float32, 1),
    'rgb8': (np.uint8, 3),
    'bgr8': (np.uint8, 3),
    'rgba8': (np.uint8, 4),
    'bgra8': (np.uint8, 4),
}


def imgmsg_to_cv2(msg, desired_encoding='passthrough'):
    dtype, channels = _ENCODING_TO_DTYPE_CHANNELS.get(msg.encoding, (np.uint8, 1))
    arr = np.frombuffer(msg.data, dtype=dtype)
    arr = arr.reshape(msg.height, msg.width) if channels == 1 else arr.reshape(msg.height, msg.width, channels)
    if msg.is_bigendian and dtype != np.uint8:
        arr = arr.byteswap()

    if desired_encoding == 'passthrough' or desired_encoding == msg.encoding:
        return arr
    if desired_encoding == 'bgr8':
        if msg.encoding == 'rgb8':
            return arr[:, :, ::-1].copy()
        if msg.encoding == 'rgba8':
            return arr[:, :, [2, 1, 0]].copy()
        if msg.encoding == 'bgra8':
            return arr[:, :, :3].copy()
    return arr

# ──────────────────────────────────────────────────────────────────────────
# ★ camera_bolt 고정 World Pose (Busbar.usd -> /World/Camera_bolt prim에서 추출) ★
# 카메라가 로봇/AMR에 붙어있지 않고 /World에 고정 마운트된 탑다운(수직 아래) 카메라라서
# 실행 중 갱신할 필요 없이 상수로 고정.
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
    """픽셀 좌표(u,v) + depth(m)를 camera_bolt 고정 pose 기준 world 좌표(m)로 변환.
    depth_m은 광축(Z) 방향 평면 depth(distance_to_image_plane)라고 가정.
    """
    x_opt = (u - cx) * depth_m / fx
    y_opt = (v - cy) * depth_m / fy
    z_opt = depth_m
    p_cam_usd = _OPTICAL_TO_USD_CAM @ np.array([x_opt, y_opt, z_opt])
    return _CAMERA_BOLT_ROT.apply(p_cam_usd) + CAMERA_BOLT_WORLD_POS


class BatteryAssemblyVisionNode(Node):
    def __init__(self):
        super().__init__('battery_assembly_vision_node')
        
        # ----------------------------------------------------------------------
        # ROS 2 Topic Subscriber (독립 콜백)
        # /camera_bolt 기준 RGB와 Depth를 각각 구독
        # ----------------------------------------------------------------------
        self.rgb_sub = self.create_subscription(
            Image, 
            '/camera_bolt/rgb', 
            self.rgb_callback, 
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            '/camera_bolt/depth',
            self.depth_callback,
            10
        )
        self.caminfo_sub = self.create_subscription(
            CameraInfo,
            '/camera_bolt/camera_info',
            self.caminfo_callback,
            10
        )

        # world 좌표 보정값 퍼블리셔 (linear.xyz: world dx,dy,dz[m], angular.z: dTheta[rad])
        self.alignment_pub = self.create_publisher(Twist, '/busbar_alignment_error', 10)

        # 최신 Depth 프레임 저장 변수
        self.current_depth_frame = None
        self.current_depth_frame_resized = None  # RGB 해상도에 맞춰 정렬된 depth (world 변환용)

        # 카메라 intrinsics (camera_info 최초 1회 수신 시 캐시, 고정 카메라라 갱신 불필요)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # 비전 처리용 상태 변수
        self.fixed_bolt_coords = []
        self.busbar_hole_coords = []
        self.busbar_hole_depths = [] # 구멍 위치의 실제 Depth 값 (거리)
        self.battery_angle = 0.0
        self.busbar_angle = 0.0
        
        self.battery_line_data = None
        self.busbar_line_data = None
        self.roi_rect = None

        # 상태 제어 플래그
        self.bolts_detected = False

        # 고정 볼트 검출 안정화(디바운스): Play 직후 렌더링/조명이 안정화되기 전
        # 첫 프레임에서 배터리 셀 격자 등 엉뚱한 원이 스퓨리어스로 잡혀 영구 고정되는
        # 것을 방지하기 위해, 같은 위치가 연속 N프레임 유지될 때만 확정한다.
        self._bolt_candidate_history = []
        self.BOLT_STABILITY_FRAMES = 5
        self.BOLT_STABILITY_TOL_PX = 3

        # 버스바 미검출 시 이전 값 유지(Hold)용 변수 및 카운터
        self.last_valid_dx = 0
        self.last_valid_dy = 0
        self.last_valid_dtheta = 0.0
        self.busbar_missing_count = 0
        self.MAX_MISSING_FRAMES = 5

        # world 좌표 보정값 (m, rad) - 픽셀 hold 값과 별도로 유지
        self.last_valid_world_dx = 0.0
        self.last_valid_world_dy = 0.0
        self.last_valid_world_dz = 0.0

        self.get_logger().info("🚀 Depth 시각화 포함 Battery Assembly Vision Node가 시작되었습니다...")

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
            # Depth 이미지 수신 및 float-nan 예외 처리
            depth_img = imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.current_depth_frame = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")

    def rgb_callback(self, msg):
        try:
            frame = imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert RGB image: {e}")
            return

        display_img = frame.copy()
        h, w = frame.shape[:2]

        # ----------------------------------------------------------------------
        # [STEP 1] 최초 볼트 미검출 상태일 때: 고정 볼트 탐색 및 위치 고정
        # ----------------------------------------------------------------------
        if not self.bolts_detected:
            self.detect_static_bolts_hough(frame)
            
            if len(self.fixed_bolt_coords) > 0:
                self.bolts_detected = True
                print()
                self.get_logger().info(f"✅ 고정 볼트 검출 완료! 선택된 고정 볼트: X={self.fixed_bolt_coords[0][0]}, Y={self.fixed_bolt_coords[0][1]}")
            else:
                cv2.putText(display_img, "Searching Static Bolts...", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ----------------------------------------------------------------------
        # [STEP 2] Depth 및 RGB 기반 버스바 구멍 추적 및 오차 계산
        # ----------------------------------------------------------------------
        if self.bolts_detected:
            if self.current_depth_frame is not None:
                self.detect_yellow_busbar_holes_depth(frame, self.current_depth_frame, min_busbar_area=2000)
            else:
                cv2.putText(display_img, "Waiting for Depth Image...", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            # 1. ROI 영역 표시 (연두색)
            if self.roi_rect is not None:
                rx, ry, rw, rh = self.roi_rect
                cv2.rectangle(display_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 128), 2)

            # 2. 수평 기준선 시각화
            self.draw_extended_line(
                display_img, self.battery_line_data, 
                color=(0, 255, 0), thickness=2, label=f"Battery Edge ({self.battery_angle:.1f}deg)"
            )
            self.draw_extended_line(
                display_img, self.busbar_line_data, 
                color=(0, 255, 255), thickness=2, label=f"Busbar Edge ({self.busbar_angle:.1f}deg)"
            )

            # 3. 고정 볼트 표시 (초록색 원)
            for bx, by in self.fixed_bolt_coords:
                cv2.circle(display_img, (bx, by), 3, (0, 255, 0), -1)

            # 4. Depth로 검출된 버스바 구멍 표시 및 X, Y 오차, Depth 값 출력
            if len(self.busbar_hole_coords) > 0:
                self.busbar_missing_count = 0
                
                num_pairs = min(len(self.fixed_bolt_coords), len(self.busbar_hole_coords))
                log_parts = []
                for i in range(num_pairs):
                    bx, by = self.fixed_bolt_coords[i]
                    hx, hy = self.busbar_hole_coords[i]
                    hole_depth = self.busbar_hole_depths[i] if i < len(self.busbar_hole_depths) else 0.0
                    
                    dx = hx - bx
                    dy = hy - by
                    
                    self.last_valid_dx = dx
                    self.last_valid_dy = dy
                    
                    cv2.circle(display_img, (hx, hy), 5, (0, 0, 255), -1)
                    cv2.line(display_img, (bx, by), (hx, hy), (255, 0, 255), 1, cv2.LINE_AA)
                    
                    # 텍스트로 오차 및 Depth 값 표기
                    cv2.putText(display_img, f"E{i}:({dx},{dy}) Z:{hole_depth:.2f}", (hx + 8, hy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

                    log_parts.append(f"Hole[{i}]->Bolt[{i}] dx:{dx:+3d}px, dy:{dy:+3d}px, Z:{hole_depth:.2f}")

                    # ── world 좌표 보정값 계산 (intrinsics + depth 확보된 첫 pair만 사용) ──
                    if i == 0 and self.fx is not None and self.current_depth_frame_resized is not None and hole_depth > 0.0:
                        bolt_depth = float(self.current_depth_frame_resized[by, bx])
                        if bolt_depth > 0.0:
                            bolt_world = pixel_depth_to_world(bx, by, bolt_depth, self.fx, self.fy, self.cx, self.cy)
                            hole_world = pixel_depth_to_world(hx, hy, hole_depth, self.fx, self.fy, self.cx, self.cy)
                            world_err = hole_world - bolt_world

                            self.last_valid_world_dx = float(world_err[0])
                            self.last_valid_world_dy = float(world_err[1])
                            self.last_valid_world_dz = float(world_err[2])

                            log_parts.append(
                                f"World dx:{world_err[0]*1000:+.1f}mm dy:{world_err[1]*1000:+.1f}mm dz:{world_err[2]*1000:+.1f}mm"
                            )

                angle_error = self.battery_angle - self.busbar_angle
                self.last_valid_dtheta = angle_error
                log_parts.append(f"dTheta:{angle_error:+.2f}deg")

                if self.fx is not None:
                    twist = Twist()
                    twist.linear.x = self.last_valid_world_dx
                    twist.linear.y = self.last_valid_world_dy
                    twist.linear.z = self.last_valid_world_dz
                    twist.angular.z = math.radians(angle_error)
                    self.alignment_pub.publish(twist)

                status_line = " | ".join(log_parts)
                sys.stdout.write(f"\r\033[K[Depth Tracking Status] {status_line}")
                sys.stdout.flush()

                cv2.putText(display_img, f"dTheta: {angle_error:+.2f} deg", 
                            (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                self.busbar_missing_count += 1
                if self.busbar_missing_count <= self.MAX_MISSING_FRAMES:
                    status_line = f"Hole[0]->Bolt[0] dx:{self.last_valid_dx:+3d}px, dy:{self.last_valid_dy:+3d}px | dTheta:{self.last_valid_dtheta:+.2f}deg (Hold)"
                    sys.stdout.write(f"\r\033[K[Depth Tracking Status] {status_line}")
                else:
                    sys.stdout.write("\r\033[K[Depth Tracking Status] Busbar Not Detected")
                sys.stdout.flush()
                
                cv2.putText(display_img, "Busbar Not Detected", (20, h - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # OpenCV GUI (시스템 python3 + /opt/ros/humble 환경에서만 실행 가능: GUI 포함 opencv 필요.
        # isaac_python은 opencv-python-headless라 창을 못 띄움)
        cv2.imshow("Battery Assembly Vision - RGB Tracking", display_img)
        cv2.waitKey(1)

    # ==========================================================================
    # 비전 알고리즘 함수
    # ==========================================================================
    def detect_static_bolts_hough(self, frame):
        h, w = frame.shape[:2]
        x_start, y_start = int(w * 0.45), int(h * 0.25)
        roi_w, roi_h = int(w * 0.40), int(h * 0.30)
        self.roi_rect = (x_start, y_start, roi_w, roi_h)
        
        roi = frame[y_start:y_start + roi_h, x_start:x_start + roi_w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        
        _, thresh = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blurred_bolt = cv2.GaussianBlur(thresh, (5, 5), 0)

        circles = cv2.HoughCircles(
            blurred_bolt, cv2.HOUGH_GRADIENT, dp=1.0, 
            minDist=15, 
            param1=50, 
            param2=18, 
            minRadius=4, 
            maxRadius=12
        )
        
        candidate = None
        if circles is not None:
            circles = np.uint16(np.around(circles))
            detected_candidates = []
            for i in circles[0, :]:
                detected_candidates.append((int(i[0]) + x_start, int(i[1]) + y_start))

            detected_candidates.sort(key=lambda c: c[0])
            candidate = detected_candidates[0]

        # 디바운스: 연속 N프레임 동안 같은 위치(허용오차 내)가 나올 때만 확정
        self.fixed_bolt_coords = []
        if candidate is not None:
            if self._bolt_candidate_history:
                px, py = self._bolt_candidate_history[-1]
                if abs(candidate[0] - px) > self.BOLT_STABILITY_TOL_PX or abs(candidate[1] - py) > self.BOLT_STABILITY_TOL_PX:
                    self._bolt_candidate_history = []  # 위치가 튀면 스트릭 초기화
            self._bolt_candidate_history.append(candidate)
            if len(self._bolt_candidate_history) >= self.BOLT_STABILITY_FRAMES:
                self.fixed_bolt_coords = [candidate]
        else:
            self._bolt_candidate_history = []

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
        
        # 1. 버스바 마스크 추출 (HSV)
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

        cv2.imshow("Busbar Yellow Mask", filtered_mask)

        # 2. 버스바 각도 계산 (HoughLinesP)
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

        # ----------------------------------------------------------------------
        # 3. Depth 이미지 처리 및 Visualized Map 생성
        # ----------------------------------------------------------------------
        depth_h, depth_w = depth_img.shape[:2]
        if (depth_h, depth_w) != (h, w):
            depth_img_resized = cv2.resize(depth_img, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            depth_img_resized = depth_img

        # RGB 픽셀 좌표계와 정렬된 depth 프레임 (고정 볼트 depth 샘플링에 재사용)
        self.current_depth_frame_resized = depth_img_resized

        # Depth 이미지 정규화 및 가우시안 블러
        depth_norm = cv2.normalize(depth_img_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_blur = cv2.GaussianBlur(depth_norm, (5, 5), 0)

        # 🎨 Depth 시각화용 Color Map 생성 (창에 깊이를 컬러로 표시)
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        # 라플라시안 필터로 깊이가 급격히 변하는 구멍 경계 추출
        laplacian = cv2.Laplacian(depth_blur, cv2.CV_8U, ksize=3)
        masked_laplacian = cv2.bitwise_and(laplacian, laplacian, mask=filtered_mask)

        cv2.imshow("Debug - Depth Laplacian Edge", masked_laplacian)

        # 원 검출
        circles = cv2.HoughCircles(
            masked_laplacian, cv2.HOUGH_GRADIENT, dp=1.0,
            minDist=10,
            param1=30,
            param2=12,
            minRadius=3,
            maxRadius=12
        )

        detected_holes = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            mask_h, mask_w = filtered_mask.shape[:2]
            for i in circles[0, :]:
                cx, cy = int(i[0]), int(i[1])
                if not (0 <= cx < mask_w and 0 <= cy < mask_h):
                    continue  # HoughCircles가 이미지 경계(예: cx==w)를 반환하는 경우 방지
                if filtered_mask[cy, cx] > 0:
                    # 해당 좌표의 실제 Depth 수치(거리) 추출
                    val = float(depth_img_resized[cy, cx])
                    detected_holes.append((cx, cy, val))

        # X좌표 기준 정렬
        detected_holes.sort(key=lambda item: item[0])

        self.busbar_hole_coords = [(h[0], h[1]) for h in detected_holes]
        self.busbar_hole_depths = [h[2] for h in detected_holes]

        # 🎨 Depth Color Map 위에 검출된 원 위치 overlay 시각화
        for cx, cy, val in detected_holes:
            cv2.circle(depth_colormap, (cx, cy), 6, (255, 255, 255), 2)
            cv2.putText(depth_colormap, f"{val:.2f}", (cx + 8, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Debug - Depth Map Visualized", depth_colormap)

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
        print()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
#!/usr bend/env python3
import sys
import math
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class BatteryAssemblyVisionNode(Node):
    def __init__(self):
        super().__init__('battery_assembly_vision_node')
        
        # ROS 2 Topic & Bridge 설정
        self.subscription = self.create_subscription(
            Image,
            '/camera_bolt/rgb',  # rosbag 내의 RGB 이미지 토픽명
            self.image_callback,
            10
        )
        self.bridge = CvBridge()

        # 비전 처리용 상태 변수
        self.fixed_bolt_coords = []
        self.busbar_hole_coords = []
        self.battery_angle = 0.0
        self.busbar_angle = 0.0
        
        self.battery_line_data = None
        self.busbar_line_data = None
        self.roi_rect = None

        # 상태 제어 플래그
        self.bolts_detected = False

        # 버스바 미검출 시 이전 값 유지(Hold)용 변수 및 카운터
        self.last_valid_dx = 0
        self.last_valid_dy = 0
        self.last_valid_dtheta = 0.0
        self.busbar_missing_count = 0
        self.MAX_MISSING_FRAMES = 5  # 최대 5프레임 유효성 유지

        # 구멍 위치 예측(캘리브레이션)용 변수
        # 버스바는 강체이므로, 구멍은 busbar_line_data(테두리 직선) 기준
        # 항상 같은 상대 위치(라인 방향 성분 along, 법선 방향 성분 perp)에 있다.
        # 처음 몇 번의 "신뢰할 수 있는" 실측값을 평균 내서 이 상대 위치를 고정하고,
        # 이후에는 실측이 튀거나 없어도 이 고정 오프셋으로 예측 위치를 표시한다.
        self.hole_offset_along = None
        self.hole_offset_perp = None
        self.hole_is_predicted = False   # 이번 프레임 구멍 좌표가 실측인지 예측인지
        self._hole_calib_samples = []
        self.HOLE_CALIB_SAMPLES_NEEDED = 8      # 이 개수만큼 모이면 오프셋 확정
        self.MAX_PREDICT_DEVIATION_PX = 20      # 예측 위치에서 이보다 멀리 튄 실측은 노이즈로 버림

        self.get_logger().info("🚀 Battery Assembly Vision Node가 시작되었습니다. 볼트를 탐색합니다...")

    def image_callback(self, msg):
        try:
            # ROS Image -> OpenCV BGR Image 변환
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
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
                print()  # 터미널 줄바꿈
                self.get_logger().info(f"✅ 고정 볼트 검출 완료! 선택된 고정 볼트: X={self.fixed_bolt_coords[0][0]}, Y={self.fixed_bolt_coords[0][1]}")
            else:
                cv2.putText(display_img, "Searching Static Bolts...", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ----------------------------------------------------------------------
        # [STEP 2] 볼트 위치 고정 완료 후: 실시간 버스바 구멍 추적 및 오차 계산
        # ----------------------------------------------------------------------
        if self.bolts_detected:
            self.detect_yellow_busbar_holes(frame, min_busbar_area=2000)

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

            # 3. 고정 볼트 표시 (초록색 원, 반지름 3px)
            for bx, by in self.fixed_bolt_coords:
                cv2.circle(display_img, (bx, by), 3, (0, 255, 0), -1)

            # 4. 버스바 구멍 표시 및 X, Y 오차 계산 (빨간색 원)
            if len(self.busbar_hole_coords) > 0:
                self.busbar_missing_count = 0  # 카운터 리셋
                
                hole_color = (0, 165, 255) if self.hole_is_predicted else (0, 0, 255)  # 예측=주황, 실측=빨강
                for hx, hy in self.busbar_hole_coords:
                    cv2.circle(display_img, (hx, hy), 5, hole_color, -1)
                    if self.hole_is_predicted:
                        cv2.putText(display_img, "(pred)", (hx + 8, hy + 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, hole_color, 1)

                # 🎯 오차(Error) 계산
                num_pairs = min(len(self.fixed_bolt_coords), len(self.busbar_hole_coords))
                
                log_parts = []
                for i in range(num_pairs):
                    bx, by = self.fixed_bolt_coords[i]
                    hx, hy = self.busbar_hole_coords[i]
                    
                    dx = hx - bx  # X축 오차 (px)
                    dy = hy - by  # Y축 오차 (px)
                    
                    self.last_valid_dx = dx
                    self.last_valid_dy = dy
                    
                    log_parts.append(f"Hole[{i}]->Bolt[{i}] dx:{dx:+3d}px, dy:{dy:+3d}px")
                    
                    # 시각화: 볼트와 구멍을 잇는 오차 선 표시
                    cv2.line(display_img, (bx, by), (hx, hy), (255, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(display_img, f"E{i}:({dx},{dy})", (hx + 8, hy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

                angle_error = self.battery_angle - self.busbar_angle
                self.last_valid_dtheta = angle_error
                log_parts.append(f"dTheta:{angle_error:+.2f}deg")
                
                # 터미널 한 줄 실시간 출력
                status_line = " | ".join(log_parts)
                sys.stdout.write(f"\r\033[K[Tracking Status] {status_line}")
                sys.stdout.flush()

                # GUI 오차 표시
                cv2.putText(display_img, f"dTheta: {angle_error:+.2f} deg", 
                            (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # 감지 실패 시 Hold 처리
                self.busbar_missing_count += 1
                if self.busbar_missing_count <= self.MAX_MISSING_FRAMES:
                    status_line = f"Hole[0]->Bolt[0] dx:{self.last_valid_dx:+3d}px, dy:{self.last_valid_dy:+3d}px | dTheta:{self.last_valid_dtheta:+.2f}deg (Hold)"
                    sys.stdout.write(f"\r\033[K[Tracking Status] {status_line}")
                else:
                    sys.stdout.write("\r\033[K[Tracking Status] Busbar Not Detected")
                sys.stdout.flush()
                
                cv2.putText(display_img, "Busbar Not Detected", (20, h - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ----------------------------------------------------------------------
        # [STEP 3] OpenCV GUI 출력
        # ----------------------------------------------------------------------
        cv2.imshow("Battery Assembly Vision - ROS2 Tracking", display_img)
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
        
        self.fixed_bolt_coords = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            detected_candidates = []
            for i in circles[0, :]:
                detected_candidates.append((int(i[0]) + x_start, int(i[1]) + y_start))
            
            detected_candidates.sort(key=lambda c: c[0])
            self.fixed_bolt_coords = [detected_candidates[0]]

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

    def detect_yellow_busbar_holes(self, frame, min_busbar_area=2000):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_yellow = np.array([15, 40, 60])
        upper_yellow = np.array([35, 255, 255])

        raw_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, open_kernel)

        # 반사광으로 마스크가 끊기는 것을 메우기 위해 넓은 커널로 CLOSE
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, close_kernel)

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

        # 🖥️ 노란색 마스크 윈도우 창 표시
        cv2.imshow("Busbar Yellow Mask", filtered_mask)

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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.0,
            minDist=10,
            param1=50,
            param2=20,
            minRadius=5,
            maxRadius=12
        )

        raw_hole_candidates = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cx, cy = int(i[0]), int(i[1])

                if filtered_mask[cy, cx] > 0:
                    raw_hole_candidates.append((cx, cy))

        raw_hole_candidates.sort(key=lambda c: c[0])

        # ------------------------------------------------------------------
        # 구멍 위치 캘리브레이션 & 예측
        # 버스바는 강체라서, 구멍은 busbar_line_data(테두리 직선) 기준
        # 항상 같은 상대 위치(along: 라인 방향 성분, perp: 법선 방향 성분)에 있다.
        # 원시 검출(raw_hole_candidates)은 반사/노이즈로 프레임마다 튈 수 있으므로,
        # 이 상대 위치를 몇 프레임 평균으로 고정해두고, 이후에는 실측이 예측과
        # 가까울 때만 신뢰하고, 없거나 튀면 예측 위치를 그대로 사용한다.
        # ------------------------------------------------------------------
        self.busbar_hole_coords = []

        if self.busbar_line_data is not None:
            vx, vy, lcx, lcy = self.busbar_line_data
            nx, ny = -vy, vx  # 라인의 법선 벡터

            predicted = None
            if self.hole_offset_along is not None:
                predicted = (
                    lcx + self.hole_offset_along * vx + self.hole_offset_perp * nx,
                    lcy + self.hole_offset_along * vy + self.hole_offset_perp * ny,
                )

            best_candidate = None
            if raw_hole_candidates:
                if predicted is not None:
                    # 예측 위치와 가장 가깝고, 허용 오차 이내인 실측만 채택
                    best_candidate = min(
                        raw_hole_candidates,
                        key=lambda c: math.hypot(c[0] - predicted[0], c[1] - predicted[1])
                    )
                    if math.hypot(best_candidate[0] - predicted[0], best_candidate[1] - predicted[1]) > self.MAX_PREDICT_DEVIATION_PX:
                        best_candidate = None  # 너무 멀리 튐 -> 노이즈로 간주
                else:
                    # 아직 캘리브레이션 전: 첫 후보로 캘리브레이션 시작
                    best_candidate = raw_hole_candidates[0]

            if best_candidate is not None:
                rel_x, rel_y = best_candidate[0] - lcx, best_candidate[1] - lcy
                along = rel_x * vx + rel_y * vy
                perp = rel_x * nx + rel_y * ny

                if self.hole_offset_along is None:
                    self._hole_calib_samples.append((along, perp))
                    if len(self._hole_calib_samples) >= self.HOLE_CALIB_SAMPLES_NEEDED:
                        # 평균 대신 중앙값: 캘리브레이션 도중 섞여든 노이즈 표본에 덜 흔들린다
                        self.hole_offset_along = float(np.median([s[0] for s in self._hole_calib_samples]))
                        self.hole_offset_perp = float(np.median([s[1] for s in self._hole_calib_samples]))
                        self.get_logger().info(
                            f"✅ 버스바 구멍 상대 위치 캘리브레이션 완료 (along={self.hole_offset_along:.1f}px, perp={self.hole_offset_perp:.1f}px)"
                        )
                else:
                    # 살짝씩 갱신(느린 이동평균)해서 드리프트에 대응
                    self.hole_offset_along = 0.95 * self.hole_offset_along + 0.05 * along
                    self.hole_offset_perp = 0.95 * self.hole_offset_perp + 0.05 * perp

                self.busbar_hole_coords = [best_candidate]
                self.hole_is_predicted = False
            elif predicted is not None:
                # 실측이 없거나 노이즈뿐이면, 캘리브레이션된 예측 위치를 그대로 사용
                self.busbar_hole_coords = [(int(predicted[0]), int(predicted[1]))]
                self.hole_is_predicted = True
        else:
            # 버스바 라인 자체를 못 잡은 경우엔 예측도 불가능하므로 원시 검출만 사용
            self.busbar_hole_coords = raw_hole_candidates
            self.hole_is_predicted = False

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
        print()  # 커서 정리
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
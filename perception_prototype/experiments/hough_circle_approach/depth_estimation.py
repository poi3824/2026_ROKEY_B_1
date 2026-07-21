"""픽셀 좌표(u, v) + depth 값을 3D 월드 좌표로 변환.

TODO: 아래 값들은 실제 정보가 없어서 임시로 잡은 가정값. 확인되면 교체 필요.
- D455_HFOV_DEG / D455_VFOV_DEG: 실제 Isaac Sim 카메라 설정이 아니라 D455 스펙값
- IMAGE_WIDTH / IMAGE_HEIGHT: 지금 테스트 중인 "배터리 모듈팩.png" 해상도 기준
- CAMERA_WORLD_POSITION: AMR 마스트의 실제 장착 위치/자세 (회전 없이 정확히
  아래를 본다고 가정한 자리표시자)
- DUMMY_DEPTH_M: 실제 depth 이미지가 없어서 쓰는 상수 (스펙상 카메라-단자
  수직거리 0.64m)
"""
import math

import cv2
import numpy as np

D455_HFOV_DEG = 87.0
D455_VFOV_DEG = 62.0
IMAGE_WIDTH = 1975
IMAGE_HEIGHT = 1123

FX = IMAGE_WIDTH / (2 * math.tan(math.radians(D455_HFOV_DEG / 2)))
FY = IMAGE_HEIGHT / (2 * math.tan(math.radians(D455_VFOV_DEG / 2)))
CX = IMAGE_WIDTH / 2
CY = IMAGE_HEIGHT / 2

CAMERA_WORLD_POSITION = (0.0, 0.0, 1.0)  # (x, y, z), 회전 없음(정면 하방) 가정
DUMMY_DEPTH_M = 0.64


def pixel_to_camera(u: int, v: int, depth: float) -> tuple[float, float, float]:
    """픽셀 좌표 + depth -> 카메라 좌표계 3D 좌표 (핀홀 카메라 역투영)."""
    z = depth
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    return x, y, z


def camera_to_world(x_cam: float, y_cam: float, z_cam: float) -> tuple[float, float, float]:
    """카메라 좌표 -> 월드 좌표. 카메라가 회전 없이 정확히 아래를 본다고 가정."""
    cam_x, cam_y, cam_z = CAMERA_WORLD_POSITION
    return cam_x + x_cam, cam_y + y_cam, cam_z - z_cam


def pixel_to_world(u: int, v: int, depth: float) -> tuple[float, float, float]:
    """픽셀 좌표 + depth -> 월드 좌표. 이 파일의 메인 진입점."""
    x_cam, y_cam, z_cam = pixel_to_camera(u, v, depth)
    return camera_to_world(x_cam, y_cam, z_cam)


def pixel_to_world_from_depth_image(
    u: int, v: int, depth_image: np.ndarray
) -> tuple[float, float, float]:
    """depth 이미지에서 (u, v) 위치의 depth 값을 조회해서 3D 월드 좌표로 변환."""
    depth = float(depth_image[v, u])
    return pixel_to_world(u, v, depth)


def main():
    # 단자 홀 검출 로직은 이 파일의 자체 테스트에서만 필요해서 여기서만 import
    from pose_estimation import DEFAULT_IMG_PATH, detect_terminal_holes, pair_terminal_holes

    img = cv2.imdecode(np.fromfile(DEFAULT_IMG_PATH, dtype=np.uint8), cv2.IMREAD_COLOR)
    holes = detect_terminal_holes(img)
    pairs = pair_terminal_holes(img, holes)

    # 실제 depth 이미지가 아직 없어서, 전부 DUMMY_DEPTH_M으로 채운 가짜
    # depth "이미지"(2차원 배열)를 만들어서 조회 로직까지 검증.
    dummy_depth_image = np.full(img.shape[:2], DUMMY_DEPTH_M, dtype=np.float32)

    print(f"paired {len(pairs)} sets, dummy depth image = {DUMMY_DEPTH_M} m (전체 고정)")
    for p in pairs:
        u, v = p["mid"]
        u, v = int(round(u)), int(round(v))
        wx, wy, wz = pixel_to_world_from_depth_image(u, v, dummy_depth_image)
        print(
            f"mid_pixel=({u},{v}) angle_deg={p['angle_deg']:.1f} "
            f"-> world=({wx:.3f}, {wy:.3f}, {wz:.3f})"
        )


if __name__ == "__main__":
    main()

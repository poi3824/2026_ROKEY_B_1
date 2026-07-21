"""팔 끝 D455 카메라 전용 좌표 계산 (픽셀 -> 카메라 3D -> 로봇 베이스 3D).

이 로봇은 스캔(단자 위치 확인)이랑 버스바/볼트 집기를 전부 팔 끝에 붙은
D455 카메라 하나로 처리함 (별도 고정 스캔 카메라 없음). 카메라가 팔을 따라
계속 움직이므로, 이 파일의 계산 로직은 어떤 검출 방식(YOLO든 다른 방식이든)을
쓰든 공통으로 재사용 가능하도록 분리해둠.

TODO:
- D455_HFOV_DEG/VFOV_DEG/IMAGE_WIDTH/IMAGE_HEIGHT: 이 카메라의 실제 Isaac
  Sim 설정을 몰라서 D455 스펙 기준으로 추정한 임시값.
- world_to_base(): 로봇 팔 관절 각도(FK, 또는 tf2)가 있어야 계산 가능한데
  아직 그 정보가 없어서 미구현 상태. 스캔하는 순간마다 팔 위치가 달라지므로
  고정값으로 대체할 수 없음 (CAMERA_TO_GRIPPER_OFFSET만 고정값으로 확보됨).
"""
import math

D455_HFOV_DEG = 87.0
D455_VFOV_DEG = 62.0
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

FX = IMAGE_WIDTH / (2 * math.tan(math.radians(D455_HFOV_DEG / 2)))
FY = IMAGE_HEIGHT / (2 * math.tan(math.radians(D455_VFOV_DEG / 2)))
CX = IMAGE_WIDTH / 2
CY = IMAGE_HEIGHT / 2

DUMMY_DEPTH_M = 0.5

# 그리퍼(angle_bracket) 기준 카메라 고정 오프셋
# (Isaac Sim의 Collected_m0609_camera/m0609_camera.usd에서 확인, 2026-07-21)
CAMERA_TO_GRIPPER_OFFSET = {
    "translate_m": (0.0, 0.045, 0.05),
    "rpy_deg": (-90.0, 0.0, -90.0),
}


def pixel_to_camera(u: float, v: float, depth: float) -> tuple[float, float, float]:
    """픽셀 좌표 + depth -> 카메라 좌표계 3D 좌표 (핀홀 카메라 역투영)."""
    z = depth
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    return x, y, z


def world_to_base(x_cam: float, y_cam: float, z_cam: float) -> tuple[float, float, float]:
    """카메라 기준 좌표 -> 로봇 베이스 기준 좌표.

    로봇 팔이 움직이는 동안 카메라 위치도 계속 바뀌므로, 이 순간의 관절
    각도(FK)와 CAMERA_TO_GRIPPER_OFFSET을 함께 써야 계산 가능함.
    관절 각도를 받아올 방법이 아직 정해지지 않아 미구현 상태.
    """
    raise NotImplementedError(
        "로봇 팔 관절 각도(FK) 정보가 아직 없어 미구현. "
        "CAMERA_TO_GRIPPER_OFFSET 값은 이미 확보되어 있음."
    )

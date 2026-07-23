"""디버그용 검출 오버레이 그리기.

rqt_image_view 등으로 확인할 수 있도록, 검출된 물체의 keypoints(cuboid 9점)/bbox와
좌표 변환 전(카메라 프레임)/후(world 프레임) 값을 이미지 위에 그린다. 물체가 이미지 가장자리에
있어도 텍스트 라벨 전체가 항상 보이도록, 라벨을 그릴 위치를 이미지 경계 안쪽으로
clamp한다 (draw_label 참고).
"""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.42
FONT_THICKNESS = 1
LINE_PADDING = 4
BOX_COLOR = (0, 255, 255)
CENTROID_COLOR = (0, 0, 255)
KEYPOINT_COLOR = (0, 200, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 0, 0)


def draw_label(image: np.ndarray, lines: list[str], anchor_xy: tuple[float, float]) -> None:
    """lines를 anchor_xy 근처에 그리되, 텍스트 블록 전체가 이미지 안에 들어오도록
    top-left 좌표를 이미지 경계 안쪽으로 clamp한다."""
    h_img, w_img = image.shape[:2]

    line_sizes = [cv2.getTextSize(line, FONT, FONT_SCALE, FONT_THICKNESS)[0] for line in lines]
    line_height = max(sz[1] for sz in line_sizes) + LINE_PADDING * 2
    block_w = max(sz[0] for sz in line_sizes) + LINE_PADDING * 2
    block_h = line_height * len(lines)

    x = int(np.clip(anchor_xy[0], 0, max(0, w_img - block_w)))
    y = int(np.clip(anchor_xy[1], 0, max(0, h_img - block_h)))

    cv2.rectangle(image, (x, y), (x + block_w, y + block_h), TEXT_BG_COLOR, -1)
    for i, line in enumerate(lines):
        baseline_y = y + (i + 1) * line_height - LINE_PADDING
        cv2.putText(image, line, (x + LINE_PADDING, baseline_y),
                    FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)


def _format_point(point, status: str, label: str) -> str:
    if point is not None:
        x, y, z = point
        return f'{label} ({x:.2f},{y:.2f},{z:.2f})'
    return f'{label} {status or "n/a"}'


def draw_detection(image: np.ndarray, det: dict, camera_point, world_point, status: str) -> None:
    """det(YoloPoseDetector.detect()의 항목 하나) + 좌표 변환 결과를 image에 그린다.

    camera_point / world_point: (x, y, z) 튜플 또는 None (해당 단계 실패 시).
    status: world_point가 None일 때의 실패 사유 (예: "no depth", "tf fail").
    """
    keypoints_px = det['keypoints_px']
    for i, (kx, ky) in enumerate(keypoints_px):
        # index 0 = Center(파지점, 빨간색 강조), 나머지 8개 = cuboid corner(초록 점).
        color = CENTROID_COLOR if i == 0 else KEYPOINT_COLOR
        cv2.circle(image, (int(round(kx)), int(round(ky))), 3, color, -1)

    x0, y0, x1, y1 = (int(round(v)) for v in det['bbox_px'])
    cv2.rectangle(image, (x0, y0), (x1, y1), BOX_COLOR, 1)

    u, v = det['pixel']  # Center 키포인트, 위에서 이미 강조색으로 그려짐

    lines = [
        f'{det["label"]} {det["score"]:.2f}',
        f'px ({u:.0f},{v:.0f})',
        _format_point(camera_point, status, 'cam'),
        _format_point(world_point, status, 'world'),
    ]
    draw_label(image, lines, (u + 8, v + 8))

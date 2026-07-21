"""YOLO 세그멘테이션 모델로 물체(버스바, 볼트/너트 등)를 검출해서
마스크 중심 좌표를 계산하고, 카메라 기준 3D 좌표까지 변환.

카메라 관련 계산(intrinsics, 카메라->베이스 변환)은 camera_geometry.py에
있음 -> 이 파일은 "YOLO로 뭐가 어디 있는지 찾기"만 담당.

TODO:
- MODEL_PATH: 아직 학습된 모델(.pt)이 없어서 자리만 비워둠. 학습 끝나면
  실제 .pt 파일 경로로 교체.
- 이 파일은 이 컴퓨터에 인터넷이 안 되어 사전학습 모델도 다운로드 못 해서
  실제 실행 검증을 하지 못했음 (문법만 확인함). 모델 파일이 생기면
  반드시 한 번 실행해서 확인 필요.
- 마스크 중심은 박스 중심이 아니라 "실제 채워진 모양의 무게중심"으로 계산함
  (버스바가 ㄹ자/Z자라 비대칭이라, 박스 중심을 쓰면 빈 공간이 중심으로
  잡힐 수 있어서).
- depth 값: 아직 실제 depth 이미지가 없어서 DUMMY_DEPTH_M 고정값 사용.
"""
import cv2
import numpy as np
from ultralytics import YOLO

from camera_geometry import DUMMY_DEPTH_M, pixel_to_camera

MODEL_PATH = "yolov8n-seg.pt"  # TODO: 학습된 모델(.pt)로 교체


def mask_centroid(mask_xy: np.ndarray, image_shape: tuple[int, int]) -> tuple[float, float]:
    """마스크 폴리곤 좌표 -> 실제 채워진 영역의 무게중심(centroid).

    폴리곤 꼭짓점들의 평균이 아니라, 그 폴리곤을 실제로 채운 픽셀들의
    평균 위치를 계산함 (오목한 모양에서도 정확하게 나오도록 cv2.moments 사용).
    """
    canvas = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(canvas, [mask_xy.astype(np.int32)], 255)
    m = cv2.moments(canvas, binaryImage=True)
    if m["m00"] == 0:
        return float(mask_xy[:, 0].mean()), float(mask_xy[:, 1].mean())
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def detect_objects(
    image: np.ndarray, model_path: str = MODEL_PATH, depth: float = DUMMY_DEPTH_M
) -> list[dict]:
    """이미지에서 물체를 검출하고, 각 물체의 카메라 기준 3D 좌표까지 계산.

    반환: [{"label": str, "confidence": float, "box": (x0,y0,x1,y1),
            "mask_center": (u, v), "camera_3d": (x, y, z)}, ...]
    """
    model = YOLO(model_path)
    results = model(image, verbose=False)[0]

    detections = []
    if results.masks is None:
        return detections

    for box, mask_xy, cls, conf in zip(
        results.boxes.xyxy, results.masks.xy, results.boxes.cls, results.boxes.conf
    ):
        u, v = mask_centroid(np.array(mask_xy), image.shape)
        detections.append(
            {
                "label": results.names[int(cls)],
                "confidence": float(conf),
                "box": tuple(box.tolist()),
                "mask_center": (u, v),
                "camera_3d": pixel_to_camera(u, v, depth),
            }
        )
    return detections


def main():
    IMG_PATH = r"C:\Users\meihi\Downloads\배터리 모듈팩.png"
    img = cv2.imdecode(np.fromfile(IMG_PATH, dtype=np.uint8), cv2.IMREAD_COLOR)

    detections = detect_objects(img)
    print(f"detected {len(detections)} objects")
    for d in detections:
        print(d)


if __name__ == "__main__":
    main()

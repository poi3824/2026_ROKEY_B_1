"""YOLO 세그멘테이션으로 물체(bolt/nut/busbar)를 검출하고 마스크 중심 픽셀을 계산.

perception 브랜치 프로토타입(perception_prototype/detection.py)의 마스크 중심 계산
로직을 그대로 재사용한다. 마스크 중심은 박스 중심이 아니라 "실제 채워진 모양의
무게중심"으로 계산한다 (busbar처럼 비대칭인 물체는 박스 중심이 빈 공간에 잡힐 수 있음).
"""
import cv2
import numpy as np
from ultralytics import YOLO


def mask_centroid(mask_xy: np.ndarray, image_shape: tuple[int, int]) -> tuple[float, float]:
    canvas = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(canvas, [mask_xy.astype(np.int32)], 255)
    m = cv2.moments(canvas, binaryImage=True)
    if m["m00"] == 0:
        return float(mask_xy[:, 0].mean()), float(mask_xy[:, 1].mean())
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


class YoloSegDetector:

    def __init__(self, model_path: str):
        self._model = YOLO(model_path)
        self.names = self._model.names

    def detect(self, image: np.ndarray, conf_threshold: float) -> list[dict]:
        """반환: [{"label", "score", "pixel": (u, v), "bbox_size_px": (w, h)}, ...]"""
        results = self._model(image, conf=conf_threshold, verbose=False)[0]
        detections = []
        if results.masks is None:
            return detections

        for box, mask_xy, cls, conf in zip(
            results.boxes.xyxy, results.masks.xy, results.boxes.cls, results.boxes.conf
        ):
            u, v = mask_centroid(np.asarray(mask_xy), image.shape)
            x0, y0, x1, y1 = box.tolist()
            detections.append({
                "label": self.names[int(cls)],
                "score": float(conf),
                "pixel": (u, v),
                "bbox_size_px": (x1 - x0, y1 - y0),
            })
        return detections

"""YOLO 세그멘테이션(레거시) / keypoints(pose)로 물체(bolt/nut/busbar)를 검출.

perception 브랜치 프로토타입(perception_prototype/detection.py)의 마스크 중심 계산
로직을 그대로 재사용한다. 마스크 중심은 박스 중심이 아니라 "실제 채워진 모양의
무게중심"으로 계산한다 (busbar처럼 비대칭인 물체는 박스 중심이 빈 공간에 잡힐 수 있음).

keypoints 모델(YoloPoseDetector)은 9개 cuboid keypoint(Center + 8 corner,
training/keypoints/data/data.yaml 순서)를 낸다. 파지점은 기하 평균이 아니라
Center 키포인트를 그대로 쓴다 — training/eval/compare_grasp_point.py의 predict_pose()와
동일한 관례. Center는 3D ground truth를 픽셀로 투영한 값이라 그리퍼에 가려져도
정확하다는게 keypoints 모델의 이점이지만, depth 기반 역투영(camera_geometry)은
여전히 그 픽셀 위치의 depth 값을 읽으므로 가려진 상태에서의 depth 정확도는 보장하지
않는다 (occlusion-aware 3D 보정은 범위 밖).
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
        """반환: [{"label", "score", "pixel": (u, v), "bbox_px": (x0,y0,x1,y1),
        "mask_xy": np.ndarray}, ...]"""
        results = self._model(image, conf=conf_threshold, verbose=False)[0]
        detections = []
        if results.masks is None:
            return detections

        for box, mask_xy, cls, conf in zip(
            results.boxes.xyxy, results.masks.xy, results.boxes.cls, results.boxes.conf
        ):
            mask_xy = np.asarray(mask_xy)
            u, v = mask_centroid(mask_xy, image.shape)
            detections.append({
                "label": self.names[int(cls)],
                "score": float(conf),
                "pixel": (u, v),
                "bbox_px": tuple(box.tolist()),
                "mask_xy": mask_xy,
            })
        return detections


# training/keypoints/data/data.yaml의 kpt_shape: [9, 3] 순서와 동일.
KEYPOINT_ORDER = ["Center", "LDB", "LDF", "LUB", "LUF", "RDB", "RDF", "RUB", "RUF"]
CENTER_KEYPOINT_INDEX = KEYPOINT_ORDER.index("Center")


class YoloPoseDetector:
    """YOLO-pose(keypoints) 모델 기반 검출. 파지 픽셀 = Center 키포인트."""

    def __init__(self, model_path: str):
        self._model = YOLO(model_path)
        self.names = self._model.names

    def detect(self, image: np.ndarray, conf_threshold: float) -> list[dict]:
        """반환: [{"label", "score", "pixel": (u, v), "bbox_px": (x0,y0,x1,y1),
        "keypoints_px": np.ndarray shape (9,2)}, ...]"""
        results = self._model(image, conf=conf_threshold, verbose=False)[0]
        detections = []
        if results.keypoints is None:
            return detections

        keypoints_xy = results.keypoints.xy.cpu().numpy()
        for box, kpts, cls, conf in zip(
            results.boxes.xyxy, keypoints_xy, results.boxes.cls, results.boxes.conf
        ):
            u, v = kpts[CENTER_KEYPOINT_INDEX]
            detections.append({
                "label": self.names[int(cls)],
                "score": float(conf),
                "pixel": (float(u), float(v)),
                "bbox_px": tuple(box.tolist()),
                "keypoints_px": kpts,
            })
        return detections

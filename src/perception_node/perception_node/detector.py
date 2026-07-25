"""YOLO 세그멘테이션(디버그용 레거시) / keypoints(pose)로 물체(bolt/nut/busbar)를 검출.

perception 브랜치 프로토타입(perception_prototype/detection.py)의 마스크 중심 계산
로직을 그대로 재사용한다. 마스크 중심은 박스 중심이 아니라 "실제 채워진 모양의
무게중심"으로 계산한다 (busbar처럼 비대칭인 물체는 박스 중심이 빈 공간에 잡힐 수 있음).
YoloSegDetector는 런타임 파이프라인(perception_node.py)에서는 쓰지 않고
scripts/inspect_bag_frame.py 오프라인 디버그 도구 전용이다.

keypoints 모델(YoloPoseDetector)은 isaacpjt/M0609/datasets/pose/data.yaml
(kpt_shape=[6,3], flip_idx=[1,0,3,2,5,4]) 기준으로 학습됐다 — 클래스 무관하게
6개 keypoint 슬롯을 내는 단일 모델이며, busbar만 6개(hole_A/B, elbow_A/B, tip_A/B)가
실제로 채워지고 bolt/nut은 슬롯 0(top_center)만 유효, 나머지 5개는 라벨 단계부터
visibility=0으로 패딩된 더미다(kp_labels_*.json 확인). busbar의 6개는 Stage3
PnP(camera_geometry)가 그대로 써서 3D pose/orientation을 복원하므로 여기서는
순서를 바꾸거나 합치지 않고 원본 그대로 넘긴다.
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

    def detect(self, image: np.ndarray, conf_threshold: float, iou_threshold: float = 0.7) -> list[dict]:
        """반환: [{"label", "score", "pixel": (u, v), "bbox_px": (x0,y0,x1,y1),
        "mask_xy": np.ndarray}, ...]"""
        results = self._model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
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


# isaacpjt/M0609/datasets/pose/data.yaml 순서와 동일. busbar에만 6개 다 유효.
BUSBAR_LABEL = "busbar"
BUSBAR_KEYPOINT_ORDER = ["hole_A", "hole_B", "elbow_A", "elbow_B", "tip_A", "tip_B"]
# flip_idx=[1,0,3,2,5,4]와 동일한 C2(면내 180°) 대칭 짝 — Stage3 PnP의 좌우 모호성
# 처리에 쓴다 (hole_A<->hole_B, elbow_A<->elbow_B, tip_A<->tip_B).
BUSBAR_SYMMETRY_PAIRS = [(0, 1), (2, 3), (4, 5)]
# bolt/nut은 슬롯 0(top_center)만 유효.
SINGLE_KEYPOINT_SLOT = 0


class YoloPoseDetector:
    """YOLO-pose(keypoints) 모델 기반 검출."""

    def __init__(self, model_path: str):
        self._model = YOLO(model_path)
        self.names = self._model.names

    def detect(self, image: np.ndarray, conf_threshold: float, iou_threshold: float = 0.7) -> list[dict]:
        """반환: [{"label", "score", "pixel": (u, v), "bbox_px": (x0,y0,x1,y1),
        "keypoints_px": np.ndarray shape (6,2)}, ...]

        "pixel"은 클래스별 대표 픽셀이다 — bolt/nut은 top_center(슬롯 0) 그대로,
        busbar는 6-keypoint의 기하 중심(centroid, 파지/근사 위치용). busbar의 진짜
        3D pose/orientation은 이 중심이 아니라 "keypoints_px" 6개 전체를 camera_geometry의
        PnP 경로에 넘겨서 별도로 계산한다 — 슬롯 1~5는 bolt/nut에서는 학습 라벨부터
        무의미한 패딩이므로 호출부에서 클래스로 분기해 걸러야 한다.
        """
        results = self._model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
        detections = []
        if results.keypoints is None:
            return detections

        keypoints_xy = results.keypoints.xy.cpu().numpy()
        for box, kpts, cls, conf in zip(
            results.boxes.xyxy, keypoints_xy, results.boxes.cls, results.boxes.conf
        ):
            label = self.names[int(cls)]
            if label == BUSBAR_LABEL:
                u, v = kpts.mean(axis=0)
            else:
                u, v = kpts[SINGLE_KEYPOINT_SLOT]
            detections.append({
                "label": label,
                "score": float(conf),
                "pixel": (float(u), float(v)),
                "bbox_px": tuple(box.tolist()),
                "keypoints_px": kpts,
            })
        return detections

"""
YOLO segmentation 또는 keypoints 모델로 bolt, nut, busbar를 검출한다.

perception 브랜치 프로토타입(perception_prototype/detection.py)의 마스크 중심 계산
로직을 그대로 재사용한다. 마스크 중심은 박스 중심이 아니라 "실제 채워진 모양의
무게중심"으로 계산한다 (busbar처럼 비대칭인 물체는 박스 중심이 빈 공간에 잡힐 수 있음).

기본 v3와 전환용 v1 pose 모델은 모두 같은 6-keypoint 계약을 사용한다. busbar의
6점은 hole_A/B, elbow_A/B, tip_A/B이고, bolt/nut은 슬롯 0(top_center)만 유효하다.
따라서 busbar의 대표 픽셀은 confidence 0.5 이상인 화면 내 A/B 대칭쌍의
중점 평균, bolt/nut은 슬롯 0이다. 모든 원본 keypoint와 confidence도 반환해
camera_geometry의 busbar PnP 경로가 사용한다.
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

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float = 0.7,
    ) -> list[dict]:
        """
        검출 라벨, 점수, 픽셀, 경계 상자와 segmentation mask를 반환한다.

        반환 항목은 label, score, pixel, bbox_px, mask_xy 필드를 갖는다.
        """
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


# isaacpjt/M0609/datasets/pose/data.yaml(kpt_shape=[6,3]) 순서와 동일.
BUSBAR_LABEL = "busbar"
BUSBAR_KEYPOINT_ORDER = [
    "hole_A",
    "hole_B",
    "elbow_A",
    "elbow_B",
    "tip_A",
    "tip_B",
]
SINGLE_KEYPOINT_SLOT = 0
MIN_BUSBAR_KEYPOINT_CONFIDENCE = 0.5
BUSBAR_SYMMETRIC_KEYPOINT_PAIRS = (
    (0, 1),  # hole_A / hole_B
    (2, 3),  # elbow_A / elbow_B
    (4, 5),  # tip_A / tip_B
)


def busbar_keypoint_valid_mask(
    keypoints_px,
    keypoints_conf,
    image_size: tuple[int, int],
    min_confidence: float = MIN_BUSBAR_KEYPOINT_CONFIDENCE,
) -> np.ndarray:
    """Mark finite, confident busbar landmarks inside the source image."""
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if (
        not np.isfinite(min_confidence)
        or min_confidence < 0.0
        or min_confidence > 1.0
    ):
        raise ValueError("min_confidence must be finite and in [0, 1]")

    keypoints = np.asarray(keypoints_px, dtype=float)
    confidences = np.asarray(keypoints_conf, dtype=float)
    if (
        keypoints.shape != (len(BUSBAR_KEYPOINT_ORDER), 2)
        or confidences.shape != (len(BUSBAR_KEYPOINT_ORDER),)
    ):
        return np.zeros(len(BUSBAR_KEYPOINT_ORDER), dtype=bool)

    return (
        np.all(np.isfinite(keypoints), axis=1)
        & np.isfinite(confidences)
        & (confidences >= min_confidence)
        & (0.0 <= keypoints[:, 0])
        & (keypoints[:, 0] < image_width)
        & (0.0 <= keypoints[:, 1])
        & (keypoints[:, 1] < image_height)
    )


def valid_busbar_keypoints(
    keypoints_px,
    keypoints_conf,
    image_size: tuple[int, int],
    min_confidence: float = MIN_BUSBAR_KEYPOINT_CONFIDENCE,
) -> np.ndarray:
    """Return finite, confident busbar landmarks inside the source image."""
    keypoints = np.asarray(keypoints_px, dtype=float)
    valid = busbar_keypoint_valid_mask(
        keypoints,
        keypoints_conf,
        image_size,
        min_confidence,
    )
    if keypoints.shape != (len(BUSBAR_KEYPOINT_ORDER), 2):
        return np.empty((0, 2), dtype=float)
    return keypoints[valid]


def busbar_symmetric_pair_centers(
    keypoints_px,
    keypoints_conf,
    image_size: tuple[int, int],
    min_confidence: float = MIN_BUSBAR_KEYPOINT_CONFIDENCE,
) -> np.ndarray:
    """
    Return centers from complete physical A/B landmark pairs.

    Every A/B pair is symmetric about the authored busbar origin.  Averaging
    arbitrary surviving landmarks can therefore move the depth sample toward
    one end of the busbar; incomplete pairs are deliberately ignored.
    """
    keypoints = np.asarray(keypoints_px, dtype=float)
    valid = busbar_keypoint_valid_mask(
        keypoints,
        keypoints_conf,
        image_size,
        min_confidence,
    )
    if keypoints.shape != (len(BUSBAR_KEYPOINT_ORDER), 2):
        return np.empty((0, 2), dtype=float)

    pair_centers = [
        (keypoints[index_a] + keypoints[index_b]) / 2.0
        for index_a, index_b in BUSBAR_SYMMETRIC_KEYPOINT_PAIRS
        if valid[index_a] and valid[index_b]
    ]
    if not pair_centers:
        return np.empty((0, 2), dtype=float)
    return np.asarray(pair_centers, dtype=float)


def busbar_keypoints_are_reliable(
    keypoints_px,
    keypoints_conf,
    image_size: tuple[int, int],
    min_confidence: float = MIN_BUSBAR_KEYPOINT_CONFIDENCE,
) -> bool:
    """Validate every physical busbar landmark before pose estimation."""
    valid_keypoints = valid_busbar_keypoints(
        keypoints_px,
        keypoints_conf,
        image_size,
        min_confidence,
    )
    return valid_keypoints.shape == (len(BUSBAR_KEYPOINT_ORDER), 2)


def representative_pixel(
    label: str,
    keypoints_px: np.ndarray,
    bbox_px,
    *,
    keypoints_conf=(),
    image_size: tuple[int, int] | None = None,
) -> tuple[float, float]:
    """
    Return the class-specific pixel used by the depth fallback path.

    Six-point v1/v3 models use the mean center of their complete, confident,
    finite, in-frame A/B busbar landmark pairs. Bolt/nut and legacy models
    keep slot 0. The bounding-box center is used when no trustworthy
    representative exists.
    """
    x0, y0, x1, y1 = (float(value) for value in bbox_px)
    bbox_center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    keypoints = np.asarray(keypoints_px, dtype=float)
    if (
        keypoints.ndim != 2
        or keypoints.shape[1] != 2
    ):
        return bbox_center

    if label == BUSBAR_LABEL and keypoints.shape[0] == len(BUSBAR_KEYPOINT_ORDER):
        if image_size is None:
            return bbox_center
        pair_centers = busbar_symmetric_pair_centers(
            keypoints,
            keypoints_conf,
            image_size,
        )
        if not len(pair_centers):
            return bbox_center
        u, v = pair_centers.mean(axis=0)
    elif keypoints.shape[0] > SINGLE_KEYPOINT_SLOT:
        if not np.all(np.isfinite(keypoints)):
            return bbox_center
        u, v = keypoints[SINGLE_KEYPOINT_SLOT]
    else:
        return bbox_center
    return float(u), float(v)


class YoloPoseDetector:
    """YOLO-pose detector using the v1/v3 six-keypoint contract."""

    def __init__(
        self,
        model_path: str,
        inference_image_size: int | None = None,
    ):
        self._model = YOLO(model_path)
        self.names = self._model.names
        if inference_image_size is not None:
            if (
                isinstance(inference_image_size, bool)
                or int(inference_image_size) < 32
            ):
                raise ValueError(
                    "inference_image_size must be None or an integer >= 32"
                )
            inference_image_size = int(inference_image_size)
        self._inference_image_size = inference_image_size

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float,
        iou_threshold: float = 0.7,
    ) -> list[dict]:
        predict_kwargs = {
            "conf": conf_threshold,
            "iou": iou_threshold,
            "verbose": False,
        }
        if self._inference_image_size is not None:
            # The v3 checkpoint retains its 1280px training override.  On the
            # 640x640 fixed bolt view that scale suppresses the bolt directly
            # under Camera_bolt below the required 0.6 confidence.  The
            # camera-specific launch contract explicitly selects 640px.
            predict_kwargs["imgsz"] = self._inference_image_size
        results = self._model(image, **predict_kwargs)[0]
        detections = []
        if results.keypoints is None:
            return detections

        keypoints_xy = results.keypoints.xy.cpu().numpy()
        keypoints_conf = results.keypoints.conf
        if keypoints_conf is None:
            confidences = [None] * len(keypoints_xy)
        else:
            confidences = keypoints_conf.cpu().numpy()

        for box, kpts, kpts_conf, cls, conf in zip(
            results.boxes.xyxy,
            keypoints_xy,
            confidences,
            results.boxes.cls,
            results.boxes.conf,
        ):
            box_list = box.tolist()
            label = self.names[int(cls)]
            u, v = representative_pixel(
                label,
                kpts,
                box_list,
                keypoints_conf=kpts_conf,
                image_size=(image.shape[1], image.shape[0]),
            )

            detections.append({
                "label": label,
                "score": float(conf),
                "pixel": (u, v),
                "bbox_px": tuple(box_list),
                "keypoints_px": kpts,
                "keypoints_conf": kpts_conf,
            })
        return detections

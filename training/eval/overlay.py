"""Draw a side-by-side seg-vs-pose comparison overlay for one evaluated frame."""
import cv2
import numpy as np

GT_COLOR = (0, 0, 255)      # red, BGR
SEG_RAW_COLOR = (0, 165, 255)   # orange
SEG_CLEAN_COLOR = (0, 255, 0)   # green
POSE_COLOR = (255, 0, 0)        # blue


def _draw_point(img, pt, color, label, radius=6):
    if pt is None:
        return
    p = (int(round(pt[0])), int(round(pt[1])))
    cv2.circle(img, p, radius, color, -1)
    cv2.putText(img, label, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _draw_arrow(img, center, angle_rad, color, length=60):
    if center is None or angle_rad is None:
        return
    c = np.array(center, dtype=np.float64)
    d = np.array([np.cos(angle_rad), np.sin(angle_rad)]) * length / 2
    cv2.arrowedLine(img, tuple((c - d).astype(int)), tuple((c + d).astype(int)), color, 2, tipLength=0.15)


def _seg_panel(seg_img, seg_pred, gt_center):
    panel = seg_img.copy()
    mask = seg_pred["clean"]["mask_used"]
    if mask is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, contours, -1, SEG_CLEAN_COLOR, 2)
    _draw_point(panel, seg_pred["raw"]["centroid"], SEG_RAW_COLOR, "raw")
    _draw_point(panel, seg_pred["clean"]["centroid"], SEG_CLEAN_COLOR, "clean")
    _draw_arrow(panel, seg_pred["clean"]["centroid"], seg_pred["clean"]["angle_rad"], SEG_CLEAN_COLOR)
    _draw_point(panel, gt_center, GT_COLOR, "gt")
    cv2.putText(panel, "segmentation", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def _pose_panel(pose_img, pose_pred, gt_center):
    panel = pose_img.copy()
    _draw_point(panel, pose_pred["center"], POSE_COLOR, "pred")
    _draw_arrow(panel, pose_pred["center"], pose_pred["angle"], POSE_COLOR)
    _draw_point(panel, gt_center, GT_COLOR, "gt")
    if gt_center is not None:
        cv2.line(
            panel,
            tuple(np.round(pose_pred["center"]).astype(int)),
            tuple(np.round(gt_center).astype(int)),
            (200, 200, 200), 1,
        )
    cv2.putText(panel, "keypoints (pose)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def save_comparison_overlay(seg_img, pose_img, seg_pred, pose_pred, gt_center, out_path):
    left = _seg_panel(seg_img, seg_pred, gt_center)
    right = _pose_panel(pose_img, pose_pred, gt_center)
    h = min(left.shape[0], right.shape[0])
    left = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h))
    right = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))
    combined = np.hstack([left, np.full((h, 4, 3), 255, dtype=np.uint8), right])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)


def save_detection_card(image_bgr, mask, bbox_xyxy, centroid, class_name, conf,
                         cam_pt, world_pt, out_path, keypoints=None):
    """Single-panel readout matching the original detector screenshot's style: yellow
    tight bbox, red grasp-point dot, and a black info card with class/confidence +
    px/cam/world coordinates. Draws a green mask contour if `mask` is given (segmentation
    track) or the raw keypoint dots if `keypoints` is given (pose track, an (N,2) pixel
    array) -- exactly one of the two is expected depending on which model produced this
    detection."""
    panel = image_bgr.copy()

    if mask is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, contours, -1, (0, 255, 0), 2)

    if keypoints is not None:
        for kx, ky in keypoints:
            cv2.circle(panel, (int(round(kx)), int(round(ky))), 3, (255, 200, 0), -1)

    if bbox_xyxy is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in bbox_xyxy)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 255), 1)

    u, v = (int(round(centroid[0])), int(round(centroid[1])))
    cv2.circle(panel, (u, v), 5, (0, 0, 255), -1)

    lines = [
        f"{class_name} {conf:.2f}",
        f"px ({u},{v})",
        f"cam ({cam_pt[0]:.2f},{cam_pt[1]:.2f},{cam_pt[2]:.2f})",
        f"world ({world_pt[0]:.2f},{world_pt[1]:.2f},{world_pt[2]:.2f})",
    ]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    line_h = 22
    text_w = max(cv2.getTextSize(t, font, scale, thick)[0][0] for t in lines) + 20
    box_h = line_h * len(lines) + 12
    box_x, box_y = u + 15, v - box_h // 2
    box_x = min(box_x, panel.shape[1] - text_w - 2)
    box_y = max(2, min(box_y, panel.shape[0] - box_h - 2))
    overlay = panel.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + text_w, box_y + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, panel, 0.25, 0, panel)
    for i, text in enumerate(lines):
        ty = box_y + line_h * (i + 1) - 4
        cv2.putText(panel, text, (box_x + 10, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)
    return panel

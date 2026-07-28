"""Grasp-point extraction from a *predicted* segmentation mask, at inference time.

The noise this addresses comes from the trained model's prediction, not from training
labels (Isaac Sim's instance_segmentation ground truth is exact, since it's a render, not
a detector output). A jagged predicted contour still makes the raw moment centroid jump
frame to frame, and the existing pipeline (see training/README.md) never computed an
orientation at all -- for an asymmetric/Z-shaped object like the busbar, "one point, no
orientation" is not enough to grasp reliably.

This module is deliberately dependency-light and framework-agnostic (pure numpy/cv2
functions) so it can be dropped into whichever ROS node ends up doing inference.
"""
import cv2
import numpy as np


def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Morphological open (drop small speckle) then close (fill small holes) on a
    binary (0/255 or 0/1) predicted mask. Reduces contour jaggedness before any
    moment/PCA computation."""
    binary = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def mask_centroid(mask: np.ndarray):
    """Area-weighted centroid (image moments) of a binary mask. Returns (u, v) pixel
    coords, or None if the mask is empty. This mirrors the existing detector's
    cv2.moments-based approach, so it's an apples-to-apples baseline for comparison."""
    m = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
    if m["m00"] == 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def mask_orientation(mask: np.ndarray):
    """PCA of the mask's foreground pixel coordinates. Returns (angle_rad, major_axis_vec)
    where angle is measured from the image x-axis, or None if the mask is empty/degenerate.
    Absent from the current pipeline entirely -- this is what lets a grasp plan pick an
    approach direction instead of just a point."""
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 2:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    pts -= pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    angle = float(np.arctan2(major[1], major[0]))
    return angle, major


def extract_grasp_point(raw_mask: np.ndarray, apply_cleanup: bool = True):
    """End-to-end: (optionally) clean the predicted mask, then return centroid + orientation.

    Returns a dict: {"centroid": (u, v) | None, "angle_rad": float | None,
                      "axis": (dx, dy) | None, "mask_used": the mask actually measured on}.
    """
    mask = clean_mask(raw_mask) if apply_cleanup else (raw_mask > 0).astype(np.uint8) * 255
    centroid = mask_centroid(mask)
    orientation = mask_orientation(mask)
    return {
        "centroid": centroid,
        "angle_rad": orientation[0] if orientation else None,
        "axis": tuple(orientation[1]) if orientation else None,
        "mask_used": mask,
    }

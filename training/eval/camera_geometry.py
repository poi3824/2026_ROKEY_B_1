"""Pixel + depth -> camera-frame -> world-frame 3D backprojection, for rendering the
same px/cam/world readout style as the original detector screenshot.

Mirrors the pinhole math in project3's perception_node/camera_geometry.py
(pixel_to_camera_point via image_geometry.PinholeCameraModel + a tf2 lookup for
camera->world), but standalone: no ROS CameraInfo/tf2 available here, so intrinsics and
the camera's world pose are read directly from the dataset's own per-frame camera_data
(datasets/keypoints/<cam>/<idx>.json) instead.

Axis convention and both transforms below were empirically validated against this
dataset's own ground truth (see datasets/README.md's cuboid_keypoints_* fields): given a
frame's known pixel + depth for a point, this reproduces its stored camera_frame position
exactly, and chaining into camera_to_world_point reproduces its known world_frame position
to sub-millimeter error.
"""
import numpy as np


def pixel_to_camera_point(u: float, v: float, depth: float, fx: float, fy: float, cx: float, cy: float):
    """(u, v) pixel + depth (positive, along the optical axis, meters) -> camera-frame
    (x, y, z) in this dataset's stored convention (camera looks down -Z, Y flipped
    relative to the image row axis -- i.e. an OpenGL/USD-style camera frame, not OpenCV's)."""
    x = (u - cx) * depth / fx
    y = -(v - cy) * depth / fy
    z = -depth
    return np.array([x, y, z], dtype=np.float64)


def camera_to_world_point(camera_point, camera_view_matrix):
    """camera_view_matrix is the frame's world_h @ view = camera_h transform (as stored
    in camera_data, row-vector convention -- see training/eval/compare_grasp_point.py's
    ground_truth_for, which already treats it as the world<->camera extrinsic for this
    same JSON). Invert it to go camera -> world."""
    view = np.asarray(camera_view_matrix, dtype=np.float64)
    camera_h = np.append(np.asarray(camera_point, dtype=np.float64), 1.0)
    world_h = camera_h @ np.linalg.inv(view)
    return world_h[:3] / world_h[3]


def pixel_to_world_point(u, v, depth, camera_data):
    """Convenience wrapper taking a frame's whole camera_data dict (as loaded from
    datasets/keypoints/<cam>/<idx>.json)."""
    intr = camera_data["intrinsics"]
    cam_pt = pixel_to_camera_point(u, v, depth, intr["fx"], intr["fy"], intr["cx"], intr["cy"])
    world_pt = camera_to_world_point(cam_pt, camera_data["camera_view_matrix"])
    return cam_pt, world_pt

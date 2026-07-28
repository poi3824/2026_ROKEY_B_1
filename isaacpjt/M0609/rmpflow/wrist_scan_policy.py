"""Pure station-independent wrist scan target helpers."""

from __future__ import annotations

import numpy as np


# With the downward busbar grasp orientation, the D455 optical origin is about
# 45 mm behind link_6 in world X and 12 mm ahead in world Y.  Moving link_6
# 190 mm past the Mesh centre places the complete 407 mm busbar above the
# gripper silhouette instead of hiding its far keypoints behind the fingers.
DEFAULT_BUSBAR_SCAN_OFFSET_X = 0.19
DEFAULT_BUSBAR_SCAN_OFFSET_Y = -0.012


def busbar_wrist_scan_target(
    mesh_center,
    *,
    scan_z: float = 0.7,
    offset_x: float = DEFAULT_BUSBAR_SCAN_OFFSET_X,
    offset_y: float = DEFAULT_BUSBAR_SCAN_OFFSET_Y,
):
    """Return one wrist scan pose relative to the selected Mesh centre."""

    center = np.asarray(mesh_center, dtype=float)
    values = np.array([scan_z, offset_x, offset_y], dtype=float)
    if center.shape != (3,):
        raise ValueError("mesh_center must be a 3-vector")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(values)):
        raise ValueError("wrist scan target values must be finite")

    return np.array(
        [
            center[0] + offset_x,
            center[1] + offset_y,
            scan_z,
        ],
        dtype=float,
    )

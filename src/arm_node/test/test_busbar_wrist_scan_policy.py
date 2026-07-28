import sys
from pathlib import Path

import numpy as np
import pytest


RMPFLOW_DIR = (
    Path(__file__).parents[3]
    / "isaacpjt"
    / "M0609"
    / "rmpflow"
)
sys.path.insert(0, str(RMPFLOW_DIR))

from wrist_scan_policy import busbar_wrist_scan_target  # noqa: E402


@pytest.mark.parametrize(
    "center,expected_xy",
    [
        ([0.5136, 2.6705, 0.2692], [0.7036, 2.6585]),
        ([-0.2323, 2.6705, 0.2692], [-0.0423, 2.6585]),
        ([-0.9701, 2.6705, 0.2692], [-0.7801, 2.6585]),
    ],
)
def test_station_scan_target_tracks_selected_mesh(center, expected_xy):
    target = busbar_wrist_scan_target(center)

    np.testing.assert_allclose(target[:2], expected_xy)
    assert target[2] == 0.7


def test_scan_height_can_be_selected_without_changing_xy_policy():
    target = busbar_wrist_scan_target([1.0, 2.0, 3.0], scan_z=0.8)

    np.testing.assert_allclose(target, [1.19, 1.988, 0.8])


@pytest.mark.parametrize(
    "center",
    [
        [1.0, 2.0],
        [1.0, np.nan, 3.0],
    ],
)
def test_invalid_mesh_centers_are_rejected(center):
    with pytest.raises(ValueError):
        busbar_wrist_scan_target(center)

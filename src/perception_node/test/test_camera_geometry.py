# Copyright 2026 OpenAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
import numpy as np
import pytest

from perception_node.camera_geometry import (
    BUSBAR_OBJECT_POINTS_M,
    busbar_pnp_world_pose,
    dual_path_discrepancy_m,
    sample_depth,
    select_busbar_world_point,
    solve_busbar_pnp,
)


class FakeCameraModel:

    def __init__(self):
        self._camera_matrix = np.array([
            [610.0, 0.0, 320.0],
            [0.0, 608.0, 320.0],
            [0.0, 0.0, 1.0],
        ])
        self._distortion = np.zeros((1, 5))

    def intrinsicMatrix(self):
        return self._camera_matrix

    def distortionCoeffs(self):
        return self._distortion


class IdentityBuffer:

    def lookup_transform(self, target, source, stamp, timeout):
        transform = TransformStamped()
        transform.header.frame_id = target
        transform.child_frame_id = source
        transform.transform.rotation.w = 1.0
        return transform


def _synthetic_projection():
    model = FakeCameraModel()
    rvec = np.array([[0.12], [-0.08], [0.25]])
    tvec = np.array([[0.03], [-0.02], [1.10]])
    projected, _ = cv2.projectPoints(
        BUSBAR_OBJECT_POINTS_M,
        rvec,
        tvec,
        model.intrinsicMatrix(),
        model.distortionCoeffs(),
    )
    return model, rvec, tvec, projected.reshape(-1, 2)


def test_sample_depth_keeps_exact_thin_target_pixel():
    depth = np.full((5, 5), 0.30, dtype=np.float32)
    depth[2, 2] = 0.82

    assert sample_depth(depth, 2.0, 2.0) == np.float32(0.82)


def test_solve_busbar_pnp_reprojects_synthetic_six_points():
    model, _, _, keypoints = _synthetic_projection()

    solved = solve_busbar_pnp(model, keypoints)

    assert solved is not None
    rvec, tvec, error = solved
    reprojected, _ = cv2.projectPoints(
        BUSBAR_OBJECT_POINTS_M,
        rvec,
        tvec,
        model.intrinsicMatrix(),
        model.distortionCoeffs(),
    )
    assert float(tvec.reshape(3)[2]) > 0.0
    assert error < 1e-5
    np.testing.assert_allclose(
        reprojected.reshape(-1, 2), keypoints, atol=1e-4
    )


def test_solve_busbar_pnp_rejects_invalid_or_degenerate_points():
    model = FakeCameraModel()

    assert solve_busbar_pnp(model, np.zeros((5, 2))) is None
    assert solve_busbar_pnp(model, np.full((6, 2), np.nan)) is None
    assert solve_busbar_pnp(model, np.ones((6, 2))) is None


def test_busbar_pnp_world_pose_uses_six_point_geometry_not_depth():
    model, true_rvec, true_tvec, keypoints = _synthetic_projection()
    rotation, _ = cv2.Rodrigues(true_rvec)
    expected_holes = (
        rotation @ BUSBAR_OBJECT_POINTS_M[:2].T + true_tvec
    ).T

    result = busbar_pnp_world_pose(
        model,
        keypoints,
        IdentityBuffer(),
        'world',
        'camera_optical',
        Time(),
    )

    assert result['status'] == ''
    np.testing.assert_allclose(
        result['hole_world']['hole_A'], expected_holes[0], atol=1e-5
    )
    np.testing.assert_allclose(
        result['hole_world']['hole_B'], expected_holes[1], atol=1e-5
    )
    np.testing.assert_allclose(
        result['mid_xy'], expected_holes[:, :2].mean(axis=0), atol=1e-5
    )


def test_dual_path_discrepancy_is_planar_distance():
    assert dual_path_discrepancy_m(
        (1.0, 2.0), (1.03, 2.04)
    ) == pytest.approx(0.05)


def test_busbar_selector_prefers_depth_when_z_estimates_agree():
    selected, source, z_delta = select_busbar_world_point(
        (1.00, 2.00, 0.24),
        (1.01, 2.01, 0.27),
    )

    assert selected == (1.00, 2.00, 0.24)
    assert source == 'depth'
    assert z_delta == pytest.approx(0.03)


def test_busbar_selector_can_force_fixed_camera_depth_on_large_z_delta():
    selected, source, z_delta = select_busbar_world_point(
        (0.513, 2.672, 0.246),
        (0.468, 2.669, 0.462),
        prefer_depth=True,
    )

    assert selected == (0.513, 2.672, 0.246)
    assert source == 'depth'
    assert z_delta == pytest.approx(0.216)


def test_busbar_selector_uses_pnp_for_floor_depth():
    selected, source, z_delta = select_busbar_world_point(
        (1.00, 2.00, 0.004),
        (1.01, 2.01, 0.24),
    )

    assert selected == (1.01, 2.01, 0.24)
    assert source == 'pnp'
    assert z_delta == pytest.approx(0.236)


@pytest.mark.parametrize(
    ('depth_world', 'pnp_world', 'expected', 'source'),
    [
        ((1.0, 2.0, 0.3), None, (1.0, 2.0, 0.3), 'depth'),
        (None, (1.0, 2.0, 0.3), (1.0, 2.0, 0.3), 'pnp'),
        ((1.0, float('nan'), 0.3), None, None, 'none'),
        (None, None, None, 'none'),
    ],
)
def test_busbar_selector_handles_missing_or_invalid_paths(
    depth_world,
    pnp_world,
    expected,
    source,
):
    selected, selected_source, z_delta = select_busbar_world_point(
        depth_world,
        pnp_world,
    )

    assert selected == expected
    assert selected_source == source
    assert z_delta is None

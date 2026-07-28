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

from types import SimpleNamespace

import numpy as np

import perception_node.perception_node as node_module
from perception_node.detector import representative_pixel
from perception_node.perception_node import PerceptionNode


class _Logger:

    def info(self, *_args, **_kwargs):
        pass

    def warn(self, *_args, **_kwargs):
        pass


def test_low_confidence_landmarks_skip_pnp_and_keep_depth_pose(monkeypatch):
    def unexpected_pnp(*_args, **_kwargs):
        raise AssertionError('low-confidence landmarks must not reach PnP')

    monkeypatch.setattr(
        node_module,
        'busbar_pnp_world_pose',
        unexpected_pnp,
    )
    transformed_pixels = []

    def transform_pixel(pixel, _depth, _stamp):
        transformed_pixels.append(pixel)
        return (
            (0.1, 0.2, 0.3),
            (1.1, 2.2, 0.3),
            '',
        )

    node = SimpleNamespace(
        _camera_model=SimpleNamespace(width=640, height=640),
        _transform_pixel=transform_pixel,
        _prefer_busbar_depth=False,
        get_logger=lambda: _Logger(),
    )
    keypoints = np.array([
        [250.0, 300.0],
        [390.0, 300.0],
        [270.0, 320.0],
        [700.0, 320.0],
        [np.nan, 340.0],
        [380.0, 340.0],
    ])
    confidences = np.array([0.9, 0.9, 0.49, 0.9, 0.9, 0.9])
    depth_pixel = representative_pixel(
        'busbar',
        keypoints,
        (200.0, 250.0, 440.0, 390.0),
        keypoints_conf=confidences,
        image_size=(640, 640),
    )
    detection = {
        'pixel': depth_pixel,
        'keypoints_px': keypoints,
        'keypoints_conf': confidences,
    }

    camera_point, world_point, status = PerceptionNode._resolve_busbar_pose(
        node,
        detection,
        np.full((640, 640), 0.3, dtype=np.float32),
        object(),
    )

    np.testing.assert_allclose(
        transformed_pixels,
        [[320.0, 300.0]],
    )
    assert camera_point == (0.1, 0.2, 0.3)
    assert world_point == (1.1, 2.2, 0.3)
    assert status == ''

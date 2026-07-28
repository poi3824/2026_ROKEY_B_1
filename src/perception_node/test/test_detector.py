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
import pytest

import perception_node.detector as detector_module
from perception_node.detector import (
    YoloPoseDetector,
    busbar_keypoints_are_reliable,
    busbar_symmetric_pair_centers,
    representative_pixel,
    valid_busbar_keypoints,
)


def test_six_keypoint_busbar_uses_landmark_centroid():
    keypoints = np.array([
        [10.0, 20.0],
        [90.0, 80.0],
        [20.0, 35.0],
        [80.0, 65.0],
        [5.0, 10.0],
        [95.0, 90.0],
    ])

    assert representative_pixel(
        'busbar',
        keypoints,
        (0.0, 0.0, 100.0, 100.0),
        keypoints_conf=np.full(6, 0.9),
        image_size=(100, 100),
    ) == (50.0, 50.0)


def test_busbar_depth_pixel_uses_only_complete_symmetric_pairs():
    keypoints = np.array([
        [10.0, 20.0],
        [90.0, 80.0],
        [20.0, 40.0],
        [120.0, 60.0],
        [np.nan, 10.0],
        [80.0, 60.0],
    ])
    confidences = np.array([0.9, 0.49, 0.5, 0.95, 0.99, 0.8])

    np.testing.assert_allclose(
        valid_busbar_keypoints(
            keypoints,
            confidences,
            (100, 100),
        ),
        np.array([
            [10.0, 20.0],
            [20.0, 40.0],
            [80.0, 60.0],
        ]),
    )
    assert busbar_symmetric_pair_centers(
        keypoints,
        confidences,
        (100, 100),
    ).shape == (0, 2)
    assert representative_pixel(
        'busbar',
        keypoints,
        (30.0, 40.0, 70.0, 80.0),
        keypoints_conf=confidences,
        image_size=(100, 100),
    ) == (50.0, 60.0)


def test_busbar_depth_pixel_averages_complete_symmetric_pair_centers():
    keypoints = np.array([
        [10.0, 20.0],
        [90.0, 80.0],
        [20.0, 40.0],
        [80.0, 60.0],
        [5.0, 10.0],
        [95.0, 90.0],
    ])
    confidences = np.array([0.9, 0.9, 0.9, 0.9, 0.49, 0.8])

    np.testing.assert_allclose(
        busbar_symmetric_pair_centers(
            keypoints,
            confidences,
            (100, 100),
        ),
        np.array([
            [50.0, 50.0],
            [50.0, 50.0],
        ]),
    )
    assert representative_pixel(
        'busbar',
        keypoints,
        (30.0, 40.0, 70.0, 80.0),
        keypoints_conf=confidences,
        image_size=(100, 100),
    ) == (50.0, 50.0)


def test_busbar_depth_pixel_falls_back_to_bbox_without_valid_landmark():
    keypoints = np.array([
        [10.0, 20.0],
        [90.0, 80.0],
        [20.0, 40.0],
        [120.0, 60.0],
        [np.nan, 10.0],
        [80.0, 60.0],
    ])
    confidences = np.array([0.49, 0.1, np.nan, 0.95, 0.99, 0.2])

    assert representative_pixel(
        'busbar',
        keypoints,
        (30.0, 40.0, 70.0, 80.0),
        keypoints_conf=confidences,
        image_size=(100, 100),
    ) == (50.0, 60.0)


def test_six_keypoint_bolt_and_nut_use_top_center_slot():
    keypoints = np.array([
        [23.0, 31.0],
        [99.0, 98.0],
        [97.0, 96.0],
        [95.0, 94.0],
        [93.0, 92.0],
        [91.0, 90.0],
    ])

    for label in ('bolt', 'nut'):
        assert representative_pixel(
            label, keypoints, (0.0, 0.0, 100.0, 100.0)
        ) == (23.0, 31.0)


def test_legacy_nine_keypoint_model_keeps_slot_zero_center():
    keypoints = np.arange(18, dtype=float).reshape(9, 2)

    assert representative_pixel(
        'busbar', keypoints, (20.0, 40.0, 80.0, 100.0)
    ) == tuple(keypoints[0])


def test_malformed_keypoints_fall_back_to_bbox_center():
    keypoints = np.full((6, 2), np.nan)

    assert representative_pixel(
        'busbar',
        keypoints,
        (20.0, 40.0, 80.0, 100.0),
        keypoints_conf=np.full(6, 0.9),
        image_size=(100, 120),
    ) == (50.0, 70.0)


def test_busbar_pose_requires_all_six_confident_in_frame_landmarks():
    keypoints = np.array([
        [10.0, 20.0],
        [90.0, 80.0],
        [20.0, 35.0],
        [80.0, 65.0],
        [5.0, 10.0],
        [95.0, 90.0],
    ])
    confidences = np.full(6, 0.5)

    assert busbar_keypoints_are_reliable(
        keypoints,
        confidences,
        (100, 100),
    )
    low_confidence = confidences.copy()
    low_confidence[3] = 0.49
    assert not busbar_keypoints_are_reliable(
        keypoints,
        low_confidence,
        (100, 100),
    )
    clipped = keypoints.copy()
    clipped[0, 0] = 100.0
    assert not busbar_keypoints_are_reliable(
        clipped,
        confidences,
        (100, 100),
    )
    assert not busbar_keypoints_are_reliable(
        keypoints,
        (),
        (100, 100),
    )


class _FakeYolo:
    names = {}

    def __init__(self):
        self.kwargs = None

    def __call__(self, _image, **kwargs):
        self.kwargs = kwargs
        return [SimpleNamespace(keypoints=None)]


class _CpuArray:

    def __init__(self, values):
        self._values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _FakeBusbarPoseYolo:
    names = {0: 'busbar'}

    def __init__(self, keypoints, confidences):
        self._result = SimpleNamespace(
            keypoints=SimpleNamespace(
                xy=_CpuArray([keypoints]),
                conf=_CpuArray([confidences]),
            ),
            boxes=SimpleNamespace(
                xyxy=np.array([[20.0, 30.0, 80.0, 90.0]]),
                cls=np.array([0]),
                conf=np.array([0.8]),
            ),
        )

    def __call__(self, _image, **_kwargs):
        return [self._result]


def test_pose_detector_falls_back_when_busbar_has_no_complete_pair(monkeypatch):
    keypoints = np.array([
        [10.0, 20.0],
        [90.0, 80.0],
        [20.0, 40.0],
        [120.0, 60.0],
        [np.nan, 10.0],
        [80.0, 60.0],
    ])
    confidences = np.array([0.9, 0.49, 0.5, 0.95, 0.99, 0.8])
    fake = _FakeBusbarPoseYolo(keypoints, confidences)
    monkeypatch.setattr(detector_module, 'YOLO', lambda _path: fake)

    detection = YoloPoseDetector('unused.pt').detect(
        np.zeros((100, 100, 3), dtype=np.uint8),
        conf_threshold=0.6,
    )[0]

    assert detection['pixel'] == (50.0, 60.0)


def test_pose_detector_passes_camera_specific_inference_size(monkeypatch):
    fake = _FakeYolo()
    monkeypatch.setattr(detector_module, 'YOLO', lambda _path: fake)
    detector = YoloPoseDetector('unused.pt', inference_image_size=640)

    assert detector.detect(
        np.zeros((32, 32, 3), dtype=np.uint8),
        conf_threshold=0.6,
    ) == []
    assert fake.kwargs['imgsz'] == 640


def test_pose_detector_preserves_model_default_when_size_is_unset(monkeypatch):
    fake = _FakeYolo()
    monkeypatch.setattr(detector_module, 'YOLO', lambda _path: fake)
    detector = YoloPoseDetector('unused.pt')

    detector.detect(
        np.zeros((32, 32, 3), dtype=np.uint8),
        conf_threshold=0.6,
    )

    assert 'imgsz' not in fake.kwargs


@pytest.mark.parametrize('invalid', [True, 0, 31])
def test_pose_detector_rejects_invalid_explicit_inference_size(
    monkeypatch,
    invalid,
):
    monkeypatch.setattr(
        detector_module,
        'YOLO',
        lambda _path: _FakeYolo(),
    )

    with pytest.raises(ValueError):
        YoloPoseDetector('unused.pt', inference_image_size=invalid)

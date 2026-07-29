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
import collections
import threading

import numpy as np

import perception_node.perception_node as node_module
from perception_node.detector import representative_pixel
from perception_node.perception_node import PerceptionNode
from perception_node.perception_policy import (
    BoltPairAssociationGate,
    CameraFrameGate,
    SingleTargetAssociationGate,
    SourceTimestampEpochTracker,
    TargetFrameBarrier,
)


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


def _ready_barrier_after_reset():
    barrier = TargetFrameBarrier(required_frames=3)
    stale_generation = barrier.generation
    barrier.reset()
    for stamp_ns in (20, 21, 22):
        barrier.record(
            'busbar',
            stamp_ns,
            generation=barrier.generation,
        )
        barrier.record(
            'bolt',
            stamp_ns,
            generation=barrier.generation,
        )
    return barrier, stale_generation


def test_grasp_service_rejects_cache_from_previous_reset_generation():
    barrier, stale_generation = _ready_barrier_after_reset()
    node = SimpleNamespace(
        _target_barrier=barrier,
        _target_input_error=lambda _label: None,
        _latest_by_label={
            'busbar': (
                stale_generation,
                0.9,
                (1.0, 2.0, 3.0),
                object(),
            ),
        },
    )
    response = SimpleNamespace(found=None, message='')

    PerceptionNode._handle_get_grasp_pose_locked(
        node,
        SimpleNamespace(label='busbar'),
        response,
    )

    assert not response.found
    assert '이전 reset 세대' in response.message


def test_bolt_service_rejects_pair_from_previous_reset_generation():
    barrier, stale_generation = _ready_barrier_after_reset()
    node = SimpleNamespace(
        _target_barrier=barrier,
        _target_input_error=lambda _label: None,
        _latest_bolt_pair=(
            stale_generation,
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
            object(),
            3,
        ),
    )
    response = SimpleNamespace(found=None, message='')

    PerceptionNode._handle_get_bolt_pair_locked(
        node,
        SimpleNamespace(),
        response,
    )

    assert not response.found
    assert '이전 reset 세대' in response.message


def test_bolt_service_rejects_ready_cache_during_latest_frame_miss():
    barrier, _stale_generation = _ready_barrier_after_reset()
    node = SimpleNamespace(
        _target_barrier=barrier,
        _target_present_latest={'bolt': False},
        _target_input_error=lambda _label: None,
        _latest_bolt_pair=(
            barrier.generation,
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
            object(),
            3,
        ),
    )
    response = SimpleNamespace(found=None, message='')

    PerceptionNode._handle_get_bolt_pair_locked(
        node,
        SimpleNamespace(),
        response,
    )

    assert not response.found
    assert '최신 accepted inference' in response.message


def test_bolt_pair_service_owns_one_demand_driven_capture_then_returns_idle():
    """Idle cameras must not pair bolts until the explicit service is used."""
    barrier = TargetFrameBarrier(required_frames=3)
    node = SimpleNamespace(
        _state_lock=threading.RLock(),
        _target_barrier=barrier,
        _target_present_latest={},
        _latest_target_received_at={},
        _recent_bolt_a=[object()],
        _recent_bolt_b=[object()],
        _latest_bolt_pair=object(),
        _bolt_pair_association=BoltPairAssociationGate(),
        _bolt_pair_tracking_active=False,
        _bolt_pair_capture_epoch=0,
        _tag_service_generation=lambda response: response,
    )
    node._clear_bolt_track = lambda **kwargs: (
        PerceptionNode._clear_bolt_track(node, **kwargs)
    )
    node._arm_bolt_pair_tracking_locked = lambda: (
        PerceptionNode._arm_bolt_pair_tracking_locked(node)
    )
    node._cancel_bolt_pair_tracking_locked = lambda: (
        PerceptionNode._cancel_bolt_pair_tracking_locked(node)
    )

    def pending(_request, response):
        response.found = False
        response.message = 'capturing'
        return response

    node._handle_get_bolt_pair_locked = pending
    first = PerceptionNode._handle_get_bolt_pair(
        node,
        SimpleNamespace(),
        SimpleNamespace(found=None, message=''),
    )
    assert not first.found
    assert node._bolt_pair_tracking_active
    assert node._bolt_pair_capture_epoch == 1
    assert node._latest_bolt_pair is None
    assert node._target_barrier.count('bolt') == 0

    # Retry calls own the same fresh capture; they must not erase the first
    # post-request frames and make the 3-frame barrier impossible to reach.
    second = PerceptionNode._handle_get_bolt_pair(
        node,
        SimpleNamespace(),
        SimpleNamespace(found=None, message=''),
    )
    assert not second.found
    assert node._bolt_pair_tracking_active
    assert node._bolt_pair_capture_epoch == 1

    def found(_request, response):
        response.found = True
        response.message = 'stable pair'
        return response

    node._handle_get_bolt_pair_locked = found
    result = PerceptionNode._handle_get_bolt_pair(
        node,
        SimpleNamespace(),
        SimpleNamespace(found=None, message=''),
    )
    assert result.found
    assert not node._bolt_pair_tracking_active
    assert node._bolt_pair_capture_epoch == 2


def test_wrist_on_demand_capture_idles_without_entering_detector_path():
    """An idle wrist node must return before source checks or GPU inference."""
    node = SimpleNamespace(
        _state_lock=threading.RLock(),
        _on_demand_inference=True,
        _demand_capture_active=False,
        _demand_capture_epoch=4,
        count_publishers=lambda _topic: (_ for _ in ()).throw(
            AssertionError('idle capture must not inspect sources')
        ),
    )

    PerceptionNode._detect_and_publish_once(node)

    assert node._demand_capture_epoch == 4


def test_wrist_on_demand_idle_timer_does_not_take_inference_lock():
    node = SimpleNamespace(
        _state_lock=threading.RLock(),
        _on_demand_inference=True,
        _demand_capture_active=False,
        _inference_lock=SimpleNamespace(
            acquire=lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError('idle timer must not take detector lock')
            ),
        ),
    )

    PerceptionNode._detect_and_publish(node)


def test_wrist_on_demand_capture_arms_on_service_and_stops_after_success():
    node = SimpleNamespace(
        _state_lock=threading.RLock(),
        _on_demand_inference=True,
        _demand_capture_active=False,
        _demand_capture_epoch=0,
        _tag_service_generation=lambda response: response,
    )

    def found(_request, response):
        response.found = True
        response.message = 'fresh snapshot'
        return response

    node._handle_get_grasp_pose_locked = found
    result = PerceptionNode._handle_get_grasp_pose(
        node,
        SimpleNamespace(label='busbar'),
        SimpleNamespace(found=None, message=''),
    )

    assert result.found
    # First service call arms epoch 1; successful handoff fences it at 2.
    assert not node._demand_capture_active
    assert node._demand_capture_epoch == 2


def test_wrist_on_demand_reset_restarts_active_capture_epoch():
    node = SimpleNamespace(
        _on_demand_inference=True,
        _demand_capture_active=True,
        _demand_capture_epoch=7,
    )

    assert not PerceptionNode._arm_on_demand_inference_locked(node)
    assert PerceptionNode._arm_on_demand_inference_locked(
        node,
        restart=True,
    )
    assert node._demand_capture_active
    assert node._demand_capture_epoch == 8


def test_nut_service_rejects_ready_cache_during_latest_frame_miss():
    barrier = TargetFrameBarrier(required_frames=3)
    for stamp_ns in (20, 21, 22):
        barrier.record('nut', stamp_ns)
    node = SimpleNamespace(
        _target_barrier=barrier,
        _target_present_latest={'nut': False},
        _target_input_error=lambda _label: None,
        _latest_by_label={
            'nut': (
                barrier.generation,
                0.9,
                (1.0, 2.0, 3.0),
                object(),
            ),
        },
    )
    response = SimpleNamespace(found=None, message='')

    PerceptionNode._handle_get_grasp_pose_locked(
        node,
        SimpleNamespace(label='nut'),
        response,
    )

    assert not response.found
    assert '최신 accepted inference' in response.message


def test_target_cache_preserves_real_rgb_and_depth_receipt_times(
    monkeypatch,
):
    barrier = TargetFrameBarrier(required_frames=3)
    node = SimpleNamespace(
        _target_barrier=barrier,
        _latest_target_received_at={},
        get_logger=lambda: _Logger(),
    )

    assert PerceptionNode._record_target_frame(
        node,
        'nut',
        10,
        barrier.generation,
        4.25,
        4.5,
    )
    assert node._latest_target_received_at['nut'] == (4.25, 4.5)

    service_node = SimpleNamespace(
        _require_unique_camera_source=False,
        _latest_target_received_at=node._latest_target_received_at,
        _max_input_receipt_age_sec=1.0,
    )
    monkeypatch.setattr(node_module.time, 'monotonic', lambda: 5.0)
    assert PerceptionNode._target_input_error(
        service_node, 'nut') is None

    monkeypatch.setattr(node_module.time, 'monotonic', lambda: 5.26)
    error = PerceptionNode._target_input_error(service_node, 'nut')
    assert 'stale receipt' in error


def test_service_message_exposes_current_reset_generation():
    barrier = TargetFrameBarrier(required_frames=3)
    barrier.reset()
    node = SimpleNamespace(_target_barrier=barrier)
    response = SimpleNamespace(message='not ready')

    tagged = PerceptionNode._tag_service_generation(node, response)

    assert tagged is response
    assert response.message == '[generation=1] not ready'


def test_service_message_appends_stale_inference_diagnostics():
    barrier = TargetFrameBarrier(required_frames=3)
    node = SimpleNamespace(
        _target_barrier=barrier,
        _inference_diagnostic_suffix=lambda: (
            ' [stale_inference consecutive=4 latency=1.250s]'
        ),
    )
    response = SimpleNamespace(message='target frame n=0')

    PerceptionNode._tag_service_generation(node, response)

    assert response.message == (
        '[generation=0] target frame n=0 '
        '[stale_inference consecutive=4 latency=1.250s]'
    )


def test_stale_inference_diagnostics_are_service_visible_and_recover():
    node = SimpleNamespace(
        _consecutive_stale_inferences=0,
        _total_stale_inferences=0,
        _last_inference_latency_sec=None,
        _last_stale_inference_reason=None,
    )

    PerceptionNode._record_stale_inference(
        node,
        'stale receipt: rgb=1.2s',
        1.25,
    )
    PerceptionNode._record_stale_inference(
        node,
        'stale receipt: rgb=1.4s',
        1.5,
    )
    suffix = PerceptionNode._inference_diagnostic_suffix(node)

    assert 'consecutive=2' in suffix
    assert 'total=2' in suffix
    assert 'latency=1.500s' in suffix
    assert 'rgb=1.4s' in suffix

    PerceptionNode._record_successful_inference(node, 0.4)
    assert node._consecutive_stale_inferences == 0
    assert node._total_stale_inferences == 2
    assert PerceptionNode._inference_diagnostic_suffix(node) == ''


def test_transform_uses_camera_snapshot_not_mutable_current_state(
    monkeypatch,
):
    calls = []

    def transform(
        camera_model,
        _depth,
        _pixel,
        _buffer,
        _world,
        camera_frame,
        _stamp,
        *,
        on_tf_error,
    ):
        calls.append((camera_model, camera_frame, on_tf_error))
        return None, None, 'expected'

    monkeypatch.setattr(
        node_module,
        'transform_pixel_to_world',
        transform,
    )
    old_model = object()
    snapshot_model = object()
    node = SimpleNamespace(
        _camera_model=old_model,
        _camera_frame_id='newer-frame',
        _tf_buffer=object(),
        _world_frame='world',
        _log_tf_error=lambda *_args: None,
    )

    result = PerceptionNode._transform_pixel(
        node,
        (10.0, 20.0),
        object(),
        object(),
        camera_model=snapshot_model,
        camera_frame_id='snapshot-frame',
    )

    assert result == (None, None, 'expected')
    assert calls[0][:2] == (snapshot_model, 'snapshot-frame')


def test_rgb_callback_started_before_reset_cannot_repopulate_queue(
    monkeypatch,
):
    barrier = TargetFrameBarrier(required_frames=3)
    node = SimpleNamespace()

    class ResetBetweenCallbackPhases:
        def __init__(self):
            self._lock = threading.RLock()
            self.exits = 0

        def __enter__(self):
            self._lock.acquire()
            return self

        def __exit__(self, *_args):
            self.exits += 1
            self._lock.release()
            if self.exits == 1:
                node._reset_monotonic_cutoff = 9.5
                barrier.reset()

    node._state_lock = ResetBetweenCallbackPhases()
    node._reset_monotonic_cutoff = 0.0
    node._target_barrier = barrier
    node._source_epoch_tracker = SourceTimestampEpochTracker()
    node._bridge = SimpleNamespace(
        imgmsg_to_cv2=lambda *_args, **_kwargs: np.zeros(
            (2, 2, 3),
            dtype=np.uint8,
        )
    )
    node._rgb_frames = collections.deque(maxlen=16)
    node._last_observed_rgb_stamp_ns = None
    node.get_logger = lambda: _Logger()
    message = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=1, nanosec=0),
        ),
    )
    monkeypatch.setattr(node_module.time, 'monotonic', lambda: 9.0)

    PerceptionNode._on_rgb(node, message)

    assert not node._rgb_frames
    assert node._last_observed_rgb_stamp_ns is None
    assert barrier.generation == 1


def test_node_reset_seeds_uninitialized_tracker_then_hard_rebases_low_epoch():
    acknowledgements = []
    node = SimpleNamespace(
        _state_lock=threading.RLock(),
        _reset_monotonic_cutoff=float('-inf'),
        _rgb_frames=collections.deque(maxlen=16),
        _depth_frames=collections.deque(maxlen=16),
        _last_observed_rgb_stamp_ns=100_000_000_000,
        _last_observed_depth_stamp_ns=100_010_000_000,
        _latest_by_label={'bolt': object()},
        _latest_target_received_at={'bolt': (1.0, 1.0)},
        _target_present_latest={'bolt': True},
        _recent_detections=collections.defaultdict(list),
        _recent_bolt_a=[object()],
        _recent_bolt_b=[object()],
        _latest_bolt_pair=object(),
        _bolt_pair_association=BoltPairAssociationGate(),
        _nut_association=SingleTargetAssociationGate(),
        _target_barrier=TargetFrameBarrier(required_frames=3),
        _frame_gate=CameraFrameGate(),
        _source_epoch_tracker=SourceTimestampEpochTracker(
            rollback_threshold_sec=5.0,
            max_pair_skew_sec=0.1,
            confirmation_pairs=2,
            quarantine_pairs=2,
        ),
        _on_demand_inference=True,
        _demand_capture_active=False,
        _demand_capture_epoch=10,
        _consecutive_stale_inferences=0,
        _total_stale_inferences=0,
        _last_inference_latency_sec=None,
        _last_stale_inference_reason=None,
        _reset_ack_pub=SimpleNamespace(
            publish=lambda _message: acknowledgements.append(True),
        ),
        get_logger=lambda: _Logger(),
    )

    PerceptionNode._on_reset_cache(node, object())

    assert acknowledgements == [True]
    assert node._target_barrier.generation == 1
    assert node._frame_gate.last_rgb_stamp_ns == 100_000_000_000
    assert node._frame_gate.last_depth_stamp_ns == 100_010_000_000
    assert node._demand_capture_active
    assert node._demand_capture_epoch == 11

    first_low_pair = node._source_epoch_tracker.observe_pair(
        2_000_000_000,
        2_010_000_000,
    )
    assert first_low_pair.pending
    assert not first_low_pair.rebase
    confirming_low_pair = node._source_epoch_tracker.observe_pair(
        2_100_000_000,
        2_110_000_000,
    )
    assert confirming_low_pair.rebase

    PerceptionNode._hard_reset_source_epoch_locked(
        node,
        confirming_low_pair.epoch,
    )
    generation = node._target_barrier.generation
    assert generation == 2

    accepted_stamps = (
        (2_100_000_000, 2_110_000_000),
        (2_200_000_000, 2_210_000_000),
        (2_300_000_000, 2_310_000_000),
    )
    for index, (rgb_stamp_ns, depth_stamp_ns) in enumerate(
        accepted_stamps,
        start=1,
    ):
        if index > 1:
            epoch_result = node._source_epoch_tracker.observe_pair(
                rgb_stamp_ns,
                depth_stamp_ns,
            )
            assert epoch_result.accept_pair
            assert not epoch_result.rebase
        gate_result = node._frame_gate.evaluate(
            rgb_stamp_ns=rgb_stamp_ns,
            depth_stamp_ns=depth_stamp_ns,
            rgb_received_at=9.5,
            depth_received_at=9.5,
            now=10.0,
            publisher_counts={
                'rgb': 1,
                'depth': 1,
                'camera_info': 1,
            },
        )
        assert gate_result.accepted
        assert PerceptionNode._record_target_frame(
            node,
            'bolt',
            rgb_stamp_ns,
            generation,
            9.5,
            9.5,
        )
        assert node._target_barrier.ready('bolt') is (index == 3)

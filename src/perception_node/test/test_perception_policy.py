"""Focused tests for camera freshness and target-selection policy."""

from types import SimpleNamespace

import pytest

from perception_node.perception_policy import (
    CameraFrameGate,
    TargetFrameBarrier,
    scale_roi,
    select_central_candidates,
    stamp_to_nanoseconds,
)


def _evaluate(gate, rgb_ns=1_000_000_000, depth_ns=1_000_000_000, **kwargs):
    return gate.evaluate(
        rgb_stamp_ns=rgb_ns,
        depth_stamp_ns=depth_ns,
        rgb_received_at=kwargs.pop('rgb_received_at', 10.0),
        depth_received_at=kwargs.pop('depth_received_at', 10.0),
        now=kwargs.pop('now', 10.5),
        publisher_counts=kwargs.pop(
            'publisher_counts',
            {'rgb': 1, 'depth': 1, 'camera_info': 1},
        ),
        **kwargs,
    )


def test_frame_gate_accepts_fresh_synchronized_unique_sources():
    gate = CameraFrameGate()

    result = _evaluate(
        gate,
        rgb_ns=1_000_000_000,
        depth_ns=1_100_000_000,
    )

    assert result.accepted


@pytest.mark.parametrize(
    ('overrides', 'reason'),
    [
        ({'publisher_counts': {'rgb': 2, 'depth': 1}}, 'publisher count'),
        ({'rgb_received_at': 8.9}, 'stale receipt'),
        ({'depth_ns': 1_100_000_001}, 'skew'),
    ],
)
def test_frame_gate_rejects_ambiguous_stale_or_skewed_inputs(overrides, reason):
    gate = CameraFrameGate()

    result = _evaluate(gate, **overrides)

    assert not result.accepted
    assert reason in result.reason


def test_frame_gate_requires_both_source_stamps_to_increase():
    gate = CameraFrameGate()
    assert _evaluate(gate).accepted

    duplicate_rgb = _evaluate(
        gate,
        rgb_ns=1_000_000_000,
        depth_ns=1_010_000_000,
    )
    assert not duplicate_rgb.accepted
    assert 'duplicate RGB' in duplicate_rgb.reason

    retrograde_depth = _evaluate(
        gate,
        rgb_ns=1_020_000_000,
        depth_ns=999_999_999,
    )
    assert not retrograde_depth.accepted
    assert 'retrograde depth' in retrograde_depth.reason

    assert _evaluate(
        gate,
        rgb_ns=1_020_000_000,
        depth_ns=1_020_000_000,
    ).accepted


def test_target_barrier_needs_three_strictly_increasing_frames_after_reset():
    barrier = TargetFrameBarrier(required_frames=3)

    assert not barrier.record('busbar', 10)
    assert not barrier.record('busbar', 10)
    assert barrier.count('busbar') == 1
    assert not barrier.record('busbar', 11)
    assert barrier.record('busbar', 12)
    assert barrier.ready('busbar')

    barrier.reset()

    assert not barrier.ready('busbar')
    assert barrier.count('busbar') == 0


def test_center_selection_uses_confidence_floor_then_image_distance():
    candidates = [
        {'score': 0.99, 'pixel': (10, 10), 'name': 'outer'},
        {'score': 0.61, 'pixel': (51, 49), 'name': 'center'},
        {'score': 0.59, 'pixel': (50, 50), 'name': 'below-threshold'},
    ]

    selected = select_central_candidates(candidates, (100, 100), limit=1)

    assert [candidate['name'] for candidate in selected] == ['center']


def test_bolt_selection_returns_two_confident_candidates_nearest_center():
    candidates = [
        {'score': 0.9, 'pixel': (50, 50), 'name': 'a'},
        {'score': 0.8, 'pixel': (55, 50), 'name': 'b'},
        {'score': 0.99, 'pixel': (90, 90), 'name': 'outer'},
    ]

    selected = select_central_candidates(candidates, (100, 100), limit=2)

    assert [candidate['name'] for candidate in selected] == ['a', 'b']


def test_central_roi_scales_with_live_image_resolution():
    assert scale_roi(
        image_size=(1280, 720),
        reference_size=(640, 640),
        reference_bounds=(150, 490, 180, 450),
    ) == (300, 980, 202, 506)


def test_stamp_conversion_rejects_invalid_nanoseconds():
    assert stamp_to_nanoseconds(SimpleNamespace(sec=2, nanosec=3)) == 2_000_000_003
    with pytest.raises(ValueError):
        stamp_to_nanoseconds(SimpleNamespace(sec=2, nanosec=1_000_000_000))

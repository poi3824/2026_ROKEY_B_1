"""Focused tests for camera freshness and target-selection policy."""

from types import SimpleNamespace

import pytest

from perception_node.perception_policy import (
    BoltPairAssociationGate,
    CameraFrameGate,
    DEFAULT_BOLT_PAIR_ASSOCIATION_MAX_ENDPOINT_M,
    TargetFrameBarrier,
    advance_source_boundary,
    associate_bolt_pair_endpoints,
    scale_roi,
    select_central_bolt_pair,
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


def test_frame_gate_reset_advances_to_last_callback_observed_boundary():
    gate = CameraFrameGate()
    assert _evaluate(
        gate,
        rgb_ns=1_000_000_000,
        depth_ns=1_000_000_000,
    ).accepted

    gate.reset(
        rgb_boundary_ns=1_050_000_000,
        depth_boundary_ns=1_040_000_000,
    )

    queued_before_reset = _evaluate(
        gate,
        rgb_ns=1_050_000_000,
        depth_ns=1_040_000_000,
    )
    assert not queued_before_reset.accepted
    assert 'duplicate RGB' in queued_before_reset.reason
    assert _evaluate(
        gate,
        rgb_ns=1_060_000_000,
        depth_ns=1_060_000_000,
    ).accepted


def test_frame_gate_reset_never_moves_an_accepted_boundary_backwards():
    gate = CameraFrameGate()
    assert _evaluate(
        gate,
        rgb_ns=2_000_000_000,
        depth_ns=2_000_000_000,
    ).accepted

    gate.reset(
        rgb_boundary_ns=1_000_000_000,
        depth_boundary_ns=1_000_000_000,
    )

    assert not _evaluate(
        gate,
        rgb_ns=1_500_000_000,
        depth_ns=1_500_000_000,
    ).accepted


def test_callback_observed_boundary_is_monotonic_across_queued_frames():
    boundary = advance_source_boundary(None, 10)
    boundary = advance_source_boundary(boundary, 6)
    boundary = advance_source_boundary(boundary, 12)

    assert boundary == 12
    with pytest.raises(ValueError):
        advance_source_boundary(boundary, -1)


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


def test_target_barrier_can_restart_one_label_only():
    barrier = TargetFrameBarrier(required_frames=3)
    barrier.record('bolt', 10)
    barrier.record('busbar', 20)

    barrier.reset_target('bolt')

    assert barrier.count('bolt') == 0
    assert barrier.count('busbar') == 1
    assert not barrier.record('bolt', 11)


def test_target_barrier_keeps_valid_frames_across_detection_dropout():
    barrier = TargetFrameBarrier(required_frames=3)

    assert not barrier.record('bolt', 10)
    assert not barrier.record('bolt', 11)
    # An accepted camera frame without a target neither advances nor resets
    # the post-reset target-frame barrier.
    assert barrier.count('bolt') == 2
    assert barrier.record('bolt', 13)


def test_center_selection_uses_confidence_floor_then_image_distance():
    candidates = [
        {'score': 0.99, 'pixel': (10, 10), 'name': 'outer'},
        {'score': 0.61, 'pixel': (51, 49), 'name': 'center'},
        {'score': 0.59, 'pixel': (50, 50), 'name': 'below-threshold'},
    ]

    selected = select_central_candidates(candidates, (100, 100), limit=1)

    assert [candidate['name'] for candidate in selected] == ['center']


def _bolt(name, pixel, world_point, score=0.8):
    return {
        'name': name,
        'score': score,
        'pixel': pixel,
        'world_point': world_point,
    }


def test_bolt_pair_rejects_cross_pack_nearest_individuals():
    candidates = [
        _bolt('target_a', (120, 200), (1.0552, 0.3722, 0.16)),
        _bolt('target_b', (280, 200), (1.26352, 0.009754, 0.16)),
        _bolt('neighbor_a', (198, 200), (1.0552, -0.2047, 0.16)),
        _bolt('neighbor_b', (390, 200), (1.26352, -0.567146, 0.16)),
    ]

    selected = select_central_bolt_pair(candidates, (400, 400))

    assert [candidate['name'] for candidate in selected] == [
        'target_a',
        'target_b',
    ]


def test_bolt_pair_applies_confidence_floor_before_pairing():
    candidates = [
        _bolt('center_a', (150, 200), (1.0552, 0.3722, 0.16)),
        _bolt(
            'center_b',
            (250, 200),
            (1.26352, 0.009754, 0.16),
            score=0.59,
        ),
        _bolt('outer_a', (250, 300), (1.0552, -0.2047, 0.16)),
        _bolt('outer_b', (350, 300), (1.26352, -0.567146, 0.16)),
    ]

    selected = select_central_bolt_pair(candidates, (400, 400))

    assert [candidate['name'] for candidate in selected] == [
        'outer_a',
        'outer_b',
    ]


def test_bolt_pair_returns_none_when_only_cross_pack_geometry_exists():
    candidates = [
        _bolt('pack_a', (198, 200), (1.0552, 0.3722, 0.16)),
        _bolt('pack_b', (202, 200), (1.26352, -0.567146, 0.16)),
    ]

    assert select_central_bolt_pair(candidates, (400, 400)) is None


def test_bolt_pair_preserves_signed_world_vector_relationship():
    wrong_sign_pair = [
        _bolt('bolt_a', (150, 200), (1.0552, 0.0, 0.16)),
        _bolt('bolt_b', (250, 200), (1.26352, 0.362446, 0.16)),
    ]

    assert select_central_bolt_pair(wrong_sign_pair, (400, 400)) is None


def test_bolt_pair_output_order_is_independent_of_detector_order():
    candidates = [
        _bolt('bolt_b', (280, 200), (1.26352, 0.009754, 0.16)),
        _bolt('bolt_a', (120, 200), (1.0552, 0.3722, 0.16)),
    ]

    forward = select_central_bolt_pair(candidates, (400, 400))
    reverse = select_central_bolt_pair(list(reversed(candidates)), (400, 400))

    assert [candidate['name'] for candidate in forward] == ['bolt_a', 'bolt_b']
    assert [candidate['name'] for candidate in reverse] == ['bolt_a', 'bolt_b']


def test_bolt_pair_association_detects_pack_switch_and_endpoint_order():
    station_3 = (
        (1.0552, 0.3722, 0.16),
        (1.26352, 0.009754, 0.16),
    )
    station_3_reversed_with_jitter = (
        (1.2640, 0.0102, 0.159),
        (1.0548, 0.3718, 0.161),
    )
    station_4 = (
        (1.0552, -0.2047, 0.16),
        (1.26352, -0.567146, 0.16),
    )

    jitter_distance, swap = associate_bolt_pair_endpoints(
        station_3,
        station_3_reversed_with_jitter,
    )
    switch_distance, switch_swap = associate_bolt_pair_endpoints(
        station_3,
        station_4,
    )

    assert swap
    assert jitter_distance < DEFAULT_BOLT_PAIR_ASSOCIATION_MAX_ENDPOINT_M
    assert not switch_swap
    assert switch_distance > DEFAULT_BOLT_PAIR_ASSOCIATION_MAX_ENDPOINT_M


def test_pack_switch_restarts_three_frame_barrier():
    barrier = TargetFrameBarrier(required_frames=3)
    station_3 = (
        (1.0552, 0.3722, 0.16),
        (1.26352, 0.009754, 0.16),
    )
    station_4 = (
        (1.0552, -0.2047, 0.16),
        (1.26352, -0.567146, 0.16),
    )

    previous = station_3
    assert not barrier.record('bolt', 10)
    assert not barrier.record('bolt', 11)

    endpoint_distance, _swap = associate_bolt_pair_endpoints(
        previous,
        station_4,
    )
    if endpoint_distance > DEFAULT_BOLT_PAIR_ASSOCIATION_MAX_ENDPOINT_M:
        barrier.reset_target('bolt')
    assert not barrier.record('bolt', 12)
    assert barrier.count('bolt') == 1

    previous = station_4
    for stamp in (13, 14):
        endpoint_distance, _swap = associate_bolt_pair_endpoints(
            previous,
            station_4,
        )
        assert endpoint_distance <= DEFAULT_BOLT_PAIR_ASSOCIATION_MAX_ENDPOINT_M
        barrier.record('bolt', stamp)
    assert barrier.ready('bolt')


def test_bolt_pair_association_uses_first_pair_as_fixed_anchor():
    gate = BoltPairAssociationGate(max_endpoint_distance_m=0.05)
    anchor = (
        (1.0552, 0.3722, 0.16),
        (1.26352, 0.009754, 0.16),
    )
    shifted_30_mm = tuple(
        (point[0], point[1] + 0.03, point[2])
        for point in anchor
    )
    shifted_60_mm = tuple(
        (point[0], point[1] + 0.06, point[2])
        for point in anchor
    )

    _ordered, restarted, distance = gate.record(anchor)
    assert not restarted
    assert distance == 0.0

    _ordered, restarted, distance = gate.record(shifted_30_mm)
    assert not restarted
    assert distance == pytest.approx(0.03)

    _ordered, restarted, distance = gate.record(shifted_60_mm)
    assert restarted
    assert distance == pytest.approx(0.06)


def test_bolt_pair_association_reset_requires_a_new_anchor():
    gate = BoltPairAssociationGate()
    pair = (
        (1.0552, 0.3722, 0.16),
        (1.26352, 0.009754, 0.16),
    )
    gate.record(pair)
    assert gate.active

    gate.reset()

    assert not gate.active
    _ordered, restarted, distance = gate.record(pair)
    assert not restarted
    assert distance == 0.0


def test_bolt_pair_ranks_ordered_bolt_2_before_midpoint_and_confidence():
    candidates = [
        _bolt(
            'target_a',
            (0, 200),
            (1.0552, 0.3722, 0.16),
            score=0.61,
        ),
        _bolt(
            'target_b',
            (200, 200),
            (1.26352, 0.009754, 0.16),
            score=0.61,
        ),
        _bolt(
            'midpoint_a',
            (140, 200),
            (1.0552, -0.2047, 0.16),
            score=0.99,
        ),
        _bolt(
            'midpoint_b',
            (260, 200),
            (1.26352, -0.567146, 0.16),
            score=0.99,
        ),
    ]

    selected = select_central_bolt_pair(candidates, (400, 400))

    assert [candidate['name'] for candidate in selected] == [
        'target_a',
        'target_b',
    ]


@pytest.mark.parametrize(
    'kwargs',
    [
        {'expected_delta_xy': (0.0, 0.0)},
        {'expected_delta_xy': (0.2, float('nan'))},
        {'x_tolerance_m': 0.0},
        {'y_tolerance_m': 0.0},
        {'max_z_delta_m': -0.01},
    ],
)
def test_bolt_pair_rejects_invalid_geometry_configuration(kwargs):
    with pytest.raises(ValueError):
        select_central_bolt_pair([], (400, 400), **kwargs)


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

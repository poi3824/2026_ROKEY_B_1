"""Focused tests for camera freshness and target-selection policy."""

from types import SimpleNamespace

import pytest

from perception_node.perception_policy import (
    BoltPairAssociationGate,
    CameraFrameGate,
    DEFAULT_BOLT_PAIR_ASSOCIATION_MAX_ENDPOINT_M,
    SingleTargetAssociationGate,
    SourceTimestampEpochTracker,
    TargetFrameBarrier,
    TimestampedFrame,
    advance_source_boundary,
    associate_bolt_pair_endpoints,
    frame_receipt_error,
    scale_roi,
    select_central_bolt_pair,
    select_central_candidates,
    select_newest_synchronized_pair,
    select_sticky_single_target,
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
        # The Isaac cameras can arrive a little more than one render tick
        # apart; the default permits this bounded 133 ms source skew.
        depth_ns=1_133_000_000,
    )

    assert result.accepted


@pytest.mark.parametrize(
    ('overrides', 'reason'),
    [
        ({'publisher_counts': {'rgb': 2, 'depth': 1}}, 'publisher count'),
        ({'rgb_received_at': 8.9}, 'stale receipt'),
        ({'depth_ns': 1_200_000_001}, 'skew'),
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


def _epoch_tracker_at_old_boundary():
    tracker = SourceTimestampEpochTracker(
        rollback_threshold_sec=5.0,
        max_pair_skew_sec=0.1,
        confirmation_pairs=2,
        quarantine_pairs=2,
    )
    initial = tracker.observe_pair(
        100_000_000_000,
        100_010_000_000,
    )
    assert initial.accept_pair
    return tracker


def test_source_epoch_tracker_rejects_small_out_of_order_without_rebase():
    tracker = _epoch_tracker_at_old_boundary()

    out_of_order = tracker.observe_pair(
        99_900_000_000,
        99_910_000_000,
    )

    assert not out_of_order.accept_pair
    assert not out_of_order.pending
    assert not out_of_order.rebase
    assert out_of_order.epoch == 0
    assert out_of_order.rgb_boundary_ns == 100_000_000_000
    recovered = tracker.observe_pair(
        100_100_000_000,
        100_110_000_000,
    )
    assert recovered.accept_pair
    assert not recovered.rebase


def test_source_epoch_tracker_needs_two_coherent_two_modality_pairs():
    tracker = _epoch_tracker_at_old_boundary()

    one_sided = tracker.observe_pair(
        2_000_000_000,
        100_020_000_000,
    )
    assert one_sided.pending
    assert not one_sided.rebase
    assert tracker.pending_confirmation_count == 0

    first_pair = tracker.observe_pair(
        2_000_000_000,
        2_010_000_000,
    )
    assert first_pair.pending
    assert not first_pair.accept_pair
    assert not first_pair.rebase
    assert tracker.pending_confirmation_count == 1

    duplicate_pair = tracker.observe_pair(
        2_000_000_000,
        2_010_000_000,
    )
    assert duplicate_pair.pending
    assert not duplicate_pair.rebase
    assert tracker.pending_confirmation_count == 1

    confirming_pair = tracker.observe_pair(
        2_100_000_000,
        2_110_000_000,
    )
    assert confirming_pair.accept_pair
    assert confirming_pair.rebase
    assert not confirming_pair.pending
    assert confirming_pair.epoch == 1
    assert confirming_pair.rgb_boundary_ns == 2_100_000_000
    assert confirming_pair.depth_boundary_ns == 2_110_000_000


def test_reset_seeded_tracker_rebases_low_epoch_before_frame_gate_deadlock():
    tracker = SourceTimestampEpochTracker(
        rollback_threshold_sec=5.0,
        max_pair_skew_sec=0.1,
        confirmation_pairs=2,
        quarantine_pairs=2,
    )
    gate = CameraFrameGate()
    barrier = TargetFrameBarrier(required_frames=3)
    old_rgb_boundary = 100_000_000_000
    old_depth_boundary = 100_010_000_000

    assert tracker.seed_reset_boundaries(
        old_rgb_boundary,
        old_depth_boundary,
    )
    gate.reset(
        rgb_boundary_ns=old_rgb_boundary,
        depth_boundary_ns=old_depth_boundary,
    )
    barrier.reset()

    first_low_pair = tracker.observe_pair(
        2_000_000_000,
        2_010_000_000,
    )
    assert first_low_pair.pending
    assert not first_low_pair.accept_pair
    assert not first_low_pair.rebase

    confirming_low_pair = tracker.observe_pair(
        2_100_000_000,
        2_110_000_000,
    )
    assert confirming_low_pair.accept_pair
    assert confirming_low_pair.rebase

    # Mirror the node's atomic hard-reset path. The confirming pair becomes
    # frame one in the new target generation, followed by two fresh pairs.
    barrier.reset()
    gate.reset(preserve_source_sequence=False)
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
            epoch_result = tracker.observe_pair(
                rgb_stamp_ns,
                depth_stamp_ns,
            )
            assert epoch_result.accept_pair
            assert not epoch_result.rebase
        gate_result = _evaluate(
            gate,
            rgb_ns=rgb_stamp_ns,
            depth_ns=depth_stamp_ns,
        )
        assert gate_result.accepted
        ready = barrier.record(
            'bolt',
            rgb_stamp_ns,
            generation=barrier.generation,
        )
        assert ready is (index == 3)

    assert barrier.ready('bolt')


def test_reset_seed_requires_both_modalities_and_keeps_rollback_strict():
    tracker = SourceTimestampEpochTracker(
        rollback_threshold_sec=5.0,
        max_pair_skew_sec=0.1,
        confirmation_pairs=2,
    )
    assert not tracker.seed_reset_boundaries(
        100_000_000_000,
        None,
    )
    initial = tracker.observe_pair(
        100_000_000_000,
        100_010_000_000,
    )
    assert initial.accept_pair

    small_rollback = tracker.observe_pair(
        99_900_000_000,
        99_910_000_000,
    )
    assert not small_rollback.accept_pair
    assert not small_rollback.pending
    assert not small_rollback.rebase

    one_sided_rollback = tracker.observe_pair(
        2_000_000_000,
        100_020_000_000,
    )
    assert not one_sided_rollback.accept_pair
    assert one_sided_rollback.pending
    assert not one_sided_rollback.rebase
    assert tracker.pending_confirmation_count == 0


def test_source_epoch_tracker_old_epoch_recovery_cancels_pending():
    tracker = _epoch_tracker_at_old_boundary()
    assert tracker.observe_pair(
        2_000_000_000,
        100_020_000_000,
    ).pending

    recovered = tracker.observe_pair(
        100_100_000_000,
        100_110_000_000,
    )

    assert recovered.accept_pair
    assert not recovered.pending
    assert not recovered.rebase
    assert 'pending canceled' in recovered.reason
    assert tracker.pending_confirmation_count == 0


def test_source_epoch_tracker_quarantines_delayed_old_high_stamps():
    tracker = _epoch_tracker_at_old_boundary()
    tracker.observe_pair(2_000_000_000, 2_010_000_000)
    rebased = tracker.observe_pair(2_100_000_000, 2_110_000_000)
    assert rebased.rebase
    assert tracker.quarantine_active
    assert tracker.stamp_is_quarantined(
        'RGB',
        99_900_000_000,
    )
    assert tracker.stamp_is_quarantined(
        'depth',
        99_910_000_000,
    )
    assert not tracker.stamp_is_quarantined(
        'RGB',
        2_200_000_000,
    )

    delayed_old_pair = tracker.observe_pair(
        99_900_000_000,
        99_910_000_000,
    )
    assert not delayed_old_pair.accept_pair
    assert delayed_old_pair.quarantined
    assert delayed_old_pair.epoch == 1
    assert delayed_old_pair.rgb_boundary_ns == 2_100_000_000

    one_sided_old_packet = tracker.observe_pair(
        99_920_000_000,
        2_120_000_000,
    )
    assert one_sided_old_packet.quarantined

    assert tracker.observe_pair(
        2_200_000_000,
        2_210_000_000,
    ).accept_pair
    assert tracker.quarantine_active
    assert tracker.observe_pair(
        2_300_000_000,
        2_310_000_000,
    ).accept_pair
    assert not tracker.quarantine_active
    assert not tracker.stamp_is_quarantined(
        'RGB',
        99_900_000_000,
    )


@pytest.mark.parametrize(
    'kwargs',
    [
        {'rollback_threshold_sec': 0.0},
        {'max_pair_skew_sec': 0.0},
        {'confirmation_pairs': 1},
        {'quarantine_pairs': 0},
    ],
)
def test_source_epoch_tracker_rejects_unsafe_configuration(kwargs):
    with pytest.raises(ValueError):
        SourceTimestampEpochTracker(**kwargs)


def _frame(stamp_ns, received_at, payload):
    return TimestampedFrame(
        stamp_ns=stamp_ns,
        received_at=received_at,
        payload=payload,
    )


def test_pair_selector_recovers_matching_frames_when_individual_latest_skew():
    rgb_frames = [
        _frame(1_000_000_000, 9.7, 'rgb-matched'),
        _frame(1_300_000_000, 9.9, 'rgb-latest'),
    ]
    depth_frames = [
        _frame(1_000_000_000, 9.8, 'depth-matched'),
        _frame(1_150_000_000, 9.95, 'depth-latest'),
    ]

    pair = select_newest_synchronized_pair(
        rgb_frames,
        depth_frames,
        max_skew_sec=0.1,
        max_receipt_age_sec=1.0,
        now=10.0,
    )

    assert pair is not None
    assert pair.rgb.payload == 'rgb-matched'
    assert pair.depth.payload == 'depth-matched'


def test_pair_selector_uses_newest_common_time_then_smallest_skew():
    rgb_frames = [
        _frame(1_000_000_000, 9.6, 'rgb-old'),
        _frame(1_200_000_000, 9.8, 'rgb-new'),
    ]
    depth_frames = [
        _frame(1_010_000_000, 9.65, 'depth-old'),
        _frame(1_190_000_000, 9.85, 'depth-new'),
        _frame(1_200_000_000, 9.9, 'depth-new-exact'),
    ]

    pair = select_newest_synchronized_pair(
        rgb_frames,
        depth_frames,
        max_skew_sec=0.1,
        max_receipt_age_sec=1.0,
        now=10.0,
    )

    assert pair is not None
    assert pair.rgb.payload == 'rgb-new'
    assert pair.depth.payload == 'depth-new-exact'


def test_pair_selector_keeps_strict_skew_age_and_reset_boundaries():
    exact_limit_pair = select_newest_synchronized_pair(
        [_frame(1_000_000_000, 9.5, 'rgb')],
        [_frame(1_100_000_000, 9.5, 'depth')],
        max_skew_sec=0.1,
        max_receipt_age_sec=1.0,
        now=10.0,
    )
    assert exact_limit_pair is not None

    assert select_newest_synchronized_pair(
        [_frame(1_000_000_000, 9.5, 'rgb')],
        [_frame(1_100_000_001, 9.5, 'depth')],
        max_skew_sec=0.1,
        max_receipt_age_sec=1.0,
        now=10.0,
    ) is None
    assert select_newest_synchronized_pair(
        [_frame(2_000_000_000, 8.9, 'stale-rgb')],
        [_frame(2_000_000_000, 9.5, 'depth')],
        max_skew_sec=0.1,
        max_receipt_age_sec=1.0,
        now=10.0,
    ) is None
    assert select_newest_synchronized_pair(
        [_frame(3_000_000_000, 9.5, 'pre-reset-rgb')],
        [_frame(3_000_000_000, 9.5, 'pre-reset-depth')],
        max_skew_sec=0.1,
        max_receipt_age_sec=1.0,
        now=10.0,
        rgb_after_ns=3_000_000_000,
        depth_after_ns=3_000_000_000,
    ) is None


def test_receipt_freshness_recheck_uses_both_actual_callback_times():
    assert frame_receipt_error(
        rgb_received_at=9.0,
        depth_received_at=9.2,
        now=10.0,
        max_receipt_age_sec=1.0,
    ) is None
    assert 'stale receipt' in frame_receipt_error(
        rgb_received_at=8.999,
        depth_received_at=9.9,
        now=10.0,
        max_receipt_age_sec=1.0,
    )
    assert 'moved backwards' in frame_receipt_error(
        rgb_received_at=10.1,
        depth_received_at=9.9,
        now=10.0,
        max_receipt_age_sec=1.0,
    )


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


def test_target_barrier_rejects_inflight_frames_from_previous_reset_generation():
    barrier = TargetFrameBarrier(required_frames=3)
    stale_generation = barrier.generation

    assert not barrier.record(
        'bolt',
        10,
        generation=stale_generation,
    )
    barrier.reset()

    assert barrier.generation == stale_generation + 1
    assert not barrier.record(
        'bolt',
        11,
        generation=stale_generation,
    )
    assert barrier.count('bolt') == 0

    current_generation = barrier.generation
    assert not barrier.record(
        'bolt',
        20,
        generation=current_generation,
    )
    assert not barrier.record(
        'bolt',
        20,
        generation=current_generation,
    )
    assert barrier.count('bolt') == 1
    assert not barrier.record(
        'bolt',
        21,
        generation=current_generation,
    )
    assert barrier.record(
        'bolt',
        22,
        generation=current_generation,
    )


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


def _nut(name, pixel, world_point, score=0.8):
    return {
        'name': name,
        'score': score,
        'pixel': pixel,
        'world_point': world_point,
    }


def test_single_target_initial_selection_is_central_after_confidence_floor():
    candidates = [
        _nut('outer-high', (10, 10), (0.0, 0.0, 0.1), score=0.99),
        _nut('center', (51, 49), (0.1, 0.0, 0.1), score=0.60),
        _nut('below-floor', (50, 50), (0.2, 0.0, 0.1), score=0.59),
    ]

    selected = select_sticky_single_target(candidates, (100, 100))

    assert selected['name'] == 'center'


def test_single_target_selection_stays_on_world_anchor():
    gate = SingleTargetAssociationGate(max_distance_m=0.05)
    associated, distance = gate.record((1.0, 2.0, 0.1))
    assert associated
    assert distance == 0.0

    candidates = [
        _nut('same-nut', (10, 10), (1.01, 2.0, 0.1), score=0.61),
        _nut('other-central', (50, 50), (1.2, 2.0, 0.1), score=0.99),
    ]
    selected = select_sticky_single_target(
        candidates,
        (100, 100),
        anchor_point=gate.anchor,
        max_anchor_distance_m=gate.max_distance_m,
    )
    reversed_selected = select_sticky_single_target(
        list(reversed(candidates)),
        (100, 100),
        anchor_point=gate.anchor,
        max_anchor_distance_m=gate.max_distance_m,
    )

    assert selected['name'] == 'same-nut'
    assert reversed_selected['name'] == 'same-nut'
    associated, distance = gate.record(selected['world_point'])
    assert associated
    assert distance == pytest.approx(0.01)


def test_single_target_requires_three_misses_before_reselection():
    gate = SingleTargetAssociationGate(
        max_distance_m=0.05,
        miss_limit=3,
    )
    gate.record((1.0, 2.0, 0.1))
    far_candidate = _nut(
        'other-nut',
        (50, 50),
        (1.2, 2.0, 0.1),
    )

    assert select_sticky_single_target(
        [far_candidate],
        (100, 100),
        anchor_point=gate.anchor,
        max_anchor_distance_m=gate.max_distance_m,
    ) is None
    associated, distance = gate.record(far_candidate['world_point'])
    assert not associated
    assert distance == pytest.approx(0.2)
    assert not gate.record_miss()
    assert not gate.record_miss()
    assert gate.active
    assert gate.consecutive_misses == 2
    assert gate.record_miss()
    assert not gate.active

    selected = select_sticky_single_target(
        [far_candidate],
        (100, 100),
    )
    assert selected['name'] == 'other-nut'
    assert gate.record(selected['world_point']) == (True, 0.0)


def test_single_target_return_resets_miss_run_and_reset_clears_anchor():
    gate = SingleTargetAssociationGate(miss_limit=3)
    gate.record((1.0, 2.0, 0.1))
    assert not gate.record_miss()
    assert gate.consecutive_misses == 1

    associated, _distance = gate.record((1.01, 2.0, 0.1))
    assert associated
    assert gate.consecutive_misses == 0

    gate.reset()
    assert not gate.active
    assert gate.anchor is None
    assert gate.consecutive_misses == 0


def test_single_target_selector_ignores_malformed_or_nonfinite_candidates():
    candidates = [
        _nut('nan-world', (50, 50), (float('nan'), 0.0, 0.1)),
        _nut('bad-pixel', (50,), (0.0, 0.0, 0.1)),
        _nut('valid', (60, 50), (0.0, 0.0, 0.1)),
    ]

    selected = select_sticky_single_target(candidates, (100, 100))

    assert selected['name'] == 'valid'


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


def test_first_bolt_anchor_has_no_station_prior_then_stays_world_sticky():
    station_5 = (
        (1.0552, -0.7312, 0.16),
        (1.26352, -1.093646, 0.16),
    )
    candidates = [
        _bolt('anchor_a', (70, 200), station_5[0]),
        _bolt('anchor_b', (110, 200), station_5[1]),
        _bolt('neighbour_a', (160, 200), (1.0552, -0.2047, 0.16)),
        _bolt(
            'neighbour_b',
            (200, 200),
            (1.26352, -0.567146, 0.16),
        ),
    ]

    # With no station input, the first selection can only use camera center.
    initial = select_central_bolt_pair(candidates, (400, 400))
    sticky = select_central_bolt_pair(
        candidates,
        (400, 400),
        anchor_pair=station_5,
    )

    assert [candidate['name'] for candidate in initial] == [
        'neighbour_a',
        'neighbour_b',
    ]
    assert [candidate['name'] for candidate in sticky] == [
        'anchor_a',
        'anchor_b',
    ]


def test_anchor_requires_consecutive_misses_before_reselection():
    gate = BoltPairAssociationGate(
        max_endpoint_distance_m=0.05,
        miss_limit=3,
    )
    anchor = (
        (1.0552, -0.7312, 0.16),
        (1.26352, -1.093646, 0.16),
    )
    gate.record(anchor)

    assert not gate.record_miss()
    assert not gate.record_miss()
    assert gate.active
    assert gate.consecutive_misses == 2

    # One good anchored frame cancels the transient-miss run.
    gate.record(anchor)
    assert gate.consecutive_misses == 0
    assert not gate.record_miss()
    assert not gate.record_miss()
    assert gate.record_miss()
    assert not gate.active


def test_one_neighbour_pack_frame_does_not_restart_three_frame_barrier():
    gate = BoltPairAssociationGate(miss_limit=3)
    barrier = TargetFrameBarrier(required_frames=3)
    anchor = (
        (1.0552, -0.7312, 0.16),
        (1.26352, -1.093646, 0.16),
    )
    gate.record(anchor)
    assert not barrier.record('bolt', 10)
    assert not barrier.record('bolt', 11)

    # A frame containing only another pack is a sticky-anchor miss, not a
    # target switch and not a barrier reset.
    assert not gate.record_miss()
    assert barrier.count('bolt') == 2

    gate.record(anchor)
    assert barrier.record('bolt', 13)
    assert barrier.ready('bolt')


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

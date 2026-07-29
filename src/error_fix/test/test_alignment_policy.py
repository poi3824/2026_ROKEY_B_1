"""Unit tests for the ROS-free fine-alignment convergence policy."""

import math

import pytest

from error_fix.alignment_policy import (
    FineAlignmentGate,
    FreshFramePairGate,
    LatestFramePairSynchronizer,
    advance_source_boundary,
)


def _submit_window(gate, start_stamp, observations):
    decision = None
    for index, (dx_px, dy_px, dtheta_deg) in enumerate(observations):
        assert gate.submit(
            start_stamp + index,
            valid=True,
            dx_px=dx_px,
            dy_px=dy_px,
            dtheta_deg=dtheta_deg,
        )
        decision = gate.consume()
        if index < len(observations) - 1:
            assert decision is None
    assert decision is not None
    return decision


def test_three_fresh_target_frames_are_required_per_decision():
    gate = FineAlignmentGate()

    assert gate.submit(1, valid=True, dx_px=1)
    assert gate.consume() is None
    assert gate.submit(2, valid=True, dx_px=-1)
    assert gate.consume() is None
    assert gate.submit(3, valid=True, dx_px=0)

    decision = gate.consume()
    assert decision is not None
    assert decision.stamp_ns == 3
    assert decision.frame_count == 3
    assert decision.dx_px == pytest.approx(0.0)
    assert decision.aligned
    assert decision.hold_count == 3


def test_plus_minus_two_pixel_quantization_holds_for_thirty_target_frames():
    gate = FineAlignmentGate()
    pattern = [
        (-2, 2, 3),
        (2, -2, -3),
        (-2, 2, 0),
    ]

    decisions = []
    for start_stamp in range(1, 31, 3):
        decisions.append(_submit_window(gate, start_stamp, pattern))

    assert [decision.hold_count for decision in decisions] == list(
        range(3, 31, 3))
    assert all(decision.aligned for decision in decisions)
    assert all(decision.correction_x_m == 0.0 for decision in decisions)
    assert all(decision.correction_y_m == 0.0 for decision in decisions)
    assert not any(decision.success for decision in decisions[:-1])
    assert decisions[-1].success
    assert gate.completed


@pytest.mark.parametrize(
    ('dx_px', 'dy_px', 'expected_x_m', 'expected_y_m'),
    [
        (3, 0, 0.0, -0.0002),
        (-3, 0, 0.0, 0.0002),
        (0, 3, -0.0002, 0.0),
        (0, -3, 0.0002, 0.0),
        (3, -3, 0.0002, -0.0002),
    ],
)
def test_consistent_error_gets_one_signed_fixed_step(
    dx_px,
    dy_px,
    expected_x_m,
    expected_y_m,
):
    gate = FineAlignmentGate()
    decision = _submit_window(
        gate,
        1,
        [(dx_px, dy_px, 0.0)] * 3,
    )

    assert decision.consensus
    assert not decision.aligned
    assert decision.correction_x_m == pytest.approx(expected_x_m)
    assert decision.correction_y_m == pytest.approx(expected_y_m)
    assert decision.hold_count == 0
    assert gate.waiting_for_settle
    assert gate.pending_correction_stamp_ns == 3
    assert not decision.deferred


def test_mixed_near_boundary_window_requires_a_second_fresh_window_before_move():
    gate = FineAlignmentGate()

    strict = _submit_window(gate, 1, [(-2, 0, 0)] * 3)
    assert strict.aligned
    assert strict.hold_count == 3

    # This reproduces the live Hough-centre jitter: one centred sample plus
    # two near-boundary samples.  It must neither become a success sample nor
    # reset earned strict evidence or cause a blind 0.2 mm move.
    first = _submit_window(
        gate,
        4,
        [(-2, 0, 0), (-4, 0, 0), (-6, 0, 0)],
    )
    assert first.consensus
    assert not first.aligned
    assert first.deferred
    assert first.correction_x_m == 0.0
    assert first.correction_y_m == 0.0
    assert first.hold_count == 3
    assert not gate.waiting_for_settle

    # The same signed error in a wholly new strict-frame window is now real
    # enough to receive the original fixed correction.
    confirmed = _submit_window(
        gate,
        7,
        [(-2, 0, 0), (-4, 0, 0), (-6, 0, 0)],
    )
    assert not confirmed.deferred
    assert confirmed.correction_y_m == pytest.approx(0.0002)
    assert confirmed.hold_count == 0
    assert gate.waiting_for_settle


def test_one_mixed_near_boundary_window_does_not_falsely_advance_hold():
    gate = FineAlignmentGate(hold_frames=6)

    first_strict = _submit_window(gate, 1, [(0, 2, 0)] * 3)
    assert first_strict.hold_count == 3
    assert not first_strict.success

    jitter = _submit_window(
        gate,
        4,
        [(0, -2, 0), (0, -4, 0), (0, -6, 0)],
    )
    assert jitter.deferred
    assert jitter.hold_count == 3
    assert not jitter.success

    final_strict = _submit_window(gate, 7, [(0, -2, 0)] * 3)
    assert final_strict.aligned
    assert final_strict.hold_count == 6
    assert final_strict.success


def test_one_near_boundary_outlier_counts_only_strict_frames_and_never_moves():
    gate = FineAlignmentGate(hold_frames=8)

    initial = _submit_window(gate, 1, [(-2, -2, 1.0)] * 3)
    assert initial.aligned
    assert initial.hold_count == 3

    # One integer-Hough excursion outside ±2 px must not discard the five
    # strict source stamps collected so far.  Repeating the same shape must
    # still never turn that lone outlier into a blind 0.2 mm correction.
    first_jitter = _submit_window(
        gate,
        4,
        [(-2, -2, 1.0), (-3, -2, 1.6), (-2, -2, 0.0)],
    )
    assert first_jitter.deferred
    assert not first_jitter.aligned
    assert not first_jitter.success
    assert first_jitter.hold_count == 5
    assert first_jitter.correction_x_m == 0.0
    assert first_jitter.correction_y_m == 0.0
    assert not gate.waiting_for_settle

    second_jitter = _submit_window(
        gate,
        7,
        [(-2, -2, 0.5), (-3, -2, 1.0), (-2, -2, 1.6)],
    )
    assert second_jitter.deferred
    assert not second_jitter.success
    assert second_jitter.hold_count == 7
    assert second_jitter.correction_x_m == 0.0
    assert second_jitter.correction_y_m == 0.0
    assert not gate.waiting_for_settle

    # Completion is still emitted only by a wholly strict fresh window.
    final_strict = _submit_window(gate, 10, [(-2, -2, 1.6)] * 3)
    assert final_strict.aligned
    assert final_strict.hold_count == 8
    assert final_strict.success


@pytest.mark.parametrize(
    'observations',
    [
        # Two position excursions are not the narrow lone-outlier case.
        [(-3, -2, 0.0), (-3, -2, 0.0), (-2, -2, 0.0)],
        # A large position excursion is not Hough boundary quantization.
        [(-2, -2, 0.0), (-7, -2, 0.0), (-2, -2, 0.0)],
        # Angle violations retain the normal correction/reset policy.
        [(-2, -2, 0.0), (-3, -2, 4.0), (-2, -2, 0.0)],
    ],
)
def test_lone_outlier_exception_remains_narrow(observations):
    gate = FineAlignmentGate()

    decision = _submit_window(gate, 1, observations)

    assert decision.hold_count == 0
    assert not decision.success


def test_only_one_nonzero_correction_is_available_before_settle_ack():
    gate = FineAlignmentGate()
    first = _submit_window(gate, 10, [(3, 0, 0)] * 3)

    assert first.correction_y_m == pytest.approx(-0.0002)
    for stamp_ns in range(13, 30):
        assert not gate.submit(
            stamp_ns,
            valid=True,
            dx_px=3,
            dy_px=0,
        )
        assert gate.consume() is None
    assert gate.pending_correction_stamp_ns == 12
    assert gate.observation_count == 0


def test_stale_ack_is_ignored_and_exact_ack_releases_pending_step():
    gate = FineAlignmentGate()
    _submit_window(gate, 100, [(3, 0, 0)] * 3)

    assert not gate.acknowledge_settled(101)
    assert gate.waiting_for_settle
    assert not gate.acknowledge_settled(103)
    assert gate.waiting_for_settle
    assert gate.acknowledge_settled(102)
    assert not gate.waiting_for_settle
    assert not gate.acknowledge_settled(102)


def test_ack_requires_three_new_post_motion_observations():
    gate = FineAlignmentGate()
    _submit_window(gate, 1, [(3, 0, 0)] * 3)

    # These frames arrived while the correction was in flight and are ignored.
    assert not gate.submit(4, valid=True)
    assert not gate.submit(5, valid=True)
    assert gate.acknowledge_settled(3)

    # Their stamps remain behind the strict ordering boundary.
    assert not gate.submit(5, valid=True)
    assert gate.submit(6, valid=True)
    assert gate.consume() is None
    assert gate.submit(7, valid=True)
    assert gate.consume() is None
    assert gate.submit(8, valid=True)
    post_motion = gate.consume()

    assert post_motion is not None
    assert post_motion.stamp_ns == 8
    assert post_motion.frame_count == 3
    assert post_motion.hold_count == 3


def test_dropouts_do_not_count_or_erase_valid_target_observations():
    gate = FineAlignmentGate()

    assert gate.submit(1, valid=True, dx_px=1)
    assert gate.observation_count == 1
    assert gate.submit(2, valid=False)
    assert gate.observation_count == 1
    assert gate.submit(3, valid=True, dx_px=math.nan)
    assert gate.observation_count == 1
    assert gate.submit(4, valid=True, dx_px=-1)
    assert gate.observation_count == 2
    assert gate.submit(5, valid=False)
    assert gate.observation_count == 2
    assert gate.submit(6, valid=True, dx_px=0)

    decision = gate.consume()
    assert decision is not None
    assert decision.frame_count == 3
    assert decision.hold_count == 3


def test_inconsistent_three_frame_error_does_not_move_or_make_progress():
    gate = FineAlignmentGate()
    decision = _submit_window(
        gate,
        1,
        [(4, 0, 0), (-4, 0, 0), (4, 0, 0)],
    )

    assert not decision.consensus
    assert not decision.aligned
    assert decision.correction_x_m == 0.0
    assert decision.correction_y_m == 0.0
    assert decision.hold_count == 0
    assert not gate.waiting_for_settle


def test_yaw_correction_waits_only_outside_three_degree_threshold():
    gate = FineAlignmentGate()
    at_threshold = _submit_window(gate, 1, [(0, 0, 3.0)] * 3)

    assert at_threshold.aligned
    assert not gate.waiting_for_settle

    gate.reset()
    over_threshold = _submit_window(gate, 10, [(0, 0, 3.1)] * 3)

    assert over_threshold.consensus
    assert not over_threshold.aligned
    assert over_threshold.dtheta_deg == pytest.approx(3.1)
    assert gate.waiting_for_settle
    assert gate.pending_correction_stamp_ns == 12


def test_duplicate_retrograde_and_zero_stamps_never_count():
    gate = FineAlignmentGate()

    with pytest.raises(ValueError, match='positive'):
        gate.submit(0, valid=True)
    assert gate.submit(10, valid=True)
    assert not gate.submit(10, valid=True)
    assert not gate.submit(9, valid=True)
    assert gate.observation_count == 1
    assert gate.submit(11, valid=True)
    assert gate.consume() is None
    assert gate.submit(12, valid=True)
    assert gate.consume() is not None


def test_frame_pair_gate_enforces_skew_age_and_strict_stream_order():
    gate = FreshFramePairGate(
        max_skew_sec=0.2,
        max_receipt_age_sec=1.0,
    )

    accepted, reason = gate.accept(
        rgb_stamp_ns=1_000_000_000,
        depth_stamp_ns=950_000_000,
        rgb_receipt_ns=5_000_000_000,
        depth_receipt_ns=4_900_000_000,
        now_receipt_ns=5_000_000_000,
    )
    assert accepted
    assert reason == ''

    duplicate_depth, reason = gate.accept(
        rgb_stamp_ns=1_100_000_000,
        depth_stamp_ns=950_000_000,
        rgb_receipt_ns=5_100_000_000,
        depth_receipt_ns=5_050_000_000,
        now_receipt_ns=5_100_000_000,
    )
    assert not duplicate_depth
    assert reason == 'duplicate/retrograde depth stamp'

    skewed, reason = gate.accept(
        rgb_stamp_ns=1_300_000_000,
        depth_stamp_ns=1_099_999_999,
        rgb_receipt_ns=5_200_000_000,
        depth_receipt_ns=5_150_000_000,
        now_receipt_ns=5_200_000_000,
    )
    assert not skewed
    assert reason == 'RGB/depth skew'

    stale, reason = gate.accept(
        rgb_stamp_ns=1_200_000_000,
        depth_stamp_ns=1_150_000_000,
        rgb_receipt_ns=5_300_000_000,
        depth_receipt_ns=4_000_000_000,
        now_receipt_ns=5_300_000_000,
    )
    assert not stale
    assert reason == 'stale receipt'


def test_frame_pair_reset_rejects_pre_move_replay():
    gate = FreshFramePairGate()
    gate.reset(
        rgb_boundary_ns=2_000_000_000,
        depth_boundary_ns=1_990_000_000,
    )

    accepted, reason = gate.accept(
        rgb_stamp_ns=2_000_000_000,
        depth_stamp_ns=2_010_000_000,
        rgb_receipt_ns=10,
        depth_receipt_ns=10,
        now_receipt_ns=10,
    )
    assert not accepted
    assert reason == 'duplicate/retrograde RGB stamp'

    accepted, reason = gate.accept(
        rgb_stamp_ns=2_100_000_000,
        depth_stamp_ns=2_090_000_000,
        rgb_receipt_ns=20,
        depth_receipt_ns=20,
        now_receipt_ns=20,
    )
    assert accepted
    assert reason == ''


@pytest.mark.parametrize('callback_order', [('rgb', 'depth'), ('depth', 'rgb')])
def test_latest_cache_accepts_three_pairs_in_either_callback_order(
    callback_order,
):
    synchronizer = LatestFramePairSynchronizer()

    for index in range(1, 4):
        stamp_ns = index * 1_000_000_000
        receipt_ns = 10_000_000_000 + index * 10_000_000
        payloads = {
            'rgb': f'rgb-{index}',
            'depth': f'depth-{index}',
        }
        cache = {
            'rgb': synchronizer.cache_rgb,
            'depth': synchronizer.cache_depth,
        }

        first, second = callback_order
        cache[first](payloads[first], stamp_ns, receipt_ns)
        pair, reason = synchronizer.try_accept(
            now_receipt_ns=receipt_ns,
        )
        assert pair is None
        assert reason == f'{second} unavailable'.replace(
            'rgb', 'RGB')

        cache[second](payloads[second], stamp_ns, receipt_ns)
        pair, reason = synchronizer.try_accept(
            now_receipt_ns=receipt_ns,
        )
        assert reason == ''
        assert pair.rgb.payload == payloads['rgb']
        assert pair.depth.payload == payloads['depth']
        assert pair.rgb.stamp_ns == stamp_ns
        assert pair.depth.stamp_ns == stamp_ns


def test_history_recovers_match_hidden_by_newer_rgb_and_retains_future_frame():
    synchronizer = LatestFramePairSynchronizer(history_size=5)
    receipt_ns = 15_000_000_000

    synchronizer.cache_rgb(
        'rgb-match',
        2_000_000_000,
        receipt_ns,
    )
    pair, reason = synchronizer.try_accept(now_receipt_ns=receipt_ns)
    assert pair is None
    assert reason == 'depth unavailable'

    synchronizer.cache_rgb(
        'rgb-future',
        2_300_000_000,
        receipt_ns,
    )
    pair, reason = synchronizer.try_accept(now_receipt_ns=receipt_ns)
    assert pair is None
    assert reason == 'depth unavailable'

    synchronizer.cache_depth(
        'depth-match',
        2_050_000_000,
        receipt_ns,
    )
    pair, reason = synchronizer.try_accept(now_receipt_ns=receipt_ns)
    assert reason == ''
    assert pair.rgb.payload == 'rgb-match'
    assert pair.depth.payload == 'depth-match'

    synchronizer.cache_depth(
        'depth-future',
        2_310_000_000,
        receipt_ns,
    )
    pair, reason = synchronizer.try_accept(now_receipt_ns=receipt_ns)
    assert reason == ''
    assert pair.rgb.payload == 'rgb-future'
    assert pair.depth.payload == 'depth-future'


def test_history_skips_stale_closest_pair_for_fresh_valid_pair():
    synchronizer = LatestFramePairSynchronizer(
        max_skew_sec=0.1,
        max_receipt_age_sec=1.0,
    )
    now_ns = 50_000_000_000

    synchronizer.cache_rgb(
        'rgb-stale-closest',
        3_000_000_000,
        now_ns - 2_000_000_000,
    )
    synchronizer.cache_rgb(
        'rgb-fresh',
        3_080_000_000,
        now_ns,
    )
    synchronizer.cache_depth(
        'depth-fresh',
        3_010_000_000,
        now_ns,
    )

    pair, reason = synchronizer.try_accept(now_receipt_ns=now_ns)

    assert reason == ''
    assert pair.rgb.payload == 'rgb-fresh'
    assert pair.depth.payload == 'depth-fresh'


def test_history_size_is_bounded_and_evicts_oldest_distinct_stamp():
    synchronizer = LatestFramePairSynchronizer(history_size=2)
    receipt_ns = 18_000_000_000

    synchronizer.cache_rgb('rgb-1', 1_000_000_000, receipt_ns)
    synchronizer.cache_rgb('rgb-2', 2_000_000_000, receipt_ns)
    synchronizer.cache_rgb('rgb-3', 3_000_000_000, receipt_ns)
    synchronizer.cache_depth('depth-1', 1_000_000_000, receipt_ns)

    evicted, reason = synchronizer.try_accept(now_receipt_ns=receipt_ns)
    assert evicted is None
    assert reason == 'RGB/depth skew'

    synchronizer.cache_depth('depth-3', 3_000_000_000, receipt_ns)
    pair, reason = synchronizer.try_accept(now_receipt_ns=receipt_ns)
    assert reason == ''
    assert pair.rgb.payload == 'rgb-3'
    assert pair.depth.payload == 'depth-3'


def test_ten_frame_backlog_retains_the_matching_counterpart():
    """A callback backlog must not lose a strict source-stamp match."""
    synchronizer = LatestFramePairSynchronizer(history_size=10)
    receipt_ns = 25_000_000_000

    # Ten RGB callbacks can run before their queued depth counterpart gets a
    # turn.  The exact first RGB match remains in the configured history.
    for index in range(10):
        stamp_ns = 5_000_000_000 + index * 10_000_000
        synchronizer.cache_rgb(
            f'rgb-{index}',
            stamp_ns,
            receipt_ns,
        )
    synchronizer.cache_depth(
        'depth-match',
        5_000_000_000,
        receipt_ns,
    )

    pair, reason = synchronizer.try_accept(now_receipt_ns=receipt_ns)

    assert reason == ''
    assert pair.rgb.payload == 'rgb-0'
    assert pair.depth.payload == 'depth-match'


@pytest.mark.parametrize('history_size', [0, -1, 1.5, True])
def test_history_size_must_be_a_positive_integer(history_size):
    with pytest.raises(ValueError, match='positive integer'):
        LatestFramePairSynchronizer(history_size=history_size)


def test_rejected_old_counterpart_is_retried_with_the_same_new_rgb():
    synchronizer = LatestFramePairSynchronizer()
    receipt_ns = 20_000_000_000

    synchronizer.cache_rgb('rgb-1', 1_000_000_000, receipt_ns)
    synchronizer.cache_depth('depth-1', 1_000_000_000, receipt_ns)
    first, _ = synchronizer.try_accept(now_receipt_ns=receipt_ns)
    assert first is not None

    synchronizer.cache_rgb('rgb-2', 2_000_000_000, receipt_ns)
    synchronizer.cache_depth('old-depth-1', 1_000_000_000, receipt_ns)
    rejected, reason = synchronizer.try_accept(
        now_receipt_ns=receipt_ns,
    )
    assert rejected is None
    assert reason == 'duplicate/retrograde depth stamp'

    synchronizer.cache_depth('depth-2', 2_000_000_000, receipt_ns)
    retried, reason = synchronizer.try_accept(
        now_receipt_ns=receipt_ns,
    )
    assert reason == ''
    assert retried.rgb.payload == 'rgb-2'
    assert retried.depth.payload == 'depth-2'


def test_accepted_pair_is_cleared_and_cannot_be_reused():
    synchronizer = LatestFramePairSynchronizer()
    receipt_ns = 30_000_000_000

    synchronizer.cache_rgb('rgb-1', 1_000_000_000, receipt_ns)
    synchronizer.cache_depth('depth-1', 1_000_000_000, receipt_ns)
    accepted, _ = synchronizer.try_accept(now_receipt_ns=receipt_ns)
    assert accepted is not None

    reused, reason = synchronizer.try_accept(now_receipt_ns=receipt_ns)
    assert reused is None
    assert reason == 'RGB unavailable'

    synchronizer.cache_rgb('old-rgb-1', 1_000_000_000, receipt_ns)
    synchronizer.cache_depth('depth-2', 2_000_000_000, receipt_ns)
    reused, reason = synchronizer.try_accept(now_receipt_ns=receipt_ns)
    assert reused is None
    assert reason == 'duplicate/retrograde RGB stamp'

    synchronizer.cache_rgb('rgb-2', 2_000_000_000, receipt_ns)
    next_pair, reason = synchronizer.try_accept(
        now_receipt_ns=receipt_ns,
    )
    assert reason == ''
    assert next_pair.rgb.payload == 'rgb-2'
    assert next_pair.depth.payload == 'depth-2'


def test_synchronizer_reset_clears_cache_and_enforces_both_boundaries():
    synchronizer = LatestFramePairSynchronizer()
    receipt_ns = 40_000_000_000

    synchronizer.cache_rgb('queued-rgb', 19, receipt_ns)
    synchronizer.cache_depth('queued-depth', 19, receipt_ns)
    synchronizer.reset(rgb_boundary_ns=20, depth_boundary_ns=20)

    pair, reason = synchronizer.try_accept(now_receipt_ns=receipt_ns)
    assert pair is None
    assert reason == 'RGB unavailable'

    synchronizer.cache_rgb('boundary-rgb', 20, receipt_ns)
    synchronizer.cache_depth('new-depth', 21, receipt_ns)
    pair, reason = synchronizer.try_accept(now_receipt_ns=receipt_ns)
    assert pair is None
    assert reason == 'duplicate/retrograde RGB stamp'

    synchronizer.cache_rgb('new-rgb', 21, receipt_ns)
    pair, reason = synchronizer.try_accept(now_receipt_ns=receipt_ns)
    assert reason == ''
    assert pair.rgb.payload == 'new-rgb'
    assert pair.depth.payload == 'new-depth'


def test_error_fix_callback_boundary_never_regresses_or_clears_on_zero():
    boundary = advance_source_boundary(None, 10)
    boundary = advance_source_boundary(boundary, 6)
    boundary = advance_source_boundary(boundary, 0)
    boundary = advance_source_boundary(boundary, 12)

    assert boundary == 12

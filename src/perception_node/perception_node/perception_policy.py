"""Deterministic input and target-selection policy for camera perception."""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


MIN_TARGET_CONFIDENCE = 0.6
DEFAULT_BOLT_PAIR_DX_M = 0.20832
DEFAULT_BOLT_PAIR_DY_M = -0.362446
DEFAULT_BOLT_PAIR_X_TOLERANCE_M = 0.04
DEFAULT_BOLT_PAIR_Y_TOLERANCE_M = 0.05
DEFAULT_BOLT_PAIR_MAX_Z_DELTA_M = 0.03
DEFAULT_BOLT_PAIR_ASSOCIATION_MAX_ENDPOINT_M = 0.05
DEFAULT_BOLT_PAIR_ASSOCIATION_MISS_LIMIT = 3
DEFAULT_SOURCE_EPOCH_ROLLBACK_SEC = 5.0
DEFAULT_SOURCE_EPOCH_CONFIRMATION_PAIRS = 2
DEFAULT_SOURCE_EPOCH_QUARANTINE_PAIRS = 2
DEFAULT_SINGLE_TARGET_ASSOCIATION_MAX_DISTANCE_M = 0.05
DEFAULT_SINGLE_TARGET_ASSOCIATION_MISS_LIMIT = 3


@dataclass(frozen=True)
class FrameGateResult:
    """Result of validating one RGB/depth source-frame pair."""

    accepted: bool
    reason: str


@dataclass(frozen=True)
class TimestampedFrame:
    """One decoded camera frame with source and local receipt timestamps."""

    stamp_ns: int
    received_at: float
    payload: object

    def __post_init__(self) -> None:
        if isinstance(self.stamp_ns, bool) or int(self.stamp_ns) < 0:
            raise ValueError('frame source timestamp must be non-negative')
        if not math.isfinite(self.received_at):
            raise ValueError('frame receipt timestamp must be finite')


@dataclass(frozen=True)
class SynchronizedFramePair:
    """A fresh RGB/depth pair selected from independent bounded queues."""

    rgb: TimestampedFrame
    depth: TimestampedFrame


@dataclass(frozen=True)
class SourceTimestampEpochResult:
    """One source-pair decision from :class:`SourceTimestampEpochTracker`."""

    accept_pair: bool
    rebase: bool
    pending: bool
    quarantined: bool
    epoch: int
    reason: str
    rgb_boundary_ns: int | None
    depth_boundary_ns: int | None


class SourceTimestampEpochTracker:
    """
    Detect a real RGB/depth source-clock restart without trusting one packet.

    A small or one-sided timestamp regression never requests a generation
    change. A rebase is signalled only after both modalities produce the
    configured number of coherent, strictly increasing low-timestamp pairs.
    The confirming pair is accepted as the first pair of the new epoch.

    Immediately after a rebase, a short pair-count quarantine rejects large
    forward jumps toward the old epoch. This prevents delayed DDS packets from
    poisoning the newly rebased anti-replay boundary.
    """

    def __init__(
        self,
        *,
        rollback_threshold_sec: float = DEFAULT_SOURCE_EPOCH_ROLLBACK_SEC,
        max_pair_skew_sec: float = 0.2,
        confirmation_pairs: int = DEFAULT_SOURCE_EPOCH_CONFIRMATION_PAIRS,
        quarantine_pairs: int = DEFAULT_SOURCE_EPOCH_QUARANTINE_PAIRS,
    ):
        rollback_threshold_sec = float(rollback_threshold_sec)
        max_pair_skew_sec = float(max_pair_skew_sec)
        if (
            not math.isfinite(rollback_threshold_sec)
            or rollback_threshold_sec <= 0.0
        ):
            raise ValueError(
                'rollback_threshold_sec must be finite and positive')
        if not math.isfinite(max_pair_skew_sec) or max_pair_skew_sec <= 0.0:
            raise ValueError(
                'max_pair_skew_sec must be finite and positive')
        if (
            isinstance(confirmation_pairs, bool)
            or int(confirmation_pairs) != confirmation_pairs
            or int(confirmation_pairs) < 2
        ):
            raise ValueError(
                'confirmation_pairs must be an integer of at least two')
        if (
            isinstance(quarantine_pairs, bool)
            or int(quarantine_pairs) != quarantine_pairs
            or int(quarantine_pairs) < 1
        ):
            raise ValueError(
                'quarantine_pairs must be a positive integer')

        self._rollback_threshold_ns = round(
            rollback_threshold_sec * 1_000_000_000)
        self._max_pair_skew_ns = round(
            max_pair_skew_sec * 1_000_000_000)
        self._confirmation_pairs = int(confirmation_pairs)
        self._quarantine_pairs = int(quarantine_pairs)
        self._epoch = 0
        self._last_rgb_stamp_ns = None
        self._last_depth_stamp_ns = None
        self._pending = False
        self._pending_last_pair = None
        self._pending_confirmation_count = 0
        self._quarantine_remaining = 0
        self._quarantine_rgb_cutoff_ns = None
        self._quarantine_depth_cutoff_ns = None

    @staticmethod
    def _validated_stamp(value, label: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f'{label} source timestamp must be non-negative')
        try:
            parsed = int(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                f'{label} source timestamp must be an integer') from exc
        if parsed < 0 or parsed != value:
            raise ValueError(
                f'{label} source timestamp must be a non-negative integer')
        return parsed

    def _result(
        self,
        *,
        accept_pair: bool,
        rebase: bool = False,
        quarantined: bool = False,
        reason: str,
    ) -> SourceTimestampEpochResult:
        return SourceTimestampEpochResult(
            accept_pair=accept_pair,
            rebase=rebase,
            pending=self._pending,
            quarantined=quarantined,
            epoch=self._epoch,
            reason=reason,
            rgb_boundary_ns=self._last_rgb_stamp_ns,
            depth_boundary_ns=self._last_depth_stamp_ns,
        )

    def _clear_pending(self) -> None:
        self._pending = False
        self._pending_last_pair = None
        self._pending_confirmation_count = 0

    def _clear_quarantine(self) -> None:
        self._quarantine_remaining = 0
        self._quarantine_rgb_cutoff_ns = None
        self._quarantine_depth_cutoff_ns = None

    def _consume_quarantine_pair(self) -> None:
        if self._quarantine_remaining <= 0:
            return
        self._quarantine_remaining -= 1
        if self._quarantine_remaining == 0:
            self._clear_quarantine()

    def observe_pair(
        self,
        rgb_stamp_ns: int,
        depth_stamp_ns: int,
    ) -> SourceTimestampEpochResult:
        """
        Observe one timestamp pair and return a node-consumable epoch decision.

        ``accept_pair`` covers only source sequence/epoch policy. The caller
        must still enforce receipt age, publisher uniqueness, and image skew.
        On ``rebase=True``, it should atomically clear old-generation caches,
        queues, associations, and frame-gate boundaries before committing this
        accepted confirming pair as new-epoch frame one.
        """
        rgb_stamp_ns = self._validated_stamp(rgb_stamp_ns, 'RGB')
        depth_stamp_ns = self._validated_stamp(depth_stamp_ns, 'depth')
        pair_skew_ns = abs(rgb_stamp_ns - depth_stamp_ns)

        if (
            self._last_rgb_stamp_ns is None
            or self._last_depth_stamp_ns is None
        ):
            if pair_skew_ns > self._max_pair_skew_ns:
                return self._result(
                    accept_pair=False,
                    reason='initial RGB/depth source stamps are not coherent',
                )
            self._last_rgb_stamp_ns = rgb_stamp_ns
            self._last_depth_stamp_ns = depth_stamp_ns
            return self._result(
                accept_pair=True,
                reason='initial source epoch established',
            )

        if self._quarantine_remaining > 0:
            rgb_old_epoch = (
                self._quarantine_rgb_cutoff_ns is not None
                and rgb_stamp_ns >= self._quarantine_rgb_cutoff_ns
            )
            depth_old_epoch = (
                self._quarantine_depth_cutoff_ns is not None
                and depth_stamp_ns >= self._quarantine_depth_cutoff_ns
            )
            if rgb_old_epoch or depth_old_epoch:
                return self._result(
                    accept_pair=False,
                    quarantined=True,
                    reason='delayed old-epoch source stamp quarantined',
                )

        rgb_forward = rgb_stamp_ns > self._last_rgb_stamp_ns
        depth_forward = depth_stamp_ns > self._last_depth_stamp_ns
        if rgb_forward and depth_forward:
            if pair_skew_ns > self._max_pair_skew_ns:
                return self._result(
                    accept_pair=False,
                    reason='forward RGB/depth source stamps are not coherent',
                )
            pending_was_active = self._pending
            self._clear_pending()
            self._last_rgb_stamp_ns = rgb_stamp_ns
            self._last_depth_stamp_ns = depth_stamp_ns
            self._consume_quarantine_pair()
            return self._result(
                accept_pair=True,
                reason=(
                    'old source epoch recovered; rollback pending canceled'
                    if pending_was_active
                    else 'source stamps increased'
                ),
            )

        rgb_rollback_ns = max(
            0, self._last_rgb_stamp_ns - rgb_stamp_ns)
        depth_rollback_ns = max(
            0, self._last_depth_stamp_ns - depth_stamp_ns)
        rgb_large_rollback = (
            rgb_rollback_ns >= self._rollback_threshold_ns)
        depth_large_rollback = (
            depth_rollback_ns >= self._rollback_threshold_ns)

        if not (rgb_large_rollback or depth_large_rollback):
            return self._result(
                accept_pair=False,
                reason='small duplicate/out-of-order source stamp rejected',
            )

        self._pending = True
        if not (rgb_large_rollback and depth_large_rollback):
            self._pending_last_pair = None
            self._pending_confirmation_count = 0
            return self._result(
                accept_pair=False,
                reason='one-sided large source rollback pending',
            )

        if pair_skew_ns > self._max_pair_skew_ns:
            self._pending_last_pair = None
            self._pending_confirmation_count = 0
            return self._result(
                accept_pair=False,
                reason='two-sided rollback is not RGB/depth coherent',
            )

        current_pair = (rgb_stamp_ns, depth_stamp_ns)
        if self._pending_last_pair is None:
            self._pending_last_pair = current_pair
            self._pending_confirmation_count = 1
        else:
            previous_rgb, previous_depth = self._pending_last_pair
            if (
                rgb_stamp_ns > previous_rgb
                and depth_stamp_ns > previous_depth
            ):
                self._pending_last_pair = current_pair
                self._pending_confirmation_count += 1
            elif (
                rgb_stamp_ns < previous_rgb
                or depth_stamp_ns < previous_depth
            ):
                self._pending_last_pair = current_pair
                self._pending_confirmation_count = 1

        if self._pending_confirmation_count < self._confirmation_pairs:
            return self._result(
                accept_pair=False,
                reason=(
                    'coherent rollback confirmation '
                    f'{self._pending_confirmation_count}/'
                    f'{self._confirmation_pairs}'
                ),
            )

        old_rgb_stamp_ns = self._last_rgb_stamp_ns
        old_depth_stamp_ns = self._last_depth_stamp_ns
        self._epoch += 1
        self._last_rgb_stamp_ns = rgb_stamp_ns
        self._last_depth_stamp_ns = depth_stamp_ns
        self._clear_pending()
        self._quarantine_remaining = self._quarantine_pairs
        self._quarantine_rgb_cutoff_ns = (
            old_rgb_stamp_ns + rgb_stamp_ns
        ) // 2
        self._quarantine_depth_cutoff_ns = (
            old_depth_stamp_ns + depth_stamp_ns
        ) // 2
        return self._result(
            accept_pair=True,
            rebase=True,
            reason='coherent RGB/depth source epoch rollback confirmed',
        )

    def seed_reset_boundaries(
        self,
        rgb_boundary_ns: int | None,
        depth_boundary_ns: int | None,
    ) -> bool:
        """
        Seed an uninitialized tracker from a reset's observed source boundary.

        Camera callbacks can run before camera info is available or before the
        first inference timer tick. In that case ``CameraFrameGate`` already
        has anti-replay boundaries at reset time while this tracker has never
        observed a pair. Without synchronizing the two, a restarted source's
        low timestamps would be accepted as this tracker's initial epoch but
        rejected forever by the frame gate's old high boundary.

        Both independently observed modality boundaries are required. A
        one-sided boundary cannot prove that both source clocks existed in the
        old epoch and therefore must never enable automatic rebasing.
        """
        if rgb_boundary_ns is None or depth_boundary_ns is None:
            return False
        rgb_boundary_ns = self._validated_stamp(
            rgb_boundary_ns,
            'RGB',
        )
        depth_boundary_ns = self._validated_stamp(
            depth_boundary_ns,
            'depth',
        )
        if (
            self._last_rgb_stamp_ns is not None
            or self._last_depth_stamp_ns is not None
        ):
            return False
        self._last_rgb_stamp_ns = rgb_boundary_ns
        self._last_depth_stamp_ns = depth_boundary_ns
        return True

    @property
    def epoch(self) -> int:
        """Return the number of confirmed automatic source rebases."""
        return self._epoch

    @property
    def pending_confirmation_count(self) -> int:
        """Return coherent low-timestamp pairs accumulated for a rebase."""
        return self._pending_confirmation_count

    @property
    def quarantine_active(self) -> bool:
        """Return whether delayed old-epoch high stamps are still guarded."""
        return self._quarantine_remaining > 0

    def stamp_is_quarantined(self, modality: str, stamp_ns: int) -> bool:
        """Check one callback stamp against the active old-epoch cutoff."""
        stamp_ns = self._validated_stamp(stamp_ns, modality)
        if self._quarantine_remaining <= 0:
            return False
        if modality == 'RGB':
            cutoff = self._quarantine_rgb_cutoff_ns
        elif modality == 'depth':
            cutoff = self._quarantine_depth_cutoff_ns
        else:
            raise ValueError("modality must be 'RGB' or 'depth'")
        return cutoff is not None and stamp_ns >= cutoff


class CameraFrameGate:
    """Accept only fresh, synchronized, strictly increasing camera frames."""

    def __init__(self, max_skew_sec: float = 0.2, max_receipt_age_sec: float = 1.0):
        if not math.isfinite(max_skew_sec) or max_skew_sec <= 0.0:
            raise ValueError('max_skew_sec must be a finite positive number')
        if not math.isfinite(max_receipt_age_sec) or max_receipt_age_sec <= 0.0:
            raise ValueError('max_receipt_age_sec must be a finite positive number')
        self._max_skew_ns = round(max_skew_sec * 1_000_000_000)
        self._max_receipt_age_sec = max_receipt_age_sec
        self._last_rgb_stamp_ns = None
        self._last_depth_stamp_ns = None

    def reset(
        self,
        preserve_source_sequence: bool = True,
        *,
        rgb_boundary_ns: int | None = None,
        depth_boundary_ns: int | None = None,
    ) -> None:
        """Reset while advancing anti-replay boundaries to observed callbacks."""
        if not preserve_source_sequence:
            self._last_rgb_stamp_ns = None
            self._last_depth_stamp_ns = None
        boundaries = (
            ('RGB', rgb_boundary_ns),
            ('depth', depth_boundary_ns),
        )
        for label, boundary in boundaries:
            if boundary is not None and int(boundary) < 0:
                raise ValueError(f'{label} boundary must be non-negative')
        if rgb_boundary_ns is not None:
            self._last_rgb_stamp_ns = max(
                int(rgb_boundary_ns),
                self._last_rgb_stamp_ns
                if self._last_rgb_stamp_ns is not None
                else int(rgb_boundary_ns),
            )
        if depth_boundary_ns is not None:
            self._last_depth_stamp_ns = max(
                int(depth_boundary_ns),
                self._last_depth_stamp_ns
                if self._last_depth_stamp_ns is not None
                else int(depth_boundary_ns),
            )

    def evaluate(
        self,
        *,
        rgb_stamp_ns: int,
        depth_stamp_ns: int,
        rgb_received_at: float,
        depth_received_at: float,
        now: float,
        publisher_counts: Mapping[str, int],
    ) -> FrameGateResult:
        """Validate a source pair and advance the sequence only when it is accepted."""
        invalid_sources = [
            f'{topic}={count}'
            for topic, count in publisher_counts.items()
            if count != 1
        ]
        if invalid_sources:
            return FrameGateResult(
                False,
                'publisher count must be exactly one: ' + ', '.join(invalid_sources),
            )

        receipt_error = frame_receipt_error(
            rgb_received_at=rgb_received_at,
            depth_received_at=depth_received_at,
            now=now,
            max_receipt_age_sec=self._max_receipt_age_sec,
        )
        if receipt_error is not None:
            return FrameGateResult(
                False,
                receipt_error,
            )

        skew_ns = abs(rgb_stamp_ns - depth_stamp_ns)
        if skew_ns > self._max_skew_ns:
            return FrameGateResult(
                False,
                f'RGB/depth skew {skew_ns / 1e9:.3f}s exceeds '
                f'{self._max_skew_ns / 1e9:.3f}s',
            )

        if (
            self._last_rgb_stamp_ns is not None
            and rgb_stamp_ns <= self._last_rgb_stamp_ns
        ):
            relation = (
                'duplicate'
                if rgb_stamp_ns == self._last_rgb_stamp_ns
                else 'retrograde'
            )
            return FrameGateResult(False, f'{relation} RGB source stamp')
        if (
            self._last_depth_stamp_ns is not None
            and depth_stamp_ns <= self._last_depth_stamp_ns
        ):
            relation = (
                'duplicate'
                if depth_stamp_ns == self._last_depth_stamp_ns
                else 'retrograde'
            )
            return FrameGateResult(False, f'{relation} depth source stamp')

        self._last_rgb_stamp_ns = rgb_stamp_ns
        self._last_depth_stamp_ns = depth_stamp_ns
        return FrameGateResult(True, 'accepted')

    @property
    def last_rgb_stamp_ns(self) -> int | None:
        """Return the accepted/reset RGB anti-replay boundary."""
        return self._last_rgb_stamp_ns

    @property
    def last_depth_stamp_ns(self) -> int | None:
        """Return the accepted/reset depth anti-replay boundary."""
        return self._last_depth_stamp_ns


def frame_receipt_error(
    *,
    rgb_received_at: float,
    depth_received_at: float,
    now: float,
    max_receipt_age_sec: float,
) -> str | None:
    """Validate the actual local receipt times of one RGB/depth pair."""
    values = (rgb_received_at, depth_received_at, now, max_receipt_age_sec)
    if not all(math.isfinite(float(value)) for value in values):
        return 'receipt timestamp/limit must be finite'
    if max_receipt_age_sec <= 0.0:
        return 'receipt age limit must be positive'

    rgb_age = now - rgb_received_at
    depth_age = now - depth_received_at
    if rgb_age < 0.0 or depth_age < 0.0:
        return 'receipt clock moved backwards'
    if (
        rgb_age > max_receipt_age_sec
        or depth_age > max_receipt_age_sec
    ):
        return (
            f'stale receipt: rgb={rgb_age:.3f}s '
            f'depth={depth_age:.3f}s'
        )
    return None


def select_newest_synchronized_pair(
    rgb_frames: Sequence[TimestampedFrame],
    depth_frames: Sequence[TimestampedFrame],
    *,
    max_skew_sec: float,
    max_receipt_age_sec: float,
    now: float,
    rgb_after_ns: int | None = None,
    depth_after_ns: int | None = None,
) -> SynchronizedFramePair | None:
    """
    Select the newest pair without combining unrelated latest frames.

    ROS image publishers and inference timers run independently.  Taking only
    each topic's latest callback can therefore combine RGB from one source
    frame with depth from another.  This selector searches bounded queues and
    ranks pairs by their newest common source time, then by the other source
    time and finally by the smallest skew.
    """
    if not math.isfinite(max_skew_sec) or max_skew_sec <= 0.0:
        raise ValueError('max_skew_sec must be a finite positive number')
    if (
        not math.isfinite(max_receipt_age_sec)
        or max_receipt_age_sec <= 0.0
    ):
        raise ValueError(
            'max_receipt_age_sec must be a finite positive number')
    if not math.isfinite(now):
        raise ValueError('now must be finite')
    for label, boundary in (
        ('RGB', rgb_after_ns),
        ('depth', depth_after_ns),
    ):
        if boundary is not None and int(boundary) < 0:
            raise ValueError(f'{label} boundary must be non-negative')

    max_skew_ns = round(max_skew_sec * 1_000_000_000)

    def eligible(
        frame: TimestampedFrame,
        boundary: int | None,
    ) -> bool:
        receipt_age = now - frame.received_at
        return (
            0.0 <= receipt_age <= max_receipt_age_sec
            and (
                boundary is None
                or frame.stamp_ns > int(boundary)
            )
        )

    candidates = []
    for rgb in rgb_frames:
        if not eligible(rgb, rgb_after_ns):
            continue
        for depth in depth_frames:
            if not eligible(depth, depth_after_ns):
                continue
            skew_ns = abs(rgb.stamp_ns - depth.stamp_ns)
            if skew_ns > max_skew_ns:
                continue
            rank = (
                min(rgb.stamp_ns, depth.stamp_ns),
                max(rgb.stamp_ns, depth.stamp_ns),
                -skew_ns,
                min(rgb.received_at, depth.received_at),
            )
            candidates.append((rank, rgb, depth))

    if not candidates:
        return None
    _rank, rgb, depth = max(candidates, key=lambda candidate: candidate[0])
    return SynchronizedFramePair(rgb=rgb, depth=depth)


class TargetFrameBarrier:
    """Track strictly increasing post-reset detections for each target label."""

    def __init__(self, required_frames: int = 3):
        if isinstance(required_frames, bool) or required_frames < 1:
            raise ValueError('required_frames must be a positive integer')
        self.required_frames = int(required_frames)
        self._generation = 0
        self._counts = {}
        self._last_stamps = {}

    def reset(self) -> None:
        """Discard every target observation from the previous camera position."""
        self._generation += 1
        self._counts.clear()
        self._last_stamps.clear()

    def reset_target(self, target: str) -> None:
        """Restart one target without disturbing independent labels."""
        self._counts.pop(target, None)
        self._last_stamps.pop(target, None)

    def record(
        self,
        target: str,
        stamp_ns: int,
        *,
        generation: int | None = None,
    ) -> bool:
        """Record one current-generation target frame and report readiness."""
        if generation is not None and not self.is_current(generation):
            return False
        previous = self._last_stamps.get(target)
        if previous is not None and stamp_ns <= previous:
            return self.ready(target)
        self._last_stamps[target] = stamp_ns
        self._counts[target] = min(
            self.required_frames,
            self._counts.get(target, 0) + 1,
        )
        return self.ready(target)

    def count(self, target: str) -> int:
        """Return the number of accepted target frames since the latest reset."""
        return self._counts.get(target, 0)

    def ready(self, target: str) -> bool:
        """Return whether a target has crossed the required frame barrier."""
        return self.count(target) >= self.required_frames

    @property
    def generation(self) -> int:
        """Return the reset generation associated with current observations."""
        return self._generation

    def is_current(self, generation: int) -> bool:
        """Return whether work captured before inference still belongs here."""
        return int(generation) == self._generation


def advance_source_boundary(previous, candidate: int) -> int:
    """Advance a callback-observed timestamp without allowing regression."""
    candidate = int(candidate)
    if candidate < 0:
        raise ValueError('source boundary stamp must be non-negative')
    if previous is None:
        return candidate
    return max(int(previous), candidate)


def stamp_to_nanoseconds(stamp) -> int:
    """Convert a ROS-style time object to a validated integer nanosecond stamp."""
    sec = int(stamp.sec)
    nanosec = int(stamp.nanosec)
    if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
        raise ValueError(f'invalid ROS stamp: {sec}.{nanosec:09d}')
    return sec * 1_000_000_000 + nanosec


def select_central_candidates(
    candidates: Sequence[Mapping],
    image_size: tuple[int, int],
    limit: int,
    min_confidence: float = MIN_TARGET_CONFIDENCE,
) -> list[Mapping]:
    """Select confident candidates nearest the image center with stable tie-breaks."""
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError('image dimensions must be positive')
    if limit < 1:
        raise ValueError('limit must be positive')

    center_x = image_width / 2.0
    center_y = image_height / 2.0
    ranked = []
    for source_index, candidate in enumerate(candidates):
        score = float(candidate['score'])
        pixel_x, pixel_y = candidate['pixel']
        if not math.isfinite(score) or score < min_confidence:
            continue
        distance_sq = (
            (float(pixel_x) - center_x) ** 2
            + (float(pixel_y) - center_y) ** 2
        )
        ranked.append((distance_sq, -score, source_index, candidate))
    ranked.sort(key=lambda item: item[:3])
    return [item[3] for item in ranked[:limit]]


def _validated_world_point(point, label: str) -> tuple[float, float, float]:
    try:
        if len(point) != 3:
            raise ValueError
        parsed = tuple(float(value) for value in point)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} must be a finite 3D point') from exc
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError(f'{label} must be a finite 3D point')
    return parsed


class SingleTargetAssociationGate:
    """Keep a single detected object tied to its first finite world point."""

    def __init__(
        self,
        max_distance_m: float = (
            DEFAULT_SINGLE_TARGET_ASSOCIATION_MAX_DISTANCE_M
        ),
        miss_limit: int = DEFAULT_SINGLE_TARGET_ASSOCIATION_MISS_LIMIT,
    ):
        max_distance_m = float(max_distance_m)
        if not math.isfinite(max_distance_m) or max_distance_m <= 0.0:
            raise ValueError('max_distance_m must be finite and positive')
        if (
            isinstance(miss_limit, bool)
            or int(miss_limit) != miss_limit
            or int(miss_limit) < 1
        ):
            raise ValueError('miss_limit must be a positive integer')
        self.max_distance_m = max_distance_m
        self.miss_limit = int(miss_limit)
        self._anchor = None
        self._consecutive_misses = 0

    @property
    def active(self) -> bool:
        """Return whether an immutable target anchor is latched."""
        return self._anchor is not None

    @property
    def anchor(self):
        """Return the current immutable world-point anchor."""
        return self._anchor

    @property
    def consecutive_misses(self) -> int:
        """Return consecutive committed frames without the anchored target."""
        return self._consecutive_misses

    def reset(self) -> None:
        """Clear the target anchor and its miss run."""
        self._anchor = None
        self._consecutive_misses = 0

    def record(self, world_point) -> tuple[bool, float]:
        """
        Record one candidate and return ``(associated, distance_m)``.

        A far candidate never replaces an active anchor. The caller should
        treat it as a miss; only the explicit miss policy can expire an anchor.
        """
        parsed = _validated_world_point(world_point, 'world_point')
        if self._anchor is None:
            self._anchor = parsed
            self._consecutive_misses = 0
            return True, 0.0
        distance = math.dist(self._anchor, parsed)
        if distance > self.max_distance_m:
            return False, distance
        self._consecutive_misses = 0
        return True, distance

    def record_miss(self) -> bool:
        """Expire and clear the anchor only at the configured miss limit."""
        if self._anchor is None:
            return False
        self._consecutive_misses += 1
        if self._consecutive_misses < self.miss_limit:
            return False
        self.reset()
        return True


def select_sticky_single_target(
    candidates: Sequence[Mapping],
    image_size: tuple[int, int],
    *,
    anchor_point: Sequence[float] | None = None,
    max_anchor_distance_m: float = (
        DEFAULT_SINGLE_TARGET_ASSOCIATION_MAX_DISTANCE_M
    ),
    min_confidence: float = MIN_TARGET_CONFIDENCE,
) -> Mapping | None:
    """
    Select a nut centrally at first, then stay on its world-point anchor.

    Every usable candidate must meet the confidence floor and contain finite
    ``pixel`` and ``world_point`` fields. With an anchor, 3D anchor distance is
    the primary rank and candidates outside the association radius are
    ineligible; image-center distance and confidence are deterministic
    tie-breakers.
    """
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError('image dimensions must be positive')
    max_anchor_distance_m = float(max_anchor_distance_m)
    min_confidence = float(min_confidence)
    if (
        not math.isfinite(max_anchor_distance_m)
        or max_anchor_distance_m <= 0.0
    ):
        raise ValueError(
            'max_anchor_distance_m must be finite and positive')
    if not math.isfinite(min_confidence):
        raise ValueError('minimum confidence must be finite')
    parsed_anchor = (
        _validated_world_point(anchor_point, 'anchor_point')
        if anchor_point is not None
        else None
    )

    center_x = image_width / 2.0
    center_y = image_height / 2.0
    ranked = []
    for source_index, candidate in enumerate(candidates):
        try:
            score = float(candidate['score'])
            pixel = tuple(float(value) for value in candidate['pixel'])
            world_point = _validated_world_point(
                candidate['world_point'], 'candidate world_point')
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not math.isfinite(score)
            or score < min_confidence
            or len(pixel) != 2
            or not all(math.isfinite(value) for value in pixel)
        ):
            continue
        center_distance_sq = (
            (pixel[0] - center_x) ** 2
            + (pixel[1] - center_y) ** 2
        )
        anchor_distance = 0.0
        if parsed_anchor is not None:
            anchor_distance = math.dist(parsed_anchor, world_point)
            if anchor_distance > max_anchor_distance_m:
                continue
        ranked.append((
            anchor_distance if parsed_anchor is not None else 0.0,
            center_distance_sq,
            -score,
            source_index,
            candidate,
        ))

    if not ranked:
        return None
    ranked.sort(key=lambda item: item[:4])
    return ranked[0][4]


def associate_bolt_pair_endpoints(
    previous_pair: Sequence[Sequence[float]],
    current_pair: Sequence[Sequence[float]],
) -> tuple[float, bool]:
    """
    Return the smallest endpoint displacement and whether current order swaps.

    The caller uses the distance to distinguish the same physical battery pack
    from another geometrically valid pack between consecutive frames.
    """
    pairs = []
    for name, pair in (
        ('previous_pair', previous_pair),
        ('current_pair', current_pair),
    ):
        if len(pair) != 2:
            raise ValueError(f'{name} must contain exactly two endpoints')
        parsed = []
        for point in pair:
            if len(point) != 3:
                raise ValueError(f'{name} endpoints must be 3D points')
            values = tuple(float(value) for value in point)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f'{name} endpoints must be finite')
            parsed.append(values)
        pairs.append(parsed)

    previous, current = pairs

    def distance(left, right):
        return math.sqrt(sum(
            (left[index] - right[index]) ** 2
            for index in range(3)
        ))

    direct_distances = (
        distance(previous[0], current[0]),
        distance(previous[1], current[1]),
    )
    swapped_distances = (
        distance(previous[0], current[1]),
        distance(previous[1], current[0]),
    )
    direct_cost = sum(direct_distances)
    swapped_cost = sum(swapped_distances)
    if swapped_cost < direct_cost:
        return max(swapped_distances), True
    return max(direct_distances), False


class BoltPairAssociationGate:
    """Keep a three-frame bolt track tied to its first physical pair."""

    def __init__(
        self,
        max_endpoint_distance_m: float = (
            DEFAULT_BOLT_PAIR_ASSOCIATION_MAX_ENDPOINT_M
        ),
        miss_limit: int = DEFAULT_BOLT_PAIR_ASSOCIATION_MISS_LIMIT,
    ):
        max_endpoint_distance_m = float(max_endpoint_distance_m)
        if (
            not math.isfinite(max_endpoint_distance_m)
            or max_endpoint_distance_m <= 0.0
        ):
            raise ValueError(
                'max_endpoint_distance_m must be finite and positive')
        self.max_endpoint_distance_m = max_endpoint_distance_m
        if isinstance(miss_limit, bool) or int(miss_limit) < 1:
            raise ValueError('miss_limit must be a positive integer')
        self.miss_limit = int(miss_limit)
        self._anchor = None
        self._consecutive_misses = 0

    @property
    def active(self) -> bool:
        """Return whether a physical pair anchor is currently latched."""
        return self._anchor is not None

    @property
    def anchor(self):
        """Return the immutable current endpoint anchor, if one is latched."""
        return self._anchor

    @property
    def consecutive_misses(self) -> int:
        """Return how many consecutive frames missed the anchored pair."""
        return self._consecutive_misses

    def reset(self) -> None:
        """Forget the current physical pair association."""
        self._anchor = None
        self._consecutive_misses = 0

    def record_miss(self) -> bool:
        """
        Record one frame without the anchored pair.

        Returns true only when the explicit consecutive-miss policy expires
        the anchor. A single frame from a neighbouring pack therefore cannot
        replace the pair being accumulated by the target-frame barrier.
        """
        if self._anchor is None:
            return False
        self._consecutive_misses += 1
        if self._consecutive_misses < self.miss_limit:
            return False
        self.reset()
        return True

    def record(
        self,
        current_pair: Sequence[Sequence[float]],
    ) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], bool, float]:
        """
        Order and record one pair.

        Returns ``(ordered_pair, restarted, endpoint_distance_m)``. The first
        pair starts an anchor without reporting a restart. A later pair farther
        than the endpoint limit replaces the anchor and reports ``restarted``.
        """
        if self._anchor is None:
            associate_bolt_pair_endpoints(current_pair, current_pair)
            ordered = tuple(
                tuple(float(value) for value in point)
                for point in current_pair
            )
            self._anchor = ordered
            self._consecutive_misses = 0
            return ordered, False, 0.0

        endpoint_distance, swap_current = associate_bolt_pair_endpoints(
            self._anchor,
            current_pair,
        )
        source = reversed(current_pair) if swap_current else current_pair
        ordered = tuple(
            tuple(float(value) for value in point)
            for point in source
        )
        restarted = endpoint_distance > self.max_endpoint_distance_m
        if restarted:
            self._anchor = ordered
        self._consecutive_misses = 0
        return ordered, restarted, endpoint_distance


def select_central_bolt_pair(
    candidates: Sequence[Mapping],
    image_size: tuple[int, int],
    *,
    expected_delta_xy: tuple[float, float] = (
        DEFAULT_BOLT_PAIR_DX_M,
        DEFAULT_BOLT_PAIR_DY_M,
    ),
    x_tolerance_m: float = DEFAULT_BOLT_PAIR_X_TOLERANCE_M,
    y_tolerance_m: float = DEFAULT_BOLT_PAIR_Y_TOLERANCE_M,
    max_z_delta_m: float = DEFAULT_BOLT_PAIR_MAX_Z_DELTA_M,
    min_confidence: float = MIN_TARGET_CONFIDENCE,
    anchor_pair: Sequence[Sequence[float]] | None = None,
    max_anchor_endpoint_distance_m: float = (
        DEFAULT_BOLT_PAIR_ASSOCIATION_MAX_ENDPOINT_M
    ),
) -> tuple[Mapping, Mapping] | None:
    """
    Select the same-pack pair whose ordered ``bolt_2`` is nearest center.

    The fixed camera is translated over the active station's ``bolt_2`` while
    retaining its calibrated orientation. Candidate pairs must match the
    signed ``bolt_1 -> bolt_2`` world vector (in either input order).
    Pair midpoint, geometry error, confidence, and source order are
    deterministic tie-breakers after the ``bolt_2`` endpoint distance.
    Once ``anchor_pair`` is supplied, only pairs close to those physical
    endpoints are eligible and anchor distance takes priority over central
    ranking. Initial selection therefore retains the approved central rule,
    while an in-progress three-frame barrier remains sticky to one pack.
    This API receives no station identifier: the first anchor is necessarily
    camera-center/geometry based and cannot be corrected by a station prior.
    """
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError('image dimensions must be positive')

    expected_dx, expected_dy = (
        float(expected_delta_xy[0]),
        float(expected_delta_xy[1]),
    )
    x_tolerance_m = float(x_tolerance_m)
    y_tolerance_m = float(y_tolerance_m)
    max_z_delta_m = float(max_z_delta_m)
    min_confidence = float(min_confidence)
    max_anchor_endpoint_distance_m = float(
        max_anchor_endpoint_distance_m)
    if (
        not math.isfinite(expected_dx)
        or not math.isfinite(expected_dy)
        or math.hypot(expected_dx, expected_dy) <= 0.0
    ):
        raise ValueError('expected bolt-pair XY vector must be finite and non-zero')
    if not math.isfinite(x_tolerance_m) or x_tolerance_m <= 0.0:
        raise ValueError('bolt-pair X tolerance must be finite and positive')
    if not math.isfinite(y_tolerance_m) or y_tolerance_m <= 0.0:
        raise ValueError('bolt-pair Y tolerance must be finite and positive')
    if not math.isfinite(max_z_delta_m) or max_z_delta_m < 0.0:
        raise ValueError('bolt-pair max Z delta must be finite and non-negative')
    if not math.isfinite(min_confidence):
        raise ValueError('minimum confidence must be finite')
    if (
        not math.isfinite(max_anchor_endpoint_distance_m)
        or max_anchor_endpoint_distance_m <= 0.0
    ):
        raise ValueError(
            'anchor endpoint distance must be finite and positive')
    if anchor_pair is not None:
        # Reuse the endpoint parser/validation without changing the anchor.
        associate_bolt_pair_endpoints(anchor_pair, anchor_pair)

    usable = []
    for source_index, candidate in enumerate(candidates):
        try:
            score = float(candidate['score'])
            pixel = tuple(float(value) for value in candidate['pixel'])
            world_point = tuple(
                float(value) for value in candidate['world_point'])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not math.isfinite(score)
            or score < min_confidence
            or len(pixel) != 2
            or len(world_point) != 3
            or not all(math.isfinite(value) for value in pixel + world_point)
        ):
            continue
        usable.append((source_index, score, pixel, world_point, candidate))

    center_x = image_width / 2.0
    center_y = image_height / 2.0
    ranked_pairs = []
    for left_index, left in enumerate(usable):
        for right in usable[left_index + 1:]:
            world_dx = right[3][0] - left[3][0]
            world_dy = right[3][1] - left[3][1]
            world_dz = abs(left[3][2] - right[3][2])
            forward_errors = (
                abs(world_dx - expected_dx),
                abs(world_dy - expected_dy),
            )
            reverse_errors = (
                abs(world_dx + expected_dx),
                abs(world_dy + expected_dy),
            )
            forward_valid = (
                forward_errors[0] <= x_tolerance_m
                and forward_errors[1] <= y_tolerance_m
            )
            reverse_valid = (
                reverse_errors[0] <= x_tolerance_m
                and reverse_errors[1] <= y_tolerance_m
            )
            if world_dz > max_z_delta_m or not (forward_valid or reverse_valid):
                continue
            if forward_valid and (
                not reverse_valid or sum(forward_errors) <= sum(reverse_errors)
            ):
                geometry_errors = forward_errors
                ordered_pair = (left[4], right[4])
                bolt_2_entry = right
            else:
                geometry_errors = reverse_errors
                ordered_pair = (right[4], left[4])
                bolt_2_entry = left

            bolt_2_distance_sq = (
                (bolt_2_entry[2][0] - center_x) ** 2
                + (bolt_2_entry[2][1] - center_y) ** 2
            )
            midpoint_x = (left[2][0] + right[2][0]) / 2.0
            midpoint_y = (left[2][1] + right[2][1]) / 2.0
            midpoint_distance_sq = (
                (midpoint_x - center_x) ** 2
                + (midpoint_y - center_y) ** 2
            )
            geometry_error = (
                geometry_errors[0]
                + geometry_errors[1]
                + world_dz
            )
            minimum_score = min(left[1], right[1])
            score_sum = left[1] + right[1]
            anchor_distance = 0.0
            if anchor_pair is not None:
                anchor_distance, _swap = associate_bolt_pair_endpoints(
                    anchor_pair,
                    (
                        ordered_pair[0]['world_point'],
                        ordered_pair[1]['world_point'],
                    ),
                )
                if (
                    anchor_distance
                    > max_anchor_endpoint_distance_m
                ):
                    continue
            ranked_pairs.append((
                anchor_distance if anchor_pair is not None else 0.0,
                bolt_2_distance_sq,
                midpoint_distance_sq,
                geometry_error,
                -minimum_score,
                -score_sum,
                left[0],
                right[0],
                ordered_pair,
            ))

    if not ranked_pairs:
        return None

    ranked_pairs.sort(key=lambda item: item[:8])
    return ranked_pairs[0][8]


def scale_roi(
    *,
    image_size: tuple[int, int],
    reference_size: tuple[int, int],
    reference_bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Scale a central ROI from its calibration resolution to the live image."""
    image_width, image_height = image_size
    reference_width, reference_height = reference_size
    if min(image_width, image_height, reference_width, reference_height) <= 0:
        raise ValueError('image and reference dimensions must be positive')
    x_min, x_max, y_min, y_max = reference_bounds
    scaled = (
        round(x_min * image_width / reference_width),
        round(x_max * image_width / reference_width),
        round(y_min * image_height / reference_height),
        round(y_max * image_height / reference_height),
    )
    sx_min, sx_max, sy_min, sy_max = scaled
    if not (0 <= sx_min < sx_max <= image_width):
        raise ValueError('scaled ROI x bounds are outside the image')
    if not (0 <= sy_min < sy_max <= image_height):
        raise ValueError('scaled ROI y bounds are outside the image')
    return scaled

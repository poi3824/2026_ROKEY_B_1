"""Pure convergence policy for the single-bolt fine-alignment loop."""

from dataclasses import dataclass
import math
from typing import Any, Optional


def advance_source_boundary(previous, candidate: int):
    """Advance a callback-observed timestamp without allowing regression."""
    candidate = int(candidate)
    if candidate <= 0:
        return previous
    if previous is None:
        return candidate
    return max(int(previous), candidate)


@dataclass(frozen=True)
class AlignmentDecision:
    """Describe one decision made from fresh, valid target observations."""

    stamp_ns: int
    dx_px: float
    dy_px: float
    dtheta_deg: float
    correction_x_m: float
    correction_y_m: float
    aligned: bool
    consensus: bool
    frame_count: int
    hold_count: int
    success: bool
    # A mixed in/out-of-tolerance three-frame window can be a Hough-circle
    # quantization flicker around the optical centre.  Such a window is never
    # considered aligned or successful; it merely has to repeat with the same
    # direction before a 0.2 mm motion is allowed.
    deferred: bool


@dataclass(frozen=True)
class _TargetObservation:
    stamp_ns: int
    dx_px: float
    dy_px: float
    dtheta_deg: float


@dataclass(frozen=True)
class _PendingDecision:
    stamp_ns: int
    dx_px: float
    dy_px: float
    dtheta_deg: float
    correction_x_m: float
    correction_y_m: float
    aligned: bool
    consensus: bool
    frame_count: int
    deferred: bool = False
    hold_increment: int = 0


class FreshFramePairGate:
    """Accept only fresh, synchronized, strictly increasing RGB/depth pairs."""

    def __init__(
        self,
        *,
        max_skew_sec: float = 0.2,
        max_receipt_age_sec: float = 1.0,
    ):
        if not math.isfinite(max_skew_sec) or max_skew_sec < 0.0:
            raise ValueError('max_skew_sec must be finite and non-negative')
        if (
            not math.isfinite(max_receipt_age_sec)
            or max_receipt_age_sec <= 0.0
        ):
            raise ValueError(
                'max_receipt_age_sec must be finite and positive')
        self.max_skew_ns = int(max_skew_sec * 1_000_000_000)
        self.max_receipt_age_ns = int(
            max_receipt_age_sec * 1_000_000_000)
        self.reset()

    def reset(
        self,
        *,
        rgb_boundary_ns: Optional[int] = None,
        depth_boundary_ns: Optional[int] = None,
    ):
        """Require both streams to advance past the supplied reset boundary."""
        self.last_rgb_stamp_ns = (
            None if rgb_boundary_ns is None else int(rgb_boundary_ns)
        )
        self.last_depth_stamp_ns = (
            None if depth_boundary_ns is None else int(depth_boundary_ns)
        )

    def accept(
        self,
        *,
        rgb_stamp_ns: int,
        depth_stamp_ns: int,
        rgb_receipt_ns: int,
        depth_receipt_ns: int,
        now_receipt_ns: int,
    ):
        """Return ``(accepted, reason)`` for one candidate frame pair."""
        rgb_stamp_ns = int(rgb_stamp_ns)
        depth_stamp_ns = int(depth_stamp_ns)
        rgb_receipt_ns = int(rgb_receipt_ns)
        depth_receipt_ns = int(depth_receipt_ns)
        now_receipt_ns = int(now_receipt_ns)

        if rgb_stamp_ns <= 0 or depth_stamp_ns <= 0:
            return False, 'zero source stamp'
        if (
            self.last_rgb_stamp_ns is not None
            and rgb_stamp_ns <= self.last_rgb_stamp_ns
        ):
            return False, 'duplicate/retrograde RGB stamp'
        if (
            self.last_depth_stamp_ns is not None
            and depth_stamp_ns <= self.last_depth_stamp_ns
        ):
            return False, 'duplicate/retrograde depth stamp'
        if abs(rgb_stamp_ns - depth_stamp_ns) > self.max_skew_ns:
            return False, 'RGB/depth skew'

        receipt_ages = (
            now_receipt_ns - rgb_receipt_ns,
            now_receipt_ns - depth_receipt_ns,
        )
        if any(age < 0 for age in receipt_ages):
            return False, 'future receipt time'
        if any(age > self.max_receipt_age_ns for age in receipt_ages):
            return False, 'stale receipt'

        self.last_rgb_stamp_ns = rgb_stamp_ns
        self.last_depth_stamp_ns = depth_stamp_ns
        return True, ''


@dataclass(frozen=True)
class CachedSourceFrame:
    """One converted camera frame with source and receipt timestamps."""

    payload: Any
    stamp_ns: int
    receipt_ns: int


@dataclass(frozen=True)
class SynchronizedFramePair:
    """One strict RGB/depth pair accepted for tracking exactly once."""

    rgb: CachedSourceFrame
    depth: CachedSourceFrame


class LatestFramePairSynchronizer:
    """
    Retry a bounded RGB/depth history when either stream advances.

    DDS may invoke the RGB callback before the matching depth callback, or in
    the opposite order.  A faster stream can also advance more than once before
    the matching callback runs, so a small history is retained and the closest
    valid source-stamp pair is selected.  Accepted and older candidates are
    removed immediately and remain protected from reuse by
    :class:`FreshFramePairGate`'s strict per-stream source-stamp boundary.
    """

    def __init__(
        self,
        *,
        max_skew_sec: float = 0.2,
        max_receipt_age_sec: float = 1.0,
        history_size: int = 5,
    ):
        if (
            not isinstance(history_size, int)
            or isinstance(history_size, bool)
            or history_size <= 0
        ):
            raise ValueError('history_size must be a positive integer')
        self.frame_gate = FreshFramePairGate(
            max_skew_sec=max_skew_sec,
            max_receipt_age_sec=max_receipt_age_sec,
        )
        self.history_size = history_size
        self._rgb_history = []
        self._depth_history = []

    def reset(
        self,
        *,
        rgb_boundary_ns: Optional[int] = None,
        depth_boundary_ns: Optional[int] = None,
    ):
        """Clear both caches and require streams newer than the boundaries."""
        self._rgb_history.clear()
        self._depth_history.clear()
        self.frame_gate.reset(
            rgb_boundary_ns=rgb_boundary_ns,
            depth_boundary_ns=depth_boundary_ns,
        )

    def cache_rgb(self, payload: Any, stamp_ns: int, receipt_ns: int):
        """Append one RGB candidate to the bounded source-stamp history."""
        self._cache_frame(
            self._rgb_history,
            payload=payload,
            stamp_ns=int(stamp_ns),
            receipt_ns=int(receipt_ns),
        )

    def cache_depth(self, payload: Any, stamp_ns: int, receipt_ns: int):
        """Append one depth candidate to the bounded source-stamp history."""
        self._cache_frame(
            self._depth_history,
            payload=payload,
            stamp_ns=int(stamp_ns),
            receipt_ns=int(receipt_ns),
        )

    def _cache_frame(
        self,
        history,
        *,
        payload: Any,
        stamp_ns: int,
        receipt_ns: int,
    ):
        frame = CachedSourceFrame(
            payload=payload,
            stamp_ns=stamp_ns,
            receipt_ns=receipt_ns,
        )
        history[:] = [
            candidate
            for candidate in history
            if candidate.stamp_ns != frame.stamp_ns
        ]
        history.append(frame)
        del history[:-self.history_size]

    def _discard_consumed_history(self):
        rgb_boundary = self.frame_gate.last_rgb_stamp_ns
        depth_boundary = self.frame_gate.last_depth_stamp_ns
        if rgb_boundary is not None:
            self._rgb_history[:] = [
                frame
                for frame in self._rgb_history
                if frame.stamp_ns > rgb_boundary
            ]
        if depth_boundary is not None:
            self._depth_history[:] = [
                frame
                for frame in self._depth_history
                if frame.stamp_ns > depth_boundary
            ]

    def try_accept(self, *, now_receipt_ns: int):
        """Return the closest valid pair without consuming rejected frames."""
        if not self._rgb_history:
            return None, 'RGB unavailable'
        if not self._depth_history:
            return None, 'depth unavailable'

        candidates = [
            (rgb, depth)
            for rgb in self._rgb_history
            for depth in self._depth_history
        ]
        candidates.sort(
            key=lambda pair: (
                abs(pair[0].stamp_ns - pair[1].stamp_ns),
                -max(pair[0].stamp_ns, pair[1].stamp_ns),
                -min(pair[0].stamp_ns, pair[1].stamp_ns),
            ),
        )

        first_reason = ''
        for rgb, depth in candidates:
            accepted, reason = self.frame_gate.accept(
                rgb_stamp_ns=rgb.stamp_ns,
                depth_stamp_ns=depth.stamp_ns,
                rgb_receipt_ns=rgb.receipt_ns,
                depth_receipt_ns=depth.receipt_ns,
                now_receipt_ns=now_receipt_ns,
            )
            if not accepted:
                if not first_reason:
                    first_reason = reason
                continue

            pair = SynchronizedFramePair(rgb=rgb, depth=depth)
            self._discard_consumed_history()
            return pair, ''

        return None, first_reason


class FineAlignmentGate:
    """
    Build one closed-loop decision from three fresh target observations.

    A non-zero correction is a delta, not an absolute pose.  Once one is
    consumed, no later observation can create another correction until the
    executor acknowledges that exact source stamp as settled.
    """

    def __init__(
        self,
        *,
        position_tolerance_px: float = 2.0,
        angle_tolerance_deg: float = 3.0,
        hold_frames: int = 30,
        correction_step_m: float = 0.0002,
        decision_frames: int = 3,
        near_boundary_limit_px: Optional[float] = None,
    ):
        if (
            not math.isfinite(position_tolerance_px)
            or position_tolerance_px < 0.0
        ):
            raise ValueError(
                'position_tolerance_px must be finite and non-negative')
        if (
            not math.isfinite(angle_tolerance_deg)
            or angle_tolerance_deg < 0.0
        ):
            raise ValueError(
                'angle_tolerance_deg must be finite and non-negative')
        if not isinstance(hold_frames, int) or hold_frames <= 0:
            raise ValueError('hold_frames must be a positive integer')
        if not isinstance(decision_frames, int) or decision_frames <= 0:
            raise ValueError('decision_frames must be a positive integer')
        if (
            not math.isfinite(correction_step_m)
            or correction_step_m <= 0.0
        ):
            raise ValueError(
                'correction_step_m must be finite and positive')
        if near_boundary_limit_px is None:
            # The Hough detector reports an integer centre.  Around the
            # ±2-pixel completion threshold it can alternate between a
            # centred sample and a sample a few pixels farther out.  Six
            # pixels is *not* a completion tolerance: it is only the largest
            # one-window excursion eligible for confirmation before motion.
            near_boundary_limit_px = position_tolerance_px + 4.0
        if (
            not math.isfinite(near_boundary_limit_px)
            or near_boundary_limit_px < position_tolerance_px
        ):
            raise ValueError(
                'near_boundary_limit_px must be finite and no smaller than '
                'position_tolerance_px')

        self.position_tolerance_px = float(position_tolerance_px)
        self.angle_tolerance_deg = float(angle_tolerance_deg)
        self.hold_frames = hold_frames
        self.correction_step_m = float(correction_step_m)
        self.decision_frames = decision_frames
        self.near_boundary_limit_px = float(near_boundary_limit_px)
        self.reset()

    def reset(self):
        """Discard frame ordering, pending correction, and convergence state."""
        self.last_seen_stamp_ns: Optional[int] = None
        self.hold_count = 0
        self.completed = False
        self.pending_correction_stamp_ns: Optional[int] = None
        self._observations = []
        self._pending: Optional[_PendingDecision] = None
        self._deferred_correction_signature = None

    @property
    def waiting_for_settle(self) -> bool:
        return self.pending_correction_stamp_ns is not None

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    def submit(
        self,
        stamp_ns: int,
        *,
        valid: bool,
        dx_px: float = 0.0,
        dy_px: float = 0.0,
        dtheta_deg: float = 0.0,
    ) -> bool:
        """
        Accept one strictly newer source stamp.

        Invalid target frames and non-finite measurements never count toward a
        decision, but they also do not erase already collected valid target
        frames.  This permits three target observations to accumulate across
        camera dropouts without fabricating hold progress.
        """
        stamp_ns = int(stamp_ns)
        if stamp_ns <= 0:
            raise ValueError('stamp_ns must be positive')
        if self.completed:
            return False
        if (
            self.last_seen_stamp_ns is not None
            and stamp_ns <= self.last_seen_stamp_ns
        ):
            return False

        self.last_seen_stamp_ns = stamp_ns
        if self.waiting_for_settle or self._pending is not None:
            return False
        if not valid:
            return True

        values = (float(dx_px), float(dy_px), float(dtheta_deg))
        if not all(math.isfinite(value) for value in values):
            return True

        dx_px, dy_px, dtheta_deg = values
        self._observations.append(_TargetObservation(
            stamp_ns=stamp_ns,
            dx_px=dx_px,
            dy_px=dy_px,
            dtheta_deg=dtheta_deg,
        ))
        if len(self._observations) < self.decision_frames:
            return True

        observations = self._observations
        self._observations = []
        mean_dx, dx_aligned, dx_consensus = self._axis_state(
            [observation.dx_px for observation in observations],
            self.position_tolerance_px,
        )
        mean_dy, dy_aligned, dy_consensus = self._axis_state(
            [observation.dy_px for observation in observations],
            self.position_tolerance_px,
        )
        mean_dtheta, theta_aligned, theta_consensus = self._axis_state(
            [observation.dtheta_deg for observation in observations],
            self.angle_tolerance_deg,
        )
        consensus = dx_consensus and dy_consensus and theta_consensus
        aligned = (
            consensus
            and dx_aligned
            and dy_aligned
            and theta_aligned
        )
        correction_x_m = (
            self._axis_correction(mean_dy)
            if consensus else 0.0
        )
        correction_y_m = (
            self._axis_correction(mean_dx)
            if consensus else 0.0
        )
        yaw_needs_correction = (
            consensus
            and abs(mean_dtheta) > self.angle_tolerance_deg
        )

        deferred = False
        hold_increment = 0
        if self._has_one_near_boundary_position_outlier(observations):
            # Hough centres are integer pixels.  One ±3..6 px sample among
            # otherwise strict ±2 px observations is quantization noise, not
            # evidence for either a blind correction or loss of all previously
            # earned strict frames.  Count only the strict observations and
            # require a later wholly strict window before success.
            hold_increment = sum(
                self._observation_is_strict(observation)
                for observation in observations
            )
            self._deferred_correction_signature = None
            correction_x_m = 0.0
            correction_y_m = 0.0
            deferred = True
        elif self._is_mixed_near_boundary_jitter(
            observations,
            correction_x_m=correction_x_m,
            correction_y_m=correction_y_m,
            yaw_needs_correction=yaw_needs_correction,
        ):
            signature = self._correction_signature(
                correction_x_m,
                correction_y_m,
            )
            if signature == self._deferred_correction_signature:
                # The same signed error survived a wholly new three-frame
                # sample.  It is now evidence of a real offset, not a single
                # quantization flicker, so issue the normal fixed step.
                self._deferred_correction_signature = None
            else:
                # Do not reset a previously earned hold count or fabricate a
                # zero-pose command.  Wait for a second independent window.
                self._deferred_correction_signature = signature
                correction_x_m = 0.0
                correction_y_m = 0.0
                deferred = True
        else:
            self._deferred_correction_signature = None

        self._pending = _PendingDecision(
            stamp_ns=observations[-1].stamp_ns,
            dx_px=mean_dx,
            dy_px=mean_dy,
            dtheta_deg=mean_dtheta,
            correction_x_m=correction_x_m,
            correction_y_m=correction_y_m,
            aligned=aligned,
            consensus=consensus,
            frame_count=len(observations),
            deferred=deferred,
            hold_increment=hold_increment,
        )
        return True

    def consume(self) -> Optional[AlignmentDecision]:
        """Consume one three-target-frame decision at most once."""
        decision = self._pending
        self._pending = None
        if decision is None or self.completed:
            return None

        if decision.deferred:
            # A first mixed near-boundary window is neither a success sample
            # nor proof that the robot should move.  Preserve already earned
            # strict-hold evidence until the next independent window settles
            # the ambiguity.  A window with exactly one bounded position
            # outlier may additionally contribute only its individually strict
            # observations; the outlier itself never advances hold.
            self.hold_count = min(
                self.hold_frames,
                self.hold_count + decision.hold_increment,
            )
            success = False
        elif decision.aligned:
            self.hold_count = min(
                self.hold_frames,
                self.hold_count + decision.frame_count,
            )
            success = self.hold_count >= self.hold_frames
        else:
            self.hold_count = 0
            success = False
        if success:
            self.completed = True

        correction_is_nonzero = (
            decision.correction_x_m != 0.0
            or decision.correction_y_m != 0.0
            or (
                decision.consensus
                and abs(decision.dtheta_deg) > self.angle_tolerance_deg
            )
        )
        if correction_is_nonzero:
            self.pending_correction_stamp_ns = decision.stamp_ns

        return AlignmentDecision(
            stamp_ns=decision.stamp_ns,
            dx_px=decision.dx_px,
            dy_px=decision.dy_px,
            dtheta_deg=decision.dtheta_deg,
            correction_x_m=decision.correction_x_m,
            correction_y_m=decision.correction_y_m,
            aligned=decision.aligned,
            consensus=decision.consensus,
            frame_count=decision.frame_count,
            hold_count=self.hold_count,
            success=success,
            deferred=decision.deferred,
        )

    def acknowledge_settled(self, stamp_ns: int) -> bool:
        """Release one in-flight correction only for its exact source stamp."""
        stamp_ns = int(stamp_ns)
        if (
            self.pending_correction_stamp_ns is None
            or stamp_ns != self.pending_correction_stamp_ns
        ):
            return False
        self.pending_correction_stamp_ns = None
        self._observations = []
        self._pending = None
        self._deferred_correction_signature = None
        return True

    def _observation_is_strict(self, observation) -> bool:
        """Whether one fresh observation satisfies every completion bound."""
        return (
            abs(observation.dx_px) <= self.position_tolerance_px
            and abs(observation.dy_px) <= self.position_tolerance_px
            and abs(observation.dtheta_deg) <= self.angle_tolerance_deg
        )

    def _has_one_near_boundary_position_outlier(self, observations) -> bool:
        """Return true for exactly one bounded pixel-quantization excursion.

        This deliberately does not widen the completion tolerance.  The two
        strict observations may advance hold; the lone outlier is ignored and
        can never create a correction.  Any angle excursion, a second position
        outlier, or a position beyond the small near-boundary band retains the
        normal reset/correction policy.
        """
        strict = [
            self._observation_is_strict(observation)
            for observation in observations
        ]
        if strict.count(False) != 1:
            return False
        outlier = observations[strict.index(False)]
        if abs(outlier.dtheta_deg) > self.angle_tolerance_deg:
            return False
        return (
            abs(outlier.dx_px) <= self.near_boundary_limit_px
            and abs(outlier.dy_px) <= self.near_boundary_limit_px
            and (
                abs(outlier.dx_px) > self.position_tolerance_px
                or abs(outlier.dy_px) > self.position_tolerance_px
            )
        )

    def _is_mixed_near_boundary_jitter(
        self,
        observations,
        *,
        correction_x_m: float,
        correction_y_m: float,
        yaw_needs_correction: bool,
    ) -> bool:
        """Whether one correction window straddles the pixel deadband.

        This is intentionally narrower than changing the completion
        tolerance.  Every axis which would command a translation must include
        at least one strictly in-tolerance sample and all of its samples must
        stay inside the small near-boundary band.  A consistently 3-pixel (or
        larger) error therefore still commands immediately.
        """
        if yaw_needs_correction:
            return False
        corrected_axes = []
        if correction_x_m != 0.0:
            corrected_axes.append(
                [observation.dy_px for observation in observations]
            )
        if correction_y_m != 0.0:
            corrected_axes.append(
                [observation.dx_px for observation in observations]
            )
        if not corrected_axes:
            return False
        return all(
            any(
                abs(value) <= self.position_tolerance_px
                for value in values
            )
            and all(
                abs(value) <= self.near_boundary_limit_px
                for value in values
            )
            for values in corrected_axes
        )

    @staticmethod
    def _correction_signature(correction_x_m: float, correction_y_m: float):
        """Return the direction-only identity of a deferred XY command."""
        def sign(value):
            if value > 0.0:
                return 1
            if value < 0.0:
                return -1
            return 0

        return sign(correction_x_m), sign(correction_y_m)

    @staticmethod
    def _axis_state(values, tolerance):
        """Return mean, aligned, and consensus for one measured axis."""
        mean_value = sum(values) / len(values)
        if all(abs(value) <= tolerance for value in values):
            return mean_value, True, True

        positive_votes = sum(value > tolerance for value in values)
        negative_votes = sum(value < -tolerance for value in values)
        if (
            mean_value > tolerance
            and positive_votes >= 2
            and negative_votes == 0
        ):
            return mean_value, False, True
        if (
            mean_value < -tolerance
            and negative_votes >= 2
            and positive_votes == 0
        ):
            return mean_value, False, True
        return mean_value, False, False

    def _axis_correction(self, error_px: float) -> float:
        if abs(error_px) <= self.position_tolerance_px:
            return 0.0
        if error_px < 0.0:
            return self.correction_step_m
        return -self.correction_step_m

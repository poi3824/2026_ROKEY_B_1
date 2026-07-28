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


class FreshFramePairGate:
    """Accept only fresh, synchronized, strictly increasing RGB/depth pairs."""

    def __init__(
        self,
        *,
        max_skew_sec: float = 0.1,
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
    Retry the latest RGB/depth candidates when either stream advances.

    DDS may invoke the RGB callback before the matching depth callback, or in
    the opposite order.  A rejected candidate is therefore retained so that a
    newly arrived counterpart can complete the pair.  Accepted candidates are
    removed immediately and remain protected from reuse by
    :class:`FreshFramePairGate`'s strict source-stamp boundary.
    """

    def __init__(
        self,
        *,
        max_skew_sec: float = 0.1,
        max_receipt_age_sec: float = 1.0,
    ):
        self.frame_gate = FreshFramePairGate(
            max_skew_sec=max_skew_sec,
            max_receipt_age_sec=max_receipt_age_sec,
        )
        self._rgb: Optional[CachedSourceFrame] = None
        self._depth: Optional[CachedSourceFrame] = None

    def reset(
        self,
        *,
        rgb_boundary_ns: Optional[int] = None,
        depth_boundary_ns: Optional[int] = None,
    ):
        """Clear both caches and require streams newer than the boundaries."""
        self._rgb = None
        self._depth = None
        self.frame_gate.reset(
            rgb_boundary_ns=rgb_boundary_ns,
            depth_boundary_ns=depth_boundary_ns,
        )

    def cache_rgb(self, payload: Any, stamp_ns: int, receipt_ns: int):
        """Replace the latest RGB candidate."""
        self._rgb = CachedSourceFrame(
            payload=payload,
            stamp_ns=int(stamp_ns),
            receipt_ns=int(receipt_ns),
        )

    def cache_depth(self, payload: Any, stamp_ns: int, receipt_ns: int):
        """Replace the latest depth candidate."""
        self._depth = CachedSourceFrame(
            payload=payload,
            stamp_ns=int(stamp_ns),
            receipt_ns=int(receipt_ns),
        )

    def try_accept(self, *, now_receipt_ns: int):
        """Return ``(pair, reason)`` without consuming rejected candidates."""
        if self._rgb is None:
            return None, 'RGB unavailable'
        if self._depth is None:
            return None, 'depth unavailable'

        accepted, reason = self.frame_gate.accept(
            rgb_stamp_ns=self._rgb.stamp_ns,
            depth_stamp_ns=self._depth.stamp_ns,
            rgb_receipt_ns=self._rgb.receipt_ns,
            depth_receipt_ns=self._depth.receipt_ns,
            now_receipt_ns=now_receipt_ns,
        )
        if not accepted:
            return None, reason

        pair = SynchronizedFramePair(
            rgb=self._rgb,
            depth=self._depth,
        )
        self._rgb = None
        self._depth = None
        return pair, ''


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

        self.position_tolerance_px = float(position_tolerance_px)
        self.angle_tolerance_deg = float(angle_tolerance_deg)
        self.hold_frames = hold_frames
        self.correction_step_m = float(correction_step_m)
        self.decision_frames = decision_frames
        self.reset()

    def reset(self):
        """Discard frame ordering, pending correction, and convergence state."""
        self.last_seen_stamp_ns: Optional[int] = None
        self.hold_count = 0
        self.completed = False
        self.pending_correction_stamp_ns: Optional[int] = None
        self._observations = []
        self._pending: Optional[_PendingDecision] = None

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
        self._pending = _PendingDecision(
            stamp_ns=observations[-1].stamp_ns,
            dx_px=mean_dx,
            dy_px=mean_dy,
            dtheta_deg=mean_dtheta,
            correction_x_m=(
                self._axis_correction(mean_dy)
                if consensus else 0.0
            ),
            correction_y_m=(
                self._axis_correction(mean_dx)
                if consensus else 0.0
            ),
            aligned=aligned,
            consensus=consensus,
            frame_count=len(observations),
        )
        return True

    def consume(self) -> Optional[AlignmentDecision]:
        """Consume one three-target-frame decision at most once."""
        decision = self._pending
        self._pending = None
        if decision is None or self.completed:
            return None

        if decision.aligned:
            self.hold_count += decision.frame_count
        else:
            self.hold_count = 0

        success = self.hold_count >= self.hold_frames
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
        return True

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

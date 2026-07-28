"""Deterministic input and target-selection policy for camera perception."""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


MIN_TARGET_CONFIDENCE = 0.6


@dataclass(frozen=True)
class FrameGateResult:
    """Result of validating one RGB/depth source-frame pair."""

    accepted: bool
    reason: str


class CameraFrameGate:
    """Accept only fresh, synchronized, strictly increasing camera frames."""

    def __init__(self, max_skew_sec: float = 0.1, max_receipt_age_sec: float = 1.0):
        if not math.isfinite(max_skew_sec) or max_skew_sec <= 0.0:
            raise ValueError('max_skew_sec must be a finite positive number')
        if not math.isfinite(max_receipt_age_sec) or max_receipt_age_sec <= 0.0:
            raise ValueError('max_receipt_age_sec must be a finite positive number')
        self._max_skew_ns = round(max_skew_sec * 1_000_000_000)
        self._max_receipt_age_sec = max_receipt_age_sec
        self._last_rgb_stamp_ns = None
        self._last_depth_stamp_ns = None

    def reset(self, preserve_source_sequence: bool = True) -> None:
        """Reset the gate, optionally preserving the anti-replay timestamp boundary."""
        if not preserve_source_sequence:
            self._last_rgb_stamp_ns = None
            self._last_depth_stamp_ns = None

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

        rgb_age = now - rgb_received_at
        depth_age = now - depth_received_at
        if rgb_age < 0.0 or depth_age < 0.0:
            return FrameGateResult(False, 'receipt clock moved backwards')
        if (
            rgb_age > self._max_receipt_age_sec
            or depth_age > self._max_receipt_age_sec
        ):
            return FrameGateResult(
                False,
                f'stale receipt: rgb={rgb_age:.3f}s depth={depth_age:.3f}s',
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


class TargetFrameBarrier:
    """Track strictly increasing post-reset detections for each target label."""

    def __init__(self, required_frames: int = 3):
        if isinstance(required_frames, bool) or required_frames < 1:
            raise ValueError('required_frames must be a positive integer')
        self.required_frames = int(required_frames)
        self._counts = {}
        self._last_stamps = {}

    def reset(self) -> None:
        """Discard every target observation from the previous camera position."""
        self._counts.clear()
        self._last_stamps.clear()

    def record(self, target: str, stamp_ns: int) -> bool:
        """Record one target frame and return whether its barrier is now ready."""
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

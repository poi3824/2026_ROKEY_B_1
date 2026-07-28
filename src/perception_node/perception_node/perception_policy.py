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

    def reset_target(self, target: str) -> None:
        """Restart one target without disturbing independent labels."""
        self._counts.pop(target, None)
        self._last_stamps.pop(target, None)

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
    ):
        max_endpoint_distance_m = float(max_endpoint_distance_m)
        if (
            not math.isfinite(max_endpoint_distance_m)
            or max_endpoint_distance_m <= 0.0
        ):
            raise ValueError(
                'max_endpoint_distance_m must be finite and positive')
        self.max_endpoint_distance_m = max_endpoint_distance_m
        self._anchor = None

    @property
    def active(self) -> bool:
        """Return whether a physical pair anchor is currently latched."""
        return self._anchor is not None

    def reset(self) -> None:
        """Forget the current physical pair association."""
        self._anchor = None

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
) -> tuple[Mapping, Mapping] | None:
    """
    Select the same-pack pair whose ordered ``bolt_2`` is nearest center.

    The fixed camera is translated over the active station's ``bolt_2`` while
    retaining its calibrated orientation. Candidate pairs must match the
    signed ``bolt_1 -> bolt_2`` world vector (in either input order).
    Pair midpoint, geometry error, confidence, and source order are
    deterministic tie-breakers after the ``bolt_2`` endpoint distance.
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
            ranked_pairs.append((
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

    ranked_pairs.sort(key=lambda item: item[:7])
    return ranked_pairs[0][7]


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

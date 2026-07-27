#!/usr/bin/env python3
"""ROS/Isaac import 없이 비전 검출을 AMR 접근 목표로 만드는 순수 로직."""

import math
from itertools import combinations
from collections import deque
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Iterable, Tuple


class VisionUnavailable(ValueError):
    """현재 비전 표본으로 안전한 목표를 만들 수 없을 때 발생한다."""


@dataclass(frozen=True)
class DetectionCandidate:
    x: float
    y: float
    z: float
    score: float


@dataclass(frozen=True)
class VisionSnapshot:
    x: float
    y: float
    z: float
    min_score: float
    sample_count: int
    std_xy_m: float
    age_sec: float
    source_stamp_ns: int
    pair_distance_m: float
    pair_distance_std_m: float


@dataclass(frozen=True)
class WorldGoal:
    x: float
    y: float
    yaw: float


_SCAN_SEARCH_PATTERN = (
    (0.0, 0.0),
    (0.0, +1.0),
    (0.0, -1.0),
)


def compute_busbar_scan_xy(
    target_xy: Tuple[float, float],
    approach_yaw_rad: float,
    attempt_index: int,
    step_m: float,
) -> Tuple[float, float]:
    """선택 busbar 중심 주위의 제한된 손목-camera 탐색 XY를 반환한다.

    pattern의 첫 자세는 검출 좌표 그 자체다. 이후 자세는 작업 셀 접근 normal과
    tangent를 기준으로만 움직이며, 절대 station 좌표는 사용하지 않는다.
    """
    target_x, target_y = target_xy
    if (
        isinstance(attempt_index, bool)
        or not isinstance(attempt_index, int)
        or attempt_index < 0
    ):
        raise ValueError("attempt_index는 0 이상의 정수여야 합니다")
    if not all(
        math.isfinite(value)
        for value in (
            target_x,
            target_y,
            approach_yaw_rad,
            step_m,
        )
    ):
        raise ValueError("scan target/yaw/step에 NaN/inf가 있습니다")
    if step_m <= 0.0:
        raise ValueError("scan 탐색 step은 양수여야 합니다")

    normal_steps, tangent_steps = _SCAN_SEARCH_PATTERN[
        attempt_index % len(_SCAN_SEARCH_PATTERN)
    ]
    normal_x = math.cos(approach_yaw_rad)
    normal_y = math.sin(approach_yaw_rad)
    tangent_x = -normal_y
    tangent_y = normal_x
    return (
        target_x
        + step_m
        * (normal_steps * normal_x + tangent_steps * tangent_x),
        target_y
        + step_m
        * (normal_steps * normal_y + tangent_steps * tangent_y),
    )


def compute_busbar_pick_targets(
    target_xy: Tuple[float, float],
    pick_z: float,
    approach_z: float,
):
    """비전 XY를 유지하는 접근·파지·수직 상승 목표를 반환한다."""
    target_x, target_y = target_xy
    values = (target_x, target_y, pick_z, approach_z)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("busbar pick target에 NaN/inf가 있습니다")
    if approach_z <= pick_z:
        raise ValueError("busbar approach_z는 pick_z보다 커야 합니다")

    approach = (target_x, target_y, approach_z)
    pick = (target_x, target_y, pick_z)
    lift = (target_x, target_y, approach_z)
    return approach, pick, lift


def select_nearest_busbar_path(
    target_xy: Tuple[float, float],
    candidates,
    max_distance_m: float,
) -> str:
    """비전 target과 가장 가까운 유효 USD busbar prim path를 고른다.

    candidates는 ``(path, world_x, world_y)`` iterable이다. 좌표 자체를
    하드코딩하지 않고 현재 stage의 world transform을 입력으로 받는다.
    """
    target_x, target_y = target_xy
    if not all(
        math.isfinite(value)
        for value in (target_x, target_y, max_distance_m)
    ):
        raise ValueError("busbar prim 연관 값에 NaN/inf가 있습니다")
    if max_distance_m <= 0.0:
        raise ValueError("busbar prim 연관 최대 거리는 양수여야 합니다")

    ranked = []
    for path, world_x, world_y in candidates:
        if not isinstance(path, str) or not path:
            raise ValueError("busbar prim path는 비어 있지 않은 문자열이어야 합니다")
        if not all(math.isfinite(value) for value in (world_x, world_y)):
            raise ValueError("busbar prim world 좌표에 NaN/inf가 있습니다")
        ranked.append(
            (
                math.hypot(world_x - target_x, world_y - target_y),
                path,
            )
        )

    if not ranked:
        raise VisionUnavailable("USD에 유효한 busbar prim 후보가 없습니다")
    ranked.sort()
    distance, path = ranked[0]
    if distance > max_distance_m:
        raise VisionUnavailable(
            "비전 busbar와 USD prim의 최근접 거리가 "
            f"{distance:.3f}m로 제한 {max_distance_m:.3f}m를 초과합니다"
        )
    return path


def select_highest_confidence(
    candidates: Iterable[DetectionCandidate],
) -> Tuple[DetectionCandidate, ...]:
    """복수 후보 중 유일하게 confidence가 가장 높은 후보 하나를 반환한다."""
    candidates = tuple(candidates)
    if not candidates:
        return ()

    if not all(math.isfinite(candidate.score) for candidate in candidates):
        raise VisionUnavailable("후보 confidence에 NaN/inf가 있습니다")

    ranked = sorted(
        candidates,
        key=lambda candidate: candidate.score,
        reverse=True,
    )
    if (
        len(ranked) > 1
        and ranked[0].score == ranked[1].score
    ):
        raise VisionUnavailable(
            "최고 confidence가 같은 후보가 여러 개라 하나를 선택할 수 없습니다"
        )
    return (ranked[0],)


def select_nearest_to_reference(
    candidates: Iterable[DetectionCandidate],
    reference_xy: Tuple[float, float],
) -> Tuple[DetectionCandidate, ...]:
    """주행 시작 때 선택한 대상을 XY 최근접 연관으로 계속 유지한다."""
    candidates = tuple(candidates)
    if not candidates:
        return ()

    reference_x, reference_y = reference_xy
    if not all(
        math.isfinite(value)
        for value in (reference_x, reference_y)
    ):
        raise VisionUnavailable("대상 연관 기준 좌표에 NaN/inf가 있습니다")
    if not all(
        math.isfinite(value)
        for candidate in candidates
        for value in (
            candidate.x,
            candidate.y,
            candidate.z,
            candidate.score,
        )
    ):
        raise VisionUnavailable("후보 좌표 또는 confidence에 NaN/inf가 있습니다")

    ranked = sorted(
        candidates,
        key=lambda candidate: math.hypot(
            candidate.x - reference_x,
            candidate.y - reference_y,
        ),
    )
    if len(ranked) > 1:
        nearest_distances = [
            math.hypot(
                candidate.x - reference_x,
                candidate.y - reference_y,
            )
            for candidate in ranked[:2]
        ]
        if math.isclose(
            nearest_distances[0],
            nearest_distances[1],
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise VisionUnavailable(
                "기존 대상과 같은 거리에 있는 후보가 여러 개라 "
                "동일 대상을 연관할 수 없습니다"
            )
    return (ranked[0],)


def select_nearest_valid_pair(
    candidates: Iterable[DetectionCandidate],
    reference_xy: Tuple[float, float],
    *,
    expected_spacing_m: float,
    spacing_tolerance_m: float,
) -> Tuple[DetectionCandidate, DetectionCandidate]:
    """물체 치수에 맞는 두 검출을 묶고 그 중 reference에 가장 가까운 쌍을 고른다.

    reference는 현재 AMR 위치(최초 선택) 또는 이미 선택한 쌍의 중점(주행 중 추적)이다.
    과거 station 좌표나 후보 순서는 사용하지 않는다.
    """
    candidates = tuple(candidates)
    reference_x, reference_y = reference_xy
    values = (
        reference_x,
        reference_y,
        expected_spacing_m,
        spacing_tolerance_m,
        *(
            value
            for candidate in candidates
            for value in (
                candidate.x,
                candidate.y,
                candidate.z,
                candidate.score,
            )
        ),
    )
    if not all(math.isfinite(value) for value in values):
        raise VisionUnavailable(
            "볼트쌍 선택 좌표, 치수 또는 confidence에 NaN/inf가 있습니다"
        )
    if expected_spacing_m <= 0.0 or spacing_tolerance_m <= 0.0:
        raise ValueError("볼트쌍 기준 간격과 허용오차는 양수여야 합니다")
    if len(candidates) < 2:
        raise VisionUnavailable(
            f"볼트 후보 수 {len(candidates)}개 (최소 2개 필요)"
        )

    valid_pairs = []
    for first, second in combinations(candidates, 2):
        spacing = math.hypot(
            second.x - first.x,
            second.y - first.y,
        )
        spacing_error = abs(spacing - expected_spacing_m)
        if spacing_error > spacing_tolerance_m:
            continue
        midpoint_x = 0.5 * (first.x + second.x)
        midpoint_y = 0.5 * (first.y + second.y)
        reference_distance = math.hypot(
            midpoint_x - reference_x,
            midpoint_y - reference_y,
        )
        valid_pairs.append(
            (
                reference_distance,
                spacing_error,
                -(first.score + second.score),
                midpoint_x,
                midpoint_y,
                first,
                second,
            )
        )

    if not valid_pairs:
        minimum_error = min(
            abs(
                math.hypot(
                    second.x - first.x,
                    second.y - first.y,
                )
                - expected_spacing_m
            )
            for first, second in combinations(candidates, 2)
        )
        raise VisionUnavailable(
            "유효한 볼트쌍이 없습니다: "
            f"기준 간격 {expected_spacing_m:.3f}m ± "
            f"{spacing_tolerance_m:.3f}m, "
            f"최소 간격 오차 {minimum_error:.3f}m"
        )

    valid_pairs.sort(key=lambda item: item[:5])
    if (
        len(valid_pairs) > 1
        and math.isclose(
            valid_pairs[0][0],
            valid_pairs[1][0],
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
        and math.isclose(
            valid_pairs[0][1],
            valid_pairs[1][1],
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
        and math.isclose(
            valid_pairs[0][2],
            valid_pairs[1][2],
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
    ):
        raise VisionUnavailable(
            "AMR와 같은 거리에 있는 유효 볼트쌍이 여러 개라 "
            "대상을 결정할 수 없습니다"
        )

    selected = valid_pairs[0]
    return (selected[5], selected[6])


class DetectionWindow:
    """후보 수·신뢰도·연속 표본·분산·신선도를 모두 통과한 좌표만 반환한다."""

    def __init__(
        self,
        *,
        expected_candidates: int,
        min_score: float,
        min_samples: int,
        max_samples: int,
        max_age_sec: float,
        max_std_m: float,
    ):
        if expected_candidates < 1:
            raise ValueError("expected_candidates는 1 이상이어야 합니다")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score는 0~1 범위여야 합니다")
        if min_samples < 1 or max_samples < min_samples:
            raise ValueError("표본 수 설정이 올바르지 않습니다")
        if max_age_sec <= 0.0 or max_std_m <= 0.0:
            raise ValueError("age/std 제한은 양수여야 합니다")

        self.expected_candidates = expected_candidates
        self.min_score = min_score
        self.min_samples = min_samples
        self.max_age_sec = max_age_sec
        self.max_std_m = max_std_m
        self._samples = deque(maxlen=max_samples)
        self.last_rejection = "아직 비전 표본이 없습니다"

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def reset(self, reason: str):
        self._samples.clear()
        self.last_rejection = reason

    def note_transient_rejection(self, reason: str):
        """일시 dropout 사유만 기록하고 이미 검증한 표본은 유지한다."""
        self.last_rejection = reason

    def add_frame(
        self,
        candidates: Iterable[DetectionCandidate],
        received_monotonic: float,
        source_stamp_ns: int,
    ) -> bool:
        candidates = tuple(candidates)
        if len(candidates) != self.expected_candidates:
            self.reset(
                f"후보 수 {len(candidates)}개 "
                f"(필요 {self.expected_candidates}개)"
            )
            return False
        if not math.isfinite(received_monotonic):
            self.reset("수신 시간이 유한하지 않습니다")
            return False
        if (
            not isinstance(source_stamp_ns, int)
            or isinstance(source_stamp_ns, bool)
            or source_stamp_ns < 0
        ):
            self.reset("source timestamp가 유효하지 않습니다")
            return False
        if self._samples and source_stamp_ns <= self._samples[-1][2]:
            self.reset(
                "동일하거나 과거인 source frame timestamp를 거부했습니다"
            )
            return False

        for candidate in candidates:
            values = (
                candidate.x,
                candidate.y,
                candidate.z,
                candidate.score,
            )
            if not all(math.isfinite(value) for value in values):
                self.reset("검출 좌표 또는 score에 NaN/inf가 있습니다")
                return False
            if candidate.score < self.min_score:
                self.reset(
                    f"검출 score {candidate.score:.3f}가 "
                    f"최소 {self.min_score:.3f}보다 낮습니다"
                )
                return False

        # 볼트쌍은 두 점의 순서나 과거 하드코딩 좌표를 사용하지 않고 순수 중점만 쓴다.
        count = float(len(candidates))
        centroid = DetectionCandidate(
            x=sum(item.x for item in candidates) / count,
            y=sum(item.y for item in candidates) / count,
            z=sum(item.z for item in candidates) / count,
            score=min(item.score for item in candidates),
        )
        pair_distance = float("nan")
        if len(candidates) == 2:
            pair_distance = math.hypot(
                candidates[1].x - candidates[0].x,
                candidates[1].y - candidates[0].y,
            )
            if not math.isfinite(pair_distance) or pair_distance <= 0.0:
                self.reset("볼트쌍 간격이 유효한 양수가 아닙니다")
                return False

        self._samples.append(
            (
                centroid,
                received_monotonic,
                source_stamp_ns,
                pair_distance,
            )
        )
        self.last_rejection = ""
        return True

    def snapshot(
        self,
        now_monotonic: float,
        *,
        max_age_sec: float = None,
        enforce_age: bool = True,
    ) -> VisionSnapshot:
        if len(self._samples) < self.min_samples:
            detail = self.last_rejection or (
                f"연속 표본 부족: {len(self._samples)}/{self.min_samples}"
            )
            raise VisionUnavailable(detail)

        age_sec = now_monotonic - self._samples[-1][1]
        if not math.isfinite(age_sec) or age_sec < 0.0:
            raise VisionUnavailable("비전 수신 시간 순서가 올바르지 않습니다")
        if enforce_age:
            age_limit = self.max_age_sec if max_age_sec is None else max_age_sec
            if not math.isfinite(age_limit) or age_limit <= 0.0:
                raise ValueError("snapshot max_age_sec은 유한한 양수여야 합니다")
            if age_sec > age_limit:
                raise VisionUnavailable(
                    f"비전 좌표가 {age_sec:.2f}s 전 데이터로 오래됐습니다 "
                    f"(제한 {age_limit:.2f}s)"
                )

        points = [item[0] for item in self._samples]
        xs = [item.x for item in points]
        ys = [item.y for item in points]
        zs = [item.z for item in points]
        std_x = pstdev(xs)
        std_y = pstdev(ys)
        std_xy_m = max(std_x, std_y)
        if std_xy_m > self.max_std_m:
            raise VisionUnavailable(
                f"비전 XY 표준편차 {std_xy_m * 1000.0:.1f}mm가 "
                f"제한 {self.max_std_m * 1000.0:.1f}mm보다 큽니다"
            )

        pair_distances = [
            item[3]
            for item in self._samples
            if math.isfinite(item[3])
        ]
        pair_distance_m = (
            fmean(pair_distances) if pair_distances else float("nan")
        )
        pair_distance_std_m = (
            pstdev(pair_distances) if pair_distances else float("nan")
        )
        if (
            pair_distances
            and pair_distance_std_m > self.max_std_m
        ):
            raise VisionUnavailable(
                f"볼트쌍 간격 표준편차 "
                f"{pair_distance_std_m * 1000.0:.1f}mm가 "
                f"제한 {self.max_std_m * 1000.0:.1f}mm보다 큽니다"
            )

        return VisionSnapshot(
            x=fmean(xs),
            y=fmean(ys),
            z=fmean(zs),
            min_score=min(item.score for item in points),
            sample_count=len(points),
            std_xy_m=std_xy_m,
            age_sec=age_sec,
            source_stamp_ns=self._samples[-1][2],
            pair_distance_m=pair_distance_m,
            pair_distance_std_m=pair_distance_std_m,
        )


def compute_standoff_goal(
    *,
    current_xy: Tuple[float, float],
    target_xy: Tuple[float, float],
    standoff_m: float,
) -> WorldGoal:
    """현재 위치에서 검출점을 바라보며 지정 거리 앞에 멈추는 world 목표를 계산한다."""
    values = (*current_xy, *target_xy, standoff_m)
    if not all(math.isfinite(value) for value in values):
        raise VisionUnavailable("접근 목표 계산값에 NaN/inf가 있습니다")
    if standoff_m <= 0.0:
        raise VisionUnavailable("standoff_m은 명시적인 양수여야 합니다")

    dx = target_xy[0] - current_xy[0]
    dy = target_xy[1] - current_xy[1]
    distance = math.hypot(dx, dy)
    if distance <= standoff_m:
        raise VisionUnavailable(
            f"검출점까지 거리 {distance:.3f}m가 "
            f"standoff {standoff_m:.3f}m 이하입니다"
        )

    ux = dx / distance
    uy = dy / distance
    return WorldGoal(
        x=target_xy[0] - ux * standoff_m,
        y=target_xy[1] - uy * standoff_m,
        yaw=math.atan2(dy, dx),
    )


def compute_directional_standoff_goal(
    *,
    target_xy: Tuple[float, float],
    approach_yaw_rad: float,
    standoff_m: float,
    docking_yaw_offset_rad: float = 0.0,
) -> WorldGoal:
    """작업 셀 normal 위치와 장착물 방향을 함께 반영한 docking 목표를 계산한다."""
    values = (
        *target_xy,
        approach_yaw_rad,
        standoff_m,
        docking_yaw_offset_rad,
    )
    if not all(math.isfinite(value) for value in values):
        raise VisionUnavailable("방향 지정 접근 목표에 NaN/inf가 있습니다")
    if standoff_m <= 0.0:
        raise VisionUnavailable("standoff_m은 명시적인 양수여야 합니다")

    approach_yaw = (
        (approach_yaw_rad + math.pi) % (2.0 * math.pi) - math.pi
    )
    docking_yaw = (
        (approach_yaw + docking_yaw_offset_rad + math.pi)
        % (2.0 * math.pi)
        - math.pi
    )
    return WorldGoal(
        x=target_xy[0] - math.cos(approach_yaw) * standoff_m,
        y=target_xy[1] - math.sin(approach_yaw) * standoff_m,
        yaw=docking_yaw,
    )

#!/usr/bin/env python3
"""로봇 관절 명령의 불연속 점프를 제한하는 순수 계산 유틸리티."""

import math
from typing import Iterable, Tuple


def limit_revolute_joint_step(
    current_positions: Iterable[float],
    target_positions: Iterable[float],
    max_step_rad: float,
) -> Tuple[Tuple[float, ...], bool]:
    """최단 각도 기준으로 각 관절의 한 프레임 이동량을 제한한다."""

    current = tuple(float(value) for value in current_positions)
    target = tuple(float(value) for value in target_positions)
    max_step_rad = float(max_step_rad)
    if not current or len(current) != len(target):
        raise ValueError("current/target 관절 배열 길이가 같고 비어 있지 않아야 합니다")
    if not all(math.isfinite(value) for value in (*current, *target)):
        raise ValueError("관절값에 NaN/inf가 있습니다")
    if not math.isfinite(max_step_rad) or max_step_rad <= 0.0:
        raise ValueError("max_step_rad는 유한한 양수여야 합니다")

    limited = []
    clipped = False
    for current_value, target_value in zip(current, target):
        delta = (
            target_value - current_value + math.pi
        ) % (2.0 * math.pi) - math.pi
        limited_delta = max(-max_step_rad, min(max_step_rad, delta))
        if not math.isclose(
            limited_delta,
            delta,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            clipped = True
        limited.append(current_value + limited_delta)
    return tuple(limited), clipped

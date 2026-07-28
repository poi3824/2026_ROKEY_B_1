"""육각 너트 외곽 면에 평행 그리퍼를 정렬하는 순수 기하 계획기."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


Point2D = Tuple[float, float]
_COLLINEAR_EPSILON = 1.0e-9


@dataclass(frozen=True)
class FlatGraspPlan:
    """선택한 평면 법선과 그리퍼에 적용할 월드 Z 회전량."""

    yaw_delta_deg: float
    face_normal_yaw_deg: float
    edge_length: float
    hull_vertex_count: int


def _cross(origin: Point2D, a: Point2D, b: Point2D) -> float:
    return (
        (a[0] - origin[0]) * (b[1] - origin[1])
        - (a[1] - origin[1]) * (b[0] - origin[0])
    )


def convex_hull_2d(points: Iterable[Sequence[float]]) -> list[Point2D]:
    """Andrew monotone chain으로 중복/내부점을 제거한 CCW 외곽을 구한다."""

    unique = sorted({
        (round(float(point[0]), 9), round(float(point[1]), 9))
        for point in points
        if len(point) >= 2
        and math.isfinite(float(point[0]))
        and math.isfinite(float(point[1]))
    })
    if len(unique) < 3:
        return []

    lower: list[Point2D] = []
    for point in unique:
        while (
            len(lower) >= 2
            and _cross(lower[-2], lower[-1], point) <= _COLLINEAR_EPSILON
        ):
            lower.pop()
        lower.append(point)

    upper: list[Point2D] = []
    for point in reversed(unique):
        while (
            len(upper) >= 2
            and _cross(upper[-2], upper[-1], point) <= _COLLINEAR_EPSILON
        ):
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _unoriented_axis_delta(target: float, current: float) -> float:
    """평행 그리퍼 축은 180도 대칭이므로 최소 회전량을 [-pi/2, pi/2)로 반환."""

    return (target - current + math.pi / 2.0) % math.pi - math.pi / 2.0


def plan_flat_grasp(
    world_xy_points: Iterable[Sequence[float]],
    base_closing_axis_yaw_rad: float,
    *,
    min_edge_ratio: float = 0.25,
) -> FlatGraspPlan:
    """Mesh 외곽 직선 변의 법선에 그리퍼 닫힘축을 맞춘다.

    육각 너트의 꼭짓점을 향해 닫는 대신, 볼록 외곽의 실제 변과 수직인 축 후보 중
    현재 그리퍼 자세에서 회전량이 가장 작은 것을 선택한다. 같은 직선 위에 Mesh
    vertex가 여러 개 있어도 짧은 조각을 제외해 모따기/삼각분할에 끌려가지 않는다.
    """

    if not 0.0 < min_edge_ratio <= 1.0:
        raise ValueError("min_edge_ratio는 (0, 1] 범위여야 합니다")
    if not math.isfinite(base_closing_axis_yaw_rad):
        raise ValueError("그리퍼 닫힘축 yaw가 유효하지 않습니다")

    hull = convex_hull_2d(world_xy_points)
    if len(hull) < 3:
        raise ValueError("너트 외곽을 계산할 Mesh 점이 부족합니다")

    edges = []
    for index, start in enumerate(hull):
        end = hull[(index + 1) % len(hull)]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length > 1.0e-9:
            edges.append((length, math.atan2(dy, dx)))
    if not edges:
        raise ValueError("너트 외곽에서 유효한 직선 변을 찾지 못했습니다")

    minimum_length = max(length for length, _ in edges) * min_edge_ratio
    candidates = []
    for length, edge_yaw in edges:
        if length < minimum_length:
            continue
        normal_yaw = edge_yaw + math.pi / 2.0
        delta = _unoriented_axis_delta(
            normal_yaw, base_closing_axis_yaw_rad
        )
        candidates.append((abs(delta), -length, delta, normal_yaw, length))
    if not candidates:
        raise ValueError("너트 면 후보를 찾지 못했습니다")

    _, _, delta, normal_yaw, length = min(candidates)
    normalized_normal = (normal_yaw + math.pi) % (2.0 * math.pi) - math.pi
    return FlatGraspPlan(
        yaw_delta_deg=math.degrees(delta),
        face_normal_yaw_deg=math.degrees(normalized_normal),
        edge_length=length,
        hull_vertex_count=len(hull),
    )

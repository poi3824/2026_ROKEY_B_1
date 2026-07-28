import math

import pytest

from nut_grasp_planner import (
    convex_hull_2d,
    plan_flat_grasp,
    rg2_gap_for_joint,
    rg2_joint_for_width,
)


def _hexagon(radius=1.0, rotation_deg=0.0):
    rotation = math.radians(rotation_deg)
    return [
        (
            radius * math.cos(rotation + index * math.pi / 3.0),
            radius * math.sin(rotation + index * math.pi / 3.0),
        )
        for index in range(6)
    ]


def test_rotated_hex_selects_nearest_flat_instead_of_corner():
    # 꼭짓점이 17도에 있으면 면 법선은 47/107/167도다. 닫힘축 0도에서는
    # 167도와 같은 축인 -13도 면이 가장 가까워야 한다.
    plan = plan_flat_grasp(_hexagon(rotation_deg=17.0), 0.0)

    assert plan.yaw_delta_deg == pytest.approx(-13.0)
    assert plan.hull_vertex_count == 6


def test_selected_closing_axis_is_perpendicular_to_real_hull_edge():
    points = _hexagon(rotation_deg=8.0)
    plan = plan_flat_grasp(points, math.radians(35.0))
    hull = convex_hull_2d(points)
    target = math.radians(plan.face_normal_yaw_deg)

    perpendicular_to_an_edge = False
    for index, start in enumerate(hull):
        end = hull[(index + 1) % len(hull)]
        edge_yaw = math.atan2(end[1] - start[1], end[0] - start[0])
        dot = abs(math.cos(target - edge_yaw))
        perpendicular_to_an_edge |= dot < 1.0e-9
    assert perpendicular_to_an_edge


def test_collinear_mesh_vertices_do_not_change_flat_selection():
    points = _hexagon(rotation_deg=0.0)
    subdivided = []
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        subdivided.extend([
            start,
            ((2.0 * start[0] + end[0]) / 3.0,
             (2.0 * start[1] + end[1]) / 3.0),
            ((start[0] + 2.0 * end[0]) / 3.0,
             (start[1] + 2.0 * end[1]) / 3.0),
        ])

    plan = plan_flat_grasp(subdivided, 0.0)
    assert abs(plan.yaw_delta_deg) == pytest.approx(30.0)
    assert plan.hull_vertex_count == 6


def test_invalid_mesh_points_are_rejected():
    with pytest.raises(ValueError):
        plan_flat_grasp([(0.0, 0.0), (1.0, 0.0)], 0.0)


def test_hex_grasp_width_is_across_flats_not_across_corners():
    plan = plan_flat_grasp(_hexagon(radius=0.013, rotation_deg=11.0), 0.0)

    assert plan.grasp_width == pytest.approx(
        2.0 * 0.013 * math.cos(math.pi / 6.0)
    )
    assert plan.grasp_width < 2.0 * 0.013


def test_rg2_joint_target_matches_nut_width_with_preload():
    nut_width = 0.022606
    preload = 0.0003
    joint = rg2_joint_for_width(nut_width, preload)

    assert joint == pytest.approx(0.98398, abs=1.0e-4)
    assert rg2_gap_for_joint(joint) == pytest.approx(nut_width - preload)
    assert rg2_gap_for_joint(0.961) > nut_width

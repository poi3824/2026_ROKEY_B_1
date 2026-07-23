"""
busbar_mesh.py
--------------
버스바(busbar) 프로시저럴 메쉬 생성 로직.
명세서 8장: "사각형 - 원2개 -> triangulate" 를 mapbox_earcut으로 구현.

- 시각 메쉬(Visual): 볼트홀 2개가 실제로 뚫린 형상. GeomSubset으로 상/하면(CoatedFace)과
  측면+홀내벽(CopperFace)을 분리해둠 (텍스처/머티리얼 배정용, 가정치이니 실제 외관에 맞게 조정 요망).
- 충돌 프록시(Collision): 홀 없는 단순 박스. 명세서 8장 코멘트대로 "convexHull이 어차피
  홀을 메꿔버리므로" 홀 형상을 충돌 계산에 넣는 건 낭비 -> 박스 하나로 대체.

전부 numpy 배열로 반환하고, USD 오써링은 make_busbar_usd.py 에서 담당 (관심사 분리).
"""

import numpy as np
import mapbox_earcut as earcut


def _circle_points(cx, cy, radius, segments):
    """중심(cx,cy), 반지름 radius인 원을 segments개 정점으로 근사 (CCW, +Z에서 바라볼 때)."""
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    return np.stack([xs, ys], axis=1)


def _rect_ring(length, width):
    """직사각형 외곽 링, CCW (+Z에서 바라볼 때), (-L/2,-W/2)에서 시작."""
    hl, hw = length / 2.0, width / 2.0
    return np.array(
        [[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]],
        dtype=np.float64,
    )


def _fix_winding_outward(tri, pts3d, ref_xy):
    """
    tri = (i, j, k) 인덱스 튜플, pts3d = (N,3) 정점 배열.
    삼각형 법선의 XY 성분이 ref_xy 방향을 향하도록 필요시 정점 순서를 뒤집어서 반환.
    (측벽/홀내벽 법선 방향 보정용)
    """
    i, j, k = tri
    p0, p1, p2 = pts3d[i], pts3d[j], pts3d[k]
    normal = np.cross(p1 - p0, p2 - p0)
    if np.dot(normal[:2], ref_xy) < 0:
        return (i, k, j)
    return tri


def build_busbar_visual_mesh(length, width, thickness, hole_dia, hole_pitch, hole_segments=16):
    """
    볼트홀 2개가 뚫린 버스바 시각 메쉬를 생성.

    Returns
    -------
    points : (N,3) float32 ndarray
    face_vertex_counts : (F,) int32 ndarray, 전부 3 (삼각형)
    face_vertex_indices : (3F,) int32 ndarray
    subset_faces : dict[str, list[int]]  # 페이스 인덱스(0-based, face_vertex_counts 기준)
    """
    hole_r = hole_dia / 2.0
    hole_centers = [(-hole_pitch / 2.0, 0.0), (hole_pitch / 2.0, 0.0)]

    outer2d = _rect_ring(length, width)
    hole0_2d = _circle_points(hole_centers[0][0], hole_centers[0][1], hole_r, hole_segments)
    hole1_2d = _circle_points(hole_centers[1][0], hole_centers[1][1], hole_r, hole_segments)

    ring2d = np.vstack([outer2d, hole0_2d, hole1_2d])
    ring_ends = np.array(
        [len(outer2d), len(outer2d) + len(hole0_2d), len(outer2d) + len(hole0_2d) + len(hole1_2d)],
        dtype=np.uint32,
    )
    tri_idx_flat = earcut.triangulate_float64(ring2d, ring_ends)
    top_tris = tri_idx_flat.reshape(-1, 3).astype(np.int64)  # earcut 결과는 CCW == +Z 법선

    n_ring = len(ring2d)
    t_half = thickness / 2.0

    top_pts = np.hstack([ring2d, np.full((n_ring, 1), t_half)])
    bot_pts = np.hstack([ring2d, np.full((n_ring, 1), -t_half)])
    points = np.vstack([top_pts, bot_pts])  # [0:n_ring)=top, [n_ring:2n_ring)=bottom
    bottom_offset = n_ring

    faces = []
    subset_top_bottom = []  # CoatedFace
    subset_sides = []  # CopperFace

    # 상면 (법선 +Z, earcut 결과 그대로)
    for tri in top_tris:
        faces.append(tuple(int(v) for v in tri))
        subset_top_bottom.append(len(faces) - 1)

    # 하면 (법선 -Z, winding 반전 + bottom 오프셋)
    for tri in top_tris:
        i, j, k = tri
        faces.append((int(k) + bottom_offset, int(j) + bottom_offset, int(i) + bottom_offset))
        subset_top_bottom.append(len(faces) - 1)

    def add_wall_ring(start, count, ref_dir_fn):
        """start..start+count-1 범위의 2D 링(외곽 or 홀)에 대해 측벽 생성."""
        for a in range(count):
            b = (a + 1) % count
            ti, tj = start + a, start + b
            bi, bj = bottom_offset + start + a, bottom_offset + start + b
            centroid_xy = ring2d[[start + a, start + b]].mean(axis=0)
            ref = ref_dir_fn(centroid_xy)
            tri1 = _fix_winding_outward((ti, bi, bj), points, ref)
            tri2 = _fix_winding_outward((ti, bj, tj), points, ref)
            faces.append(tri1)
            subset_sides.append(len(faces) - 1)
            faces.append(tri2)
            subset_sides.append(len(faces) - 1)

    # 외곽 측벽: 바깥쪽으로 향하는 법선
    add_wall_ring(0, len(outer2d), lambda c: np.array([c[0], c[1]]))

    # 홀 내벽: 홀 중심 쪽(안쪽)으로 향하는 법선
    off = len(outer2d)
    add_wall_ring(off, len(hole0_2d), lambda c: np.array([hole_centers[0][0] - c[0], hole_centers[0][1] - c[1]]))
    off += len(hole0_2d)
    add_wall_ring(off, len(hole1_2d), lambda c: np.array([hole_centers[1][0] - c[0], hole_centers[1][1] - c[1]]))

    face_vertex_counts = np.full(len(faces), 3, dtype=np.int32)
    face_vertex_indices = np.array(faces, dtype=np.int32).flatten()

    subset_faces = {"CoatedFace": subset_top_bottom, "CopperFace": subset_sides}
    return points.astype(np.float32), face_vertex_counts, face_vertex_indices, subset_faces


def build_box_mesh(length, width, thickness):
    """충돌 프록시용 단순 박스 (8정점, 12삼각형), 홀 없음."""
    hl, hw, ht = length / 2.0, width / 2.0, thickness / 2.0
    points = np.array(
        [
            [-hl, -hw, -ht], [hl, -hw, -ht], [hl, hw, -ht], [-hl, hw, -ht],  # bottom 0-3
            [-hl, -hw, ht], [hl, -hw, ht], [hl, hw, ht], [-hl, hw, ht],      # top 4-7
        ],
        dtype=np.float32,
    )
    # 각 면 CCW(바깥에서 봤을 때)
    quads = [
        (0, 3, 2, 1),  # bottom (-Z)
        (4, 5, 6, 7),  # top (+Z)
        (0, 1, 5, 4),  # -Y
        (1, 2, 6, 5),  # +X
        (2, 3, 7, 6),  # +Y
        (3, 0, 4, 7),  # -X
    ]
    faces = []
    for a, b, c, d in quads:
        faces.append((a, b, c))
        faces.append((a, c, d))
    face_vertex_counts = np.full(len(faces), 3, dtype=np.int32)
    face_vertex_indices = np.array(faces, dtype=np.int32).flatten()
    return points, face_vertex_counts, face_vertex_indices

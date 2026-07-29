"""Unit and contract tests for deterministic busbar-hole selection."""

import ast
import math
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).parents[1] / 'error_fix' / 'error_fix.py'
)


def _load_selector():
    source = SOURCE_PATH.read_text(encoding='utf-8')
    tree = ast.parse(source)
    selector = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == 'select_closest_hole_candidate'
    )
    module = ast.Module(body=[selector], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {'math': math}
    exec(compile(module, SOURCE_PATH, 'exec'), namespace)
    return namespace['select_closest_hole_candidate']


def test_closest_candidate_wins_instead_of_leftmost_candidate():
    select = _load_selector()

    selected = select(
        [
            (308, 248, 0.42),
            (318, 242, 0.43),
            (340, 240, 0.44),
        ],
        (320, 240),
    )

    assert selected == (318, 242, 0.43)


def test_selection_rejects_invalid_reference_and_depth_candidates():
    select = _load_selector()

    candidates = [
        (320, 240, math.nan),
        (320, 240, math.inf),
        (320, 240, 0.0),
        (320, 240, -0.1),
        (324, 240, 0.4),
    ]

    assert select(candidates, (math.nan, 240)) is None
    assert select(candidates, (320, 240)) == (324, 240, 0.4)


def test_equal_distance_selection_has_a_stable_coordinate_tiebreak():
    select = _load_selector()

    selected = select(
        [
            (322, 240, 0.5),
            (318, 240, 0.5),
        ],
        (320, 240),
    )

    assert selected == (318, 240, 0.5)


def test_detector_filters_bounds_mask_and_depth_before_reference_selection():
    source = SOURCE_PATH.read_text(encoding='utf-8')
    detector_source = source[
        source.index('def detect_yellow_busbar_holes_depth'):
        source.index('def draw_extended_line')
    ]

    assert '0 <= cx < w and 0 <= cy < h' in detector_source
    assert 'filtered_mask[cy, cx] == 0' in detector_source
    assert 'not math.isfinite(val) or val <= 0.0' in detector_source
    assert 'if self.fixed_bolt_coords:' in detector_source
    assert 'select_closest_hole_candidate(' in detector_source
    assert 'detected_holes.sort(key=lambda item: item[0])' not in (
        detector_source
    )

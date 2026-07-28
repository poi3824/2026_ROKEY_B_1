"""Regression test for the ordered error-fix reset command path."""

import ast
from pathlib import Path


def _method_calls(source, method_name):
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    return {
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }


def test_error_fix_has_only_the_ordered_string_reset_path():
    source = (
        Path(__file__).parents[1] / 'error_fix' / 'error_fix.py'
    ).read_text(encoding='utf-8')

    assert "'/errorfix_command'" in source
    assert '"RESET_BOLT_DETECTION"' in source
    assert '/bolt_cam/perception/reset_cache' not in source
    assert 'std_msgs.msg import Empty' not in source


def test_start_and_camera_reset_clear_alignment_frame_state():
    source = (
        Path(__file__).parents[1] / 'error_fix' / 'error_fix.py'
    ).read_text(encoding='utf-8')

    assert '_reset_bolt_detection' in _method_calls(
        source, 'task_command_callback')
    assert '_reset_alignment_tracking' in _method_calls(
        source, '_reset_bolt_detection')
    assert 'reset' in _method_calls(source, '_reset_alignment_tracking')
    assert 'reset' in _method_calls(source, '_reset_bolt_detection')


def test_error_fix_requires_synchronized_post_move_frames():
    source = (
        Path(__file__).parents[1] / 'error_fix' / 'error_fix.py'
    ).read_text(encoding='utf-8')

    assert 'LatestFramePairSynchronizer' in source
    assert 'max_skew_sec=0.1' in source
    assert 'max_receipt_age_sec=1.0' in source
    assert 'BOLT_REFERENCE_FRAMES = 3' in source
    assert '_try_process_cached_pair' in _method_calls(
        source, 'rgb_callback')
    assert '_try_process_cached_pair' in _method_calls(
        source, 'depth_callback')
    assert '_update_bolt_reference' in _method_calls(
        source, '_process_accepted_pair')
    assert 'frame.shape[1] // 2' in source
    assert 'frame.shape[0] // 2' in source
    assert 'detect_static_bolts_hough(frame)' not in source
    try_source = source[
        source.index('def _try_process_cached_pair'):
        source.index('def _process_accepted_pair')
    ]
    pair_rejection = try_source[
        try_source.index('if pair is None:'):
        try_source.index('self._process_accepted_pair')
    ]
    assert '_submit_invalid_frame' not in pair_rejection
    assert '_reset_alignment_tracking' not in pair_rejection


def test_both_callbacks_cache_converted_frames_before_common_pair_retry():
    source = (
        Path(__file__).parents[1] / 'error_fix' / 'error_fix.py'
    ).read_text(encoding='utf-8')

    rgb_calls = _method_calls(source, 'rgb_callback')
    depth_calls = _method_calls(source, 'depth_callback')
    common_calls = _method_calls(source, '_try_process_cached_pair')

    assert 'imgmsg_to_cv2' in rgb_calls
    assert 'cache_rgb' in rgb_calls
    assert '_try_process_cached_pair' in rgb_calls
    assert 'imgmsg_to_cv2' in depth_calls
    assert 'cache_depth' in depth_calls
    assert '_try_process_cached_pair' in depth_calls
    assert 'try_accept' in common_calls
    assert '_process_accepted_pair' in common_calls


def test_accepted_frame_path_consumes_third_frame_decision_without_timer():
    source = (
        Path(__file__).parents[1] / 'error_fix' / 'error_fix.py'
    ).read_text(encoding='utf-8')
    process_source = source[
        source.index('def _process_accepted_pair'):
        source.index('@staticmethod', source.index('def _process_accepted_pair'))
    ]
    timer_source = source[
        source.index('def control_loop'):
        source.index('def _consume_alignment_decision')
    ]
    consumer_source = source[
        source.index('def _consume_alignment_decision'):
        source.index(
            '# --------------------------------------------------------------------------',
            source.index('def _consume_alignment_decision'),
        )
    ]

    # FineAlignmentGate makes its decision on submit of the third accepted
    # observation. Consume it in this same callback so an always-ready image
    # subscription cannot starve progress in a SingleThreadedExecutor.
    assert process_source.index('self.alignment_gate.submit(') < (
        process_source.index('self._consume_alignment_decision()')
    )
    assert 'self._consume_alignment_decision()' in timer_source
    assert consumer_source.count('self.alignment_gate.consume()') == 1


def test_settle_ack_must_match_and_reopens_a_fresh_frame_boundary():
    source = (
        Path(__file__).parents[1] / 'error_fix' / 'error_fix.py'
    ).read_text(encoding='utf-8')

    callback_source = source[
        source.index('def task_command_callback'):
        source.index('def _reset_alignment_tracking')
    ]
    assert 'FINE_ALIGNMENT_STEP_SETTLED:' in callback_source
    assert 'acknowledge_settled' in callback_source
    assert 'frame_pair_sync.reset' in callback_source
    assert 'rgb_boundary_ns=self._last_observed_rgb_stamp_ns' in (
        callback_source)
    assert 'depth_boundary_ns=self._last_observed_depth_stamp_ns' in (
        callback_source)
    assert 'self.current_depth_frame' not in callback_source


def test_alignment_delta_uses_decision_source_stamp_and_explicit_frame():
    source = (
        Path(__file__).parents[1] / 'error_fix' / 'error_fix.py'
    ).read_text(encoding='utf-8')
    control_source = source[
        source.index('def control_loop'):
        source.index(
            '# --------------------------------------------------------------------------',
            source.index('def control_loop'),
        )
    ]

    assert 'decision.stamp_ns // 1_000_000_000' in control_source
    assert 'decision.stamp_ns % 1_000_000_000' in control_source
    assert 'fine_alignment_delta' in control_source
    assert 'self.get_clock().now().to_msg()' not in control_source
    assert 'dtheta > self.TOLERANCE_DEG' in control_source
    assert 'dtheta < -self.TOLERANCE_DEG' in control_source


def test_alignment_delta_echoes_the_active_task_generation():
    source = (
        Path(__file__).parents[1] / 'error_fix' / 'error_fix.py'
    ).read_text(encoding='utf-8')
    callback_source = source[
        source.index('def task_command_callback'):
        source.index('def _reset_alignment_tracking')
    ]
    control_source = source[
        source.index('def control_loop'):
        source.index(
            '# --------------------------------------------------------------------------',
            source.index('def control_loop'),
        )
    ]

    assert 'START_ERRORFIX_CORRECTION:' in callback_source
    assert 'self.alignment_generation = generation' in callback_source
    assert 'if self.alignment_generation is not None:' in control_source
    assert 'f":{self.alignment_generation}"' in control_source


def test_alignment_success_is_scoped_to_the_active_generation():
    source = (
        Path(__file__).parents[1] / 'error_fix' / 'error_fix.py'
    ).read_text(encoding='utf-8')
    control_source = source[
        source.index('def control_loop'):
        source.index(
            '# --------------------------------------------------------------------------',
            source.index('def control_loop'),
        )
    ]

    assert '"ALIGNMENT_SUCCESS:"' in control_source
    assert 'f"{self.alignment_generation}"' in control_source
    assert 'elif self.alignment_legacy_session:' in control_source
    assert 'msg.data = "ALIGNMENT_SUCCESS"' in control_source
    assert 'self.alignment_generation = None' in control_source
    assert 'self.alignment_legacy_session = False' in control_source


def test_bare_start_is_an_explicit_legacy_only_session():
    source = (
        Path(__file__).parents[1] / 'error_fix' / 'error_fix.py'
    ).read_text(encoding='utf-8')
    callback_source = source[
        source.index('def task_command_callback'):
        source.index('def _reset_alignment_tracking')
    ]
    reset_source = source[
        source.index('def _reset_bolt_detection'):
        source.index('def depth_callback')
    ]

    assert 'self.alignment_legacy_session = generation is None' in (
        callback_source)
    assert 'self.alignment_generation = None' in reset_source
    assert 'self.alignment_legacy_session = False' in reset_source

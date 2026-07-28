"""Regression test for the ordered error-fix reset command path."""

from pathlib import Path


def test_error_fix_has_only_the_ordered_string_reset_path():
    source = (
        Path(__file__).parents[1] / 'error_fix' / 'error_fix.py'
    ).read_text(encoding='utf-8')

    assert "'/errorfix_command'" in source
    assert '"RESET_BOLT_DETECTION"' in source
    assert '/bolt_cam/perception/reset_cache' not in source
    assert 'std_msgs.msg import Empty' not in source

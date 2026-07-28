"""Tests for BehaviorNode's optional AMR move deadline."""

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from behavior_node.amr_timeout import (  # noqa: E402
    DEFAULT_AMR_MOVE_TIMEOUT_SEC,
    amr_move_deadline,
    amr_move_timed_out,
)


def test_amr_move_timeout_is_disabled_by_default():
    deadline = amr_move_deadline(
        now=10.0,
        timeout_sec=DEFAULT_AMR_MOVE_TIMEOUT_SEC,
    )

    assert deadline is None
    assert not amr_move_timed_out(deadline, now=1_000_000.0)


def test_positive_amr_move_timeout_still_sets_a_deadline():
    deadline = amr_move_deadline(now=10.0, timeout_sec=120.0)

    assert deadline == 130.0
    assert not amr_move_timed_out(deadline, now=130.0)
    assert amr_move_timed_out(deadline, now=130.1)

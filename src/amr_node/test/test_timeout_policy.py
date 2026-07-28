"""Tests for the AMR bridge watchdog defaults."""

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from amr_node.timeout_policy import (  # noqa: E402
    DEFAULT_BRIDGE_TIMEOUT_SEC,
    bridge_timeout_expired,
)


def test_bridge_watchdog_is_disabled_by_default():
    assert DEFAULT_BRIDGE_TIMEOUT_SEC == 0.0
    assert not bridge_timeout_expired(
        moving=True,
        timeout_sec=DEFAULT_BRIDGE_TIMEOUT_SEC,
        last_activity=1.0,
        now=1_000_000.0,
    )


def test_positive_bridge_watchdog_still_expires():
    assert bridge_timeout_expired(
        moving=True,
        timeout_sec=10.0,
        last_activity=1.0,
        now=11.1,
    )

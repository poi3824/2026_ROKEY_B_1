"""Pure deadline helpers for BehaviorNode AMR moves."""

from typing import Optional


DEFAULT_AMR_MOVE_TIMEOUT_SEC = 0.0


def amr_move_deadline(
    now: float,
    timeout_sec: float,
) -> Optional[float]:
    """Return a deadline, or None when the timeout is disabled."""
    if timeout_sec < 0.0:
        raise ValueError('timeout_sec must be zero or greater')
    if timeout_sec == 0.0:
        return None
    return now + timeout_sec


def amr_move_timed_out(
    deadline: Optional[float],
    now: float,
) -> bool:
    """Return whether an enabled AMR move deadline has passed."""
    return deadline is not None and now > deadline

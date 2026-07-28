"""Pure timeout policy for the FMS-to-Isaac AMR bridge."""


DEFAULT_BRIDGE_TIMEOUT_SEC = 0.0


def bridge_timeout_expired(
    *,
    moving: bool,
    timeout_sec: float,
    last_activity: float,
    now: float,
) -> bool:
    """Return whether an enabled bridge watchdog has expired."""
    if timeout_sec < 0.0:
        raise ValueError('timeout_sec must be zero or greater')
    return (
        moving
        and timeout_sec > 0.0
        and now - last_activity > timeout_sec
    )

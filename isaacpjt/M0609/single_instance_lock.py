"""Nonblocking, ROS-domain-scoped process lock for the Isaac executor."""

from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path


class InstanceLockError(RuntimeError):
    """Raised when another executor owns the requested ROS domain."""


def parse_ros_domain_id(value: str | None) -> int:
    """Parse the ROS domain using the same 0..232 contract as the runtime."""
    raw_value = "0" if value is None else value
    try:
        domain_id = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("ROS_DOMAIN_ID는 정수여야 합니다") from exc
    if not 0 <= domain_id <= 232:
        raise ValueError("ROS_DOMAIN_ID는 0~232 범위여야 합니다")
    return domain_id


class DomainInstanceLock:
    """Own one open ``flock`` file descriptor until explicitly released."""

    def __init__(self, fd: int, path: Path, domain_id: int):
        self._fd = fd
        self.path = path
        self.domain_id = domain_id

    @property
    def fileno(self) -> int:
        """Return the live lock descriptor, or -1 after release."""
        return self._fd

    def release(self) -> None:
        """Release the lock idempotently without unlinking its stable inode."""
        if self._fd < 0:
            return
        fd = self._fd
        self._fd = -1
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def acquire_domain_instance_lock(
    domain_id: int,
    *,
    lock_directory: str | Path = "/tmp",
) -> DomainInstanceLock:
    """Acquire a nonblocking lock for one ROS domain.

    The lock file deliberately remains on disk after release. Unlinking a
    flock file can create two lockable inodes for the same path and allow
    duplicate executors during a startup race.
    """
    if not 0 <= domain_id <= 232:
        raise ValueError("ROS_DOMAIN_ID는 0~232 범위여야 합니다")

    directory = Path(lock_directory)
    path = directory / (
        f"ev_combine_execute_isaac_domain{domain_id}.lock"
    )
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            owner = os.read(fd, 512).decode(
                "utf-8", errors="replace"
            ).strip()
        finally:
            os.close(fd)
        owner_detail = owner or "owner metadata unavailable"
        raise InstanceLockError(
            "같은 ROS_DOMAIN_ID에서 execute_isaac.py가 이미 실행 중입니다: "
            f"domain={domain_id}, lock={path}, {owner_detail}"
        ) from exc
    except Exception:
        os.close(fd)
        raise

    metadata = (
        f"pid={os.getpid()} domain={domain_id} "
        f"started_unix={time.time():.6f}\n"
    ).encode("utf-8")
    try:
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, metadata)
        os.fsync(fd)
    except Exception:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        raise

    return DomainInstanceLock(fd, path, domain_id)


def acquire_ros_domain_instance_lock(
    *,
    lock_directory: str | Path = "/tmp",
) -> DomainInstanceLock:
    """Acquire the executor lock for the current ``ROS_DOMAIN_ID``."""
    domain_id = parse_ros_domain_id(os.environ.get("ROS_DOMAIN_ID"))
    return acquire_domain_instance_lock(
        domain_id,
        lock_directory=lock_directory,
    )

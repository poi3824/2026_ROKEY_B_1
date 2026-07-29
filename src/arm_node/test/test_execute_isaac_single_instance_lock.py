"""Tests for the ROS-domain-scoped Isaac executor lock."""

import importlib.util
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_MODULE_PATH = (
    REPO_ROOT / "isaacpjt/M0609/single_instance_lock.py"
)
EXECUTOR_PATH = REPO_ROOT / "isaacpjt/M0609/execute_isaac.py"


@pytest.fixture(scope="module")
def lock_module():
    spec = importlib.util.spec_from_file_location(
        "single_instance_lock_under_test",
        LOCK_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [(None, 0), ("0", 0), ("129", 129), ("232", 232)],
)
def test_ros_domain_parser_accepts_supported_values(
    lock_module,
    raw_value,
    expected,
):
    assert lock_module.parse_ros_domain_id(raw_value) == expected


@pytest.mark.parametrize("raw_value", ["abc", "-1", "233"])
def test_ros_domain_parser_rejects_invalid_values(
    lock_module,
    raw_value,
):
    with pytest.raises(ValueError):
        lock_module.parse_ros_domain_id(raw_value)


def test_same_domain_fails_fast_until_owner_releases(
    lock_module,
    tmp_path,
):
    owner = lock_module.acquire_domain_instance_lock(
        129,
        lock_directory=tmp_path,
    )
    lock_path = owner.path

    try:
        assert owner.fileno >= 0
        os.fstat(owner.fileno)
        metadata = lock_path.read_text(encoding="utf-8")
        assert f"pid={os.getpid()}" in metadata
        assert "domain=129" in metadata

        with pytest.raises(
            lock_module.InstanceLockError,
            match=r"이미 실행 중.*domain=129",
        ):
            lock_module.acquire_domain_instance_lock(
                129,
                lock_directory=tmp_path,
            )
    finally:
        owner.release()

    # Keep the stable inode on disk, but prove the OS lock was released.
    assert lock_path.is_file()
    replacement = lock_module.acquire_domain_instance_lock(
        129,
        lock_directory=tmp_path,
    )
    replacement.release()
    replacement.release()
    assert replacement.fileno == -1


def test_different_domains_can_run_independently(
    lock_module,
    tmp_path,
):
    first = lock_module.acquire_domain_instance_lock(
        128,
        lock_directory=tmp_path,
    )
    second = lock_module.acquire_domain_instance_lock(
        129,
        lock_directory=tmp_path,
    )
    try:
        assert first.path != second.path
        assert first.fileno >= 0
        assert second.fileno >= 0
    finally:
        second.release()
        first.release()


def test_executor_acquires_lock_before_simulation_app():
    source = EXECUTOR_PATH.read_text(encoding="utf-8")

    lock_index = source.index(
        "_INSTANCE_LOCK = acquire_ros_domain_instance_lock()"
    )
    simulation_import_index = source.index(
        "from isaacsim import SimulationApp"
    )
    simulation_start_index = source.index(
        'simulation_app = SimulationApp({"headless": _HEADLESS})'
    )

    assert lock_index < simulation_import_index < simulation_start_index
    assert "atexit.register(_INSTANCE_LOCK.release)" in source[
        lock_index:simulation_import_index
    ]

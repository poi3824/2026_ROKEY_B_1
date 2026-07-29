"""Functional tests for the default station workflow."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


class _Logger:
    def info(self, *_args, **_kwargs):
        pass

    def warn(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


class _Node:
    def get_logger(self):
        return self._logger


class _ActionClient:
    pass


class _ExecuteArmTask:
    class Goal:
        def __init__(self):
            self.task_type = ""


class _AmrGoal:
    def __init__(self):
        self.station_id = ""
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0


class _AmrStatus:
    STATE_MOVING = 1
    STATE_ARRIVED = 2
    STATE_ERROR = 3

    def __init__(self):
        self.station_id = ""
        self.state = 0
        self.message = ""


class _FleetJob:
    def __init__(self):
        self.job_id = ""
        self.station_id = ""
        self.job_type = ""


class _FleetReport:
    pass


class _Publisher:
    def __init__(self):
        self.published = []

    def publish(self, message):
        self.published.append(message)


class _Timer:
    def __init__(self):
        self.canceled = False

    def cancel(self):
        self.canceled = True


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture()
def behavior_module(monkeypatch):
    """Load BehaviorNode with lightweight ROS message and client doubles."""
    package_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(package_root))

    rclpy = _module(
        "rclpy",
        init=lambda **_kwargs: None,
        shutdown=lambda: None,
        spin=lambda _node: None,
    )
    modules = {
        "rclpy": rclpy,
        "rclpy.node": _module("rclpy.node", Node=_Node),
        "rclpy.action": _module(
            "rclpy.action",
            ActionClient=_ActionClient,
        ),
        "fms_interfaces": _module("fms_interfaces"),
        "fms_interfaces.action": _module(
            "fms_interfaces.action",
            ExecuteArmTask=_ExecuteArmTask,
        ),
        "fms_interfaces.msg": _module(
            "fms_interfaces.msg",
            AmrGoal=_AmrGoal,
            AmrStatus=_AmrStatus,
            FleetJob=_FleetJob,
            FleetReport=_FleetReport,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = (
        package_root / "behavior_node" / "behavior_node.py"
    )
    spec = importlib.util.spec_from_file_location(
        "behavior_node.behavior_node_workflow_under_test",
        module_path,
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def _bare_node(module, station="station_5"):
    node = object.__new__(module.BehaviorNode)
    node._logger = _Logger()
    node.state = module.ProcessState.IDLE
    node.is_waiting_action = False
    node.next_state_on_success = module.ProcessState.IDLE
    node._amr_goal_sent = False
    node._waiting_amr_station = None
    node._amr_deadline = None
    node._amr_move_timeout_sec = 0.0
    node._work_station = station
    node._amr_goal_pub = _Publisher()
    node._active_job = None
    node._auto_start = True
    node.has_started = False
    node.start_timer = _Timer()
    return node


def _arrived(module, station_id):
    status = module.AmrStatus()
    status.station_id = station_id
    status.state = module.AmrStatus.STATE_ARRIVED
    status.message = "arrived"
    return status


def test_auto_and_fleet_start_at_busbar_without_battery_scan(
    behavior_module,
):
    auto_node = _bare_node(behavior_module)
    auto_node.auto_start_trigger()

    assert auto_node.start_timer.canceled
    assert (
        auto_node.state
        == behavior_module.ProcessState.MOVE_AMR_BUSBAR
    )

    fleet_node = _bare_node(behavior_module)
    job = behavior_module.FleetJob()
    job.job_id = "job_station_5"
    job.station_id = "station_5"
    job.job_type = "ASSEMBLE"
    fleet_node._on_fleet_job(job)

    assert (
        fleet_node.state
        == behavior_module.ProcessState.MOVE_AMR_BUSBAR
    )
    assert fleet_node._active_job is job


def test_default_flow_emits_only_busbar_then_battery_amr_goals(
    behavior_module,
):
    node = _bare_node(behavior_module)
    node.state = behavior_module.ProcessState.MOVE_AMR_BUSBAR
    arm_tasks = []

    def complete_arm_task(task_type, next_state):
        arm_tasks.append(task_type)
        node.state = next_state

    node.send_arm_goal = complete_arm_task

    node.fsm_loop()
    assert [
        goal.station_id for goal in node._amr_goal_pub.published
    ] == ["busbar5"]

    node._on_amr_status(_arrived(
        behavior_module,
        "busbar5",
    ))
    assert node.state == behavior_module.ProcessState.INITIALIZE_ARM

    node.fsm_loop()
    assert arm_tasks == ["RETURN_HOME"]
    assert node.state == behavior_module.ProcessState.SCAN_BUSBAR

    node.fsm_loop()
    node.fsm_loop()
    assert arm_tasks == [
        "RETURN_HOME",
        "LATCH_BUSBAR",
        "PICK_BUSBAR",
    ]
    assert "SCAN_BATTERY" not in arm_tasks
    assert "SCAN_BUSBAR" not in arm_tasks
    assert (
        node.state
        == behavior_module.ProcessState.MOVE_AMR_BATTERY_ASSEMBLY
    )

    node.fsm_loop()
    assert [
        goal.station_id for goal in node._amr_goal_pub.published
    ] == ["busbar5", "battery5"]

    node._on_amr_status(_arrived(
        behavior_module,
        "battery5",
    ))
    assert node.state == behavior_module.ProcessState.MOVE_BATTERY_CENTER

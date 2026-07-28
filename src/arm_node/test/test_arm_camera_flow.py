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


class _Publisher:
    def __init__(self, topic, events):
        self.topic = topic
        self.published = []
        self._events = events

    def publish(self, message):
        self.published.append(message)
        self._events.append((self.topic, message))


class _Client:
    def __init__(self, topic):
        self.topic = topic


class _Node:
    def __init__(self, name):
        self.name = name
        self._logger = _Logger()
        self._events = []
        self._clients = {}
        self._publishers = {}

    def get_logger(self):
        return self._logger

    def get_clock(self):
        return types.SimpleNamespace(
            now=lambda: types.SimpleNamespace(nanoseconds=1_000)
        )

    def create_client(self, _service_type, topic, **_kwargs):
        client = _Client(topic)
        self._clients[topic] = client
        return client

    def create_publisher(self, _message_type, topic, _depth):
        publisher = _Publisher(topic, self._events)
        self._publishers[topic] = publisher
        return publisher

    def create_subscription(
        self, _message_type, topic, callback, _depth, **_kwargs
    ):
        return types.SimpleNamespace(topic=topic, callback=callback)

    def destroy_node(self):
        pass


class _ActionServer:
    def __init__(self, *_args, **kwargs):
        self.kwargs = kwargs


class _CancelResponse:
    ACCEPT = "accept"


class _Stamp:
    def __init__(self, nanoseconds=0):
        self.sec, self.nanosec = divmod(nanoseconds, 1_000_000_000)


class _PoseStamped:
    def __init__(self):
        self.header = types.SimpleNamespace(
            stamp=_Stamp(),
            frame_id="world",
        )
        self.pose = types.SimpleNamespace(
            position=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
            orientation=types.SimpleNamespace(
                x=0.0, y=0.0, z=0.0, w=1.0
            ),
        )


class _String:
    def __init__(self):
        self.data = ""


class _Float32:
    def __init__(self):
        self.data = 0.0


class _Empty:
    pass


class _Feedback:
    def __init__(self):
        self.sub_phase = ""
        self.progress_pct = 0.0


class _Result:
    def __init__(self):
        self.success = False
        self.error_code = ""
        self.message = ""


class _ExecuteArmTask:
    Feedback = _Feedback
    Result = _Result


class _GetGraspPose:
    class Request:
        def __init__(self):
            self.label = ""


class _GetBoltPair:
    class Request:
        pass


class _NutPose:
    def __init__(self):
        self.pose = _PoseStamped()


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture()
def arm_module(monkeypatch):
    rclpy = _module(
        "rclpy",
        ok=lambda: True,
        init=lambda **_kwargs: None,
        shutdown=lambda: None,
    )
    modules = {
        "rclpy": rclpy,
        "rclpy.node": _module("rclpy.node", Node=_Node),
        "rclpy.action": _module(
            "rclpy.action",
            ActionServer=_ActionServer,
            CancelResponse=_CancelResponse,
        ),
        "rclpy.callback_groups": _module(
            "rclpy.callback_groups",
            ReentrantCallbackGroup=type(
                "ReentrantCallbackGroup", (), {}
            ),
        ),
        "rclpy.executors": _module(
            "rclpy.executors",
            MultiThreadedExecutor=type(
                "MultiThreadedExecutor", (), {}
            ),
        ),
        "geometry_msgs": _module("geometry_msgs"),
        "geometry_msgs.msg": _module(
            "geometry_msgs.msg", PoseStamped=_PoseStamped
        ),
        "std_msgs": _module("std_msgs"),
        "std_msgs.msg": _module(
            "std_msgs.msg",
            Empty=_Empty,
            Float32=_Float32,
            String=_String,
        ),
        "fms_interfaces": _module("fms_interfaces"),
        "fms_interfaces.action": _module(
            "fms_interfaces.action",
            ExecuteArmTask=_ExecuteArmTask,
        ),
        "fms_interfaces.srv": _module(
            "fms_interfaces.srv",
            GetBoltPair=_GetBoltPair,
            GetGraspPose=_GetGraspPose,
        ),
        "fms_interfaces.msg": _module(
            "fms_interfaces.msg", NutPose=_NutPose
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = (
        Path(__file__).parents[1] / "arm_node" / "arm_node.py"
    )
    spec = importlib.util.spec_from_file_location(
        "arm_node_camera_flow_under_test", module_path
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


class _Goal:
    def __init__(self, task_type):
        self.request = types.SimpleNamespace(task_type=task_type)
        self.is_cancel_requested = False
        self.transition = None
        self.feedback = []

    def publish_feedback(self, feedback):
        self.feedback.append(feedback)

    def succeed(self):
        self.transition = "succeeded"

    def abort(self):
        self.transition = "aborted"

    def canceled(self):
        self.transition = "canceled"


def _pose(module, x, y, z, stamp_ns):
    pose = module.PoseStamped()
    pose.header.stamp = _Stamp(stamp_ns)
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    return pose


def test_camera_clients_resets_and_cancel_contract(arm_module):
    node = arm_module.ArmNode()

    assert set(node._clients) == {
        "/wrist/perception/get_grasp_pose",
        "/busbar_cam/perception/get_grasp_pose",
        "/bolt_cam/perception/get_bolt_pair",
    }
    assert {
        "/wrist/perception/reset_cache",
        "/busbar_cam/perception/reset_cache",
        "/bolt_cam/perception/reset_cache",
    }.issubset(node._publishers)
    assert node._action_server.kwargs["cancel_callback"] is not None
    assert (
        node.cancel_callback(_Goal("SCAN_BUSBAR"))
        == arm_module.CancelResponse.ACCEPT
    )


def test_overlapping_action_is_rejected_as_busy(arm_module):
    node = arm_module.ArmNode()
    goal = _Goal("SCAN_BUSBAR")
    assert node._execution_lock.acquire(blocking=False)

    try:
        result = node.execute_callback(goal)
    finally:
        node._execution_lock.release()

    assert not result.success
    assert result.error_code == "BUSY"
    assert goal.transition == "aborted"
    assert not node.pub_task_command.published
    assert not node.pub_target_pose.published


def test_reset_waits_for_the_matching_perception_ack(
    arm_module,
    monkeypatch,
):
    node = arm_module.ArmNode()
    goal = _Goal("SCAN_BUSBAR")
    sleep_calls = []

    def acknowledge(_duration):
        sleep_calls.append(True)
        node._record_perception_reset_ack("busbar_cam")

    monkeypatch.setattr(arm_module.time, "sleep", acknowledge)

    assert node._reset_perception_cache(
        node.pub_busbar_reset,
        goal,
    )
    assert len(node.pub_busbar_reset.published) == 1
    assert sleep_calls == [True]


def test_phase_wait_forwards_feedback_until_camera_ready(
    arm_module,
    monkeypatch,
):
    node = arm_module.ArmNode()
    node.isaac_phase = "SCAN_BUSBAR_NAV"
    node.isaac_progress = 42.0
    goal = _Goal("SCAN_BUSBAR")
    sleeps = []

    def advance_to_ready(_duration):
        sleeps.append(True)
        node.isaac_phase = "BUSBAR_CAMERA_READY"

    monkeypatch.setattr(arm_module.time, "sleep", advance_to_ready)

    assert node.wait_for_isaac_phase(
        "BUSBAR_CAMERA_READY",
        goal,
        arm_module.ExecuteArmTask.Feedback(),
    )
    assert sleeps == [True]
    assert len(goal.feedback) == 2
    assert goal.feedback[-1].sub_phase == "BUSBAR_CAMERA_READY"
    assert goal.feedback[-1].progress_pct == pytest.approx(42.0)


def test_phase_wait_stops_on_isaac_failure(arm_module, monkeypatch):
    node = arm_module.ArmNode()
    node.isaac_status = "FAILURE: camera move failed"
    goal = _Goal("SCAN_BUSBAR")
    monkeypatch.setattr(
        arm_module.time,
        "sleep",
        lambda _duration: pytest.fail("terminal failure must not sleep"),
    )

    assert not node.wait_for_isaac_phase(
        "BUSBAR_CAMERA_READY",
        goal,
        arm_module.ExecuteArmTask.Feedback(),
    )


def test_short_ordering_wait_stops_on_isaac_failure(
    arm_module,
    monkeypatch,
):
    node = arm_module.ArmNode()
    node.isaac_status = "FAILURE: playback restart"
    monkeypatch.setattr(
        arm_module.time,
        "sleep",
        lambda _duration: pytest.fail("terminal failure must not sleep"),
    )

    assert not node._wait_while_active(
        0.1,
        _Goal("PICK_BUSBAR"),
    )


def test_wrist_confirmation_stops_on_isaac_failure(arm_module):
    node = arm_module.ArmNode()
    node.isaac_status = "FAILURE: arm task conflict"
    node._call_grasp_pose = (
        lambda *_args: pytest.fail(
            "terminal failure must stop before a service retry"
        )
    )

    found, pose, message = node.wait_for_wrist_busbar_confirmation(
        _pose(arm_module, 0.0, 0.0, 0.0, 1),
        _Goal("SCAN_BUSBAR"),
    )

    assert not found
    assert pose is None
    assert "Isaac failure" in message


def test_bolt_pair_wait_stops_on_isaac_failure(arm_module):
    node = arm_module.ArmNode()
    node.isaac_status = "FAILURE: playback restart"
    node.client_get_bolt_pair = types.SimpleNamespace(
        wait_for_service=lambda **_kwargs: pytest.fail(
            "terminal failure must stop before service discovery"
        )
    )

    found, pose, message = node.request_bolt_pair_midpoint_async(
        _Goal("SCAN_BATTERY"),
    )

    assert not found
    assert pose is None
    assert "Isaac failure" in message


def test_wrist_confirmation_requires_fresh_increasing_samples(arm_module):
    node = arm_module.ArmNode()
    fixed = _pose(arm_module, 1.0, 2.0, 0.3, 90)
    observations = [
        (True, _pose(arm_module, 1.02, 2.00, 0.3, 101), "first"),
        (True, _pose(arm_module, 1.03, 2.00, 0.3, 101), "duplicate"),
        (True, _pose(arm_module, 1.04, 2.00, 0.3, 102), "second"),
    ]
    calls = []

    def call_once(client, label, goal):
        calls.append((client, label, goal))
        return observations.pop(0)

    node._call_grasp_pose = call_once
    node._wait_while_active = lambda _duration, _goal: True
    goal = _Goal("SCAN_BUSBAR")

    found, confirmed_pose, message = (
        node.wait_for_wrist_busbar_confirmation(
            fixed, goal
        )
    )

    assert found
    assert confirmed_pose.pose.position.x == pytest.approx(1.04)
    assert "연속 2표본" in message
    assert len(calls) == 3
    assert all(call[0] is node.client_get_wrist_pose for call in calls)


def test_wrist_confirmation_restarts_after_large_sample_shift(arm_module):
    node = arm_module.ArmNode()
    fixed = _pose(arm_module, 0.0, 0.0, 0.0, 90)
    observations = [
        (True, _pose(arm_module, 0.0, 0.0, 0.00, 101), "first"),
        (True, _pose(arm_module, 0.0, 0.0, 0.08, 102), "3d shift"),
        (True, _pose(arm_module, 0.0, 0.0, 0.09, 103), "stable"),
    ]
    node._call_grasp_pose = (
        lambda _client, _label, _goal: observations.pop(0)
    )
    node._wait_while_active = lambda _duration, _goal: True

    found, confirmed_pose, _message = (
        node.wait_for_wrist_busbar_confirmation(
            fixed, _Goal("SCAN_BUSBAR")
        )
    )

    assert found
    assert confirmed_pose.pose.position.z == pytest.approx(0.09)
    assert not observations


def test_fixed_grasp_request_retries_until_barrier_reports_found(arm_module):
    node = arm_module.ArmNode()
    fresh = _pose(arm_module, 0.3, 0.2, 0.1, 102)
    responses = [
        (False, None, "not ready"),
        (True, fresh, "fresh"),
    ]
    calls = []

    def call_once(client, label, goal):
        calls.append((client, label, goal))
        return responses.pop(0)

    node._call_grasp_pose = call_once
    node._wait_while_active = lambda _duration, _goal: True
    goal = _Goal("SCAN_BUSBAR")

    found, pose, message = node.request_grasp_pose_until_found(
        node.client_get_busbar_pose,
        "busbar",
        goal,
    )

    assert found
    assert pose is fresh
    assert message == "fresh"
    assert len(calls) == 2


def test_scan_latches_fixed_pose_and_pick_reuses_exact_snapshot(
    arm_module,
):
    node = arm_module.ArmNode()
    fixed = _pose(arm_module, 1.2, -0.4, 0.2, 201)
    wrist = _pose(arm_module, 1.21, -0.39, 0.21, 301)
    reset_topics = []
    service_requests = []
    delays = []
    scan_order = []

    def wait_for_phase(expected, _goal, _feedback):
        scan_order.append(f"phase:{expected}")
        return True

    def wait_for_completion(request_goal, _feedback):
        if request_goal.request.task_type == "SCAN_BUSBAR":
            scan_order.append("isaac:success")
            assert [
                message.data
                for message in node.pub_task_command.published
            ] == ["SCAN_BUSBAR", "CONTINUE_BUSBAR_WRIST_SCAN"]
        return True

    node.wait_for_isaac_phase = wait_for_phase
    node.wait_for_isaac_completion = wait_for_completion

    def reset(publisher, _goal):
        reset_topics.append(publisher.topic)
        scan_order.append(f"reset:{publisher.topic}")
        return True

    def fixed_request(client, label, goal):
        scan_order.append("fixed:latch")
        assert [
            message.data
            for message in node.pub_task_command.published
        ] == ["SCAN_BUSBAR"]
        service_requests.append(
            (
                client.topic,
                label,
                goal.request.task_type,
            )
        )
        return True, fixed, "fixed"

    node._reset_perception_cache = reset
    node.request_grasp_pose_until_found = fixed_request

    def confirm_wrist(selected, _goal):
        scan_order.append("wrist:confirm")
        assert selected is fixed
        return (
            True,
            wrist,
            f"selected={selected.pose.position.x}",
        )

    node.wait_for_wrist_busbar_confirmation = confirm_wrist
    node._wait_while_active = (
        lambda duration, _goal: delays.append(duration) or True
    )

    scan_goal = _Goal("SCAN_BUSBAR")
    scan_result = node.execute_callback(scan_goal)

    assert scan_result.success
    assert scan_goal.transition == "succeeded"
    assert node.scanned_busbar_pose is fixed
    assert reset_topics == [
        "/busbar_cam/perception/reset_cache",
        "/wrist/perception/reset_cache",
    ]
    assert service_requests == [
        (
            "/busbar_cam/perception/get_grasp_pose",
            "busbar",
            "SCAN_BUSBAR",
        )
    ]
    assert [
        message.data
        for message in node.pub_task_command.published
    ] == ["SCAN_BUSBAR", "CONTINUE_BUSBAR_WRIST_SCAN"]
    assert not node.pub_target_pose.published
    assert scan_order == [
        "phase:BUSBAR_CAMERA_READY",
        "reset:/busbar_cam/perception/reset_cache",
        "fixed:latch",
        "isaac:success",
        "reset:/wrist/perception/reset_cache",
        "wrist:confirm",
    ]

    pick_goal = _Goal("PICK_BUSBAR")
    pick_result = node.execute_callback(pick_goal)

    assert pick_result.success
    assert pick_goal.transition == "succeeded"
    assert node.pub_target_pose.published == [fixed]
    assert delays == [0.1]
    assert [
        message.data
        for message in node.pub_task_command.published
    ] == [
        "SCAN_BUSBAR",
        "CONTINUE_BUSBAR_WRIST_SCAN",
        "PICK_BUSBAR",
    ]
    target_event = next(
        index
        for index, (topic, _message) in enumerate(node._events)
        if topic == "/target_pose"
    )
    pick_event = next(
        index
        for index, (topic, message) in enumerate(node._events)
        if topic == "/task_command"
        and message.data == "PICK_BUSBAR"
    )
    assert target_event < pick_event


def test_pick_rejects_missing_fixed_snapshot_without_fallback(arm_module):
    node = arm_module.ArmNode()
    node.scanned_busbar_pose = None
    node.scanned_battery_midpoint = _pose(
        arm_module, 9.0, 9.0, 9.0, 1
    )
    node.request_grasp_pose_until_found = lambda *_args, **_kwargs: (
        pytest.fail("PICK_BUSBAR must not refresh perception")
    )

    goal = _Goal("PICK_BUSBAR")
    result = node.execute_callback(goal)

    assert not result.success
    assert result.error_code == "NO_FIXED_CAMERA_BUSBAR_POSE"
    assert goal.transition == "aborted"
    assert not node.pub_target_pose.published
    assert not node.pub_task_command.published


def test_scan_cancel_is_reported_as_canceled_not_aborted(arm_module):
    node = arm_module.ArmNode()
    previous_snapshot = _pose(arm_module, 0.4, 0.5, 0.6, 99)
    node.scanned_busbar_pose = previous_snapshot
    goal = _Goal("SCAN_BUSBAR")
    goal.is_cancel_requested = True

    result = node.execute_callback(goal)

    assert not result.success
    assert result.error_code == "SCAN_BUSBAR_CANCELED"
    assert goal.transition == "canceled"
    assert node.scanned_busbar_pose is previous_snapshot
    assert not node.pub_busbar_reset.published
    assert not node.pub_wrist_reset.published
    assert [
        message.data
        for message in node.pub_task_command.published
    ] == []


def test_scan_cancel_fence_before_continue_releases_isaac_pause(arm_module):
    node = arm_module.ArmNode()
    goal = _Goal("SCAN_BUSBAR")
    fixed = _pose(arm_module, 1.2, -0.4, 0.2, 201)
    node.wait_for_isaac_phase = (
        lambda _phase, _goal, _feedback: True
    )
    node._reset_perception_cache = (
        lambda _publisher, _goal: True
    )

    def cancel_fixed_request(_client, _label, request_goal):
        request_goal.is_cancel_requested = True
        return True, fixed, "fixed before cancel"

    node.request_grasp_pose_until_found = cancel_fixed_request
    node.wait_for_isaac_completion = lambda *_args: pytest.fail(
        "cancel fence must stop before waiting for wrist scan"
    )

    result = node.execute_callback(goal)

    assert not result.success
    assert result.error_code == "SCAN_BUSBAR_CANCELED"
    assert goal.transition == "canceled"
    assert [
        message.data
        for message in node.pub_task_command.published
    ] == ["SCAN_BUSBAR", "CANCEL_ARM_TASK"]


def test_scan_failure_before_continue_releases_isaac_pause(arm_module):
    node = arm_module.ArmNode()
    goal = _Goal("SCAN_BUSBAR")
    node.wait_for_isaac_phase = (
        lambda _phase, _goal, _feedback: True
    )
    node._reset_perception_cache = (
        lambda _publisher, _goal: True
    )

    def fail_fixed_request(_client, _label, _goal):
        node.isaac_status = "FAILURE: camera bridge stopped"
        return False, None, "Isaac failure"

    node.request_grasp_pose_until_found = fail_fixed_request

    result = node.execute_callback(goal)

    assert not result.success
    assert result.error_code == "BUSBAR_VISION_INTERRUPTED"
    assert goal.transition == "aborted"
    assert [
        message.data
        for message in node.pub_task_command.published
    ] == ["SCAN_BUSBAR", "CANCEL_ARM_TASK"]


def test_scan_battery_resets_bolt_cache_and_latches_midpoint(arm_module):
    node = arm_module.ArmNode()
    midpoint = _pose(arm_module, 0.5, -0.2, 0.1, 501)
    reset_topics = []
    requests = []
    node.wait_for_isaac_completion = (
        lambda _goal, _feedback: True
    )
    node._reset_perception_cache = (
        lambda publisher, _goal: (
            reset_topics.append(publisher.topic) or 500
        )
    )

    def request(goal):
        requests.append(goal.request.task_type)
        return True, midpoint, "bolts"

    node.request_bolt_pair_midpoint_async = request
    goal = _Goal("SCAN_BATTERY")
    result = node.execute_callback(goal)

    assert result.success
    assert goal.transition == "succeeded"
    assert node.scanned_battery_midpoint is midpoint
    assert reset_topics == ["/bolt_cam/perception/reset_cache"]
    assert requests == ["SCAN_BATTERY"]


def test_bolt_pair_retries_until_barrier_reports_found(arm_module):
    node = arm_module.ArmNode()
    fresh_a = _pose(arm_module, 0.2, 0.0, 0.1, 101)
    fresh_b = _pose(arm_module, 2.2, 0.0, 0.1, 101)
    responses = [
        types.SimpleNamespace(found=False, message="not ready"),
        types.SimpleNamespace(
            found=True,
            pose_a=fresh_a,
            pose_b=fresh_b,
            message="fresh",
        ),
    ]

    class ImmediateFuture:
        def __init__(self, response):
            self._response = response

        def done(self):
            return True

        def result(self):
            return self._response

    class BoltClient:
        def __init__(self):
            self.calls = 0

        def wait_for_service(self, timeout_sec):
            assert timeout_sec == 1.0
            return True

        def call_async(self, _request):
            self.calls += 1
            return ImmediateFuture(responses.pop(0))

    client = BoltClient()
    node.client_get_bolt_pair = client
    node._wait_while_active = lambda _duration, _goal: True

    found, midpoint, message = node.request_bolt_pair_midpoint_async(
        _Goal("SCAN_BATTERY"),
    )

    assert found
    assert message == "fresh"
    assert midpoint.pose.position.x == pytest.approx(1.2)
    assert midpoint.pose.position.y == pytest.approx(0.0)
    assert midpoint.pose.position.z == pytest.approx(0.1)
    assert client.calls == 2

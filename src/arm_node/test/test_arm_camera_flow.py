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


def test_reset_waits_for_service_generation_not_uncorrelated_empty_ack(
    arm_module,
):
    node = arm_module.ArmNode()
    goal = _Goal("SCAN_BUSBAR")
    generations = iter((4, 4, 5))
    probes = []

    def probe(camera_name, request_goal):
        probes.append((camera_name, request_goal))
        # A delayed Empty ACK must not complete the reset.
        node._record_perception_reset_ack("busbar_cam")
        return next(generations)

    node._probe_perception_generation = probe
    node._wait_while_active = lambda _duration, _goal: True

    assert node._reset_perception_cache(
        node.pub_busbar_reset,
        goal,
    )
    assert len(node.pub_busbar_reset.published) == 1
    assert [camera for camera, _goal in probes] == [
        "busbar_cam",
        "busbar_cam",
        "busbar_cam",
    ]
    assert node._perception_reset_ack_counts["busbar_cam"] == 3
    assert node._required_perception_generation["busbar_cam"] == 5


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("[generation=0] waiting", 0),
        ("[generation=17] ready", 17),
        ("generation=2", None),
        ("[generation=-1] invalid", None),
        ("[generation=nope] invalid", None),
    ],
)
def test_perception_generation_message_parser(
    arm_module,
    message,
    expected,
):
    assert arm_module.ArmNode._perception_generation(message) == expected


def test_generation_probe_retries_transport_but_rejects_legacy_server(
    arm_module,
):
    node = arm_module.ArmNode()
    goal = _Goal("SCAN_BATTERY")
    retries = []
    calls = [
        (None, "GetBoltPair 호출 실패: transient"),
        (
            types.SimpleNamespace(
                message="[generation=7] target not ready"
            ),
            "unused",
        ),
    ]
    node._call_bolt_pair_once = lambda _goal: calls.pop(0)
    node._wait_while_active = (
        lambda _duration, _goal: retries.append(True) or True
    )

    assert node._probe_perception_generation(
        "bolt_cam",
        goal,
    ) == 7
    assert retries == [True]

    node._call_bolt_pair_once = lambda _goal: (
        types.SimpleNamespace(message="'bolt' target not ready"),
        "unused",
    )
    with pytest.raises(
        arm_module.PerceptionProtocolError,
        match="protocol mismatch",
    ):
        node._probe_perception_generation("bolt_cam", goal)


def test_reset_reports_protocol_mismatch_without_publishing(
    arm_module,
):
    node = arm_module.ArmNode()
    node._probe_perception_generation = (
        lambda _camera, _goal: (_ for _ in ()).throw(
            arm_module.PerceptionProtocolError(
                "bolt_cam perception protocol mismatch"
            )
        )
    )

    assert not node._reset_perception_cache(
        node.pub_bolt_reset,
        _Goal("SCAN_BATTERY"),
    )
    assert node.pub_bolt_reset.published == []
    assert (
        node._last_perception_reset_error
        == "bolt_cam perception protocol mismatch"
    )


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


@pytest.mark.parametrize(
    ("found", "message"),
    [
        (False, "'busbar' target not ready"),
        (False, "[generation=2] stale"),
        (True, "'busbar' target ready"),
        (True, "[generation=2] stale"),
    ],
)
def test_grasp_request_fails_on_missing_or_regressed_generation(
    arm_module,
    found,
    message,
):
    node = arm_module.ArmNode()
    node._required_perception_generation["busbar_cam"] = 3
    response = types.SimpleNamespace(
        found=found,
        pose=_pose(arm_module, 0.1, 0.2, 0.3, 10),
        message=message,
    )
    node._call_grasp_pose_response_once = (
        lambda _client, _label, _goal: (response, response.message)
    )

    with pytest.raises(
        arm_module.PerceptionProtocolError,
        match="protocol mismatch|generation regression",
    ):
        node.request_grasp_pose_until_found(
            node.client_get_busbar_pose,
            "busbar",
            _Goal("SCAN_BUSBAR"),
        )


@pytest.mark.parametrize(
    ("found", "message"),
    [
        (False, "'busbar' target not ready"),
        (False, "[generation=2] stale"),
        (True, "'busbar' target ready"),
        (True, "[generation=2] stale"),
    ],
)
def test_wrist_confirmation_fails_on_every_invalid_generation_response(
    arm_module,
    found,
    message,
):
    node = arm_module.ArmNode()
    node._required_perception_generation["wrist"] = 3
    response = types.SimpleNamespace(
        found=found,
        pose=_pose(arm_module, 0.0, 0.0, 0.0, 10),
        message=message,
    )
    node._call_grasp_pose_response_once = (
        lambda _client, _label, _goal: (response, response.message)
    )

    with pytest.raises(
        arm_module.PerceptionProtocolError,
        match="protocol mismatch|generation regression",
    ):
        node.wait_for_wrist_busbar_confirmation(
            _pose(arm_module, 0.0, 0.0, 0.0, 1),
            _Goal("SCAN_BUSBAR"),
        )


def test_post_reset_grasp_transport_failure_remains_retryable(
    arm_module,
):
    node = arm_module.ArmNode()
    node._required_perception_generation["busbar_cam"] = 3
    fresh = _pose(arm_module, 0.3, 0.2, 0.1, 102)
    response = types.SimpleNamespace(
        found=True,
        pose=fresh,
        message="[generation=3] fresh",
    )
    calls = [
        (None, "GetGraspPose 호출 실패: transient"),
        (response, response.message),
    ]
    node._call_grasp_pose_response_once = (
        lambda _client, _label, _goal: calls.pop(0)
    )
    node._wait_while_active = lambda _duration, _goal: True

    found, pose, message = node.request_grasp_pose_until_found(
        node.client_get_busbar_pose,
        "busbar",
        _Goal("SCAN_BUSBAR"),
    )

    assert found
    assert pose is fresh
    assert message == "[generation=3] fresh"
    assert not calls


def test_nut_scan_request_never_uses_legacy_topic_fallback(
    arm_module,
):
    node = arm_module.ArmNode()
    node.latest_nut_pose = _pose(
        arm_module, 7.0, 8.0, 9.0, 100)
    node._call_grasp_pose = (
        lambda _client, _label, _goal: (
            False,
            None,
            "[generation=1] barrier 2/3",
        )
    )
    node._wait_while_active = lambda _duration, _goal: False

    found, pose, message = node.request_grasp_pose_until_found(
        node.client_get_wrist_pose,
        "nut",
        _Goal("SCAN_NUT1"),
    )

    assert not found
    assert pose is None
    assert "너트 토픽 데이터 사용" not in message


def test_scan_nut_action_fails_without_strict_service_barrier_result(
    arm_module,
):
    node = arm_module.ArmNode()
    node.latest_nut_pose = _pose(
        arm_module, 7.0, 8.0, 9.0, 100)
    node.wait_for_isaac_completion = (
        lambda _goal, _feedback: True
    )
    node._reset_perception_cache = (
        lambda _publisher, _goal: True
    )
    node.request_grasp_pose_until_found = (
        lambda _client, _label, _goal: (
            False,
            None,
            "strict 3-frame service barrier interrupted",
        )
    )

    goal = _Goal("SCAN_NUT1")
    result = node.execute_callback(goal)

    assert not result.success
    assert result.error_code == "NUT_VISION_FAILED"
    assert goal.transition == "aborted"
    assert node.scanned_nut1_pose is None


def test_default_latch_skips_wrist_scan_and_pick_reuses_exact_snapshot(
    arm_module,
):
    node = arm_module.ArmNode()
    fixed = _pose(arm_module, 1.2, -0.4, 0.2, 201)
    reset_topics = []
    service_requests = []
    scan_order = []

    def wait_for_phase(expected, _goal, _feedback):
        scan_order.append(f"phase:{expected}")
        return True

    def wait_for_completion(request_goal, _feedback):
        scan_order.append(
            f"isaac:{request_goal.request.task_type}:success"
        )
        if request_goal.request.task_type == "LATCH_BUSBAR":
            assert [
                message.data
                for message in node.pub_task_command.published
            ] == [
                "SCAN_BUSBAR",
                "COMPLETE_BUSBAR_FIXED_SCAN",
            ]
        return True

    node.wait_for_isaac_phase = wait_for_phase
    node.wait_for_isaac_completion = wait_for_completion

    def reset(publisher, _goal):
        reset_topics.append(publisher.topic)
        scan_order.append(f"reset:{publisher.topic}")
        return True

    def fixed_request(client, label, goal):
        scan_order.append("fixed:latch")
        service_requests.append(
            (client.topic, label, goal.request.task_type)
        )
        return True, fixed, "fixed"

    node._reset_perception_cache = reset
    node.request_grasp_pose_until_found = fixed_request
    node.wait_for_wrist_busbar_confirmation = (
        lambda *_args: pytest.fail(
            "default fixed-camera latch must not request wrist samples"
        )
    )
    node._wait_while_active = lambda _duration, _goal: True

    latch_goal = _Goal("LATCH_BUSBAR")
    latch_result = node.execute_callback(latch_goal)

    assert latch_result.success
    assert latch_goal.transition == "succeeded"
    assert node.scanned_busbar_pose is fixed
    assert reset_topics == ["/busbar_cam/perception/reset_cache"]
    assert not node.pub_wrist_reset.published
    assert service_requests == [
        (
            "/busbar_cam/perception/get_grasp_pose",
            "busbar",
            "LATCH_BUSBAR",
        )
    ]
    assert scan_order == [
        "phase:BUSBAR_CAMERA_READY",
        "reset:/busbar_cam/perception/reset_cache",
        "fixed:latch",
        "isaac:LATCH_BUSBAR:success",
    ]
    assert not node.pub_target_pose.published

    pick_goal = _Goal("PICK_BUSBAR")
    pick_result = node.execute_callback(pick_goal)

    assert pick_result.success
    assert pick_goal.transition == "succeeded"
    assert node.pub_target_pose.published == [fixed]
    assert [
        message.data
        for message in node.pub_task_command.published
    ] == [
        "SCAN_BUSBAR",
        "COMPLETE_BUSBAR_FIXED_SCAN",
        "PICK_BUSBAR",
    ]


def test_fixed_camera_latch_drops_snapshot_if_completion_handshake_fails(
    arm_module,
):
    node = arm_module.ArmNode()
    fixed = _pose(arm_module, 1.2, -0.4, 0.2, 201)
    node.wait_for_isaac_phase = (
        lambda _phase, _goal, _feedback: True
    )
    node._reset_perception_cache = (
        lambda _publisher, _goal: True
    )
    node.request_grasp_pose_until_found = (
        lambda _client, _label, _goal: (True, fixed, "fixed")
    )

    def fail_completion(_goal, _feedback):
        node.isaac_status = "FAILURE:INVALID_BUSBAR_FIXED_SCAN_COMPLETE"
        return False

    node.wait_for_isaac_completion = fail_completion
    node.wait_for_wrist_busbar_confirmation = (
        lambda *_args: pytest.fail(
            "fixed-only handshake failure must not fall through to wrist"
        )
    )

    goal = _Goal("LATCH_BUSBAR")
    result = node.execute_callback(goal)

    assert not result.success
    assert result.error_code == "SCAN_BUSBAR_FAILED"
    assert goal.transition == "aborted"
    assert node.scanned_busbar_pose is None
    assert not node.pub_wrist_reset.published
    assert [
        message.data
        for message in node.pub_task_command.published
    ] == ["SCAN_BUSBAR", "COMPLETE_BUSBAR_FIXED_SCAN"]


def test_explicit_legacy_scan_latches_fixed_pose_and_confirms_wrist(
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


def test_move_battery_center_uses_isaac_live_bolts_without_target_pose(
    arm_module,
):
    node = arm_module.ArmNode()
    # Legacy SCAN_BATTERY remains callable and may leave this snapshot
    # behind, but MOVE_BATTERY_CENTER must not consume or publish it.
    legacy_midpoint = _pose(arm_module, 9.0, -8.0, 7.0, 501)
    node.scanned_battery_midpoint = legacy_midpoint
    node.wait_for_isaac_completion = (
        lambda _goal, _feedback: True
    )

    goal = _Goal("MOVE_BATTERY_CENTER")
    result = node.execute_callback(goal)

    assert result.success
    assert goal.transition == "succeeded"
    assert node.pub_target_pose.published == []
    assert [
        message.data
        for message in node.pub_task_command.published
    ] == ["MOVE_BATTERY_CENTER"]
    assert node.scanned_battery_midpoint is legacy_midpoint


def test_move_battery_center_cancel_still_cancels_without_target_pose(
    arm_module,
):
    node = arm_module.ArmNode()
    goal = _Goal("MOVE_BATTERY_CENTER")
    goal.is_cancel_requested = True

    result = node.execute_callback(goal)

    assert not result.success
    assert result.error_code == "MOVE_BATTERY_CENTER_FAILED"
    assert goal.transition == "canceled"
    assert node.pub_target_pose.published == []
    assert [
        message.data
        for message in node.pub_task_command.published
    ] == ["MOVE_BATTERY_CENTER", "CANCEL_ARM_TASK"]


def test_scan_battery_aborts_before_detection_on_reset_protocol_failure(
    arm_module,
):
    node = arm_module.ArmNode()
    node.wait_for_isaac_completion = (
        lambda _goal, _feedback: True
    )

    def fail_reset(_publisher, _goal):
        node._last_perception_reset_error = (
            "bolt_cam perception protocol mismatch"
        )
        return False

    node._reset_perception_cache = fail_reset
    node.request_bolt_pair_midpoint_async = (
        lambda _goal: pytest.fail(
            "protocol mismatch must not enter an infinite target wait"
        )
    )

    goal = _Goal("SCAN_BATTERY")
    result = node.execute_callback(goal)

    assert not result.success
    assert result.error_code == "PERCEPTION_PROTOCOL_MISMATCH"
    assert "protocol mismatch" in result.message
    assert goal.transition == "aborted"


@pytest.mark.parametrize("task_type", ["SCAN_BATTERY", "SCAN_NUT1", "SCAN_NUT2"])
def test_scan_action_maps_post_reset_protocol_failure(
    arm_module,
    task_type,
):
    node = arm_module.ArmNode()
    node.wait_for_isaac_completion = (
        lambda _goal, _feedback: True
    )
    node._reset_perception_cache = (
        lambda _publisher, _goal: True
    )
    error = arm_module.PerceptionProtocolError(
        "perception generation regression after reset"
    )
    if task_type == "SCAN_BATTERY":
        node.request_bolt_pair_midpoint_async = (
            lambda _goal: (_ for _ in ()).throw(error)
        )
    else:
        node.request_grasp_pose_until_found = (
            lambda *_args: (_ for _ in ()).throw(error)
        )

    goal = _Goal(task_type)
    result = node.execute_callback(goal)

    assert not result.success
    assert result.error_code == "PERCEPTION_PROTOCOL_MISMATCH"
    assert "generation regression" in result.message
    assert goal.transition == "aborted"


def test_scan_busbar_protocol_failure_releases_camera_ready_pause(
    arm_module,
):
    node = arm_module.ArmNode()
    node.wait_for_isaac_phase = (
        lambda _phase, _goal, _feedback: True
    )
    node._reset_perception_cache = (
        lambda _publisher, _goal: True
    )
    node.request_grasp_pose_until_found = (
        lambda *_args: (_ for _ in ()).throw(
            arm_module.PerceptionProtocolError(
                "busbar_cam protocol mismatch after reset"
            )
        )
    )

    goal = _Goal("SCAN_BUSBAR")
    result = node.execute_callback(goal)

    assert not result.success
    assert result.error_code == "PERCEPTION_PROTOCOL_MISMATCH"
    assert goal.transition == "aborted"
    assert [
        message.data
        for message in node.pub_task_command.published
    ] == ["SCAN_BUSBAR", "CANCEL_ARM_TASK"]


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


@pytest.mark.parametrize(
    ("found", "message"),
    [
        (False, "'bolt' target not ready"),
        (False, "[generation=2] stale"),
        (True, "'bolt' target ready"),
        (True, "[generation=2] stale"),
    ],
)
def test_bolt_pair_all_responses_fail_on_invalid_generation(
    arm_module,
    found,
    message,
):
    node = arm_module.ArmNode()
    node._required_perception_generation["bolt_cam"] = 3
    response = types.SimpleNamespace(
        found=found,
        message=message,
    )

    class ImmediateFuture:
        def done(self):
            return True

        def result(self):
            return response

    node.client_get_bolt_pair = types.SimpleNamespace(
        wait_for_service=lambda timeout_sec: timeout_sec == 1.0,
        call_async=lambda _request: ImmediateFuture(),
    )

    with pytest.raises(
        arm_module.PerceptionProtocolError,
        match="protocol mismatch|generation regression",
    ):
        node.request_bolt_pair_midpoint_async(
            _Goal("SCAN_BATTERY"),
        )


def test_post_reset_bolt_transport_failure_remains_retryable(
    arm_module,
):
    node = arm_module.ArmNode()
    node._required_perception_generation["bolt_cam"] = 3
    pose_a = _pose(arm_module, 0.2, 0.0, 0.1, 101)
    pose_b = _pose(arm_module, 2.2, 0.0, 0.1, 101)
    outcomes = [
        RuntimeError("transient transport failure"),
        types.SimpleNamespace(
            found=True,
            pose_a=pose_a,
            pose_b=pose_b,
            message="[generation=3] fresh",
        ),
    ]

    class ImmediateFuture:
        def __init__(self, outcome):
            self.outcome = outcome

        def done(self):
            return True

        def result(self):
            if isinstance(self.outcome, Exception):
                raise self.outcome
            return self.outcome

    node.client_get_bolt_pair = types.SimpleNamespace(
        wait_for_service=lambda timeout_sec: timeout_sec == 1.0,
        call_async=lambda _request: ImmediateFuture(outcomes.pop(0)),
    )
    node._wait_while_active = lambda _duration, _goal: True

    found, midpoint, message = node.request_bolt_pair_midpoint_async(
        _Goal("SCAN_BATTERY"),
    )

    assert found
    assert midpoint.pose.position.x == pytest.approx(1.2)
    assert message == "[generation=3] fresh"
    assert not outcomes

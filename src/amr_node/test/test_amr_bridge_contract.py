"""Regression tests for AMR bridge goal and arrival ordering."""

import json

import pytest
import rclpy
from geometry_msgs.msg import Pose2D
from std_msgs.msg import String

from fms_interfaces.msg import AmrGoal, AmrStatus
from amr_node.amr_node import AmrNode


class _Recorder:
    """Minimal publisher replacement that records messages."""

    def __init__(self):
        self.messages = []

    def publish(self, message):
        """Record one published ROS message."""
        self.messages.append(message)


@pytest.fixture
def node():
    """Create an AMR node with observable publishers."""
    if not rclpy.ok():
        rclpy.init()
    instance = AmrNode()
    instance._status_pub = _Recorder()
    instance._goal_pose_pub = _Recorder()
    instance._cancel_pub = _Recorder()
    yield instance
    instance.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def _goal(station_id, x, y, theta=0.0):
    message = AmrGoal()
    message.station_id = station_id
    message.x = x
    message.y = y
    message.theta = theta
    return message


def test_replacement_goal_does_not_publish_cross_topic_cancel(node):
    """A late cancel must not be able to cancel the replacement goal."""
    node._on_goal(_goal('station3', 1.0, 2.0))
    node._on_goal(_goal('station4', 3.0, 4.0))

    assert node._cancel_pub.messages == []
    assert len(node._goal_pose_pub.messages) == 2
    assert node._moving
    assert node._goal_station_id == 'station4'


def test_identical_active_goal_keeps_existing_wheel_command(node):
    """A transport retry must not make Isaac interrupt the same drive."""
    goal = _goal('station3', 1.0, 2.0, 0.5)
    node._on_goal(goal)
    original_stamp = node._goal_stamp
    published_statuses = len(node._status_pub.messages)

    node._on_goal(_goal('station3', 1.0, 2.0, 0.5))

    assert node._moving
    assert node._goal_station_id == 'station3'
    assert node._goal_stamp == original_stamp
    assert len(node._goal_pose_pub.messages) == 1
    assert len(node._status_pub.messages) == published_statuses
    assert node._cancel_pub.messages == []


def test_same_station_with_changed_pose_replaces_active_goal(node):
    """Deduplication must not hide a real correction for one station."""
    node._on_goal(_goal('station3', 1.0, 2.0, 0.5))
    node._on_goal(_goal('station3', 1.01, 2.0, 0.5))

    assert node._moving
    assert node._goal_station_id == 'station3'
    assert node._goal_xy == (1.01, 2.0)
    assert len(node._goal_pose_pub.messages) == 2
    assert node._cancel_pub.messages == []


def test_sim_pose_never_short_circuits_physical_arrival(node):
    """Position-only feedback cannot satisfy yaw and settle requirements."""
    node._on_goal(_goal('station3', 1.0, 2.0, 0.5))
    feedback = Pose2D(x=1.0, y=2.0, theta=0.5)

    for _ in range(20):
        node._on_sim_pose(feedback)

    assert node._moving
    assert all(
        message.state != AmrStatus.STATE_ARRIVED
        for message in node._status_pub.messages
    )


def test_only_matching_drive_state_can_finish_goal(node):
    """Terminal state is accepted only for the current goal stamp."""
    node._on_goal(_goal('station3', 1.0, 2.0))
    current_stamp = node._goal_stamp

    stale = String()
    stale.data = json.dumps({
        'state': 'ARRIVED',
        'goal_stamp': 'old:stamp',
    })
    node._on_drive_state(stale)
    assert node._moving

    arrived = String()
    arrived.data = json.dumps({
        'state': 'ARRIVED',
        'goal_stamp': current_stamp,
        'message': '12 ticks stable',
    })
    node._on_drive_state(arrived)

    assert not node._moving
    assert node._status_pub.messages[-1].state == AmrStatus.STATE_ARRIVED

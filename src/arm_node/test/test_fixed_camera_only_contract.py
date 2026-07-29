"""Static contracts for the default fixed-camera-only busbar flow."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ARM_SOURCE = (
    REPO_ROOT / "src/arm_node/arm_node/arm_node.py"
).read_text(encoding="utf-8")
BEHAVIOR_SOURCE = (
    REPO_ROOT
    / "src/behavior_node/behavior_node/behavior_node.py"
).read_text(encoding="utf-8")
ISAAC_SOURCE = (
    REPO_ROOT / "isaacpjt/M0609/execute_isaac.py"
).read_text(encoding="utf-8")


def _between(source, start, end):
    return source[source.index(start):source.index(end, source.index(start))]


def test_default_behavior_requests_fixed_camera_latch_not_legacy_scan():
    state_branch = _between(
        BEHAVIOR_SOURCE,
        "elif self.state == ProcessState.SCAN_BUSBAR:",
        "elif self.state == ProcessState.PICK_BUSBAR:",
    )

    assert 'task_type="LATCH_BUSBAR"' in state_branch
    assert 'task_type="SCAN_BUSBAR"' not in state_branch


def test_fixed_camera_latch_finishes_before_legacy_wrist_path():
    fixed_only = _between(
        ARM_SOURCE,
        "if fixed_camera_only:",
        "continue_msg = String()",
    )

    assert 'complete_msg.data = "COMPLETE_BUSBAR_FIXED_SCAN"' in fixed_only
    assert "self.scanned_busbar_pose = fixed_pose" in fixed_only
    assert "self.wait_for_isaac_completion(" in fixed_only
    assert "self.pub_wrist_reset" not in fixed_only
    assert "wait_for_wrist_busbar_confirmation" not in fixed_only


def test_isaac_fixed_scan_handshake_holds_arm_and_reports_success():
    fixed_handshake = _between(
        ISAAC_SOURCE,
        'elif task == "COMPLETE_BUSBAR_FIXED_SCAN":',
        'elif task == "CONTINUE_BUSBAR_WRIST_SCAN":',
    )
    dispatcher_guard = _between(
        ISAAC_SOURCE,
        "busbar_camera_handshake = (",
        "if (\n                task",
    )

    assert '"COMPLETE_BUSBAR_FIXED_SCAN"' in dispatcher_guard
    assert '"CONTINUE_BUSBAR_WRIST_SCAN"' in dispatcher_guard
    assert 'phase == "WAIT_BUSBAR_CAMERA_LATCH"' in dispatcher_guard
    assert "hold_current_arm_pose()" in fixed_handshake
    assert 'publish_status("SUCCESS")' in fixed_handshake
    assert "busbar_wrist_scan_target(" not in fixed_handshake

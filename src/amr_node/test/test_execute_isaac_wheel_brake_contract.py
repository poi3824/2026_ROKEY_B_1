"""Source contracts for physical wheel driving and stopped arm work."""

from pathlib import Path


EXECUTOR_SOURCE = (
    Path(__file__).parents[3]
    / "isaacpjt"
    / "M0609"
    / "execute_isaac.py"
).read_text(encoding="utf-8")


def _source_between(start_marker, end_marker):
    start = EXECUTOR_SOURCE.index(start_marker)
    end = EXECUTOR_SOURCE.index(end_marker, start)
    return EXECUTOR_SOURCE[start:end]


def test_stopped_amr_uses_finite_wheel_position_brake():
    brake = _source_between(
        "def _configure_runtime_wheel_brake(robot):",
        "def _apply_amr_wheel_speeds(",
    )
    lock = _source_between(
        "def lock_amr_base(",
        "def unlock_amr_base(",
    )

    assert "joint_positions=wheel_positions" in brake
    assert "joint_velocities=np.zeros(2" in brake
    assert "stiffness=1.0e4" in brake
    assert "max_force=200.0" in brake
    assert "_configure_runtime_wheel_brake(robot)" in lock


def test_unlock_restores_velocity_drive_before_new_goal():
    unlock = _source_between(
        "def unlock_amr_base(",
        "def stiffen_gripper_grip(",
    )

    assert "_configure_runtime_wheel_drive(robot)" in unlock
    assert "set_world_pose" not in unlock


def test_smoke_unlocks_position_brake_before_velocity_commands():
    smoke_entry = _source_between(
        "if _SMOKE_TEST:",
        "# battery4_main의 검증된 너트 운반 방식",
    )

    unlock_at = smoke_entry.index("unlock_amr_base(")
    run_at = smoke_entry.index("run_wheel_smoke_test(")
    assert unlock_at < run_at


def test_arm_phase_rejects_new_amr_goal_without_unlocking():
    interlock = _source_between(
        "# 팔 FSM과 AMR wheel 주행은 동시에 실행하지 않는다.",
        "if isaac_node.amr_goal_pose is not None:",
    )

    assert 'phase not in {"IDLE", "DONE"}' in interlock
    assert "_apply_amr_wheel_speeds(" in interlock
    assert "lock_amr_base(" in interlock
    assert '"FAILED"' in interlock
    assert "unlock_amr_base(" not in interlock


def test_playback_restart_cancels_active_goal_with_original_stamp():
    restart = _source_between(
        "if playing and not was_playing:",
        "# battery4_main의 너트 AMR glue 추종",
    )

    capture_at = restart.index("restart_amr_goal_stamp = (")
    zero_at = restart.index("_apply_amr_wheel_speeds(", capture_at)
    reset_at = restart.index("world.reset()", zero_at)
    lock_at = restart.index("lock_amr_base(", reset_at)
    terminal_at = restart.index("publish_amr_drive_state(", lock_at)

    assert capture_at < zero_at < reset_at < lock_at < terminal_at
    terminal = restart[terminal_at:]
    assert '"CANCELED"' in terminal
    assert "restart_amr_goal_stamp," in terminal
    assert "pose=restart_amr_pose" in terminal


def test_busbar_arrival_latch_uses_approved_amr_tolerance():
    resolver = _source_between(
        "def resolve_busbar_stop_from_amr_xy(",
        "def busbar_station_pose_error(",
    )
    goal_setup = _source_between(
        "# 새 목표를 해석하기 전에 이전 command를 즉시 끊는다.",
        "pose = measure_amr_pose(robot)\n\n"
        "            if amr_moving",
    )
    arrival = _source_between(
        "if command.arrived:",
        "elif amr_drive_publish_step % 60 == 0:",
    )
    restart = _source_between(
        "if playing and not was_playing:",
        "# battery4_main의 너트 AMR glue 추종",
    )

    assert "tolerance=AMR_POS_TOL" in resolver
    assert "last_arrived_busbar_station = None" in goal_setup
    assert "resolve_busbar_stop_from_amr_xy(" in goal_setup
    assert (
        "last_arrived_busbar_station = (\n"
        "                        amr_goal_busbar_station"
        in arrival
    )
    assert "amr_goal_busbar_station = None" in arrival
    assert "last_arrived_busbar_station = None" in restart


def test_runtime_overlay_keeps_camera_source_stamps_monotonic_on_restart():
    overlay = _source_between(
        "def configure_monotonic_simulation_time(stage):",
        "def create_runtime_stage_overlay(",
    )
    preflight = _source_between(
        "def validate_stage_contract(stage):",
        "def run_wheel_smoke_test(",
    )

    assert '"inputs:resetSimulationTimeOnStop"' in overlay
    assert "reset_attr.Set(False)" in overlay
    assert '"inputs:resetSimulationTimeOnStop"' in preflight
    assert "bool(reset_attr.Get())" in preflight

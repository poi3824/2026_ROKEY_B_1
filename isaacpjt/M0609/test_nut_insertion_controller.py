import math

from nut_insertion_controller import (
    ForceGuidedNutInsertion,
    InsertionConfig,
    InsertionState,
    NutSensorFilter,
    NutSensorSample,
    RawNutSensors,
    evaluate_screw_state,
)


def _sample(*, axial=10.0, lateral=0.0, torque=1.0, z=1.0):
    return NutSensorSample(
        axial_force=axial,
        lateral_force=lateral,
        torque=torque,
        gripper_force=8.0,
        tcp_z=z,
        valid=True,
    )


def _fast_config(**overrides):
    values = dict(
        contact_hold_steps=2,
        center_min_time_s=0.02,
        spiral_duration_s=0.1,
        backoff_deg=10.0,
        thread_find_deg=30.0,
        thread_find_speed_deg_s=100.0,
        min_engagement_rotation_deg=5.0,
        engagement_drop_m=0.0004,
        cross_thread_hold_steps=1,
        recovery_duration_s=0.1,
    )
    values.update(overrides)
    return InsertionConfig(**values)


def test_sensor_filter_subtracts_tare_and_uses_named_wrist_signal():
    sensor_filter = NutSensorFilter(alpha=1.0)
    tare = RawNutSensors(
        wrench=(1.0, 2.0, 3.0, 0.0, 0.0, 4.0),
        wrist_effort=5.0,
        gripper_effort=7.0,
        tcp_z=1.0,
    )
    assert sensor_filter.add_tare_sample(tare)
    assert sensor_filter.finish_tare()

    measured = RawNutSensors(
        wrench=(1.0, 2.0, 13.0, 0.0, 0.0, 9.0),
        wrist_effort=7.0,
        gripper_effort=8.0,
        tcp_z=0.9,
    )
    sample = sensor_filter.update(measured)

    assert sample.valid
    assert sample.axial_force == 10.0
    assert sample.lateral_force == 0.0
    assert sample.torque == 5.0
    assert sample.gripper_force == 8.0


def test_contact_center_backoff_and_thread_engagement_sequence():
    controller = ForceGuidedNutInsertion(_fast_config())
    controller.reset(1.0)

    controller.update(_sample(), 0.01)
    command = controller.update(_sample(), 0.01)
    assert command.state == InsertionState.CENTER_SEARCH

    command = controller.update(_sample(), 0.02)
    assert command.state == InsertionState.THREAD_BACKOFF

    command = controller.update(_sample(), 0.1)
    assert command.state == InsertionState.THREAD_FORWARD
    assert command.rotation_deg == -10.0

    command = controller.update(_sample(z=0.999), 0.1)
    assert command.state == InsertionState.ENGAGED
    assert command.engaged


def test_cross_thread_recovery_stops_after_attempt_limit():
    controller = ForceGuidedNutInsertion(
        _fast_config(contact_hold_steps=1, max_attempts=1)
    )
    controller.reset(1.0)
    controller.update(_sample(), 0.01)
    controller.update(_sample(), 0.02)
    controller.update(_sample(), 0.1)

    command = controller.update(_sample(torque=30.0, z=1.0), 0.1)
    assert command.state == InsertionState.RECOVER

    command = controller.update(_sample(), 0.1)
    assert command.state == InsertionState.FAILED
    assert command.failed


def test_spiral_search_never_exceeds_configured_radius():
    config = _fast_config(
        contact_hold_steps=1,
        center_min_time_s=10.0,
        spiral_radius_m=0.0015,
    )
    controller = ForceGuidedNutInsertion(config)
    controller.reset(1.0)
    controller.update(_sample(), 0.01)

    for _ in range(5):
        command = controller.update(
            _sample(lateral=10.0), config.spiral_duration_s / 10.0
        )
        radius = math.hypot(command.x_offset, command.y_offset)
        assert radius <= config.spiral_radius_m + 1.0e-12


def test_seated_requires_depth_torque_stall_and_low_lateral_force():
    cross = evaluate_screw_state(
        planned_depth_m=0.002,
        actual_advance_m=0.0001,
        torque_nm=30.0,
        lateral_force_n=2.0,
        z_movement_m=0.0,
        engage_length_m=0.0125,
        seated_torque_nm=45.0,
        cross_thread_torque_nm=25.0,
        max_lateral_force_n=20.0,
        stall_z_movement_m=0.0001,
    )
    assert cross.cross_thread
    assert not cross.seated

    seated = evaluate_screw_state(
        planned_depth_m=0.010,
        actual_advance_m=0.010,
        torque_nm=50.0,
        lateral_force_n=2.0,
        z_movement_m=0.00005,
        engage_length_m=0.0125,
        seated_torque_nm=45.0,
        cross_thread_torque_nm=25.0,
        max_lateral_force_n=20.0,
        stall_z_movement_m=0.0001,
    )
    assert not seated.cross_thread
    assert seated.seated

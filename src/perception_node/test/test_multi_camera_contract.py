"""Contract tests for the three camera-specific perception instances."""

from pathlib import Path

from perception_node.multi_camera_config import (
    CAMERA_CONFIGS,
    DEFAULT_MODEL_FILENAME,
    FALLBACK_MODEL_FILENAME,
    perception_remappings,
)


def test_three_camera_sources_are_distinct_and_use_optical_frames():
    assert [config.namespace for config in CAMERA_CONFIGS] == [
        'wrist',
        'busbar_cam',
        'bolt_cam',
    ]
    assert len({config.rgb_topic for config in CAMERA_CONFIGS}) == 3
    assert all(config.optical_frame.endswith('_optical_frame') for config in CAMERA_CONFIGS)


def test_required_services_and_reset_topics_are_namespaced():
    remappings = {
        config.namespace: dict(perception_remappings(config))
        for config in CAMERA_CONFIGS
    }

    assert remappings['wrist']['/perception/get_grasp_pose'] == (
        '/wrist/perception/get_grasp_pose'
    )
    assert remappings['busbar_cam']['/perception/get_grasp_pose'] == (
        '/busbar_cam/perception/get_grasp_pose'
    )
    assert remappings['bolt_cam']['/perception/get_bolt_pair'] == (
        '/bolt_cam/perception/get_bolt_pair'
    )
    for namespace in remappings:
        assert remappings[namespace]['/perception/reset_cache'] == (
            f'/{namespace}/perception/reset_cache'
        )
        assert remappings[namespace]['/perception/reset_ack'] == (
            f'/{namespace}/perception/reset_ack'
        )


def test_only_intended_legacy_vision_outputs_remain_global():
    remappings = {
        config.namespace: dict(perception_remappings(config))
        for config in CAMERA_CONFIGS
    }

    assert remappings['wrist']['/vision/nut_pose'] == '/vision/nut_pose'
    assert remappings['busbar_cam']['/vision/busbar_grasp'] == '/vision/busbar_grasp'
    assert remappings['bolt_cam']['/vision/nut_pose'] == '/bolt_cam/vision/nut_pose'


def test_only_fixed_busbar_camera_forces_depth_priority():
    priorities = {
        config.namespace: config.prefer_busbar_depth
        for config in CAMERA_CONFIGS
    }
    assert priorities == {
        'wrist': False,
        'busbar_cam': True,
        'bolt_cam': False,
    }


def test_only_wrist_camera_uses_on_demand_inference():
    on_demand = {
        config.namespace: config.on_demand_inference
        for config in CAMERA_CONFIGS
    }

    assert on_demand == {
        'wrist': True,
        'busbar_cam': False,
        'bolt_cam': False,
    }


def test_only_bolt_camera_uses_calibrated_640_inference_and_roi():
    configs = {
        config.namespace: config
        for config in CAMERA_CONFIGS
    }

    assert configs['wrist'].inference_image_size is None
    assert configs['busbar_cam'].inference_image_size is None
    assert configs['bolt_cam'].inference_image_size == 640
    assert configs['bolt_cam'].bolt_roi_bounds == (150, 560, 180, 450)


def test_offline_bolt_pixels_keep_target_pack_and_reject_neighbours():
    bounds = next(
        config.bolt_roi_bounds
        for config in CAMERA_CONFIGS
        if config.namespace == 'bolt_cam'
    )
    x_min, x_max, y_min, y_max = bounds

    def accepted(pixel, score):
        x, y = pixel
        return (
            score >= 0.6
            and x_min <= x <= x_max
            and y_min <= y <= y_max
        )

    # v3 inference on the live station-5 Camera_bolt frame. TF/USD matching
    # assigns (531,199) to bolt_1 and camera-centred (320,321) to bolt_2.
    assert accepted((531, 199), 0.615)
    assert accepted((320, 321), 0.732)
    assert not accepted((320, 521), 0.613)
    assert not accepted((531, 400), 0.572)
    assert not accepted((628, 320), 0.571)
    assert not accepted((629, 511), 0.317)


def test_default_and_fallback_models_are_packaged():
    package_root = Path(__file__).parents[1]
    setup_source = (package_root / 'setup.py').read_text(encoding='utf-8')

    for filename in (DEFAULT_MODEL_FILENAME, FALLBACK_MODEL_FILENAME):
        assert (package_root / 'models' / filename).is_file()
        assert f"'models/{filename}'" in setup_source
    assert (package_root / 'models' / DEFAULT_MODEL_FILENAME).stat().st_size == 6_591_443


def test_reset_uses_last_callback_observed_rgb_and_depth_boundaries():
    source = (
        Path(__file__).parents[1]
        / 'perception_node'
        / 'perception_node.py'
    ).read_text(encoding='utf-8')

    assert 'self._last_observed_rgb_stamp_ns' in source
    assert 'self._last_observed_depth_stamp_ns' in source
    reset_source = source[
        source.index('def _on_reset_cache'):
        source.index('def _detect_and_publish')
    ]
    assert 'rgb_boundary_ns=self._last_observed_rgb_stamp_ns' in reset_source
    assert (
        'depth_boundary_ns=self._last_observed_depth_stamp_ns'
        in reset_source
    )


def test_inference_cannot_starve_reset_or_service_callbacks():
    source = (
        Path(__file__).parents[1]
        / 'perception_node'
        / 'perception_node.py'
    ).read_text(encoding='utf-8')

    assert 'MultiThreadedExecutor(num_threads=4)' in source
    assert 'self._sensor_callback_group = ReentrantCallbackGroup()' in source
    assert 'self._control_callback_group = ReentrantCallbackGroup()' in source
    assert (
        'self._inference_callback_group = '
        'MutuallyExclusiveCallbackGroup()'
    ) in source
    assert 'self._inference_lock.acquire(blocking=False)' in source
    assert 'discarding inference with stale input at commit' in source
    assert 'callback_generation = self._target_barrier.generation' in source
    assert 'self._target_barrier.is_current(' in source
    assert 'rgb_frame.received_at' in source
    assert 'depth_frame.received_at' in source
    assert 'self._reset_monotonic_cutoff' in source
    assert 'received_at <= self._reset_monotonic_cutoff' in source
    assert 'SourceTimestampEpochTracker(' in source
    assert 'epoch_result.rebase' in source
    assert '_hard_reset_source_epoch_locked' in source
    assert 'camera_model = self._camera_model' in source
    assert 'camera_frame_id = self._camera_frame_id' in source
    assert 'select_sticky_single_target(' in source
    assert 'self._target_present_latest[BOLT_LABEL] = False' in source
    assert 'stale_inference' in source


def test_bolt_pair_association_is_idle_until_its_explicit_service_request():
    source = (
        Path(__file__).parents[1]
        / 'perception_node'
        / 'perception_node.py'
    ).read_text(encoding='utf-8')
    inference_source = source[
        source.index('    def _detect_and_publish_once(self):'):
        source.index('    def _record_stale_inference(')
    ]
    service_source = source[
        source.index('    def _handle_get_bolt_pair(self, request, response):'):
        source.index('    def _handle_get_bolt_pair_locked(')
    ]

    # Generic Detection3D output remains available, but same-pack selection
    # and its warnings are not allowed to churn while no caller requested a
    # bolt-pair snapshot.
    assert 'bolt_pair = None' in inference_source
    assert 'if bolt_pair_tracking_active:' in inference_source
    assert inference_source.index('if bolt_pair_tracking_active:') < (
        inference_source.index('select_central_bolt_pair(')
    )
    assert 'if bolt_pair_capture_current:' in inference_source
    assert 'self._arm_bolt_pair_tracking_locked()' in service_source
    assert 'self._cancel_bolt_pair_tracking_locked()' in service_source


def test_wrist_on_demand_mode_fences_idle_gpu_work_and_keeps_services():
    source = (
        Path(__file__).parents[1]
        / 'perception_node'
        / 'perception_node.py'
    ).read_text(encoding='utf-8')
    inference_source = source[
        source.index('    def _detect_and_publish_once(self):'):
        source.index('    def _record_stale_inference(')
    ]
    grasp_service = source[
        source.index('    def _handle_get_grasp_pose(self, request, response):'):
        source.index('    def _tag_service_generation(')
    ]

    assert "self.declare_parameter('on_demand_inference', False)" in source
    launch_source = (
        Path(__file__).parents[2]
        / 'fms_bringup'
        / 'launch'
        / 'fms_bringup.launch.py'
    ).read_text(encoding='utf-8')
    assert "'on_demand_inference': config.on_demand_inference" in launch_source
    assert 'def _arm_on_demand_inference_locked' in source
    assert 'def _finish_on_demand_inference_locked' in source
    assert 'and not getattr(self, \'_demand_capture_active\', False)' in inference_source
    assert 'PerceptionNode._arm_on_demand_inference_locked(self)' in grasp_service
    assert 'PerceptionNode._finish_on_demand_inference_locked(self)' in grasp_service


def test_inference_never_moves_callback_observed_reset_boundary_backwards():
    source = (
        Path(__file__).parents[1]
        / 'perception_node'
        / 'perception_node.py'
    ).read_text(encoding='utf-8')
    inference_source = source[
        source.index('    def _detect_and_publish_once(self):'):
        source.index('    def _record_stale_inference(')
    ]

    assert (
        'self._last_observed_rgb_stamp_ns = advance_source_boundary('
        in inference_source
    )
    assert (
        'self._last_observed_depth_stamp_ns = advance_source_boundary('
        in inference_source
    )


def test_low_confidence_busbar_skips_pnp_but_keeps_depth_fallback():
    source = (
        Path(__file__).parents[1]
        / 'perception_node'
        / 'perception_node.py'
    ).read_text(encoding='utf-8')
    resolver = source[
        source.index('def _resolve_busbar_pose'):
        source.index('def _make_detection3d')
    ]
    detection_loop = source[
        source.index('for det in self._detector.detect'):
        source.index('if debug_image is not None:', source.index(
            'for det in self._detector.detect'))
    ]

    depth_at = resolver.index('self._transform_pixel(')
    gate_at = resolver.index('busbar_keypoints_are_reliable(')
    pnp_at = resolver.index('busbar_pnp_world_pose(')
    selector_at = resolver.index('select_busbar_world_point(')
    rejected_landmarks = resolver[
        resolver.index('if not keypoints_reliable:'):
        resolver.index('elif keypoints.shape')
    ]

    assert depth_at < gate_at < pnp_at < selector_at
    assert 'return' not in rejected_landmarks
    assert 'depth_world,' in resolver[selector_at:]
    assert 'busbar_keypoints_are_reliable(' not in detection_loop

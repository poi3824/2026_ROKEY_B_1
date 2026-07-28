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


def test_only_bolt_camera_uses_calibrated_640_inference_and_roi():
    configs = {
        config.namespace: config
        for config in CAMERA_CONFIGS
    }

    assert configs['wrist'].inference_image_size is None
    assert configs['busbar_cam'].inference_image_size is None
    assert configs['bolt_cam'].inference_image_size == 640
    assert configs['bolt_cam'].bolt_roi_bounds == (80, 560, 150, 450)


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

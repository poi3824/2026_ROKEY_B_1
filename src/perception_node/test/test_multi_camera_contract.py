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


def test_default_and_fallback_models_are_packaged():
    package_root = Path(__file__).parents[1]
    setup_source = (package_root / 'setup.py').read_text(encoding='utf-8')

    for filename in (DEFAULT_MODEL_FILENAME, FALLBACK_MODEL_FILENAME):
        assert (package_root / 'models' / filename).is_file()
        assert f"'models/{filename}'" in setup_source
    assert (package_root / 'models' / DEFAULT_MODEL_FILENAME).stat().st_size == 6_591_443

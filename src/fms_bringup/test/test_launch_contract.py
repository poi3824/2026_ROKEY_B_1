"""Executable contract test for the composed FMS launch description."""

import importlib.util
from pathlib import Path
import sys
import types


class _LaunchDescription:
    def __init__(self, entities):
        self.entities = entities


class _Action:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _Node(_Action):
    pass


class _Substitution(_Action):
    pass


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def test_launch_wires_three_isolated_perception_instances(monkeypatch):
    """Load the actual launch file and inspect every generated camera node."""
    modules = {
        'launch': _module(
            'launch',
            LaunchDescription=_LaunchDescription,
        ),
        'launch.actions': _module(
            'launch.actions',
            DeclareLaunchArgument=_Action,
        ),
        'launch.substitutions': _module(
            'launch.substitutions',
            LaunchConfiguration=_Substitution,
            PathJoinSubstitution=_Substitution,
        ),
        'launch_ros': _module('launch_ros'),
        'launch_ros.actions': _module(
            'launch_ros.actions',
            Node=_Node,
        ),
        'launch_ros.substitutions': _module(
            'launch_ros.substitutions',
            FindPackageShare=_Substitution,
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    src_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(src_root / 'perception_node'))
    launch_path = (
        Path(__file__).resolve().parents[1]
        / 'launch'
        / 'fms_bringup.launch.py'
    )
    spec = importlib.util.spec_from_file_location(
        'fms_bringup_launch_under_test',
        launch_path,
    )
    launch_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch_module)

    description = launch_module.generate_launch_description()
    perception_nodes = [
        entity
        for entity in description.entities
        if isinstance(entity, _Node)
        and entity.kwargs['package'] == 'perception_node'
    ]

    assert [node.kwargs['name'] for node in perception_nodes] == [
        'wrist_perception_node',
        'busbar_perception_node',
        'bolt_perception_node',
    ]
    for node, config in zip(
        perception_nodes,
        launch_module.CAMERA_CONFIGS,
    ):
        parameters = node.kwargs['parameters'][0]
        assert parameters['rgb_topic'] == config.rgb_topic
        assert parameters['depth_topic'] == config.depth_topic
        assert parameters['camera_info_topic'] == config.camera_info_topic
        assert parameters['camera_frame_override'] == config.optical_frame
        assert isinstance(parameters['model_path'], _Substitution)
        assert parameters['model_path'].args == ('perception_model_path',)
        assert parameters['max_rgb_depth_skew_sec'] == 0.1
        assert parameters['max_input_receipt_age_sec'] == 1.0
        assert parameters['require_unique_camera_source'] is True
        assert parameters['required_target_frames'] == 3
        assert parameters['inference_image_size'] == (
            config.inference_image_size
            if config.inference_image_size is not None
            else 0
        )
        assert (
            parameters['bolt_roi_x_min'],
            parameters['bolt_roi_x_max'],
            parameters['bolt_roi_y_min'],
            parameters['bolt_roi_y_max'],
        ) == config.bolt_roi_bounds

    remappings = [
        dict(node.kwargs['remappings'])
        for node in perception_nodes
    ]
    assert remappings[0]['/perception/get_grasp_pose'] == (
        '/wrist/perception/get_grasp_pose'
    )
    assert remappings[1]['/perception/get_grasp_pose'] == (
        '/busbar_cam/perception/get_grasp_pose'
    )
    assert remappings[2]['/perception/get_bolt_pair'] == (
        '/bolt_cam/perception/get_bolt_pair'
    )
    for mapping, namespace in zip(
        remappings,
        ('wrist', 'busbar_cam', 'bolt_cam'),
    ):
        assert mapping['/perception/reset_cache'] == (
            f'/{namespace}/perception/reset_cache'
        )
        assert mapping['/perception/reset_ack'] == (
            f'/{namespace}/perception/reset_ack'
        )

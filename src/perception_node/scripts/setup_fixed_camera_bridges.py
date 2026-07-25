"""setup_fixed_camera_bridges.py -- World0123.usd에 고정 카메라(Camera_busbar, Camera_bolt)용
ROS2 카메라 브리지(RGB/Depth/CameraInfo)와 optical_frame 자식 프림을 추가하고 저장한다
(1회성 스크립트, setup_ros2_arm_bridge.py와 동일 패턴).

배경: 지금은 손목(그리퍼) RealSense 카메라 하나(RSD455/Camera_OmniVision_OV9782_Color)만
/rgb, /depth, /camera_info로 퍼블리시되고 있어, 버스바/볼트 검출이 팔 위치에 종속된다.
Camera_busbar/Camera_bolt는 /World 밑에 고정으로 이미 존재하지만 어떤 ROS2 그래프에도
연결이 안 돼 있었다. targetPrims에 두 카메라를 추가하는 것까지는 사용자가 GUI에서 직접
했으나(ros2_publish_transform_tree), 그 프림 자체를 넣어서 축 컨벤션이 안 맞고(USD 카메라:
Y-up/-Z-forward, ROS optical: Y-down/+Z-forward), camera_graph(RGB/Depth/CameraInfo)도
아직 확장 전이었다.

이 스크립트가 하는 일:
  1. Camera_busbar, Camera_bolt 밑에 optical_frame 자식 Xform 추가
     (RSD455/Camera_OmniVision_OV9782_Color/camera_color_optical_frame과 동일하게
     translate=(0,0,0), orient=180도 X축 회전 -- USD 카메라 -> ROS optical 축 변환)
  2. ros2_publish_transform_tree의 targetPrims에서 기존에 추가돼 있던 카메라 프림
     자체(Camera_busbar, Camera_bolt) 대신 새로 만든 optical_frame 자식으로 교체
  3. /World/Graph/camera_graph에 카메라별 RenderProduct + RGB/Depth/CameraInfo Publish
     노드를 추가 (기존 OnPlaybackTick/Context 재사용, RunOnce/RenderProduct는 카메라마다
     새로 생성 -- 기존 손목캠 서브그래프와 동일 패턴)
     busbar: /busbar_cam/rgb, /busbar_cam/depth, /busbar_cam/camera_info
     bolt:   /bolt_cam/rgb,   /bolt_cam/depth,   /bolt_cam/camera_info

실행:
  /home/rokey/dev_ws/isaac_sim/isaacsim/_build/linux-x86_64/release/python.sh \
      /home/rokey/EV_combine/src/perception_node/scripts/setup_fixed_camera_bridges.py
"""
import os

_HEADLESS = os.environ.get("BOLT_HEADLESS", "1") == "1"
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": _HEADLESS})

import omni.usd
import omni.kit.app
import omni.graph.core as og
from pxr import Gf, UsdGeom

WORLD_USD = "/home/rokey/EV_combine/src/Collected_World_0123/World0123.usd"
GRAPH_PATH = "/World/Graph/camera_graph"
TF_GRAPH_PATH = "/World/ActionGraph"

# RSD455/Camera_OmniVision_OV9782_Color/camera_color_optical_frame과 동일한 값
# (USD 카메라 축 -> ROS optical 축 변환: X축 180도 회전)
OPTICAL_FRAME_ORIENT = Gf.Quatf(6.123233995736766e-17, Gf.Vec3f(1.0, 0.0, 0.0))

CAMERAS = [
    {
        "label": "busbar",
        "camera_prim": "/World/Camera_busbar",
        "optical_frame_name": "busbar_cam_optical_frame",
        "rgb_topic": "/busbar_cam/rgb",
        "depth_topic": "/busbar_cam/depth",
        "camera_info_topic": "/busbar_cam/camera_info",
        "frame_id": "busbar_cam",
    },
    {
        "label": "bolt",
        "camera_prim": "/World/Camera_bolt",
        "optical_frame_name": "bolt_cam_optical_frame",
        "rgb_topic": "/bolt_cam/rgb",
        "depth_topic": "/bolt_cam/depth",
        "camera_info_topic": "/bolt_cam/camera_info",
        "frame_id": "bolt_cam",
    },
]

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.core.nodes", True)
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
for _ in range(10):
    simulation_app.update()

context = omni.usd.get_context()
context.open_stage(WORLD_USD)
for _ in range(20):
    simulation_app.update()

stage = context.get_stage()

# 1. optical_frame 자식 프림 추가
optical_frame_paths = {}
for cam in CAMERAS:
    parent = stage.GetPrimAtPath(cam["camera_prim"])
    assert parent.IsValid(), f'카메라 프림 없음: {cam["camera_prim"]}'
    child_path = f'{cam["camera_prim"]}/{cam["optical_frame_name"]}'
    xform = UsdGeom.Xform.Define(stage, child_path)
    existing_ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
    if "xformOp:translate" in existing_ops:
        existing_ops["xformOp:translate"].Set(Gf.Vec3d(0.0, 0.0, 0.0))
        existing_ops["xformOp:orient"].Set(OPTICAL_FRAME_ORIENT)
        existing_ops["xformOp:scale"].Set(Gf.Vec3f(1.0, 1.0, 1.0))
    else:
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        xform.AddOrientOp().Set(OPTICAL_FRAME_ORIENT)
        xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    optical_frame_paths[cam["label"]] = child_path
    print(f'[optical_frame 추가] {child_path}')

# 2. ros2_publish_transform_tree의 targetPrims 교체
#    (기존 카메라 프림 자체 참조 -> optical_frame 자식 참조로 대체)
tf_prim = stage.GetPrimAtPath(f"{TF_GRAPH_PATH}/ros2_publish_transform_tree")
assert tf_prim.IsValid(), "ros2_publish_transform_tree 프림 없음"
target_rel = tf_prim.GetRelationship("inputs:targetPrims")
current_targets = list(target_rel.GetTargets())
optical_frame_target_strs = set(optical_frame_paths.values())
camera_prim_strs = {CAMERAS[0]["camera_prim"], CAMERAS[1]["camera_prim"]}
new_targets = []
for t in current_targets:
    t_str = str(t)
    if t_str in camera_prim_strs or t_str in optical_frame_target_strs:
        continue  # 카메라 프림 자체 참조는 제거, optical_frame은 아래서 새로 추가하니 중복 방지
    new_targets.append(t)
for cam in CAMERAS:
    new_targets.append(optical_frame_paths[cam["label"]])
target_rel.SetTargets(new_targets)
print(f'[targetPrims 갱신] {new_targets}')

# 3. camera_graph에 카메라별 RenderProduct + RGB/Depth/CameraInfo Publish 노드 추가
#    (OnPlaybackTick, Context는 기존 것 재사용)
keys = og.Controller.Keys
create_nodes = []
set_values = []
connect = []

for cam in CAMERAS:
    label = cam["label"]
    run_once = f"RunOnce_{label}"
    render_product = f"RenderProduct_{label}"
    cam_info = f"CameraInfoPublish_{label}"
    rgb_pub = f"RGBPublish_{label}"
    depth_pub = f"DepthPublish_{label}"

    for node_name, node_type in [
        (run_once, "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
        (render_product, "isaacsim.core.nodes.IsaacCreateRenderProduct"),
        (cam_info, "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
        (rgb_pub, "isaacsim.ros2.bridge.ROS2CameraHelper"),
        (depth_pub, "isaacsim.ros2.bridge.ROS2CameraHelper"),
    ]:
        # og.Controller.edit의 CREATE_NODES는 upsert가 아니라 이미 존재하는 노드 경로면
        # 그냥 에러(OmniGraphError: already exists) -- 재실행 멱등성을 위해 이미 있는
        # 노드는 건너뛰고 SET_VALUES/CONNECT만 다시 적용한다(둘은 기존 값 덮어쓰기라 안전).
        if not stage.GetPrimAtPath(f"{GRAPH_PATH}/{node_name}").IsValid():
            create_nodes.append((node_name, node_type))
    set_values += [
        (f"{render_product}.inputs:width", 640),
        (f"{render_product}.inputs:height", 640),
        # optical_frame 자식(UsdGeom.Xform, 렌즈 속성 없음)이 아니라 실제 Camera 프림을
        # 렌더 소스로 써야 한다. 자식을 cameraPrim으로 넣었더니(2026-07-25 실측)
        # busbar_cam/bolt_cam 두 RenderProduct가 렌즈 속성 없는 프림을 못 찾고 똑같이
        # Camera_bolt로 fallback되는 버그가 발견됨 -- /busbar_cam/rgb와 /bolt_cam/rgb가
        # 완전히 동일한 이미지(오블리크 Camera_bolt 뷰)로 나왔고, tf는 여전히 정상인
        # busbar_cam_optical_frame(수직 하향)을 가리켜서 depth/PnP가 "존재하지 않는
        # 카메라 자세"로 좌표를 계산해 world z가 바닥 아래로 나오는 원인이 됐다.
        # optical_frame 자식은 ros2_publish_transform_tree(tf 좌표계 의미론)에만 쓰고,
        # 렌더링은 실제 Camera 프림 기준으로 한다.
        (f"{render_product}.inputs:cameraPrim", cam["camera_prim"]),
        (f"{cam_info}.inputs:topicName", cam["camera_info_topic"]),
        (f"{cam_info}.inputs:frameId", cam["frame_id"]),
        (f"{rgb_pub}.inputs:topicName", cam["rgb_topic"]),
        (f"{rgb_pub}.inputs:frameId", cam["frame_id"]),
        (f"{rgb_pub}.inputs:type", "rgb"),
        (f"{depth_pub}.inputs:topicName", cam["depth_topic"]),
        (f"{depth_pub}.inputs:frameId", cam["frame_id"]),
        (f"{depth_pub}.inputs:type", "depth"),
    ]
    connect += [
        (f"{GRAPH_PATH}/OnPlaybackTick.outputs:tick", f"{run_once}.inputs:execIn"),
        (f"{run_once}.outputs:step", f"{render_product}.inputs:execIn"),
        (f"{render_product}.outputs:execOut", f"{cam_info}.inputs:execIn"),
        (f"{render_product}.outputs:renderProductPath", f"{cam_info}.inputs:renderProductPath"),
        (f"{GRAPH_PATH}/Context.outputs:context", f"{cam_info}.inputs:context"),
        (f"{render_product}.outputs:execOut", f"{rgb_pub}.inputs:execIn"),
        (f"{render_product}.outputs:renderProductPath", f"{rgb_pub}.inputs:renderProductPath"),
        (f"{GRAPH_PATH}/Context.outputs:context", f"{rgb_pub}.inputs:context"),
        (f"{render_product}.outputs:execOut", f"{depth_pub}.inputs:execIn"),
        (f"{render_product}.outputs:renderProductPath", f"{depth_pub}.inputs:renderProductPath"),
        (f"{GRAPH_PATH}/Context.outputs:context", f"{depth_pub}.inputs:context"),
    ]

og.Controller.edit(
    GRAPH_PATH,
    {
        keys.CREATE_NODES: create_nodes,
        keys.SET_VALUES: set_values,
        keys.CONNECT: connect,
    },
)

for _ in range(5):
    simulation_app.update()

stage.GetRootLayer().Save()

_log_f = open(os.path.join(os.path.dirname(__file__), "setup_fixed_camera_bridges_result.txt"), "w")
_log_f.write(f"[저장 완료] {WORLD_USD}\n")
for cam in CAMERAS:
    _log_f.write(
        f'{cam["label"]}: optical_frame={optical_frame_paths[cam["label"]]} '
        f'rgb={cam["rgb_topic"]} depth={cam["depth_topic"]} info={cam["camera_info_topic"]} '
        f'frame_id={cam["frame_id"]}\n'
    )
_log_f.close()

simulation_app.close()

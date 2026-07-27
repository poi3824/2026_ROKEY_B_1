#!/usr/bin/env python3
"""rosbag2에서 프레임 딱 하나만 오프라인으로 읽어 YOLO 검출 -> 좌표 변환 -> 오버레이
PNG로 저장하는 디버그 도구.

이 노트북은 CPU-only라 YOLO 세그멘테이션 추론에 프레임당 8~10초가 걸린다.
`ros2 bag play`로 실시간 재생하면서 perception_node의 0.5초 주기 타이머로 검출하면,
추론이 끝날 때쯤엔 이미 몇 초가 지나버려서 tf2 버퍼(기본 10초 캐시)가 그 시점의
tf를 이미 밀어낸 뒤라 "tf fail"이 거의 항상 발생한다 (실시간 재생 자체의 구조적
한계이지 코드 버그가 아님).

이 스크립트는 실시간 재생을 아예 쓰지 않는다: bag 파일에서 /tf 전체를 미리 다 읽어
tf2 Buffer에 채워 넣고, 원하는 인덱스의 rgb/depth/camera_info 프레임 하나만 골라
곧바로 검출 + 좌표 변환을 수행한다. 타이머도 실시간 재생도 없으니 추론이 아무리
오래 걸려도 tf 조회는 항상 성공한다(그 프레임 시점의 tf가 이미 버퍼에 다 있으므로).

perception_node.py와 동일한 camera_geometry.py / detector.py / overlay.py 로직을
그대로 재사용한다.

사용법 (ROS 2 언더레이는 source 필요, 이 워크스페이스는 colcon build 불필요):
    source /opt/ros/humble/setup.bash
    python3 inspect_bag_frame.py <bag_dir> --index 50 --out frame_50.png
"""
import argparse
import os
import sys

# perception_node 패키지는 순수 파이썬이라 colcon build/install 없이도 소스에서 바로
# import 가능하게, 이 스크립트 위치 기준 부모 디렉터리(src/perception_node/)를
# sys.path에 추가한다. install/setup.bash를 source했다면 이 경로가 site-packages의
# egg-link가 가리키는 곳과 동일해서 동작 차이 없음.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage
from tf2_ros.buffer import Buffer

from perception_node import overlay
from perception_node.camera_geometry import make_camera_model, transform_pixel_to_world
from perception_node.detector import YoloSegDetector

DEFAULT_MODEL_PATH = None  # 아래 main()에서 패키지 share 경로로 채움


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag_dir', help='rosbag2 디렉터리 (metadata.yaml이 있는 곳)')
    parser.add_argument('--index', type=int, default=0,
                         help='몇 번째 rgb/depth/camera_info 프레임을 볼지 (0-based)')
    parser.add_argument('--rgb-topic', default='/rgb')
    parser.add_argument('--depth-topic', default='/depth')
    parser.add_argument('--camera-info-topic', default='/camera_info')
    parser.add_argument('--tf-topic', default='/tf')
    parser.add_argument('--world-frame', default='world')
    parser.add_argument('--camera-frame-override', default='camera_color_optical_frame',
                         help='tf 조회에 쓸 프레임 이름 (이미지 헤더 frame_id와 다를 수 있음)')
    parser.add_argument('--conf-threshold', type=float, default=0.5)
    parser.add_argument('--model-path', default=None,
                         help='기본값: perception_node 패키지의 share/models/best.pt')
    parser.add_argument('--out', default=None, help='기본값: <bag_dir>_frame<index>.png')
    return parser.parse_args()


def read_frame(bag_dir, index, rgb_topic, depth_topic, camera_info_topic, tf_topic):
    """bag에서 /tf 전체 + 지정한 인덱스의 rgb/depth/camera_info 메시지만 읽어온다."""
    storage_options = StorageOptions(uri=bag_dir, storage_id='sqlite3')
    converter_options = ConverterOptions('cdr', 'cdr')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    tf_buffer = Buffer(cache_time=Duration(seconds=3600))  # 오프라인 분석용, 넉넉하게

    counts = {rgb_topic: 0, depth_topic: 0, camera_info_topic: 0}
    picked = {rgb_topic: None, depth_topic: None, camera_info_topic: None}

    while reader.has_next():
        topic, data, _t = reader.read_next()
        if topic == tf_topic:
            tf_msg = deserialize_message(data, TFMessage)
            for transform in tf_msg.transforms:
                tf_buffer.set_transform(transform, 'bag')
        elif topic in counts:
            if counts[topic] == index:
                msg_type = Image if topic != camera_info_topic else CameraInfo
                picked[topic] = deserialize_message(data, msg_type)
            counts[topic] += 1

    missing = [t for t, msg in picked.items() if msg is None]
    if missing:
        raise RuntimeError(
            f'index={index} 프레임을 못 찾음 (topics without enough messages: {missing}, '
            f'실제 메시지 수: { {t: c for t, c in counts.items()} })')

    return picked[rgb_topic], picked[depth_topic], picked[camera_info_topic], tf_buffer


def main():
    args = parse_args()

    model_path = args.model_path
    if model_path is None:
        try:
            model_path = f'{get_package_share_directory("perception_node")}/models/best.pt'
        except Exception:
            # install/setup.bash를 source 안 했으면 ament index에 없음 -> 소스 트리
            # 경로(src/perception_node/models/best.pt)로 대체.
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'best.pt')

    rgb_msg, depth_msg, camera_info_msg, tf_buffer = read_frame(
        args.bag_dir, args.index, args.rgb_topic, args.depth_topic,
        args.camera_info_topic, args.tf_topic)

    bridge = CvBridge()
    rgb = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
    depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

    camera_model = make_camera_model(camera_info_msg)
    camera_frame_id = args.camera_frame_override or camera_info_msg.header.frame_id

    detector = YoloSegDetector(model_path)
    print(f'추론 중... (index={args.index}, stamp={rgb_msg.header.stamp.sec}.'
          f'{rgb_msg.header.stamp.nanosec:09d})', file=sys.stderr)
    detections = detector.detect(rgb, args.conf_threshold)
    print(f'{len(detections)}개 검출됨', file=sys.stderr)

    debug_image = rgb.copy()
    for det in detections:
        def log_tf_error(ex):
            print(f'  [{det["label"]}] tf 조회 실패: {ex}', file=sys.stderr)

        camera_point, world_point, status = transform_pixel_to_world(
            camera_model, depth, det['pixel'], tf_buffer, args.world_frame,
            camera_frame_id, rgb_msg.header.stamp, on_tf_error=log_tf_error)

        overlay.draw_detection(debug_image, det, camera_point, world_point, status)
        print(f'  {det["label"]} score={det["score"]:.2f} pixel={det["pixel"]} '
              f'camera={camera_point} world={world_point} status="{status}"', file=sys.stderr)

    out_path = args.out or f'{args.bag_dir.rstrip("/")}_frame{args.index}.png'
    cv2.imwrite(out_path, debug_image)
    print(f'저장됨: {out_path}', file=sys.stderr)


if __name__ == '__main__':
    main()

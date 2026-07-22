import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'perception_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), ['models/best.pt']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='soo',
    maintainer_email='poi3824@gmail.com',
    description='YOLO 세그멘테이션 기반 bolt/nut/busbar 검출 -> world 좌표 변환 -> 토픽 발행 노드',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'perception_node = perception_node.perception_node:main'
        ],
    },
)

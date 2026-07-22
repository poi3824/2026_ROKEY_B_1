import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'arm_node'

setup(
    name=package_name,
    version='0.0.0',
    # find_packages()만 사용하여 실제 파이썬 모듈만 패키징
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # data 폴더 내부의 json 파일들을 share/arm_node/data 디렉터리로 복사
        (os.path.join('share', package_name, 'data'), glob('arm_node/data/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='soo',
    maintainer_email='poi3824@gmail.com',
    description='버스바 파지·삽입·너트 체결을 수행하는 매니퓰레이터 노드',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'arm_node = arm_node.arm_node:main'
        ],
    },
)
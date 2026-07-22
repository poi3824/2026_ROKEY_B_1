from setuptools import find_packages, setup

package_name = 'dummy_executor_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='soo',
    maintainer_email='poi3824@gmail.com',
    description='GPU/Isaac Sim 없이 behavior_node·fleet_manager_node의 FSM을 검증하기 위한 더미 실행 노드',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'dummy_executor_node = dummy_executor_node.dummy_executor_node:main'
        ],
    },
)

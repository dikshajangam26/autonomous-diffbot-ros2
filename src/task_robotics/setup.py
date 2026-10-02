import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'task_robotics'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Diksha',
    maintainer_email='dikshajangam26@gmail.com',
    description='Warehouse robot: odometry, sensor fusion, object SLAM and Nav2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'wheel_tick_pub = task_robotics.wheel_tick_pub:main',
            'odom_calculator = task_robotics.odom_calculator:main',
            'sensor_check = task_robotics.sensor_check:main',
            'cloud_filter = task_robotics.cloud_filter:main',
        ],
    },
)

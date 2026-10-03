from glob import glob

from setuptools import find_packages, setup

package_name = 'odom_computation'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Diksha',
    maintainer_email='dikshajangam26@gmail.com',
    description='Differential-drive odometry from wheel encoder ticks, plus a synthetic IMU publisher',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'odom_node = odom_computation.odom_node:main',
            'wheel_tick_pub = odom_computation.wheel_tick_pub:main',
            'imu_publisher = odom_computation.imu_publisher:main',
        ],
    },
)

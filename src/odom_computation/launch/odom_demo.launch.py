"""Starts the simulated encoder publisher and the odometry node (and optionally the IMU)."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('imu', default_value='false',
                              description='also start the synthetic IMU publisher'),
        Node(package='odom_computation', executable='wheel_tick_pub', output='screen'),
        Node(package='odom_computation', executable='odom_node', output='screen'),
        Node(package='odom_computation', executable='imu_publisher', output='screen',
             condition=IfCondition(LaunchConfiguration('imu'))),
    ])

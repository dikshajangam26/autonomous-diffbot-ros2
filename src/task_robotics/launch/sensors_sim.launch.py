"""Start Gazebo, spawn the robot with IMU + LiDAR + stereo camera, publish TF, open RViz.

    ros2 launch task_robotics sensors_sim.launch.py
    ros2 launch task_robotics sensors_sim.launch.py rviz:=false gui:=false   (no windows)

Drive the robot (in a 2nd terminal):
    ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}}"

Also starts wheel_tick_pub (gazebo mode) and odom_calculator, so /wheel_ticks, /odom and
the odom -> base_link transform follow the REAL simulated wheels. Switch off with odometry:=false.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('task_robotics')
    gazebo_pkg = get_package_share_directory('gazebo_ros')

    xacro_file = os.path.join(pkg, 'urdf', 'task_robot.urdf.xacro')
    rviz_file = os.path.join(pkg, 'rviz', 'sensors.rviz')
    robot_description = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true', description='Show the Gazebo window'),
        DeclareLaunchArgument('rviz', default_value='true', description='Open RViz'),
        DeclareLaunchArgument('odometry', default_value='true',
                              description='Run wheel_tick_pub + odom_calculator'),

        # Stops Gazebo waiting for an online model database (no internet needed)
        SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),

        # Gazebo itself (also publishes /clock = simulated time)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_pkg, 'launch', 'gazebo.launch.py')),
            launch_arguments={'gui': LaunchConfiguration('gui')}.items()),

        # Reads the URDF and publishes the TF tree (base_link -> sensor frames)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description,
                         'use_sim_time': True}],
            output='screen'),

        # Puts the robot into the Gazebo world
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-topic', 'robot_description', '-entity', 'task_robot', '-z', '0.08', '-timeout', '120'],
            output='screen'),

        # "Encoders": wheel angles from Gazebo -> ticks on /wheel_ticks (50 Hz)
        Node(
            package='task_robotics',
            executable='wheel_tick_pub',
            parameters=[{'mode': 'gazebo', 'use_sim_time': True}],
            condition=IfCondition(LaunchConfiguration('odometry')),
            output='screen'),

        # Ticks -> /odom and the odom -> base_link transform
        Node(
            package='task_robotics',
            executable='odom_calculator',
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(LaunchConfiguration('odometry')),
            output='screen'),

        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_file],
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(LaunchConfiguration('rviz')),
            output='screen'),
    ])

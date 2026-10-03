"""Gazebo simulation + sensors + wheel odometry + EKF (Phase 2).

    ros2 launch task_robotics sensors_sim.launch.py
    ros2 launch task_robotics sensors_sim.launch.py gui:=false rviz:=false   (fewer windows, less CPU)
    ros2 launch task_robotics sensors_sim.launch.py ekf:=false               (wheel odometry only)
    ros2 launch task_robotics sensors_sim.launch.py perception:=false        (no stereo pipeline: lighter)
    ros2 launch task_robotics sensors_sim.launch.py detector:=true people:=true   (YOLO + walking person)
    ros2 launch task_robotics sensors_sim.launch.py slam:=true                    (LiDAR SLAM: /map + map->odom)
    ros2 launch task_robotics sensors_sim.launch.py object_slam:=true people:=true (object SLAM: semantic map + map->odom)
    (people:=true also starts worker_walker: the walking person has an invisible body the LiDAR can see)

Drive the robot (2nd terminal):
    ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}, angular: {z: 0.5}}"

What runs:
    Gazebo -> wheels/IMU/LiDAR/cameras ... -> /joint_states, /imu/data, /scan, /stereo/...
    wheel_tick_pub (gazebo mode)   /joint_states -> /wheel_ticks
    odom_calculator                /wheel_ticks  -> /odom            (raw wheel odometry)
    ekf_filter_node                /odom + /imu/data -> /odometry/filtered + TF odom -> base_link
When ekf:=true the EKF owns the TF, so odom_calculator's own TF broadcast is switched off.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
                            SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, PushRosNamespace, SetParameter
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('task_robotics')
    gazebo_pkg = get_package_share_directory('gazebo_ros')

    xacro_file = os.path.join(pkg, 'urdf', 'task_robot.urdf.xacro')
    rviz_file = os.path.join(pkg, 'rviz', 'sensors.rviz')
    ekf_file = os.path.join(pkg, 'config', 'ekf.yaml')
    world_file = os.path.join(pkg, 'worlds', 'warehouse.world')
    world_people_file = os.path.join(pkg, 'worlds', 'warehouse_people.world')
    people = LaunchConfiguration('people')
    world = PythonExpression(["'", world_people_file, "' if '", people, "' == 'true' else '", world_file, "'"])
    stereo_pkg = get_package_share_directory('stereo_image_proc')
    slam_pkg = get_package_share_directory('slam_toolbox')
    slam_params = os.path.join(pkg, 'config', 'slam_toolbox.yaml')
    slam_params_objects = os.path.join(pkg, 'config', 'slam_toolbox_objects.yaml')
    object_slam = LaunchConfiguration('object_slam')
    # object_slam:=true needs LiDAR SLAM and the detector, so it switches them on
    slam_on = IfCondition(PythonExpression(["'", LaunchConfiguration('slam'), "' == 'true' or '", object_slam, "' == 'true'"]))
    detector_on = IfCondition(PythonExpression(["('", LaunchConfiguration('detector'), "' == 'true' or '", object_slam, "' == 'true') and '", LaunchConfiguration('perception'), "' == 'true'"]))
    slam_file = PythonExpression(["'", slam_params_objects, "' if '", object_slam, "' == 'true' else '", slam_params, "'"])
    gazebo_params = os.path.join(pkg, 'config', 'gazebo_params.yaml')
    robot_description = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)

    odometry = LaunchConfiguration('odometry')
    ekf = LaunchConfiguration('ekf')
    # odom_calculator publishes the TF only when the EKF is NOT running
    odom_with_tf = IfCondition(PythonExpression(
        ["'", odometry, "' == 'true' and '", ekf, "' != 'true'"]))
    odom_without_tf = IfCondition(PythonExpression(
        ["'", odometry, "' == 'true' and '", ekf, "' == 'true'"]))

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true', description='Show the Gazebo window'),
        DeclareLaunchArgument('rviz', default_value='true', description='Open RViz'),
        DeclareLaunchArgument('odometry', default_value='true',
                              description='Run wheel_tick_pub + odom_calculator'),
        DeclareLaunchArgument('ekf', default_value='true', description='Run the EKF'),
        DeclareLaunchArgument('perception', default_value='true',
                              description='Run stereo depth + point cloud + cloud filter'),
        DeclareLaunchArgument('object_slam', default_value='false',
                              description='Semantic object SLAM: publishes map->odom from LiDAR + objects '
                                          '(switches on slam and detector)'),
        DeclareLaunchArgument('slam', default_value='false',
                              description='Run slam_toolbox (LiDAR map + map->odom transform)'),
        DeclareLaunchArgument('detector', default_value='false',
                              description='Run the YOLO object detector (needs the YOLO install)'),
        DeclareLaunchArgument('people', default_value='false',
                              description='Add a walking person to the world'),

        # Stops Gazebo waiting for an online model database (no internet needed)
        SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),

        # Gazebo itself (also publishes /clock = simulated time, here at 100 Hz)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_pkg, 'launch', 'gazebo.launch.py')),
            launch_arguments={'gui': LaunchConfiguration('gui'),
                              'world': world,
                              'params_file': gazebo_params}.items()),

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
            arguments=['-topic', 'robot_description', '-entity', 'task_robot',
                       '-z', '0.08', '-timeout', '120'],
            output='screen'),

        # "Encoders": wheel angles from Gazebo -> ticks on /wheel_ticks
        Node(
            package='task_robotics',
            executable='wheel_tick_pub',
            parameters=[{'mode': 'gazebo', 'use_sim_time': True}],
            condition=IfCondition(odometry),
            output='screen'),

        # Ticks -> /odom (two variants: with or without the TF broadcast)
        Node(
            package='task_robotics',
            executable='odom_calculator',
            name='odom_calculator',
            parameters=[{'use_sim_time': True, 'publish_tf': True}],
            condition=odom_with_tf,
            output='screen'),
        Node(
            package='task_robotics',
            executable='odom_calculator',
            name='odom_calculator',
            parameters=[{'use_sim_time': True, 'publish_tf': False}],
            condition=odom_without_tf,
            output='screen'),

        # Sensor fusion: wheel speed + IMU gyro -> /odometry/filtered and odom -> base_link
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            parameters=[ekf_file, {'use_sim_time': True}],
            condition=IfCondition(ekf),
            output='screen'),

        # ---------- Phase 3, Task 6: stereo images -> 3D point cloud ----------
        # stereo_image_proc: rectify both images -> disparity (how far each pixel shifted between
        # left and right) -> point cloud. Everything lives in the /stereo namespace, so it reads
        # /stereo/left/image_raw, /stereo/left/camera_info, /stereo/right/... and writes
        # /stereo/disparity and /stereo/points2.
        GroupAction(
            condition=IfCondition(LaunchConfiguration('perception')),
            actions=[
                PushRosNamespace('stereo'),
                SetParameter(name='use_sim_time', value=True),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(stereo_pkg, 'launch', 'stereo_image_proc.launch.py')),
                    # left/right pictures come from two separate simulated cameras, so allow a
                    # tiny time difference when pairing them
                    launch_arguments={'approximate_sync': 'True'}.items()),
            ]),

        # Cleans the cloud: pass-through (floor / ceiling / far) + voxel grid -> /stereo/points_filtered
        Node(
            package='task_robotics',
            executable='cloud_filter',
            parameters=[{'use_sim_time': True,
                         'voxel_size': 0.05,
                         'z_min': 0.02, 'z_max': 1.5,
                         'x_min': 0.2, 'x_max': 6.0,
                         'y_min': -4.0, 'y_max': 4.0}],
            condition=IfCondition(LaunchConfiguration('perception')),
            output='screen'),

        # ---------- Phase 3, Task 7b: LiDAR SLAM -> /map and the map -> odom transform ----------
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(slam_pkg, 'launch', 'online_async_launch.py')),
            launch_arguments={'slam_params_file': slam_file,
                              'use_sim_time': 'true'}.items(),
            condition=slam_on),

        # ---------- Phase 3, Task 7a: YOLO finds objects, the stereo cloud gives their 3D position
        Node(
            package='task_robotics',
            executable='object_detector',
            parameters=[{'use_sim_time': True}],
            condition=detector_on,
            output='screen'),

        # ---------- Phase 3, Task 7c: semantic object map (factor graph) + map -> odom ----------
        Node(
            package='task_robotics',
            executable='object_slam',
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(object_slam),
            output='screen'),

        # ---------- Phase 4, Task 9: a collision body follows the walking person (LiDAR can see it) ----------
        Node(
            package='task_robotics',
            executable='worker_walker',
            parameters=[{'use_sim_time': True, 'world_file': world_people_file}],
            condition=IfCondition(people),
            output='screen'),

        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_file],
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(LaunchConfiguration('rviz')),
            output='screen'),
    ])

"""Nav2 for the warehouse robot (Phase 4, Task 8).

Terminal 1 (simulation, SLAM and the semantic object map):
    ros2 launch task_robotics sensors_sim.launch.py object_slam:=true people:=true gui:=false rviz:=false
Terminal 2 (this file):
    ros2 launch task_robotics nav2.launch.py

It starts
  * Nav2 (planner, controller, smoother, behaviours, bt_navigator, waypoint follower, velocity smoother)
    using config/nav2_params.yaml and behavior_trees/warehouse_nav.xml
  * semantic_costmap: object map + people -> keep-out points that both costmaps read

Nav2's own launch file needs two settings that depend on the installed Nav2 version (the list of
behaviour-tree node libraries) and on where this package lives (the path of our behaviour tree).
They are filled in here, into a temporary copy of nav2_params.yaml, so the file you edit stays clean.
"""
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _prepare_params(pkg_share, nav2_share):
    with open(os.path.join(pkg_share, 'config', 'nav2_params.yaml')) as f:
        params = yaml.safe_load(f)
    bt = params['bt_navigator']['ros__parameters']
    bt['default_nav_to_pose_bt_xml'] = os.path.join(pkg_share, 'behavior_trees', 'warehouse_nav.xml')
    # copy the list of BT node plugins from this Nav2 version's own example file
    try:
        with open(os.path.join(nav2_share, 'params', 'nav2_params.yaml')) as f:
            ref = yaml.safe_load(f)
        libs = ref['bt_navigator']['ros__parameters'].get('plugin_lib_names')
        if libs:
            bt['plugin_lib_names'] = libs
    except (OSError, KeyError, TypeError):
        pass                                    # fall back to Nav2's built-in default list
    fd, path = tempfile.mkstemp(prefix='nav2_params_', suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.safe_dump(params, f)
    return path


def generate_launch_description():
    pkg = get_package_share_directory('task_robotics')
    nav2 = get_package_share_directory('nav2_bringup')
    params_file = _prepare_params(pkg, nav2)

    return LaunchDescription([
        DeclareLaunchArgument('map_file', default_value='/ws/maps/semantic_map.yaml',
                              description='Saved semantic map used until object_slam publishes'),
        DeclareLaunchArgument('person_radius', default_value='0.35',
                              description='Keep-out radius around a detected person (m)'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav2, 'launch', 'navigation_launch.py')),
            launch_arguments={'params_file': params_file,
                              'use_sim_time': 'true',
                              'autostart': 'true',
                              'use_composition': 'False'}.items()),

        Node(
            package='task_robotics',
            executable='semantic_costmap',
            parameters=[{'use_sim_time': True,
                         'map_file': LaunchConfiguration('map_file'),
                         'person_radius': ParameterValue(LaunchConfiguration('person_radius'), value_type=float)}],
            output='screen'),
    ])

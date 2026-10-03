# odom_computation (Task 2)

Differential-drive odometry from simulated wheel-encoder ticks, exactly as the task brief specifies.

| Item | Value |
|---|---|
| Subscribes | `/left_wheel_ticks`, `/right_wheel_ticks` (`std_msgs/Int32`, cumulative) |
| Publishes | `/odom` (`nav_msgs/Odometry`) at 10 Hz, orientation as a quaternion, pose and twist filled |
| Parameters | `wheel_radius` 0.065 m, `wheel_base` 0.21 m, `ticks_per_revolution` 512 |
| Bonus | `imu_publisher` publishes noise-free `sensor_msgs/Imu` on `/imu/data` |

## Run

```bash
cd /ws && colcon build --symlink-install --packages-select odom_computation && source install/setup.bash
ros2 launch odom_computation odom_demo.launch.py          # simulated ticks + odometry node
ros2 launch odom_computation odom_demo.launch.py imu:=true   # plus the synthetic IMU
ros2 topic echo /odom --once
```

Expected: `x` increases (about 0.2 m/s), `y` stays about 0, orientation stays near (0, 0, 0, 1).
Individual nodes: `ros2 run odom_computation odom_node`, `wheel_tick_pub`, `imu_publisher`
(`--ros-args -p mode:=stationary` for a still IMU).

## Notes

* `odom_node` does not broadcast `odom -> base_link` unless `-p publish_tf:=true`.
* Do not run it together with the main stack's `odom_calculator` or the Gazebo IMU: they publish the same topics.
* Unit tests for the maths (no ROS needed): `python3 -m pytest src/odom_computation/test`

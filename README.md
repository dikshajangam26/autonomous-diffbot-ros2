# Warehouse Robot – Task_Robotics (ROS 2 Humble)

Autonomous warehouse robot: odometry → sensor fusion → object-based SLAM → Nav2.

## Progress
- [x] Phase 1.1 Workspace + `task_robotics` package
- [x] Phase 1.2 `wheel_tick_pub` – `/wheel_ticks` at 50 Hz (sim or serial)
- [x] Phase 1.3 `odom_calculator` – `/odom` + `odom -> base_link` TF
- [ ] Phase 2 Sensor fusion (IMU, LiDAR, stereo camera, EKF)
- [ ] Phase 3 Perception and object-based SLAM
- [ ] Phase 4 Nav2 navigation + Gazebo validation

## Run with Docker (recommended)
```bash
docker compose build              # first time only
docker compose up -d
docker compose exec ros bash      # terminal inside the container
cd /ws && colcon build --symlink-install && source install/setup.bash
ros2 run task_robotics wheel_tick_pub     # terminal 1
ros2 run task_robotics odom_calculator    # terminal 2 (open with another `docker compose exec ros bash`)
```

## Run without Docker
Needs Ubuntu 22.04 + ROS 2 Humble:
```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

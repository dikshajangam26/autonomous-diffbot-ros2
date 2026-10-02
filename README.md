# Autonomous Differential-Drive Warehouse Robot (ROS 2 Humble)

An end-to-end autonomy stack for a small differential-drive warehouse robot, built and validated in Gazebo:
**wheel odometry → EKF sensor fusion → stereo/LiDAR perception → object-based SLAM → Nav2 navigation with a semantic costmap → automatic validation against ground truth.**

Everything runs in Docker on ROS 2 Humble, so there is nothing to install on the host except Docker.

---

## What it does

| Phase | Task | What was built |
|---|---|---|
| 1 | Odometry | `wheel_tick_pub` (50 Hz `/wheel_ticks`, simulated or serial) and `odom_calculator` (`/odom` + `odom → base_link` TF) |
| 2 | Sensors and fusion | Robot URDF with 2D LiDAR, IMU and stereo camera; `robot_localization` EKF fusing wheel speed (vx) with gyro yaw rate; `ekf_compare` tool |
| 3 | Perception | Stereo point-cloud pipeline with `cloud_filter`; YOLOv8-World `object_detector` (pallets, racks, boxes, people) giving 3D detections |
| 3 | Object-based SLAM | `slam_toolbox` for the occupancy map + `object_slam`: a GTSAM **iSAM2 factor graph** of robot keyframes and object landmarks that publishes `map → odom` and a persistent semantic map |
| 4 | Nav2 configuration | Global and local costmaps that consume the semantic map, narrow-aisle inflation and footprint, custom behaviour tree |
| 4 | Planning and validation | A* vs Dijkstra global planner, DWB vs MPPI local planner, a walking-person dynamic obstacle, and `validate_pipeline.py` which scores the whole pipeline against the Gazebo world file |

## Architecture

```
 Gazebo (headless)  ──►  /scan  /imu/data  /stereo/*  /odom  /clock
        │
        ├─ ekf_filter_node ───────────────► odom → base_link  (smooth odometry)
        ├─ cloud_filter ──► object_detector (YOLOv8-World) ──► /objects/detections
        ├─ slam_toolbox ──► /map, /pose  (LiDAR scan matching)
        └─ object_slam (GTSAM iSAM2) ◄── /pose + detections + odometry
                 ├─► TF map → odom
                 └─► /semantic_map/landmarks  (Detection3DArray, map frame)
                              │
                       semantic_costmap ──► class-sized keep-out PointCloud2
                              │
 Nav2:  planner_server (NavFn A* / Dijkstra) · controller_server (DWB / MPPI)
        smoother · behavior_server · bt_navigator · velocity_smoother
        costmaps = static_layer (/map) + obstacle_layer (/scan + semantic cloud) + inflation_layer
```

### Semantic costmap

`semantic_costmap` turns each landmark into a keep-out disc sized by class (rack 0.30 m, pallet 0.25 m, box 0.20 m, person 0.35 m with a 1.5 s time-to-live). A **ghost-landmark filter** drops detections that have no mapped LiDAR surface within 0.20 m (pallets are exempt), so false YOLO detections cannot block an aisle.

### Narrow aisles

The 0.65 m aisle is tight for a 0.26 × 0.24 m footprint. The configuration uses an explicit footprint polygon, an inflation radius of 0.45 m (global) and 0.35 m (local), and `tools/check_aisles.py` verifies offline that each aisle is passable.

## Repository layout

```
docker/Dockerfile, docker-compose.yml     ROS 2 Humble + Gazebo + Nav2 + slam_toolbox + torch/ultralytics/gtsam
src/task_robotics/
  task_robotics/    wheel_tick_pub, odom_calculator, cloud_filter, object_detector, object_slam,
                    slam_ops, semantic_ops, semantic_costmap, validation_ops, worker_walker
  launch/           sensors_sim.launch.py (sim + perception + SLAM), nav2.launch.py (Nav2)
  config/           ekf.yaml, slam_toolbox*.yaml, nav2_params.yaml
  behavior_trees/   warehouse_nav.xml (A*+DWB), warehouse_nav_dijkstra.xml, warehouse_nav_mppi.xml
  worlds/           warehouse.world, warehouse_people.world
  urdf/, rviz/
tools/              validate_pipeline.py, nav_goal_test.py, slam_compare.py, ekf_compare.py,
                    check_aisles.py, test_*.py (offline unit tests)
maps/               saved semantic map
```

## Quick start

Use the Docker setup (recommended). On Windows, run it inside WSL2.

```bash
git clone https://github.com/dikshajangam26/autonomous-diffbot-ros2.git
cd autonomous-diffbot-ros2
docker compose build            # first time only; installs torch, ultralytics, gtsam
docker compose up -d
docker compose exec ros bash    # a terminal inside the container (project is mounted at /ws)

cd /ws && colcon build --symlink-install && source install/setup.bash
```

Open extra terminals with the same `docker compose exec ros bash` and `source /ws/install/setup.bash`.

> **WSL2 note:** after unzipping files on Windows, delete the `*:Zone.Identifier` files:
> `find . -name '*:Zone.Identifier' -type f -delete`. New data files need a rebuild with `--symlink-install`.

### 1. Simulation, perception and object SLAM (terminal 1)

```bash
ros2 launch task_robotics sensors_sim.launch.py object_slam:=true people:=true gui:=false rviz:=false
```

Useful switches: `ekf:=false` (wheel odometry only), `slam:=true` (LiDAR SLAM only), `detector:=true`, `perception:=false` (lighter), `gui:=false rviz:=false` (less CPU).

### 2. Nav2 (terminal 2)

```bash
ros2 launch task_robotics nav2.launch.py      # wait for "Managed nodes are active"
```

Send a single goal and see what really happened:

```bash
python3 tools/nav_goal_test.py 3.575 1.6      # x y in the map frame
```

### 3. Full validation (terminal 3)

```bash
python3 tools/validate_pipeline.py                                   # tour + missions + dynamic, all planners
python3 tools/validate_pipeline.py --skip-tour --configs astar_mppi --leg-timeout 40 --no-diagnose
```

Flags: `--configs astar_dwb,dijkstra_dwb,astar_mppi`, `--skip-tour`, `--no-dynamic`, `--leg-timeout` (simulated seconds, default 90), `--no-diagnose`, `--world`, `--out /ws/reports`.
A full run takes roughly 12–15 minutes of wall time per planner combination, because the simulation runs close to real time.

Offline unit tests (no simulator needed):

```bash
python3 tools/test_semantic_ops.py
python3 tools/test_validation_ops.py
```

## Validation method

The Gazebo world file is the ground truth. `validate_pipeline.py`:

1. **Maps the warehouse** with a tour of 11 waypoints, then scores the SLAM map against the world: wall and rack recall, precision (tolerance 0.10 m).
2. **Scores the semantic map**: landmark distance to the true object footprint and class accuracy.
3. **Runs a 7-waypoint mission** and records, per leg, the status, time, path length, final error, planning time and closest LiDAR return.
4. **Runs dynamic-obstacle legs** with an invisible-to-camera, LiDAR-visible walking person (a collision cylinder driven by `worker_walker`, with true position from `/worker/ground_truth`) and records the minimum separation.
5. Prints a PASS/FAIL table against fixed targets and saves JSON and Markdown to `reports/`.

| Target | Threshold |
|---|---|
| Wall recall / rack recall / precision | ≥ 0.85 / ≥ 0.40 / ≥ 0.85 |
| Landmark inside correct footprint | ≥ 0.75 |
| Mission success / final goal error | ≥ 0.90 / ≤ 15 cm |
| Min LiDAR clearance | ≥ 0.16 m |
| Dynamic success / person separation | ≥ 0.75 / ≥ 0.45 m |

## Results

Measured in simulation (astar_dwb configuration):

- **Map accuracy:** walls 95%, racks 82%, boxes 82.5%, precision 99.8%.
- **Semantic map:** 7 landmarks, all inside the correct footprint with the correct class, covering 6 of 10 world objects.
- **Global planners:** A* and Dijkstra give equal paths (3.34 m vs 3.33 m) with 38 ms vs 40 ms planning time on the test leg.
- **Navigation:** 6 of 7 mission waypoints reached; successful legs end within 3–5 cm of the goal, and a tight-aisle goal reached with a closest LiDAR return of 25.7 cm.

The DWB-vs-MPPI comparison reports are written to `reports/validation_*.md` and `.json` by each run.

## Known limitations

These are documented honestly rather than tuned away:

- **Aisle entrances:** a few legs end with Nav2 "Failed to make progress" at the tight east aisle entrance and when exiting the aisle behind a rack. The robot footprint sits inside the inscribed zone with only about 7 cm of slack, so the controller stalls.
- **Dynamic person:** minimum separation was about 0.38 m against a 0.45 m target. DWB has no motion prediction and the person walks at 0.6 m/s against a 0.25 m/s robot.
- **Detection coverage:** 6 of 10 world objects are found; the rest are not seen by the detector from the tour poses.
- **TEB:** TEB Local Planner has no Humble apt package and was not built; MPPI is used as the second local planner instead.
- **Simulation only:** no real hardware testing; the Gazebo actor has no collision body, so a separate invisible collision cylinder stands in for the person.

### Next steps

Nav2 collision monitor and speed limiting near people, a pre-aisle approach waypoint, a TEB build from source, and a wider detector tour.

## Tech stack

ROS 2 Humble · Gazebo Classic · Nav2 (NavFn, DWB, MPPI) · slam_toolbox · robot_localization · GTSAM (iSAM2) · YOLOv8-World / PyTorch · Docker · GitHub Actions (colcon build on `ros:humble`)

## Author

Diksha — MSc Robotics, University of Birmingham · [github.com/dikshajangam26](https://github.com/dikshajangam26)

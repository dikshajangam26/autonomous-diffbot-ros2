#!/usr/bin/env python3
"""
nav_goal_test.py - sends ONE navigation goal to Nav2 and reports what really happened.

    python3 tools/nav_goal_test.py 3.575 1.6            # goal x y in the map frame (default: tight east aisle)
    python3 tools/nav_goal_test.py 3.575 1.6 --timeout 120

Every 2 s it prints where the robot is (SLAM map frame, and Gazebo's true position), the distance still
to go, and the speed Nav2 is commanding. At the end it prints SUCCEEDED / FAILED / TIMEOUT with the final
error in the map frame and in the real world, and the closest the robot ever got to a wall or rack
(estimated from the LiDAR). Needs the sim and Nav2 running; the robot must not be driven by anything else.
"""
import argparse
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener
from action_msgs.msg import GoalStatus

STATUS = {GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED', GoalStatus.STATUS_ABORTED: 'ABORTED (Nav2 gave up)',
          GoalStatus.STATUS_CANCELED: 'CANCELED'}


class Tester(Node):
    def __init__(self, gx, gy, gyaw):
        super().__init__('nav_goal_test', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.gx, self.gy, self.gyaw = gx, gy, gyaw
        self.tf = Buffer()
        self.tl = TransformListener(self.tf, self)
        self.truth = None
        self.cmd = (0.0, 0.0)
        self.min_range = 99.0
        self.recoveries = 0
        self.create_subscription(Odometry, '/ground_truth/odom', self.on_truth, 10)
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, 10)
        self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    def on_truth(self, m):
        p = m.pose.pose.position
        self.truth = (p.x, p.y)

    def on_cmd(self, m):
        self.cmd = (m.linear.x, m.angular.z)

    def on_scan(self, m):
        # ignore the robot's own body (< 0.13 m); the closest remaining return is the clearance
        good = [r for r in m.ranges if 0.13 < r < m.range_max and math.isfinite(r)]
        if good:
            self.min_range = min(self.min_range, min(good))

    def map_pose(self):
        try:
            t = self.tf.lookup_transform('map', 'base_link', Time()).transform.translation
            return t.x, t.y
        except TransformException:
            return None

    def line(self, tag=''):
        mp = self.map_pose()
        s = f'{tag}map='
        s += f'({mp[0]:6.2f},{mp[1]:6.2f})' if mp else '(  n/a  )'
        s += f'  truth=({self.truth[0]:6.2f},{self.truth[1]:6.2f})' if self.truth else '  truth=n/a'
        if mp:
            s += f'  to_go={math.hypot(self.gx - mp[0], self.gy - mp[1]):5.2f} m'
        s += f'  cmd v={self.cmd[0]:5.2f} w={self.cmd[1]:5.2f}  closest_lidar={self.min_range:4.2f} m'
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('x', type=float, nargs='?', default=3.575)
    ap.add_argument('y', type=float, nargs='?', default=1.6)
    ap.add_argument('--yaw', type=float, default=0.0)
    ap.add_argument('--timeout', type=float, default=90.0, help='seconds of SIMULATED time')
    a = ap.parse_args()

    rclpy.init()
    n = Tester(a.x, a.y, a.yaw)
    print('waiting for the Nav2 action server ...')
    if not n.client.wait_for_server(timeout_sec=20.0):
        print('FAIL: navigate_to_pose is not available - is nav2.launch.py running and "active"?')
        return 1
    for _ in range(40):                       # let TF / clock arrive
        rclpy.spin_once(n, timeout_sec=0.1)

    goal = NavigateToPose.Goal()
    goal.pose = PoseStamped()
    goal.pose.header.frame_id = 'map'
    goal.pose.pose.position.x, goal.pose.pose.position.y = a.x, a.y
    goal.pose.pose.orientation.z = math.sin(a.yaw / 2)
    goal.pose.pose.orientation.w = math.cos(a.yaw / 2)

    n.min_range = 99.0
    print(f'goal: ({a.x}, {a.y}) in the map frame')
    print(n.line('start  '))
    fut = n.client.send_goal_async(goal)
    rclpy.spin_until_future_complete(n, fut, timeout_sec=10.0)
    handle = fut.result()
    if handle is None or not handle.accepted:
        print('FAIL: Nav2 rejected the goal')
        return 1

    res_fut = handle.get_result_async()
    t0 = n.get_clock().now().nanoseconds * 1e-9
    last = t0
    status = None
    while rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.2)
        now = n.get_clock().now().nanoseconds * 1e-9
        if now - last >= 2.0:
            last = now
            print(n.line(f't={now - t0:5.1f}  '))
        if res_fut.done():
            status = res_fut.result().status
            break
        if now - t0 > a.timeout:
            handle.cancel_goal_async()
            status = 'TIMEOUT'
            break

    mp = n.map_pose()
    print('\n==== RESULT ====')
    print(STATUS.get(status, status))
    if mp:
        print(f'final map-frame error : {math.hypot(a.x - mp[0], a.y - mp[1]) * 100:5.1f} cm')
    if n.truth:
        print(f'final real position   : ({n.truth[0]:.2f}, {n.truth[1]:.2f})  '
              f'-> {math.hypot(a.x - n.truth[0], a.y - n.truth[1]) * 100:5.1f} cm from the goal')
    print(f'closest LiDAR return  : {n.min_range * 100:5.1f} cm (robot half-width is 12 cm)')
    print(f'took                  : {n.get_clock().now().nanoseconds * 1e-9 - t0:5.1f} s of simulated time')
    rclpy.shutdown()
    return 0 if status == GoalStatus.STATUS_SUCCEEDED else 1


if __name__ == '__main__':
    sys.exit(main())

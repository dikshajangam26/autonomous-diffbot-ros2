#!/usr/bin/env python3
"""
worker_walker.py - Task 9. Moves the invisible collision body of the walking person (world model
"worker_body") along the SAME scripted route as the animated actor, so that the LiDAR sees a real moving
obstacle and the robot can actually touch it.

Reads  : /worker/ground_truth   where the body really is (Gazebo p3d plugin)
Writes : /worker/cmd_vel        velocity for the planar-move plugin of the body
How    : the actor's route comes from the world file (waypoints with times, looping). Each cycle the
         target is where the actor should be right now; the command is the route's own velocity (feed-forward)
         plus a correction proportional to the position error.
"""
import math
import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from task_robotics import validation_ops as vo


class WorkerWalker(Node):
    def __init__(self):
        super().__init__('worker_walker')
        default = os.path.join(get_package_share_directory('task_robotics'), 'worlds', 'warehouse_people.world')
        self.declare_parameter('world_file', default)
        self.declare_parameter('gain', 3.0)
        self.declare_parameter('max_speed', 1.0)
        self.route = vo.parse_actor(self.get_parameter('world_file').value)
        if not self.route:
            self.get_logger().error('no <actor> route found in the world file')
        self.gain = self.get_parameter('gain').value
        self.max_speed = self.get_parameter('max_speed').value
        self.pose = None
        self.create_subscription(Odometry, '/worker/ground_truth', self.on_pose, 10)
        self.pub = self.create_publisher(Twist, '/worker/cmd_vel', 10)
        self.create_timer(0.05, self.step)
        self.get_logger().info(f'following the actor route ({len(self.route)} waypoints, '
                               f'period {self.route[-1][0] if self.route else 0:.2f} s)')

    def on_pose(self, m):
        q = m.pose.pose.orientation
        self.pose = (m.pose.pose.position.x, m.pose.pose.position.y,
                     vo.yaw_from_quaternion(q.x, q.y, q.z, q.w))

    def step(self):
        if self.pose is None or not self.route:
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        tx, ty = vo.person_position(self.route, t)
        nx, ny = vo.person_position(self.route, t + 0.05)
        vx_w = (nx - tx) / 0.05 + self.gain * (tx - self.pose[0])
        vy_w = (ny - ty) / 0.05 + self.gain * (ty - self.pose[1])
        speed = math.hypot(vx_w, vy_w)
        if speed > self.max_speed:
            vx_w, vy_w = vx_w * self.max_speed / speed, vy_w * self.max_speed / speed
        c, s = math.cos(self.pose[2]), math.sin(self.pose[2])          # world -> body frame
        cmd = Twist()
        cmd.linear.x = c * vx_w + s * vy_w
        cmd.linear.y = -s * vx_w + c * vy_w
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = WorkerWalker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
odom_calculator.py  -  Phase 1, Task 3

Dead reckoning for a 2-wheel differential-drive robot.

Listens to : /wheel_ticks   (std_msgs/Int32MultiArray, data = [left, right])
Publishes  : /odom          (nav_msgs/Odometry)
Broadcasts : TF  odom -> base_link

Idea in plain words
-------------------
1. Ticks -> distance:   each wheel moved (new ticks - old ticks) / ticks_per_rev
                        turns, and one turn = 2 * pi * wheel_radius meters.
2. Two wheels -> robot: the robot moved forward by the AVERAGE of the two wheel
                        distances, and rotated by their DIFFERENCE divided by
                        the distance between the wheels (wheel_base).
3. Add up the little moves (integration) to get the pose (x, y, theta).
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32MultiArray
from tf2_ros import TransformBroadcaster


def yaw_to_quaternion(yaw):
    """Heading angle (rotation about Z) -> quaternion (x, y, z, w)."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class OdomCalculator(Node):

    def __init__(self):
        super().__init__('odom_calculator')

        # ---------- Parameters ----------
        self.declare_parameter('wheel_radius', 0.065)        # meters
        self.declare_parameter('wheel_base', 0.21)           # meters between wheels
        self.declare_parameter('ticks_per_revolution', 512)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('publish_tf', True)

        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.wheel_base = self.get_parameter('wheel_base').value
        self.ticks_per_rev = self.get_parameter('ticks_per_revolution').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.publish_tf = self.get_parameter('publish_tf').value

        # Meters of wheel travel for ONE tick
        self.meters_per_tick = (2.0 * math.pi * self.wheel_radius) / self.ticks_per_rev

        # ---------- Robot state ----------
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.prev_left = None     # previous tick counts (None until first message)
        self.prev_right = None
        self.prev_time = None

        # ---------- ROS interfaces ----------
        self.sub = self.create_subscription(
            Int32MultiArray, '/wheel_ticks', self.ticks_callback, 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(
            f'odom_calculator started: {self.odom_frame} -> {self.base_frame}, '
            f'{self.meters_per_tick * 1000:.3f} mm per tick')

    # ------------------------------------------------------------------
    def ticks_callback(self, msg):
        if len(msg.data) < 2:
            self.get_logger().warn('/wheel_ticks needs 2 values [left, right]')
            return

        left, right = msg.data[0], msg.data[1]
        now = self.get_clock().now()

        # The very first message only gives us a starting point
        if self.prev_time is None:
            self.prev_left, self.prev_right, self.prev_time = left, right, now
            return

        dt = (now - self.prev_time).nanoseconds * 1e-9
        if dt <= 0.0:
            return

        # 1) Ticks -> distance each wheel travelled since last message
        d_left = (left - self.prev_left) * self.meters_per_tick
        d_right = (right - self.prev_right) * self.meters_per_tick
        self.prev_left, self.prev_right, self.prev_time = left, right, now

        # 2) Two wheels -> robot motion
        d_center = (d_left + d_right) / 2.0               # forward distance
        d_theta = (d_right - d_left) / self.wheel_base    # rotation (rad)

        # 3) Integrate. Using the MIDDLE heading of the step is more accurate
        #    than the heading at the start of the step.
        self.x += d_center * math.cos(self.theta + d_theta / 2.0)
        self.y += d_center * math.sin(self.theta + d_theta / 2.0)
        self.theta = math.atan2(math.sin(self.theta + d_theta),
                                math.cos(self.theta + d_theta))  # keep in -pi..pi

        # Velocities
        v = d_center / dt        # linear  (m/s)
        w = d_theta / dt         # angular (rad/s)

        self.publish_odometry(now, v, w)

    # ------------------------------------------------------------------
    def publish_odometry(self, now, v, w):
        stamp = now.to_msg()
        qx, qy, qz, qw = yaw_to_quaternion(self.theta)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        # Velocities are expressed in the robot's own frame (base_link)
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w

        # Covariance = how much we doubt each number (6x6 matrix, row-major).
        # Order: x, y, z, roll, pitch, yaw. Big number = "don't trust this".
        # These are starting guesses; we tune them in Phase 2 (EKF).
        odom.pose.covariance[0] = 0.001      # x
        odom.pose.covariance[7] = 0.001      # y
        odom.pose.covariance[14] = 1e6       # z      (robot is flat)
        odom.pose.covariance[21] = 1e6       # roll
        odom.pose.covariance[28] = 1e6       # pitch
        odom.pose.covariance[35] = 0.01      # yaw
        odom.twist.covariance[0] = 0.001     # vx
        odom.twist.covariance[7] = 1e6       # vy  (diff drive can't slide sideways)
        odom.twist.covariance[14] = 1e6
        odom.twist.covariance[21] = 1e6
        odom.twist.covariance[28] = 1e6
        odom.twist.covariance[35] = 0.01     # yaw rate

        self.odom_pub.publish(odom)

        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.translation.z = 0.0
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomCalculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
    
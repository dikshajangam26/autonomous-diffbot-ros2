#!/usr/bin/env python3
"""
odom_node.py - Task 2: differential-drive odometry from simulated wheel encoder ticks.

Subscribes : /left_wheel_ticks   std_msgs/Int32   (cumulative encoder count)
             /right_wheel_ticks  std_msgs/Int32
Publishes  : /odom               nav_msgs/Odometry at 10 Hz

Pose (x, y, theta) and velocity come from differential-drive kinematics. Heading is converted
to a quaternion for the orientation field. The node starts publishing /odom immediately (zeros
until the first ticks arrive), so `ros2 topic echo /odom --once` always returns a message.

Parameters (defaults are the values from the task brief):
  wheel_radius 0.065 m, wheel_base 0.21 m, ticks_per_revolution 512, publish_rate 10 Hz
  odom_frame 'odom', base_frame 'base_link'
  publish_tf false   (set true to also broadcast odom -> base_link)
"""
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Int32
from tf2_ros import TransformBroadcaster

from odom_computation.odom_math import integrate, meters_per_tick, yaw_to_quaternion


class OdomNode(Node):

    def __init__(self):
        super().__init__('odom_computation')

        self.declare_parameter('wheel_radius', 0.065)
        self.declare_parameter('wheel_base', 0.21)
        self.declare_parameter('ticks_per_revolution', 512)
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('publish_tf', False)

        self.wheel_base = self.get_parameter('wheel_base').value
        self.mpt = meters_per_tick(self.get_parameter('wheel_radius').value,
                                   self.get_parameter('ticks_per_revolution').value)
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.publish_tf = self.get_parameter('publish_tf').value
        period = 1.0 / self.get_parameter('publish_rate').value

        # latest cumulative ticks and the counts used at the previous update
        self.left = None
        self.right = None
        self.prev_left = None
        self.prev_right = None
        self.x = self.y = self.theta = 0.0
        self.v = self.w = 0.0

        self.create_subscription(Int32, '/left_wheel_ticks', self.on_left, 10)
        self.create_subscription(Int32, '/right_wheel_ticks', self.on_right, 10)
        self.pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None
        self.period = period
        self.create_timer(period, self.update)

        self.get_logger().info(
            f'odom_computation started: {self.mpt * 1000:.3f} mm per tick, '
            f'publishing /odom at {1.0 / period:.0f} Hz')

    def on_left(self, msg):
        self.left = msg.data

    def on_right(self, msg):
        self.right = msg.data

    def update(self):
        if self.left is not None and self.right is not None:
            if self.prev_left is None:            # first pair of counts = starting point
                self.prev_left, self.prev_right = self.left, self.right
            d_left = (self.left - self.prev_left) * self.mpt
            d_right = (self.right - self.prev_right) * self.mpt
            self.prev_left, self.prev_right = self.left, self.right
            self.x, self.y, self.theta, d_center, d_theta = integrate(
                self.x, self.y, self.theta, d_left, d_right, self.wheel_base)
            self.v = d_center / self.period
            self.w = d_theta / self.period
        self.publish()

    def publish(self):
        stamp = self.get_clock().now().to_msg()
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
        odom.twist.twist.linear.x = self.v           # in the robot frame
        odom.twist.twist.angular.z = self.w

        # 6x6 covariances (x, y, z, roll, pitch, yaw); large = "not measured"
        pc = [0.0] * 36
        tc = [0.0] * 36
        for i, val in enumerate((0.01, 0.01, 1e6, 1e6, 1e6, 0.05)):
            pc[i * 7] = val
        for i, val in enumerate((0.01, 1e6, 1e6, 1e6, 1e6, 0.05)):
            tc[i * 7] = val
        odom.pose.covariance = pc
        odom.twist.covariance = tc
        self.pub.publish(odom)

        if self.tf_broadcaster is not None:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

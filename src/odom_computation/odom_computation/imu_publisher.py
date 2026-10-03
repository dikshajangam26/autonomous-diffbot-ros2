#!/usr/bin/env python3
"""
imu_publisher.py - Task 2 bonus: simulated, noise-free IMU data on /imu/data (sensor_msgs/Imu).

Modes (parameter 'mode'):
  stationary : robot at rest. angular velocity 0, linear acceleration = gravity only (0, 0, 9.81).
  circle     : robot driving a circle at 'speed' m/s and 'yaw_rate' rad/s. The gyro reports the yaw
               rate, the accelerometer reports the centripetal acceleration (speed * yaw_rate) on y plus
               gravity on z, and the orientation yaw is integrated from the yaw rate.

Other parameters: publish_rate (Hz, default 50), frame_id (default 'imu_link'), speed, yaw_rate.
In the full Gazebo stack the IMU comes from the simulator plugin, so do not run both at once.
"""
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

from odom_computation.odom_math import wrap_angle, yaw_to_quaternion

GRAVITY = 9.81


class ImuPublisher(Node):

    def __init__(self):
        super().__init__('imu_publisher')
        self.declare_parameter('mode', 'circle')
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('speed', 0.2)
        self.declare_parameter('yaw_rate', 0.2)

        self.mode = self.get_parameter('mode').value
        self.frame_id = self.get_parameter('frame_id').value
        self.speed = self.get_parameter('speed').value
        self.yaw_rate = self.get_parameter('yaw_rate').value if self.mode == 'circle' else 0.0
        self.dt = 1.0 / self.get_parameter('publish_rate').value
        self.yaw = 0.0

        self.pub = self.create_publisher(Imu, '/imu/data', 10)
        self.create_timer(self.dt, self.tick)
        self.get_logger().info(f'imu_publisher started in "{self.mode}" mode on /imu/data')

    def tick(self):
        self.yaw = wrap_angle(self.yaw + self.yaw_rate * self.dt)
        qx, qy, qz, qw = yaw_to_quaternion(self.yaw)

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.orientation.x, msg.orientation.y = qx, qy
        msg.orientation.z, msg.orientation.w = qz, qw
        msg.angular_velocity.z = self.yaw_rate
        msg.linear_acceleration.y = self.speed * self.yaw_rate   # centripetal, 0 when stationary
        msg.linear_acceleration.z = GRAVITY

        # small fixed covariances (noise-free data, so these are nominal)
        for i in (0, 4, 8):
            msg.orientation_covariance[i] = 1e-4
            msg.angular_velocity_covariance[i] = 1e-4
            msg.linear_acceleration_covariance[i] = 1e-3
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImuPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

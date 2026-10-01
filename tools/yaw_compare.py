#!/usr/bin/env python3
"""
yaw_compare.py - prints the robot's heading and turn rate from the wheels (/odom) and from
the IMU (/imu/data) side by side, once per second. Both headings start at 0 deg when this
tool starts, so the two columns should move together if wheels and IMU agree.

Also shows every stage of the wheel chain (joint angles -> ticks -> odom) to find where it stalls.

Run (inside the container, simulation already running):
    python3 /ws/tools/yaw_compare.py --ros-args -p use_sim_time:=true

Then drive:   ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist "{angular: {z: 0.5}}"
"""
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Int32MultiArray


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Unwrapper:
    """Turns angles that jump at +-180 deg into one smooth running angle, starting at 0."""

    def __init__(self):
        self.prev = None
        self.total = 0.0

    def update(self, raw):
        if self.prev is not None:
            d = math.atan2(math.sin(raw - self.prev), math.cos(raw - self.prev))
            self.total += d
        self.prev = raw
        return self.total


class YawCompare(Node):

    def __init__(self):
        super().__init__('yaw_compare')
        self.odom_u, self.imu_u = Unwrapper(), Unwrapper()
        self.odom_yaw = self.imu_yaw = None
        self.odom_w = self.imu_w = None
        self.ticks = (None, None)
        self.joints = (None, None)
        self.n = {'odom': 0, 'imu': 0, 'ticks': 0, 'joints': 0}
        self.create_subscription(Int32MultiArray, '/wheel_ticks', self.on_ticks, 10)
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)
        self.create_subscription(Imu, '/imu/data', self.on_imu, qos_profile_sensor_data)
        self.create_timer(1.0, self.report)
        self.get_logger().info('yaw_compare running (headings start at 0 deg now)')

    def on_ticks(self, msg):
        self.n['ticks'] += 1
        self.ticks = (msg.data[0], msg.data[1])

    def on_joints(self, msg):
        self.n['joints'] += 1
        p = dict(zip(msg.name, msg.position))
        self.joints = (p.get('left_wheel_joint'), p.get('right_wheel_joint'))

    def on_odom(self, msg):
        self.n['odom'] += 1
        self.odom_yaw = self.odom_u.update(yaw_of(msg.pose.pose.orientation))
        self.odom_w = msg.twist.twist.angular.z

    def on_imu(self, msg):
        self.n['imu'] += 1
        self.imu_yaw = self.imu_u.update(yaw_of(msg.orientation))
        self.imu_w = msg.angular_velocity.z

    def report(self):
        if self.odom_yaw is None or self.imu_yaw is None:
            self.get_logger().info('waiting for /odom and /imu/data ...')
            return
        o, i = math.degrees(self.odom_yaw), math.degrees(self.imu_yaw)
        self.get_logger().info(
            f'heading  wheels={o:8.1f} deg   imu={i:8.1f} deg   diff={o - i:7.1f} deg | '
            f'turn rate  wheels={self.odom_w:6.3f}  imu={self.imu_w:6.3f} rad/s')
        j = [f'{v:9.3f}' if v is not None else '     none' for v in self.joints]
        self.get_logger().info(
            f'   joint angles L,R = {j[0]},{j[1]} rad | ticks L,R = {self.ticks[0]},{self.ticks[1]} | '
            f'msgs/s: joints={self.n["joints"]} ticks={self.n["ticks"]} '
            f'odom={self.n["odom"]} imu={self.n["imu"]}')
        self.n = {k: 0 for k in self.n}


def main(args=None):
    rclpy.init(args=args)
    node = YawCompare()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
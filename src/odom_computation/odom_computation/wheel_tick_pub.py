#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32


class WheelTickPublisher(Node):

    def __init__(self):
        super().__init__('wheel_tick_publisher')

        # Robot parameters
        self.wheel_radius = 0.065       # meters
        self.wheel_base = 0.21         # meters
        self.ticks_per_revolution = 512

        # Simulated robot motion
        self.linear_velocity = 0.2     # m/s
        self.angular_velocity = 0.0    # rad/s

        # Encoder state
        self.left_ticks = 0
        self.right_ticks = 0

        # Fractional tick accumulation
        self.left_tick_accumulator = 0.0
        self.right_tick_accumulator = 0.0

        # Publishers
        self.left_tick_pub = self.create_publisher(
            Int32,
            '/left_wheel_ticks',
            10
        )

        self.right_tick_pub = self.create_publisher(
            Int32,
            '/right_wheel_ticks',
            10
        )

        # 10 Hz update rate
        self.publish_rate = 10.0
        self.timer_period = 1.0 / self.publish_rate

        self.timer = self.create_timer(
            self.timer_period,
            self.timer_callback
        )

        self.get_logger().info(
            'Simulated wheel encoder publisher started'
        )

    def timer_callback(self):

        dt = self.timer_period

        # Differential-drive wheel velocities
        v_left = (
            self.linear_velocity
            - (self.angular_velocity * self.wheel_base / 2.0)
        )

        v_right = (
            self.linear_velocity
            + (self.angular_velocity * self.wheel_base / 2.0)
        )

        # Distance travelled by each wheel
        left_distance = v_left * dt
        right_distance = v_right * dt

        # Wheel circumference
        wheel_circumference = 2.0 * math.pi * self.wheel_radius

        # Convert distance to encoder ticks
        ticks_per_meter = (
            self.ticks_per_revolution / wheel_circumference
        )

        left_tick_increment = left_distance * ticks_per_meter
        right_tick_increment = right_distance * ticks_per_meter

        # Keep fractional ticks instead of throwing them away
        self.left_tick_accumulator += left_tick_increment
        self.right_tick_accumulator += right_tick_increment

        left_increment = math.trunc(self.left_tick_accumulator)
        right_increment = math.trunc(self.right_tick_accumulator)

        self.left_tick_accumulator -= left_increment
        self.right_tick_accumulator -= right_increment

        # Update cumulative encoder counts
        self.left_ticks += left_increment
        self.right_ticks += right_increment

        # Publish
        left_msg = Int32()
        left_msg.data = self.left_ticks

        right_msg = Int32()
        right_msg.data = self.right_ticks

        self.left_tick_pub.publish(left_msg)
        self.right_tick_pub.publish(right_msg)


def main(args=None):

    rclpy.init(args=args)

    node = WheelTickPublisher()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

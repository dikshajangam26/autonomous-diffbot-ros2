#!/usr/bin/env python3
"""
wheel_tick_pub.py  -  Phase 1, Task 2

Publishes the robot's cumulative wheel encoder ticks on /wheel_ticks at a
fixed rate (default 50 Hz).

Message: std_msgs/Int32MultiArray
    data[0] = LEFT  wheel total ticks
    data[1] = RIGHT wheel total ticks

Two modes (chosen with the 'mode' parameter):
    sim    (default) - no hardware needed. Fake ticks are generated from a
                       pretend robot speed. Use this with Gazebo/testing.
    serial           - reads real ticks from a microcontroller (Arduino/ESP32/
                       Teensy) over a USB serial port. The microcontroller must
                       print one line per update, like:   1234,1230\\n
                       (left_ticks,right_ticks as cumulative totals).

Examples:
    ros2 run task_robotics wheel_tick_pub
    ros2 run task_robotics wheel_tick_pub --ros-args -p publish_rate:=50.0
    ros2 run task_robotics wheel_tick_pub --ros-args -p mode:=serial \
        -p serial_port:=/dev/ttyUSB0 -p baud_rate:=115200
"""

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray


class WheelTickPublisher(Node):

    def __init__(self):
        super().__init__('wheel_tick_publisher')

        # ---------- Parameters (can be changed from the command line) ----------
        self.declare_parameter('mode', 'sim')               # 'sim' or 'serial'
        self.declare_parameter('publish_rate', 50.0)        # Hz
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 115200)

        # Robot numbers (used only in sim mode to make fake ticks)
        self.declare_parameter('wheel_radius', 0.065)       # meters
        self.declare_parameter('wheel_base', 0.21)          # meters
        self.declare_parameter('ticks_per_revolution', 512)
        self.declare_parameter('sim_linear_velocity', 0.2)  # m/s
        self.declare_parameter('sim_angular_velocity', 0.0)  # rad/s

        self.mode = self.get_parameter('mode').value
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.wheel_base = self.get_parameter('wheel_base').value
        self.ticks_per_rev = self.get_parameter('ticks_per_revolution').value
        self.linear_velocity = self.get_parameter('sim_linear_velocity').value
        self.angular_velocity = self.get_parameter('sim_angular_velocity').value

        # ---------- State ----------
        self.left_ticks = 0
        self.right_ticks = 0
        self.left_remainder = 0.0    # fractional ticks we keep for next time
        self.right_remainder = 0.0
        self.serial = None

        if self.mode == 'serial':
            self._open_serial()
        elif self.mode != 'sim':
            raise ValueError("Parameter 'mode' must be 'sim' or 'serial'")

        # ---------- Publisher ----------
        self.pub = self.create_publisher(Int32MultiArray, '/wheel_ticks', 10)

        # ---------- Timer: this is what makes the rate steady ----------
        self.dt = 1.0 / self.publish_rate
        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.get_logger().info(
            f"wheel_tick_publisher started: mode={self.mode}, "
            f"rate={self.publish_rate} Hz, topic=/wheel_ticks"
        )

    # ------------------------------------------------------------------
    def _open_serial(self):
        import serial  # pyserial; only needed in serial mode
        port = self.get_parameter('serial_port').value
        baud = self.get_parameter('baud_rate').value
        # timeout=0 means "never wait": reading never blocks our 50 Hz timer
        self.serial = serial.Serial(port, baud, timeout=0)
        self.serial_buffer = b''
        self.get_logger().info(f'Opened serial port {port} at {baud} baud')

    # ------------------------------------------------------------------
    def _update_from_sim(self):
        """Make fake ticks from a pretend robot speed (differential drive)."""
        half = self.angular_velocity * self.wheel_base / 2.0
        v_left = self.linear_velocity - half
        v_right = self.linear_velocity + half

        ticks_per_meter = self.ticks_per_rev / (2.0 * math.pi * self.wheel_radius)

        self.left_remainder += v_left * self.dt * ticks_per_meter
        self.right_remainder += v_right * self.dt * ticks_per_meter

        # Whole ticks only; keep the leftover fraction for the next cycle
        left_whole = math.trunc(self.left_remainder)
        right_whole = math.trunc(self.right_remainder)
        self.left_remainder -= left_whole
        self.right_remainder -= right_whole

        self.left_ticks += left_whole
        self.right_ticks += right_whole

    # ------------------------------------------------------------------
    def _update_from_serial(self):
        """Read any lines that have arrived; keep the newest valid one."""
        waiting = self.serial.in_waiting
        if waiting:
            self.serial_buffer += self.serial.read(waiting)

        # Split into complete lines; the last piece may be half-received
        *lines, self.serial_buffer = self.serial_buffer.split(b'\n')

        for line in lines:
            try:
                left_str, right_str = line.decode().strip().split(',')
                self.left_ticks = int(left_str)
                self.right_ticks = int(right_str)
            except ValueError:
                self.get_logger().warn(f'Ignoring bad serial line: {line!r}')

    # ------------------------------------------------------------------
    def timer_callback(self):
        if self.mode == 'sim':
            self._update_from_sim()
        else:
            self._update_from_serial()

        msg = Int32MultiArray()
        msg.data = [int(self.left_ticks), int(self.right_ticks)]
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WheelTickPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.serial is not None:
            node.serial.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
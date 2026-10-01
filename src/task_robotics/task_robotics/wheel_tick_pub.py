#!/usr/bin/env python3
"""
wheel_tick_pub.py  -  Phase 1 Task 2 (+ Gazebo mode added in Phase 2)

Publishes the robot's cumulative wheel encoder ticks on /wheel_ticks at a
fixed rate (default 50 Hz).

Message: std_msgs/Int32MultiArray
    data[0] = LEFT  wheel total ticks
    data[1] = RIGHT wheel total ticks

Three modes (parameter 'mode'):
    sim     (default) - fake ticks from a pretend constant speed. No hardware.
    gazebo            - the wheels in the Gazebo simulation ARE the encoders:
                        wheel angle from /joint_states -> ticks.
                        This is what we use with the simulated robot. Ticks are published the
                        moment each /joint_states message arrives (Gazebo sets the 50 Hz rate),
                        and only the NEWEST message is kept, so the ticks never lag behind.
    serial            - real ticks from a microcontroller (Arduino/ESP32/Teensy) over USB.
                        It must print one line per update:   1234,1230\\n
                        (left_ticks,right_ticks as cumulative totals).

Examples:
    ros2 run task_robotics wheel_tick_pub
    ros2 run task_robotics wheel_tick_pub --ros-args -p mode:=gazebo -p use_sim_time:=true
    ros2 run task_robotics wheel_tick_pub --ros-args -p mode:=serial \
        -p serial_port:=/dev/ttyUSB0 -p baud_rate:=115200
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32MultiArray


class WheelTickPublisher(Node):

    def __init__(self):
        super().__init__('wheel_tick_publisher')

        # ---------- Parameters (can be changed from the command line) ----------
        self.declare_parameter('mode', 'sim')               # 'sim', 'gazebo' or 'serial'
        self.declare_parameter('publish_rate', 50.0)        # Hz
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 115200)

        # Robot numbers
        self.declare_parameter('wheel_radius', 0.065)       # meters
        self.declare_parameter('wheel_base', 0.21)          # meters
        self.declare_parameter('ticks_per_revolution', 512)
        self.declare_parameter('sim_linear_velocity', 0.2)  # m/s   (sim mode only)
        self.declare_parameter('sim_angular_velocity', 0.0)  # rad/s (sim mode only)

        # Joint names (gazebo mode only)
        self.declare_parameter('left_joint', 'left_wheel_joint')
        self.declare_parameter('right_joint', 'right_wheel_joint')

        self.mode = self.get_parameter('mode').value
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.wheel_base = self.get_parameter('wheel_base').value
        self.ticks_per_rev = self.get_parameter('ticks_per_revolution').value
        self.linear_velocity = self.get_parameter('sim_linear_velocity').value
        self.angular_velocity = self.get_parameter('sim_angular_velocity').value
        self.left_joint = self.get_parameter('left_joint').value
        self.right_joint = self.get_parameter('right_joint').value

        # ---------- State ----------
        self.left_ticks = 0
        self.right_ticks = 0
        self.left_remainder = 0.0    # fractional ticks we keep for next time
        self.right_remainder = 0.0
        self.serial = None
        self.joint_pos = {}          # latest wheel angles from /joint_states
        self.joint_zero = None       # wheel angles when we started (so ticks start at 0)

        if self.mode == 'serial':
            self._open_serial()
        elif self.mode == 'gazebo':
            # depth=1 + best effort: always work on the newest wheel angles, never a backlog
            latest_only = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST,
                                     reliability=QoSReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(JointState, '/joint_states',
                                     self._on_joint_states, latest_only)
        elif self.mode != 'sim':
            raise ValueError("Parameter 'mode' must be 'sim', 'gazebo' or 'serial'")

        # ---------- Publisher ----------
        self.pub = self.create_publisher(Int32MultiArray, '/wheel_ticks', 10)

        # ---------- Timer: this is what makes the rate steady ----------
        self.dt = 1.0 / self.publish_rate
        self.timer = None
        if self.mode != 'gazebo':      # gazebo mode is driven by incoming joint states instead
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
        return True

    # ------------------------------------------------------------------
    def _on_joint_states(self, msg):
        for name, position in zip(msg.name, msg.position):
            if name in (self.left_joint, self.right_joint):
                self.joint_pos[name] = position
        if self._update_from_gazebo():
            self._publish()

    def _update_from_gazebo(self):
        """Turn the wheel angles (radians) coming from Gazebo into encoder ticks."""
        if self.left_joint not in self.joint_pos or self.right_joint not in self.joint_pos:
            return False                       # nothing from Gazebo yet

        left = self.joint_pos[self.left_joint]
        right = self.joint_pos[self.right_joint]
        if self.joint_zero is None:            # first reading becomes tick zero
            self.joint_zero = (left, right)

        ticks_per_rad = self.ticks_per_rev / (2.0 * math.pi)
        self.left_ticks = round((left - self.joint_zero[0]) * ticks_per_rad)
        self.right_ticks = round((right - self.joint_zero[1]) * ticks_per_rad)
        return True

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
        return True

    # ------------------------------------------------------------------
    def timer_callback(self):
        if self.mode == 'sim':
            ready = self._update_from_sim()
        else:
            ready = self._update_from_serial()

        if ready:
            self._publish()

    def _publish(self):
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
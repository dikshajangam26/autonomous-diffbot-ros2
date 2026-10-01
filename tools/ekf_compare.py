#!/usr/bin/env python3
"""
ekf_compare.py - how good is our position estimate?  (Phase 2, Task 5 test tool)

Once per second prints, for the true pose (Gazebo ground truth), the raw wheel odometry
(/odom) and the EKF output (/odometry/filtered):
    x, y (meters), heading (degrees), and the ERROR against the truth.
All poses are shown relative to where the robot was when this tool started.

Run (inside the container, simulation running):
    python3 /ws/tools/ekf_compare.py --ros-args -p use_sim_time:=true

Ground truth is only for testing. The robot itself never uses it.
"""
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Track:
    """Pose relative to the first message, with heading unwrapped (so 2 turns = 720 deg)."""

    def __init__(self):
        self.x0 = self.y0 = self.yaw0 = None
        self.prev = None
        self.total = 0.0
        self.x = self.y = self.yaw = None

    def update(self, msg):
        p = msg.pose.pose
        raw = yaw_of(p.orientation)
        if self.x0 is None:
            self.x0, self.y0, self.yaw0 = p.position.x, p.position.y, raw
            self.prev = raw
        self.total += math.atan2(math.sin(raw - self.prev), math.cos(raw - self.prev))
        self.prev = raw
        dx, dy = p.position.x - self.x0, p.position.y - self.y0
        c, s = math.cos(-self.yaw0), math.sin(-self.yaw0)      # express in the starting frame
        self.x, self.y, self.yaw = c * dx - s * dy, s * dx + c * dy, self.total


class EkfCompare(Node):

    def __init__(self):
        super().__init__('ekf_compare')
        self.t = {'truth': Track(), 'wheels': Track(), 'ekf': Track()}
        self.create_subscription(Odometry, '/ground_truth/odom', lambda m: self.t['truth'].update(m), 10)
        self.create_subscription(Odometry, '/odom', lambda m: self.t['wheels'].update(m), 10)
        self.create_subscription(Odometry, '/odometry/filtered', lambda m: self.t['ekf'].update(m), 10)
        self.create_timer(1.0, self.report)
        self.get_logger().info('ekf_compare running (poses start at 0 now)')

    def report(self):
        missing = [k for k, v in self.t.items() if v.x is None]
        if missing:
            self.get_logger().info('waiting for: ' + ', '.join(missing))
            return
        tr = self.t['truth']
        lines = [f"  TRUTH   x={tr.x:7.3f}  y={tr.y:7.3f}  heading={math.degrees(tr.yaw):8.1f} deg"]
        for name in ('wheels', 'ekf'):
            k = self.t[name]
            pos_err = math.hypot(k.x - tr.x, k.y - tr.y)
            yaw_err = math.degrees(k.yaw - tr.yaw)
            lines.append(f"  {name.upper():<7} x={k.x:7.3f}  y={k.y:7.3f}  heading={math.degrees(k.yaw):8.1f} deg"
                         f"   ERROR: position {pos_err:6.3f} m, heading {yaw_err:7.1f} deg")
        self.get_logger().info('\n' + '\n'.join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = EkfCompare()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

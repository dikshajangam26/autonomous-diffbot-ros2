#!/usr/bin/env python3
"""
slam_compare.py - does SLAM make the position better?  (Phase 3, Task 7b test tool)

Once per second prints the true pose (Gazebo ground truth) next to
    EKF   = odometry only  (/odometry/filtered, the odom frame: drifts over time)
    SLAM  = odometry + LiDAR map correction  (TF  map -> base_link)
with the ERROR of each against the truth. Poses are relative to where the robot was when this
tool started, so start it right at the beginning and then drive around.

Run (inside the container, simulation + slam:=true running):
    python3 /ws/tools/slam_compare.py --ros-args -p use_sim_time:=true

Ground truth is only for testing. The robot itself never uses it.
"""
import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Track:
    """Pose relative to the first sample, heading unwrapped (so 2 turns = 720 deg)."""

    def __init__(self):
        self.x0 = self.y0 = self.yaw0 = None
        self.prev = None
        self.total = 0.0
        self.x = self.y = self.yaw = None

    def update(self, x, y, raw):
        if self.x0 is None:
            self.x0, self.y0, self.yaw0, self.prev = x, y, raw, raw
        self.total += math.atan2(math.sin(raw - self.prev), math.cos(raw - self.prev))
        self.prev = raw
        dx, dy = x - self.x0, y - self.y0
        c, s = math.cos(-self.yaw0), math.sin(-self.yaw0)
        self.x, self.y, self.yaw = c * dx - s * dy, s * dx + c * dy, self.total


class SlamCompare(Node):

    def __init__(self):
        super().__init__('slam_compare')
        self.t = {'truth': Track(), 'ekf': Track(), 'slam': Track()}
        self.create_subscription(Odometry, '/ground_truth/odom', self.on_truth, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self.on_ekf, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(0.1, self.poll_slam)
        self.create_timer(1.0, self.report)
        self.get_logger().info('slam_compare running (poses start at 0 now)')

    def on_truth(self, m):
        p = m.pose.pose
        self.t['truth'].update(p.position.x, p.position.y, yaw_of(p.orientation))

    def on_ekf(self, m):
        p = m.pose.pose
        self.t['ekf'].update(p.position.x, p.position.y, yaw_of(p.orientation))

    def poll_slam(self):
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_link', Time())
        except TransformException:
            return
        v, q = tf.transform.translation, tf.transform.rotation
        self.t['slam'].update(v.x, v.y, yaw_of(q))

    def report(self):
        missing = [k for k, v in self.t.items() if v.x is None]
        if missing:
            self.get_logger().info('waiting for: ' + ', '.join(missing))
            return
        tr = self.t['truth']
        lines = [f"  TRUTH  x={tr.x:7.3f}  y={tr.y:7.3f}  heading={math.degrees(tr.yaw):8.1f} deg"]
        for name in ('ekf', 'slam'):
            k = self.t[name]
            pos_err = math.hypot(k.x - tr.x, k.y - tr.y)
            yaw_err = math.degrees(k.yaw - tr.yaw)
            lines.append(f"  {name.upper():<6} x={k.x:7.3f}  y={k.y:7.3f}  heading={math.degrees(k.yaw):8.1f} deg"
                         f"   ERROR: position {pos_err:6.3f} m, heading {yaw_err:7.1f} deg")
        self.get_logger().info('\n' + '\n'.join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = SlamCompare()
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
#!/usr/bin/env python3
"""
sensor_check.py  -  Phase 2, Task 4 verification tool

Every 2 seconds prints one line per sensor topic showing:
  - rate     : messages per second
  - frame_id : which frame the data is stamped with
  - age      : how old the timestamp is compared with the current (sim) time
  - TF       : whether that frame is connected to base_link in the TF tree
and a final verdict. Run it with simulated time while Gazebo is running:

    ros2 run task_robotics sensor_check --ros-args -p use_sim_time:=true
"""
from functools import partial

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, Imu, LaserScan
from tf2_ros import Buffer, TransformListener

# topic, message type, frame the data SHOULD be stamped with
SENSORS = [
    ('/scan', LaserScan, 'lidar_link'),
    ('/imu/data', Imu, 'imu_link'),
    ('/stereo/left/image_raw', Image, 'stereo_left_optical_frame'),
    ('/stereo/right/image_raw', Image, 'stereo_right_optical_frame'),
]
REPORT_PERIOD = 2.0     # seconds
MAX_AGE = 1.0           # seconds: a stamp further from "now" than this is suspicious


class SensorCheck(Node):

    def __init__(self):
        super().__init__('sensor_check')
        self.declare_parameter('base_frame', 'base_link')
        self.base_frame = self.get_parameter('base_frame').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.stats = {}
        for topic, msg_type, expected in SENSORS:
            self.stats[topic] = {'count': 0, 'last_count': 0, 'frame': '', 'age': None,
                                 'zero_stamp': False, 'expected': expected}
            # Best-effort QoS can listen to both reliable and best-effort publishers
            self.create_subscription(msg_type, topic, partial(self.on_msg, topic),
                                     qos_profile_sensor_data)

        self.create_timer(REPORT_PERIOD, self.report)
        self.get_logger().info('sensor_check running - waiting for data...')

    def on_msg(self, topic, msg):
        s = self.stats[topic]
        s['count'] += 1
        s['frame'] = msg.header.frame_id
        stamp = Time.from_msg(msg.header.stamp)
        s['zero_stamp'] = stamp.nanoseconds == 0
        s['age'] = (self.get_clock().now() - stamp).nanoseconds * 1e-9

    def report(self):
        all_ok = True
        lines = []
        for topic, _, expected in SENSORS:
            s = self.stats[topic]
            rate = (s['count'] - s['last_count']) / REPORT_PERIOD
            s['last_count'] = s['count']

            problems = []
            if rate <= 0.0:
                problems.append('no data')
            else:
                if s['frame'] != expected:
                    problems.append(f"frame should be '{expected}'")
                if s['zero_stamp']:
                    problems.append('timestamp is zero')
                elif s['age'] is None or abs(s['age']) > MAX_AGE:
                    problems.append('timestamp far from current time (use_sim_time?)')
                if not self.tf_buffer.can_transform(self.base_frame, s['frame'], Time()):
                    problems.append(f"no TF {self.base_frame}->{s['frame']}")

            ok = not problems
            all_ok = all_ok and ok
            age = 'n/a' if s['age'] is None else f"{s['age']:+.3f}s"
            lines.append(f"  {'OK  ' if ok else 'FAIL'} {topic:<26} {rate:6.1f} Hz  "
                         f"frame={s['frame'] or '-':<28} age={age:<9}"
                         + ('' if ok else '  <- ' + '; '.join(problems)))

        self.get_logger().info('\n' + '\n'.join(lines) +
                               f"\n  ==> {'ALL SENSORS OK' if all_ok else 'PROBLEMS FOUND'}")


def main(args=None):
    rclpy.init(args=args)
    node = SensorCheck()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

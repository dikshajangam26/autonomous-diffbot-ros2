#!/usr/bin/env python3
"""
cloud_filter.py  -  Phase 3, Task 6

Cleans up the stereo point cloud so later stages (object detection, SLAM) have less to chew on.

Listens to : /stereo/points2          (sensor_msgs/PointCloud2, made by stereo_image_proc,
                                       expressed in the LEFT CAMERA's optical frame)
Publishes  : /stereo/points_filtered  (sensor_msgs/PointCloud2, in base_link, so z = height)

Steps for every cloud, in this order (cheapest first, so the expensive step sees fewer points):
    1. drop invalid points (NaN: the stereo matcher found no match there)
    2. move the points from the camera frame into base_link, so "z" means height above the floor
       (floor is about z = -0.065 because base_link sits at wheel-axle height)
    3. PASS-THROUGH filter: keep only a box in front of the robot
           z_min..z_max   throws away the FLOOR (below) and the CEILING / tall stuff (above)
           x_min..x_max   throws away points too close and too far (stereo gets noisy far away)
           y_min..y_max   throws away points far to the sides
    4. VOXEL GRID filter: one averaged point per voxel_size cube. Millions of points on a
       wall become a few hundred, but the shapes stay.
"""
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener

from task_robotics import cloud_ops


class CloudFilter(Node):

    def __init__(self):
        super().__init__('cloud_filter')

        self.declare_parameter('input_topic', '/stereo/points2')
        self.declare_parameter('output_topic', '/stereo/points_filtered')
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('voxel_size', 0.05)      # meters (cube side)
        # pass-through box, in the target frame (meters)
        self.declare_parameter('x_min', 0.2)
        self.declare_parameter('x_max', 6.0)
        self.declare_parameter('y_min', -4.0)
        self.declare_parameter('y_max', 4.0)
        self.declare_parameter('z_min', 0.02)           # just above the floor (floor ~ -0.065)
        self.declare_parameter('z_max', 1.5)            # below the ceiling

        g = self.get_parameter
        self.target = g('target_frame').value
        self.leaf = float(g('voxel_size').value)
        self.limits = {a: (float(g(f'{a}_min').value), float(g(f'{a}_max').value))
                       for a in 'xyz'}

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # depth 1 + best effort: if we are slow we skip old clouds instead of building a backlog
        qos = QoSProfile(depth=1, history=QoSHistoryPolicy.KEEP_LAST,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.sub = self.create_subscription(
            PointCloud2, g('input_topic').value, self.on_cloud, qos)
        self.pub = self.create_publisher(PointCloud2, g('output_topic').value, 5)

        self.stats = dict(frames=0, raw=0, valid=0, floor=0, high=0, kept_box=0, voxel=0, ms=0.0)
        self.last_report = time.monotonic()
        self.get_logger().info(
            f'cloud_filter: {g("input_topic").value} -> {g("output_topic").value} '
            f'in {self.target} | box x{self.limits["x"]} y{self.limits["y"]} z{self.limits["z"]} '
            f'| voxel {self.leaf} m')

    # ------------------------------------------------------------------
    def on_cloud(self, msg):
        t0 = time.monotonic()

        # camera frame -> base_link. The camera is bolted to the robot, so this transform never
        # changes; asking for the "latest" one (Time()) is always right and never has to wait.
        try:
            tf = self.tf_buffer.lookup_transform(self.target, msg.header.frame_id, Time())
        except TransformException as err:
            self.get_logger().warn(f'waiting for TF {msg.header.frame_id} -> {self.target}: {err}',
                                   throttle_duration_sec=2.0)
            return
        q = tf.transform.rotation
        t = tf.transform.translation
        rot = cloud_ops.quaternion_to_matrix(q.x, q.y, q.z, q.w)
        trans = np.array([t.x, t.y, t.z])

        fields = [(f.name, f.offset, f.datatype) for f in msg.fields]
        xyz, rgb = cloud_ops.unpack_cloud(fields, msg.point_step, bytes(msg.data))
        n_raw = msg.width * msg.height
        n_valid = len(xyz)

        xyz = cloud_ops.transform_points(xyz, rot, trans)

        # statistics: how many are floor / ceiling (only counting points inside the x/y window)
        in_xy = ((xyz[:, 0] >= self.limits['x'][0]) & (xyz[:, 0] <= self.limits['x'][1]) &
                 (xyz[:, 1] >= self.limits['y'][0]) & (xyz[:, 1] <= self.limits['y'][1]))
        n_floor = int(np.count_nonzero(in_xy & (xyz[:, 2] < self.limits['z'][0])))
        n_high = int(np.count_nonzero(in_xy & (xyz[:, 2] > self.limits['z'][1])))

        keep = cloud_ops.passthrough_masks(xyz, self.limits)
        xyz = xyz[keep]
        rgb = rgb[keep] if rgb is not None else None
        n_box = len(xyz)

        xyz, rgb = cloud_ops.voxel_downsample(xyz, rgb, self.leaf)

        out = PointCloud2()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.target
        out.height = 1
        out.width = len(xyz)
        out.fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                      PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                      PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                      PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)]
        out.is_bigendian = False
        out.point_step = 16
        out.row_step = 16 * out.width
        out.is_dense = True
        out.data = cloud_ops.pack_cloud(xyz, rgb)
        self.pub.publish(out)

        s = self.stats
        s['frames'] += 1
        s['raw'] += n_raw
        s['valid'] += n_valid
        s['floor'] += n_floor
        s['high'] += n_high
        s['kept_box'] += n_box
        s['voxel'] += len(xyz)
        s['ms'] += (time.monotonic() - t0) * 1000.0
        self.report()

    # ------------------------------------------------------------------
    def report(self):
        now = time.monotonic()
        if now - self.last_report < 2.0 or self.stats['frames'] == 0:
            return
        s = self.stats
        n = s['frames']
        self.get_logger().info(
            f'per cloud: raw {s["raw"] // n} -> valid {s["valid"] // n} '
            f'| removed floor {s["floor"] // n}, ceiling/high {s["high"] // n} '
            f'-> in box {s["kept_box"] // n} -> after voxel {s["voxel"] // n} '
            f'| {s["ms"] / n:.0f} ms/cloud, {n / (now - self.last_report):.1f} clouds/s')
        self.stats = dict(frames=0, raw=0, valid=0, floor=0, high=0, kept_box=0, voxel=0, ms=0.0)
        self.last_report = now


def main(args=None):
    rclpy.init(args=args)
    node = CloudFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

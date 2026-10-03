#!/usr/bin/env python3
"""
semantic_costmap.py  -  Phase 4, Task 8 (the bridge between the semantic object map and Nav2)

Nav2's costmaps only understand "obstacle points". This node turns the semantic object map into
such points, so the global AND local costmaps treat recognised racks, pallets, boxes and people as
obstacles with a class-specific keep-out patch.

Listens to : /semantic_map/landmarks   the object map from object_slam (Detection3DArray, frame map)
             /objects/detections       YOLO sightings; only 'person' is used here (they move)
             TF map <- base_link       to place people in the map
Publishes  : /semantic_costmap/obstacles  PointCloud2 (frame map) - read by both costmaps as the
                                          observation source "semantic"
             /semantic_costmap/markers    RViz: the keep-out discs (orange = static objects, pink = people)
Start-up   : if object_slam has not published yet, the saved file maps/semantic_map.yaml is used.

Static objects stay in the cloud as long as they are in the map AND the LiDAR map shows a surface next to them
(a landmark over empty floor is a ghost: it would plug an aisle that is really free). People are only kept for
`person_ttl` seconds after the last sighting, so the patch follows them and disappears when they leave.
"""
import os

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

import numpy as np

from task_robotics import semantic_ops as so


class SemanticCostmap(Node):
    def __init__(self):
        super().__init__('semantic_costmap')
        self.declare_parameter('map_file', '/ws/maps/semantic_map.yaml')
        self.declare_parameter('rack_radius', so.DEFAULT_KEEPOUT['shelving rack'])
        self.declare_parameter('pallet_radius', so.DEFAULT_KEEPOUT['pallet'])
        self.declare_parameter('box_radius', so.DEFAULT_KEEPOUT['cardboard box'])
        self.declare_parameter('person_radius', 0.35)
        self.declare_parameter('person_ttl', 1.5)
        self.declare_parameter('publish_rate', 2.0)
        self.declare_parameter('min_person_score', 0.4)
        self.declare_parameter('support_radius', 0.20)   # a rack/box landmark needs a mapped surface this close

        self.table = {'shelving rack': self.get_parameter('rack_radius').value,
                      'pallet': self.get_parameter('pallet_radius').value,
                      'cardboard box': self.get_parameter('box_radius').value}
        self.person_radius = self.get_parameter('person_radius').value
        self.person_ttl = self.get_parameter('person_ttl').value
        self.min_person_score = self.get_parameter('min_person_score').value
        self.support_radius = self.get_parameter('support_radius').value
        self.occupied = np.zeros((0, 2))
        self.last_rejected = -1

        self.landmarks = []          # list of {label, x, y, size}
        self.live = False            # True once object_slam has published
        self.people = {}             # slot -> (x, y, time_seen) in the map frame
        self.last_count = -1

        path = self.get_parameter('map_file').value
        if path and os.path.exists(path):
            try:
                self.landmarks = so.load_landmarks_yaml(path)
                self.get_logger().info(f'loaded {len(self.landmarks)} landmarks from {path}')
            except Exception as err:  # noqa: BLE001 - a bad file must not stop navigation
                self.get_logger().warn(f'could not read {path}: {err}')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            OccupancyGrid, '/map', self.on_map,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(Detection3DArray, '/semantic_map/landmarks', self.on_landmarks, 5)
        self.create_subscription(Detection3DArray, '/objects/detections', self.on_detections, 5)
        self.cloud_pub = self.create_publisher(PointCloud2, '/semantic_costmap/obstacles', 5)
        self.marker_pub = self.create_publisher(MarkerArray, '/semantic_costmap/markers', 5)
        self.create_timer(1.0 / self.get_parameter('publish_rate').value, self.publish)

    # ------------------------------------------------------------------ inputs
    def on_map(self, m):
        self.occupied = so.occupied_cells_xy(m.data, m.info.width, m.info.height, m.info.resolution,
                                             m.info.origin.position.x, m.info.origin.position.y)

    def on_landmarks(self, msg):
        lms = []
        for d in msg.detections:
            if not d.results:
                continue
            lms.append({'label': d.results[0].hypothesis.class_id,
                        'x': d.bbox.center.position.x, 'y': d.bbox.center.position.y,
                        'size': (d.bbox.size.x, d.bbox.size.y, d.bbox.size.z)})
        self.landmarks = lms          # the live map replaces the start-up file completely
        self.live = True

    def on_detections(self, msg):
        persons = [d for d in msg.detections if d.results
                   and d.results[0].hypothesis.class_id == 'person'
                   and d.results[0].hypothesis.score >= self.min_person_score]
        if not persons:
            return
        frame = msg.header.frame_id or 'base_link'
        try:
            tf = self.tf_buffer.lookup_transform('map', frame, Time())     # newest available
        except TransformException:
            return
        t = tf.transform
        yaw = so.yaw_from_quaternion(t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w)
        now = self.get_clock().now().nanoseconds * 1e-9
        for d in persons:
            x, y = so.transform_xy(d.bbox.center.position.x, d.bbox.center.position.y,
                                   t.translation.x, t.translation.y, yaw)
            # same person if within 0.6 m of an existing slot, otherwise a new slot
            slot = next((k for k, (px, py, _) in self.people.items()
                         if (px - x) ** 2 + (py - y) ** 2 < 0.36), None)
            if slot is None:
                slot = max(self.people, default=-1) + 1
            self.people[slot] = (x, y, now)

    # ------------------------------------------------------------------ output
    def publish(self):
        now_s = self.get_clock().now().nanoseconds * 1e-9
        self.people = {k: v for k, v in self.people.items() if now_s - v[2] <= self.person_ttl}

        kept, rejected = so.filter_supported(self.landmarks, self.occupied, self.support_radius)
        if len(rejected) != self.last_rejected:
            self.last_rejected = len(rejected)
            for lm in rejected:
                self.get_logger().info(f"ignored ghost landmark {lm['label']} at ({lm['x']:.2f}, {lm['y']:.2f}): "
                                       f'no mapped surface within {self.support_radius} m')
        pts, discs = so.landmark_points(kept, self.table)
        chunks = [pts]
        person_discs = []
        for x, y, _ in self.people.values():
            chunks.append(so.disc_points(x, y, self.person_radius))
            person_discs.append(('person', x, y, self.person_radius))
        cloud_pts = np.vstack(chunks) if chunks else np.zeros((0, 3), np.float32)

        stamp = self.get_clock().now().to_msg()
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = 'map'
        msg.height = 1
        msg.width = int(len(cloud_pts))
        msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                      for i, n in enumerate('xyz')]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * msg.width
        msg.is_dense = True
        msg.data = so.pack_xyz(cloud_pts)
        self.cloud_pub.publish(msg)

        if len(self.landmarks) != self.last_count:
            self.last_count = len(self.landmarks)
            src = 'live object map' if self.live else 'saved file'
            self.get_logger().info(f'semantic costmap source: {self.last_count} landmarks ({src})')
        self.publish_markers(stamp, discs, person_discs)

    def publish_markers(self, stamp, discs, person_discs):
        arr = MarkerArray()
        wipe = Marker()
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)
        for i, (label, x, y, r) in enumerate(discs + person_discs):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = 'map'
            m.ns = 'keepout'
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), 0.02
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = float(2.0 * r)
            m.scale.z = 0.04
            if label == 'person':
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.1, 0.8, 0.5
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.55, 0.0, 0.5
            arr.markers.append(m)
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = SemanticCostmap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()

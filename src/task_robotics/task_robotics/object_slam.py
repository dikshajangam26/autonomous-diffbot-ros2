#!/usr/bin/env python3
"""
object_slam.py  -  Phase 3, Task 7 (step c: the semantic map and the map -> odom transform)

Builds a map of recognised OBJECTS (pallets, shelving racks, boxes) as a factor graph and uses it,
together with the LiDAR SLAM pose and the odometry, to publish the transform  map -> odom.

Listens to : /pose                    slam_toolbox: robot pose in the map from LiDAR scan matching
             /objects/detections      YOLO objects with 3D position (object_detector, in base_link)
             TF odom -> base_link     smooth odometry from the EKF
Publishes  : TF  map -> odom          (slam_toolbox's own map -> odom is switched off by the launch file)
             /semantic_map/markers    RViz: persistent object boxes, graph lines, keyframes, people
             /semantic_map/landmarks  the object map as vision_msgs/Detection3DArray (frame map)
Saves      : the object map to a YAML file (default /ws/maps/semantic_map.yaml)

How one cycle works (see slam_ops.py for the factor graph itself)
  * every ~25 cm of driving the robot makes a KEYFRAME: a new unknown "where was I" X_i in the graph,
    tied to the LiDAR pose and to the previous keyframe through the odometry.
  * every YOLO sighting of a pallet / rack / box becomes a bearing+range factor between the current
    keyframe and that object. Seen 3 times from 2+ keyframes -> the object joins the map for good.
  * the solver (iSAM2) re-solves all robot poses and object positions together.
  * map -> odom = (best estimate of the robot in the map) * (odometry of the robot in odom)^-1
People are moving, so they are shown briefly but never stored as landmarks.
"""
import math
import os
import time

import gtsam
import rclpy
import yaml
from geometry_msgs.msg import Point, PoseWithCovarianceStamped, TransformStamped
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from task_robotics.slam_ops import SemanticGraph

COLOURS = {
    'pallet': (1.0, 0.6, 0.0),
    'shelving rack': (0.2, 0.5, 1.0),
    'cardboard box': (0.8, 0.5, 0.2),
    'person': (1.0, 0.1, 0.8),
}


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ObjectSlam(Node):

    def __init__(self):
        super().__init__('object_slam')
        self.declare_parameter('kf_distance', 0.25)       # new keyframe after this many meters ...
        self.declare_parameter('kf_angle', 0.25)          # ... or this many radians of turning
        self.declare_parameter('min_score', 0.15)         # ignore weaker YOLO detections
        self.declare_parameter('min_hits', 3)             # sightings needed before an object is kept
        self.declare_parameter('min_keyframes', 2)        # ... from at least this many keyframes
        self.declare_parameter('min_range', 0.3)
        self.declare_parameter('max_range', 6.0)
        self.declare_parameter('dynamic_classes', ['person'])
        self.declare_parameter('save_path', '/ws/maps/semantic_map.yaml')

        g = self.get_parameter
        self.kf_distance = float(g('kf_distance').value)
        self.kf_angle = float(g('kf_angle').value)
        self.min_score = float(g('min_score').value)
        self.min_range = float(g('min_range').value)
        self.max_range = float(g('max_range').value)
        self.dynamic = set(g('dynamic_classes').value)
        self.save_path = g('save_path').value

        self.graph = SemanticGraph(min_hits=int(g('min_hits').value),
                                   min_keyframes=int(g('min_keyframes').value))
        self.latest_kf = None
        self.last_kf_map = None
        self.map_to_odom = (0.0, 0.0, 0.0)       # identity until the first keyframe exists
        self.people = []                         # (x, y, time) of people seen recently
        self.n_dets = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(PoseWithCovarianceStamped, '/pose', self.on_pose, 10)
        self.create_subscription(Detection3DArray, '/objects/detections', self.on_detections, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/semantic_map/markers', 5)
        self.lm_pub = self.create_publisher(Detection3DArray, '/semantic_map/landmarks', 5)

        self.create_timer(0.05, self.publish_tf)
        self.create_timer(0.5, self.publish_markers)
        self.create_timer(1.0, self.publish_landmarks)
        self.create_timer(5.0, self.report)
        self.create_timer(10.0, self.save)
        self.get_logger().info('object_slam started: waiting for /pose from slam_toolbox '
                               'and detections from object_detector')

    # ------------------------------------------------------------------ helpers
    def odom_pose(self, stamp):
        """Odometry pose (x, y, yaw) of base_link in odom at `stamp` (latest if not available)."""
        for when in (Time.from_msg(stamp), Time()):
            try:
                tf = self.tf_buffer.lookup_transform('odom', 'base_link', when)
            except TransformException:
                continue
            t, q = tf.transform.translation, tf.transform.rotation
            return (t.x, t.y, yaw_of(q))
        return None

    def refresh_correction(self):
        mo = self.graph.map_to_odom(self.latest_kf)
        self.map_to_odom = (mo.x(), mo.y(), mo.theta())

    # ------------------------------------------------------------------ inputs
    def on_pose(self, msg):
        p = msg.pose.pose
        yaw = yaw_of(p.orientation)
        if self.last_kf_map is not None:
            d = math.hypot(p.position.x - self.last_kf_map[0], p.position.y - self.last_kf_map[1])
            if d < self.kf_distance and abs(wrap(yaw - self.last_kf_map[2])) < self.kf_angle:
                return
        odom = self.odom_pose(msg.header.stamp)
        if odom is None:
            return
        map_pose = (p.position.x, p.position.y, yaw)
        self.latest_kf = self.graph.add_keyframe(map_pose, odom)
        self.last_kf_map = map_pose
        self.graph.update()
        self.refresh_correction()

    def on_detections(self, msg):
        if self.latest_kf is None:
            return
        kf = self.latest_kf
        odom_t = self.odom_pose(msg.header.stamp)
        if odom_t is None:
            return
        # where the robot was when the picture was taken, expressed in the keyframe's frame
        rel = self.graph.kf_odom[kf].between(gtsam.Pose2(*odom_t))
        kf_pose = self.graph.pose(kf)
        now = time.monotonic()
        self.people = [p for p in self.people if now - p[2] < 1.5]

        for det in msg.detections:
            if not det.results:
                continue
            label = det.results[0].hypothesis.class_id
            if det.results[0].hypothesis.score < self.min_score:
                continue
            c, s = det.bbox.center.position, det.bbox.size
            p_kf = rel.transformFrom(gtsam.Point2(c.x, c.y))
            rng = math.hypot(p_kf[0], p_kf[1])
            if not (self.min_range <= rng <= self.max_range):
                continue
            if label in self.dynamic:
                p_map = kf_pose.transformFrom(p_kf)
                self.people.append((p_map[0], p_map[1], now))
                continue
            self.graph.observe(kf, label, math.atan2(p_kf[1], p_kf[0]), rng, (s.x, s.y, s.z))
            self.n_dets += 1
        self.graph.update()
        self.refresh_correction()

    # ------------------------------------------------------------------ outputs
    def publish_tf(self):
        x, y, th = self.map_to_odom
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.rotation.z = math.sin(th / 2.0)
        t.transform.rotation.w = math.cos(th / 2.0)
        self.tf_broadcaster.sendTransform(t)

    def publish_landmarks(self):
        arr = Detection3DArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.header.frame_id = 'map'
        for lm in self.graph.landmarks.values():
            d = Detection3D()
            d.header = arr.header
            d.id = f'{lm.label}_{lm.id}'
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = lm.label
            hyp.hypothesis.score = float(min(1.0, lm.n_obs / 10.0))
            d.results.append(hyp)
            d.bbox.center.position.x, d.bbox.center.position.y = float(lm.xy[0]), float(lm.xy[1])
            d.bbox.center.position.z = float(lm.size[2] / 2.0)
            d.bbox.center.orientation.w = 1.0
            d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = (float(v) for v in lm.size)
            arr.detections.append(d)
        self.lm_pub.publish(arr)

    def _marker(self, ns, mid, mtype, life=0):
        m = Marker()
        m.header.frame_id = 'map'
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.pose.orientation.w = 1.0
        m.lifetime.sec = life
        return m

    def publish_markers(self):
        arr = MarkerArray()
        # persistent objects: a coloured box + a text label (these stay)
        for lm in self.graph.landmarks.values():
            r, g, b = COLOURS.get(lm.label, (0.7, 0.7, 0.7))
            sx, sy, sz = (max(float(v), 0.2) for v in lm.size)
            box = self._marker('landmark', lm.id, Marker.CUBE)
            box.pose.position.x, box.pose.position.y, box.pose.position.z = float(lm.xy[0]), float(lm.xy[1]), sz / 2.0
            box.scale.x, box.scale.y, box.scale.z = sx, sy, sz
            box.color.r, box.color.g, box.color.b, box.color.a = r, g, b, 0.6
            arr.markers.append(box)
            txt = self._marker('landmark_text', lm.id, Marker.TEXT_VIEW_FACING)
            txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = float(lm.xy[0]), float(lm.xy[1]), sz + 0.2
            txt.scale.z = 0.18
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            txt.text = f'{lm.label} #{lm.id} ({lm.n_obs})'
            arr.markers.append(txt)

        # the factor graph itself: robot keyframes (dots) and the object factors (lines)
        n = self.graph.n_keyframes
        first = max(0, n - 300)
        kfs = self._marker('keyframes', 0, Marker.SPHERE_LIST)
        kfs.scale.x = kfs.scale.y = kfs.scale.z = 0.07
        kfs.color.r, kfs.color.g, kfs.color.b, kfs.color.a = 0.2, 1.0, 0.2, 1.0
        for i in range(first, n):
            p = self.graph.pose(i)
            kfs.points.append(Point(x=float(p.x()), y=float(p.y()), z=0.05))
        arr.markers.append(kfs)
        lines = self._marker('factors', 0, Marker.LINE_LIST)
        lines.scale.x = 0.01
        lines.color.r = lines.color.g = lines.color.b = lines.color.a = 0.9
        for lm in self.graph.landmarks.values():
            for k in lm.keyframes:
                if k >= first:
                    p = self.graph.pose(k)
                    lines.points.append(Point(x=float(p.x()), y=float(p.y()), z=0.05))
                    lines.points.append(Point(x=float(lm.xy[0]), y=float(lm.xy[1]), z=0.05))
        arr.markers.append(lines)

        # not-yet-trusted sightings (grey, disappear if never confirmed)
        for i, c in enumerate(self.graph.candidates):
            m = self._marker('candidates', i, Marker.SPHERE, life=1)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(c.xy[0]), float(c.xy[1]), 0.2
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.color.r, m.color.g, m.color.b, m.color.a = 0.6, 0.6, 0.6, 0.8
            arr.markers.append(m)
        # people: shown for a moment only
        for i, (x, y, _) in enumerate(self.people):
            m = self._marker('people', i, Marker.CYLINDER, life=1)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), 0.85
            m.scale.x = m.scale.y = 0.4
            m.scale.z = 1.7
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.1, 0.8, 0.6
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    def report(self):
        x, y, th = self.map_to_odom
        self.get_logger().info(
            f'keyframes {self.graph.n_keyframes}, landmarks {len(self.graph.landmarks)} '
            f'({", ".join(sorted({l.label for l in self.graph.landmarks.values()})) or "none yet"}), '
            f'unconfirmed {len(self.graph.candidates)}, sightings {self.n_dets} | '
            f'map->odom x={x:.3f} y={y:.3f} yaw={math.degrees(th):.2f} deg')

    def save(self):
        if not self.graph.landmarks:
            return
        data = {'frame': 'map', 'landmarks': [
            {'id': lm.id, 'label': lm.label, 'x': round(float(lm.xy[0]), 3),
             'y': round(float(lm.xy[1]), 3), 'size': [round(float(v), 2) for v in lm.size],
             'observations': lm.n_obs} for lm in self.graph.landmarks.values()]}
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, 'w') as f:
                yaml.safe_dump(data, f, sort_keys=False)
        except OSError as err:
            self.get_logger().warn(f'could not save the object map: {err}', throttle_duration_sec=30.0)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectSlam()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
object_detector.py  -  Phase 3, Task 7 (step a: seeing objects)

Finds warehouse objects (pallet, shelving rack, cardboard box, person) in the LEFT camera
picture with a YOLO neural network, then looks up how far away each one is in the stereo
point cloud, so every detection becomes a 3D object:  label + position + size.

Listens to : /stereo/left/image_raw   (the picture)
             /stereo/points2          (stereo point cloud, one 3D point per picture pixel)
Publishes  : /objects/detections      (vision_msgs/Detection3DArray, in base_link)
             /objects/markers         (RViz boxes + text labels)
             /objects/debug_image     (the picture with the YOLO boxes drawn on it)

The network is YOLOv8-World: a pre-trained YOLOv8 that is told WHAT to look for in plain
words ("pallet", "shelving rack", ...) instead of being limited to the 80 classes of the COCO
dataset (which has "person" but no pallet or rack). The words are baked into a model file once
by tools/make_warehouse_model.py, so no extra downloads are needed while the robot runs.
Any other YOLOv8 .pt file (for example one you trained yourself) also works: set the 'model'
parameter.
"""
import os
import time

import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from task_robotics import cloud_ops, object_ops

COLOURS = {   # r, g, b for the RViz boxes
    'pallet': (1.0, 0.6, 0.0),
    'shelving rack': (0.2, 0.5, 1.0),
    'cardboard box': (0.8, 0.5, 0.2),
    'person': (1.0, 0.1, 0.8),
}


class ObjectDetector(Node):

    def __init__(self):
        super().__init__('object_detector')

        self.declare_parameter('model', '/ws/models/warehouse_world.pt')
        self.declare_parameter('conf', 0.15)            # lowest confidence we accept
        self.declare_parameter('imgsz', 480)            # network input size (smaller = faster)
        self.declare_parameter('max_rate', 2.0)         # pictures per second to analyse
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('threads', 2)            # CPU threads for the network
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('roi_shrink', 0.8)       # use the middle 80% of each box
        self.declare_parameter('depth_window', 0.6)     # meters behind the nearest surface to keep
        self.declare_parameter('min_points', 15)
        self.declare_parameter('hfov', 1.3962634)       # camera horizontal field of view (rad)
        self.declare_parameter('use_lidar', True)       # LiDAR gives the distance when stereo has no points
        self.declare_parameter('max_range', 7.0)        # ignore objects further away

        g = self.get_parameter
        self.target = g('target_frame').value
        self.conf = float(g('conf').value)
        self.imgsz = int(g('imgsz').value)
        self.device = g('device').value
        self.shrink = float(g('roi_shrink').value)
        self.depth_window = float(g('depth_window').value)
        self.min_points = int(g('min_points').value)
        self.max_range = float(g('max_range').value)
        self.hfov = float(g('hfov').value)
        self.use_lidar = bool(g('use_lidar').value)
        self.scan = None
        self.period = 1.0 / float(g('max_rate').value)

        self.model = self._load_model(g('model').value, int(g('threads').value))

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=2, history=QoSHistoryPolicy.KEEP_LAST,
                         reliability=QoSReliabilityPolicy.BEST_EFFORT)
        img_sub = message_filters.Subscriber(self, Image, '/stereo/left/image_raw',
                                             qos_profile=qos)
        cloud_sub = message_filters.Subscriber(self, PointCloud2, '/stereo/points2',
                                               qos_profile=qos)
        # picture and cloud carry the same time stamp; allow a tiny difference anyway
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [img_sub, cloud_sub], queue_size=5, slop=0.05)
        self.sync.registerCallback(self.on_pair)

        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.det_pub = self.create_publisher(Detection3DArray, '/objects/detections', 5)
        self.marker_pub = self.create_publisher(MarkerArray, '/objects/markers', 5)
        self.debug_pub = self.create_publisher(Image, '/objects/debug_image', 2)

        self.last_run = 0.0
        self.n_runs = 0
        self.n_dets = 0
        self.ms = 0.0
        self.last_report = time.monotonic()
        self.get_logger().info(
            f'object_detector ready: classes {list(self.model.names.values())}, '
            f'conf>={self.conf}, <= {1.0 / self.period:.1f} pictures/s')

    # ------------------------------------------------------------------
    def _load_model(self, path, threads):
        if not os.path.exists(path):
            raise RuntimeError(
                f'model file {path} not found. Create it once with:\n'
                f'    python3 /ws/tools/make_warehouse_model.py')
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(threads)
        return YOLO(path)

    # ------------------------------------------------------------------
    def on_scan(self, msg):
        self.scan = msg

    def _scan_xy(self):
        """Latest LiDAR hits as N x 2 points in the robot frame (or None)."""
        if not self.use_lidar or self.scan is None:
            return None
        sc = self.scan
        ang = sc.angle_min + np.arange(len(sc.ranges)) * sc.angle_increment
        r = np.asarray(sc.ranges, dtype=np.float64)
        ok = np.isfinite(r) & (r > sc.range_min) & (r < sc.range_max)
        pts = np.stack([r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok]), np.zeros(ok.sum())], axis=1)
        try:
            tf = self.tf_buffer.lookup_transform(self.target, sc.header.frame_id, Time())
        except TransformException:
            return None
        q, t = tf.transform.rotation, tf.transform.translation
        rot = cloud_ops.quaternion_to_matrix(q.x, q.y, q.z, q.w)
        return (pts @ rot.T + np.array([t.x, t.y, t.z]))[:, :2]

    def on_pair(self, img_msg, cloud_msg):
        now = time.monotonic()
        if now - self.last_run < self.period:       # too soon: skip (keeps the CPU free)
            return
        self.last_run = now
        t0 = now

        # picture -> numpy (H x W x 3)
        h, w = img_msg.height, img_msg.width
        raw = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(h, img_msg.step)[:, :w * 3]
        rgb = raw.reshape(h, w, 3)
        if img_msg.encoding == 'bgr8':
            rgb = rgb[:, :, ::-1]
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])      # YOLO (ultralytics) expects BGR

        try:
            tf = self.tf_buffer.lookup_transform(self.target, cloud_msg.header.frame_id, Time())
        except TransformException as err:
            self.get_logger().warn(f'waiting for TF: {err}', throttle_duration_sec=2.0)
            return
        q, t = tf.transform.rotation, tf.transform.translation
        rot = cloud_ops.quaternion_to_matrix(q.x, q.y, q.z, q.w)
        trans = np.array([t.x, t.y, t.z])

        result = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf,
                                    device=self.device, verbose=False)[0]
        names = result.names
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        fields = [(f.name, f.offset, f.datatype) for f in cloud_msg.fields]
        cloud_data = bytes(cloud_msg.data)

        scan_xy = self._scan_xy()
        dets = Detection3DArray()
        dets.header.stamp = img_msg.header.stamp
        dets.header.frame_id = self.target
        markers = MarkerArray()
        drawn = []

        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            label = names[int(cls)]
            region = object_ops.box_to_region(*box, self.shrink, w, h)
            pts = object_ops.region_points(fields, cloud_msg.point_step, cloud_data,
                                           cloud_msg.height, cloud_msg.width, region)
            pts = cloud_ops.transform_points(pts, rot, trans) if len(pts) else pts
            est = object_ops.estimate_object(pts, self.depth_window, self.min_points)
            if est is not None and est[0][0] > self.max_range:
                est = None
            source = 'stereo'
            if est is None and scan_xy is not None:
                # no stereo points on this object: take the distance from the LiDAR instead
                az = object_ops.box_to_azimuths(box[0], box[2], w, self.hfov)
                est = object_ops.lidar_estimate(scan_xy, trans[:2], az, box, (w, h),
                                                self.hfov, trans[2])
                if est is not None and est[0][0] > self.max_range:
                    est = None
                source = 'lidar'
            if est is None:
                drawn.append((box, f'{label} {score:.2f} (no 3D)'))
                continue
            centre, size, n = est

            d = Detection3D()
            d.header = dets.header
            d.id = f'{label}_{i}'
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = label
            hyp.hypothesis.score = float(score)
            d.results.append(hyp)
            d.bbox.center.position.x = float(centre[0])
            d.bbox.center.position.y = float(centre[1])
            d.bbox.center.position.z = float(centre[2])
            d.bbox.center.orientation.w = 1.0
            d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = (float(v) for v in size)
            dets.detections.append(d)
            markers.markers.extend(self._markers(len(dets.detections), label, score,
                                                 centre, size))
            drawn.append((box, f'{label} {score:.2f} @{centre[0]:.1f}m {source}'))

        self.det_pub.publish(dets)
        self.marker_pub.publish(markers if markers.markers else self._clear())
        self._publish_debug(img_msg, rgb, drawn)

        self.n_runs += 1
        self.n_dets += len(dets.detections)
        self.ms += (time.monotonic() - t0) * 1000.0
        self._report()

    # ------------------------------------------------------------------
    def _markers(self, idx, label, score, centre, size):
        r, g, b = COLOURS.get(label, (0.7, 0.7, 0.7))
        box = Marker()
        box.header.frame_id = self.target          # stamp 0 = "use the latest transform"
        box.ns, box.id, box.type, box.action = 'box', idx, Marker.CUBE, Marker.ADD
        box.pose.position.x, box.pose.position.y, box.pose.position.z = (float(v) for v in centre)
        box.pose.orientation.w = 1.0
        box.scale.x, box.scale.y, box.scale.z = (float(v) for v in size)
        box.color.r, box.color.g, box.color.b, box.color.a = r, g, b, 0.45
        box.lifetime.sec = 1
        text = Marker()
        text.header.frame_id = self.target
        text.ns, text.id, text.type, text.action = 'label', idx, Marker.TEXT_VIEW_FACING, Marker.ADD
        text.pose.position.x, text.pose.position.y = float(centre[0]), float(centre[1])
        text.pose.position.z = float(centre[2] + size[2] / 2.0 + 0.15)
        text.pose.orientation.w = 1.0
        text.scale.z = 0.15
        text.color.r = text.color.g = text.color.b = text.color.a = 1.0
        text.text = f'{label} {score:.2f}'
        text.lifetime.sec = 1
        return [box, text]

    def _clear(self):
        m = Marker()
        m.action = Marker.DELETEALL
        arr = MarkerArray()
        arr.markers.append(m)
        return arr

    def _publish_debug(self, img_msg, rgb, drawn):
        if self.debug_pub.get_subscription_count() == 0:
            return
        import cv2
        canvas = np.ascontiguousarray(rgb).copy()
        for box, text in drawn:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(canvas, text, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 0), 1, cv2.LINE_AA)
        out = Image()
        out.header = img_msg.header
        out.height, out.width = canvas.shape[:2]
        out.encoding = 'rgb8'
        out.step = out.width * 3
        out.data = canvas.tobytes()
        self.debug_pub.publish(out)

    def _report(self):
        now = time.monotonic()
        if now - self.last_report < 5.0 or self.n_runs == 0:
            return
        self.get_logger().info(
            f'{self.n_runs} pictures analysed, {self.n_dets / self.n_runs:.1f} objects with 3D '
            f'position per picture, {self.ms / self.n_runs:.0f} ms per picture')
        self.n_runs = self.n_dets = 0
        self.ms = 0.0
        self.last_report = now


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

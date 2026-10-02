"""
semantic_ops.py - the maths behind the semantic costmap layer. Plain Python + numpy, no ROS,
so it can be tested on its own (tools/test_semantic_ops.py).

The idea in plain words
Nav2's costmaps are 2D pictures: every 5 cm cell is "free", "unknown" or "obstacle". The LiDAR map
already gives walls and the front faces of racks. The SEMANTIC map adds what the LiDAR cannot say:
  * "this thing is a shelving rack / pallet / box"  -> a keep-out patch whose size depends on the CLASS
  * "this is a person"                              -> a short-lived keep-out patch that follows them
A keep-out patch is just a disc of points. Nav2 reads those points like extra LiDAR hits.

Why the discs are SMALL
Landmark centres from the object map sit on the FRONT face of the object (Task 7 measured 0.3-0.4 m
toward the robot) and the stereo size is too small. A big disc around such a centre would spill
into free aisle space and wrongly close tight aisles. So the disc is only the class minimum (or the
measured size when that is bigger); the walls and the rest of each rack still come from the LiDAR map.
"""
import math

import numpy as np
import yaml

# Lethal keep-out radius (metres) around a landmark centre, per class. Tunable from the launch file.
DEFAULT_KEEPOUT = {
    'shelving rack': 0.30,
    'pallet': 0.25,
    'cardboard box': 0.20,
}
MAX_RADIUS = 0.80          # never trust a size estimate beyond this


def keepout_radius(label, size_xy, table=None):
    """Radius of the keep-out disc: the class minimum, or half of the measured size if larger."""
    table = DEFAULT_KEEPOUT if table is None else table
    base = table.get(label, 0.20)
    measured = 0.5 * max(float(size_xy[0]), float(size_xy[1]))
    return min(max(base, measured), MAX_RADIUS)


def disc_points(cx, cy, radius, step=0.05, z=0.1):
    """Grid of points (spacing `step`) filling a disc -> array (N, 3). z is a fixed height."""
    n = int(math.ceil(radius / step))
    g = np.arange(-n, n + 1) * step
    gx, gy = np.meshgrid(g, g)
    keep = gx * gx + gy * gy <= radius * radius + 1e-9
    pts = np.zeros((int(keep.sum()), 3), dtype=np.float32)
    pts[:, 0] = cx + gx[keep]
    pts[:, 1] = cy + gy[keep]
    pts[:, 2] = z
    return pts


def landmark_points(landmarks, table=None, step=0.05):
    """landmarks: list of dicts {label, x, y, size}. -> (points (N,3), list of (label, x, y, radius))."""
    chunks, discs = [], []
    for lm in landmarks:
        r = keepout_radius(lm['label'], lm.get('size', (0.0, 0.0, 0.0)), table)
        chunks.append(disc_points(lm['x'], lm['y'], r, step))
        discs.append((lm['label'], lm['x'], lm['y'], r))
    if not chunks:
        return np.zeros((0, 3), dtype=np.float32), discs
    return np.vstack(chunks), discs


def load_landmarks_yaml(path):
    """Read the file saved by object_slam (maps/semantic_map.yaml) -> list of dicts."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    out = []
    for lm in data.get('landmarks', []):
        out.append({'label': str(lm['label']), 'x': float(lm['x']), 'y': float(lm['y']),
                    'size': tuple(float(v) for v in lm.get('size', (0.0, 0.0, 0.0)))})
    return out


def yaw_from_quaternion(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def transform_xy(x, y, tx, ty, yaw):
    """Point (x, y) given in a child frame -> parent frame, where the child sits at (tx, ty, yaw)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return tx + c * x - s * y, ty + s * x + c * y


def pack_xyz(points):
    """(N,3) float32 -> bytes for a PointCloud2 with fields x y z (12 bytes per point)."""
    return np.ascontiguousarray(points, dtype='<f4').tobytes()

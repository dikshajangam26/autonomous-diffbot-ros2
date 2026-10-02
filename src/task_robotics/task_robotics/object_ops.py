"""
object_ops.py - the geometry used by object_detector.py (plain numpy, testable without ROS).

The detector gives a 2D box on the picture. This file turns "a box on the picture" into
"an object at x, y, z in the robot's frame, about this big":

    box_to_region     shrink the box to its middle part (the edges often show the background)
    region_points     read the 3D points of the stereo cloud that sit under that part of the picture
    estimate_object   clean those points and measure the object's centre and size
"""
import numpy as np

FLOAT32 = 7


def box_to_region(x1, y1, x2, y2, shrink, width, height):
    """Pixel box -> (u0, u1, v0, v1) of its middle part, kept inside the picture.

    shrink = 0.5 keeps the middle 50% in each direction.
    """
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = (x2 - x1) / 2.0 * shrink, (y2 - y1) / 2.0 * shrink
    u0 = int(max(0, np.floor(cx - hw)))
    u1 = int(min(width, np.ceil(cx + hw) + 1))
    v0 = int(max(0, np.floor(cy - hh)))
    v1 = int(min(height, np.ceil(cy + hh) + 1))
    return u0, u1, v0, v1


def region_points(fields, point_step, data, height, width, region):
    """3D points (N x 3, camera frame) under a pixel region of an ORGANIZED point cloud.

    An organized cloud keeps one point per picture pixel, row by row, so the point under
    picture pixel (u, v) is cloud[v, u]. Points the stereo matcher could not find are NaN
    and are dropped here.
    """
    if height <= 1:
        raise ValueError('point cloud is not organized (height == 1): cannot map pixels to points')
    by_name = {name: offset for name, offset, dtype in fields}
    dt = np.dtype({'names': ['x', 'y', 'z'], 'formats': ['<f4'] * 3,
                   'offsets': [by_name['x'], by_name['y'], by_name['z']],
                   'itemsize': point_step})
    cloud = np.frombuffer(data, dtype=dt).reshape(height, width)
    u0, u1, v0, v1 = region
    part = cloud[v0:v1, u0:u1]
    xyz = np.stack([part['x'], part['y'], part['z']], axis=-1).reshape(-1, 3)
    return xyz[np.isfinite(xyz).all(axis=1)]


def estimate_object(points, depth_window=0.6, min_points=25, z_limits=(0.02, 2.0)):
    """points = N x 3 in the robot frame (x forward, y left, z up).

    1. drop floor / ceiling points (z outside z_limits)
    2. keep only points near the NEAREST surface in the box (20th percentile of x): an object
       stands in front of whatever is behind it, so this throws away the wall that is visible
       behind or beside the object
    3. centre = middle (median) of what is left; size = spread (5th..95th percentile)

    Returns (centre[3], size[3], n_points) or None when there is too little data.
    Note the cameras only see the FRONT of an object, so the size along x (depth) is a minimum.
    """
    if len(points) == 0:
        return None
    z = points[:, 2]
    pts = points[(z >= z_limits[0]) & (z <= z_limits[1])]
    if len(pts) < min_points:
        return None
    near_x = np.percentile(pts[:, 0], 20)
    pts = pts[(pts[:, 0] >= near_x - 0.3) & (pts[:, 0] <= near_x + depth_window)]
    if len(pts) < min_points:
        return None
    centre = np.median(pts, axis=0)
    lo = np.percentile(pts, 5, axis=0)
    hi = np.percentile(pts, 95, axis=0)
    size = np.maximum(hi - lo, 0.05)
    return centre, size, len(pts)


def box_to_azimuths(x1, x2, width, hfov):
    """Left/right pixel edges of a box -> (az_low, az_high) angles in radians as seen from the
    camera, in the robot convention (0 = straight ahead, positive = to the LEFT)."""
    fx = (width / 2.0) / np.tan(hfov / 2.0)
    cx = width / 2.0
    a1 = -np.arctan((x1 - cx) / fx)
    a2 = -np.arctan((x2 - cx) / fx)
    return min(a1, a2), max(a1, a2)


def lidar_estimate(scan_xy, cam_xy, az_range, box_px, image_size, hfov, cam_z,
                   near_window=0.5, min_points=3, default_depth=0.3):
    """Position of an object from the 2D LiDAR when the stereo cloud had no points there.

    scan_xy  : N x 2 LiDAR hits in the robot frame (x forward, y left)
    cam_xy   : camera position in the robot frame
    az_range : (low, high) bearing of the detection box, as seen from the camera
    box_px   : (x1, y1, x2, y2) of the box in the picture

    Takes the LiDAR hits that lie inside the box's angular span, keeps the nearest surface
    (an object stands in front of the wall behind it) and measures it.
    Returns (centre[3], size[3], n_points) or None. The LiDAR sees one thin slice, so height
    comes from the picture box and distance: height = box height in pixels * range / focal length.
    """
    if len(scan_xy) == 0:
        return None
    d = scan_xy - np.asarray(cam_xy)[None, :]
    bearing = np.arctan2(d[:, 1], d[:, 0])
    rng = np.hypot(d[:, 0], d[:, 1])
    inside = (bearing >= az_range[0]) & (bearing <= az_range[1])
    pts, rng = scan_xy[inside], rng[inside]
    if len(pts) < min_points:
        return None
    near = np.percentile(rng, 20)
    keep = rng <= near + near_window
    pts = pts[keep]
    if len(pts) < min_points:
        return None
    cx_, cy_ = np.median(pts[:, 0]), np.median(pts[:, 1])
    width = max(float(np.ptp(pts[:, 1])), 0.1)
    depth = max(float(np.ptp(pts[:, 0])), default_depth)

    w, h = image_size
    fx = (w / 2.0) / np.tan(hfov / 2.0)
    x1, y1, x2, y2 = box_px
    r = float(np.median(rng[keep]))
    height = float(np.clip((y2 - y1) * r / fx, 0.1, 2.5))
    z_centre = cam_z - ((y1 + y2) / 2.0 - h / 2.0) * r / fx
    return (np.array([cx_, cy_, z_centre]), np.array([depth, width, height]), len(pts))

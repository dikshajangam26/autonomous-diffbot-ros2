"""
cloud_ops.py - the point-cloud maths used by cloud_filter.py (plain numpy, no ROS needed,
so it can be tested on its own).

Four small steps, in the order the node uses them:
    unpack_cloud     bytes of a PointCloud2  ->  N x 3 positions (+ N x 3 colours)
    transform_points move the points from the camera frame into the robot frame
    passthrough      keep only points inside a box (this removes floor / ceiling / far points)
    voxel_downsample many points -> one average point per small cube (voxel)
    pack_cloud       positions + colours -> bytes of a new PointCloud2
"""
import numpy as np

FLOAT32 = 7   # sensor_msgs/PointField.FLOAT32


# ----------------------------------------------------------------------
def unpack_cloud(fields, point_step, data):
    """fields = [(name, offset, datatype), ...]. Returns (xyz float32 Nx3, rgb uint8 Nx3 or None).

    Points that are NaN / infinite (the stereo matcher puts those where it found no match)
    are dropped here.
    """
    by_name = {name: (offset, dtype) for name, offset, dtype in fields}
    for axis in ('x', 'y', 'z'):
        if axis not in by_name or by_name[axis][1] != FLOAT32:
            raise ValueError(f"cloud has no float32 '{axis}' field")

    names = ['x', 'y', 'z']
    formats = ['<f4', '<f4', '<f4']
    offsets = [by_name['x'][0], by_name['y'][0], by_name['z'][0]]
    has_rgb = 'rgb' in by_name and by_name['rgb'][1] == FLOAT32
    if has_rgb:
        names.append('rgb')
        formats.append('<u4')      # the colour float is really 3 bytes packed into 32 bits
        offsets.append(by_name['rgb'][0])
    dt = np.dtype({'names': names, 'formats': formats, 'offsets': offsets,
                   'itemsize': point_step})

    arr = np.frombuffer(data, dtype=dt)
    xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=1)
    good = np.isfinite(xyz).all(axis=1)
    xyz = xyz[good]
    rgb = None
    if has_rgb:
        packed = arr['rgb'][good]
        rgb = np.stack([(packed >> 16) & 255, (packed >> 8) & 255, packed & 255],
                       axis=1).astype(np.uint8)
    return xyz, rgb


# ----------------------------------------------------------------------
def quaternion_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])


def transform_points(xyz, rotation, translation):
    """Apply 'p_new = R * p + t' to every row of xyz."""
    return (xyz @ rotation.T + translation).astype(np.float32)


# ----------------------------------------------------------------------
def passthrough_masks(xyz, limits):
    """limits = {'x': (min, max), 'y': (min, max), 'z': (min, max)}.

    Returns (keep, below_z, above_z): which points stay, and how many fell under / over the
    z limits (just for the statistics; z is "height above the floor" in the robot frame).
    """
    keep = np.ones(len(xyz), dtype=bool)
    for i, axis in enumerate('xyz'):
        lo, hi = limits[axis]
        keep &= (xyz[:, i] >= lo) & (xyz[:, i] <= hi)
    return keep


def voxel_downsample(xyz, rgb, leaf):
    """Cut space into cubes of side 'leaf' and replace all points in a cube by their average."""
    if len(xyz) == 0:
        return xyz, rgb
    idx = np.floor(xyz / leaf).astype(np.int64)
    idx -= idx.min(axis=0)                       # all indices >= 0
    dims = idx.max(axis=0) + 1
    key = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]   # one number per cube
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    n = len(counts)

    out_xyz = np.empty((n, 3), dtype=np.float32)
    for k in range(3):
        out_xyz[:, k] = np.bincount(inverse, weights=xyz[:, k], minlength=n) / counts
    out_rgb = None
    if rgb is not None:
        out_rgb = np.empty((n, 3), dtype=np.uint8)
        for k in range(3):
            out_rgb[:, k] = np.bincount(inverse, weights=rgb[:, k], minlength=n) / counts
    return out_xyz, out_rgb


# ----------------------------------------------------------------------
def pack_cloud(xyz, rgb):
    """Positions (+ colours) -> raw bytes (16 bytes per point: x y z rgb)."""
    n = len(xyz)
    dt = np.dtype({'names': ['x', 'y', 'z', 'rgb'],
                   'formats': ['<f4', '<f4', '<f4', '<u4'],
                   'offsets': [0, 4, 8, 12], 'itemsize': 16})
    out = np.zeros(n, dtype=dt)
    out['x'], out['y'], out['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if rgb is None:
        out['rgb'] = 0x00FFFFFF
    else:
        r, g, b = (rgb[:, k].astype(np.uint32) for k in range(3))
        out['rgb'] = (r << 16) | (g << 8) | b
    return out.tobytes()

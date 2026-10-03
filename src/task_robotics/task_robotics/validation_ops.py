"""
validation_ops.py - the scoring maths for the pipeline validation (Task 9). Plain Python + numpy,
no ROS, so it can be tested on its own (tools/test_validation_ops.py).

What it does, in plain words
The Gazebo world file IS the ground truth: it lists every wall, rack, box and pallet with an exact size
and position. So after the robot has mapped the room we can ask three questions with numbers:
  1. Does the SLAM occupancy map show the real surfaces?   (recall = "how much of the truth did we see",
                                                             precision = "how much of what we drew is real")
  2. Are the semantic landmarks on real objects, with the right class?
  3. During a run, how close did the robot get to the walking person? (the person's route is scripted in
     the world file, so its position at any simulation time is known exactly)
"""
import math
import xml.etree.ElementTree as ET

import numpy as np

LIDAR_Z = 0.15            # LiDAR height above the floor (base_link 0.065 + lidar joint 0.085)
ROOM_HALF = 3.95          # inner face of the walls; anything beyond is the OUTSIDE of the walls
KIND_BY_PREFIX = {'wall': 'wall', 'shelf': 'shelving rack', 'box': 'cardboard box', 'pallet': 'pallet'}


def parse_world_objects(path):
    """Every box-shaped collision in the world -> list of dicts (floor/ceiling are skipped)."""
    out = []
    root = ET.parse(path).getroot()
    for model in root.iter('model'):
        name = model.get('name', '')
        prefix = name.split('_')[0]
        if prefix not in KIND_BY_PREFIX:
            continue
        pose = (model.findtext('pose') or '0 0 0 0 0 0').split()
        px, py, pz = (float(v) for v in pose[:3])
        for col in model.iter('collision'):
            size = col.find('./geometry/box/size')
            if size is None:
                continue
            sx, sy, sz = (float(v) for v in size.text.split())
            zlo, zhi = pz - sz / 2, pz + sz / 2
            out.append({'name': name, 'kind': KIND_BY_PREFIX[prefix], 'cx': px, 'cy': py,
                        'sx': sx, 'sy': sy, 'zlo': zlo, 'zhi': zhi,
                        'lidar_visible': zlo < LIDAR_Z < zhi})
    return out


def surface_points(obj, step=0.025, room_half=ROOM_HALF):
    """Points along the outline of a box, keeping only those inside the room (-> the faces a robot can see)."""
    x0, x1 = obj['cx'] - obj['sx'] / 2, obj['cx'] + obj['sx'] / 2
    y0, y1 = obj['cy'] - obj['sy'] / 2, obj['cy'] + obj['sy'] / 2
    xs = np.arange(x0, x1 + 1e-9, step)
    ys = np.arange(y0, y1 + 1e-9, step)
    pts = np.vstack([np.column_stack([xs, np.full_like(xs, y0)]), np.column_stack([xs, np.full_like(xs, y1)]),
                     np.column_stack([np.full_like(ys, x0), ys]), np.column_stack([np.full_like(ys, x1), ys])])
    keep = (np.abs(pts[:, 0]) <= room_half + 1e-9) & (np.abs(pts[:, 1]) <= room_half + 1e-9)
    return pts[keep]


def distance_to_rect(px, py, obj):
    """0 if the point is inside the box footprint, otherwise the distance to its edge."""
    dx = max(abs(px - obj['cx']) - obj['sx'] / 2, 0.0)
    dy = max(abs(py - obj['cy']) - obj['sy'] / 2, 0.0)
    return math.hypot(dx, dy)


def min_distance(a, b, chunk=500):
    """For every point in a (N,2): distance to the nearest point in b (M,2). Chunked brute force."""
    if len(b) == 0:
        return np.full(len(a), np.inf)
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        d = a[i:i + chunk, None, :] - b[None, :, :]
        out[i:i + chunk] = np.sqrt((d * d).sum(axis=2)).min(axis=1)
    return out


def occupied_points(data, width, height, resolution, origin_x, origin_y, threshold=65):
    """OccupancyGrid data -> (N,2) world coordinates of the cell centres that are occupied."""
    grid = np.asarray(data, dtype=np.int16).reshape(height, width)
    iy, ix = np.nonzero(grid >= threshold)
    return np.column_stack([origin_x + (ix + 0.5) * resolution, origin_y + (iy + 0.5) * resolution])


def score_map(objects, map_pts, tol=0.10):
    """Compare the SLAM map with the world.
    recall    per object: share of its visible surface that has a map cell within `tol`
    precision share of map cells that lie within `tol` of ANY true surface
    Pallets (top at LiDAR height) are not expected in the map, so they only count for precision."""
    per = {}
    for o in objects:
        pts = surface_points(o)
        if o['lidar_visible'] and len(pts):
            per[o['name']] = {'kind': o['kind'],
                              'recall': float((min_distance(pts, map_pts) <= tol).mean())}
    walls = [v['recall'] for v in per.values() if v['kind'] == 'wall']
    racks = [v['recall'] for v in per.values() if v['kind'] == 'shelving rack']
    boxes = [v['recall'] for v in per.values() if v['kind'] == 'cardboard box']
    truth_all = np.vstack([surface_points(o) for o in objects]) if objects else np.zeros((0, 2))
    prec = float((min_distance(map_pts, truth_all) <= tol).mean()) if len(map_pts) else 0.0
    mean = lambda v: float(np.mean(v)) if v else float('nan')   # noqa: E731
    return {'per_object': per, 'wall_recall': mean(walls), 'rack_recall': mean(racks),
            'box_recall': mean(boxes), 'precision': prec, 'map_cells': int(len(map_pts))}


def score_landmarks(objects, landmarks, near=0.5, found=0.6):
    """landmarks: list of dicts {label, x, y}. A landmark is POSITION-OK if it lies within `near` metres of a
    real object's footprint (inside it counts as 0). A real object is FOUND when a landmark of the right
    class lies within `found` metres of its footprint."""
    things = [o for o in objects if o['kind'] != 'wall']
    rows = []
    for lm in landmarks:
        best = min(things, key=lambda o: distance_to_rect(lm['x'], lm['y'], o))
        rows.append({'label': lm['label'], 'x': lm['x'], 'y': lm['y'], 'nearest': best['name'],
                     'true_kind': best['kind'], 'footprint_dist': distance_to_rect(lm['x'], lm['y'], best),
                     'centre_dist': math.hypot(lm['x'] - best['cx'], lm['y'] - best['cy']),
                     'class_ok': lm['label'] == best['kind']})
    pos_ok = [r['footprint_dist'] <= near for r in rows]
    cls_ok = [r['class_ok'] for r in rows]
    status = {}
    for o in things:
        status[o['name']] = any(lm['label'] == o['kind'] and distance_to_rect(lm['x'], lm['y'], o) <= found
                                for lm in landmarks)
    racks = [n for n, o in ((o['name'], o) for o in things) if o['kind'] == 'shelving rack']
    return {'rows': rows,
            'position_ok': float(np.mean(pos_ok)) if rows else 0.0,
            'class_ok': float(np.mean(cls_ok)) if rows else 0.0,
            'found': status,
            'racks_found': sum(status[n] for n in racks), 'racks_total': len(racks)}


def parse_actor(path):
    """Walking route of the first <actor> in the world: list of (time, x, y). Empty if there is none."""
    wps = []
    root = ET.parse(path).getroot()
    for actor in root.iter('actor'):
        for wp in actor.iter('waypoint'):
            t = float(wp.findtext('time'))
            pose = wp.findtext('pose').split()
            wps.append((t, float(pose[0]), float(pose[1])))
        break
    return wps


def person_position(waypoints, t):
    """Where the scripted (looping) person is at simulation time t (linear interpolation, as Gazebo does)."""
    if not waypoints:
        return None
    period = waypoints[-1][0]
    t = t % period if period > 0 else 0.0
    for (t0, x0, y0), (t1, x1, y1) in zip(waypoints, waypoints[1:]):
        if t0 <= t <= t1:
            f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return x0 + f * (x1 - x0), y0 + f * (y1 - y0)
    return waypoints[-1][1], waypoints[-1][2]


def yaw_from_quaternion(qx, qy, qz, qw):
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def path_length(points):
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:])))


def summarise(legs):
    """Per-configuration numbers from a list of leg result dicts."""
    ok = [leg for leg in legs if leg['status'] == 'SUCCEEDED']
    n = len(legs)
    seps = [leg['min_sep'] for leg in legs if leg.get('min_sep') is not None]
    return {'legs': n, 'succeeded': len(ok), 'success_rate': len(ok) / n if n else 0.0,
            'mean_time': float(np.mean([leg['time'] for leg in ok])) if ok else float('nan'),
            'mean_path': float(np.mean([leg['path_len'] for leg in ok])) if ok else float('nan'),
            'mean_err_cm': float(np.mean([leg['err_map'] for leg in ok]) * 100) if ok else float('nan'),
            'max_err_cm': float(np.max([leg['err_map'] for leg in ok]) * 100) if ok else float('nan'),
            'min_lidar': float(min(leg['min_range'] for leg in legs)) if legs else float('nan'),
            'min_sep': float(min(seps)) if seps else None}

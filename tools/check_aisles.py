#!/usr/bin/env python3
"""
check_aisles.py - checks the Nav2 footprint / inflation numbers against the real warehouse.world
BEFORE you drive anything (no ROS or Gazebo needed):

    python3 tools/check_aisles.py [path/to/warehouse.world] [path/to/semantic_map.yaml]

It rasterises the world's boxes into a 5 cm grid (what a perfect LiDAR map would show), computes for every
cell the distance to the nearest obstacle, and applies the SAME rule as Nav2's inflation layer:
    distance <= inscribed radius          -> the robot centre cannot be here (cost 253+)
    otherwise cost = 252 * exp(-k * (distance - inscribed radius))   for distance <= inflation radius
Then it asks: can the robot get from the start to the goals? What does it cost in the middle of an aisle?
"""
import heapq
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

RES = 0.05
HALF = 4.3                       # grid covers +-4.3 m
INSCRIBED = 0.12                 # half of the robot's width (wheels): from the footprint
PADDING = 0.01                   # footprint_padding in nav2_params.yaml
GLOBAL_INFL = (0.45, 4.0)        # inflation_radius, cost_scaling_factor of the global costmap
LIDAR_Z = 0.15                   # LiDAR height above the floor (base_link 0.065 + lidar joint 0.085)
HERE = os.path.dirname(os.path.abspath(__file__))


def parse_world(path):
    """-> list of (name, cx, cy, sx, sy, z_lo, z_hi) for every static box that is a collision shape."""
    boxes = []
    root = ET.parse(path).getroot()
    for model in root.iter('model'):
        pose = (model.findtext('pose') or '0 0 0 0 0 0').split()
        px, py, pz = (float(v) for v in pose[:3])
        for col in model.iter('collision'):
            size = col.find('./geometry/box/size')
            if size is None:
                continue
            sx, sy, sz = (float(v) for v in size.text.split())
            boxes.append((model.get('name'), px, py, sx, sy, pz - sz / 2, pz + sz / 2))
    return boxes


def rasterise(boxes):
    n = int(2 * HALF / RES)
    occ = np.zeros((n, n), bool)                      # index [iy, ix]
    xs = (np.arange(n) + 0.5) * RES - HALF
    X, Y = np.meshgrid(xs, xs)
    hidden = []
    for name, cx, cy, sx, sy, zlo, zhi in boxes:
        if zlo > LIDAR_Z or zhi < 0.02:               # ceiling, or nothing at robot level
            continue
        if not (zlo < LIDAR_Z < zhi):                 # e.g. a 15 cm pallet: top is AT the LiDAR plane
            hidden.append(name)
            continue
        occ |= (abs(X - cx) <= sx / 2) & (abs(Y - cy) <= sy / 2)
    return occ, X, Y, hidden


def distance_to_obstacle(occ):
    from scipy import ndimage
    # EDT measures to the CENTRE of the nearest occupied cell, which lies half a cell inside the real
    # surface; subtract half a cell to get the distance to the surface itself (what a LiDAR hit marks).
    return np.maximum(ndimage.distance_transform_edt(~occ) * RES - RES / 2, 0.0)


def cost(dist, radius, k):
    return np.where(dist <= INSCRIBED + PADDING, 253.0,
                    np.where(dist <= radius, 252.0 * np.exp(-k * (dist - INSCRIBED)), 0.0))


def cell(x, y):
    return int((y + HALF) / RES), int((x + HALF) / RES)


def plan(costs, start, goal):
    """A* on the cost grid (8-connected). Cells with cost >= 253 are blocked. -> path length or None."""
    n = costs.shape[0]
    s, g = cell(*start), cell(*goal)
    if costs[s] >= 253 or costs[g] >= 253:
        return None
    best = {s: 0.0}
    heap = [(0.0, 0.0, s)]
    while heap:
        _, gc, cur = heapq.heappop(heap)
        if cur == g:
            return gc * RES
        if gc > best.get(cur, 1e9):
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb = (cur[0] + dy, cur[1] + dx)
                if not (0 <= nb[0] < n and 0 <= nb[1] < n) or costs[nb] >= 253:
                    continue
                ng = gc + math.hypot(dy, dx) * (1.0 + costs[nb] / 252.0 * 0.8)
                if ng < best.get(nb, 1e9):
                    best[nb] = ng
                    heapq.heappush(heap, (ng + math.hypot(nb[0] - g[0], nb[1] - g[1]), ng, nb))
    return None


def main():
    world = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, '..', 'src', 'task_robotics', 'worlds', 'warehouse.world')
    boxes = parse_world(world)
    occ, X, Y, hidden = rasterise(boxes)
    print(f'world: {len(boxes)} collision boxes')
    if hidden:
        print(f'  at or below the LiDAR plane (only the semantic layer can add these): {", ".join(hidden)}')

    if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):      # add semantic keep-out discs
        sys.path.insert(0, os.path.join(HERE, '..', 'src', 'task_robotics'))
        from task_robotics import semantic_ops as so
        lms = so.load_landmarks_yaml(sys.argv[2])
        for lm in lms:
            r = so.keepout_radius(lm['label'], lm['size'])
            occ |= (X - lm['x']) ** 2 + (Y - lm['y']) ** 2 <= r * r
        print(f'  + {len(lms)} semantic keep-out discs from {sys.argv[2]}')

    dist = distance_to_obstacle(occ)
    costs = cost(dist, *GLOBAL_INFL)

    print('\nCost across the east aisle (rack face x=3.25 .. wall x=3.90) at y = 1.6:')
    for x in (3.35, 3.45, 3.575, 3.70, 3.80):
        iy, ix = cell(x, 1.6)
        print(f'  x={x:5.3f}  distance to obstacle {dist[iy, ix]:.2f} m   cost {costs[iy, ix]:5.1f}')

    tests = [
        ('start -> middle of the east aisle behind rack 1 (3.575, 1.6)', (0, 0), (3.575, 1.6), True),
        ('start -> middle of the east aisle behind rack 2 (3.575, -1.6)', (0, 0), (3.575, -1.6), True),
        ('start -> gap behind rack 3 (-1.5, 3.775), only 0.25 m wide', (0, 0), (-1.5, 3.775), False),
        ('start -> between box 1 and rack 2 (2.4, -0.6)', (0, 0), (2.4, -0.6), True),
    ]
    ok = True
    print('\nPlanning checks (A* on the global costmap rule):')
    for label, a, b, expect in tests:
        length = plan(costs, a, b)
        good = (length is not None) == expect
        ok &= good
        res = f'path {length:.2f} m' if length is not None else 'NO PATH'
        print(f"  {'PASS' if good else 'FAIL'}  {label}: {res} (expected {'a path' if expect else 'no path'})")
    print('\nALL PASSED' if ok else '\nSOME FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

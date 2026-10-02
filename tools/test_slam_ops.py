#!/usr/bin/env python3
"""Self-test of the factor graph (no ROS needed):  python3 tools/test_slam_ops.py

A pretend robot drives circles around three known objects. Its odometry drifts, its LiDAR pose is
slightly noisy, the object sightings are noisy and now and then YOLO "sees" something that is not
there. We check that the graph (1) finds exactly the three real objects, (2) puts them in the right
place, (3) ignores the one-off ghosts.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'src', 'task_robotics', 'task_robotics'))
from slam_ops import SemanticGraph  # noqa: E402

rng = np.random.default_rng(1)
TRUTH = {'shelving rack': (3.0, 1.0), 'cardboard box': (-3.0, 3.5), 'pallet': (0.5, -3.0)}

g = SemanticGraph()
odom_x = odom_y = odom_th = 0.0
prev = None
n_ghost = 0
for step in range(600):
    t = step * 0.05
    th = 0.35 * t                                  # drive a circle of radius 2 m, 0.7 m/s
    x, y = 2.0 * math.sin(th), 2.0 * (1 - math.cos(th))
    yaw = th
    # odometry: integrates the true motion but over-reports turning by 5 % (wheel slip) -> drifts
    if prev is not None:
        dx, dy, dth = x - prev[0], y - prev[1], yaw - prev[2]
        c, s = math.cos(prev[2]), math.sin(prev[2])
        fwd, lat = c * dx + s * dy, -s * dx + c * dy
        odom_th += dth * 1.05
        odom_x += math.cos(odom_th) * fwd - math.sin(odom_th) * lat
        odom_y += math.sin(odom_th) * fwd + math.cos(odom_th) * lat
    prev = (x, y, yaw)

    if step % 6 == 0:                              # new keyframe every 0.3 s
        lidar = (x + rng.normal(0, 0.03), y + rng.normal(0, 0.03), yaw + rng.normal(0, 0.01))
        kf = g.add_keyframe(lidar, (odom_x, odom_y, odom_th))
        g.update()
        for label, (lx, ly) in TRUTH.items():
            dx, dy = lx - x, ly - y
            r = math.hypot(dx, dy)
            b = math.atan2(dy, dx) - yaw
            b = math.atan2(math.sin(b), math.cos(b))
            if r < 6.0 and abs(b) < math.radians(40):
                g.observe(kf, label, b + rng.normal(0, 0.04), r + rng.normal(0, 0.1), (0.5, 1.0, 1.0))
        if rng.random() < 0.05:                    # a ghost: one random sighting somewhere
            g.observe(kf, 'pallet', rng.uniform(-0.5, 0.5), rng.uniform(1.5, 5.0), (0.5, 0.5, 0.3))
            n_ghost += 1
        g.update()

print(f'{g.n_keyframes} keyframes, {n_ghost} ghosts injected')
print(f'landmarks in graph: {len(g.landmarks)}   still-unconfirmed candidates: {len(g.candidates)}')
ok = len(g.landmarks) == 3
for lm in g.landmarks.values():
    tx, ty = TRUTH[lm.label]
    err = math.hypot(lm.xy[0] - tx, lm.xy[1] - ty)
    print(f'  #{lm.id} {lm.label:<14} at ({lm.xy[0]:6.2f}, {lm.xy[1]:6.2f})  truth ({tx:5.2f}, {ty:5.2f})  '
          f'error {err:.2f} m, seen from {len(lm.keyframes)} keyframes')
    ok &= err < 0.35
last = g.n_keyframes - 1
mo = g.map_to_odom(last)
print(f'map->odom correction at the end: x={mo.x():.2f} y={mo.y():.2f} yaw={math.degrees(mo.theta()):.1f} deg '
      f'(odometry had drifted to yaw {math.degrees(odom_th - 0.35 * 599 * 0.05):.1f} deg)')
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)

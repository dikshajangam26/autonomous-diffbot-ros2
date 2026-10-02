#!/usr/bin/env python3
"""Self-test for semantic_ops.py (no ROS needed):  python3 tools/test_semantic_ops.py"""
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'task_robotics'))
from task_robotics import semantic_ops as so  # noqa: E402


def check(cond, msg):
    print(('PASS  ' if cond else 'FAIL  ') + msg)
    if not cond:
        check.failed = True


check.failed = False

# 1. radius: class minimum wins for the tiny stereo sizes, measured size wins when larger, capped
check(so.keepout_radius('shelving rack', (0.1, 0.1, 1.1)) == 0.30, 'tiny measured rack -> class minimum 0.30')
check(abs(so.keepout_radius('pallet', (0.9, 0.6, 0.1)) - 0.45) < 1e-9, 'big pallet -> half of measured size')
check(so.keepout_radius('shelving rack', (5.0, 5.0, 1.0)) == so.MAX_RADIUS, 'huge size is capped')
check(so.keepout_radius('mystery', (0, 0, 0)) == 0.20, 'unknown class -> 0.20 default')

# 2. disc: every point inside the radius, centred correctly, no gap on the axes
p = so.disc_points(2.0, -1.0, 0.30)
d = np.hypot(p[:, 0] - 2.0, p[:, 1] + 1.0)
check(d.max() <= 0.30 + 1e-6, 'all disc points within the radius')
check(abs(p[:, 0].mean() - 2.0) < 1e-6 and abs(p[:, 1].mean() + 1.0) < 1e-6, 'disc is centred')
check(len(p) > 100, f'disc has enough points ({len(p)})')

# 3. YAML round trip with the real file format written by object_slam
text = """frame: map
landmarks:
- id: 0
  label: shelving rack
  x: 2.844
  y: 1.2
  size: [0.1, 0.1, 1.14]
  observations: 177
- id: 4
  label: cardboard box
  x: 1.565
  y: -0.207
  size: [0.19, 0.18, 0.14]
  observations: 453
"""
with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
    f.write(text)
lms = so.load_landmarks_yaml(f.name)
os.unlink(f.name)
check(len(lms) == 2 and lms[0]['label'] == 'shelving rack' and abs(lms[1]['y'] + 0.207) < 1e-9, 'semantic_map.yaml loads')
pts, discs = so.landmark_points(lms)
check(len(discs) == 2 and pts.shape[1] == 3 and len(pts) > 0, 'landmarks -> cloud')
check(so.landmark_points([])[0].shape == (0, 3), 'no landmarks -> empty cloud')

# 4. the key property: the east aisle (rack face x=3.25, wall x=3.90) must stay open.
#    Rack landmark was saved at x=2.844 (0.4 m in FRONT of the true centre 3.0).
rack_disc_edge = 2.844 + so.keepout_radius('shelving rack', (0.1, 0.1, 1.1))
check(rack_disc_edge < 3.25, f'rack disc ends at x={rack_disc_edge:.2f}, before the aisle (3.25-3.90)')

# 5. frame maths
x, y = so.transform_xy(1.0, 0.0, 2.0, 3.0, math.pi / 2)
check(abs(x - 2.0) < 1e-9 and abs(y - 4.0) < 1e-9, 'transform_xy rotates then shifts')
check(abs(so.yaw_from_quaternion(0, 0, math.sin(0.4), math.cos(0.4)) - 0.8) < 1e-9, 'quaternion -> yaw')
check(len(so.pack_xyz(pts)) == 12 * len(pts), 'packed cloud is 12 bytes/point')

print('\nALL PASSED' if not check.failed else '\nSOME FAILED')
sys.exit(1 if check.failed else 0)

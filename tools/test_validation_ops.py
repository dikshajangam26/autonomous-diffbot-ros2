#!/usr/bin/env python3
"""Self-test for validation_ops.py (no ROS needed):  python3 tools/test_validation_ops.py [warehouse_people.world]"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'task_robotics'))
from task_robotics import validation_ops as vo  # noqa: E402

WORLD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, '..', 'src', 'task_robotics', 'worlds', 'warehouse_people.world')


def check(cond, msg):
    print(('PASS  ' if cond else 'FAIL  ') + msg)
    if not cond:
        check.failed = True


check.failed = False

objs = vo.parse_world_objects(WORLD)
kinds = {}
for o in objs:
    kinds[o['kind']] = kinds.get(o['kind'], 0) + 1
check(kinds == {'wall': 4, 'shelving rack': 3, 'cardboard box': 5, 'pallet': 2}, f'world parsed: {kinds}')
check({o['name'] for o in objs if o['lidar_visible']}.isdisjoint({'pallet_1', 'pallet_2'}),
      'pallets (15 cm) are not LiDAR-visible')

# 1. a perfect map scores 1.0 recall / 1.0 precision
truth = np.vstack([vo.surface_points(o) for o in objs if o['lidar_visible']])
sc = vo.score_map(objs, truth)
check(sc['wall_recall'] > 0.99 and sc['rack_recall'] > 0.99 and sc['precision'] > 0.99, 'perfect map -> 1.0 / 1.0')
# 2. delete shelf_1 from the map -> its recall drops to ~0, the others stay
shelf1 = vo.surface_points(next(o for o in objs if o['name'] == 'shelf_1'))
mp = truth[vo.min_distance(truth, shelf1) > 0.2]
sc = vo.score_map(objs, mp)
check(sc['per_object']['shelf_1']['recall'] < 0.1 and sc['per_object']['shelf_2']['recall'] > 0.99, 'missing rack detected')
# 3. clutter lowers precision
junk = np.array([[0.0, 0.0], [0.5, 0.5], [-1.0, 0.2], [1.0, -0.5]])
sc = vo.score_map(objs, np.vstack([truth, junk]))
check(0.9 < sc['precision'] < 1.0, f"4 false cells lower precision ({sc['precision']:.4f})")
# 4. occupied_points converts grid indices correctly
pts = vo.occupied_points([0, 100, 0, 0], 2, 2, 0.05, -1.0, 2.0)
check(len(pts) == 1 and abs(pts[0][0] + 0.925) < 1e-9 and abs(pts[0][1] - 2.025) < 1e-9, 'grid cell -> world coordinates')

# 5. the 5 landmarks saved in Task 7 against the real world
lms = [{'label': 'shelving rack', 'x': 2.844, 'y': 1.2}, {'label': 'shelving rack', 'x': -1.73, 'y': 3.148},
       {'label': 'cardboard box', 'x': -0.278, 'y': 2.457}, {'label': 'cardboard box', 'x': -1.706, 'y': 1.512},
       {'label': 'cardboard box', 'x': 1.565, 'y': -0.207}]
ls = vo.score_landmarks(objs, lms)
names = [r['nearest'] for r in ls['rows']]
check(names == ['shelf_1', 'shelf_3', 'box_4', 'box_2', 'box_1'], f'landmarks matched to {names}')
check(ls['position_ok'] == 1.0 and ls['class_ok'] == 1.0, 'all 5 inside the real footprints with the right class')
check(ls['racks_found'] == 2 and ls['racks_total'] == 3, 'racks found 2 of 3 (shelf_2 never mapped)')
check(abs(ls['rows'][0]['centre_dist'] - math.hypot(0.156, 0.4)) < 1e-6, 'centre error of landmark 0 is 43 cm')

# 6. the walking person
wps = vo.parse_actor(WORLD)
check(len(wps) == 5 and abs(wps[-1][0] - 8.67) < 1e-9, 'actor route parsed')
x, y = vo.person_position(wps, 0.0)
check(abs(x) < 1e-9 and abs(y + 1.3) < 1e-9, 'person starts at (0, -1.3)')
x, y = vo.person_position(wps, 3.67)
check(abs(x - 2.2) < 1e-9 and abs(y + 1.3) < 1e-9, 'person reaches (2.2, -1.3) at 3.67 s')
x, y = vo.person_position(wps, 8.67 + 1.835)
check(abs(x - 1.1) < 1e-3 and abs(y + 1.3) < 1e-9, 'route loops (t + one period)')
check(vo.person_position([], 1.0) is None, 'no actor -> None')

# 7. summary
legs = [{'status': 'SUCCEEDED', 'time': 10.0, 'path_len': 2.0, 'err_map': 0.08, 'min_range': 0.3, 'min_sep': 0.9},
        {'status': 'ABORTED', 'time': 5.0, 'path_len': 1.0, 'err_map': 0.9, 'min_range': 0.2, 'min_sep': 0.5}]
s = vo.summarise(legs)
check(s['success_rate'] == 0.5 and abs(s['mean_err_cm'] - 8.0) < 1e-9 and s['min_sep'] == 0.5, 'summary numbers')

print('\nALL PASSED' if not check.failed else '\nSOME FAILED')
sys.exit(1 if check.failed else 0)

#!/usr/bin/env python3
"""
validate_pipeline.py - Task 9: validates the WHOLE pipeline inside Gazebo, automatically, and writes a report.

Run it with the simulation and Nav2 already running (see the instructions in the chat):

    python3 tools/validate_pipeline.py                       # everything
    python3 tools/validate_pipeline.py --skip-tour           # you already mapped; only score + missions
    python3 tools/validate_pipeline.py --configs astar_dwb   # only one planner combination (faster)
    python3 tools/validate_pipeline.py --no-dynamic          # sim was started without people:=true

What it does, in order
  1. MAPPING TOUR   drives the robot through 11 waypoints (down both aisles behind the racks, around the boxes)
                    so SLAM and the object map see the whole warehouse.
  2. MAP SCORE      compares the SLAM occupancy map with the true surfaces in warehouse.world
                    (walls, racks, boxes: recall; share of the drawn cells that are real: precision).
  3. LANDMARK SCORE compares the semantic object map with the true objects (position and class).
  4. GLOBAL PLANNERS asks Nav2 for paths with A* and with Dijkstra between the mission waypoints
                    (no driving): path length and planning time.
  5. MISSIONS       drives 7 waypoints for each planner combination:
                       astar_dwb      A* + DWB          dijkstra_dwb   Dijkstra + DWB        astar_mppi   A* + MPPI
  6. DYNAMIC TEST   crosses the route of the walking person 4 times per combination and records the
                    smallest distance to the person (the person's route is scripted in the world file,
                    so its true position at every simulated second is known).
  7. REPORT         prints a table and saves reports/validation_<time>.json and .md
Targets are listed in TARGETS below (the person is a 0.25 m radius body, the robot 0.12-0.18 m, so closer than ~0.43 m can be contact); the report says PASS or FAIL against each, with the real numbers.
"""
import argparse
import datetime
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray

from task_robotics import validation_ops as vo

TOUR = [(1.0, 1.2), (3.575, 1.6), (3.575, -1.6), (2.0, -1.9), (0.0, -1.6), (-0.8, -1.2),
        (-1.8, 0.0), (-0.8, 1.0), (-1.2, 2.5), (0.0, 1.3), (0.0, 0.0)]
MISSION = [(1.0, 1.2), (3.575, 1.6), (3.575, -1.6), (2.0, -1.9), (-0.8, -1.2), (-1.2, 2.5), (0.0, 0.0)]
DYNAMIC = [(2.4, -0.5), (-0.8, -1.2), (2.4, -0.5), (-0.8, -1.2)]      # crosses the worker's loop (y -1.3..-0.9)
DYNAMIC_START = (-0.8, -1.2)

CONFIGS = {
    'astar_dwb': ('warehouse_nav.xml', 'A* + DWB'),
    'dijkstra_dwb': ('warehouse_nav_dijkstra.xml', 'Dijkstra + DWB'),
    'astar_mppi': ('warehouse_nav_mppi.xml', 'A* + MPPI'),
}

TARGETS = {                      # (description, threshold)
    'wall_recall': ('SLAM map: share of wall surface that is mapped', 0.85),
    'rack_recall': ('SLAM map: average share of rack surface that is mapped', 0.40),
    'precision': ('SLAM map: share of drawn cells that are real surface', 0.85),
    'landmark_position': ('Semantic map: landmarks within 0.5 m of a real object', 0.75),
    'racks_found': ('Semantic map: racks found (of 3), right class, within 0.6 m', 2),
    'success': ('Missions: legs that reached the goal', 0.90),
    'goal_error_cm': ('Missions: worst final error (cm, map frame)', 15.0),
    'clearance': ('Missions: closest LiDAR return (m) - robot half-width is 0.12', 0.16),
    'dyn_success': ('Dynamic: legs that reached the goal', 0.75),
    'separation': ('Dynamic: closest distance to the walking person (m) - touching is 0.37-0.43', 0.45),
}


class Validator(Node):
    def __init__(self, leg_timeout, actor, diagnose=True):
        super().__init__('validate_pipeline', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.leg_timeout = leg_timeout
        self.diagnose = diagnose
        self.actor = actor
        self.tf = Buffer()
        self.tl = TransformListener(self.tf, self)
        self.map_msg = None
        self.landmarks = []
        self.truth = None
        self.worker = None
        self.cmd = (0.0, 0.0)
        self.min_range = 99.0
        self.tracking = False
        self.path_pts = []
        self.min_sep = 99.0
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, qos)
        self.create_subscription(Detection3DArray, '/semantic_map/landmarks', self.on_landmarks, 5)
        self.create_subscription(Odometry, '/ground_truth/odom', self.on_truth, 20)
        self.create_subscription(Odometry, '/worker/ground_truth', self.on_worker, 20)
        self.gcost = None
        self.plan_msg = None
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.on_gcost, qos)
        self.create_subscription(Path, '/plan', self.on_plan, 5)
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, 10)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.plan = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')

    # ---------------------------------------------------------------- callbacks
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_map(self, m):
        self.map_msg = m

    def on_landmarks(self, m):
        self.landmarks = [{'label': d.results[0].hypothesis.class_id, 'x': d.bbox.center.position.x,
                           'y': d.bbox.center.position.y} for d in m.detections if d.results]

    def on_gcost(self, m):
        self.gcost = m

    def on_plan(self, m):
        self.plan_msg = m

    def on_worker(self, m):
        self.worker = (m.pose.pose.position.x, m.pose.pose.position.y)

    def on_cmd(self, m):
        self.cmd = (m.linear.x, m.angular.z)

    def person_xy(self):
        """True position of the walking person: the collision body's ground truth, else the scripted route."""
        if self.worker is not None:
            return self.worker
        return vo.person_position(self.actor, self.now()) if self.actor else None

    def on_truth(self, m):
        p = m.pose.pose.position
        self.truth = (p.x, p.y)
        if self.tracking:
            if not self.path_pts or math.hypot(p.x - self.path_pts[-1][0], p.y - self.path_pts[-1][1]) > 0.01:
                self.path_pts.append((p.x, p.y))
            if self.actor:
                person = self.person_xy()
                if person:
                    self.min_sep = min(self.min_sep, math.hypot(p.x - person[0], p.y - person[1]))

    def on_scan(self, m):
        good = [r for r in m.ranges if 0.13 < r < m.range_max and math.isfinite(r)]   # skip the robot's own body
        if good and self.tracking:
            self.min_range = min(self.min_range, min(good))

    def map_pose(self):
        try:
            t = self.tf.lookup_transform('map', 'base_link', Time()).transform.translation
            return t.x, t.y
        except TransformException:
            return None

    def spin_for(self, seconds):
        end = self.now() + seconds
        while rclpy.ok() and self.now() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for(self, cond, what, timeout=60.0):
        t0 = time.time()
        while rclpy.ok() and not cond():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - t0 > timeout:
                print(f'FAIL: timed out waiting for {what}')
                return False
        return True

    # ---------------------------------------------------------------- one navigation leg
    def go(self, x, y, yaw=0.0, bt=None, label=''):
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        if bt:
            goal.behavior_tree = bt
        self.min_range, self.min_sep = 99.0, 99.0
        self.path_pts = [self.truth] if self.truth else []
        self.tracking = True
        t0 = self.now()
        fut = self.nav.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        handle = fut.result()
        if handle is None or not handle.accepted:
            self.tracking = False
            return self._result(label, x, y, 'REJECTED', t0)
        res = handle.get_result_async()
        status = 'TIMEOUT'
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if res.done():
                code = res.result().status
                status = {GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED', GoalStatus.STATUS_ABORTED: 'ABORTED',
                          GoalStatus.STATUS_CANCELED: 'CANCELED'}.get(code, str(code))
                break
            if self.now() - t0 > self.leg_timeout:
                handle.cancel_goal_async()
                end = time.time() + 10
                while not res.done() and time.time() < end:
                    rclpy.spin_once(self, timeout_sec=0.05)
                break
        self.tracking = False
        r = self._result(label, x, y, status, t0)
        if status != 'SUCCEEDED' and self.diagnose:
            self.dump_diag(x, y)
        self.spin_for(0.5)
        return r

    def dump_diag(self, gx, gy):
        """Printed when a leg fails: what the global costmap and the planner look like around the robot."""
        mp = self.map_pose()
        g = self.gcost
        print('    ---- stuck diagnosis (global costmap 4 m x 4 m around the robot) ----')
        if g is None or mp is None:
            print('    (no global costmap / pose received)')
            return
        res, ox, oy = g.info.resolution, g.info.origin.position.x, g.info.origin.position.y
        grid = np.asarray(g.data, dtype=np.int16).reshape(g.info.height, g.info.width)
        path = set()
        if self.plan_msg:
            path = {(int((q.pose.position.x - ox) / res), int((q.pose.position.y - oy) / res))
                    for q in self.plan_msg.poses}
        cx, cy = int((mp[0] - ox) / res), int((mp[1] - oy) / res)
        gcx, gcy = int((gx - ox) / res), int((gy - oy) / res)
        half = int(2.0 / res)
        near = [grid[y, x] for y in range(max(cy - 4, 0), min(cy + 5, grid.shape[0]))
                for x in range(max(cx - 4, 0), min(cx + 5, grid.shape[1]))]
        print(f'    robot map=({mp[0]:.2f},{mp[1]:.2f})  cost at robot {grid[cy, cx]}  max within 0.2 m {max(near)}  '
              f'(99/100 = blocked)   plan has {len(path)} cells   legend: R robot, G goal, * plan, # blocked, '
              'o high cost, . low cost, ? unknown')
        for iy in range(cy + half, cy - half - 1, -2):
            row = ''
            for ix in range(cx - half, cx + half + 1):
                if not (0 <= iy < grid.shape[0] and 0 <= ix < grid.shape[1]):
                    row += ' '
                elif abs(ix - cx) <= 1 and abs(iy - cy) <= 1:
                    row += 'R'
                elif abs(ix - gcx) <= 1 and abs(iy - gcy) <= 1:
                    row += 'G'
                elif (ix, iy) in path or (ix, iy + 1) in path:
                    row += '*'
                else:
                    c = int(grid[iy, ix])
                    row += '?' if c < 0 else ' ' if c == 0 else '.' if c < 50 else 'o' if c < 99 else '#'
            print(f'    y={oy + iy * res:5.2f} |{row}|')
        print(f'    x from {ox + (cx - half) * res:.2f} to {ox + (cx + half) * res:.2f}')

    def _result(self, label, x, y, status, t0):
        mp = self.map_pose()
        err_map = math.hypot(x - mp[0], y - mp[1]) if mp else float('nan')
        err_true = math.hypot(x - self.truth[0], y - self.truth[1]) if self.truth else float('nan')
        r = {'label': label, 'goal': [x, y], 'status': status, 'time': self.now() - t0,
             'path_len': vo.path_length(self.path_pts), 'err_map': err_map, 'err_truth': err_true,
             'min_range': self.min_range, 'min_sep': self.min_sep if self.actor and self.min_sep < 99 else None}
        sep = f"  sep={r['min_sep']:.2f}" if r['min_sep'] is not None else ''
        if status != 'SUCCEEDED':
            sep += f"  [last cmd v={self.cmd[0]:.2f} w={self.cmd[1]:.2f}, robot at "
            sep += f"({mp[0]:.2f},{mp[1]:.2f})]" if mp else 'unknown]'
        print(f"  {label:<22} -> ({x:5.2f},{y:5.2f})  {status:<9} {r['time']:5.1f}s  path {r['path_len']:4.1f}m  "
              f"err {err_map * 100:5.1f}cm  closest {self.min_range:4.2f}m{sep}")
        return r

    # ---------------------------------------------------------------- planner only
    def plan_only(self, start, goal, planner_id):
        g = ComputePathToPose.Goal()
        g.planner_id = planner_id
        g.use_start = True
        for pose, (x, y) in ((g.start, start), (g.goal, goal)):
            pose.header.frame_id = 'map'
            pose.pose.position.x, pose.pose.position.y = float(x), float(y)
            pose.pose.orientation.w = 1.0
        t_start = time.perf_counter()
        fut = self.plan.send_goal_async(g)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        h = fut.result()
        if h is None or not h.accepted:
            return None
        res = h.get_result_async()
        rclpy.spin_until_future_complete(self, res, timeout_sec=15.0)
        if not res.done() or res.result().status != GoalStatus.STATUS_SUCCEEDED:
            return None
        r = res.result().result
        pts = [(p.pose.position.x, p.pose.position.y) for p in r.path.poses]
        return {'length': vo.path_length(pts), 'ms': (time.perf_counter() - t_start) * 1e3}


def fmt(v, spec='.2f'):
    return 'n/a' if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-tour', action='store_true')
    ap.add_argument('--no-dynamic', action='store_true')
    ap.add_argument('--no-diagnose', action='store_true', help='do not print the costmap picture when a leg fails')
    ap.add_argument('--configs', default=','.join(CONFIGS), help='comma separated: ' + ', '.join(CONFIGS))
    ap.add_argument('--leg-timeout', type=float, default=90.0, help='simulated seconds per leg')
    ap.add_argument('--world', default=None, help='warehouse.world (default: the installed one)')
    ap.add_argument('--out', default='/ws/reports')
    a = ap.parse_args()

    share = get_package_share_directory('task_robotics')
    world_dir = os.path.join(share, 'worlds')
    world = a.world or os.path.join(world_dir, 'warehouse.world')
    objects = vo.parse_world_objects(world)
    actor = [] if a.no_dynamic else vo.parse_actor(os.path.join(world_dir, 'warehouse_people.world'))
    bt_dir = os.path.join(share, 'behavior_trees')
    configs = [c for c in a.configs.split(',') if c in CONFIGS]

    rclpy.init()
    v = Validator(a.leg_timeout, actor, diagnose=not a.no_diagnose)
    print('waiting for Nav2, the map and the clock ...')
    if not v.nav.wait_for_server(timeout_sec=30.0) or not v.plan.wait_for_server(timeout_sec=30.0):
        print('FAIL: Nav2 action servers not available - is nav2.launch.py running and "active"?')
        return 1
    if not v.wait_for(lambda: v.map_msg is not None and v.truth is not None and v.map_pose() is not None,
                      '/map, /ground_truth/odom and the map->base_link transform'):
        return 1
    if actor:
        v.spin_for(2.0)
        print('walking person: ' + ('collision body found on /worker/ground_truth' if v.worker is not None else
              'no /worker/ground_truth (old world?) - falling back to the scripted route'))
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    report = {'time': stamp, 'configs': configs}
    t_wall = time.time()

    # 1 -------------------------------------------------------------------------------------- mapping tour
    if not a.skip_tour:
        print('\n[1/6] MAPPING TOUR - SLAM and the object map learn the warehouse')
        report['tour'] = [v.go(x, y, label=f'tour {i + 1}/{len(TOUR)}') for i, (x, y) in enumerate(TOUR)]
        v.spin_for(8.0)                                    # let slam_toolbox publish the final map
    else:
        print('\n[1/6] mapping tour skipped')

    # 2 + 3 ------------------------------------------------------------------------------------ map scores
    print('\n[2/6] SLAM MAP vs the real world')
    m = v.map_msg
    pts = vo.occupied_points(m.data, m.info.width, m.info.height, m.info.resolution,
                             m.info.origin.position.x, m.info.origin.position.y)
    ms = vo.score_map(objects, pts)
    report['map'] = ms
    for name, d in sorted(ms['per_object'].items()):
        print(f"  {name:<9} ({d['kind']:<13}) surface mapped: {d['recall'] * 100:5.1f} %")
    print(f"  walls {ms['wall_recall'] * 100:.1f} %   racks {ms['rack_recall'] * 100:.1f} %   boxes "
          f"{ms['box_recall'] * 100:.1f} %   precision {ms['precision'] * 100:.1f} %   ({ms['map_cells']} occupied cells)")

    print('\n[3/6] SEMANTIC OBJECT MAP vs the real world')
    ls = vo.score_landmarks(objects, v.landmarks)
    report['landmarks'] = ls
    for r in ls['rows']:
        print(f"  {r['label']:<14} at ({r['x']:5.2f},{r['y']:5.2f}) -> {r['nearest']:<8} footprint distance "
              f"{r['footprint_dist'] * 100:4.0f} cm, centre error {r['centre_dist'] * 100:4.0f} cm, "
              f"class {'OK' if r['class_ok'] else 'WRONG (real: ' + r['true_kind'] + ')'}")
    missing = [n for n, ok in ls['found'].items() if not ok]
    print(f"  objects found: {len(ls['found']) - len(missing)} of {len(ls['found'])}   "
          f"racks {ls['racks_found']} of {ls['racks_total']}   not found: {', '.join(missing) or 'none'}")

    # 4 ------------------------------------------------------------------------------------- global planners
    print('\n[4/6] GLOBAL PLANNERS (path only, no driving; time = request to answer, in wall-clock ms)')
    chain = [(0.0, 0.0)] + MISSION
    rows = {'GridBased': [], 'Dijkstra': []}
    for s, g in zip(chain, chain[1:]):
        for pid in rows:
            r = v.plan_only(s, g, pid)
            if r:
                rows[pid].append(r)
    report['planners'] = {}
    for pid, name in (('GridBased', 'A*'), ('Dijkstra', 'Dijkstra')):
        rr = rows[pid]
        if rr:
            report['planners'][name] = {'paths': len(rr), 'mean_length': float(np.mean([x['length'] for x in rr])),
                                        'mean_ms': float(np.mean([x['ms'] for x in rr]))}
            print(f"  {name:<9} {len(rr)} paths   mean length {np.mean([x['length'] for x in rr]):.2f} m   "
                  f"mean planning time {np.mean([x['ms'] for x in rr]):.1f} ms")
        else:
            print(f'  {name}: no path returned')

    # 5 + 6 ---------------------------------------------------------------------------------------- missions
    report['missions'], report['dynamic'] = {}, {}
    for cfg in configs:
        bt = os.path.join(bt_dir, CONFIGS[cfg][0])
        print(f'\n[5/6] MISSION  {CONFIGS[cfg][1]}')
        legs = [v.go(x, y, bt=bt, label=f'{cfg} wp{i + 1}') for i, (x, y) in enumerate(MISSION)]
        report['missions'][cfg] = {'legs': legs, 'summary': vo.summarise(legs)}
        if actor:
            print(f'[6/6] DYNAMIC OBSTACLE  {CONFIGS[cfg][1]}  (robot crosses the walking person\'s route)')
            v.go(*DYNAMIC_START, bt=bt, label=f'{cfg} reposition')
            legs = [v.go(x, y, bt=bt, label=f'{cfg} cross {i + 1}') for i, (x, y) in enumerate(DYNAMIC)]
            report['dynamic'][cfg] = {'legs': legs, 'summary': vo.summarise(legs)}
    if not actor:
        print('\n[6/6] dynamic test skipped (--no-dynamic)')

    # 7 ------------------------------------------------------------------------------------------------ verdict
    print('\n==================== REPORT ====================')
    checks = []

    def add(key, value, ok, spec='.2f'):
        checks.append({'key': key, 'what': TARGETS[key][0], 'target': TARGETS[key][1],
                       'value': value, 'pass': bool(ok)})
        print(f"  {'PASS' if ok else 'FAIL'}  {TARGETS[key][0]}: {fmt(value, spec)}  (target {TARGETS[key][1]})")

    add('wall_recall', ms['wall_recall'], ms['wall_recall'] >= TARGETS['wall_recall'][1])
    add('rack_recall', ms['rack_recall'], ms['rack_recall'] >= TARGETS['rack_recall'][1])
    add('precision', ms['precision'], ms['precision'] >= TARGETS['precision'][1])
    add('landmark_position', ls['position_ok'], ls['position_ok'] >= TARGETS['landmark_position'][1])
    add('racks_found', ls['racks_found'], ls['racks_found'] >= TARGETS['racks_found'][1], 'd')
    for cfg in configs:
        s_ = report['missions'][cfg]['summary']
        ok = s_['success_rate'] >= TARGETS['success'][1] and s_['max_err_cm'] <= TARGETS['goal_error_cm'][1] \
            and s_['min_lidar'] >= TARGETS['clearance'][1]
        print(f"  {'PASS' if ok else 'FAIL'}  mission {CONFIGS[cfg][1]:<16} {s_['succeeded']}/{s_['legs']} legs, "
              f"mean time {fmt(s_['mean_time'], '.1f')} s, mean path {fmt(s_['mean_path'], '.1f')} m, "
              f"mean/max error {fmt(s_['mean_err_cm'], '.1f')}/{fmt(s_['max_err_cm'], '.1f')} cm, "
              f"closest return {s_['min_lidar']:.2f} m")
    bestc = max(configs, key=lambda c: report['missions'][c]['summary']['success_rate'])
    sm = report['missions'][bestc]['summary']
    print(f'  (scored on the best combination: {CONFIGS[bestc][1]})')
    add('success', sm['success_rate'], sm['success_rate'] >= TARGETS['success'][1])
    add('goal_error_cm', sm['max_err_cm'], sm['max_err_cm'] <= TARGETS['goal_error_cm'][1], '.1f')
    add('clearance', sm['min_lidar'], sm['min_lidar'] >= TARGETS['clearance'][1])
    if actor:
        for cfg in configs:
            s_ = report['dynamic'][cfg]['summary']
            ok = s_['success_rate'] >= TARGETS['dyn_success'][1] and (s_['min_sep'] or 0) >= TARGETS['separation'][1]
            print(f"  {'PASS' if ok else 'FAIL'}  dynamic {CONFIGS[cfg][1]:<16} {s_['succeeded']}/{s_['legs']} legs, "
                  f"closest distance to the person {fmt(s_['min_sep'])} m")
        bestd = max(configs, key=lambda c: (report['dynamic'][c]['summary']['success_rate'],
                                            report['dynamic'][c]['summary']['min_sep'] or 0))
        sd = report['dynamic'][bestd]['summary']
        print(f'  (scored on the best combination: {CONFIGS[bestd][1]})')
        add('dyn_success', sd['success_rate'], sd['success_rate'] >= TARGETS['dyn_success'][1])
        add('separation', sd['min_sep'], (sd['min_sep'] or 0) >= TARGETS['separation'][1])
    overall = all(c['pass'] for c in checks)
    print(f"\n  OVERALL: {'PASS' if overall else 'SOME TARGETS NOT MET'}   "
          f"(wall-clock {int(time.time() - t_wall)} s, simulated time per leg in the table above)")
    report['checks'], report['overall'] = checks, overall

    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, f'validation_{stamp}')
    with open(base + '.json', 'w') as f:
        json.dump(report, f, indent=2, default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o))
    with open(base + '.md', 'w') as f:
        f.write(f'# Pipeline validation {stamp}\n\n| Result | Check | Value | Target |\n|---|---|---|---|\n')
        for c in checks:
            f.write(f"| {'PASS' if c['pass'] else 'FAIL'} | {c['what']} | {fmt(c['value'])} | {c['target']} |\n")
        f.write('\n## Missions\n\n| Planner | Legs OK | Mean time (s) | Mean path (m) | Mean / max error (cm) | Closest return (m) |\n'
                '|---|---|---|---|---|---|\n')
        for cfg in configs:
            s = report['missions'][cfg]['summary']
            f.write(f"| {CONFIGS[cfg][1]} | {s['succeeded']}/{s['legs']} | {fmt(s['mean_time'], '.1f')} | "
                    f"{fmt(s['mean_path'], '.1f')} | {fmt(s['mean_err_cm'], '.1f')} / {fmt(s['max_err_cm'], '.1f')} | "
                    f"{s['min_lidar']:.2f} |\n")
        if actor:
            f.write('\n## Dynamic obstacle (walking person)\n\n| Planner | Legs OK | Closest to person (m) |\n|---|---|---|\n')
            for cfg in configs:
                s = report['dynamic'][cfg]['summary']
                f.write(f"| {CONFIGS[cfg][1]} | {s['succeeded']}/{s['legs']} | {fmt(s['min_sep'])} |\n")
    print(f'\nsaved {base}.json and {base}.md')
    rclpy.shutdown()
    return 0 if overall else 2


if __name__ == '__main__':
    sys.exit(main())

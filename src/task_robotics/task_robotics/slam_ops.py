"""
slam_ops.py - the factor graph behind the semantic (object-based) SLAM. Plain Python + GTSAM, no ROS,
so it can be tested on its own.

What a factor graph is, in plain words
--------------------------------------
Draw a circle (a "variable") for every unknown number we want to find out, and a line (a "factor")
for every piece of evidence about it:

    variables :  X0, X1, X2 ...   where the robot was at each keyframe (x, y, heading in the map)
                 L0, L1 ...       where each object (pallet, rack, box) is (x, y in the map)

    factors   :  LiDAR     "the scan matcher says the robot was HERE at keyframe i"   (on one X)
                 odometry  "from keyframe i-1 to i the robot moved THIS much"          (between two X)
                 object    "from keyframe i, object j was seen at THIS bearing and range"
                                                                                     (between X and L)

Every factor is a little "spring" that pulls the numbers towards what that sensor says. The solver
(iSAM2) finds the robot poses and object positions that satisfy all springs as well as possible,
weighting each spring by how much we trust that sensor (its noise). When the robot sees the same
shelf again later, the objects pull the poses back into line - that is how objects help SLAM.
"""
import math

import gtsam
import numpy as np
from gtsam import symbol_shorthand

X = symbol_shorthand.X
L = symbol_shorthand.L

# How far (m) a new sighting may be from a known object of the SAME class to count as that object.
DEFAULT_GATES = {'shelving rack': 1.2, 'pallet': 0.9, 'cardboard box': 0.7}


class Landmark:
    def __init__(self, lid, label, xy, size):
        self.id = lid
        self.label = label
        self.xy = np.array(xy, dtype=float)
        self.size = np.array(size, dtype=float)
        self.n_obs = 1
        self.keyframes = set()          # keyframes that have an object factor to this landmark


class Candidate:
    """Something seen but not yet trusted: one-off YOLO mistakes never make it into the graph."""

    def __init__(self, label, xy, size, kf, bearing, rng):
        self.label = label
        self.xy_sum = np.array(xy, dtype=float)
        self.size_sum = np.array(size, dtype=float)
        self.n = 1
        self.obs = {kf: (bearing, rng)}  # keyframe -> (bearing, range): one sighting per keyframe
        self.last_kf = kf

    @property
    def xy(self):
        return self.xy_sum / self.n


class SemanticGraph:

    def __init__(self, prior_sigmas=(0.08, 0.08, 0.03), odom_sigmas=(0.08, 0.08, 0.03),
                 bearing_sigma=0.06, range_sigma=0.15, range_rel=0.05,
                 gates=None, default_gate=0.8, min_hits=3, min_keyframes=2, candidate_ttl=25):
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(0.01)
        params.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(params)
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.est = gtsam.Values()

        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(prior_sigmas))
        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(odom_sigmas))
        self.bearing_sigma, self.range_sigma, self.range_rel = bearing_sigma, range_sigma, range_rel
        self.gates = dict(DEFAULT_GATES if gates is None else gates)
        self.default_gate = default_gate
        self.min_hits, self.min_keyframes, self.candidate_ttl = min_hits, min_keyframes, candidate_ttl

        self.kf_odom = []               # odometry pose (odom frame) of every keyframe
        self.kf_label = []
        self.landmarks = {}
        self.candidates = []
        self.next_lid = 0

    # ------------------------------------------------------------------ keyframes
    @property
    def n_keyframes(self):
        return len(self.kf_odom)

    def add_keyframe(self, map_pose, odom_pose):
        """map_pose = (x, y, yaw) from the LiDAR SLAM; odom_pose = (x, y, yaw) from the EKF.
        Adds the variable X_i, the LiDAR factor and (if not the first) the odometry factor."""
        i = self.n_keyframes
        mp = gtsam.Pose2(*map_pose)
        op = gtsam.Pose2(*odom_pose)
        self.values.insert(X(i), mp)
        self.graph.add(gtsam.PriorFactorPose2(X(i), mp, self.prior_noise))        # LiDAR factor
        if i > 0:
            delta = self.kf_odom[i - 1].between(op)                               # odometry factor
            self.graph.add(gtsam.BetweenFactorPose2(X(i - 1), X(i), delta, self.odom_noise))
        self.kf_odom.append(op)
        return i

    def pose(self, i):
        """Current best estimate of keyframe i (a gtsam.Pose2 in the map frame)."""
        if self.est.exists(X(i)):
            return self.est.atPose2(X(i))
        return self.values.atPose2(X(i))

    def map_to_odom(self, i):
        """The SLAM correction: where the odom frame is in the map, given keyframe i's estimate."""
        return self.pose(i).compose(self.kf_odom[i].inverse())

    # ------------------------------------------------------------------ objects
    def _gate(self, label):
        return self.gates.get(label, self.default_gate)

    def _object_noise(self, rng):
        return gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(1.345),
            gtsam.noiseModel.Diagonal.Sigmas(np.array(
                [self.bearing_sigma, self.range_sigma + self.range_rel * rng])))

    def _add_object_factor(self, kf, lid, bearing, rng):
        self.graph.add(gtsam.BearingRangeFactor2D(
            X(kf), L(lid), gtsam.Rot2.fromAngle(bearing), float(rng), self._object_noise(rng)))
        self.landmarks[lid].keyframes.add(kf)

    def observe(self, kf, label, bearing, rng, size):
        """One sighting of an object of class `label`, seen from keyframe kf at this bearing (rad,
        0 = straight ahead, + = left) and range (m). Returns 'landmark', 'candidate' or 'promoted'."""
        size = np.asarray(size, dtype=float)
        p_kf = gtsam.Point2(rng * math.cos(bearing), rng * math.sin(bearing))
        p_map = self.pose(kf).transformFrom(p_kf)
        xy = np.array([p_map[0], p_map[1]])

        # 1) is it a landmark we already trust?
        best, best_d = None, self._gate(label)
        for lm in self.landmarks.values():
            if lm.label != label:
                continue
            d = np.linalg.norm(lm.xy - xy)
            if d < best_d:
                best, best_d = lm, d
        if best is not None:
            best.size = (best.size * best.n_obs + size) / (best.n_obs + 1)
            best.n_obs += 1
            if kf not in best.keyframes:                 # one object factor per (keyframe, object)
                self._add_object_factor(kf, best.id, bearing, rng)
            return 'landmark'

        # 2) otherwise it is (or becomes) a candidate
        cand, best_d = None, self._gate(label)
        for c in self.candidates:
            if c.label != label:
                continue
            d = np.linalg.norm(c.xy - xy)
            if d < best_d:
                cand, best_d = c, d
        if cand is None:
            self.candidates.append(Candidate(label, xy, size, kf, bearing, rng))
            return 'candidate'
        cand.xy_sum += xy
        cand.size_sum += size
        cand.n += 1
        cand.last_kf = kf
        if kf not in cand.obs:
            cand.obs[kf] = (bearing, rng)

        # 3) seen often enough, from enough different places? -> it joins the graph
        if cand.n >= self.min_hits and len(cand.obs) >= self.min_keyframes:
            lid = self.next_lid
            self.next_lid += 1
            lm = Landmark(lid, label, cand.xy, cand.size_sum / cand.n)
            lm.n_obs = cand.n
            self.landmarks[lid] = lm
            self.values.insert(L(lid), gtsam.Point2(*lm.xy))
            for k, (b, r) in cand.obs.items():
                self._add_object_factor(k, lid, b, r)
            self.candidates.remove(cand)
            return 'promoted'
        return 'candidate'

    # ------------------------------------------------------------------ solve
    def update(self):
        """Run the solver on everything added since the last call."""
        if self.graph.size() == 0 and self.values.size() == 0:
            return
        self.isam.update(self.graph, self.values)
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.est = self.isam.calculateEstimate()
        for lid, lm in self.landmarks.items():
            if self.est.exists(L(lid)):
                p = self.est.atPoint2(L(lid))
                lm.xy = np.array([p[0], p[1]])
        last = self.n_keyframes - 1
        self.candidates = [c for c in self.candidates if last - c.last_kf <= self.candidate_ttl]

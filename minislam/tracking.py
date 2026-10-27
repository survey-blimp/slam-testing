"""Per-frame tracking state machine.

States: ``NO_IMAGES -> NOT_INITIALIZED -> OK <-> LOST``.

  * NOT_INITIALIZED: feed frames to the two-view ``Initializer`` until it
    builds the first map.
  * OK: estimate the pose of every new frame against the map:

      1. **Predict** with a constant-velocity model. ``V = T_last T_prev^-1``
         is the motion between the last two frames; assume it repeats:
         ``T_pred = V T_last``.
      2. **Match by projection**: project the last frame's map points with
         the predicted pose and search nearby (15 px x octave scale, then
         30 px). If that fails, brute-force match against the reference
         keyframe's map points.
      3. **Pose-only optimization** (4 rounds with chi-square outlier
         re-classification).
      4. **Track the local map**: project the points of the keyframes that
         share points with this frame (plus their best covisible
         neighbours) and match those too, then optimize again. This finds
         many more correspondences than the last frame alone and ties the
         pose to the map rather than to the previous frame (less drift).
      5. **Success** if enough inliers remain, else LOST.
      6. **Keyframe decision**: insert a keyframe when tracking is getting
         weaker than at the reference keyframe, or after a while anyway.

ORB-SLAM3 equivalent: ``Tracking.cc`` (``Track``, ``TrackWithMotionModel``,
``TrackReferenceKeyFrame``, ``TrackLocalMap``, ``UpdateLocalKeyFrames``,
``SearchLocalPoints``, ``NeedNewKeyFrame``).
"""

import logging
import time
from collections import Counter
from enum import Enum

import numpy as np

from minislam import geometry
from minislam.camera import PinholeCamera
from minislam.config import Config
from minislam.frame import Frame
from minislam.initializer import Initializer
from minislam.map import KeyFrame, Map, MapPoint
from minislam.matcher import match_brute_force, search_by_projection
from minislam.optimizer import optimize_pose

log = logging.getLogger(__name__)


class TrackingState(Enum):
    NO_IMAGES = "NO_IMAGES"
    NOT_INITIALIZED = "NOT_INITIALIZED"
    OK = "OK"
    LOST = "LOST"


class Tracker:
    def __init__(self, cfg: Config, camera: PinholeCamera, map_: Map) -> None:
        self.cfg = cfg
        self.camera = camera
        self.K = camera.K
        self.map = map_
        self.initializer = Initializer(cfg, camera.K)
        self.reset()

    def reset(self) -> None:
        self.state = TrackingState.NO_IMAGES
        self.initializer.reset()
        self.last_frame: Frame | None = None
        self.velocity: np.ndarray | None = None  # T_last T_prev^-1
        self.reference_kf: KeyFrame | None = None
        self.last_kf_frame_id = 0
        self.local_keyframes: list[KeyFrame] = []
        self.local_map_points: list[MapPoint] = []
        self.message = ""
        self.timings = {"track": 0.0, "local_map": 0.0}
        # Trajectory stored relative to a reference keyframe, as ORB-SLAM does:
        # T_cw = T_cr @ T_rw. When BA later moves the keyframe, the frame moves with it.
        self.trajectory: list[tuple[float, KeyFrame, np.ndarray]] = []

    # ------------------------------------------------------------ main entry

    def process(self, frame: Frame) -> None:
        t0 = time.perf_counter()
        self.timings = {"track": 0.0, "local_map": 0.0}
        if self.state == TrackingState.NO_IMAGES:
            self.state = TrackingState.NOT_INITIALIZED

        if self.state == TrackingState.NOT_INITIALIZED:
            self._initialize(frame)
        elif self.state == TrackingState.OK:
            self._track(frame)
        self.timings["track"] = 1000 * (time.perf_counter() - t0) - self.timings["local_map"]
        self.last_frame = frame

    # ------------------------------------------------------- initialization

    def _initialize(self, frame: Frame) -> None:
        kfs = self.initializer.try_initialize(frame, self.map)
        self.message = self.initializer.message
        if kfs is None:
            return
        kf0, kf1 = kfs
        self.reference_kf = kf1
        self.last_kf_frame_id = frame.id
        self.local_keyframes = [kf0, kf1]
        self.local_map_points = self.map.good_points()
        self.velocity = None
        self.state = TrackingState.OK
        self._record_pose(self.initializer.reference)
        self._record_pose(frame)

    # ------------------------------------------------------------- tracking

    def _track(self, frame: Frame) -> None:
        last = self.last_frame
        ok = self._track_last_frame(frame, last)
        n_inliers = self._track_local_map(frame) if ok else 0
        if n_inliers < self.cfg.tracking.min_inliers:
            log.info("Tracking LOST at frame %d (%d inliers < %d)", frame.id, n_inliers,
                     self.cfg.tracking.min_inliers)
            self.message = f"lost: only {n_inliers} inliers"
            self.state = TrackingState.LOST
            self.velocity = None
            frame.T_cw = None
            return
        self.message = ""
        self.velocity = frame.T_cw @ geometry.se3_inverse(last.T_cw)
        self._record_pose(frame)

    def _track_last_frame(self, frame: Frame, last: Frame) -> bool:
        """Steps 1-3: predict, match the last frame's points by projection, optimize."""
        c = self.cfg.tracking
        fc = self.cfg.features
        T_pred = last.T_cw if self.velocity is None else self.velocity @ last.T_cw
        frame.T_cw = T_pred.copy()

        idx = last.matched_indices()
        points = [last.map_points[i] for i in idx]
        n = 0
        if points:
            p_w = np.array([mp.p_w for mp in points])
            descs = np.array([mp.descriptor for mp in points])
            for radius in (c.projection_radius_px, c.projection_radius_wide_px):
                frame.clear_matches()
                pairs = search_by_projection(
                    frame, p_w, descs, last.octaves[idx], self.K, radius, fc.scale_factor,
                    c.octave_tolerance, c.max_distance, c.projection_ratio,
                    self.camera.width, self.camera.height)
                for p, k in pairs:
                    frame.map_points[k] = points[p]
                n = len(pairs)
                if n >= c.min_projection_matches:
                    break

        if n < c.min_projection_matches:
            n = self._match_reference_keyframe(frame, last)
            log.debug("frame %d: projection found too few matches, reference KF brute force: %d",
                      frame.id, n)
            if n < c.min_projection_matches:
                return False

        o = self.cfg.optimizer
        n_in = optimize_pose(frame, self.K, o.chi2_2dof, o.pose_rounds, o.pose_max_nfev)
        self._drop_outliers(frame)
        return n_in >= c.min_projection_matches

    def _match_reference_keyframe(self, frame: Frame, last: Frame) -> int:
        """Fallback: brute-force descriptors against the reference keyframe's map points."""
        c = self.cfg.tracking
        ref = self.reference_kf
        frame.clear_matches()
        frame.T_cw = last.T_cw.copy()
        idx = np.array([i for i, mp in enumerate(ref.map_points)
                        if mp is not None and not mp.is_bad], dtype=int)
        if len(idx) == 0:
            return 0
        pairs = match_brute_force(frame.descriptors, ref.descriptors[idx], c.reference_ratio,
                                  c.max_distance, cross_check=False)
        for i, j in pairs:
            frame.map_points[i] = ref.map_points[idx[j]]
        return len(pairs)

    def _drop_outliers(self, frame: Frame) -> None:
        for i in np.nonzero(frame.outlier)[0]:
            frame.map_points[i] = None
        frame.outlier[:] = False

    # ------------------------------------------------------------ local map

    def _update_local_map(self, frame: Frame) -> None:
        """Local keyframes = those sharing points with the frame + their best neighbours."""
        c = self.cfg.tracking
        counts: Counter = Counter()
        for i in frame.matched_indices():
            for kf in frame.map_points[i].observations:
                counts[kf] += 1
        if not counts:
            return
        ordered = [kf for kf, _ in counts.most_common()]
        local = ordered[:c.local_map_max_keyframes]
        for kf in ordered:
            if len(local) >= c.local_map_max_keyframes:
                break
            for nb in kf.best_covisible(c.local_map_max_keyframes):
                if nb not in local:
                    local.append(nb)
                    if len(local) >= c.local_map_max_keyframes:
                        break
        self.local_keyframes = local
        self.reference_kf = ordered[0]  # the keyframe sharing the most points

        seen, points = set(), []
        for kf in local:
            for mp in kf.map_points:
                if mp is not None and not mp.is_bad and mp.id not in seen:
                    seen.add(mp.id)
                    points.append(mp)
        self.local_map_points = points

    def _track_local_map(self, frame: Frame) -> int:
        """Step 4: match more points from the local map and re-optimize. Returns inliers."""
        c = self.cfg.tracking
        fc = self.cfg.features
        self._update_local_map(frame)

        already = {id(frame.map_points[i]) for i in frame.matched_indices()}
        for i in frame.matched_indices():
            frame.map_points[i].n_visible += 1

        # Which local points are in the field of view and at a plausible scale?
        cands = [mp for mp in self.local_map_points if id(mp) not in already]
        if cands:
            p_w = np.array([mp.p_w for mp in cands])
            C = frame.camera_center
            uv, z = geometry.project(self.K, frame.T_cw, p_w)
            ray = p_w - C
            dist = np.linalg.norm(ray, axis=1)
            normals = np.array([mp.normal for mp in cands])
            view_cos = np.sum(ray * normals, axis=1) / np.maximum(dist, 1e-9)
            min_d = np.array([mp.min_distance for mp in cands])
            max_d = np.array([mp.max_distance for mp in cands])
            visible = ((z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < self.camera.width)
                       & (uv[:, 1] >= 0) & (uv[:, 1] < self.camera.height)
                       & (dist >= 0.8 * min_d) & (dist <= 1.2 * max_d)
                       & (view_cos >= c.min_viewing_cos))
            vis = np.nonzero(visible)[0]
            for k in vis:
                cands[k].n_visible += 1
            if len(vis):
                pred_oct = np.array([cands[k].predict_octave(dist[k], fc.scale_factor, fc.n_levels)
                                     for k in vis])
                descs = np.array([cands[k].descriptor for k in vis])
                pairs = search_by_projection(
                    frame, p_w[vis], descs, pred_oct, self.K, c.local_map_radius_px,
                    fc.scale_factor, c.octave_tolerance, c.max_distance, c.projection_ratio,
                    self.camera.width, self.camera.height)
                for p, kp in pairs:
                    frame.map_points[kp] = cands[vis[p]]

        o = self.cfg.optimizer
        n_in = optimize_pose(frame, self.K, o.chi2_2dof, o.pose_rounds, o.pose_max_nfev)
        for i in frame.matched_indices():
            frame.map_points[i].n_found += 1
        return n_in

    # ------------------------------------------------------------ keyframes

    def need_new_keyframe(self, frame: Frame) -> bool:
        c = self.cfg.tracking
        since = frame.id - self.last_kf_frame_id
        if since < c.min_frames_between_kf:
            return False
        n_in = frame.n_tracked_inliers()
        ref_tracked = self.reference_kf.tracked_map_points(c.kf_ref_min_obs)
        weak = n_in < c.kf_inlier_ratio * ref_tracked and n_in >= c.kf_min_inliers
        return weak or since >= c.max_frames_between_kf

    # ------------------------------------------------------------ trajectory

    def _record_pose(self, frame: Frame) -> None:
        ref = self.reference_kf
        self.trajectory.append((frame.timestamp, ref, frame.T_cw @ geometry.se3_inverse(ref.T_cw)))

    def trajectory_poses_cw(self) -> tuple[list[float], list[np.ndarray]]:
        """Timestamps and current ``T_cw`` of every tracked frame (re-anchored on its keyframe)."""
        ts = [t for t, _, _ in self.trajectory]
        poses = [T_cr @ ref.T_cw for _, ref, T_cr in self.trajectory]
        return ts, poses

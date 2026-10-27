"""Two-view monocular map initialization.

A single camera cannot measure depth from one image, so the map is
bootstrapped from two images taken from different positions:

1. Pick a **reference frame** with enough features.
2. Match each new frame against it (brute force, ratio test, cross-check).
   Too few matches -> keep waiting; far too few -> the view changed too
   much, so the new frame becomes the reference.
3. Estimate the **essential matrix** ``E = [t]_x R`` from the matches with
   RANSAC (5-point algorithm) and decompose it into the relative pose
   ``R, t``. ``t`` is only known up to scale (monocular!), so ``|t| = 1``.
4. **Triangulate** the inlier matches and keep points that are in front of
   both cameras, reproject well (chi-square) and have >= 1 deg parallax.
5. Accept if enough points survive *and* the median parallax is large
   enough. Pure rotation gives zero parallax and is rejected here: the
   camera must translate.
6. Create two keyframes and the map points, **normalize scale** so the
   median scene depth is 1, and refine everything with full bundle
   adjustment (first keyframe fixed).

Simplification: ORB-SLAM3 (``TwoViewReconstruction``) estimates both a
homography (for planar or low-parallax scenes) and a fundamental matrix in
parallel and picks the better-scoring model. minislam only implements the
essential-matrix path, so a perfectly planar scene may initialize poorly.

ORB-SLAM3 equivalent: ``TwoViewReconstruction.cc`` and
``Tracking::MonocularInitialization`` / ``CreateInitialMapMonocular``.
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from minislam import geometry
from minislam.config import Config, InitializerConfig
from minislam.frame import Frame
from minislam.map import KeyFrame, Map
from minislam.matcher import match_brute_force
from minislam.optimizer import bundle_adjustment

log = logging.getLogger(__name__)


@dataclass
class TwoViewResult:
    T_21: np.ndarray  # pose of view 2 relative to view 1 (= T_c2w with world = view 1)
    points: np.ndarray  # (M, 3) triangulated points in the view-1 frame
    match_idx: np.ndarray  # (M,) indices into the input match arrays of the kept points
    median_parallax_deg: float


def reconstruct_two_view(uv1: np.ndarray, uv2: np.ndarray, sigma2_1: np.ndarray,
                         sigma2_2: np.ndarray, K: np.ndarray, cfg: InitializerConfig,
                         chi2_threshold: float) -> tuple[TwoViewResult | None, str]:
    """Relative pose + 3D points from matched undistorted pixels of two views.

    Returns (result, "") on success or (None, reason) on failure.
    """
    # BLACKBOX: cv2.findEssentialMat runs RANSAC around Nister's 5-point
    # solver: repeatedly pick 5 matches, solve the polynomial system for up
    # to 10 candidate E's, count matches whose epipolar (Sampson) error is
    # below the threshold, keep the best. By hand: implement the 5-point
    # solver (or the simpler, less accurate 8-point algorithm on normalized
    # coordinates) inside a RANSAC loop.
    E, mask = cv2.findEssentialMat(uv1, uv2, K, method=cv2.RANSAC, prob=cfg.ransac_prob,
                                   threshold=cfg.ransac_threshold_px)
    if E is None or E.shape != (3, 3):
        return None, "essential matrix estimation failed"
    # BLACKBOX: cv2.recoverPose decomposes E = U diag(1,1,0) V^T into the four
    # candidate poses (R = U W V^T or U W^T V^T, t = +-u3) and keeps the one
    # that puts the most triangulated points in front of both cameras
    # (cheirality check). By hand: an SVD plus four triangulations.
    _, R, t, pose_mask = cv2.recoverPose(E, uv1, uv2, K, mask=mask.copy())
    inliers = np.nonzero(pose_mask.ravel() > 0)[0]
    if len(inliers) < cfg.min_points:
        return None, f"too few essential-matrix inliers ({len(inliers)})"

    T_11 = np.eye(4)
    T_21 = geometry.make_T(R, t.ravel())
    pts = geometry.triangulate(K, T_11, T_21, uv1[inliers], uv2[inliers])
    good, parallax = geometry.check_triangulation(
        K, T_11, T_21, pts, uv1[inliers], uv2[inliers], sigma2_1[inliers], sigma2_2[inliers],
        chi2_threshold, cfg.min_parallax_deg)
    n_good = int(good.sum())
    if n_good < cfg.min_points:
        median_all = float(np.median(parallax[np.isfinite(parallax)])) if len(parallax) else 0.0
        return None, (f"too few good triangulated points ({n_good} < {cfg.min_points}, "
                      f"median parallax {median_all:.2f} deg)")
    median_parallax = float(np.median(parallax[good]))
    if median_parallax < cfg.min_parallax_deg:
        return None, f"low parallax ({median_parallax:.2f} deg), translate the camera more"
    return TwoViewResult(T_21, pts[good], inliers[good], median_parallax), ""


class Initializer:
    """Holds the reference frame and decides when two views are good enough."""

    def __init__(self, cfg: Config, K: np.ndarray) -> None:
        self.cfg = cfg
        self.K = K
        self.reference: Frame | None = None
        self.matches = np.zeros((0, 2), dtype=int)  # last reference<->current matches (for the GUI)
        self.message = "waiting for a reference frame"

    def reset(self) -> None:
        self.reference = None
        self.matches = np.zeros((0, 2), dtype=int)

    def try_initialize(self, frame: Frame, map_: Map) -> tuple[KeyFrame, KeyFrame] | None:
        """Feed one frame. Returns the two initial keyframes once the map is built."""
        c = self.cfg.initializer
        if frame.n_features < c.min_features:
            self.message = f"init: too few features ({frame.n_features} < {c.min_features})"
            return None
        if self.reference is None:
            self.reference = frame
            self.matches = np.zeros((0, 2), dtype=int)
            self.message = "init: reference frame set, now translate the camera sideways"
            log.info("Initialization: reference frame %d (%d features)", frame.id, frame.n_features)
            return None

        ref = self.reference
        self.matches = match_brute_force(ref.descriptors, frame.descriptors, c.ratio,
                                         c.max_distance, cross_check=True)
        n = len(self.matches)
        if n < c.reset_below_matches:
            log.info("Initialization: only %d matches, replacing reference with frame %d", n, frame.id)
            self.reference = frame
            self.matches = np.zeros((0, 2), dtype=int)
            self.message = f"init: lost the reference ({n} matches), new reference set"
            return None
        if n < c.min_matches:
            self.message = f"init: too few matches ({n} < {c.min_matches})"
            return None

        i1, i2 = self.matches[:, 0], self.matches[:, 1]
        result, reason = reconstruct_two_view(
            ref.keypoints_un[i1], frame.keypoints_un[i2], ref.sigma2[i1], frame.sigma2[i2],
            self.K, c, self.cfg.optimizer.chi2_2dof)
        if result is None:
            self.message = f"init: {reason}"
            log.debug("Initialization attempt with frame %d failed: %s", frame.id, reason)
            return None

        kfs = self.create_initial_map(ref, frame, self.matches[result.match_idx], result, map_)
        if kfs is None:
            return None
        log.info("Initialization SUCCESS with frames %d/%d: %d points, median parallax %.1f deg",
                 ref.id, frame.id, len(map_.map_points), result.median_parallax_deg)
        self.message = ""
        return kfs

    def create_initial_map(self, ref: Frame, cur: Frame, matches: np.ndarray,
                            result: TwoViewResult, map_: Map) -> tuple[KeyFrame, KeyFrame] | None:
        fc = self.cfg.features
        ref.T_cw = np.eye(4)
        cur.T_cw = result.T_21.copy()
        ref.clear_matches()
        cur.clear_matches()
        kf0 = KeyFrame.from_frame(ref, map_.new_keyframe_id())
        kf1 = KeyFrame.from_frame(cur, map_.new_keyframe_id())
        map_.add_keyframe(kf0)
        map_.add_keyframe(kf1)

        # Scale normalization: the median depth of the scene becomes 1.
        scale = 1.0 / float(np.median(result.points[:, 2]))
        kf1.T_cw[:3, 3] *= scale
        for (i0, i1), p in zip(matches, result.points * scale):
            mp = map_.new_map_point(p, kf0)
            map_.add_observation(mp, kf0, int(i0))
            map_.add_observation(mp, kf1, int(i1))

        points = map_.good_points()
        ba = bundle_adjustment([kf1], [kf0], points, self.K, self.cfg.optimizer.chi2_2dof,
                               self.cfg.optimizer.init_ba_max_nfev, label="init BA")
        map_.cull_observations(ba.observations, ba.chi2 > self.cfg.optimizer.chi2_2dof, min_obs=2)

        n_left = len(map_.good_points())
        if n_left < self.cfg.initializer.min_points:
            self.message = f"init: only {n_left} points survived BA, retrying"
            log.info("Initialization rejected after BA: %d points left", n_left)
            map_.clear()
            self.reference = cur
            return None

        for mp in map_.good_points():
            mp.update_descriptor()
            mp.update_normal_and_depth(fc.scale_factor, fc.n_levels)
        for kf in (kf0, kf1):
            map_.update_covisibility(kf, self.cfg.map.min_covisibility_weight)

        # The current frame continues as the "last frame" of tracking.
        cur.T_cw = kf1.T_cw.copy()
        cur.map_points = list(kf1.map_points)
        cur.outlier = np.zeros(cur.n_features, dtype=bool)
        return kf0, kf1

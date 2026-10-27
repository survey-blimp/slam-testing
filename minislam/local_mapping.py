"""Local mapping: grow and refine the map around each new keyframe.

When tracking inserts a keyframe, this module:

1. **Inserts it**: adds the keyframe's tracked map points as observations,
   refreshes their representative descriptors and viewing directions, and
   updates the covisibility graph (edges need >= 15 shared points).
2. **Culls recently created points** that turn out unreliable:
   * found in fewer than 25% of the frames where they should have been
     visible (``n_found / n_visible < 0.25``), or
   * two keyframes after creation still seen by fewer than 2 keyframes.
   Points that survive 3 keyframes are considered mature.
3. **Triangulates new points** between the new keyframe and its best
   covisible neighbours. Feature pairs that have no map point yet are
   matched along epipolar lines, triangulated, and accepted with the same
   depth / chi-square / parallax checks as initialization. Pairs whose
   baseline is tiny compared with the scene depth are skipped (the points
   would be hopelessly uncertain).
4. **Local bundle adjustment** over the new keyframe, its covisible
   keyframes and all points they see. Other keyframes that see those points
   contribute residuals but stay fixed, and the very first keyframe is
   always fixed (gauge freedom). Afterwards observations failing the
   chi-square test are removed, and points left with fewer than 2
   observations are deleted.

In minislam this runs synchronously inside ``Tracker`` on keyframe
insertion (tracking briefly stalls). ORB-SLAM3 runs it in its own thread
so tracking never waits for it.

ORB-SLAM3 equivalent: ``LocalMapping.cc`` (``ProcessNewKeyFrame``,
``MapPointCulling``, ``CreateNewMapPoints``, ``Optimizer::LocalBundleAdjustment``).
ORB-SLAM3 additionally fuses duplicate points (``SearchInNeighbors``) and
culls redundant keyframes; both are outside minislam v1.
"""

import logging

import numpy as np

from minislam import geometry
from minislam.config import Config
from minislam.map import KeyFrame, Map, MapPoint
from minislam.matcher import search_for_triangulation
from minislam.optimizer import bundle_adjustment

log = logging.getLogger(__name__)


class LocalMapper:
    def __init__(self, cfg: Config, K: np.ndarray, map_: Map) -> None:
        self.cfg = cfg
        self.K = K
        self.map = map_
        self.recent_points: list[MapPoint] = []

    def reset(self) -> None:
        self.recent_points = []

    def process_keyframe(self, kf: KeyFrame) -> None:
        self._insert_keyframe(kf)
        n_culled = self._cull_recent_points(kf)
        n_new = self._triangulate_new_points(kf)
        self.map.update_covisibility(kf, self.cfg.map.min_covisibility_weight)
        n_removed = self._local_bundle_adjustment(kf)
        log.info("KF %d (frame %d): +%d points, culled %d recent, %d removed after BA; map %d KFs, %d points",
                 kf.kf_id, kf.id, n_new, n_culled, n_removed, len(self.map.keyframes),
                 len(self.map.map_points))

    # ------------------------------------------------------------- steps

    def _update_point(self, mp: MapPoint) -> None:
        fc = self.cfg.features
        mp.update_descriptor()
        mp.update_normal_and_depth(fc.scale_factor, fc.n_levels)

    def _insert_keyframe(self, kf: KeyFrame) -> None:
        self.map.add_keyframe(kf)
        for idx, mp in enumerate(kf.map_points):
            if mp is None:
                continue
            if mp.is_bad:
                kf.map_points[idx] = None
                continue
            self.map.add_observation(mp, kf, idx)
            self._update_point(mp)
        self.map.update_covisibility(kf, self.cfg.map.min_covisibility_weight)

    def _cull_recent_points(self, kf: KeyFrame) -> int:
        c = self.cfg.local_mapping
        keep, culled = [], 0
        for mp in self.recent_points:
            age = kf.kf_id - mp.first_kf_id
            if mp.is_bad:
                continue
            if mp.found_ratio() < c.cull_min_found_ratio:
                self.map.remove_map_point(mp)
                culled += 1
            elif age >= c.cull_keyframe_age and len(mp.observations) < c.cull_min_observations:
                self.map.remove_map_point(mp)
                culled += 1
            elif age < c.recent_point_maturity:
                keep.append(mp)  # still on probation
        self.recent_points = keep
        return culled

    def _triangulate_new_points(self, kf: KeyFrame) -> int:
        c = self.cfg.local_mapping
        chi2 = self.cfg.optimizer.chi2_2dof
        C1 = kf.camera_center
        n_new = 0
        for nb in kf.best_covisible(c.triangulation_neighbors):
            baseline = np.linalg.norm(nb.camera_center - C1)
            if baseline / nb.median_depth() < c.min_baseline_depth_ratio:
                continue
            pairs = search_for_triangulation(kf, nb, self.K, c.triangulation_max_distance,
                                             c.triangulation_ratio, c.epipolar_sigma_factor)
            if len(pairs) == 0:
                continue
            i1, i2 = pairs[:, 0], pairs[:, 1]
            p_w = geometry.triangulate(self.K, kf.T_cw, nb.T_cw, kf.keypoints_un[i1],
                                       nb.keypoints_un[i2])
            good, _ = geometry.check_triangulation(
                self.K, kf.T_cw, nb.T_cw, p_w, kf.keypoints_un[i1], nb.keypoints_un[i2],
                kf.sigma2[i1], nb.sigma2[i2], chi2, self.cfg.initializer.min_parallax_deg)
            for k in np.nonzero(good)[0]:
                mp = self.map.new_map_point(p_w[k], kf)
                self.map.add_observation(mp, kf, int(i1[k]))
                self.map.add_observation(mp, nb, int(i2[k]))
                self._update_point(mp)
                self.recent_points.append(mp)
                n_new += 1
        return n_new

    def _local_bundle_adjustment(self, kf: KeyFrame) -> int:
        c = self.cfg.local_mapping
        o = self.cfg.optimizer
        local = [kf] + kf.best_covisible(c.local_ba_neighbors)
        local_set = set(local)

        points, seen = [], set()
        for k in local:
            for mp in k.map_points:
                if mp is not None and not mp.is_bad and mp.id not in seen:
                    seen.add(mp.id)
                    points.append(mp)

        fixed = {other for mp in points for other in mp.observations if other not in local_set}
        first = self.map.keyframes[0]
        if first in local_set:  # gauge freedom: the first keyframe never moves
            local.remove(first)
            fixed.add(first)
        if not local or not points:
            return 0

        ba = bundle_adjustment(local, list(fixed), points, self.K, o.chi2_2dof, o.ba_max_nfev,
                               label="local BA")
        removed = self.map.cull_observations(ba.observations, ba.chi2 > o.chi2_2dof,
                                             min_obs=c.cull_min_observations)
        # BA moved the points, so refresh viewing directions; descriptors only
        # depend on the observation set, which BA does not grow.
        fc = self.cfg.features
        for mp in points:
            if not mp.is_bad:
                mp.update_normal_and_depth(fc.scale_factor, fc.n_levels)
        for k in local:
            self.map.update_covisibility(k, self.cfg.map.min_covisibility_weight)
        return removed

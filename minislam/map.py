"""The map: MapPoints (3D landmarks), KeyFrames, and the covisibility graph.

* A **MapPoint** is a 3D point ``p_w`` plus the keyframes that observe it
  (``observations: {KeyFrame: keypoint index}``). For matching it keeps a
  *representative descriptor* (the observed descriptor with the smallest
  median Hamming distance to all the others, i.e. the "medoid") and a mean
  viewing direction. ``n_visible`` / ``n_found`` count how often the point
  was predicted to be in view versus actually matched; points that are
  rarely found when expected are culled.

* A **KeyFrame** is a Frame made permanent. Besides the Frame data it holds
  covisibility weights ``{other KeyFrame: number of shared map points}``.

* The **covisibility graph** links keyframes that see the same points. It
  decides which keyframes form the "local map" for tracking, which
  neighbours to triangulate against, and which keyframes local BA
  optimizes. Edges need at least ``min_weight`` (15) shared points.

* **Scale prediction.** A keypoint seen at distance ``d`` on octave ``o``
  would be seen on octave ``o + log_s(d / d')`` at distance ``d'``. Each
  MapPoint stores the distance range ``[min_distance, max_distance]`` over
  which it should be detectable at any pyramid level, and predicts the
  octave from the current distance. This sets the search radius and rejects
  points seen from much too close / far.

``Map`` keeps both directions of every association consistent
(``MapPoint.observations`` and ``KeyFrame.map_points``): always go through
its ``add_*`` / ``remove_*`` methods.

ORB-SLAM3 equivalent: ``MapPoint.cc``, ``KeyFrame.cc``, ``Map.cc``.
"""

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from minislam import geometry
from minislam.frame import Frame

# Popcount of every byte value, for vectorized Hamming distances.
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between (N, 32) and (M, 32) uint8 descriptors -> (N, M)."""
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)), dtype=np.int32)
    return POPCOUNT[np.bitwise_xor(A[:, None, :], B[None, :, :])].sum(axis=2, dtype=np.int32)


@dataclass(eq=False)
class KeyFrame(Frame):
    kf_id: int = -1
    covisibility: dict["KeyFrame", int] = field(default_factory=dict)

    @staticmethod
    def from_frame(frame: Frame, kf_id: int) -> "KeyFrame":
        """Promote a tracked frame. Only inlier associations are carried over."""
        map_points = [mp if (mp is not None and not frame.outlier[i] and not mp.is_bad) else None
                      for i, mp in enumerate(frame.map_points)]
        return KeyFrame(id=frame.id, timestamp=frame.timestamp, image=frame.image,
                        keypoints=frame.keypoints, keypoints_un=frame.keypoints_un,
                        octaves=frame.octaves, descriptors=frame.descriptors,
                        sigma2=frame.sigma2, T_cw=frame.T_cw.copy(), map_points=map_points,
                        kf_id=kf_id)

    def best_covisible(self, n: int) -> list["KeyFrame"]:
        """Up to ``n`` neighbours, most shared points first."""
        return [kf for kf, _ in sorted(self.covisibility.items(), key=lambda kv: -kv[1])[:n]]

    def tracked_map_points(self, min_obs: int) -> int:
        """Number of good map points of this keyframe seen by at least ``min_obs`` keyframes."""
        return sum(1 for mp in self.map_points
                   if mp is not None and not mp.is_bad and len(mp.observations) >= min_obs)

    def median_depth(self) -> float:
        pts = np.array([mp.p_w for mp in self.map_points if mp is not None and not mp.is_bad])
        if len(pts) == 0:
            return 1.0
        return float(np.median(geometry.transform(self.T_cw, pts)[:, 2]))


@dataclass(eq=False)
class MapPoint:
    id: int
    p_w: np.ndarray  # (3,) position in the world frame
    first_kf_id: int  # keyframe that created it (used for culling young points)
    ref_kf: KeyFrame | None = None  # keyframe that created it (used for scale prediction)
    observations: dict[KeyFrame, int] = field(default_factory=dict)
    descriptor: np.ndarray = field(default_factory=lambda: np.zeros(32, dtype=np.uint8))
    normal: np.ndarray = field(default_factory=lambda: np.zeros(3))  # mean viewing direction
    min_distance: float = 0.0
    max_distance: float = math.inf
    n_visible: int = 1
    n_found: int = 1
    is_bad: bool = False

    def found_ratio(self) -> float:
        return self.n_found / max(self.n_visible, 1)

    def update_descriptor(self) -> None:
        """Pick the observed descriptor with the least median distance to the others."""
        descs = np.array([kf.descriptors[idx] for kf, idx in self.observations.items()])
        if len(descs) == 0:
            return
        if len(descs) <= 2:
            self.descriptor = descs[0]
            return
        D = hamming_matrix(descs, descs)
        self.descriptor = descs[int(np.argmin(np.median(D, axis=1)))]

    def update_normal_and_depth(self, scale_factor: float, n_levels: int) -> None:
        """Mean viewing direction and the distance range for scale prediction."""
        if not self.observations or self.ref_kf is None:
            return
        rays = [self.p_w - kf.camera_center for kf in self.observations]
        rays = [r / np.linalg.norm(r) for r in rays]
        n = np.mean(rays, axis=0)
        self.normal = n / np.linalg.norm(n)
        ref = self.ref_kf if self.ref_kf in self.observations else next(iter(self.observations))
        dist = float(np.linalg.norm(self.p_w - ref.camera_center))
        octave = int(ref.octaves[self.observations[ref]])
        self.max_distance = dist * scale_factor ** octave
        self.min_distance = self.max_distance / scale_factor ** (n_levels - 1)

    def predict_octave(self, distance: np.ndarray | float, scale_factor: float,
                       n_levels: int) -> np.ndarray:
        """Pyramid level at which this point should appear at ``distance``."""
        ratio = self.max_distance / np.maximum(distance, 1e-9)
        octave = np.ceil(np.log(ratio) / math.log(scale_factor))
        return np.clip(octave, 0, n_levels - 1).astype(int)


class Map:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        """Forget everything (used on reset or a rejected initialization)."""
        for mp in getattr(self, "map_points", {}).values():
            mp.is_bad = True  # frames that still reference these points must ignore them
        self.keyframes: list[KeyFrame] = []
        self.map_points: dict[int, MapPoint] = {}
        self._next_kf_id = 0
        self._next_mp_id = 0

    # ------------------------------------------------------------- creation

    def new_keyframe_id(self) -> int:
        self._next_kf_id += 1
        return self._next_kf_id - 1

    def add_keyframe(self, kf: KeyFrame) -> None:
        self.keyframes.append(kf)

    def new_map_point(self, p_w: np.ndarray, ref_kf: KeyFrame) -> MapPoint:
        mp = MapPoint(id=self._next_mp_id, p_w=np.asarray(p_w, dtype=np.float64).copy(),
                      first_kf_id=ref_kf.kf_id, ref_kf=ref_kf)
        self._next_mp_id += 1
        self.map_points[mp.id] = mp
        return mp

    # --------------------------------------------------------- associations

    def add_observation(self, mp: MapPoint, kf: KeyFrame, idx: int) -> None:
        mp.observations[kf] = idx
        kf.map_points[idx] = mp

    def remove_observation(self, mp: MapPoint, kf: KeyFrame) -> None:
        idx = mp.observations.pop(kf, None)
        if idx is not None and kf.map_points[idx] is mp:
            kf.map_points[idx] = None

    def remove_map_point(self, mp: MapPoint) -> None:
        """Mark bad and detach from every keyframe. Frames holding it check ``is_bad``."""
        mp.is_bad = True
        for kf, idx in list(mp.observations.items()):
            if kf.map_points[idx] is mp:
                kf.map_points[idx] = None
        mp.observations.clear()
        self.map_points.pop(mp.id, None)

    def cull_observations(self, observations: list[tuple[MapPoint, KeyFrame]],
                          bad: np.ndarray, min_obs: int) -> int:
        """Remove the flagged observations; remove points left with < ``min_obs``.

        Used after bundle adjustment with ``bad = chi2 > threshold``.
        Returns the number of map points removed.
        """
        touched = set()
        for (mp, kf), is_bad in zip(observations, bad):
            if is_bad and not mp.is_bad:
                self.remove_observation(mp, kf)
                touched.add(mp)
        removed = 0
        for mp in touched:
            if len(mp.observations) < min_obs:
                self.remove_map_point(mp)
                removed += 1
        return removed

    # ---------------------------------------------------------- covisibility

    def update_covisibility(self, kf: KeyFrame, min_weight: int) -> None:
        """Recount shared map points between ``kf`` and every other keyframe.

        Keeps edges with weight >= ``min_weight``; if none qualifies, keeps
        the single strongest edge so no keyframe becomes disconnected (as
        ORB-SLAM's ``UpdateConnections`` does).
        """
        counts: Counter = Counter()
        for mp in kf.map_points:
            if mp is None or mp.is_bad:
                continue
            for other in mp.observations:
                if other is not kf:
                    counts[other] += 1
        edges = {other: w for other, w in counts.items() if w >= min_weight}
        if not edges and counts:
            best, w = counts.most_common(1)[0]
            edges = {best: w}
        for old in list(kf.covisibility):
            if old not in edges:
                old.covisibility.pop(kf, None)
        kf.covisibility = edges
        for other, w in edges.items():
            other.covisibility[kf] = w

    # --------------------------------------------------------------- queries

    def good_points(self) -> list[MapPoint]:
        return [mp for mp in self.map_points.values() if not mp.is_bad]

    def points_array(self) -> tuple[np.ndarray, np.ndarray]:
        """(ids (P,), positions (P, 3)) of all good map points."""
        pts = self.good_points()
        if not pts:
            return np.zeros(0, dtype=int), np.zeros((0, 3))
        return np.array([mp.id for mp in pts]), np.array([mp.p_w for mp in pts])

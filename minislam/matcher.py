"""Descriptor matching: brute force, by projection, and epipolar-guided.

ORB descriptors are 256-bit strings compared with the **Hamming distance**
(number of differing bits, 0..256). Two tests make a match trustworthy:

  * **absolute**: distance <= ``max_distance`` (50 in ORB-SLAM's TH_LOW),
  * **ratio** (Lowe): best distance < ``ratio`` x second-best distance, so
    the match is not ambiguous (repeated texture gives two similar
    candidates and is rejected).

Three strategies, from least to most prior knowledge:

1. ``match_brute_force``: compare every descriptor with every other. Used
   when nothing is known about the geometry (initialization, relocalization,
   tracking fallback). Optional cross-check: i's best is j *and* j's best is i.
2. ``search_by_projection``: when the pose is (approximately) known, project
   each 3D map point into the image and only compare it with keypoints
   inside a small radius. Far cheaper and far less ambiguous.
3. ``search_for_triangulation``: between two posed keyframes, a feature in
   image 1 can only match features near its epipolar line in image 2.

ORB-SLAM3 equivalent: ``ORBmatcher.cc`` (``SearchByProjection``,
``SearchForInitialization``, ``SearchForTriangulation``). ORB-SLAM3 speeds
up brute-force matching with a bag-of-words vocabulary (DBoW2); minislam
does true brute force instead (no vocabulary, a v1 non-goal).
"""

import cv2
import numpy as np

from minislam import geometry
from minislam.frame import Frame
from minislam.map import POPCOUNT, hamming_matrix


def hamming_to_many(d: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Hamming distance from one descriptor (32,) to many (M, 32) -> (M,)."""
    return POPCOUNT[np.bitwise_xor(D, d)].sum(axis=1, dtype=np.int32)


def _best_two(dists: np.ndarray) -> tuple[int, int, int]:
    """(index of best, best distance, second-best distance or a large value)."""
    if len(dists) == 1:
        return 0, int(dists[0]), 10 ** 6
    two = np.argpartition(dists, 1)[:2]
    a, b = (two[0], two[1]) if dists[two[0]] <= dists[two[1]] else (two[1], two[0])
    return int(a), int(dists[a]), int(dists[b])


def _unique_greedy(cands: list[tuple[int, int, int]]) -> np.ndarray:
    """From (distance, i, j) candidates keep the best one per i and per j."""
    used_i, used_j, out = set(), set(), []
    for dist, i, j in sorted(cands):
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        out.append((i, j))
    return np.array(out, dtype=int).reshape(-1, 2)


def match_brute_force(desc1: np.ndarray, desc2: np.ndarray, ratio: float, max_distance: int,
                      cross_check: bool) -> np.ndarray:
    """All-pairs matching. Returns (K, 2) index pairs (i into desc1, j into desc2)."""
    if len(desc1) < 2 or len(desc2) < 2:
        return np.zeros((0, 2), dtype=int)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # knnMatch(k=2) gives the best and second best neighbour of each query descriptor.
    knn = bf.knnMatch(desc1, desc2, k=2)
    reverse_best = None
    if cross_check:
        reverse_best = {m[0].queryIdx: m[0].trainIdx for m in bf.knnMatch(desc2, desc1, k=1) if m}
    cands = []
    for pair in knn:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance > max_distance or best.distance >= ratio * second.distance:
            continue
        if reverse_best is not None and reverse_best.get(best.trainIdx) != best.queryIdx:
            continue
        cands.append((int(best.distance), best.queryIdx, best.trainIdx))
    return _unique_greedy(cands)


def search_by_projection(frame: Frame, points_w: np.ndarray, descriptors: np.ndarray,
                         predicted_octaves: np.ndarray, K: np.ndarray, radius_px: float,
                         scale_factor: float, octave_tolerance: int, max_distance: int,
                         ratio: float, width: int, height: int) -> np.ndarray:
    """Match 3D points to keypoints of ``frame`` (whose ``T_cw`` is set) by projection.

    Each point is projected with the frame's pose and compared only with
    unassigned keypoints within ``radius_px * scale_factor^octave`` pixels
    whose octave is within ``octave_tolerance`` of the predicted octave.
    Returns (K, 2) pairs (point index, keypoint index).
    """
    if len(points_w) == 0 or frame.n_features == 0:
        return np.zeros((0, 2), dtype=int)
    uv, z = geometry.project(K, frame.T_cw, points_w)
    valid = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    point_ids = np.nonzero(valid)[0]
    if len(point_ids) == 0:
        return np.zeros((0, 2), dtype=int)
    radii = radius_px * scale_factor ** predicted_octaves[point_ids]
    neighbours = frame.kd_tree.query_ball_point(uv[point_ids], radii)
    free = np.array([mp is None or mp.is_bad for mp in frame.map_points])

    cands = []
    for p, near in zip(point_ids, neighbours):
        if not near:
            continue
        near = np.asarray(near)
        near = near[free[near]
                    & (np.abs(frame.octaves[near] - predicted_octaves[p]) <= octave_tolerance)]
        if len(near) == 0:
            continue
        dists = hamming_to_many(descriptors[p], frame.descriptors[near])
        k, best, second = _best_two(dists)
        if best <= max_distance and best < ratio * second:
            cands.append((best, int(p), int(near[k])))
    return _unique_greedy(cands)


def search_for_triangulation(kf1: Frame, kf2: Frame, K: np.ndarray, max_distance: int,
                             ratio: float, epipolar_sigma_factor: float) -> np.ndarray:
    """Match *unassociated* features of two posed keyframes along epipolar lines.

    A candidate j in kf2 for feature i in kf1 must lie within
    ``epipolar_sigma_factor * sigma_j`` pixels of the epipolar line of i.
    Returns (K, 2) pairs (keypoint index in kf1, keypoint index in kf2).
    """
    idx1 = np.array([i for i, mp in enumerate(kf1.map_points) if mp is None or mp.is_bad], dtype=int)
    idx2 = np.array([i for i, mp in enumerate(kf2.map_points) if mp is None or mp.is_bad], dtype=int)
    if len(idx1) == 0 or len(idx2) == 0:
        return np.zeros((0, 2), dtype=int)
    F21 = geometry.fundamental_from_poses(K, kf1.T_cw, kf2.T_cw)
    lines = geometry.epipolar_lines(F21, kf1.keypoints_un[idx1])  # (N1, 3)
    x2 = np.hstack([kf2.keypoints_un[idx2], np.ones((len(idx2), 1))])  # (N2, 3)
    epi_dist = np.abs(lines @ x2.T)  # (N1, N2) pixel distance of each kp2 to each line
    allowed = epi_dist < epipolar_sigma_factor * np.sqrt(kf2.sigma2[idx2])[None, :]

    D = hamming_matrix(kf1.descriptors[idx1], kf2.descriptors[idx2])
    D = np.where(allowed, D, 10 ** 6)
    cands = []
    for r in np.nonzero(allowed.any(axis=1))[0]:
        k, best, second = _best_two(D[r])
        if best <= max_distance and best < ratio * second:
            cands.append((best, int(idx1[r]), int(idx2[k])))
    return _unique_greedy(cands)

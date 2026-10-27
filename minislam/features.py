"""ORB feature extraction with a grid distribution over an image pyramid.

ORB = oriented FAST corners + rotated BRIEF descriptors (Rublee et al. 2011).

Pipeline per frame:

1. **Scale pyramid.** Level ``l`` is the image shrunk by ``s^l`` where
   ``s = scale_factor`` (1.2). Detecting on every level makes features
   repeatable when the camera moves closer or further away. The feature
   budget is split geometrically, like ORB-SLAM: level ``l`` gets
   ``N (1 - 1/s) / (1 - (1/s)^L) * (1/s)^l`` features (smaller levels have
   less area, so they get fewer).

2. **FAST + grid distribution.** Plain "keep the N strongest corners"
   clumps features on the single most contrasted object, which is bad for
   pose estimation. Instead each level is divided into ``cell_size`` px
   cells and features are taken round-robin: the best corner of every cell,
   then the second best of every cell, and so on until the level budget is
   used. Cells where nothing passes the normal FAST threshold (20) may use
   corners that only pass the low threshold (7), so low-texture regions
   still contribute. (We run FAST once per level at the low threshold and
   use the corner score to tell "strong" from "weak" corners; this is
   equivalent to ORB-SLAM's two-pass detection but much faster in Python.)
   ORB-SLAM3 uses a quadtree for step 2; the grid is a simpler version.

3. **Orientation by intensity centroid.** For a circular patch of radius
   15 px around the corner, the moments ``m10 = sum x I(x, y)`` and
   ``m01 = sum y I(x, y)`` give the direction from the corner to the
   patch's brightness centroid: ``angle = atan2(m01, m10)``. Rotating the
   BRIEF pattern by this angle makes the descriptor rotation invariant.

4. **Descriptors.** 256 binary intensity comparisons in the rotated, blurred
   patch -> 32 bytes. Compared with the Hamming distance.

Each keypoint keeps its pyramid ``octave``; its position uncertainty is
``sigma = s^octave`` pixels (a corner found on a coarse level is less
precisely located), so ``sigma^2 = s^(2 octave)``.

ORB-SLAM3 equivalent: ``ORBextractor.cc`` (``ComputePyramid``,
``ComputeKeyPointsOctTree``, ``DistributeOctTree``, ``IC_Angle``,
``computeOrbDescriptor``).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from minislam.config import FeatureConfig

HALF_PATCH = 15  # radius of the orientation patch (ORB uses a 31x31 patch)


@dataclass(frozen=True)
class Features:
    xy: np.ndarray  # (N, 2) float, level-0 pixel coordinates (raw, distorted image)
    octaves: np.ndarray  # (N,) int pyramid level each keypoint was detected on
    angles: np.ndarray  # (N,) float degrees
    descriptors: np.ndarray  # (N, 32) uint8


def level_sigma2(scale_factor: float, n_levels: int) -> np.ndarray:
    """sigma^2 of a keypoint position per octave: s^(2 o)."""
    return scale_factor ** (2.0 * np.arange(n_levels))


def features_per_level(n_features: int, scale_factor: float, n_levels: int) -> np.ndarray:
    """ORB-SLAM's geometric split of the feature budget over pyramid levels."""
    f = 1.0 / scale_factor
    first = n_features * (1.0 - f) / (1.0 - f ** n_levels)
    budget = np.round(first * f ** np.arange(n_levels)).astype(int)
    budget[-1] = max(n_features - budget[:-1].sum(), 0)  # remainder to the last level
    return budget


def _circle_offsets(radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel offsets (dx, dy) inside a disc of the given radius."""
    r = np.arange(-radius, radius + 1)
    dx, dy = np.meshgrid(r, r)
    inside = dx ** 2 + dy ** 2 <= radius ** 2
    return dx[inside], dy[inside]


class OrbExtractor:
    def __init__(self, cfg: FeatureConfig) -> None:
        self.cfg = cfg
        self.scales = cfg.scale_factor ** np.arange(cfg.n_levels)
        self.budget = features_per_level(cfg.n_features, cfg.scale_factor, cfg.n_levels)
        self.fast = cv2.FastFeatureDetector_create(threshold=cfg.fast_threshold_low,
                                                   nonmaxSuppression=True)
        # nlevels=1: we hand ORB one pyramid level at a time with our own keypoints.
        self.orb = cv2.ORB_create(nlevels=1, edgeThreshold=cfg.edge_threshold,
                                  patchSize=cfg.patch_size)
        self.circle_dx, self.circle_dy = _circle_offsets(HALF_PATCH)

    def extract(self, gray: np.ndarray) -> Features:
        xy, octaves, angles, descs = [], [], [], []
        for level, (scale, budget) in enumerate(zip(self.scales, self.budget)):
            if level == 0:
                img = gray
            else:
                size = (round(gray.shape[1] / scale), round(gray.shape[0] / scale))
                img = cv2.resize(gray, size, interpolation=cv2.INTER_LINEAR)
            pts, resp = self._detect_distributed(img, int(budget))
            if len(pts) == 0:
                continue
            ang = self._ic_angles(img, pts)
            pts, ang, d = self._describe(img, pts, ang, resp)
            xy.append(pts * scale)
            octaves.append(np.full(len(pts), level, dtype=np.int32))
            angles.append(ang)
            descs.append(d)
        if not xy:
            return Features(np.zeros((0, 2)), np.zeros(0, np.int32), np.zeros(0),
                            np.zeros((0, 32), np.uint8))
        return Features(np.concatenate(xy), np.concatenate(octaves), np.concatenate(angles),
                        np.concatenate(descs))

    # ----------------------------------------------------------- internals

    def _detect_distributed(self, img: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray]:
        """FAST corners on one level, distributed over a grid (see module docstring)."""
        h, w = img.shape
        b = self.cfg.edge_threshold
        if budget <= 0 or w <= 2 * b or h <= 2 * b:
            return np.zeros((0, 2)), np.zeros(0)
        # FAST needs a 3 px ring around each pixel, so detect on a slightly larger crop.
        m = 3
        crop = img[b - m:h - b + m, b - m:w - b + m]
        kps = self.fast.detect(crop, None)
        if not kps:
            return np.zeros((0, 2)), np.zeros(0)
        pts = np.array([k.pt for k in kps]) + (b - m)
        resp = np.array([k.response for k in kps])
        inside = (pts[:, 0] >= b) & (pts[:, 0] < w - b) & (pts[:, 1] >= b) & (pts[:, 1] < h - b)
        pts, resp = pts[inside], resp[inside]

        # Assign every corner to a grid cell.
        cs = self.cfg.cell_size
        n_cols = max(int(np.ceil((w - 2 * b) / cs)), 1)
        cell = ((pts[:, 1] - b) // cs).astype(int) * n_cols + ((pts[:, 0] - b) // cs).astype(int)

        # Strong corners pass the normal threshold. Weak ones are only kept in
        # cells that have no strong corner at all (the low-threshold retry).
        strong = resp >= self.cfg.fast_threshold
        cell_has_strong = np.zeros(cell.max() + 1, dtype=bool)
        cell_has_strong[cell[strong]] = True
        keep = strong | ~cell_has_strong[cell]
        pts, resp, cell = pts[keep], resp[keep], cell[keep]

        # Rank corners inside each cell by response (0 = best in its cell)...
        order = np.lexsort((-resp, cell))
        cell_sorted = cell[order]
        first_in_cell = np.searchsorted(cell_sorted, cell_sorted, side="left")
        rank = np.empty(len(order), dtype=int)
        rank[order] = np.arange(len(order)) - first_in_cell
        # ...then take all rank-0 corners, then rank-1, ... (round robin over cells).
        pick = np.lexsort((-resp, rank))[:budget]
        return pts[pick], resp[pick]

    def _ic_angles(self, img: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Intensity-centroid orientation in degrees, vectorized over keypoints."""
        x = np.round(pts[:, 0]).astype(int)
        y = np.round(pts[:, 1]).astype(int)
        patch = img[y[:, None] + self.circle_dy[None, :],
                    x[:, None] + self.circle_dx[None, :]].astype(np.float64)
        m10 = patch @ self.circle_dx
        m01 = patch @ self.circle_dy
        return np.degrees(np.arctan2(m01, m10)) % 360.0

    def _describe(self, img: np.ndarray, pts: np.ndarray, angles: np.ndarray,
                  resp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kps = [cv2.KeyPoint(float(p[0]), float(p[1]), float(self.cfg.patch_size), float(a),
                            float(r), 0, i)
               for i, (p, a, r) in enumerate(zip(pts, angles, resp))]
        # OpenCV may drop keypoints too close to the border; class_id tells us which survived.
        kps, desc = self.orb.compute(img, kps)
        if desc is None or not kps:
            return np.zeros((0, 2)), np.zeros(0), np.zeros((0, 32), np.uint8)
        idx = np.array([k.class_id for k in kps])
        return pts[idx], angles[idx], desc

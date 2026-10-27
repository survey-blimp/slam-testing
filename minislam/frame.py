"""Frame: one image's features, pose estimate and map-point associations.

A Frame is created for every input image. It stores

  * keypoints in raw pixels (for drawing) and undistorted pixels (for all
    geometry), their pyramid octaves and position variance
    ``sigma^2 = scale_factor^(2 octave)``,
  * the 32-byte ORB descriptors,
  * ``T_cw``: the estimated world -> camera pose (``None`` until tracked),
  * ``map_points[i]``: the MapPoint keypoint ``i`` observes, or ``None``,
  * ``outlier[i]``: True if pose optimization rejected that association.

Frames are temporary. Only frames promoted to keyframes (``KeyFrame`` in
``map.py``) become part of the map.

ORB-SLAM3 equivalent: ``Frame.cc`` / ``Frame.h``.
"""

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree

from minislam import geometry

if TYPE_CHECKING:
    from minislam.map import MapPoint


@dataclass(eq=False)  # eq=False keeps identity hashing, so frames can be dict keys
class Frame:
    id: int
    timestamp: float
    image: np.ndarray  # grayscale, only used by the GUI
    keypoints: np.ndarray  # (N, 2) raw pixel coordinates
    keypoints_un: np.ndarray  # (N, 2) undistorted pixel coordinates
    octaves: np.ndarray  # (N,) pyramid level
    descriptors: np.ndarray  # (N, 32) uint8
    sigma2: np.ndarray  # (N,) position variance in px^2
    T_cw: np.ndarray | None = None
    map_points: list["MapPoint | None"] = field(default_factory=list)
    outlier: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def __post_init__(self) -> None:
        n = len(self.keypoints)
        if not self.map_points:
            self.map_points = [None] * n
        if len(self.outlier) != n:
            self.outlier = np.zeros(n, dtype=bool)

    @property
    def n_features(self) -> int:
        return len(self.keypoints)

    @property
    def camera_center(self) -> np.ndarray:
        return geometry.camera_center(self.T_cw)

    @cached_property
    def kd_tree(self) -> cKDTree:
        """Spatial index over undistorted keypoints, for radius searches."""
        return cKDTree(self.keypoints_un)

    def matched_indices(self, include_outliers: bool = False) -> np.ndarray:
        """Indices of keypoints associated with a (good) map point."""
        return np.array([i for i, mp in enumerate(self.map_points)
                         if mp is not None and not mp.is_bad
                         and (include_outliers or not self.outlier[i])], dtype=int)

    def n_tracked_inliers(self) -> int:
        return len(self.matched_indices())

    def clear_matches(self) -> None:
        self.map_points = [None] * self.n_features
        self.outlier = np.zeros(self.n_features, dtype=bool)

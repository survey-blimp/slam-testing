"""Immutable, GUI-safe copy of the SLAM state after one frame.

The SLAM worker thread owns all live objects (frames, keyframes, map
points). The GUI thread must never touch them, because the worker mutates
them concurrently. Instead the worker packs everything the GUI needs into a
``Snapshot`` made only of plain numbers, strings and freshly copied numpy
arrays, and sends it through a Qt signal.

ORB-SLAM3 equivalent: ``FrameDrawer::Update`` and ``MapDrawer``, which copy
the state the Pangolin viewer draws under a mutex.
"""

from dataclasses import dataclass, field

import numpy as np

# Per-keypoint status codes used by the overlay.
UNMATCHED = 0  # feature not associated with any map point
INLIER = 1  # matched to a map point and accepted by pose optimization
OUTLIER = 2  # matched to a map point but rejected (chi-square test)


def _empty(*shape: int) -> np.ndarray:
    return np.zeros((0, *shape), dtype=np.float64)


@dataclass(frozen=True)
class Snapshot:
    frame_id: int
    timestamp: float
    image: np.ndarray  # BGR copy of the input image
    state: str  # tracking state name
    fps: float
    timings_ms: dict[str, float] = field(default_factory=dict)  # extract / track / local_map / total

    # 2D features of the current frame (raw pixel coordinates, as drawn on the image)
    keypoints: np.ndarray = field(default_factory=lambda: _empty(2))
    octaves: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    keypoint_status: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    track_lines: np.ndarray = field(default_factory=lambda: _empty(2, 2))  # last-frame projection -> keypoint
    init_lines: np.ndarray = field(default_factory=lambda: _empty(2, 2))  # reference kp -> current kp

    # counts for the HUD
    n_features: int = 0
    n_tracked_inliers: int = 0
    n_keyframes: int = 0
    n_map_points: int = 0

    # 3D map, world frame
    map_points: np.ndarray = field(default_factory=lambda: _empty(3))
    map_point_in_local_map: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    keyframe_poses_wc: np.ndarray = field(default_factory=lambda: _empty(4, 4))
    current_pose_wc: np.ndarray | None = None
    trajectory_centers: np.ndarray = field(default_factory=lambda: _empty(3))

    message: str = ""  # latest human-readable status (e.g. why initialization failed)
    warnings: tuple[str, ...] = ()  # persistent warnings, e.g. uncalibrated camera

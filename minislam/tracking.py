"""Per-frame tracking state machine.

States: ``NO_IMAGES -> NOT_INITIALIZED -> OK <-> LOST``.

  * NOT_INITIALIZED: feed frames to the two-view ``Initializer`` until it
    builds the first map.
  * OK: estimate the pose of every new frame against the map (milestone 5).

ORB-SLAM3 equivalent: ``Tracking.cc`` (``Tracking::Track``).
"""

import logging
import time
from enum import Enum

import numpy as np

from minislam import geometry
from minislam.camera import PinholeCamera
from minislam.config import Config
from minislam.frame import Frame
from minislam.initializer import Initializer
from minislam.map import KeyFrame, Map, MapPoint

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
        self.reference_kf: KeyFrame | None = None
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
        self.local_keyframes = [kf0, kf1]
        self.local_map_points = self.map.good_points()
        self.state = TrackingState.OK
        self._record_pose(self.initializer.reference)
        self._record_pose(frame)

    def _record_pose(self, frame: Frame) -> None:
        ref = self.reference_kf
        self.trajectory.append((frame.timestamp, ref, frame.T_cw @ geometry.se3_inverse(ref.T_cw)))

    def trajectory_poses_cw(self) -> tuple[list[float], list[np.ndarray]]:
        """Timestamps and current ``T_cw`` of every tracked frame (re-anchored on its keyframe)."""
        ts = [t for t, _, _ in self.trajectory]
        poses = [T_cr @ ref.T_cw for _, ref, T_cr in self.trajectory]
        return ts, poses

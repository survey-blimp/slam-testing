"""SlamSystem: owns every SLAM component and turns images into Snapshots.

Milestone 3: extracts grid-distributed ORB features and reports timing.

ORB-SLAM3 equivalent: ``System.cc`` (``System::TrackMonocular``).
"""

import time
from pathlib import Path

import cv2
import numpy as np

from minislam.camera import PinholeCamera
from minislam.config import Config
from minislam.features import OrbExtractor, level_sigma2
from minislam.frame import Frame
from minislam.io import save_ply, save_trajectory_tum
from minislam.snapshot import Snapshot


class SlamSystem:
    def __init__(self, cfg: Config, camera: PinholeCamera) -> None:
        self.cfg = cfg
        self.camera = camera
        self.extractor = OrbExtractor(cfg.features)
        self.sigma2_per_level = level_sigma2(cfg.features.scale_factor, cfg.features.n_levels)
        self.warnings: tuple[str, ...] = ()
        if not camera.calibrated:
            self.warnings = ("UNCALIBRATED: using HFOV guess, run tools/calibrate.py",)
        self.frame_count = 0
        self._last_call: float | None = None
        self._fps = 0.0

    def _update_fps(self) -> None:
        now = time.perf_counter()
        if self._last_call is not None:
            dt = now - self._last_call
            inst = 1.0 / dt if dt > 0 else 0.0
            self._fps = inst if self._fps == 0.0 else 0.9 * self._fps + 0.1 * inst
        self._last_call = now

    def make_frame(self, timestamp: float, gray: np.ndarray) -> Frame:
        feats = self.extractor.extract(gray)
        frame = Frame(id=self.frame_count, timestamp=timestamp, image=gray,
                      keypoints=feats.xy, keypoints_un=self.camera.undistort_points(feats.xy),
                      octaves=feats.octaves, descriptors=feats.descriptors,
                      sigma2=self.sigma2_per_level[feats.octaves])
        self.frame_count += 1
        return frame

    def process(self, timestamp: float, image_bgr: np.ndarray) -> Snapshot:
        self._update_fps()
        t0 = time.perf_counter()
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        frame = self.make_frame(timestamp, gray)
        t1 = time.perf_counter()
        timings = {"extract": 1000 * (t1 - t0), "track": 0.0, "local_map": 0.0}
        return Snapshot(frame_id=frame.id, timestamp=timestamp, image=image_bgr.copy(),
                        state="NOT_INITIALIZED", fps=self._fps, timings_ms=timings,
                        keypoints=frame.keypoints.copy(), octaves=frame.octaves.copy(),
                        keypoint_status=np.zeros(frame.n_features, dtype=np.int8),
                        n_features=frame.n_features, warnings=self.warnings)

    def reset(self) -> None:
        self.frame_count = 0

    def save(self, out_dir: Path) -> Path:
        """Write the trajectory (TUM) and map points (PLY) into ``out_dir``."""
        save_trajectory_tum(out_dir / "trajectory.txt", [], [])
        save_ply(out_dir / "map_points.ply", np.zeros((0, 3)))
        return out_dir

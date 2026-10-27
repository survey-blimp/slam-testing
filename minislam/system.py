"""SlamSystem: owns every SLAM component and turns images into Snapshots.

Milestone 1 skeleton: it only measures the frame rate and forwards the
image. Later milestones add feature extraction, tracking and mapping.

ORB-SLAM3 equivalent: ``System.cc`` (``System::TrackMonocular``).
"""

import time
from pathlib import Path

import numpy as np

from minislam.camera import PinholeCamera
from minislam.config import Config
from minislam.io import save_ply, save_trajectory_tum
from minislam.snapshot import Snapshot


class SlamSystem:
    def __init__(self, cfg: Config, camera: PinholeCamera) -> None:
        self.cfg = cfg
        self.camera = camera
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

    def process(self, timestamp: float, image_bgr: np.ndarray) -> Snapshot:
        self._update_fps()
        self.frame_count += 1
        return Snapshot(frame_id=self.frame_count, timestamp=timestamp, image=image_bgr.copy(),
                        state="NO_IMAGES", fps=self._fps, warnings=self.warnings)

    def reset(self) -> None:
        self.frame_count = 0

    def save(self, out_dir: Path) -> Path:
        """Write the trajectory (TUM) and map points (PLY) into ``out_dir``."""
        save_trajectory_tum(out_dir / "trajectory.txt", [], [])
        save_ply(out_dir / "map_points.ply", np.zeros((0, 3)))
        return out_dir

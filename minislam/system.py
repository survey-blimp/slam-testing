"""SlamSystem: owns every SLAM component and turns images into Snapshots.

Per image: extract ORB features -> build a Frame (undistorting keypoint
coordinates) -> ``Tracker.process`` (which runs initialization, tracking
and, on new keyframes, local mapping) -> pack a GUI-safe ``Snapshot``.

ORB-SLAM3 equivalent: ``System.cc`` (``System::TrackMonocular``,
``System::Reset``, ``System::SaveTrajectoryTUM``).
"""

import time
from pathlib import Path

import cv2
import numpy as np

from minislam import geometry
from minislam.camera import PinholeCamera
from minislam.config import Config
from minislam.features import OrbExtractor, level_sigma2
from minislam.frame import Frame
from minislam.io import save_ply, save_trajectory_tum
from minislam.map import Map
from minislam.snapshot import INLIER, OUTLIER, UNMATCHED, Snapshot
from minislam.tracking import Tracker, TrackingState


class SlamSystem:
    def __init__(self, cfg: Config, camera: PinholeCamera) -> None:
        self.cfg = cfg
        self.camera = camera
        self.extractor = OrbExtractor(cfg.features)
        self.sigma2_per_level = level_sigma2(cfg.features.scale_factor, cfg.features.n_levels)
        self.map = Map()
        self.tracker = Tracker(cfg, camera, self.map)
        self.warnings: tuple[str, ...] = ()
        if not camera.calibrated:
            self.warnings = ("UNCALIBRATED: using HFOV guess, run tools/calibrate.py",)
        self.frame_count = 0
        self._last_call: float | None = None
        self._fps = 0.0

    # ---------------------------------------------------------------- frames

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
        t_extract = 1000 * (time.perf_counter() - t0)

        prev = self.tracker.last_frame
        prev_T_cw = None if prev is None or prev.T_cw is None else prev.T_cw.copy()
        self.tracker.process(frame)

        timings = {"extract": t_extract, **self.tracker.timings,
                   "total": 1000 * (time.perf_counter() - t0)}
        return self._snapshot(frame, image_bgr, prev_T_cw, timings)

    def reset(self) -> None:
        self.map.clear()
        self.tracker.reset()

    def save(self, out_dir: Path) -> Path:
        """Write the trajectory (TUM), keyframe trajectory and map points (PLY) into ``out_dir``."""
        ts, poses_cw = self.tracker.trajectory_poses_cw()
        save_trajectory_tum(out_dir / "trajectory.txt", ts,
                            [geometry.se3_inverse(T) for T in poses_cw])
        kfs = sorted(self.map.keyframes, key=lambda kf: kf.timestamp)
        save_trajectory_tum(out_dir / "keyframes.txt", [kf.timestamp for kf in kfs],
                            [geometry.se3_inverse(kf.T_cw) for kf in kfs])
        save_ply(out_dir / "map_points.ply", self.map.points_array()[1])
        return out_dir

    # -------------------------------------------------------------- internals

    def _update_fps(self) -> None:
        now = time.perf_counter()
        if self._last_call is not None:
            dt = now - self._last_call
            inst = 1.0 / dt if dt > 0 else 0.0
            self._fps = inst if self._fps == 0.0 else 0.9 * self._fps + 0.1 * inst
        self._last_call = now

    def _snapshot(self, frame: Frame, image_bgr: np.ndarray, prev_T_cw: np.ndarray | None,
                  timings: dict[str, float]) -> Snapshot:
        tr = self.tracker
        status = np.full(frame.n_features, UNMATCHED, dtype=np.int8)
        for i, mp in enumerate(frame.map_points):
            if mp is not None and not mp.is_bad:
                status[i] = OUTLIER if frame.outlier[i] else INLIER

        # Track lines: where each in-use landmark projected in the previous frame.
        track_lines = np.zeros((0, 2, 2))
        inl = np.nonzero(status == INLIER)[0]
        if prev_T_cw is not None and len(inl):
            p_w = np.array([frame.map_points[i].p_w for i in inl])
            uv_prev, z = geometry.project(self.camera.K, prev_T_cw, p_w)
            ok = z > 0
            uv_prev = self.camera.distort_pixels(uv_prev[ok])
            track_lines = np.stack([uv_prev, frame.keypoints[inl[ok]]], axis=1)

        init_lines = np.zeros((0, 2, 2))
        ref = tr.initializer.reference
        if tr.state == TrackingState.NOT_INITIALIZED and ref is not None and len(tr.initializer.matches):
            m = tr.initializer.matches
            if ref is not frame:
                init_lines = np.stack([ref.keypoints[m[:, 0]], frame.keypoints[m[:, 1]]], axis=1)

        ids, pts = self.map.points_array()
        local_ids = {mp.id for mp in tr.local_map_points if not mp.is_bad}
        in_local = np.array([i in local_ids for i in ids], dtype=bool)
        kf_poses = np.array([geometry.se3_inverse(kf.T_cw) for kf in self.map.keyframes]).reshape(-1, 4, 4)

        tracking_ok = tr.state == TrackingState.OK and frame.T_cw is not None
        current = geometry.se3_inverse(frame.T_cw) if tracking_ok else None
        _, poses_cw = tr.trajectory_poses_cw()
        centers = np.array([geometry.camera_center(T) for T in poses_cw]).reshape(-1, 3)

        return Snapshot(
            frame_id=frame.id, timestamp=frame.timestamp, image=image_bgr.copy(),
            state=tr.state.value, fps=self._fps, timings_ms=timings,
            keypoints=frame.keypoints.copy(), octaves=frame.octaves.copy(), keypoint_status=status,
            track_lines=track_lines, init_lines=init_lines,
            n_features=frame.n_features, n_tracked_inliers=int(len(inl)),
            n_keyframes=len(self.map.keyframes), n_map_points=len(ids),
            map_points=pts, map_point_in_local_map=in_local, keyframe_poses_wc=kf_poses,
            current_pose_wc=current, trajectory_centers=centers,
            message=tr.message, warnings=self.warnings)

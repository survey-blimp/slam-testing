"""Pinhole camera model with radial-tangential (Brown-Conrady) distortion.

A 3D point in the camera frame ``p_c = (X, Y, Z)`` (x right, y down, z forward)
projects to pixel coordinates

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

which is ``uv ~ K p_c`` with the intrinsic matrix

    K = [[fx, 0, cx],
         [0, fy, cy],
         [0,  0,  1]].

Real lenses add distortion. With normalized coordinates ``x = X/Z, y = Y/Z``
and ``r^2 = x^2 + y^2`` the OpenCV model (k1, k2, p1, p2, k3) is

    x_d = x (1 + k1 r^2 + k2 r^4 + k3 r^6) + 2 p1 x y + p2 (r^2 + 2 x^2)
    y_d = y (1 + k1 r^2 + k2 r^4 + k3 r^6) + p1 (r^2 + 2 y^2) + 2 p2 x y

minislam never undistorts whole images. Keypoints are detected on the raw
image and only their *coordinates* are undistorted (``undistort_points``);
all geometry then works on undistorted pixels with the plain pinhole model.

ORB-SLAM3 equivalent: ``CameraModels/Pinhole.cc`` plus the keypoint
undistortion in ``Frame::UndistortKeyPoints``.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PinholeCamera:
    width: int
    height: int
    K: np.ndarray  # (3, 3) intrinsic matrix
    dist: np.ndarray  # (5,) k1, k2, p1, p2, k3
    calibrated: bool  # False when using the HFOV fallback

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def has_distortion(self) -> bool:
        return bool(np.any(self.dist != 0.0))

    # ----------------------------------------------------------- construction

    @staticmethod
    def from_intrinsics(width: int, height: int, fx: float, fy: float, cx: float, cy: float,
                        dist: np.ndarray | None = None, calibrated: bool = True) -> "PinholeCamera":
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        d = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64).reshape(5)
        return PinholeCamera(width, height, K, d, calibrated)

    @staticmethod
    def uncalibrated(width: int, height: int, hfov_deg: float) -> "PinholeCamera":
        """Guess intrinsics from a horizontal field of view: fx = fy = W / (2 tan(HFOV/2))."""
        f = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
        return PinholeCamera.from_intrinsics(width, height, f, f, width / 2.0, height / 2.0,
                                             calibrated=False)

    @staticmethod
    def from_yaml(path: Path) -> "PinholeCamera":
        with Path(path).open("r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return PinholeCamera.from_intrinsics(
            int(d["image_width"]), int(d["image_height"]),
            float(d["fx"]), float(d["fy"]), float(d["cx"]), float(d["cy"]),
            np.array(d["distortion"], dtype=np.float64))

    def to_yaml(self, path: Path, extra: dict | None = None) -> None:
        data = {
            "image_width": self.width,
            "image_height": self.height,
            "fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
            "distortion": [float(v) for v in self.dist],  # k1, k2, p1, p2, k3
        }
        if extra:
            data.update(extra)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as f:
            f.write("# Pinhole intrinsics + OpenCV distortion (k1, k2, p1, p2, k3).\n")
            yaml.safe_dump(data, f, sort_keys=False)

    def scaled_to(self, width: int, height: int) -> "PinholeCamera":
        """Rescale intrinsics to another resolution (only valid for the same aspect ratio)."""
        sx, sy = width / self.width, height / self.height
        K = self.K.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        return PinholeCamera(width, height, K, self.dist.copy(), self.calibrated)

    # ------------------------------------------------------------ projection

    def project(self, p_c: np.ndarray) -> np.ndarray:
        """Project (N, 3) camera-frame points to (N, 2) undistorted pixels."""
        z = p_c[:, 2]
        u = self.fx * p_c[:, 0] / z + self.cx
        v = self.fy * p_c[:, 1] / z + self.cy
        return np.stack([u, v], axis=1)

    def distort_pixels(self, uv: np.ndarray) -> np.ndarray:
        """Map undistorted pixels (N, 2) to where they appear on the raw image."""
        if not self.has_distortion:
            return uv.copy()
        k1, k2, p1, p2, k3 = self.dist
        x = (uv[:, 0] - self.cx) / self.fx
        y = (uv[:, 1] - self.cy) / self.fy
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.stack([xd * self.fx + self.cx, yd * self.fy + self.cy], axis=1)

    def undistort_points(self, uv: np.ndarray) -> np.ndarray:
        """Map raw pixels (N, 2) to undistorted pixels (N, 2)."""
        if len(uv) == 0 or not self.has_distortion:
            return np.asarray(uv, dtype=np.float64).copy()
        # BLACKBOX: cv2.undistortPoints inverts the distortion polynomial above,
        # which has no closed form. It iterates x <- (x_d - tangential(x)) / radial(x)
        # a few times starting from x = x_d. With P=K the result is converted back
        # to pixels. A by-hand version is a ~10 line fixed-point loop.
        pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.undistortPoints(pts, self.K, self.dist, P=self.K)
        return out.reshape(-1, 2)


def load_camera(path: Path | None, width: int, height: int, hfov_deg: float) -> PinholeCamera:
    """Load ``camera.yaml`` if it exists, else fall back to an HFOV guess.

    ``width``/``height`` is the resolution the frames actually have. If the
    calibration was made at another resolution the intrinsics are rescaled
    with a warning (calibrate at the SLAM resolution to avoid this).
    """
    if path is not None and Path(path).exists():
        cam = PinholeCamera.from_yaml(Path(path))
        if (cam.width, cam.height) != (width, height):
            log.warning("camera.yaml is for %dx%d but frames are %dx%d; rescaling intrinsics. "
                        "Recalibrate at the SLAM resolution for best results.",
                        cam.width, cam.height, width, height)
            cam = cam.scaled_to(width, height)
        log.info("Loaded calibration %s: fx=%.1f fy=%.1f cx=%.1f cy=%.1f", path,
                 cam.fx, cam.fy, cam.cx, cam.cy)
        return cam
    log.warning("No camera calibration found (%s); using HFOV=%.0f deg fallback. "
                "Run tools/calibrate.py for real accuracy.", path, hfov_deg)
    return PinholeCamera.uncalibrated(width, height, hfov_deg)

"""Checkerboard camera calibration -> config/camera.yaml.

Usage (print a checkerboard, measure one square):

    python tools/calibrate.py                       # webcam 0, 9x6 inner corners, 25 mm squares
    python tools/calibrate.py --cols 9 --rows 6 --square-mm 24.5 --camera-index 1
    python tools/calibrate.py --drone               # Wi-Fi drone camera (laptop on the drone's Wi-Fi)
    python tools/calibrate.py --images path\\to\\saved_board_photos

Live mode shows the camera feed. When the board is detected *and* its
position in the image differs enough from every view captured so far, the
view is captured automatically (so tilt and move the board around: corners,
edges, close, far, tilted). After ``--views`` captures it calibrates.

Pipeline:
  1. ``findChessboardCorners``: rough corner locations (pixel accuracy).
  2. ``cornerSubPix``: refine each corner to ~0.1 px using image gradients.
  3. ``calibrateCamera``: solve for K, distortion and one pose per view.

The RMS reprojection error is reported; above 1.0 px usually means blurry
views, a board that is not flat, or the wrong square size / corner count.

Calibrate at the SAME resolution SLAM will use: the capture resolution
comes from ``config/default.yaml`` (``source.width/height``), the same
setting ``run.py`` uses.

ORB-SLAM3 equivalent: none (ORB-SLAM3 expects a calibrated settings file).
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minislam.camera import PinholeCamera  # noqa: E402
from minislam.config import load_config  # noqa: E402
from minislam.drone import DroneSource  # noqa: E402
from minislam.source import IMAGE_SUFFIXES, WebcamSource  # noqa: E402

log = logging.getLogger("calibrate")

DETECT_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cols", type=int, default=9, help="inner corners per row")
    p.add_argument("--rows", type=int, default=6, help="inner corners per column")
    p.add_argument("--square-mm", type=float, default=25.0, help="checkerboard square size (mm)")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--drone", action="store_true",
                   help="use the Wi-Fi drone camera (see the drone: section of the config) instead of a webcam")
    p.add_argument("--images", type=Path, default=None, help="calibrate from a folder of images instead")
    p.add_argument("--views", type=int, default=20, help="number of views to capture")
    p.add_argument("--min-motion-px", type=float, default=40.0,
                   help="mean corner displacement vs. every previous capture needed to accept a new view")
    p.add_argument("--subpix-window", type=int, default=11, help="cornerSubPix half-window (px)")
    p.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p.add_argument("--output", type=Path, default=Path("config/camera.yaml"))
    return p.parse_args()


def board_points(cols: int, rows: int, square_m: float) -> np.ndarray:
    """3D corner coordinates on the board plane (z = 0), in metres."""
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    return np.hstack([grid * square_m, np.zeros((len(grid), 1), np.float32)])


def detect(gray: np.ndarray, pattern: tuple[int, int], subpix_window: int,
           fast: bool) -> np.ndarray | None:
    flags = DETECT_FLAGS if fast else DETECT_FLAGS & ~cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
    if not found:
        return None
    w = subpix_window
    return cv2.cornerSubPix(gray, corners, (w, w), (-1, -1), SUBPIX_CRITERIA)


def differs_enough(corners: np.ndarray, captured: list[np.ndarray], min_motion_px: float) -> bool:
    """Proxy for "the board pose is new": its corners moved enough in the image."""
    for prev in captured:
        if np.mean(np.linalg.norm(corners - prev, axis=2)) < min_motion_px:
            return False
    return True


def calibrate(image_points: list[np.ndarray], obj: np.ndarray,
              size: tuple[int, int]) -> tuple[float, np.ndarray, np.ndarray]:
    object_points = [obj] * len(image_points)
    # BLACKBOX: cv2.calibrateCamera (Zhang's method). It first gets a closed-form
    # estimate of K from the homographies between the board plane and each
    # image, then runs Levenberg-Marquardt over K, the distortion coefficients
    # and one 6-DOF board pose per view to minimise the corner reprojection
    # error. By hand: homography DLT per view, solve the linear system for the
    # image of the absolute conic (B = K^-T K^-1), extract K, then a bundle
    # adjustment much like minislam/optimizer.py with distortion added.
    rms, K, dist, _, _ = cv2.calibrateCamera(object_points, image_points, size, None, None)
    return rms, K, dist.reshape(-1)[:5]


def capture_live(args: argparse.Namespace, pattern: tuple[int, int]) -> tuple[list[np.ndarray], tuple[int, int]]:
    cfg = load_config(args.config)
    cam = DroneSource(cfg.drone) if args.drone else WebcamSource(args.camera_index, cfg.source)
    captured: list[np.ndarray] = []
    log.info("Show the board to the camera; views are captured automatically. Q / Esc aborts.")
    try:
        while len(captured) < args.views:
            item = cam.read()
            if item is None:
                raise RuntimeError("camera stopped delivering frames")
            _, img = item
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners = detect(gray, pattern, args.subpix_window, fast=True)
            view = img.copy()
            if corners is not None:
                cv2.drawChessboardCorners(view, pattern, corners, True)
                if differs_enough(corners, captured, args.min_motion_px):
                    captured.append(corners)
                    view[:] = 255 - view  # flash to show a capture
                    log.info("captured view %d/%d", len(captured), args.views)
            cv2.putText(view, f"views {len(captured)}/{args.views}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("calibrate", view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                raise SystemExit("aborted")
        return captured, (cam.width, cam.height)
    finally:
        cam.close()
        cv2.destroyAllWindows()


def capture_folder(args: argparse.Namespace, pattern: tuple[int, int]) -> tuple[list[np.ndarray], tuple[int, int]]:
    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    captured, size = [], None
    for f in files:
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if size is None:
            size = (gray.shape[1], gray.shape[0])
        elif size != (gray.shape[1], gray.shape[0]):
            raise RuntimeError(f"{f.name} has a different resolution; all views must match")
        corners = detect(gray, pattern, args.subpix_window, fast=False)
        log.info("%s: %s", f.name, "board found" if corners is not None else "no board")
        if corners is not None:
            captured.append(corners)
    return captured, size


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    pattern = (args.cols, args.rows)
    if args.images is not None:
        captured, size = capture_folder(args, pattern)
    else:
        captured, size = capture_live(args, pattern)
    if len(captured) < 3:
        log.error("Only %d usable views; need at least 3 (20+ recommended)", len(captured))
        return 1

    rms, K, dist = calibrate(captured, board_points(*pattern, args.square_mm / 1000.0), size)
    log.info("RMS reprojection error: %.3f px over %d views", rms, len(captured))
    if rms > 1.0:
        log.warning("RMS above 1.0 px: check focus/blur, board flatness, corner count and square size")
    log.info("fx=%.2f fy=%.2f cx=%.2f cy=%.2f dist=%s", K[0, 0], K[1, 1], K[0, 2], K[1, 2],
             np.round(dist, 5).tolist())
    cam = PinholeCamera(size[0], size[1], K, dist, calibrated=True)
    cam.to_yaml(args.output, extra={"rms_reprojection_error_px": float(rms), "n_views": len(captured)})
    log.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

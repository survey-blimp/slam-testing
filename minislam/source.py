"""Frame sources: a live webcam, a video file, or a folder of images.

Every source yields ``(timestamp_seconds, bgr_image)`` pairs from ``read()``
and returns ``None`` when it runs out of frames.

  * Webcam: opened with DirectShow (``CAP_DSHOW``), falling back to Media
    Foundation (``CAP_MSMF``). Timestamps come from ``time.perf_counter()``.
    We request a resolution, then check what the camera actually delivers.
    Autofocus can be locked (many webcams ignore this; we log the result).
    Auto-exposure is explicitly switched **on** by default: the camera
    remembers the mode after the program exits, so a camera that an earlier
    run left in manual exposure comes back near-black until something turns
    auto back on. If exposure is locked (``disable_auto_exposure: true``),
    ``close()`` switches auto-exposure back on for the same reason.
  * Video file: timestamps come from a ``<video>.timestamps.txt`` written by
    ``tools/record.py`` if present, otherwise from the container's clock.
  * Image folder: a TUM RGB-D sequence with ``rgb.txt`` (``timestamp path``
    per line). Without ``rgb.txt`` the sorted images are played at 30 Hz.

ORB-SLAM3 equivalent: none in the library itself; this is the job of its
example programs (``Examples/Monocular/mono_tum.cc`` etc.).
"""

import logging
import time
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from minislam.config import SourceConfig

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

# OpenCV's DirectShow/V4L convention for CAP_PROP_AUTO_EXPOSURE (not tunables).
AUTO_EXPOSURE_MANUAL = 0.25
AUTO_EXPOSURE_AUTO = 0.75


class FrameSource(Protocol):
    is_live: bool

    def read(self) -> tuple[float, np.ndarray] | None: ...

    def close(self) -> None: ...


class WebcamSource:
    is_live = True

    def __init__(self, index: int, cfg: SourceConfig) -> None:
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        backend = "DSHOW"
        if not self.cap.isOpened():
            log.warning("DirectShow could not open camera %d, trying Media Foundation", index)
            self.cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
            backend = "MSMF"
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam {index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        if cfg.disable_autofocus:
            self._request(cv2.CAP_PROP_AUTOFOCUS, 0, "autofocus off")
        self.exposure_locked = cfg.disable_auto_exposure
        if self.exposure_locked:
            self._request(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE_MANUAL, "auto-exposure off")
        else:
            self._request(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE_AUTO, "auto-exposure on")

        ok, img = self.cap.read()
        if not ok:
            raise RuntimeError(f"Webcam {index} opened but returned no frame")
        h, w = img.shape[:2]
        self.width, self.height = w, h
        log.info("Webcam %d (%s): requested %dx%d, delivering %dx%d", index, backend,
                 cfg.width, cfg.height, w, h)
        if (w, h) != (cfg.width, cfg.height):
            log.warning("Camera does not deliver the requested resolution")

    def _request(self, prop: int, value: float, what: str) -> None:
        accepted = self.cap.set(prop, value)
        actual = self.cap.get(prop)
        if not accepted:
            result = "IGNORED"
        elif actual == -1:  # DirectShow often accepts a setting but cannot read it back
            result = "accepted (driver cannot report the value)"
        elif abs(actual - value) < 1e-3:
            result = "took effect"
        else:
            result = f"accepted but driver reports {actual}"
        log.info("Webcam request '%s': %s", what, result)

    def read(self) -> tuple[float, np.ndarray] | None:
        ok, img = self.cap.read()
        if not ok:
            return None
        return time.perf_counter(), img

    def close(self) -> None:
        if self.exposure_locked:
            # The camera keeps the manual mode after we exit; hand it back in auto.
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE_AUTO)
            log.info("Webcam: auto-exposure switched back on")
        self.cap.release()


class VideoFileSource:
    is_live = False

    def __init__(self, path: Path) -> None:
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video {path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.timestamps: list[float] | None = None
        ts_file = path.with_suffix(".timestamps.txt")
        if ts_file.exists():
            self.timestamps = [float(line.split()[0]) for line in ts_file.read_text().splitlines()
                               if line.strip() and not line.startswith("#")]
            log.info("Using %d timestamps from %s", len(self.timestamps), ts_file)
        self.index = 0

    def read(self) -> tuple[float, np.ndarray] | None:
        ok, img = self.cap.read()
        if not ok:
            return None
        if self.timestamps is not None and self.index < len(self.timestamps):
            ts = self.timestamps[self.index]
        else:
            ts = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self.index += 1
        return ts, img

    def close(self) -> None:
        self.cap.release()


class ImageFolderSource:
    is_live = False

    def __init__(self, folder: Path) -> None:
        rgb_txt = folder / "rgb.txt"
        if rgb_txt.exists():
            self.items = []
            for line in rgb_txt.read_text().splitlines():
                if not line.strip() or line.startswith("#"):
                    continue
                ts, rel = line.split()[:2]
                self.items.append((float(ts), folder / rel))
        else:
            files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            self.items = [(i / 30.0, p) for i, p in enumerate(files)]
        if not self.items:
            raise RuntimeError(f"No images found in {folder}")
        first = cv2.imread(str(self.items[0][1]))
        self.height, self.width = first.shape[:2]
        self.index = 0
        log.info("Image folder %s: %d frames (%dx%d)", folder, len(self.items), self.width, self.height)

    def read(self) -> tuple[float, np.ndarray] | None:
        if self.index >= len(self.items):
            return None
        ts, path = self.items[self.index]
        self.index += 1
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Could not read {path}")
        return ts, img

    def close(self) -> None:
        pass


def open_source(spec: str, cfg: SourceConfig) -> WebcamSource | VideoFileSource | ImageFolderSource:
    """``spec`` is a webcam index ("0"), a video file, or an image folder."""
    if spec.isdigit():
        return WebcamSource(int(spec), cfg)
    path = Path(spec)
    if path.is_dir():
        return ImageFolderSource(path)
    if path.is_file():
        return VideoFileSource(path)
    raise FileNotFoundError(f"Source '{spec}' is not a webcam index, video file, or folder")

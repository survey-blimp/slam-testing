"""Draw landmarks and the HUD onto the video frame (OpenCV), then make a QImage.

Colour code (BGR):
  * green circle : feature matched to a map point and an inlier (landmark in use)
  * red circle   : matched, but rejected as an outlier by pose optimization
  * grey dot     : feature with no map point
Circle radius grows with the pyramid octave, so coarse-scale features are
drawn bigger. During initialization, lines join reference and current matches.

ORB-SLAM3 equivalent: ``FrameDrawer.cc`` (``DrawFrame`` / ``DrawTextInfo``).
"""

import cv2
import numpy as np
from PySide6.QtGui import QImage

from minislam.snapshot import INLIER, OUTLIER, Snapshot

GREEN = (0, 220, 0)
RED = (0, 0, 255)
GREY = (160, 160, 160)
YELLOW = (0, 220, 255)
ORANGE = (0, 140, 255)


def draw_overlay(snap: Snapshot, show_track_lines: bool) -> np.ndarray:
    img = snap.image.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for (p_ref, p_cur) in snap.init_lines:
        cv2.line(img, _pt(p_ref), _pt(p_cur), YELLOW, 1, cv2.LINE_AA)

    if show_track_lines:
        for (p_prev, p_cur) in snap.track_lines:
            cv2.line(img, _pt(p_prev), _pt(p_cur), ORANGE, 1, cv2.LINE_AA)

    for (x, y), octave, status in zip(snap.keypoints, snap.octaves, snap.keypoint_status):
        c = (int(round(x)), int(round(y)))
        if status == INLIER:
            cv2.circle(img, c, 2 + int(octave), GREEN, 1, cv2.LINE_AA)
        elif status == OUTLIER:
            cv2.circle(img, c, 2 + int(octave), RED, 1, cv2.LINE_AA)
        else:
            cv2.circle(img, c, 2, (40, 40, 40), -1)  # dark ring keeps the dot visible on grey texture
            cv2.circle(img, c, 1, GREY, -1)

    _draw_hud(img, snap)
    return img


def _pt(p: np.ndarray) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _draw_hud(img: np.ndarray, snap: Snapshot) -> None:
    t = snap.timings_ms
    white, warn_red = (255, 255, 255), (0, 0, 255)
    lines = [
        (f"{snap.state}   {snap.fps:4.1f} FPS", white),
        (f"extract {t.get('extract', 0):5.1f} ms  track {t.get('track', 0):5.1f} ms  "
         f"local map {t.get('local_map', 0):5.1f} ms", white),
        (f"features {snap.n_features}  inliers {snap.n_tracked_inliers}  "
         f"KFs {snap.n_keyframes}  MPs {snap.n_map_points}", white),
    ]
    if snap.message:
        lines.append((snap.message, white))
    lines.extend((w, warn_red) for w in snap.warnings)

    font, scale, line_h = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 18
    box_w = max(cv2.getTextSize(text, font, scale, 1)[0][0] for text, _ in lines) + 12
    box_h = line_h * len(lines) + 8
    # semi-transparent dark box behind the text so it is readable on any background
    roi = img[:box_h, :box_w]
    roi[:] = (roi * 0.35).astype(img.dtype)
    y = 16
    for text, color in lines:
        cv2.putText(img, text, (6, y), font, scale, color, 1, cv2.LINE_AA)
        y += line_h


def to_qimage(img_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    # .copy() so the QImage owns its pixels after `rgb` is garbage collected
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

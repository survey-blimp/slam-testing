"""Main window, SLAM worker thread and keyboard shortcuts.

Threading model:
  * ``SlamWorker`` (a ``QThread``) owns the FrameSource and the SlamSystem.
    It loops: read frame -> ``system.process`` -> emit ``Snapshot``.
  * The GUI thread only receives Snapshots. The signal handler just stores
    the newest one ("latest wins"); two timers redraw the video (~30 Hz) and
    the 3D map (~5 Hz) from whatever is newest. Older snapshots are dropped,
    so a slow GUI never slows SLAM down and never blocks the worker.
  * GUI -> worker requests (pause, reset, save, stop) are plain boolean
    flags the worker polls between frames, so ``SlamSystem`` is only ever
    touched by the worker thread.

ORB-SLAM3 equivalent: ``Viewer.cc`` (which runs the drawers in their own
thread and talks to ``System`` through request flags, the same pattern).
"""

import logging

from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QSizePolicy, QSplitter

from minislam.config import Config
from minislam.gui.map_view import MapView
from minislam.gui.overlay import draw_overlay, to_qimage
from minislam.io import timestamped_output_dir
from minislam.snapshot import Snapshot
from minislam.source import FrameSource
from minislam.system import SlamSystem

log = logging.getLogger(__name__)


class SlamWorker(QThread):
    snapshot_ready = Signal(object)
    message = Signal(str)

    def __init__(self, source: FrameSource, system: SlamSystem, cfg: Config,
                 max_frames: int | None) -> None:
        super().__init__()
        self.source = source
        self.system = system
        self.cfg = cfg
        self.max_frames = max_frames
        # Request flags written by the GUI thread, read here. Single bool
        # assignments are atomic in CPython, so no lock is needed.
        self.paused = False
        self.reset_requested = False
        self.save_requested = False
        self.stop_requested = False

    def run(self) -> None:
        n = 0
        source_done = False
        while not self.stop_requested:
            if self.reset_requested:
                self.reset_requested = False
                self.system.reset()
                self.message.emit("SLAM reset")
            if self.save_requested:
                self.save_requested = False
                out = self.system.save(timestamped_output_dir(self.cfg.gui.output_dir))
                self.message.emit(f"Saved to {out}")
            if self.paused or source_done:
                self.msleep(20)
                continue
            item = self.source.read()
            if item is None or (self.max_frames is not None and n >= self.max_frames):
                source_done = True
                self.message.emit("End of input (press S to save, Q to quit)")
                continue
            ts, image = item
            self.snapshot_ready.emit(self.system.process(ts, image))
            n += 1
        self.source.close()


class VideoLabel(QLabel):
    """A QLabel that scales its pixmap to fit while keeping the aspect ratio."""

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: black;")

    def set_frame(self, pixmap: QPixmap) -> None:
        self.setPixmap(pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation))


class MainWindow(QMainWindow):
    def __init__(self, worker: SlamWorker, cfg: Config) -> None:
        super().__init__()
        self.setWindowTitle("minislam")
        self.worker = worker
        self.cfg = cfg
        self.show_track_lines = cfg.gui.show_track_lines

        self.video = VideoLabel()
        self.map_view = MapView(cfg.gui)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.video)
        splitter.addWidget(self.map_view)
        splitter.setSizes([640, 640])
        self.setCentralWidget(splitter)
        self.resize(1400, 600)
        self.statusBar().showMessage("Space pause | R reset | T track lines | F follow | S save | Q quit")

        self._latest: Snapshot | None = None
        self._video_dirty = False
        self._map_dirty = False
        worker.snapshot_ready.connect(self._on_snapshot)
        worker.message.connect(lambda text: self.statusBar().showMessage(text, 5000))

        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self._redraw_video)
        self.video_timer.start(int(1000 / cfg.gui.video_fps))
        self.map_timer = QTimer(self)
        self.map_timer.timeout.connect(self._redraw_map)
        self.map_timer.start(int(1000 / cfg.gui.map_fps))

        self._shortcut(Qt.Key.Key_Space, self._toggle_pause)
        self._shortcut(Qt.Key.Key_R, self._request_reset)
        self._shortcut(Qt.Key.Key_T, self._toggle_tracks)
        self._shortcut(Qt.Key.Key_F, self._toggle_follow)
        self._shortcut(Qt.Key.Key_S, self._request_save)
        self._shortcut(Qt.Key.Key_Q, self.close)
        self._shortcut(Qt.Key.Key_Escape, self.close)

    def _shortcut(self, key: Qt.Key, slot) -> None:
        sc = QShortcut(QKeySequence(key), self)
        sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
        sc.activated.connect(slot)

    # ---------------------------------------------------------------- slots

    def _on_snapshot(self, snap: Snapshot) -> None:
        self._latest = snap  # latest wins; anything older is simply dropped
        self._video_dirty = True
        self._map_dirty = True

    def _redraw_video(self) -> None:
        if not self._video_dirty or self._latest is None:
            return
        self._video_dirty = False
        img = draw_overlay(self._latest, self.show_track_lines)
        self.video.set_frame(QPixmap.fromImage(to_qimage(img)))

    def _redraw_map(self) -> None:
        if not self._map_dirty or self._latest is None:
            return
        self._map_dirty = False
        self.map_view.update_from_snapshot(self._latest)

    def _toggle_pause(self) -> None:
        self.worker.paused = not self.worker.paused
        self.statusBar().showMessage("Paused" if self.worker.paused else "Running", 3000)

    def _request_reset(self) -> None:
        self.worker.reset_requested = True

    def _toggle_tracks(self) -> None:
        self.show_track_lines = not self.show_track_lines
        self._video_dirty = True

    def _toggle_follow(self) -> None:
        self.map_view.follow_camera = not self.map_view.follow_camera
        self.statusBar().showMessage(f"Follow camera: {self.map_view.follow_camera}", 3000)

    def _request_save(self) -> None:
        self.worker.save_requested = True

    def closeEvent(self, event) -> None:
        self.worker.stop_requested = True
        self.worker.wait(3000)
        super().closeEvent(event)


def run_gui(source: FrameSource, system: SlamSystem, cfg: Config, max_frames: int | None) -> int:
    app = QApplication.instance() or QApplication([])
    worker = SlamWorker(source, system, cfg, max_frames)
    window = MainWindow(worker, cfg)
    window.show()
    worker.start()
    return app.exec()

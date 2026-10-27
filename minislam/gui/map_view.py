"""3D map view: map points, keyframe frustums, current camera, trajectory.

Built on ``pyqtgraph.opengl.GLViewWidget``, which gives mouse orbit (left
drag), pan (middle drag) and zoom (wheel) for free.

Display axes: SLAM uses the OpenCV camera convention (x right, y down,
z forward), but pyqtgraph treats +z as "up". To make the scene look upright
we draw every point as ``(x, z, -y)``: the first camera looks along +y of
the display and its "up" (-y) becomes display +z. This is a pure display
transform; the SLAM data is untouched.

ORB-SLAM3 equivalent: ``MapDrawer.cc`` (``DrawMapPoints``, ``DrawKeyFrames``,
``DrawCurrentCamera``) and the Pangolin setup in ``Viewer.cc``.
"""

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtGui import QVector3D

from minislam.config import GuiConfig
from minislam.snapshot import Snapshot

# SLAM world (x right, y down, z forward) -> display (x, forward, up)
DISPLAY_FROM_WORLD = np.array([[1.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0],
                               [0.0, -1.0, 0.0]])

GREY = (0.8, 0.8, 0.8, 0.8)
RED = (1.0, 0.15, 0.15, 1.0)
BLUE = (0.3, 0.5, 1.0, 1.0)
GREEN = (0.1, 1.0, 0.1, 1.0)
TRAJ = (1.0, 0.8, 0.2, 1.0)


def to_display(p_w: np.ndarray) -> np.ndarray:
    return p_w @ DISPLAY_FROM_WORLD.T


def frustum_segments(T_wc: np.ndarray, size: float) -> np.ndarray:
    """Line segments (16, 3) of a camera pyramid in world coordinates.

    The apex is the camera center; the base is a rectangle ``size`` in front
    of it (4:3 aspect), so the pyramid shows where the camera looks.
    """
    w, h, d = size, size * 0.75, size * 1.2
    corners_c = np.array([[-w, -h, d], [w, -h, d], [w, h, d], [-w, h, d]])
    apex_c = np.zeros(3)
    pts_c = []
    for i in range(4):
        pts_c += [apex_c, corners_c[i]]  # apex -> corner
        pts_c += [corners_c[i], corners_c[(i + 1) % 4]]  # base rectangle
    pts_c = np.array(pts_c)
    return pts_c @ T_wc[:3, :3].T + T_wc[:3, 3]


class MapView(gl.GLViewWidget):
    def __init__(self, cfg: GuiConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.follow_camera = cfg.follow_camera
        self.setCameraPosition(distance=4.0, elevation=25, azimuth=-110)

        grid = gl.GLGridItem()
        grid.setSize(10, 10)
        grid.setSpacing(0.5, 0.5)
        grid.translate(0, 0, -1)  # a floor-ish reference a little below the first camera
        self.addItem(grid)
        self.addItem(gl.GLAxisItem(size=QVector3D(0.5, 0.5, 0.5)))  # world origin = first keyframe

        self.points_item = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=cfg.point_size,
                                                color=GREY, pxMode=True)
        self.keyframes_item = gl.GLLinePlotItem(pos=np.zeros((2, 3)), mode="lines", color=BLUE,
                                                width=1.0)
        self.camera_item = gl.GLLinePlotItem(pos=np.zeros((2, 3)), mode="lines", color=GREEN,
                                             width=2.0)
        self.trajectory_item = gl.GLLinePlotItem(pos=np.zeros((2, 3)), mode="line_strip",
                                                 color=TRAJ, width=1.0)
        for item in (self.points_item, self.keyframes_item, self.camera_item, self.trajectory_item):
            item.setVisible(False)
            self.addItem(item)

    def update_from_snapshot(self, snap: Snapshot) -> None:
        if len(snap.map_points):
            colors = np.tile(np.array(GREY), (len(snap.map_points), 1))
            colors[snap.map_point_in_local_map] = RED
            self.points_item.setData(pos=to_display(snap.map_points), color=colors)
            self.points_item.setVisible(True)
        else:
            self.points_item.setVisible(False)

        if len(snap.keyframe_poses_wc):
            segs = np.concatenate([frustum_segments(T, self.cfg.frustum_size)
                                   for T in snap.keyframe_poses_wc])
            self.keyframes_item.setData(pos=to_display(segs))
            self.keyframes_item.setVisible(True)
        else:
            self.keyframes_item.setVisible(False)

        if snap.current_pose_wc is not None:
            segs = frustum_segments(snap.current_pose_wc, self.cfg.frustum_size * 1.5)
            self.camera_item.setData(pos=to_display(segs))
            self.camera_item.setVisible(True)
            if self.follow_camera:
                c = to_display(snap.current_pose_wc[:3, 3][None])[0]
                self.opts["center"] = QVector3D(*c)
                self.update()
        else:
            self.camera_item.setVisible(False)

        if len(snap.trajectory_centers) >= 2:
            self.trajectory_item.setData(pos=to_display(snap.trajectory_centers))
            self.trajectory_item.setVisible(True)
        else:
            self.trajectory_item.setVisible(False)

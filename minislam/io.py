"""Save results: camera trajectory (TUM format) and map points (PLY).

TUM trajectory format, one line per pose (camera-to-world, i.e. ``T_wc``):

    timestamp tx ty tz qx qy qz qw

where ``t`` is the camera center in the world frame and ``q`` the rotation
``R_wc`` as a unit quaternion. This is what ``evo_ape tum`` expects.

PLY is the simplest point-cloud format that MeshLab / CloudCompare open.

ORB-SLAM3 equivalent: ``System::SaveTrajectoryTUM`` /
``System::SaveKeyFrameTrajectoryTUM``.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def timestamped_output_dir(base: str | Path) -> Path:
    """``<base>/<YYYYmmdd_HHMMSS>`` for one saved run."""
    return Path(base) / datetime.now().strftime("%Y%m%d_%H%M%S")


def save_trajectory_tum(path: Path, timestamps: list[float], poses_wc: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, T_wc in zip(timestamps, poses_wc):
            t = T_wc[:3, 3]
            q = Rotation.from_matrix(T_wc[:3, :3]).as_quat()  # (x, y, z, w)
            f.write(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def load_trajectory_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps (N,), positions (N, 3)) from a TUM trajectory file."""
    data = np.loadtxt(path, comments="#", ndmin=2)
    return data[:, 0], data[:, 1:4]


def save_ply(path: Path, points_w: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points_w)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in points_w:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

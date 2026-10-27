"""Synthetic scenes, trajectories and images for the tests.

Two kinds of data:

1. Pure geometry: random 3D points, known camera poses, projected pixels
   with Gaussian noise and a fraction of outlier "matches". Used to test
   initialization, pose optimization and bundle adjustment in isolation.

2. Rendered images: a ray-cast textured room (``TexturedRoom``) seen by a
   camera moving along a known trajectory. Used to run the *whole* system
   (ORB extraction, matching, tracking, mapping) against ground truth
   without a webcam.

Conventions match minislam: OpenCV camera frame (x right, y down,
z forward) and poses ``T_cw`` (world -> camera).
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 640, 480
FX = FY = 457.0  # ~70 deg horizontal FOV at 640 px
CX, CY = WIDTH / 2, HEIGHT / 2
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1.0]])


# --------------------------------------------------------------- geometry

def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def rot_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return cv2.Rodrigues(axis * angle_rad)[0]


def look_at(C_w: np.ndarray, target_w: np.ndarray, up_w=(0.0, -1.0, 0.0)) -> np.ndarray:
    """Pose ``T_cw`` of a camera at ``C_w`` looking at ``target_w`` (y down)."""
    z = target_w - C_w
    z = z / np.linalg.norm(z)
    x = np.cross(-np.asarray(up_w), z)  # right = down x forward (right-handed x = y cross z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R_wc = np.stack([x, y, z], axis=1)
    R_cw = R_wc.T
    return make_T(R_cw, -R_cw @ C_w)


def project(T_cw: np.ndarray, p_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p_c = p_w @ T_cw[:3, :3].T + T_cw[:3, 3]
    uv = np.stack([FX * p_c[:, 0] / p_c[:, 2] + CX, FY * p_c[:, 1] / p_c[:, 2] + CY], axis=1)
    return uv, p_c[:, 2]


def random_points(rng: np.random.Generator, n: int, depth_range=(2.0, 6.0)) -> np.ndarray:
    """Points spread over the field of view of a camera at the origin looking along +z."""
    z = rng.uniform(*depth_range, n)
    u = rng.uniform(40, WIDTH - 40, n)
    v = rng.uniform(40, HEIGHT - 40, n)
    return np.stack([(u - CX) / FX * z, (v - CY) / FY * z, z], axis=1)


def add_noise_and_outliers(rng: np.random.Generator, uv: np.ndarray, noise_px: float,
                           outlier_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian pixel noise plus a fraction of observations moved to random pixels."""
    noisy = uv + rng.normal(0.0, noise_px, uv.shape)
    is_outlier = rng.random(len(uv)) < outlier_frac
    noisy[is_outlier] = rng.uniform([0, 0], [WIDTH, HEIGHT], (is_outlier.sum(), 2))
    return noisy, is_outlier


@dataclass
class TwoViewProblem:
    T_cw1: np.ndarray
    T_cw2: np.ndarray
    p_w: np.ndarray
    uv1: np.ndarray
    uv2: np.ndarray


def two_view_problem(rng: np.random.Generator, n: int = 300, noise_px: float = 0.5,
                     baseline: float = 0.4) -> TwoViewProblem:
    """Camera 1 at the origin, camera 2 translated sideways and slightly rotated."""
    p_w = random_points(rng, n)
    T_cw1 = np.eye(4)
    R_c2w = rot_axis_angle([0.1, 1.0, 0.05], np.radians(4.0))
    C2 = np.array([baseline, 0.05, 0.1])
    T_cw2 = make_T(R_c2w, -R_c2w @ C2)
    uv1, _ = project(T_cw1, p_w)
    uv2, _ = project(T_cw2, p_w)
    uv1 = uv1 + rng.normal(0, noise_px, uv1.shape)
    uv2 = uv2 + rng.normal(0, noise_px, uv2.shape)
    return TwoViewProblem(T_cw1, T_cw2, p_w, uv1, uv2)


def orbit_poses(n: int, radius: float = 0.6) -> list[np.ndarray]:
    """``n`` cameras on an arc, all looking at the center of ``random_points``."""
    poses = []
    for i in range(n):
        a = np.radians(-15 + 30 * i / max(n - 1, 1))
        C = np.array([radius * np.sin(a), 0.1 * np.sin(3 * a), 4.0 - 4.0 * np.cos(a)])
        poses.append(look_at(C, np.array([0.0, 0.0, 4.0])))
    return poses


# ------------------------------------------------------------- rendering

def make_texture(rng: np.random.Generator, size: int = 1024) -> np.ndarray:
    """Random blobs, rectangles and lines: plenty of FAST corners at many scales."""
    low = rng.uniform(60, 190, (size // 64, size // 64)).astype(np.float32)
    tex = cv2.resize(low, (size, size), interpolation=cv2.INTER_CUBIC)
    for _ in range(900):
        color = float(rng.uniform(0, 255))
        x, y = (int(v) for v in rng.integers(0, size, 2))
        s = int(rng.integers(6, 60))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.rectangle(tex, (x, y), (x + s, y + int(rng.integers(6, 60))), color, -1)
        elif kind == 1:
            cv2.circle(tex, (x, y), s // 2, color, -1)
        else:
            x2, y2 = (int(v) for v in rng.integers(-60, 60, 2))
            cv2.line(tex, (x, y), (x + x2, y + y2), color, int(rng.integers(2, 6)))
    tex = cv2.GaussianBlur(tex, (3, 3), 0.8)
    return np.clip(tex, 0, 255).astype(np.float32)


@dataclass
class Plane:
    normal: np.ndarray  # unit normal n, plane is n . X = offset
    offset: float
    axis_u: np.ndarray  # in-plane texture axes
    axis_v: np.ndarray
    texture: np.ndarray
    texel: float  # metres per texel


class TexturedRoom:
    """A closed box room with a different random texture on every wall.

    ``render`` ray-casts each pixel against the six planes and keeps the
    nearest hit, so occlusion and perspective are exact.
    """

    def __init__(self, seed: int = 0, x=(-3.0, 3.0), y=(-2.0, 1.5), z=(-3.0, 5.0),
                 texel: float = 0.006) -> None:
        rng = np.random.default_rng(seed)
        ex, ey, ez = (np.eye(3)[i] for i in range(3))
        specs = [
            (ez, z[1], ex, ey),  # front wall
            (ez, z[0], ex, ey),  # back wall
            (ex, x[0], ez, ey),  # left wall
            (ex, x[1], ez, ey),  # right wall
            (ey, y[1], ex, ez),  # floor (y down)
            (ey, y[0], ex, ez),  # ceiling
        ]
        self.planes = [Plane(n, d, a, b, make_texture(rng), texel) for n, d, a, b in specs]
        u, v = np.meshgrid(np.arange(WIDTH, dtype=np.float64), np.arange(HEIGHT, dtype=np.float64))
        self.rays_c = np.stack([(u - CX) / FX, (v - CY) / FY, np.ones_like(u)], axis=-1)

    def render(self, T_cw: np.ndarray) -> np.ndarray:
        R_wc = T_cw[:3, :3].T
        C = -R_wc @ T_cw[:3, 3]
        rays_w = self.rays_c @ R_wc.T
        best = np.full((HEIGHT, WIDTH), np.inf)
        out = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        for pl in self.planes:
            denom = rays_w @ pl.normal
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (pl.offset - pl.normal @ C) / denom
            hit = np.isfinite(t) & (t > 1e-6) & (t < best)
            if not hit.any():
                continue
            X = C + rays_w * np.where(hit, t, 0.0)[..., None]
            map_u = (X @ pl.axis_u / pl.texel).astype(np.float32)
            map_v = (X @ pl.axis_v / pl.texel).astype(np.float32)
            sample = cv2.remap(pl.texture, map_u, map_v, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_WRAP)
            out[hit] = sample[hit]
            best[hit] = t[hit]
        return np.clip(out, 0, 255).astype(np.uint8)


def room_trajectory(n_frames: int, n_init: int = 30) -> list[np.ndarray]:
    """Sideways translation for initialization, then a smooth loop with yaw.

    Frames ``0..n_init`` slide 0.3 m to the right looking at the front wall
    (monocular init needs translation). The rest is an elliptical loop of
    about 0.5 m x 0.4 m with a gently swinging look-at target.
    """
    poses = []
    for i in range(n_init):
        C = np.array([0.3 * i / n_init, 0.0, 0.0])
        poses.append(look_at(C, C + np.array([0.0, 0.0, 5.0])))
    n_loop = n_frames - n_init
    for i in range(n_loop):
        th = 2 * np.pi * i / n_loop
        C = np.array([-0.2 + 0.5 * np.cos(th), 0.08 * np.sin(2 * th), 0.4 * np.sin(th)])
        target = np.array([C[0] + 0.3 + 0.8 * np.sin(th) - 0.3 * np.cos(th), 0.2 * np.sin(th),
                           5.0])
        poses.append(look_at(C, target))
    return poses


def write_image_folder(folder: Path, n_frames: int, seed: int = 0, fps: float = 30.0) -> None:
    """Render the room trajectory to PNGs + rgb.txt + groundtruth.txt (TUM layout)."""
    from scipy.spatial.transform import Rotation

    room = TexturedRoom(seed)
    poses = room_trajectory(n_frames)
    (folder / "rgb").mkdir(parents=True, exist_ok=True)
    rgb_lines, gt_lines = [], []
    for i, T_cw in enumerate(poses):
        ts = i / fps
        name = f"rgb/{i:05d}.png"
        cv2.imwrite(str(folder / name), room.render(T_cw))
        rgb_lines.append(f"{ts:.6f} {name}")
        T_wc = np.linalg.inv(T_cw)
        q = Rotation.from_matrix(T_wc[:3, :3]).as_quat()
        t = T_wc[:3, 3]
        gt_lines.append(f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                        f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
    (folder / "rgb.txt").write_text("# timestamp filename\n" + "\n".join(rgb_lines) + "\n")
    (folder / "groundtruth.txt").write_text("# timestamp tx ty tz qx qy qz qw\n"
                                            + "\n".join(gt_lines) + "\n")
    cam_yaml = folder / "camera.yaml"
    cam_yaml.write_text(f"image_width: {WIDTH}\nimage_height: {HEIGHT}\nfx: {FX}\nfy: {FY}\n"
                        f"cx: {CX}\ncy: {CY}\ndistortion: [0.0, 0.0, 0.0, 0.0, 0.0]\n")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Render the synthetic room sequence (TUM layout)")
    p.add_argument("folder", type=Path)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    write_image_folder(a.folder, a.frames, a.seed)
    print(f"wrote {a.frames} frames to {a.folder}")

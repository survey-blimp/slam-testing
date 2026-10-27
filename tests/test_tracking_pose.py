from pathlib import Path

import numpy as np

from minislam import geometry
from minislam.config import load_config
from minislam.frame import Frame
from minislam.map import MapPoint
from minislam.optimizer import optimize_pose
from synthetic import K, add_noise_and_outliers, project, random_points, rot_axis_angle

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml").optimizer


def make_frame(rng: np.random.Generator, n: int, outlier_frac: float):
    p_w = random_points(rng, n)
    T_true = geometry.make_T(rot_axis_angle([0.2, 1, 0.1], np.radians(3)), np.array([0.1, -0.05, 0.2]))
    uv, z = project(T_true, p_w)
    keep = (z > 0) & (uv[:, 0] > 0) & (uv[:, 0] < 640) & (uv[:, 1] > 0) & (uv[:, 1] < 480)
    p_w, uv = p_w[keep], uv[keep]
    uv, is_outlier = add_noise_and_outliers(rng, uv, 0.7, outlier_frac)
    octaves = rng.integers(0, 3, len(uv))
    sigma2 = 1.2 ** (2 * octaves)
    frame = Frame(id=0, timestamp=0.0, image=np.zeros((1, 1), np.uint8), keypoints=uv,
                  keypoints_un=uv, octaves=octaves, descriptors=np.zeros((len(uv), 32), np.uint8),
                  sigma2=sigma2)
    frame.map_points = [MapPoint(id=i, p_w=p, first_kf_id=0) for i, p in enumerate(p_w)]
    return frame, T_true, is_outlier, float(np.median(z[keep]))


def perturb(T: np.ndarray, rng: np.random.Generator, angle_deg: float, trans: float) -> np.ndarray:
    dR = rot_axis_angle(rng.normal(size=3), np.radians(angle_deg))
    dt = rng.normal(size=3)
    dt = trans * dt / np.linalg.norm(dt)
    return geometry.make_T(dR, dt) @ T


def test_pose_optimization_converges_with_outliers():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        frame, T_true, is_outlier, depth = make_frame(rng, 300, outlier_frac=0.2)
        frame.T_cw = perturb(T_true, rng, 5.0, 0.1 * depth)
        n_in = optimize_pose(frame, K, CFG.chi2_2dof, CFG.pose_rounds, CFG.pose_max_nfev)

        rot_err = geometry.rotation_angle_deg(frame.T_cw[:3, :3] @ T_true[:3, :3].T)
        center_err = np.linalg.norm(geometry.camera_center(frame.T_cw) - geometry.camera_center(T_true))
        assert rot_err < 0.2, f"seed {seed}: rotation error {rot_err:.3f} deg"
        assert center_err < 0.01 * depth, f"seed {seed}: position error {center_err / depth:.4f} x depth"
        # outlier flags should mostly agree with the injected outliers
        assert np.mean(frame.outlier == is_outlier) > 0.95
        assert n_in > 0.7 * len(is_outlier)

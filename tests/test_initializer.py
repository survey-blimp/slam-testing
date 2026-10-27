from pathlib import Path

import numpy as np

from minislam import geometry
from minislam.config import load_config
from minislam.frame import Frame
from minislam.initializer import Initializer, reconstruct_two_view
from minislam.map import Map
from synthetic import K, project, random_points, rot_axis_angle, two_view_problem

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def frame_from_pixels(frame_id: int, uv: np.ndarray) -> Frame:
    """A Frame whose keypoints are given pixels (octave 0, dummy descriptors)."""
    n = len(uv)
    return Frame(id=frame_id, timestamp=float(frame_id), image=np.zeros((1, 1), np.uint8),
                 keypoints=uv, keypoints_un=uv, octaves=np.zeros(n, dtype=int),
                 descriptors=np.zeros((n, 32), np.uint8), sigma2=np.ones(n))


def initialize(prob) -> tuple[np.ndarray, Map]:
    """Run the full initialization (RANSAC + triangulation + scale + BA) on known matches."""
    ones = np.ones(len(prob.uv1))
    res, reason = reconstruct_two_view(prob.uv1, prob.uv2, ones, ones, K, CFG.initializer,
                                       CFG.optimizer.chi2_2dof)
    assert res is not None, reason
    ref, cur = frame_from_pixels(0, prob.uv1), frame_from_pixels(1, prob.uv2)
    matches = np.stack([np.arange(len(prob.uv1))] * 2, axis=1)[res.match_idx]
    map_ = Map()
    kfs = Initializer(CFG, K).create_initial_map(ref, cur, matches, res, map_)
    assert kfs is not None
    return kfs[1].T_cw, map_


def test_two_view_recovers_relative_pose():
    for seed in range(5):
        prob = two_view_problem(np.random.default_rng(seed), n=300, noise_px=0.7)
        T_21, map_ = initialize(prob)
        T_true = prob.T_cw2 @ geometry.se3_inverse(prob.T_cw1)
        rot_err = geometry.rotation_angle_deg(T_21[:3, :3] @ T_true[:3, :3].T)
        t_err = angle_between_deg(T_21[:3, 3], T_true[:3, 3])
        assert rot_err < 0.5, f"seed {seed}: rotation error {rot_err:.3f} deg"
        assert t_err < 2.0, f"seed {seed}: translation direction error {t_err:.3f} deg"
        # some points are lost to the 1 px RANSAC threshold and the 1 deg parallax cut
        assert len(map_.good_points()) >= 0.6 * 300


def test_scale_is_normalized_to_unit_median_depth():
    prob = two_view_problem(np.random.default_rng(11), n=300, noise_px=0.5)
    _, map_ = initialize(prob)
    depths = np.array([mp.p_w[2] for mp in map_.good_points()])
    assert abs(np.median(depths) - 1.0) < 0.05


def test_pure_rotation_is_rejected():
    rng = np.random.default_rng(7)
    p_w = random_points(rng, 300)
    R = rot_axis_angle([0, 1, 0], np.radians(5))
    T2 = geometry.make_T(R, np.zeros(3))
    uv1, _ = project(np.eye(4), p_w)
    uv2, _ = project(T2, p_w)
    uv1 += rng.normal(0, 0.5, uv1.shape)
    uv2 += rng.normal(0, 0.5, uv2.shape)
    ones = np.ones(len(uv1))
    res, reason = reconstruct_two_view(uv1, uv2, ones, ones, K, CFG.initializer, CFG.optimizer.chi2_2dof)
    assert res is None and ("parallax" in reason or "few" in reason)

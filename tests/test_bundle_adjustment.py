from pathlib import Path

import numpy as np

from minislam import geometry
from minislam.config import load_config
from minislam.map import KeyFrame, Map
from minislam.optimizer import bundle_adjustment
from synthetic import K, orbit_poses, project, random_points, rot_axis_angle

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml").optimizer


def build_problem(rng: np.random.Generator, n_kfs: int = 6, n_points: int = 250):
    """Keyframes on an arc observing random points, with noisy pixels and perturbed estimates."""
    poses = orbit_poses(n_kfs)
    p_true = random_points(rng, n_points, depth_range=(3.0, 5.0))
    map_ = Map()
    obs = []  # per keyframe: (point indices, pixels)
    for T in poses:
        uv, z = project(T, p_true)
        vis = np.nonzero((z > 0) & (uv[:, 0] > 0) & (uv[:, 0] < 640) & (uv[:, 1] > 0) & (uv[:, 1] < 480))[0]
        obs.append((vis, uv[vis] + rng.normal(0, 0.7, (len(vis), 2))))

    kfs = []
    for i, (T, (vis, uv)) in enumerate(zip(poses, obs)):
        T_est = T.copy()
        if i > 0:  # keyframe 0 is the fixed gauge
            dR = rot_axis_angle(rng.normal(size=3), np.radians(0.5))
            T_est = geometry.make_T(dR, rng.normal(0, 0.02, 3)) @ T
        n = len(vis)
        kf = KeyFrame(id=i, timestamp=float(i), image=np.zeros((1, 1), np.uint8), keypoints=uv,
                      keypoints_un=uv, octaves=np.zeros(n, int), descriptors=np.zeros((n, 32), np.uint8),
                      sigma2=np.ones(n), T_cw=T_est, kf_id=map_.new_keyframe_id())
        map_.add_keyframe(kf)
        kfs.append(kf)

    points = [map_.new_map_point(p + rng.normal(0, 0.05, 3), kfs[0]) for p in p_true]
    for kf, (vis, _) in zip(kfs, obs):
        for k, j in enumerate(vis):
            map_.add_observation(points[j], kf, k)
    return kfs, points, poses, p_true


def errors_after_similarity(kfs, points, poses_true, p_true) -> tuple[float, float]:
    """RMSE of camera centers and points after aligning the estimate to ground truth (Sim3)."""
    C_est = np.array([kf.camera_center for kf in kfs])
    C_true = np.array([geometry.camera_center(T) for T in poses_true])
    P_est = np.array([mp.p_w for mp in points])
    s, R, t = geometry.umeyama_alignment(np.vstack([C_est, P_est]), np.vstack([C_true, p_true]))
    align = lambda X: s * X @ R.T + t  # noqa: E731
    rmse = lambda a, b: float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))  # noqa: E731
    return rmse(align(C_est), C_true), rmse(align(P_est), p_true)


def test_bundle_adjustment_reduces_cost_and_error():
    rng = np.random.default_rng(0)
    kfs, points, poses_true, p_true = build_problem(rng)
    cam_err0, pt_err0 = errors_after_similarity(kfs, points, poses_true, p_true)

    result = bundle_adjustment(kfs[1:], kfs[:1], points, K, CFG.chi2_2dof, CFG.ba_max_nfev)
    cam_err1, pt_err1 = errors_after_similarity(kfs, points, poses_true, p_true)

    assert result.final_cost < result.initial_cost
    assert cam_err1 < cam_err0 and pt_err1 < pt_err0
    # the fixed keyframe really stays fixed
    assert np.allclose(kfs[0].T_cw, poses_true[0])


def test_bundle_adjustment_converges_to_noise_level():
    # The perturbations above put most residuals (~6 px) in the linear part of the
    # Huber loss, where scipy's trf makes slow progress; with enough iterations it
    # must still reach the accuracy the 0.7 px pixel noise allows.
    rng = np.random.default_rng(0)
    kfs, points, poses_true, p_true = build_problem(rng)
    cam_err0, pt_err0 = errors_after_similarity(kfs, points, poses_true, p_true)
    result = bundle_adjustment(kfs[1:], kfs[:1], points, K, CFG.chi2_2dof, max_nfev=300)
    cam_err1, pt_err1 = errors_after_similarity(kfs, points, poses_true, p_true)
    assert cam_err1 < 0.3 * cam_err0 and pt_err1 < 0.3 * pt_err0
    # with only Gaussian noise almost every observation passes the chi-square test
    assert np.mean(result.chi2 <= CFG.chi2_2dof) > 0.9


def test_sparsity_pattern_shape():
    from minislam.optimizer import ba_sparsity

    cam_idx = np.array([0, 1, 2])  # camera 2 is fixed (only 2 optimized poses)
    pt_idx = np.array([0, 0, 1])
    A = ba_sparsity(3, 2, 2, cam_idx, pt_idx).toarray()
    assert A.shape == (6, 2 * 6 + 2 * 3)
    assert A[0, :6].all() and not A[0, 6:12].any()  # obs 0 -> pose 0 only
    assert A[4, :12].sum() == 0  # obs 2 is from the fixed camera: no pose columns
    assert A[4, 12 + 3:].all() and not A[4, 12:15].any()  # ...and point 1 only

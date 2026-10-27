import numpy as np

from minislam import geometry
from synthetic import K, project, random_points, rot_axis_angle, two_view_problem


def random_pose(rng: np.random.Generator) -> np.ndarray:
    R = rot_axis_angle(rng.normal(size=3), rng.uniform(0, np.pi))
    return geometry.make_T(R, rng.normal(size=3))


def test_se3_inverse_and_compose_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        A, B = random_pose(rng), random_pose(rng)
        assert np.allclose(A @ geometry.se3_inverse(A), np.eye(4), atol=1e-12)
        assert np.allclose(geometry.se3_inverse(A @ B), geometry.se3_inverse(B) @ geometry.se3_inverse(A))
        assert np.allclose(geometry.vec_to_pose(geometry.pose_to_vec(A)), A, atol=1e-12)


def test_camera_center():
    rng = np.random.default_rng(1)
    T_cw = random_pose(rng)
    C = geometry.camera_center(T_cw)
    assert np.allclose(geometry.transform(T_cw, C[None])[0], 0.0, atol=1e-12)


def test_triangulation_noise_free_is_exact():
    rng = np.random.default_rng(2)
    prob = two_view_problem(rng, n=200, noise_px=0.0)
    uv1, _ = project(prob.T_cw1, prob.p_w)
    uv2, _ = project(prob.T_cw2, prob.p_w)
    p = geometry.triangulate(K, prob.T_cw1, prob.T_cw2, uv1, uv2)
    assert np.max(np.abs(p - prob.p_w)) < 1e-6


def test_triangulation_checks_reject_bad_points():
    rng = np.random.default_rng(3)
    prob = two_view_problem(rng, n=100, noise_px=0.0)
    p = prob.p_w.copy()
    p[0] = -p[0]  # behind both cameras
    p[1] += np.array([0.5, 0.0, 0.0])  # reprojects far away
    good, _ = geometry.check_triangulation(K, prob.T_cw1, prob.T_cw2, p, prob.uv1, prob.uv2,
                                           np.ones(100), np.ones(100), 5.991, 0.5)
    assert not good[0] and not good[1] and good[2:].mean() > 0.9


def test_epipolar_constraint():
    rng = np.random.default_rng(4)
    prob = two_view_problem(rng, n=50, noise_px=0.0)
    F = geometry.fundamental_from_poses(K, prob.T_cw1, prob.T_cw2)
    lines = geometry.epipolar_lines(F, prob.uv1)
    d = np.abs(np.sum(lines * np.hstack([prob.uv2, np.ones((50, 1))]), axis=1))
    assert np.max(d) < 1e-6


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(5)
    src = random_points(rng, 50)
    R = rot_axis_angle([1, 2, 3], 0.7)
    dst = 2.5 * src @ R.T + np.array([1.0, -2.0, 0.5])
    s, R_est, t = geometry.umeyama_alignment(src, dst)
    assert abs(s - 2.5) < 1e-9 and np.allclose(R_est, R) and np.allclose(s * src @ R_est.T + t, dst)

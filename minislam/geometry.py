"""Geometry helpers: SE(3) poses, projection, triangulation, epipolar lines.

Pose conventions (same as ORB-SLAM3):
  * ``T_cw`` (4x4) maps world points into the camera: ``p_c = R_cw p_w + t_cw``.
  * Its inverse is ``T_wc = [[R_cw^T, -R_cw^T t_cw], [0, 1]]``.
  * The camera center in the world is ``C = -R_cw^T t_cw``.
  * Composition: ``T_ac = T_ab @ T_bc``.

For optimization a pose is written as 6 numbers ``(r, t)`` where ``r`` is an
axis-angle (Rodrigues) vector: direction = rotation axis, norm = angle in
radians. This is a minimal, singularity-free parametrization for the small
rotations an optimizer steps through.

Two views of the same point: with relative pose ``R, t`` (camera 1 -> 2), the
essential matrix is ``E = [t]_x R`` and the fundamental matrix
``F = K^-T E K^-1``. A pixel ``x1`` in image 1 must lie on the epipolar line
``l2 = F x1`` in image 2 (``x2^T F x1 = 0``).

ORB-SLAM3 equivalent: ``Thirdparty/Sophus`` for SE(3), ``GeometricTools.cc``
(``ComputeF12``, ``Triangulate``) and the checks in
``TwoViewReconstruction::CheckRT`` / ``LocalMapping::CreateNewMapPoints``.
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# ------------------------------------------------------------------ SE(3)


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Closed-form inverse of a rigid transform (cheaper and exacter than np.linalg.inv)."""
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def camera_center(T_cw: np.ndarray) -> np.ndarray:
    return -T_cw[:3, :3].T @ T_cw[:3, 3]


def transform(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N, 3) points."""
    return p @ T[:3, :3].T + T[:3, 3]


def pose_to_vec(T: np.ndarray) -> np.ndarray:
    """4x4 pose -> 6-vector (axis-angle rotation, translation)."""
    return np.concatenate([Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])


def vec_to_pose(x: np.ndarray) -> np.ndarray:
    return make_T(Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6])


def rotation_angle_deg(R: np.ndarray) -> float:
    """Angle of a rotation matrix, e.g. of R_est @ R_true^T for an error."""
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(R).as_rotvec())))


def skew(v: np.ndarray) -> np.ndarray:
    """[v]_x such that [v]_x w = v x w."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# ------------------------------------------------------------- projection


def project(K: np.ndarray, T_cw: np.ndarray, p_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points (N, 3) -> undistorted pixels (N, 2) and depths (N,)."""
    p_c = transform(T_cw, p_w)
    z = p_c[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([K[0, 0] * p_c[:, 0] / z + K[0, 2], K[1, 1] * p_c[:, 1] / z + K[1, 2]], axis=1)
    return uv, z


def reprojection_chi2(K: np.ndarray, T_cw: np.ndarray, p_w: np.ndarray, uv: np.ndarray,
                      sigma2: np.ndarray) -> np.ndarray:
    """Squared reprojection error normalized by the keypoint variance, ||r||^2 / sigma^2.

    Under Gaussian pixel noise this is chi-square distributed with 2 degrees
    of freedom, so ``chi2 > 5.991`` rejects a measurement at 95% confidence.
    """
    proj, z = project(K, T_cw, p_w)
    err2 = np.sum((proj - uv) ** 2, axis=1)
    err2[~(z > 0)] = np.inf
    return err2 / sigma2


# ---------------------------------------------------------- triangulation


def triangulate(K: np.ndarray, T_cw1: np.ndarray, T_cw2: np.ndarray, uv1: np.ndarray,
                uv2: np.ndarray) -> np.ndarray:
    """Triangulate matched undistorted pixels (N, 2) from two posed views -> (N, 3) world points."""
    if len(uv1) == 0:
        return np.zeros((0, 3))
    P1 = K @ T_cw1[:3, :]
    P2 = K @ T_cw2[:3, :]
    # BLACKBOX: cv2.triangulatePoints solves the linear (DLT) triangulation.
    # Each view gives two equations  u * P[2] - P[0] = 0,  v * P[2] - P[1] = 0
    # in the homogeneous point X. Stacking both views gives A X = 0 (4x4), and
    # X is the right singular vector of A with the smallest singular value.
    # By hand: build A per point and call np.linalg.svd.
    X = cv2.triangulatePoints(P1, P2, uv1.T.astype(np.float64), uv2.T.astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (X[:3] / X[3]).T


def parallax_deg(C1: np.ndarray, C2: np.ndarray, p_w: np.ndarray) -> np.ndarray:
    """Angle at each point between the rays to the two camera centers.

    Small parallax means the two rays are nearly parallel, so depth is poorly
    constrained: points near the epipole or far away, or a too-small baseline.
    """
    r1 = p_w - C1
    r2 = p_w - C2
    cos = np.sum(r1 * r2, axis=1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def check_triangulation(K: np.ndarray, T_cw1: np.ndarray, T_cw2: np.ndarray, p_w: np.ndarray,
                        uv1: np.ndarray, uv2: np.ndarray, sigma2_1: np.ndarray,
                        sigma2_2: np.ndarray, chi2_threshold: float,
                        min_parallax_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Accept a triangulated point only if it is geometrically trustworthy.

    Checks, as in ORB-SLAM:
      1. finite and in front of both cameras (positive depth),
      2. reprojection error passes the chi-square test in both views,
      3. parallax angle >= ``min_parallax_deg``.
    Returns (good mask, parallax in degrees).
    """
    finite = np.all(np.isfinite(p_w), axis=1)
    p = np.where(finite[:, None], p_w, 0.0)
    z1 = transform(T_cw1, p)[:, 2]
    z2 = transform(T_cw2, p)[:, 2]
    chi1 = reprojection_chi2(K, T_cw1, p, uv1, sigma2_1)
    chi2 = reprojection_chi2(K, T_cw2, p, uv2, sigma2_2)
    par = parallax_deg(camera_center(T_cw1), camera_center(T_cw2), p)
    good = (finite & (z1 > 0) & (z2 > 0) & (chi1 <= chi2_threshold) & (chi2 <= chi2_threshold)
            & (par >= min_parallax_deg))
    return good, par


# ---------------------------------------------------------------- epipolar


def fundamental_from_poses(K: np.ndarray, T_cw1: np.ndarray, T_cw2: np.ndarray) -> np.ndarray:
    """F21 with x2^T F21 x1 = 0 for pixels x1 in view 1 and x2 in view 2."""
    T_21 = T_cw2 @ se3_inverse(T_cw1)
    E = skew(T_21[:3, 3]) @ T_21[:3, :3]
    K_inv = np.linalg.inv(K)
    return K_inv.T @ E @ K_inv


def epipolar_lines(F21: np.ndarray, uv1: np.ndarray) -> np.ndarray:
    """Lines (N, 3) in image 2, scaled so that |l . (u, v, 1)| is a distance in pixels."""
    x1 = np.hstack([uv1, np.ones((len(uv1), 1))])
    lines = x1 @ F21.T
    return lines / np.linalg.norm(lines[:, :2], axis=1, keepdims=True)


# --------------------------------------------------------------- alignment


def umeyama_alignment(src: np.ndarray, dst: np.ndarray,
                      with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) minimizing sum ||dst - (s R src + t)||^2 (Umeyama 1991).

    Monocular SLAM recovers the trajectory only up to scale, so it is
    compared with ground truth after this alignment (what ``evo -as`` does).
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / xs.var(0).sum()) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t

"""Nonlinear least squares: pose-only optimization and bundle adjustment.

Everything here minimizes the same **reprojection error**. For a keyframe
(or frame) with pose ``T_cw`` observing world point ``p_w`` at undistorted
pixel ``z`` on octave ``o``:

    p_c = R_cw p_w + t_cw
    pi(p_c) = (fx X/Z + cx, fy Y/Z + cy)
    r = (pi(p_c) - z) / sigma,    sigma = scale_factor^o

Dividing by ``sigma`` (the keypoint's pixel standard deviation) weights
precise fine-scale corners more than coarse ones and makes ``||r||^2``
chi-square distributed with 2 DOF, so ``||r||^2 > 5.991`` flags an outlier
at 95% confidence.

The total cost ``sum_k rho(||r_k||^2)`` uses a **Huber** robust loss
``rho`` (quadratic for small errors, linear for large ones, switch at
``sqrt(5.991)``), so a few bad matches cannot drag the solution.

Parameters: every pose is 6 numbers (axis-angle rotation, translation), every
point 3 numbers. ``scipy.optimize.least_squares`` (trust-region reflective)
does the minimization with finite-difference Jacobians.

* ``optimize_pose``: 6 unknowns, points fixed. Used every frame by tracking.
  4 rounds; after each round all observations are re-classified with the
  chi-square test and only inliers are used in the next round, so a point
  wrongly rejected early can come back (as ORB-SLAM's ``PoseOptimization``).
* ``bundle_adjustment``: poses of some keyframes and all their points jointly.
  Some keyframes are held fixed: at least one, to remove the gauge freedom
  (the whole solution could otherwise slide/rotate freely).

ORB-SLAM3 equivalent: ``Optimizer.cc`` (``PoseOptimization``,
``LocalBundleAdjustment``, ``GlobalBundleAdjustemnt``), which uses g2o with
analytic Jacobians instead of scipy.
"""

import logging
import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

from minislam import geometry
from minislam.frame import Frame
from minislam.map import KeyFrame, MapPoint

log = logging.getLogger(__name__)


def reprojection_residuals(poses: np.ndarray, points: np.ndarray, cam_idx: np.ndarray,
                           pt_idx: np.ndarray, uv: np.ndarray, inv_sigma: np.ndarray,
                           K: np.ndarray) -> np.ndarray:
    """Weighted reprojection residuals for many observations, flattened to (2R,).

    poses: (P, 6) axis-angle + translation (``T_cw``) per camera
    points: (M, 3) world points
    observation k: camera ``cam_idx[k]`` sees point ``pt_idx[k]`` at ``uv[k]``,
    with weight ``inv_sigma[k] = 1 / sigma``.
    """
    R = Rotation.from_rotvec(poses[:, :3]).as_matrix()  # (P, 3, 3)
    p_c = np.einsum("kij,kj->ki", R[cam_idx], points[pt_idx]) + poses[cam_idx, 3:6]
    z = p_c[:, 2]
    z = np.where(np.abs(z) < 1e-6, 1e-6, z)  # avoid division by zero for degenerate points
    u = K[0, 0] * p_c[:, 0] / z + K[0, 2]
    v = K[1, 1] * p_c[:, 1] / z + K[1, 2]
    res = (np.stack([u, v], axis=1) - uv) * inv_sigma[:, None]
    return res.ravel()


def _chi2_per_observation(residuals: np.ndarray) -> np.ndarray:
    """||r_k||^2 per observation from flattened weighted residuals (= error^2 / sigma^2)."""
    r = residuals.reshape(-1, 2)
    return np.sum(r * r, axis=1)


# ------------------------------------------------------------ pose only


def optimize_pose(frame: Frame, K: np.ndarray, chi2_threshold: float, rounds: int,
                  max_nfev: int) -> int:
    """Refine ``frame.T_cw`` from its map-point associations; set ``frame.outlier``.

    Returns the number of inlier associations.
    """
    idx = frame.matched_indices(include_outliers=True)
    if len(idx) < 4:
        frame.outlier[idx] = True
        return 0
    t0 = time.perf_counter()
    p_w = np.array([frame.map_points[i].p_w for i in idx])
    uv = frame.keypoints_un[idx]
    inv_sigma = 1.0 / np.sqrt(frame.sigma2[idx])
    cam_idx = np.zeros(len(idx), dtype=int)
    pt_idx = np.arange(len(idx))

    def residuals(x: np.ndarray, use: np.ndarray) -> np.ndarray:
        return reprojection_residuals(x[None, :], p_w, cam_idx[use], pt_idx[use], uv[use],
                                      inv_sigma[use], K)

    x = geometry.pose_to_vec(frame.T_cw)
    all_obs = np.ones(len(idx), dtype=bool)
    cost0 = 0.5 * float(np.sum(residuals(x, all_obs) ** 2))
    outlier = np.zeros(len(idx), dtype=bool)
    for _ in range(rounds):
        use = ~outlier
        if use.sum() < 4:
            break
        x = least_squares(residuals, x, args=(use,), loss="huber", f_scale=math.sqrt(chi2_threshold),
                          max_nfev=max_nfev).x
        outlier = _chi2_per_observation(residuals(x, all_obs)) > chi2_threshold

    frame.T_cw = geometry.vec_to_pose(x)
    frame.outlier[idx] = outlier
    n_in = int((~outlier).sum())
    cost1 = 0.5 * float(np.sum(residuals(x, ~outlier) ** 2)) if n_in else 0.0
    log.debug("pose opt: %d obs, %d inliers, cost %.1f -> %.1f (inliers), %.1f ms",
              len(idx), n_in, cost0, cost1, 1000 * (time.perf_counter() - t0))
    return n_in


# ------------------------------------------------------ bundle adjustment


@dataclass
class BAResult:
    observations: list[tuple[MapPoint, KeyFrame]]  # one entry per residual pair
    chi2: np.ndarray  # (R,) ||r||^2 / sigma^2 per observation after optimization
    initial_cost: float
    final_cost: float


def ba_sparsity(n_obs: int, n_opt_poses: int, n_points: int, cam_idx: np.ndarray,
                pt_idx: np.ndarray) -> coo_matrix:
    """Jacobian sparsity pattern for bundle adjustment.

    Why sparsity makes BA tractable: with C optimized cameras and M points
    there are 6C + 3M unknowns (thousands) and 2R residuals (tens of
    thousands). A dense Jacobian would have 2R x (6C + 3M) entries, and a
    finite-difference estimate would need one extra residual evaluation per
    unknown. But each residual pair only depends on the 6 parameters of
    *its* camera and the 3 of *its* point: at most 9 non-zeros per row.
    Telling ``least_squares`` this pattern lets it

      * store the Jacobian sparsely (2R x 9 numbers instead of 2R x (6C+3M)),
      * estimate it with very few residual evaluations: columns that never
        share a row (e.g. the x-coordinate of all points, or the first
        pose parameter of all cameras) are perturbed *together*,
      * solve each step with a sparse iterative solver (LSMR).

    ORB-SLAM's g2o goes further and uses the Schur complement on the
    camera-point block structure; the principle is the same.
    """
    rows, cols = [], []
    r = np.arange(n_obs)
    for axis in range(2):  # u and v residual of each observation
        row = 2 * r + axis
        has_pose = cam_idx < n_opt_poses  # observations from fixed cameras have no pose columns
        for k in range(6):
            rows.append(row[has_pose])
            cols.append(6 * cam_idx[has_pose] + k)
        for k in range(3):
            rows.append(row)
            cols.append(6 * n_opt_poses + 3 * pt_idx + k)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)),
                      shape=(2 * n_obs, 6 * n_opt_poses + 3 * n_points))


def bundle_adjustment(opt_keyframes: list[KeyFrame], fixed_keyframes: list[KeyFrame],
                      points: list[MapPoint], K: np.ndarray, chi2_threshold: float,
                      max_nfev: int, label: str = "BA") -> BAResult:
    """Jointly optimize the poses of ``opt_keyframes`` and the positions of ``points``.

    Observations of ``points`` from ``fixed_keyframes`` contribute residuals
    but those poses stay constant. Results are written back in place.
    """
    t0 = time.perf_counter()
    kfs = list(opt_keyframes) + list(fixed_keyframes)
    kf_index = {kf: i for i, kf in enumerate(kfs)}
    n_opt = len(opt_keyframes)

    observations, cam_idx, pt_idx, uv, inv_sigma = [], [], [], [], []
    for j, mp in enumerate(points):
        for kf, kp in mp.observations.items():
            i = kf_index.get(kf)
            if i is None:
                continue
            observations.append((mp, kf))
            cam_idx.append(i)
            pt_idx.append(j)
            uv.append(kf.keypoints_un[kp])
            inv_sigma.append(1.0 / math.sqrt(kf.sigma2[kp]))
    if not observations:
        return BAResult([], np.zeros(0), 0.0, 0.0)
    cam_idx, pt_idx = np.array(cam_idx), np.array(pt_idx)
    uv, inv_sigma = np.array(uv), np.array(inv_sigma)

    poses0 = np.array([geometry.pose_to_vec(kf.T_cw) for kf in kfs])
    fixed_poses = poses0[n_opt:]
    x0 = np.concatenate([poses0[:n_opt].ravel(), np.array([mp.p_w for mp in points]).ravel()])

    def residuals(x: np.ndarray) -> np.ndarray:
        poses = np.vstack([x[:6 * n_opt].reshape(-1, 6), fixed_poses])
        pts = x[6 * n_opt:].reshape(-1, 3)
        return reprojection_residuals(poses, pts, cam_idx, pt_idx, uv, inv_sigma, K)

    sparsity = ba_sparsity(len(observations), n_opt, len(points), cam_idx, pt_idx)
    r0 = residuals(x0)
    result = least_squares(residuals, x0, jac_sparsity=sparsity, loss="huber",
                           f_scale=math.sqrt(chi2_threshold), x_scale="jac", max_nfev=max_nfev,
                           method="trf")
    x = result.x
    for i, kf in enumerate(opt_keyframes):
        kf.T_cw = geometry.vec_to_pose(x[6 * i:6 * i + 6])
    for j, mp in enumerate(points):
        mp.p_w = x[6 * n_opt + 3 * j:6 * n_opt + 3 * j + 3].copy()

    r1 = residuals(x)
    cost0, cost1 = 0.5 * float(r0 @ r0), 0.5 * float(r1 @ r1)
    log.info("%s: %d KFs (%d fixed), %d points, %d obs: cost %.1f -> %.1f, %d evals, %.0f ms",
             label, len(kfs), len(fixed_keyframes), len(points), len(observations), cost0, cost1,
             result.nfev, 1000 * (time.perf_counter() - t0))
    return BAResult(observations, _chi2_per_observation(r1), cost0, cost1)

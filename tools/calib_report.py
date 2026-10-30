"""Uncertainty report for a camera calibration, from the corners saved by calibrate.py.

    python tools/calib_report.py data\\calib_corners_20261026_231500.npz
    python tools/calib_report.py corners.npz --fix-k3 --drop-worst 5 --output config\\camera_new.yaml

Why: a low reprojection error does not mean every parameter is well determined. The
distortion terms k1, k2, k3 can trade off against each other (large values of opposite sign
that nearly cancel) when the views do not cover the image edges. This script quantifies it.

Math: calibration is a nonlinear least-squares problem over the intrinsics theta (fx, fy,
cx, cy, k1, k2, p1, p2, k3) and one 6-DOF pose per view. With residual Jacobian J at the
solution, the parameter covariance is sigma^2 (J^T J)^-1, where sigma^2 is the residual
variance. The per-view poses are nuisance parameters; marginalising them out is the Schur
complement  S = H_cc - sum_i H_ci H_ii^-1 H_ic  (H = J^T J blocks), and cov(theta) =
sigma^2 S^-1. Standard deviations are sqrt(diag); the correlation matrix is cov normalised
by those. |correlation| near 1 means two parameters cannot be told apart from this data.

Also prints the per-view RMS error (worst first) and can re-fit with k3 fixed to zero
(``--fix-k3``) and/or the worst views dropped (``--drop-worst N``).

ORB-SLAM3 equivalent: none.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibrate import board_points  # noqa: E402
from minislam.camera import PinholeCamera  # noqa: E402

log = logging.getLogger("calib_report")

PARAM_NAMES = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]
N_POSE = 6                      # rvec (3) + tvec (3) per view
INTRINSIC_COLS = slice(6, 15)   # projectPoints Jacobian: rvec, tvec, f, c, then 5 distortion terms
HIGH_CORRELATION = 0.95


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("corners", type=Path, help=".npz written by tools/calibrate.py")
    p.add_argument("--fix-k3", action="store_true", help="re-fit with k3 fixed at 0")
    p.add_argument("--drop-worst", type=int, default=0, help="drop the N worst views, then re-fit")
    p.add_argument("--worst", type=int, default=8, help="how many worst views to list")
    p.add_argument("--output", type=Path, default=None, help="write the re-fitted camera.yaml here")
    return p.parse_args()


def fit(views: list[np.ndarray], obj: np.ndarray, size: tuple[int, int], fix_k3: bool):
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    # BLACKBOX: cv2.calibrateCamera, see tools/calibrate.py.
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj] * len(views), views, size, None, None,
                                                     flags=flags)
    return rms, K, dist.reshape(-1)[:5], rvecs, tvecs


def view_errors(views, obj, K, dist, rvecs, tvecs) -> np.ndarray:
    """RMS reprojection error (px) of each view."""
    errs = []
    for pts, r, t in zip(views, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, r, t, K, dist)
        errs.append(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - pts.reshape(-1, 2)) ** 2, axis=1))))
    return np.array(errs)


def intrinsics_covariance(views, obj, K, dist, rvecs, tvecs, free: np.ndarray) -> np.ndarray:
    """Covariance of the free intrinsic parameters (order PARAM_NAMES), poses marginalised out."""
    n_free = int(free.sum())
    h_cc = np.zeros((n_free, n_free))
    coupling = np.zeros((n_free, n_free))
    sq_sum, n_res = 0.0, 0
    for pts, r, t in zip(views, rvecs, tvecs):
        proj, jac = cv2.projectPoints(obj, r, t, K, dist)
        res = (pts.reshape(-1, 2) - proj.reshape(-1, 2)).reshape(-1)
        j_c = jac[:, INTRINSIC_COLS][:, free]
        j_i = jac[:, :N_POSE]
        h_ii = j_i.T @ j_i
        h_ci = j_c.T @ j_i
        h_cc += j_c.T @ j_c
        coupling += h_ci @ np.linalg.solve(h_ii, h_ci.T)
        sq_sum += float(res @ res)
        n_res += res.size
    dof = n_res - n_free - N_POSE * len(views)
    sigma2 = sq_sum / dof
    return sigma2 * np.linalg.inv(h_cc - coupling)


def print_report(names: list[str], values: np.ndarray, cov: np.ndarray) -> None:
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    print("\nParameter        value      std dev   std/|value|")
    for n, v, s in zip(names, values, std):
        print(f"  {n:<4} {v:12.5f} {s:12.5f} {s / abs(v) if v else float('inf'):10.2f}")
    print("\nCorrelation matrix:")
    print("       " + " ".join(f"{n:>6}" for n in names))
    for n, row in zip(names, corr):
        print(f"  {n:<4} " + " ".join(f"{c:6.2f}" for c in row))
    pairs = [(names[i], names[j], corr[i, j]) for i in range(len(names)) for j in range(i)
             if abs(corr[i, j]) >= HIGH_CORRELATION]
    if pairs:
        print(f"\nStrongly coupled (|corr| >= {HIGH_CORRELATION}): "
              + ", ".join(f"{a}/{b} {c:+.3f}" for a, b, c in pairs))
    else:
        print(f"\nNo parameter pair has |corr| >= {HIGH_CORRELATION}.")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    data = np.load(args.corners)
    size = (int(data["size"][0]), int(data["size"][1]))
    cols, rows = (int(v) for v in data["pattern"])
    obj = board_points(cols, rows, float(data["square_mm"]) / 1000.0)
    views = [v.astype(np.float32).reshape(-1, 1, 2) for v in data["corners"]]
    log.info("%d views, %dx%d, board %dx%d", len(views), *size, cols, rows)

    rms, K, dist, rvecs, tvecs = fit(views, obj, size, args.fix_k3)
    errs = view_errors(views, obj, K, dist, rvecs, tvecs)
    if args.drop_worst:
        keep = np.sort(np.argsort(errs)[:len(views) - args.drop_worst])
        views = [views[i] for i in keep]
        log.info("Dropped the %d worst views, re-fitting on %d", args.drop_worst, len(views))
        rms, K, dist, rvecs, tvecs = fit(views, obj, size, args.fix_k3)
        errs = view_errors(views, obj, K, dist, rvecs, tvecs)

    print(f"RMS reprojection error: {rms:.3f} px over {len(views)} views"
          + ("  (k3 fixed at 0)" if args.fix_k3 else ""))
    order = np.argsort(errs)[::-1][:args.worst]
    print("Worst views (index: RMS px): " + ", ".join(f"{i}: {errs[i]:.2f}" for i in order)
          + f"   median {np.median(errs):.2f}")

    free = np.ones(len(PARAM_NAMES), bool)
    if args.fix_k3:
        free[-1] = False
    values = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *dist])[free]
    cov = intrinsics_covariance(views, obj, K, dist, rvecs, tvecs, free)
    print_report([n for n, f in zip(PARAM_NAMES, free) if f], values, cov)

    if args.output:
        PinholeCamera(size[0], size[1], K, dist, calibrated=True).to_yaml(
            args.output, extra={"rms_reprojection_error_px": float(rms), "n_views": len(views)})
        log.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

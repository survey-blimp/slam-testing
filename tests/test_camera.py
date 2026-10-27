import numpy as np

from minislam.camera import PinholeCamera, load_camera


def make_camera() -> PinholeCamera:
    return PinholeCamera.from_intrinsics(640, 480, 600.0, 598.0, 322.0, 236.0,
                                         np.array([-0.21, 0.06, 0.001, -0.0005, 0.0]))


def test_distort_undistort_round_trip():
    cam = make_camera()
    rng = np.random.default_rng(0)
    uv = rng.uniform([20, 20], [620, 460], (200, 2))
    back = cam.undistort_points(cam.distort_pixels(uv))
    assert np.max(np.abs(back - uv)) < 1e-3


def test_yaml_round_trip(tmp_path):
    cam = make_camera()
    path = tmp_path / "camera.yaml"
    cam.to_yaml(path, extra={"rms_reprojection_error_px": 0.2})
    loaded = PinholeCamera.from_yaml(path)
    assert np.allclose(loaded.K, cam.K) and np.allclose(loaded.dist, cam.dist)


def test_uncalibrated_fallback(tmp_path):
    cam = load_camera(tmp_path / "missing.yaml", 640, 480, 70.0)
    assert not cam.calibrated
    assert abs(cam.fx - 640 / (2 * np.tan(np.radians(35)))) < 1e-9

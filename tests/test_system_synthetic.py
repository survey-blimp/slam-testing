"""End-to-end: run the whole system on rendered images of the synthetic room.

This exercises ORB extraction, matching, initialization, tracking, local
mapping and BA together, against ground truth, without a webcam. It takes
around 20-30 seconds.
"""

from pathlib import Path

import numpy as np

from minislam import geometry
from minislam.camera import PinholeCamera
from minislam.config import load_config
from minislam.system import SlamSystem
from synthetic import CX, CY, FX, FY, HEIGHT, WIDTH, TexturedRoom, room_trajectory

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
N_FRAMES = 150


def test_full_system_on_synthetic_room():
    room = TexturedRoom(seed=0)
    poses_true = room_trajectory(N_FRAMES * 2)[:N_FRAMES]  # first half of the loop
    camera = PinholeCamera.from_intrinsics(WIDTH, HEIGHT, FX, FY, CX, CY)
    system = SlamSystem(CFG, camera)

    states = []
    for i, T in enumerate(poses_true):
        snap = system.process(i / 30.0, room.render(T))
        states.append(snap.state)

    first_ok = states.index("OK")
    assert first_ok < 30, "should initialize during the sideways slide"
    assert all(s == "OK" for s in states[first_ok:]), "tracking must not be lost"
    assert len(system.map.keyframes) >= 4
    assert len(system.map.map_points) > 300

    ts, poses_cw = system.tracker.trajectory_poses_cw()
    C_est = np.array([geometry.camera_center(T) for T in poses_cw])
    C_true = np.array([geometry.camera_center(poses_true[round(t * 30)]) for t in ts])
    s, R, t = geometry.umeyama_alignment(C_est, C_true)
    ate = np.sqrt(np.mean(np.sum((s * C_est @ R.T + t - C_true) ** 2, axis=1)))
    path = np.sum(np.linalg.norm(np.diff(C_true, axis=0), axis=1))
    assert ate < 0.02 * path, f"ATE {ate:.4f} m over a {path:.2f} m path"


def test_relocalization_after_covering_the_lens():
    room = TexturedRoom(seed=2)
    poses = room_trajectory(200)
    camera = PinholeCamera.from_intrinsics(WIDTH, HEIGHT, FX, FY, CX, CY)
    system = SlamSystem(CFG, camera)
    t = 0
    for T in poses[:70]:
        system.process(t / 30.0, room.render(T))
        t += 1
    assert system.tracker.state.value == "OK"
    n_kfs = len(system.map.keyframes)

    dark = np.full((HEIGHT, WIDTH), 12, np.uint8)  # lens covered
    for _ in range(10):
        snap = system.process(t / 30.0, dark)
        t += 1
    assert snap.state == "LOST"

    # Uncover while looking at an already mapped part of the room.
    states = [system.process((t + k) / 30.0, room.render(T)).state for k, T in enumerate(poses[40:55])]
    assert "OK" in states[:5], f"did not relocalize: {states}"
    assert states[-1] == "OK"
    assert len(system.map.keyframes) >= n_kfs  # same map, not a reset


def test_save_writes_tum_and_ply(tmp_path):
    room = TexturedRoom(seed=1)
    camera = PinholeCamera.from_intrinsics(WIDTH, HEIGHT, FX, FY, CX, CY)
    system = SlamSystem(CFG, camera)
    for i, T in enumerate(room_trajectory(60)[:25]):
        system.process(i / 30.0, room.render(T))
    out = system.save(tmp_path / "run")

    traj = np.loadtxt(out / "trajectory.txt", comments="#", ndmin=2)
    assert traj.shape[1] == 8 and len(traj) > 5
    assert np.allclose(np.linalg.norm(traj[:, 4:8], axis=1), 1.0, atol=1e-5)  # unit quaternions
    ply = (out / "map_points.ply").read_text().splitlines()
    n_vertex = int(next(line for line in ply if line.startswith("element vertex")).split()[-1])
    assert n_vertex == len(system.map.map_points) > 0
    assert len(ply) == ply.index("end_header") + 1 + n_vertex

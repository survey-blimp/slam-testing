"""minislam entry point.

    python run.py --source 0                          # webcam index
    python run.py --source drone                      # Wi-Fi drone camera (laptop joined to the drone's Wi-Fi)
    python run.py --source path\\to\\video.mp4          # video file
    python run.py --source path\\to\\tum_sequence       # image folder with rgb.txt
                  --camera config\\camera.yaml --config config\\default.yaml
                  [--headless] [--max-frames N]

``--headless`` runs without the GUI and saves the trajectory and map to
``output/<timestamp>/`` at the end (used for tests and evaluation).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from minislam.camera import load_camera
from minislam.config import load_config
from minislam.io import timestamped_output_dir
from minislam.source import open_source
from minislam.system import SlamSystem

log = logging.getLogger("minislam")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="minislam: minimal monocular ORB-SLAM")
    p.add_argument("--source", default="0", help="webcam index, 'drone', video file, or image folder")
    p.add_argument("--camera", type=Path, default=Path("config/camera.yaml"),
                   help="calibration file from tools/calibrate.py")
    p.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p.add_argument("--headless", action="store_true", help="no GUI; save outputs at the end")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--output", type=Path, default=None,
                   help="output folder for --headless (default output/<timestamp>)")
    p.add_argument("--log-level", default="INFO", help="DEBUG shows every optimizer call")
    return p.parse_args()


def run_headless(source, system: SlamSystem, max_frames: int | None, out_dir: Path) -> None:
    n = 0
    t0 = time.perf_counter()
    last_state = None
    while max_frames is None or n < max_frames:
        item = source.read()
        if item is None:
            break
        snap = system.process(*item)
        if snap.state != last_state:
            log.info("frame %d: state %s", snap.frame_id, snap.state)
            last_state = snap.state
        n += 1
        if n % 100 == 0:
            log.info("frame %d: %s, %d KFs, %d MPs, %.1f FPS", n, snap.state, snap.n_keyframes,
                     snap.n_map_points, n / (time.perf_counter() - t0))
    source.close()
    log.info("Processed %d frames in %.1f s", n, time.perf_counter() - t0)
    log.info("Saved results to %s", system.save(out_dir))


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    source = open_source(args.source, cfg.source, cfg.drone)
    camera = load_camera(args.camera, source.width, source.height, cfg.camera.uncalibrated_hfov_deg)
    system = SlamSystem(cfg, camera)

    if args.headless:
        out_dir = args.output or timestamped_output_dir(cfg.gui.output_dir)
        run_headless(source, system, args.max_frames, out_dir)
        return 0

    from minislam.gui.app import run_gui

    return run_gui(source, system, cfg, args.max_frames)


if __name__ == "__main__":
    sys.exit(main())

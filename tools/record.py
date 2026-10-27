"""Record the webcam to a video + timestamps file for repeatable runs.

    python tools/record.py                         # -> recordings/<timestamp>.mp4
    python tools/record.py --output recordings\\desk.mp4 --seconds 60

Writes ``<name>.mp4`` and ``<name>.timestamps.txt`` (one ``perf_counter``
timestamp per frame, in seconds). ``run.py --source recordings\\desk.mp4``
picks the timestamps file up automatically, so a live session can be
replayed exactly while debugging. The capture settings (resolution,
autofocus / auto-exposure locks) come from ``config/default.yaml``, the
same as for SLAM. Press Q or Esc in the preview window to stop.

ORB-SLAM3 equivalent: none (dataset recording is outside the library).
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minislam.config import load_config  # noqa: E402
from minislam.source import WebcamSource  # noqa: E402

log = logging.getLogger("record")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--output", type=Path, default=None, help="video path (default recordings/<time>.mp4)")
    p.add_argument("--seconds", type=float, default=None, help="stop after this long (default: until Q)")
    p.add_argument("--no-preview", action="store_true", help="do not show a preview window")
    p.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    out = args.output or Path("recordings") / f"{datetime.now():%Y%m%d_%H%M%S}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    cam = WebcamSource(args.camera_index, cfg.source)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), float(cfg.source.fps),
                             (cam.width, cam.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out}")
    stamps: list[float] = []
    t_end = None if args.seconds is None else time.perf_counter() + args.seconds
    log.info("Recording to %s (Q / Esc to stop)", out)
    try:
        while t_end is None or time.perf_counter() < t_end:
            item = cam.read()
            if item is None:
                break
            ts, img = item
            writer.write(img)
            stamps.append(ts)
            if not args.no_preview:
                cv2.imshow("record (Q to stop)", img)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cam.close()
        writer.release()
        cv2.destroyAllWindows()

    ts_file = out.with_suffix(".timestamps.txt")
    ts_file.write_text("# perf_counter timestamp (s) per frame\n" + "".join(f"{t:.6f}\n" for t in stamps))
    dur = stamps[-1] - stamps[0] if len(stamps) > 1 else 0.0
    log.info("Wrote %d frames (%.1f s, %.1f FPS) to %s and %s", len(stamps), dur,
             (len(stamps) - 1) / dur if dur > 0 else 0.0, out, ts_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())

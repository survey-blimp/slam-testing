"""Frame source for a Wi-Fi toy drone camera (Quadbox / "JX WiFi", MJPEG over UDP).

Protocol (reverse-engineered in the drone-hacking repo, copied here so minislam is self-contained):

  -> drone:40000  ``63 63 01 00 00 00 00``  hello / keep-alive, about once per second.
                  The drone stops streaming ~4 s after the last hello and sends video
                  back to the UDP source port the hello came from.
  <- drone        ``63 63 03 ...``  video chunk. Header fields (little-endian):
                    [8:12]  u32  frame counter (+1 per frame, arbitrary start, wraps)
                    [12:16] u32  total JPEG size of the frame
                    [48]    u8   chunk index (1-based)
                    [50]    u8   chunk count
                    [52:54] u16  payload length
                    [54:]        JPEG payload; concatenate chunks in index order

The drone deliberately inverts (XOR 0xFF) one byte of every JPEG, at a position computed
from the frame counter and size; ``unscramble`` flips it back exactly as the vendor app does.

Threading: a receiver thread owns the socket, sends the keep-alive, reassembles frames and
keeps only the **newest** complete JPEG ("latest wins"). ``read()`` waits for a frame newer
than the last one it returned and decodes it, so SLAM never works through a backlog.
Timestamps are ``time.perf_counter()`` at the moment the frame was completed, matching
``WebcamSource``.

Only the hello is ever sent: this module never transmits flight-control frames.

ORB-SLAM3 equivalent: none (an example-program concern, like ``source.py``).
"""

import logging
import socket
import struct
import threading
import time

import cv2
import numpy as np

from minislam.config import DroneConfig

log = logging.getLogger(__name__)

# Wire-format constants of the drone protocol (not tunables).
HELLO = bytes.fromhex("63630100000000")
VIDEO_MAGIC = b"\x63\x63\x03"
HEADER_LEN = 54
STALE_FRAME_WINDOW = 5      # incomplete frames this far behind the newest are discarded
COUNTER_MASK = 0xFFFFFFFF


def scramble_index(frame_counter: int, size: int) -> int:
    """Position of the inverted byte (``encode_index`` in the vendor's liblewei-2.2.so)."""
    if size & 1:
        v = ((frame_counter ^ size) + size) ^ size
    else:
        v = ((frame_counter ^ size) + size + 1) ^ size
    return v % size


def unscramble(jpg: bytes, frame_counter: int) -> bytes:
    """Undo the drone's per-frame byte inversion."""
    b = bytearray(jpg)
    b[scramble_index(frame_counter, len(b))] ^= 0xFF
    return bytes(b)


def is_newer(a: int, b: int) -> bool:
    """True if 32-bit frame counter ``a`` is after ``b`` (handles wrap-around)."""
    return 0 < ((a - b) & COUNTER_MASK) < 0x80000000


class FrameAssembler:
    """Collects video chunks into complete, unscrambled JPEGs."""

    def __init__(self) -> None:
        self.partial: dict[int, dict] = {}   # counter -> {"size", "count", "chunks"}
        self.last_done: int | None = None
        self.incomplete = 0                  # frames abandoned before all chunks arrived

    def add(self, pkt: bytes) -> tuple[int, bytes] | None:
        """Feed one UDP packet; returns ``(counter, jpeg)`` when a frame completes."""
        if len(pkt) <= HEADER_LEN or pkt[:3] != VIDEO_MAGIC:
            return None
        counter, size = struct.unpack_from("<II", pkt, 8)
        index, count = pkt[48], pkt[50]
        payload_len = struct.unpack_from("<H", pkt, 52)[0]
        frame = self.partial.setdefault(counter, {"size": size, "count": count, "chunks": {}})
        frame["chunks"][index] = pkt[HEADER_LEN:HEADER_LEN + payload_len]

        done = None
        if len(frame["chunks"]) == frame["count"] and (
                self.last_done is None or is_newer(counter, self.last_done)):
            jpg = b"".join(frame["chunks"][i] for i in sorted(frame["chunks"]))
            del self.partial[counter]
            self.last_done = counter
            if len(jpg) == frame["size"]:
                done = (counter, unscramble(jpg, counter))

        for old in [k for k in self.partial
                    if is_newer(counter, k) and ((counter - k) & COUNTER_MASK) > STALE_FRAME_WINDOW]:
            del self.partial[old]
            self.incomplete += 1
        return done


class DroneSource:
    is_live = True

    def __init__(self, cfg: DroneConfig) -> None:
        self.cfg = cfg
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self.sock.bind(("0.0.0.0", cfg.local_port))
        self.sock.settimeout(0.2)
        self.assembler = FrameAssembler()
        self.cond = threading.Condition()
        self.latest: bytes | None = None
        self.latest_ts = 0.0
        self.latest_seq = 0          # increments per completed frame
        self.returned_seq = 0
        self.dropped = 0             # completed frames overwritten before read() got them
        self.stopping = False
        self.thread = threading.Thread(target=self._receive, name="drone-rx", daemon=True)
        self.thread.start()

        first = self.read()
        if first is None:
            self.close()
            raise RuntimeError(
                f"No video from drone at {cfg.host}:{cfg.port} within {cfg.read_timeout_s:.0f} s "
                "(is the laptop on the drone's Wi-Fi, and is nothing else using the stream?)")
        self.height, self.width = first[1].shape[:2]
        self._first = first
        log.info("Drone %s:%d: delivering %dx%d", cfg.host, cfg.port, self.width, self.height)

    def _receive(self) -> None:
        next_hello = 0.0
        while not self.stopping:
            now = time.monotonic()
            if now >= next_hello:
                self.sock.sendto(HELLO, (self.cfg.host, self.cfg.port))
                next_hello = now + self.cfg.hello_period_s
            try:
                pkt, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:   # Windows reports ICMP port-unreachable as ConnectionResetError
                continue
            done = self.assembler.add(pkt)
            if done is not None:
                with self.cond:
                    if self.latest_seq > self.returned_seq:
                        self.dropped += 1
                    self.latest = done[1]
                    self.latest_ts = time.perf_counter()
                    self.latest_seq += 1
                    self.cond.notify_all()

    def read(self) -> tuple[float, np.ndarray] | None:
        first = getattr(self, "_first", None)
        if first is not None:
            self._first = None
            return first
        deadline = time.monotonic() + self.cfg.read_timeout_s
        while True:
            with self.cond:
                while self.latest_seq == self.returned_seq and not self.stopping:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        log.warning("No drone frame for %.1f s, ending input", self.cfg.read_timeout_s)
                        return None
                    self.cond.wait(remaining)
                if self.stopping:
                    return None
                jpg, ts = self.latest, self.latest_ts
                self.returned_seq = self.latest_seq
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:   # a garbled frame decodes to None: skip it, wait for the next
                return ts, img
            log.debug("Skipping undecodable drone frame")

    def close(self) -> None:
        self.stopping = True
        with self.cond:
            self.cond.notify_all()
        self.thread.join(timeout=1.0)
        self.sock.close()
        log.info("Drone source closed (%d frames overwritten unread, %d incomplete)",
                 self.dropped, self.assembler.incomplete)

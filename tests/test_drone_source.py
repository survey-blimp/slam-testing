import socket
import struct
import threading
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from minislam.config import load_config
from minislam.drone import HELLO, DroneSource, FrameAssembler, scramble_index, unscramble

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
CHUNK = 1400


def make_chunks(counter: int, jpg: bytes) -> list[bytes]:
    """Build drone video packets for one frame, scrambling it like the drone does."""
    b = bytearray(jpg)
    b[scramble_index(counter, len(b))] ^= 0xFF
    parts = [bytes(b[i:i + CHUNK]) for i in range(0, len(b), CHUNK)]
    pkts = []
    for i, part in enumerate(parts, start=1):
        hdr = bytearray(54)
        hdr[0:3] = b"\x63\x63\x03"
        struct.pack_into("<II", hdr, 8, counter, len(jpg))
        hdr[48], hdr[50] = i, len(parts)
        struct.pack_into("<H", hdr, 52, len(part))
        pkts.append(bytes(hdr) + part)
    return pkts


def test_unscramble_is_its_own_inverse():
    data = bytes(range(200)) * 3
    for counter in (0, 5, 0x5A5A5A5A, 0xFFFFFFFF):
        assert unscramble(unscramble(data, counter), counter) == data


def test_assembler_rebuilds_frame_from_shuffled_chunks():
    img = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    jpg = cv2.imencode(".jpg", img)[1].tobytes()
    pkts = make_chunks(0x5A5A5A5A, jpg)
    asm = FrameAssembler()
    out = None
    for pkt in reversed(pkts):
        out = asm.add(pkt) or out
    assert out == (0x5A5A5A5A, jpg)


def test_drone_source_end_to_end_with_fake_drone():
    fake = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    fake.bind(("127.0.0.1", 0))
    fake.settimeout(2.0)
    local_port = socket.socket()
    local_port.bind(("", 0))
    port = local_port.getsockname()[1]
    local_port.close()
    cfg = replace(CFG.drone, host="127.0.0.1", port=fake.getsockname()[1], local_port=port,
                  read_timeout_s=3.0)
    img = np.full((120, 160, 3), 90, np.uint8)
    jpg = cv2.imencode(".jpg", img)[1].tobytes()

    def drone() -> None:
        pkt, addr = fake.recvfrom(100)
        assert pkt == HELLO
        for counter in range(10, 40):
            for p in make_chunks(counter, jpg):
                fake.sendto(p, addr)
            time.sleep(0.02)

    t = threading.Thread(target=drone)
    t.start()
    src = DroneSource(cfg)
    try:
        assert (src.width, src.height) == (160, 120)
        ts, frame = src.read()
        assert frame.shape == (120, 160, 3) and abs(int(frame.mean()) - 90) < 5
    finally:
        src.close()
        t.join()
        fake.close()

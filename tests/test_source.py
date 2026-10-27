from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from minislam import source
from minislam.config import load_config

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")


class FakeCapture:
    """Stands in for cv2.VideoCapture and records every property it is sent."""

    def __init__(self, index: int, api: int) -> None:
        self.sets: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return True

    def set(self, prop: int, value: float) -> bool:
        self.sets.append((prop, value))
        return True

    def get(self, prop: int) -> float:
        return -1.0  # like DirectShow: settings are accepted but not reported back

    def read(self) -> tuple[bool, np.ndarray]:
        return True, np.zeros((CFG.source.height, CFG.source.width, 3), np.uint8)

    def release(self) -> None:
        pass


def exposure_requests(cap: FakeCapture) -> list[float]:
    return [v for p, v in cap.sets if p == cv2.CAP_PROP_AUTO_EXPOSURE]


def test_auto_exposure_is_switched_on_by_default(monkeypatch):
    monkeypatch.setattr(source.cv2, "VideoCapture", FakeCapture)
    cam = source.WebcamSource(0, replace(CFG.source, disable_auto_exposure=False))
    cam.close()
    assert exposure_requests(cam.cap) == [source.AUTO_EXPOSURE_AUTO]


def test_locked_exposure_is_released_on_close(monkeypatch):
    monkeypatch.setattr(source.cv2, "VideoCapture", FakeCapture)
    cam = source.WebcamSource(0, replace(CFG.source, disable_auto_exposure=True))
    assert exposure_requests(cam.cap) == [source.AUTO_EXPOSURE_MANUAL]
    cam.close()
    assert exposure_requests(cam.cap) == [source.AUTO_EXPOSURE_MANUAL, source.AUTO_EXPOSURE_AUTO]

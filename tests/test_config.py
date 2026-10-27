from pathlib import Path

import pytest
import yaml

from minislam.config import load_config

DEFAULT = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


def test_default_config_loads():
    cfg = load_config(DEFAULT)
    assert cfg.source.width == 640
    assert isinstance(cfg.camera.uncalibrated_hfov_deg, float)


def test_unknown_key_is_rejected(tmp_path):
    raw = yaml.safe_load(DEFAULT.read_text())
    raw["source"]["widht"] = 1  # typo
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(bad)


def test_missing_key_is_rejected(tmp_path):
    raw = yaml.safe_load(DEFAULT.read_text())
    del raw["gui"]["map_fps"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="missing keys"):
        load_config(bad)

from pathlib import Path

import cv2
import numpy as np

from minislam.config import load_config
from minislam.features import OrbExtractor, features_per_level
from synthetic import TexturedRoom, look_at

CFG = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml").features


def test_budget_sums_to_total():
    b = features_per_level(1000, 1.2, 8)
    assert b.sum() == 1000 and np.all(np.diff(b) <= 0)


def test_features_cover_the_image():
    img = TexturedRoom(seed=3).render(look_at(np.zeros(3), np.array([0.5, 0.2, 5.0])))
    f = OrbExtractor(CFG).extract(img)
    assert 800 <= len(f.xy) <= CFG.n_features
    assert f.descriptors.shape == (len(f.xy), 32)
    assert set(np.unique(f.octaves)) == set(range(CFG.n_levels))
    # Spread: most 80 px cells of the image contain at least one feature.
    cells = set(map(tuple, (f.xy // 80).astype(int)))
    assert len(cells) >= 0.9 * (640 // 80) * (480 // 80)


def test_grid_beats_plain_orb_on_clumped_texture():
    # One very high-contrast blob on a faint background: plain ORB clumps on it.
    rng = np.random.default_rng(0)
    img = (rng.normal(128, 6, (480, 640))).clip(0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    for _ in range(200):
        x, y = rng.integers(60, 200, 2)
        cv2.rectangle(img, (int(x), int(y)), (int(x) + 8, int(y) + 8), int(rng.integers(0, 255)), -1)
    ours = OrbExtractor(CFG).extract(img).xy
    plain = np.array([k.pt for k in cv2.ORB_create(1000).detect(img, None)])
    spread = lambda pts: len(set(map(tuple, (pts // 80).astype(int))))  # noqa: E731
    assert spread(ours) > 2 * spread(plain)

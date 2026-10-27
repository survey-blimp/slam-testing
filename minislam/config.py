"""Load and validate the YAML configuration into typed dataclasses.

Every tunable number in minislam lives in ``config/default.yaml``. This module
turns that file into frozen dataclasses so the rest of the code can write
``cfg.tracking.min_inliers`` instead of ``cfg["tracking"]["min_inliers"]``
and so typos are caught at start-up rather than deep inside the pipeline.

Validation rules (deliberately strict):
  * every field of every section must be present in the YAML,
  * unknown keys are an error (they are usually typos),
  * values are converted to the annotated type (int / float / bool / str).

ORB-SLAM3 equivalent: the settings parsing in ``Settings.cc`` /
``Tracking::ParseORBParamFile`` etc., which read the ``.yaml`` settings file.
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass(frozen=True)
class SourceConfig:
    width: int
    height: int
    fps: int
    disable_autofocus: bool
    disable_auto_exposure: bool


@dataclass(frozen=True)
class CameraConfig:
    uncalibrated_hfov_deg: float


@dataclass(frozen=True)
class GuiConfig:
    video_fps: float
    map_fps: float
    point_size: float
    frustum_size: float
    show_track_lines: bool
    follow_camera: bool
    output_dir: str


@dataclass(frozen=True)
class FeatureConfig:
    n_features: int
    scale_factor: float
    n_levels: int
    fast_threshold: int
    fast_threshold_low: int
    cell_size: int
    edge_threshold: int
    patch_size: int


@dataclass(frozen=True)
class InitializerConfig:
    min_features: int
    min_matches: int
    reset_below_matches: int
    ratio: float
    max_distance: int
    ransac_prob: float
    ransac_threshold_px: float
    min_parallax_deg: float
    min_points: int


@dataclass(frozen=True)
class OptimizerConfig:
    chi2_2dof: float
    pose_rounds: int
    pose_max_nfev: int
    ba_max_nfev: int
    init_ba_max_nfev: int


@dataclass(frozen=True)
class TrackingConfig:
    projection_radius_px: float
    projection_radius_wide_px: float
    max_distance: int
    projection_ratio: float
    octave_tolerance: int
    min_projection_matches: int
    reference_ratio: float
    local_map_radius_px: float
    local_map_max_keyframes: int
    min_viewing_cos: float
    min_inliers: int
    min_frames_between_kf: int
    max_frames_between_kf: int
    kf_inlier_ratio: float
    kf_min_inliers: int
    kf_ref_min_obs: int


@dataclass(frozen=True)
class LocalMappingConfig:
    cull_min_found_ratio: float
    cull_min_observations: int
    cull_keyframe_age: int
    recent_point_maturity: int
    triangulation_neighbors: int
    min_baseline_depth_ratio: float
    epipolar_sigma_factor: float
    triangulation_max_distance: int
    triangulation_ratio: float
    local_ba_neighbors: int


@dataclass(frozen=True)
class MapConfig:
    min_covisibility_weight: int


@dataclass(frozen=True)
class Config:
    source: SourceConfig
    camera: CameraConfig
    gui: GuiConfig
    features: FeatureConfig
    initializer: InitializerConfig
    optimizer: OptimizerConfig
    map: MapConfig
    tracking: TrackingConfig
    local_mapping: LocalMappingConfig


def _convert(value: Any, target: type, where: str) -> Any:
    """Convert a YAML scalar to the annotated type, refusing silent nonsense."""
    if target is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{where}: expected true/false, got {value!r}")
        return value
    if target is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where}: expected an integer, got {value!r}")
        return value
    if target is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if target is str:
        return str(value)
    raise TypeError(f"{where}: unsupported config type {target}")


def _build_section(cls: type, data: Any, name: str) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"config section '{name}' is missing or not a mapping")
    hints = get_type_hints(cls)
    expected = {f.name for f in fields(cls)}
    missing = expected - data.keys()
    unknown = data.keys() - expected
    if missing:
        raise ValueError(f"config section '{name}' is missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"config section '{name}' has unknown keys: {sorted(unknown)}")
    values = {k: _convert(data[k], hints[k], f"{name}.{k}") for k in expected}
    return cls(**values)


def load_config(path: Path) -> Config:
    """Read ``path`` (YAML) and return a validated :class:`Config`."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    hints = get_type_hints(Config)
    unknown = raw.keys() - hints.keys()
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    sections = {name: _build_section(cls, raw.get(name), name) for name, cls in hints.items()}
    return Config(**sections)

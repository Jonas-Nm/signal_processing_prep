"""Configuration loading for signal-processing analyses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations used by the analysis pipeline."""

    data_dir: Path = Path("data/raw")
    reports_dir: Path = Path("reports")
    figures_dir: Path = Path("reports/figures")
    summaries_dir: Path = Path("reports/summaries")


@dataclass(frozen=True)
class LoadingConfig:
    """Settings for loading signal files."""

    file_patterns: list[str] = field(
        default_factory=lambda: ["*.csv", "*.txt", "*.npy", "*.wav"]
    )
    sampling_rate_hz: float | None = None
    label_column: str | None = None
    signal_column: str | None = None


@dataclass(frozen=True)
class WindowConfig:
    """Windowing settings for segmented analysis."""

    size_seconds: float = 1.0
    overlap_fraction: float = 0.5


@dataclass(frozen=True)
class AnalysisConfig:
    """Analysis settings shared by feature extraction and plotting."""

    frequency_bands_hz: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "low": (0.0, 100.0),
            "mid": (100.0, 1000.0),
            "high": (1000.0, 5000.0),
        }
    )
    window: WindowConfig = field(default_factory=WindowConfig)


@dataclass(frozen=True)
class FilteringConfig:
    """Optional filtering settings."""

    enabled: bool = False
    kind: str | None = None
    low_cut_hz: float | None = None
    high_cut_hz: float | None = None
    order: int = 4


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project configuration."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    loading: LoadingConfig = field(default_factory=LoadingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)


def load_config(path: str | Path) -> ProjectConfig:
    """Load a YAML configuration file into a typed project configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}
    if not isinstance(raw_config, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    return ProjectConfig(
        paths=_load_paths(_section(raw_config, "paths")),
        loading=_load_loading(_section(raw_config, "loading")),
        analysis=_load_analysis(_section(raw_config, "analysis")),
        filtering=_load_filtering(_section(raw_config, "filtering")),
    )


def _section(raw_config: dict[str, Any], name: str) -> Mapping[str, Any]:
    raw_section = raw_config.get(name) or {}
    if not isinstance(raw_section, Mapping):
        raise ValueError(f"Configuration section '{name}' must be a mapping.")
    return raw_section


def _load_paths(raw_paths: Mapping[str, Any]) -> PathsConfig:
    return PathsConfig(
        data_dir=Path(raw_paths.get("data_dir", "data/raw")),
        reports_dir=Path(raw_paths.get("reports_dir", "reports")),
        figures_dir=Path(raw_paths.get("figures_dir", "reports/figures")),
        summaries_dir=Path(raw_paths.get("summaries_dir", "reports/summaries")),
    )


def _load_loading(raw_loading: Mapping[str, Any]) -> LoadingConfig:
    return LoadingConfig(
        file_patterns=list(raw_loading.get("file_patterns", ["*.csv", "*.txt", "*.npy", "*.wav"])),
        sampling_rate_hz=_optional_float(raw_loading.get("sampling_rate_hz")),
        label_column=raw_loading.get("label_column"),
        signal_column=raw_loading.get("signal_column"),
    )


def _load_analysis(raw_analysis: Mapping[str, Any]) -> AnalysisConfig:
    raw_bands = raw_analysis.get("frequency_bands_hz", {})
    frequency_bands = {
        name: _frequency_band_tuple(value) for name, value in raw_bands.items()
    }
    raw_window = raw_analysis.get("window", {})
    return AnalysisConfig(
        frequency_bands_hz=frequency_bands or AnalysisConfig().frequency_bands_hz,
        window=WindowConfig(
            size_seconds=float(raw_window.get("size_seconds", 1.0)),
            overlap_fraction=float(raw_window.get("overlap_fraction", 0.5)),
        ),
    )


def _load_filtering(raw_filtering: Mapping[str, Any]) -> FilteringConfig:
    return FilteringConfig(
        enabled=bool(raw_filtering.get("enabled", False)),
        kind=raw_filtering.get("kind"),
        low_cut_hz=_optional_float(raw_filtering.get("low_cut_hz")),
        high_cut_hz=_optional_float(raw_filtering.get("high_cut_hz")),
        order=int(raw_filtering.get("order", 4)),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _frequency_band_tuple(value: Any) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError("Frequency bands must contain exactly two values.")
    low, high = value
    return float(low), float(high)

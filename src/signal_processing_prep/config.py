"""Validated project and analysis configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from signal_processing_prep.errors import ConfigurationError

_TOP_LEVEL_KEYS = {"paths", "loading", "analysis"}
_PATH_KEYS = {"data_dir", "reports_dir", "figures_dir", "summaries_dir"}
_LOADING_KEYS = {"file_patterns", "sampling_rate_hz", "label_column", "signal_column"}
_ANALYSIS_KEYS = {
    "frequency_bands_hz",
    "window",
    "spectrogram_window_seconds",
    "spectrogram_step_seconds",
    "high_frequency_cutoff_hz",
    "filtering",
    "quality",
    "invalid_record_policy",
    "modeling",
}
_WINDOW_KEYS = {"size_seconds", "overlap_fraction"}
_FILTERING_KEYS = {
    "enabled",
    "kind",
    "low_cut_hz",
    "high_cut_hz",
    "order",
    "zero_phase",
    "allow_causal_fallback",
}
_QUALITY_KEYS = {
    "min_duration_seconds",
    "clipping_fraction_threshold",
    "clipping_tolerance",
    "near_constant_std_threshold",
    "large_amplitude_threshold",
    "stationarity_window_seconds",
    "stationarity_mean_drift_threshold",
    "stationarity_std_cv_threshold",
    "time_step_jitter_threshold",
    "time_gap_count_threshold",
    "sampling_rate_mismatch_threshold",
}
_MODELING_KEYS = {
    "enabled",
    "anomaly_when_unlabeled",
    "anomaly_contamination",
    "random_state",
}


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations used by configured analyses."""

    data_dir: Path = Path("data/raw")
    reports_dir: Path = Path("reports")
    figures_dir: Path = Path("reports/figures")
    summaries_dir: Path = Path("reports/summaries")


@dataclass(frozen=True)
class LoadingConfig:
    """Settings for loading signal files."""

    file_patterns: tuple[str, ...] = ("*.csv", "*.txt", "*.npy", "*.wav")
    sampling_rate_hz: float | None = None
    label_column: str | None = None
    signal_column: str | None = None

    def __post_init__(self) -> None:
        """Validate loader discovery settings."""
        object.__setattr__(self, "file_patterns", tuple(self.file_patterns))
        if not self.file_patterns:
            raise ConfigurationError("loading.file_patterns must not be empty.")
        if self.sampling_rate_hz is not None and self.sampling_rate_hz <= 0:
            raise ConfigurationError("loading.sampling_rate_hz must be positive.")


@dataclass(frozen=True)
class WindowConfig:
    """Window settings for explicit segmented analysis."""

    size_seconds: float = 1.0
    overlap_fraction: float = 0.5

    def __post_init__(self) -> None:
        """Validate window settings."""
        if self.size_seconds <= 0:
            raise ConfigurationError("analysis.window.size_seconds must be positive.")
        if not 0.0 <= self.overlap_fraction < 1.0:
            raise ConfigurationError("analysis.window.overlap_fraction must be in [0, 1).")


@dataclass(frozen=True)
class FilteringConfig:
    """Optional filtering policy for the analysis pipeline."""

    enabled: bool = False
    kind: str | None = None
    low_cut_hz: float | None = None
    high_cut_hz: float | None = None
    order: int = 4
    zero_phase: bool = True
    allow_causal_fallback: bool = False

    def __post_init__(self) -> None:
        """Validate static filter policy independent of signal sampling rate."""
        kind = self.kind.lower() if self.kind is not None else None
        object.__setattr__(self, "kind", kind)
        if self.order <= 0:
            raise ConfigurationError("analysis.filtering.order must be positive.")
        if self.enabled and kind is None:
            raise ConfigurationError("analysis.filtering.kind is required when enabled.")
        if kind is not None and kind not in {"lowpass", "highpass", "bandpass"}:
            raise ConfigurationError(
                "analysis.filtering.kind must be 'lowpass', 'highpass', or 'bandpass'."
            )
        if self.low_cut_hz is not None and self.low_cut_hz <= 0:
            raise ConfigurationError("analysis.filtering.low_cut_hz must be positive.")
        if self.high_cut_hz is not None and self.high_cut_hz <= 0:
            raise ConfigurationError("analysis.filtering.high_cut_hz must be positive.")
        if kind in {"highpass", "bandpass"} and self.low_cut_hz is None:
            raise ConfigurationError("analysis.filtering.low_cut_hz is required.")
        if kind in {"lowpass", "bandpass"} and self.high_cut_hz is None:
            raise ConfigurationError("analysis.filtering.high_cut_hz is required.")
        if (
            kind == "bandpass"
            and self.low_cut_hz is not None
            and self.high_cut_hz is not None
            and self.low_cut_hz >= self.high_cut_hz
        ):
            raise ConfigurationError(
                "analysis.filtering.low_cut_hz must be below high_cut_hz."
            )


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds used by signal quality checks."""

    min_duration_seconds: float | None = None
    clipping_fraction_threshold: float = 0.01
    clipping_tolerance: float = 1e-9
    near_constant_std_threshold: float = 1e-12
    large_amplitude_threshold: float | None = None
    stationarity_window_seconds: float | None = None
    stationarity_mean_drift_threshold: float = 0.25
    stationarity_std_cv_threshold: float = 0.5
    time_step_jitter_threshold: float = 0.01
    time_gap_count_threshold: int = 0
    sampling_rate_mismatch_threshold: float = 0.01

    def __post_init__(self) -> None:
        """Validate quality threshold policy."""
        if self.min_duration_seconds is not None and self.min_duration_seconds <= 0:
            raise ConfigurationError("analysis.quality.min_duration_seconds must be positive.")
        if not 0.0 <= self.clipping_fraction_threshold <= 1.0:
            raise ConfigurationError(
                "analysis.quality.clipping_fraction_threshold must be between 0 and 1."
            )
        for name in (
            "clipping_tolerance",
            "near_constant_std_threshold",
            "stationarity_mean_drift_threshold",
            "stationarity_std_cv_threshold",
            "time_step_jitter_threshold",
            "sampling_rate_mismatch_threshold",
        ):
            if getattr(self, name) < 0:
                raise ConfigurationError(f"analysis.quality.{name} must be non-negative.")
        if self.large_amplitude_threshold is not None and self.large_amplitude_threshold <= 0:
            raise ConfigurationError(
                "analysis.quality.large_amplitude_threshold must be positive."
            )
        if self.stationarity_window_seconds is not None and self.stationarity_window_seconds <= 0:
            raise ConfigurationError(
                "analysis.quality.stationarity_window_seconds must be positive."
            )
        if self.time_gap_count_threshold < 0:
            raise ConfigurationError(
                "analysis.quality.time_gap_count_threshold must be non-negative."
            )


@dataclass(frozen=True)
class ModelingConfig:
    """Modeling policy applied by the high-level workflow."""

    enabled: bool = True
    anomaly_when_unlabeled: bool = True
    anomaly_contamination: float | str = "auto"
    random_state: int = 0

    def __post_init__(self) -> None:
        """Validate anomaly-model policy."""
        if self.anomaly_contamination != "auto" and not (
            isinstance(self.anomaly_contamination, float | int)
            and 0.0 < float(self.anomaly_contamination) <= 0.5
        ):
            raise ConfigurationError(
                "analysis.modeling.anomaly_contamination must be 'auto' or in (0, 0.5]."
            )


@dataclass(frozen=True)
class AnalysisConfig:
    """Complete policy for pipeline processing, features, quality, and modeling."""

    frequency_bands_hz: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "low": (0.0, 100.0),
            "mid": (100.0, 1000.0),
            "high": (1000.0, 5000.0),
        }
    )
    window: WindowConfig = field(default_factory=WindowConfig)
    spectrogram_window_seconds: float = 0.1
    spectrogram_step_seconds: float | None = None
    high_frequency_cutoff_hz: float | None = None
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    invalid_record_policy: str = "skip"
    modeling: ModelingConfig = field(default_factory=ModelingConfig)

    def __post_init__(self) -> None:
        """Validate analysis features and error-handling policy."""
        bands = {str(name): tuple(values) for name, values in self.frequency_bands_hz.items()}
        object.__setattr__(self, "frequency_bands_hz", bands)
        for name, (low_hz, high_hz) in bands.items():
            if low_hz < 0 or high_hz <= low_hz:
                raise ConfigurationError(f"Invalid analysis frequency band '{name}'.")
        if self.spectrogram_window_seconds <= 0:
            raise ConfigurationError("analysis.spectrogram_window_seconds must be positive.")
        if self.spectrogram_step_seconds is not None and self.spectrogram_step_seconds <= 0:
            raise ConfigurationError("analysis.spectrogram_step_seconds must be positive.")
        if self.high_frequency_cutoff_hz is not None and self.high_frequency_cutoff_hz < 0:
            raise ConfigurationError("analysis.high_frequency_cutoff_hz must be non-negative.")
        if self.invalid_record_policy not in {"skip", "raise"}:
            raise ConfigurationError("analysis.invalid_record_policy must be 'skip' or 'raise'.")


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level configured project input and analysis policy."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    loading: LoadingConfig = field(default_factory=LoadingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


def load_config(path: str | Path) -> ProjectConfig:
    """Load YAML into one validated project configuration hierarchy."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration file must contain a YAML mapping.")
    _reject_unknown_keys(raw, _TOP_LEVEL_KEYS, "configuration")
    try:
        return ProjectConfig(
            paths=_load_paths(_section(raw, "paths")),
            loading=_load_loading(_section(raw, "loading")),
            analysis=_load_analysis(_section(raw, "analysis")),
        )
    except ConfigurationError:
        raise
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"Invalid configuration value: {error}") from error


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = raw.get(name) or {}
    if not isinstance(section, Mapping):
        raise ConfigurationError(f"Configuration section '{name}' must be a mapping.")
    return section


def _load_paths(raw: Mapping[str, Any]) -> PathsConfig:
    _reject_unknown_keys(raw, _PATH_KEYS, "paths")
    return PathsConfig(
        data_dir=Path(raw.get("data_dir", "data/raw")),
        reports_dir=Path(raw.get("reports_dir", "reports")),
        figures_dir=Path(raw.get("figures_dir", "reports/figures")),
        summaries_dir=Path(raw.get("summaries_dir", "reports/summaries")),
    )


def _load_loading(raw: Mapping[str, Any]) -> LoadingConfig:
    _reject_unknown_keys(raw, _LOADING_KEYS, "loading")
    patterns = raw.get("file_patterns", ("*.csv", "*.txt", "*.npy", "*.wav"))
    if isinstance(patterns, str):
        patterns = (patterns,)
    if not isinstance(patterns, list | tuple):
        raise ConfigurationError("loading.file_patterns must be a string or list of strings.")
    return LoadingConfig(
        file_patterns=tuple(str(pattern) for pattern in patterns),
        sampling_rate_hz=_optional_float(raw.get("sampling_rate_hz")),
        label_column=raw.get("label_column"),
        signal_column=raw.get("signal_column"),
    )


def _load_analysis(raw: Mapping[str, Any]) -> AnalysisConfig:
    _reject_unknown_keys(raw, _ANALYSIS_KEYS, "analysis")
    raw_bands = raw.get("frequency_bands_hz", AnalysisConfig().frequency_bands_hz)
    if not isinstance(raw_bands, Mapping):
        raise ConfigurationError("analysis.frequency_bands_hz must be a mapping.")
    bands = {str(name): _frequency_band_tuple(value) for name, value in raw_bands.items()}
    raw_window = _mapping(raw.get("window"), "analysis.window")
    raw_filter = _mapping(raw.get("filtering"), "analysis.filtering")
    raw_quality = _mapping(raw.get("quality"), "analysis.quality")
    raw_modeling = _mapping(raw.get("modeling"), "analysis.modeling")
    _reject_unknown_keys(raw_window, _WINDOW_KEYS, "analysis.window")
    _reject_unknown_keys(raw_filter, _FILTERING_KEYS, "analysis.filtering")
    _reject_unknown_keys(raw_quality, _QUALITY_KEYS, "analysis.quality")
    _reject_unknown_keys(raw_modeling, _MODELING_KEYS, "analysis.modeling")
    return AnalysisConfig(
        frequency_bands_hz=bands,
        window=WindowConfig(
            size_seconds=float(raw_window.get("size_seconds", 1.0)),
            overlap_fraction=float(raw_window.get("overlap_fraction", 0.5)),
        ),
        spectrogram_window_seconds=float(raw.get("spectrogram_window_seconds", 0.1)),
        spectrogram_step_seconds=_optional_float(raw.get("spectrogram_step_seconds")),
        high_frequency_cutoff_hz=_optional_float(raw.get("high_frequency_cutoff_hz")),
        filtering=FilteringConfig(
            enabled=bool(raw_filter.get("enabled", False)),
            kind=raw_filter.get("kind"),
            low_cut_hz=_optional_float(raw_filter.get("low_cut_hz")),
            high_cut_hz=_optional_float(raw_filter.get("high_cut_hz")),
            order=int(raw_filter.get("order", 4)),
            zero_phase=bool(raw_filter.get("zero_phase", True)),
            allow_causal_fallback=bool(raw_filter.get("allow_causal_fallback", False)),
        ),
        quality=QualityConfig(**raw_quality),
        invalid_record_policy=str(raw.get("invalid_record_policy", "skip")),
        modeling=ModelingConfig(
            enabled=bool(raw_modeling.get("enabled", True)),
            anomaly_when_unlabeled=bool(raw_modeling.get("anomaly_when_unlabeled", True)),
            anomaly_contamination=raw_modeling.get("anomaly_contamination", "auto"),
            random_state=int(raw_modeling.get("random_state", 0)),
        ),
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{name} must be a mapping.")
    return value


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _frequency_band_tuple(value: Any) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ConfigurationError("Frequency bands must contain exactly two values.")
    return float(value[0]), float(value[1])


def _reject_unknown_keys(
    raw: Mapping[str, Any], allowed: set[str], section: str
) -> None:
    unknown = sorted(str(key) for key in raw if str(key) not in allowed)
    if unknown:
        raise ConfigurationError(
            f"Unknown {section} setting(s): {', '.join(unknown)}."
        )

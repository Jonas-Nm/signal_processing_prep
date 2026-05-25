"""Tests for unified project and analysis configuration."""

from pathlib import Path

import pytest

from signal_processing_prep.config import AnalysisConfig, FilteringConfig, ProjectConfig, load_config
from signal_processing_prep.errors import ConfigurationError


def test_load_config_reads_default_yaml() -> None:
    config = load_config(Path("configs/default.yaml"))

    assert isinstance(config, ProjectConfig)
    assert config.paths.data_dir == Path("data/raw")
    assert config.loading.file_patterns == ("*.csv", "*.txt", "*.npy", "*.wav")
    assert config.analysis.frequency_bands_hz["low"] == (0.0, 100.0)
    assert config.analysis.filtering.enabled is False


def test_load_config_reads_complete_analysis_policy(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
paths:
  data_dir: custom/raw
loading:
  file_patterns: ["*.npy"]
  sampling_rate_hz: 2048
analysis:
  frequency_bands_hz:
    bearing: [100, 300]
  window:
    size_seconds: 0.25
    overlap_fraction: 0.25
  filtering:
    enabled: true
    kind: bandpass
    low_cut_hz: 10
    high_cut_hz: 500
    order: 3
  quality:
    min_duration_seconds: 0.1
  modeling:
    enabled: false
  invalid_record_policy: raise
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.loading.file_patterns == ("*.npy",)
    assert config.analysis.filtering.kind == "bandpass"
    assert config.analysis.quality.min_duration_seconds == 0.1
    assert config.analysis.modeling.enabled is False
    assert config.analysis.invalid_record_policy == "raise"


@pytest.mark.parametrize(
    "config",
    [
        AnalysisConfig(spectrogram_window_seconds=0.1),
        AnalysisConfig(filtering=FilteringConfig()),
    ],
)
def test_programmatic_analysis_config_constructs(config: AnalysisConfig) -> None:
    assert isinstance(config, AnalysisConfig)


def test_invalid_policy_fails_during_configuration(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="spectrogram_window_seconds"):
        AnalysisConfig(spectrogram_window_seconds=0.0)
    with pytest.raises(ConfigurationError, match="high_cut_hz"):
        FilteringConfig(enabled=True, kind="lowpass")

    path = tmp_path / "bad.yaml"
    path.write_text("analysis:\n  invalid_record_policy: ignore\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid_record_policy"):
        load_config(path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            "filtering:\n  enabled: true\n  kind: lowpass\n  high_cut_hz: 10\n",
            "Unknown configuration setting",
        ),
        ("analysis:\n  modelling:\n    enabled: false\n", "Unknown analysis setting"),
        ("analysis:\n  quality:\n    typo_threshold: 1\n", "Unknown analysis.quality setting"),
    ],
)
def test_yaml_rejects_removed_or_misspelled_settings(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigurationError, match=message):
        load_config(path)


def test_yaml_conversion_errors_use_configuration_error(tmp_path: Path) -> None:
    path = tmp_path / "bad-number.yaml"
    path.write_text("analysis:\n  window:\n    size_seconds: not-a-number\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Invalid configuration value"):
        load_config(path)

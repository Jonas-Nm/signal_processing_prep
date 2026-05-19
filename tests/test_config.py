"""Tests for project configuration loading."""

from pathlib import Path

import pytest

from signal_processing_prep.config import ProjectConfig, load_config


def test_load_config_reads_default_yaml() -> None:
    """The repository default config loads into typed dataclasses."""
    config = load_config(Path("configs/default.yaml"))

    assert isinstance(config, ProjectConfig)
    assert config.paths.data_dir == Path("data/raw")
    assert config.loading.file_patterns == ["*.csv", "*.txt", "*.npy", "*.wav"]
    assert config.loading.sampling_rate_hz is None
    assert config.analysis.frequency_bands_hz["low"] == (0.0, 100.0)
    assert config.analysis.window.size_seconds == 1.0
    assert config.filtering.enabled is False


def test_load_config_applies_custom_values(tmp_path: Path) -> None:
    """Custom YAML values are parsed and converted to useful types."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
paths:
  data_dir: custom/raw
loading:
  file_patterns:
    - "*.npy"
  sampling_rate_hz: 2048
  signal_column: vibration
analysis:
  frequency_bands_hz:
    bearing:
      - 100
      - 300
  window:
    size_seconds: 0.25
    overlap_fraction: 0.25
filtering:
  enabled: true
  kind: bandpass
  low_cut_hz: 10
  high_cut_hz: 500
  order: 3
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.paths.data_dir == Path("custom/raw")
    assert config.loading.file_patterns == ["*.npy"]
    assert config.loading.sampling_rate_hz == 2048.0
    assert config.loading.signal_column == "vibration"
    assert config.analysis.frequency_bands_hz["bearing"] == (100.0, 300.0)
    assert config.analysis.window.overlap_fraction == 0.25
    assert config.filtering.kind == "bandpass"
    assert config.filtering.high_cut_hz == 500.0


def test_load_config_treats_null_sections_as_defaults(tmp_path: Path) -> None:
    """Null config sections fall back to defaults rather than failing indirectly."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("loading: null\nanalysis: null\n", encoding="utf-8")

    config = load_config(config_path)

    assert config.loading.file_patterns == ["*.csv", "*.txt", "*.npy", "*.wav"]
    assert config.analysis.window.size_seconds == 1.0


def test_load_config_rejects_non_mapping_sections(tmp_path: Path) -> None:
    """Invalid section shapes fail with a clear error."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("loading:\n  - not\n  - a\n  - mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="section 'loading' must be a mapping"):
        load_config(config_path)


def test_load_config_rejects_invalid_analysis_and_filter_values(tmp_path: Path) -> None:
    """Config validation catches invalid windows, bands, and filter shapes early."""
    bad_window = tmp_path / "bad_window.yaml"
    bad_window.write_text("analysis:\n  window:\n    overlap_fraction: 1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlap_fraction"):
        load_config(bad_window)

    bad_band = tmp_path / "bad_band.yaml"
    bad_band.write_text(
        "analysis:\n  frequency_bands_hz:\n    bad: [10, 5]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="greater than"):
        load_config(bad_band)

    bad_filter = tmp_path / "bad_filter.yaml"
    bad_filter.write_text(
        "filtering:\n  enabled: true\n  kind: bandpass\n  low_cut_hz: 20\n  high_cut_hz: 10\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="low_cut_hz"):
        load_config(bad_filter)

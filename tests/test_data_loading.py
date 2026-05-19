"""Tests for loading signal files into SignalRecord objects."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from signal_processing_prep.data_loading import (
    load_signal_dataset,
    load_signal_file,
    load_signal_file_channels,
)


def test_load_csv_uses_explicit_signal_and_label_columns(tmp_path: Path) -> None:
    """CSV loading can use explicit signal and label columns."""
    path = tmp_path / "record.csv"
    pd.DataFrame(
        {
            "time": [0.0, 0.1, 0.2],
            "vibration": [1.0, 0.0, -1.0],
            "condition": ["normal", "normal", "normal"],
        }
    ).to_csv(path, index=False)

    record = load_signal_file(
        path,
        sampling_rate_hz=10.0,
        signal_column="vibration",
        label_column="condition",
    )

    np.testing.assert_allclose(record.values, [1.0, 0.0, -1.0])
    assert record.sampling_rate_hz == 10.0
    assert record.label == "normal"
    assert record.name == "record"
    assert record.metadata["signal_column"] == "vibration"


def test_load_csv_defaults_to_first_numeric_column(tmp_path: Path) -> None:
    """CSV loading chooses the only non-time numeric column when unambiguous."""
    path = tmp_path / "numeric.csv"
    pd.DataFrame({"time": [0.0, 0.1, 0.2], "signal": [1, 2, 3]}).to_csv(
        path, index=False
    )

    record = load_signal_file(path, sampling_rate_hz=100.0)

    np.testing.assert_allclose(record.values, [1.0, 2.0, 3.0])
    assert record.metadata["signal_column"] == "signal"


def test_load_csv_infers_sampling_rate_and_time_axis_metadata(tmp_path: Path) -> None:
    """CSV time columns provide acquisition metadata and a fallback sampling rate."""
    path = tmp_path / "timed.csv"
    pd.DataFrame(
        {
            "time": [0.0, 0.1, 0.2, 0.35],
            "signal": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(path, index=False)

    record = load_signal_file(path)

    assert record.sampling_rate_hz == pytest.approx(10.0)
    assert record.metadata["sampling_rate_source"] == "time_column"
    assert record.metadata["time_column"] == "time"
    assert record.metadata["time_gap_count"] == 0
    assert record.metadata["time_step_jitter_fraction"] > 0.0


def test_load_csv_requires_signal_column_for_ambiguous_numeric_columns(
    tmp_path: Path,
) -> None:
    """CSV loading refuses to guess between multiple non-time numeric columns."""
    path = tmp_path / "ambiguous.csv"
    pd.DataFrame({"time": [0.0, 0.1], "x": [1.0, 2.0], "y": [3.0, 4.0]}).to_csv(
        path, index=False
    )

    with pytest.raises(ValueError, match="provide signal_column"):
        load_signal_file(path, sampling_rate_hz=10.0)


def test_load_csv_rejects_multiple_labels_for_one_record(tmp_path: Path) -> None:
    """A single loaded record cannot represent multiple labels."""
    path = tmp_path / "multi_label.csv"
    pd.DataFrame({"signal": [1, 2], "label": ["a", "b"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="multiple CSV labels"):
        load_signal_file(path, sampling_rate_hz=100.0, label_column="label")


def test_load_txt_requires_sampling_rate_and_returns_record(tmp_path: Path) -> None:
    """TXT loading reads one-dimensional numeric samples."""
    path = tmp_path / "record.txt"
    np.savetxt(path, np.array([0.0, 1.0, 0.0]))

    record = load_signal_file(path, sampling_rate_hz=50.0, label="txt_label")

    np.testing.assert_allclose(record.values, [0.0, 1.0, 0.0])
    assert record.label == "txt_label"
    assert record.sampling_rate_hz == 50.0


def test_load_npy_reads_one_dimensional_array(tmp_path: Path) -> None:
    """NPY loading accepts one-dimensional arrays."""
    path = tmp_path / "record.npy"
    np.save(path, np.array([1.0, 2.0, 3.0]))

    record = load_signal_file(path, sampling_rate_hz=25.0)

    np.testing.assert_allclose(record.values, [1.0, 2.0, 3.0])
    assert record.name == "record"


def test_load_wav_uses_file_sampling_rate_and_scales_integer_data(tmp_path: Path) -> None:
    """WAV loading reads the file sampling rate and normalizes integer samples."""
    path = tmp_path / "record.wav"
    samples = np.array([0, 16384, -16384], dtype=np.int16)
    wavfile.write(path, 8000, samples)

    record = load_signal_file(path)

    assert record.sampling_rate_hz == 8000.0
    assert record.metadata["format"] == "wav"
    assert record.metadata["original_dtype"] == "int16"
    np.testing.assert_allclose(record.values, [0.0, 0.5, -0.5], atol=1e-4)


def test_load_wav_centers_unsigned_integer_data(tmp_path: Path) -> None:
    """Unsigned WAV samples are centered so midpoint silence maps to zero."""
    path = tmp_path / "unsigned.wav"
    samples = np.array([0, 128, 255], dtype=np.uint8)
    wavfile.write(path, 8000, samples)

    record = load_signal_file(path)

    assert record.metadata["original_dtype"] == "uint8"
    np.testing.assert_allclose(record.values, [-1.0, 0.0, 0.9921875])


def test_load_signal_file_channels_splits_multi_channel_wav(tmp_path: Path) -> None:
    """Multi-channel files are represented as explicit single-channel records."""
    path = tmp_path / "stereo.wav"
    samples = np.array([[0, 0], [16384, -16384], [0, 0]], dtype=np.int16)
    wavfile.write(path, 8000, samples)

    records = load_signal_file_channels(path)

    assert [record.name for record in records] == ["stereo_0", "stereo_1"]
    assert [record.metadata["channel_index"] for record in records] == [0, 1]
    np.testing.assert_allclose(records[0].values, [0.0, 0.5, 0.0], atol=1e-4)
    np.testing.assert_allclose(records[1].values, [0.0, -0.5, 0.0], atol=1e-4)


def test_load_signal_dataset_uses_config_patterns_and_metadata_table(tmp_path: Path) -> None:
    """Dataset loading discovers configured files and applies external labels."""
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    signal_path = data_dir / "record.csv"
    pd.DataFrame({"time": [0.0, 0.1, 0.2], "x": [1.0, 0.0, -1.0]}).to_csv(
        signal_path,
        index=False,
    )
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame({"file_name": ["record.csv"], "label": ["normal"], "sensor": ["accel"]}).to_csv(
        metadata_path,
        index=False,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
paths:
  data_dir: {data_dir.as_posix()}
loading:
  file_patterns:
    - "*.csv"
  signal_column: x
""",
        encoding="utf-8",
    )

    records = load_signal_dataset(config_path, metadata_table=metadata_path)

    assert len(records) == 1
    assert records[0].label == "normal"
    assert records[0].metadata["external_metadata"]["sensor"] == "accel"
    assert records[0].sampling_rate_hz == pytest.approx(10.0)


def test_non_wav_files_require_sampling_rate(tmp_path: Path) -> None:
    """Formats without embedded sampling rate require explicit sampling_rate_hz."""
    path = tmp_path / "record.npy"
    np.save(path, np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="sampling_rate_hz is required"):
        load_signal_file(path)


def test_loader_rejects_unsupported_extension(tmp_path: Path) -> None:
    """Unsupported file formats fail clearly."""
    path = tmp_path / "record.bin"
    path.write_bytes(b"not a supported signal file")

    with pytest.raises(ValueError, match="Unsupported signal file extension"):
        load_signal_file(path, sampling_rate_hz=1.0)

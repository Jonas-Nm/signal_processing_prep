"""Tests for the canonical dataset-loading facade."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import wavfile

from signal_processing_prep._loading_formats import (
    CsvSignalFileLoader,
    NpySignalFileLoader,
    TxtSignalFileLoader,
    WavSignalFileLoader,
)
from signal_processing_prep.config import AnalysisConfig, LoadingConfig, ModelingConfig, PathsConfig, ProjectConfig
from signal_processing_prep.data_loading import SignalDatasetLoader
from signal_processing_prep.records import SignalRecord


def test_csv_loader_uses_typed_provenance_and_acquisition(tmp_path: Path) -> None:
    path = tmp_path / "record.csv"
    pd.DataFrame(
        {"time": [0.0, 0.1, 0.2, 0.35], "vibration": [1.0, 0.0, -1.0, 0.0], "condition": ["normal"] * 4}
    ).to_csv(path, index=False)

    record = SignalDatasetLoader().load_file(path, signal_column="vibration", label_column="condition")

    assert record.label == "normal"
    assert record.provenance.source_format == "csv"
    assert record.provenance.signal_column == "vibration"
    assert record.acquisition.sampling_rate_source == "time_column"
    assert record.acquisition.time_step_jitter_fraction > 0.0
    assert not record.attributes


def test_csv_loader_refuses_ambiguous_signal_columns(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.csv"
    pd.DataFrame({"time": [0.0, 0.1], "x": [1.0, 2.0], "y": [3.0, 4.0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="provide signal_column"):
        SignalDatasetLoader().load_file(path, sampling_rate_hz=10.0)


@pytest.mark.parametrize(
    ("loader", "suffix"),
    [
        (CsvSignalFileLoader(), ".csv"),
        (TxtSignalFileLoader(), ".txt"),
        (NpySignalFileLoader(), ".npy"),
        (WavSignalFileLoader(), ".wav"),
    ],
)
def test_built_in_parsers_share_record_contract(tmp_path: Path, loader, suffix: str) -> None:
    path = tmp_path / f"record{suffix}"
    if suffix == ".csv":
        pd.DataFrame({"signal": [0.0, 1.0]}).to_csv(path, index=False)
    elif suffix == ".txt":
        np.savetxt(path, np.array([0.0, 1.0]))
    elif suffix == ".npy":
        np.save(path, np.array([0.0, 1.0]))
    else:
        wavfile.write(path, 8000, np.array([0, 16384], dtype=np.int16))

    record = loader.load_record(path, sampling_rate_hz=None if suffix == ".wav" else 8000.0)

    assert record.provenance.source_path == str(path)
    assert record.sampling_rate_hz == 8000.0


def test_wav_loader_scales_samples_and_splits_channels(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    wavfile.write(path, 8000, np.array([[0, 0], [16384, -16384], [0, 0]], dtype=np.int16))

    dataset = SignalDatasetLoader().load_file_channels(path)

    assert [record.name for record in dataset] == ["stereo_0", "stereo_1"]
    assert [record.provenance.channel_index for record in dataset] == [0, 1]
    assert dataset.records[0].provenance.original_dtype == "int16"
    np.testing.assert_allclose(dataset.records[0].values, [0.0, 0.5, 0.0], atol=1e-4)


def test_dataset_loading_joins_typed_annotations(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    pd.DataFrame({"time": [0.0, 0.1, 0.2], "x": [1.0, 0.0, -1.0]}).to_csv(
        data_dir / "record.csv", index=False
    )
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "file_name": ["record.csv"],
            "label": ["normal"],
            "sensor": ["accel"],
            "split": ["test"],
            "is_anomalous": [True],
            "anomaly_kind": ["burst"],
            "anomaly_start_seconds": [0.1],
            "anomaly_end_seconds": [0.2],
        }
    ).to_csv(metadata, index=False)
    project = ProjectConfig(
        paths=PathsConfig(data_dir=data_dir),
        loading=LoadingConfig(file_patterns=("*.csv",), signal_column="x"),
    )

    dataset = SignalDatasetLoader().load_dataset(project, metadata_table=metadata)
    record = dataset.records[0]

    assert record.label == "normal"
    assert record.annotations.sensor == "accel"
    assert record.annotations.split == "test"
    assert record.annotations.intervals[0].start_seconds == 0.1
    assert "external_metadata" not in record.attributes


def test_dataset_loader_accepts_registered_parser(tmp_path: Path) -> None:
    (tmp_path / "capture.foo").write_text("custom", encoding="utf-8")

    class FakeLoader:
        suffix = ".foo"

        def load_record(self, path: Path, **_kwargs: object) -> SignalRecord:
            return SignalRecord([1.0, 0.0], 2.0, name=path.stem)

        def load_channels(self, path: Path, **kwargs: object) -> list[SignalRecord]:
            return [self.load_record(path, **kwargs)]

    loader = SignalDatasetLoader({".foo": FakeLoader()})
    dataset = loader.load_dataset(
        ProjectConfig(
            paths=PathsConfig(data_dir=tmp_path),
            loading=LoadingConfig(file_patterns=("*.foo",)),
        )
    )
    assert dataset.records[0].name == "capture"


def test_non_wav_data_without_rate_or_time_axis_fails(tmp_path: Path) -> None:
    path = tmp_path / "record.npy"
    np.save(path, np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="sampling_rate_hz is required"):
        SignalDatasetLoader().load_file(path)


def test_npy_loader_preserves_encoded_dtype_in_provenance(tmp_path: Path) -> None:
    path = tmp_path / "record.npy"
    np.save(path, np.array([1, 2], dtype=np.int16))

    record = SignalDatasetLoader().load_file(path, sampling_rate_hz=100.0)

    assert record.provenance.original_dtype == "int16"
    assert record.values.dtype == np.float64


def test_dataset_loader_can_skip_malformed_record_files(tmp_path: Path) -> None:
    pd.DataFrame({"time": [0.0, 0.1], "signal": ["bad", "values"]}).to_csv(
        tmp_path / "bad.csv", index=False
    )
    pd.DataFrame({"time": [0.0, 0.1], "signal": [0.0, 1.0]}).to_csv(
        tmp_path / "good.csv", index=False
    )
    project = ProjectConfig(
        paths=PathsConfig(data_dir=tmp_path),
        loading=LoadingConfig(file_patterns=("*.csv",), signal_column="signal"),
        analysis=AnalysisConfig(modeling=ModelingConfig(enabled=False)),
    )

    loading = SignalDatasetLoader().load_dataset_result(
        project, invalid_record_policy="skip"
    )

    assert [record.provenance.source_name for record in loading.dataset] == ["good"]
    assert "Loading skipped bad.csv" in loading.notes[0]


def test_dataset_loader_can_skip_invalid_record_annotations(tmp_path: Path) -> None:
    for name in ("bad.csv", "good.csv"):
        pd.DataFrame({"time": [0.0, 0.1], "signal": [0.0, 1.0]}).to_csv(
            tmp_path / name, index=False
        )
    annotations = pd.DataFrame(
        {
            "file_name": ["bad.csv", "good.csv"],
            "anomaly_start_seconds": [-1.0, 0.0],
            "anomaly_end_seconds": [0.1, 0.1],
        }
    )
    project = ProjectConfig(
        paths=PathsConfig(data_dir=tmp_path),
        loading=LoadingConfig(file_patterns=("*.csv",), signal_column="signal"),
    )

    loading = SignalDatasetLoader().load_dataset_result(
        project, metadata_table=annotations, invalid_record_policy="skip"
    )

    assert [record.provenance.source_name for record in loading.dataset] == ["good"]
    assert "Loading skipped bad.csv" in loading.notes[0]

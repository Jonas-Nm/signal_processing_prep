"""Tests for optional synthetic spectrogram autoencoder helpers."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.generate_spectrogram_autoencoder_demo_csv import generate_demo_records
import signal_processing_prep.autoencoders as autoencoders
from signal_processing_prep.autoencoders import (
    SpectrogramAutoencoderConfig,
    run_spectrogram_autoencoder,
    split_autoencoder_demo_records,
    spectrogram_patches_from_records,
)
from signal_processing_prep.config import load_config
from signal_processing_prep.data_loading import load_signal_dataset
from signal_processing_prep.records import SignalRecord


def _write_demo_config(tmp_path):
    config_path = tmp_path / "spectrogram_autoencoder_demo.yaml"
    config_path.write_text(
        f"""
paths:
  data_dir: "{(tmp_path / "demo" / "records").as_posix()}"
  reports_dir: reports
  figures_dir: reports/figures
  summaries_dir: reports/summaries

loading:
  file_patterns:
    - "*.csv"
  sampling_rate_hz: null
  label_column: null
  signal_column: voltage

analysis:
  frequency_bands_hz: {{}}
  window:
    size_seconds: 0.5
    overlap_fraction: 0.8

filtering:
  enabled: false
  kind: null
  low_cut_hz: null
  high_cut_hz: null
  order: 4
""",
        encoding="utf-8",
    )
    return config_path


def _loaded_demo_dataset(tmp_path, *, duration_seconds=2.0, sampling_rate_hz=1000.0):
    output_dir = tmp_path / "demo"
    metadata = generate_demo_records(
        output_dir=output_dir,
        n_train_records=3,
        n_test_normal_records=1,
        n_test_anomalous_records=2,
        duration_seconds=duration_seconds,
        sampling_rate_hz=sampling_rate_hz,
        random_seed=3,
    )
    config_path = _write_demo_config(tmp_path)
    records = load_signal_dataset(config_path, metadata_table=output_dir / "metadata.csv")
    return split_autoencoder_demo_records(records), metadata, output_dir


def test_generate_spectrogram_autoencoder_demo_csv_writes_records_and_metadata(tmp_path) -> None:
    """The file generator writes loader-compatible signal CSVs and metadata."""
    output_dir = tmp_path / "demo"
    metadata = generate_demo_records(
        output_dir=output_dir,
        n_train_records=2,
        n_test_normal_records=1,
        n_test_anomalous_records=2,
        duration_seconds=1.0,
        sampling_rate_hz=1000.0,
        random_seed=3,
    )

    metadata_path = output_dir / "metadata.csv"
    csv_paths = sorted((output_dir / "records").glob("*.csv"))

    assert metadata_path.exists()
    assert len(csv_paths) == 5
    assert len(metadata) == 5
    assert set(metadata["split"]) == {"train", "test"}
    assert set(metadata["label"]) == {"normal", "anomalous"}
    assert {"file_name", "source_path", "anomaly_start_seconds", "anomaly_end_seconds"}.issubset(
        metadata.columns
    )
    assert metadata["source_path"].str.startswith("records/").all()
    assert all(not Path(source_path).is_absolute() for source_path in metadata["source_path"])
    assert all((output_dir / source_path).exists() for source_path in metadata["source_path"])
    for path in csv_paths:
        frame = pd.read_csv(path)
        assert list(frame.columns) == ["time", "voltage"]
        assert len(frame) == 1000

    anomalous = metadata[metadata["label"] == "anomalous"]
    assert anomalous["anomaly_start_seconds"].notna().all()
    assert anomalous["anomaly_end_seconds"].notna().all()
    assert (anomalous["anomaly_start_seconds"] < anomalous["anomaly_end_seconds"]).all()


def test_generate_spectrogram_autoencoder_demo_csv_replaces_stale_records(tmp_path) -> None:
    """Rerunning the generator leaves signal CSVs aligned with metadata."""
    output_dir = tmp_path / "demo"
    generate_demo_records(
        output_dir=output_dir,
        n_train_records=3,
        n_test_normal_records=1,
        n_test_anomalous_records=2,
        duration_seconds=1.0,
        sampling_rate_hz=1000.0,
        random_seed=3,
    )
    stale_path = output_dir / "records" / "stale.csv"
    stale_path.write_text("time,voltage\n0.0,0.0\n", encoding="utf-8")

    metadata = generate_demo_records(
        output_dir=output_dir,
        n_train_records=1,
        n_test_normal_records=0,
        n_test_anomalous_records=1,
        duration_seconds=1.0,
        sampling_rate_hz=1000.0,
        random_seed=4,
    )

    csv_names = sorted(path.name for path in (output_dir / "records").glob("*.csv"))
    assert csv_names == sorted(metadata["file_name"].tolist())
    assert not stale_path.exists()


def test_existing_loader_splits_file_based_demo_dataset(tmp_path) -> None:
    """Existing data loaders attach metadata that can split train and test records."""
    dataset, metadata, _ = _loaded_demo_dataset(tmp_path)

    assert len(dataset.train_records) == 3
    assert len(dataset.test_records) == 3
    assert {record.label for record in dataset.train_records} == {"normal"}
    assert {record.label for record in dataset.test_records} == {"normal", "anomalous"}
    assert len(metadata) == 6

    anomalous = [record for record in dataset.test_records if record.label == "anomalous"]
    for record in anomalous:
        external_metadata = record.metadata["external_metadata"]
        assert external_metadata["split"] == "test"
        assert external_metadata["is_anomalous"] is True
        assert 0.0 <= external_metadata["anomaly_start_seconds"] < external_metadata["anomaly_end_seconds"]
        assert external_metadata["anomaly_end_seconds"] <= record.duration_seconds


def test_notebook_style_loading_resolves_relative_data_dir_from_project_root(
    tmp_path, monkeypatch
) -> None:
    """Notebook loading works when its kernel starts in the notebooks directory."""
    project_root = tmp_path / "project"
    output_dir = project_root / "data" / "raw" / "spectrogram_autoencoder_demo"
    generate_demo_records(
        output_dir=output_dir,
        n_train_records=2,
        n_test_normal_records=1,
        n_test_anomalous_records=1,
        duration_seconds=1.0,
        sampling_rate_hz=1000.0,
        random_seed=3,
    )
    source_config = Path("configs/spectrogram_autoencoder_demo.yaml").read_text(encoding="utf-8")
    config_path = project_root / "configs" / "spectrogram_autoencoder_demo.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(source_config, encoding="utf-8")
    notebooks_dir = project_root / "notebooks"
    notebooks_dir.mkdir()
    monkeypatch.chdir(notebooks_dir)

    config = load_config(config_path)
    assert not config.paths.data_dir.is_absolute()
    config = replace(
        config,
        paths=replace(config.paths, data_dir=project_root / config.paths.data_dir),
    )
    records = load_signal_dataset(config, metadata_table=output_dir / "metadata.csv")
    dataset = split_autoencoder_demo_records(records)

    assert len(records) == 4
    assert len(dataset.train_records) == 2
    assert {record.label for record in dataset.test_records} == {"normal", "anomalous"}


def test_spectrogram_patch_extraction_returns_finite_fixed_shape_patches(tmp_path) -> None:
    """Spectrogram patch extraction produces finite arrays and timing metadata."""
    dataset, _, _ = _loaded_demo_dataset(tmp_path)
    config = SpectrogramAutoencoderConfig(
        patch_window_seconds=0.4,
        patch_step_seconds=0.2,
        spectrogram_window_seconds=0.08,
        spectrogram_step_seconds=0.04,
        max_frequency_hz=450.0,
    )

    patch_set = spectrogram_patches_from_records(dataset.test_records, config)

    assert patch_set.patches.ndim == 3
    assert patch_set.patches.shape[0] == len(patch_set.metadata)
    assert patch_set.patches.shape[1:] == (
        len(patch_set.frequencies_hz),
        len(patch_set.times_seconds),
    )
    assert np.isfinite(patch_set.patches).all()
    assert {
        "record_name",
        "label",
        "patch_start_seconds",
        "patch_end_seconds",
        "patch_center_seconds",
        "known_anomaly_overlap",
    }.issubset(patch_set.metadata.columns)
    assert patch_set.metadata["known_anomaly_overlap"].any()


def test_patch_inspection_variants_show_expected_resolution_and_density_effects(tmp_path) -> None:
    """Walkthrough variants expose the intended spectrogram representation tradeoffs."""
    dataset, _, _ = _loaded_demo_dataset(
        tmp_path,
        duration_seconds=3.0,
        sampling_rate_hz=2000.0,
    )
    transient_record = next(
        record
        for record in dataset.test_records
        if record.metadata["external_metadata"].get("anomaly_kind") == "transient_tone"
    )
    baseline = SpectrogramAutoencoderConfig(
        patch_window_seconds=0.5,
        patch_step_seconds=0.1,
        spectrogram_window_seconds=0.064,
        spectrogram_step_seconds=0.032,
        max_frequency_hz=800.0,
    )
    variants = {
        "baseline": baseline,
        "shorter_context": replace(baseline, patch_window_seconds=0.25),
        "shorter_spectrogram": replace(
            baseline,
            spectrogram_window_seconds=0.032,
            spectrogram_step_seconds=0.016,
        ),
        "longer_spectrogram": replace(
            baseline,
            spectrogram_window_seconds=0.128,
            spectrogram_step_seconds=0.064,
        ),
        "reduced_frequency": replace(baseline, max_frequency_hz=400.0),
    }
    patch_sets = {
        name: spectrogram_patches_from_records([transient_record], variant)
        for name, variant in variants.items()
    }

    assert all(np.isfinite(patch_set.patches).all() for patch_set in patch_sets.values())
    assert (
        patch_sets["shorter_spectrogram"].patches.shape[1]
        < patch_sets["baseline"].patches.shape[1]
        < patch_sets["longer_spectrogram"].patches.shape[1]
    )
    assert (
        patch_sets["shorter_spectrogram"].patches.shape[2]
        > patch_sets["baseline"].patches.shape[2]
        > patch_sets["longer_spectrogram"].patches.shape[2]
    )
    assert patch_sets["reduced_frequency"].frequencies_hz[-1] <= 400.0
    assert (
        patch_sets["reduced_frequency"].patches.shape[1]
        < patch_sets["baseline"].patches.shape[1]
    )

    dense = spectrogram_patches_from_records(
        [transient_record],
        replace(baseline, patch_step_seconds=0.05),
    )
    sparse = spectrogram_patches_from_records(
        [transient_record],
        replace(baseline, patch_step_seconds=0.2),
    )
    assert len(dense.patches) > len(patch_sets["baseline"].patches) > len(sparse.patches)


def test_synthetic_autoencoder_notebook_includes_pre_training_patch_review() -> None:
    """The autoencoder notebook explains and visualizes patches before training."""
    notebook_path = Path("notebooks/04_synthetic_spectrogram_autoencoder_walkthrough.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "Inspect The Model Inputs Before Training" in source
    assert "Parameter Sensitivity" in source
    assert "variant_configs" in source
    assert "step_configs" in source


def test_short_synthetic_records_keep_valid_anomaly_intervals(tmp_path) -> None:
    """Anomaly metadata remains valid even for short synthetic demo records."""
    metadata = generate_demo_records(
        output_dir=tmp_path / "short_demo",
        n_train_records=1,
        n_test_normal_records=0,
        n_test_anomalous_records=2,
        duration_seconds=0.5,
        sampling_rate_hz=1000.0,
        random_seed=8,
    )

    anomalous = metadata[metadata["label"] == "anomalous"]
    assert (0.0 <= anomalous["anomaly_start_seconds"]).all()
    assert (anomalous["anomaly_start_seconds"] < anomalous["anomaly_end_seconds"]).all()
    assert (anomalous["anomaly_end_seconds"] <= anomalous["duration_seconds"]).all()


def test_spectrogram_patch_extraction_rejects_axis_mismatch(monkeypatch) -> None:
    """Patch extraction rejects same-shaped patches with incompatible axes."""
    records = [
        SignalRecord(np.zeros(200), sampling_rate_hz=1000.0, name="a"),
        SignalRecord(np.zeros(400), sampling_rate_hz=2000.0, name="b"),
    ]
    config = SpectrogramAutoencoderConfig(
        patch_window_seconds=0.1,
        patch_step_seconds=0.1,
        spectrogram_window_seconds=0.05,
        spectrogram_step_seconds=0.025,
    )

    def fake_patch(window, _config):
        frequencies = np.array([1.0, 2.0]) if window.sampling_rate_hz == 1000.0 else np.array([10.0, 20.0])
        return np.ones((2, 2)), frequencies, np.array([0.0, 0.1])

    monkeypatch.setattr(autoencoders, "_spectrogram_patch", fake_patch)

    with pytest.raises(ValueError, match="frequency and time axes"):
        spectrogram_patches_from_records(records, config)


def test_package_imports_without_requiring_torch() -> None:
    """Public autoencoder helpers can be imported without importing PyTorch eagerly."""
    import signal_processing_prep

    assert hasattr(signal_processing_prep, "SpectrogramAutoencoderConfig")
    assert hasattr(signal_processing_prep, "spectrogram_patches_from_records")
    assert hasattr(signal_processing_prep, "run_spectrogram_autoencoder")


def test_run_spectrogram_autoencoder_requires_torch_when_missing() -> None:
    """A missing PyTorch install fails at training time with an actionable message."""
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("PyTorch is installed; lazy missing-dependency path is not active.")
    train = [SignalRecord(np.zeros(1000), sampling_rate_hz=1000.0, label="normal", name="train")]
    test = [SignalRecord(np.zeros(1000), sampling_rate_hz=1000.0, label="normal", name="test")]

    with pytest.raises(ImportError, match="PyTorch is required"):
        run_spectrogram_autoencoder(train, test)


def test_spectrogram_autoencoder_tiny_training_run_scores_known_anomalies(tmp_path) -> None:
    """A tiny optional PyTorch run preserves metadata and ranks an anomalous patch highly."""
    pytest.importorskip("torch")
    dataset, _, _ = _loaded_demo_dataset(
        tmp_path,
        duration_seconds=3.0,
        sampling_rate_hz=1000.0,
    )
    config = SpectrogramAutoencoderConfig(
        patch_window_seconds=0.4,
        patch_step_seconds=0.1,
        spectrogram_window_seconds=0.08,
        spectrogram_step_seconds=0.04,
        max_frequency_hz=450.0,
        n_epochs=2,
        batch_size=8,
        latent_channels=2,
        random_state=7,
    )

    result = run_spectrogram_autoencoder(dataset.train_records, dataset.test_records, config)
    predictions = result.evaluation.predictions

    assert len(result.training_losses) == 2
    assert np.isfinite(predictions["anomaly_score"]).all()
    assert {"patch_start_seconds", "patch_end_seconds", "known_anomaly_overlap"}.issubset(
        predictions.columns
    )
    top = predictions.sort_values("anomaly_score", ascending=False).head(5)
    assert top["known_anomaly_overlap"].any()

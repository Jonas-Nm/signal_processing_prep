"""Architecture contracts for the deliberate workflow reset."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import signal_processing_prep as package
from signal_processing_prep.artifacts import FeatureTable, PredictionTable, QualityTable
from signal_processing_prep.config import AnalysisConfig, ModelingConfig, load_config
from signal_processing_prep.data_loading import SignalDatasetLoader
from signal_processing_prep.pipeline import SignalAnalysisPipeline, run_synthetic_analysis


def test_root_api_is_small_and_workflow_focused() -> None:
    assert set(package.__all__) == {
        "AnalysisConfig", "AnalysisResult", "ProjectConfig", "SignalAnalysisPipeline",
        "SignalDataset", "SignalDatasetLoader", "SignalRecord", "__version__",
        "load_config", "run_synthetic_analysis",
    }


def test_pipeline_result_has_typed_artifacts_and_frame_views() -> None:
    result = run_synthetic_analysis(AnalysisConfig(modeling=ModelingConfig(enabled=False)))

    assert isinstance(result.features, FeatureTable)
    assert isinstance(result.quality, QualityTable)
    assert isinstance(result.feature_frame, pd.DataFrame)
    assert {"record_name", "rms", "dominant_frequency_hz"}.issubset(result.feature_frame.columns)
    assert "# Signal Analysis Summary" in result.markdown_summary


def test_configured_loading_creates_typed_facts_without_legacy_metadata(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    np.save(data_dir / "multi.npy", np.column_stack([np.arange(4.0), -np.arange(4.0)]))
    pd.DataFrame({"time": [0.0, 0.1, 0.2], "signal": [0.0, 1.0, 0.0]}).to_csv(
        data_dir / "record.csv", index=False
    )
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"file_name": ["record.csv"], "label": ["normal"], "sensor": ["accel"]}).to_csv(
        metadata, index=False
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f'paths:\n  data_dir: "{data_dir.as_posix()}"\nloading:\n  file_patterns: ["*.csv"]\n',
        encoding="utf-8",
    )

    dataset = SignalDatasetLoader().load_dataset(load_config(config), metadata_table=metadata)
    record = dataset.records[0]
    assert record.annotations.sensor == "accel"
    assert record.acquisition.time_column == "time"
    assert "external_metadata" not in record.attributes


def test_notebooks_do_not_read_removed_legacy_metadata() -> None:
    for path in Path("notebooks").glob("*.ipynb"):
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]
        )
        assert 'metadata["external_metadata"]' not in source

"""Tests for report objects and artifact-driven report building."""

import pandas as pd

from signal_processing_prep.artifacts import FeatureTable, QualityTable
from signal_processing_prep.reporting import AnalysisReport, AnalysisReportBuilder, dataset_overview


def test_analysis_report_renders_and_saves_standard_sections(tmp_path) -> None:
    report = AnalysisReport(
        dataset_overview="Two synthetic records.",
        figures={"FFT": "reports/figures/fft.png"},
    )
    markdown = report.render_markdown()
    path = report.save(tmp_path, stem="Demo Report", run_date="2026-05-19")

    assert "# Signal Analysis Summary" in markdown
    assert "## Signal quality observations" in markdown
    assert "FFT" in markdown
    assert path.name == "2026-05-19_Demo_Report.md"
    assert path.read_text(encoding="utf-8") == markdown


def test_dataset_overview_and_builder_consume_artifacts() -> None:
    features = FeatureTable.from_dataframe(
        pd.DataFrame({"record_name": ["a", "b"], "label": ["x", "y"], "rms": [1.0, 2.0]}),
        feature_columns=("rms",),
    )
    quality = QualityTable(pd.DataFrame({"record_name": ["a", "b"], "issues": ["", ""]}))

    text = dataset_overview(2, features)
    report = AnalysisReportBuilder().build(
        record_count=2,
        quality=quality,
        features=features,
        supervised_models={},
        anomaly_model=None,
        modeling_notes=[],
        processing_notes=[],
    )

    assert "Observed label counts" in text
    assert "RMS ranged" in report.render_markdown()

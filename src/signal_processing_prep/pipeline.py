"""High-level orchestration for the preferred signal-analysis workflow."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from signal_processing_prep.features import (
    FeatureExtractionConfig,
    FrequencyBand,
    extract_features,
)
from signal_processing_prep.modeling import (
    ModelEvaluation,
    anomaly_summary_text,
    run_isolation_forest,
    run_supervised_baselines,
)
from signal_processing_prep.config import FilteringConfig
from signal_processing_prep.preprocessing import apply_configured_filter
from signal_processing_prep.quality import (
    QualityCheckConfig,
    assess_dataset_quality,
)
from signal_processing_prep.records import SignalRecord
from signal_processing_prep.reporting import (
    MarkdownReport,
    dataset_overview_from_records,
    generate_markdown_summary,
)
from signal_processing_prep.synthetic import make_synthetic_dataset


@dataclass(frozen=True)
class AnalysisPipelineConfig:
    """Configuration for the lightweight analysis pipeline."""

    frequency_bands: Sequence[FrequencyBand] = field(default_factory=tuple)
    run_modeling: bool = True
    run_anomaly_when_unlabeled: bool = True
    anomaly_contamination: float | str = "auto"
    random_state: int = 0
    quality: QualityCheckConfig = field(default_factory=QualityCheckConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    features: FeatureExtractionConfig | None = None


@dataclass(frozen=True)
class AnalysisPipelineResult:
    """Artifacts produced by the high-level analysis pipeline."""

    records: tuple[SignalRecord, ...]
    quality: pd.DataFrame
    features: pd.DataFrame
    supervised_models: dict[str, ModelEvaluation]
    anomaly_model: ModelEvaluation | None
    markdown_summary: str
    modeling_notes: tuple[str, ...]


def analyze_records(
    records: Sequence[SignalRecord],
    config: AnalysisPipelineConfig | None = None,
) -> AnalysisPipelineResult:
    """Run quality checks, feature extraction, optional modeling, and summary text."""
    if len(records) == 0:
        raise ValueError("At least one SignalRecord is required.")
    if config is None:
        config = AnalysisPipelineConfig()

    feature_config = config.features or FeatureExtractionConfig(
        frequency_bands=config.frequency_bands
    )
    raw_records = tuple(records)
    quality = assess_dataset_quality(raw_records, config.quality)
    analysis_records = tuple(
        apply_configured_filter(record, config.filtering) for record in raw_records
    )
    features = extract_features(analysis_records, feature_config)
    supervised_models: dict[str, ModelEvaluation] = {}
    anomaly_model: ModelEvaluation | None = None
    modeling_notes: list[str] = []

    if config.run_modeling:
        supervised_models, anomaly_model, modeling_notes = _run_optional_modeling(
            features,
            contamination=config.anomaly_contamination,
            random_state=config.random_state,
            run_anomaly_when_unlabeled=config.run_anomaly_when_unlabeled,
        )

    markdown_summary = _build_pipeline_summary(
        record_count=len(raw_records),
        quality=quality,
        features=features,
        supervised_models=supervised_models,
        anomaly_model=anomaly_model,
        modeling_notes=modeling_notes,
    )
    return AnalysisPipelineResult(
        records=analysis_records,
        quality=quality,
        features=features,
        supervised_models=supervised_models,
        anomaly_model=anomaly_model,
        markdown_summary=markdown_summary,
        modeling_notes=tuple(modeling_notes),
    )


def run_synthetic_analysis(
    config: AnalysisPipelineConfig | None = None,
) -> AnalysisPipelineResult:
    """Run the complete analysis workflow on the built-in synthetic dataset."""
    return analyze_records(make_synthetic_dataset(), config=config)


def _run_optional_modeling(
    features: pd.DataFrame,
    *,
    contamination: float | str,
    random_state: int,
    run_anomaly_when_unlabeled: bool,
) -> tuple[dict[str, ModelEvaluation], ModelEvaluation | None, list[str]]:
    notes: list[str] = []
    labels_available = "label" in features.columns and not features["label"].isna().all()
    has_unlabeled_rows = "label" in features.columns and features["label"].isna().any()
    if labels_available:
        try:
            supervised = run_supervised_baselines(features, random_state=random_state)
            notes.append("Supervised baselines were trained because labels were available.")
            anomaly = None
            if run_anomaly_when_unlabeled and has_unlabeled_rows:
                anomaly = run_isolation_forest(
                    features,
                    contamination=contamination,
                    random_state=random_state,
                )
                notes.append(
                    "Isolation Forest anomaly scoring was also run because some rows were unlabeled."
                )
            return supervised, anomaly, notes
        except ValueError as error:
            notes.append(f"Supervised baselines skipped: {error}")

    try:
        anomaly = run_isolation_forest(
            features,
            contamination=contamination,
            random_state=random_state,
        )
        notes.append("Isolation Forest anomaly scoring was run as an exploratory baseline.")
        return {}, anomaly, notes
    except ValueError as error:
        notes.append(f"Anomaly scoring skipped: {error}")
        return {}, None, notes


def _build_pipeline_summary(
    *,
    record_count: int,
    quality: pd.DataFrame,
    features: pd.DataFrame,
    supervised_models: dict[str, ModelEvaluation],
    anomaly_model: ModelEvaluation | None,
    modeling_notes: Sequence[str],
) -> str:
    quality_issues = _quality_issue_summary(quality)
    feature_model_findings = _feature_model_summary(
        supervised_models,
        anomaly_model,
        modeling_notes,
    )
    report = MarkdownReport(
        dataset_overview=dataset_overview_from_records(
            record_count,
            features=features,
            labels=features["label"] if "label" in features.columns else None,
        ),
        signal_quality_observations=quality_issues,
        time_domain_findings="Time-domain features include mean, spread, RMS, extrema, crest factor, skewness, kurtosis, and zero-crossing rate.",
        frequency_domain_findings="Frequency-domain features include dominant frequency, spectral descriptors, and configured band energies where available.",
        time_frequency_findings="Time-frequency summaries include spectrogram energy and high-frequency transient energy indicators.",
        feature_model_findings=feature_model_findings,
        limitations="Synthetic and exploratory results are not substitutes for domain validation, calibrated thresholds, or independent test data.",
        recommended_next_steps="Inspect top anomaly candidates, compare signal plots across conditions, and rerun with real acquisition metadata when available.",
    )
    return generate_markdown_summary(report)


def _quality_issue_summary(quality: pd.DataFrame) -> str:
    if quality.empty or "issues" not in quality.columns:
        return "No quality rows were produced."
    issue_counts = (
        quality["issues"]
        .dropna()
        .astype(str)
        .str.split("; ")
        .explode()
    )
    issue_counts = issue_counts[issue_counts != ""].value_counts()
    if issue_counts.empty:
        return "No quality issues were flagged by the configured checks."
    return f"Quality checks flagged these issue counts: {issue_counts.to_dict()}."


def _feature_model_summary(
    supervised_models: dict[str, ModelEvaluation],
    anomaly_model: ModelEvaluation | None,
    modeling_notes: Sequence[str],
) -> str:
    note_text = " ".join(modeling_notes)
    if supervised_models:
        metric_parts = []
        for name, evaluation in supervised_models.items():
            metric_parts.append(
                f"{name}: accuracy={evaluation.metrics['accuracy']:.3f}, "
                f"f1_weighted={evaluation.metrics['f1_weighted']:.3f}, "
                f"split={evaluation.split_strategy}"
            )
        return f"{note_text} Supervised baseline results: {'; '.join(metric_parts)}"
    if anomaly_model is not None:
        return f"{note_text} {anomaly_summary_text(anomaly_model)}"
    return note_text or "Modeling was disabled or skipped."

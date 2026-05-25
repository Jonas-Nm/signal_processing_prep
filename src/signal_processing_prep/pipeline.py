"""High-level orchestration for the preferred signal-analysis workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from signal_processing_prep.config import FilteringConfig, ProjectConfig, load_config
from signal_processing_prep.data_loading import load_signal_dataset
from signal_processing_prep.features import (
    FeatureExtractionConfig,
    FrequencyBand,
    extract_features,
    frequency_bands_from_mapping,
)
from signal_processing_prep.modeling import (
    ModelEvaluation,
    anomaly_summary_text,
    run_isolation_forest,
    run_supervised_baselines,
)
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
    invalid_record_policy: str = "skip"

    @classmethod
    def from_project_config(
        cls,
        project_config: ProjectConfig,
        **overrides: Any,
    ) -> AnalysisPipelineConfig:
        """Create runtime analysis settings from a loaded project configuration."""
        values: dict[str, Any] = {
            "frequency_bands": frequency_bands_from_mapping(
                project_config.analysis.frequency_bands_hz
            ),
            "filtering": project_config.filtering,
        }
        values.update(overrides)
        return cls(**values)


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
    processing_notes: tuple[str, ...] = ()


def analyze_records(
    records: Sequence[SignalRecord],
    config: AnalysisPipelineConfig | None = None,
) -> AnalysisPipelineResult:
    """Run quality checks, feature extraction, optional modeling, and summary text."""
    if len(records) == 0:
        raise ValueError("At least one SignalRecord is required.")
    if config is None:
        config = AnalysisPipelineConfig()
    if config.invalid_record_policy not in {"skip", "raise"}:
        raise ValueError("invalid_record_policy must be 'skip' or 'raise'.")

    feature_config = config.features or FeatureExtractionConfig(
        frequency_bands=config.frequency_bands
    )
    raw_records = tuple(records)
    quality = assess_dataset_quality(raw_records, config.quality)
    analysis_records, processing_notes = _prepare_analysis_records(
        raw_records,
        config.filtering,
        invalid_record_policy=config.invalid_record_policy,
    )
    features, feature_notes = _extract_features_with_policy(
        analysis_records,
        feature_config,
        invalid_record_policy=config.invalid_record_policy,
    )
    processing_notes.extend(feature_notes)
    supervised_models: dict[str, ModelEvaluation] = {}
    anomaly_model: ModelEvaluation | None = None
    modeling_notes: list[str] = []

    if config.run_modeling and not features.empty:
        supervised_models, anomaly_model, modeling_notes = _run_optional_modeling(
            features,
            contamination=config.anomaly_contamination,
            random_state=config.random_state,
            run_anomaly_when_unlabeled=config.run_anomaly_when_unlabeled,
        )
    elif config.run_modeling:
        modeling_notes.append("Modeling skipped: no feature rows were available.")

    markdown_summary = _build_pipeline_summary(
        record_count=len(raw_records),
        quality=quality,
        features=features,
        supervised_models=supervised_models,
        anomaly_model=anomaly_model,
        modeling_notes=modeling_notes,
        processing_notes=processing_notes,
    )
    return AnalysisPipelineResult(
        records=analysis_records,
        quality=quality,
        features=features,
        supervised_models=supervised_models,
        anomaly_model=anomaly_model,
        markdown_summary=markdown_summary,
        modeling_notes=tuple(modeling_notes),
        processing_notes=tuple(processing_notes),
    )


def run_synthetic_analysis(
    config: AnalysisPipelineConfig | None = None,
) -> AnalysisPipelineResult:
    """Run the complete analysis workflow on the built-in synthetic dataset."""
    return analyze_records(make_synthetic_dataset(), config=config)


def analyze_dataset(
    config: ProjectConfig | str | Path,
    *,
    metadata_table: str | Path | pd.DataFrame | None = None,
    labels_by_name: Mapping[str, str] | None = None,
    split_channels: bool = True,
    pipeline_config: AnalysisPipelineConfig | None = None,
) -> AnalysisPipelineResult:
    """Load and analyze a configured dataset through one explicit workflow."""
    project_config = load_config(config) if isinstance(config, str | Path) else config
    records = load_signal_dataset(
        project_config,
        metadata_table=metadata_table,
        labels_by_name=labels_by_name,
        split_channels=split_channels,
    )
    runtime_config = pipeline_config or AnalysisPipelineConfig.from_project_config(
        project_config
    )
    return analyze_records(records, runtime_config)


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


def _prepare_analysis_records(
    records: Sequence[SignalRecord],
    filtering: FilteringConfig,
    *,
    invalid_record_policy: str,
) -> tuple[tuple[SignalRecord, ...], list[str]]:
    analysis_records: list[SignalRecord] = []
    notes: list[str] = []
    for record in records:
        try:
            analysis_records.append(apply_configured_filter(record, filtering))
        except ValueError as error:
            if invalid_record_policy == "raise":
                raise
            notes.append(f"Preprocessing skipped {record.name or '<unnamed>'}: {error}")
    return tuple(analysis_records), notes


def _extract_features_with_policy(
    records: Sequence[SignalRecord],
    feature_config: FeatureExtractionConfig,
    *,
    invalid_record_policy: str,
) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    notes: list[str] = []
    for record in records:
        try:
            frames.append(extract_features([record], feature_config))
        except ValueError as error:
            if invalid_record_policy == "raise":
                raise
            notes.append(f"Feature extraction skipped {record.name or '<unnamed>'}: {error}")
    if not frames:
        return pd.DataFrame(), notes
    return pd.concat(frames, ignore_index=True), notes


def _build_pipeline_summary(
    *,
    record_count: int,
    quality: pd.DataFrame,
    features: pd.DataFrame,
    supervised_models: dict[str, ModelEvaluation],
    anomaly_model: ModelEvaluation | None,
    modeling_notes: Sequence[str],
    processing_notes: Sequence[str],
) -> str:
    quality_issues = _quality_issue_summary(quality)
    feature_model_findings = _feature_model_summary(
        supervised_models,
        anomaly_model,
        modeling_notes,
    )
    limitations = _limitations_summary(processing_notes)
    report = MarkdownReport(
        dataset_overview=dataset_overview_from_records(
            record_count,
            features=features,
            labels=features["label"] if "label" in features.columns else None,
        ),
        signal_quality_observations=quality_issues,
        time_domain_findings=_time_domain_summary(features),
        frequency_domain_findings=_frequency_domain_summary(features),
        time_frequency_findings=_time_frequency_summary(features),
        feature_model_findings=feature_model_findings,
        limitations=limitations,
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


def _time_domain_summary(features: pd.DataFrame) -> str:
    if features.empty or "rms" not in features.columns:
        return "No time-domain feature rows were available."
    rms = pd.to_numeric(features["rms"], errors="coerce").dropna()
    if rms.empty:
        return "Time-domain features were computed, but RMS values were not finite."
    parts = [f"RMS ranged from {rms.min():.3g} to {rms.max():.3g} across feature rows."]
    if "record_name" in features.columns:
        max_index = rms.idxmax()
        parts.append(f"The largest RMS row was {features.loc[max_index, 'record_name']}.")
    label_text = _label_group_summary(features, "rms", "RMS")
    if label_text:
        parts.append(label_text)
    return " ".join(parts)


def _frequency_domain_summary(features: pd.DataFrame) -> str:
    if features.empty or "dominant_frequency_hz" not in features.columns:
        return "No frequency-domain feature rows were available."
    dominant = pd.to_numeric(features["dominant_frequency_hz"], errors="coerce").dropna()
    if dominant.empty:
        return "Frequency-domain features were computed, but dominant frequencies were not finite."
    parts = [
        f"Dominant frequency ranged from {dominant.min():.3g} Hz to {dominant.max():.3g} Hz."
    ]
    band_columns = [column for column in features.columns if column.startswith("band_energy_")]
    if band_columns:
        band_means = features[band_columns].apply(pd.to_numeric, errors="coerce").mean().dropna()
        if not band_means.empty:
            parts.append(f"Highest mean configured band energy was {band_means.idxmax()}.")
    label_text = _label_group_summary(features, "dominant_frequency_hz", "dominant frequency")
    if label_text:
        parts.append(label_text)
    return " ".join(parts)


def _time_frequency_summary(features: pd.DataFrame) -> str:
    column = "high_frequency_transient_energy"
    if features.empty or column not in features.columns:
        return "No time-frequency feature rows were available."
    transient = pd.to_numeric(features[column], errors="coerce").dropna()
    if transient.empty:
        return "Time-frequency summaries were computed, but transient-energy values were not finite."
    parts = [
        f"High-frequency transient energy ranged from {transient.min():.3g} to {transient.max():.3g}."
    ]
    if "record_name" in features.columns:
        max_index = transient.idxmax()
        parts.append(f"The largest transient-energy row was {features.loc[max_index, 'record_name']}.")
    return " ".join(parts)


def _label_group_summary(features: pd.DataFrame, feature_column: str, label: str) -> str | None:
    if "label" not in features.columns or features["label"].isna().all():
        return None
    grouped = (
        features.assign(_feature_value=pd.to_numeric(features[feature_column], errors="coerce"))
        .dropna(subset=["label", "_feature_value"])
        .groupby("label")["_feature_value"]
        .mean()
    )
    if grouped.empty:
        return None
    values = {str(index): round(float(value), 4) for index, value in grouped.items()}
    return f"Mean {label} by label: {values}."


def _limitations_summary(processing_notes: Sequence[str]) -> str:
    base = (
        "Results are exploratory and depend on dataset size, labels, acquisition assumptions, "
        "calibrated thresholds, and independent test data."
    )
    if processing_notes:
        return f"{base} Processing notes: {' '.join(processing_notes)}"
    return base

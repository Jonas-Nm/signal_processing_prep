"""Report objects and builders for exploratory signal analyses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from signal_processing_prep._modeling_common import ModelEvaluation, anomaly_summary_text
from signal_processing_prep.artifacts import FeatureTable, QualityTable


@dataclass(frozen=True)
class AnalysisReport:
    """Render-ready analysis summary owned by the reporting boundary."""

    dataset_overview: str
    acquisition_assumptions: str = "Sampling rates and sensor context are taken from typed record observations."
    signal_quality_observations: str = "No quality observations were provided."
    time_domain_findings: str = "Time-domain findings were not summarized."
    frequency_domain_findings: str = "Frequency-domain findings were not summarized."
    time_frequency_findings: str = "Time-frequency findings were not summarized."
    feature_model_findings: str = "Feature and model findings were not summarized."
    limitations: str = "Results are exploratory and depend on dataset size, labels, and acquisition assumptions."
    recommended_next_steps: str = "Validate assumptions with domain context and repeat analysis on more data."
    figures: Mapping[str, str | Path] = field(default_factory=dict)
    title: str = "Signal Analysis Summary"

    def render_markdown(self) -> str:
        """Render this report in the standard Markdown presentation format."""
        sections = [
            ("Dataset overview", self.dataset_overview),
            ("Acquisition assumptions", self.acquisition_assumptions),
            ("Signal quality observations", self.signal_quality_observations),
            ("Time-domain findings", self.time_domain_findings),
            ("Frequency-domain findings", self.frequency_domain_findings),
            ("Time-frequency findings", self.time_frequency_findings),
            ("Feature/model findings", self.feature_model_findings),
            ("Limitations", self.limitations),
            ("Recommended next steps", self.recommended_next_steps),
        ]
        lines = [f"# {self.title}", ""]
        for heading, body in sections:
            lines.extend([f"## {heading}", "", body.strip(), ""])
        if self.figures:
            lines.extend(["## Figures", ""])
            for label, path in self.figures.items():
                lines.append(f"- {label}: `{path}`")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def save(
        self,
        summaries_dir: str | Path = "reports/summaries",
        *,
        stem: str = "signal_analysis_summary",
        run_date: date | str | None = None,
    ) -> Path:
        """Render and save the report under the summaries directory."""
        date_part = (
            date.today().isoformat()
            if run_date is None
            else run_date.isoformat()
            if isinstance(run_date, date)
            else run_date
        )
        output_dir = Path(summaries_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{date_part}_{_safe_filename(stem)}.md"
        output_path.write_text(self.render_markdown(), encoding="utf-8")
        return output_path


def dataset_overview(
    record_count: int, features: FeatureTable
) -> str:
    """Build a factual dataset overview from an artifact."""
    frame = features.to_dataframe()
    parts = [
        f"Dataset contains {record_count} signal record(s).",
        f"Feature table has {frame.shape[0]} row(s) and {frame.shape[1]} column(s).",
    ]
    labels = features.labels()
    if labels is not None:
        counts = labels.dropna().astype(str).value_counts().to_dict()
        parts.append(f"Observed label counts: {counts}." if counts else "No labels were provided.")
    return " ".join(parts)


@dataclass(frozen=True)
class AnalysisReportBuilder:
    """Build standard exploratory reports from typed workflow artifacts."""

    recommended_next_steps: str = (
        "Inspect top anomaly candidates, compare signal plots across conditions, "
        "and rerun with complete acquisition context when available."
    )

    def build(
        self,
        *,
        record_count: int,
        quality: QualityTable,
        features: FeatureTable,
        supervised_models: dict[str, ModelEvaluation],
        anomaly_model: ModelEvaluation | None,
        modeling_notes: Sequence[str],
        processing_notes: Sequence[str],
    ) -> AnalysisReport:
        """Create a report object without exposing orchestration details."""
        frame = features.to_dataframe()
        return AnalysisReport(
            dataset_overview=dataset_overview(record_count, features),
            signal_quality_observations=self._quality_issue_summary(quality),
            time_domain_findings=self._time_domain_summary(frame),
            frequency_domain_findings=self._frequency_domain_summary(frame),
            time_frequency_findings=self._time_frequency_summary(frame),
            feature_model_findings=self._feature_model_summary(
                supervised_models, anomaly_model, modeling_notes
            ),
            limitations=self._limitations_summary(processing_notes),
            recommended_next_steps=self.recommended_next_steps,
        )

    @staticmethod
    def _quality_issue_summary(quality: QualityTable) -> str:
        frame = quality.to_dataframe()
        if frame.empty or "issues" not in frame.columns:
            return "No quality rows were produced."
        counts = frame["issues"].dropna().astype(str).str.split("; ").explode()
        counts = counts[counts != ""].value_counts()
        return (
            "No quality issues were flagged by the configured checks."
            if counts.empty
            else f"Quality checks flagged these issue counts: {counts.to_dict()}."
        )

    @staticmethod
    def _feature_model_summary(
        supervised_models: dict[str, ModelEvaluation],
        anomaly_model: ModelEvaluation | None,
        notes: Sequence[str],
    ) -> str:
        note_text = " ".join(notes)
        if supervised_models:
            results = "; ".join(
                f"{name}: accuracy={item.metrics['accuracy']:.3f}, "
                f"f1_weighted={item.metrics['f1_weighted']:.3f}, split={item.split_strategy}"
                for name, item in supervised_models.items()
            )
            return f"{note_text} Supervised baseline results: {results}"
        if anomaly_model is not None:
            return f"{note_text} {anomaly_summary_text(anomaly_model)}"
        return note_text or "Modeling was disabled or skipped."

    @classmethod
    def _time_domain_summary(cls, frame: pd.DataFrame) -> str:
        if frame.empty or "rms" not in frame.columns:
            return "No time-domain feature rows were available."
        values = pd.to_numeric(frame["rms"], errors="coerce").dropna()
        if values.empty:
            return "Time-domain features were computed, but RMS values were not finite."
        parts = [f"RMS ranged from {values.min():.3g} to {values.max():.3g} across feature rows."]
        if "record_name" in frame.columns:
            parts.append(f"The largest RMS row was {frame.loc[values.idxmax(), 'record_name']}.")
        group = cls._label_group_summary(frame, "rms", "RMS")
        if group:
            parts.append(group)
        return " ".join(parts)

    @classmethod
    def _frequency_domain_summary(cls, frame: pd.DataFrame) -> str:
        if frame.empty or "dominant_frequency_hz" not in frame.columns:
            return "No frequency-domain feature rows were available."
        values = pd.to_numeric(frame["dominant_frequency_hz"], errors="coerce").dropna()
        if values.empty:
            return "Frequency-domain features were computed, but dominant frequencies were not finite."
        parts = [f"Dominant frequency ranged from {values.min():.3g} Hz to {values.max():.3g} Hz."]
        bands = [column for column in frame.columns if column.startswith("band_energy_")]
        if bands:
            means = frame[bands].apply(pd.to_numeric, errors="coerce").mean().dropna()
            if not means.empty:
                parts.append(f"Highest mean configured band energy was {means.idxmax()}.")
        group = cls._label_group_summary(frame, "dominant_frequency_hz", "dominant frequency")
        if group:
            parts.append(group)
        return " ".join(parts)

    @staticmethod
    def _time_frequency_summary(frame: pd.DataFrame) -> str:
        column = "high_frequency_transient_energy"
        if frame.empty or column not in frame.columns:
            return "No time-frequency feature rows were available."
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            return "Time-frequency summaries were computed, but transient-energy values were not finite."
        summary = (
            f"High-frequency transient energy ranged from {values.min():.3g} "
            f"to {values.max():.3g}."
        )
        if "record_name" in frame.columns:
            summary += f" The largest transient-energy row was {frame.loc[values.idxmax(), 'record_name']}."
        return summary

    @staticmethod
    def _label_group_summary(frame: pd.DataFrame, feature_column: str, label: str) -> str | None:
        if "label" not in frame.columns or frame["label"].isna().all():
            return None
        grouped = (
            frame.assign(_value=pd.to_numeric(frame[feature_column], errors="coerce"))
            .dropna(subset=["label", "_value"])
            .groupby("label")["_value"]
            .mean()
        )
        if grouped.empty:
            return None
        values = {str(index): round(float(value), 4) for index, value in grouped.items()}
        return f"Mean {label} by label: {values}."

    @staticmethod
    def _limitations_summary(notes: Sequence[str]) -> str:
        base = (
            "Results are exploratory and depend on dataset size, labels, acquisition assumptions, "
            "calibrated thresholds, and independent test data."
        )
        return f"{base} Processing notes: {' '.join(notes)}" if notes else base


def _safe_filename(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    safe = safe.strip("_")
    if not safe:
        raise ValueError("stem must contain at least one filename-safe character.")
    return safe

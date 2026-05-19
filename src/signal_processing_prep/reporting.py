"""Markdown reporting helpers for signal-analysis summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class MarkdownReport:
    """Inputs for a concise signal-analysis Markdown summary."""

    dataset_overview: str
    acquisition_assumptions: str = "Sampling rates and sensor context are taken from metadata when available."
    signal_quality_observations: str = "No quality observations were provided."
    time_domain_findings: str = "Time-domain findings were not summarized."
    frequency_domain_findings: str = "Frequency-domain findings were not summarized."
    time_frequency_findings: str = "Time-frequency findings were not summarized."
    feature_model_findings: str = "Feature and model findings were not summarized."
    limitations: str = "Results are exploratory and depend on dataset size, labels, and acquisition assumptions."
    recommended_next_steps: str = "Validate assumptions with domain context and repeat analysis on more data."
    figures: Mapping[str, str | Path] = field(default_factory=dict)


def generate_markdown_summary(
    report: MarkdownReport,
    *,
    title: str = "Signal Analysis Summary",
) -> str:
    """Generate a concise Markdown summary with the standard report sections."""
    sections = [
        ("Dataset overview", report.dataset_overview),
        ("Acquisition assumptions", report.acquisition_assumptions),
        ("Signal quality observations", report.signal_quality_observations),
        ("Time-domain findings", report.time_domain_findings),
        ("Frequency-domain findings", report.frequency_domain_findings),
        ("Time-frequency findings", report.time_frequency_findings),
        ("Feature/model findings", report.feature_model_findings),
        ("Limitations", report.limitations),
        ("Recommended next steps", report.recommended_next_steps),
    ]
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", "", body.strip(), ""])
    if report.figures:
        lines.extend(["## Figures", ""])
        for label, path in report.figures.items():
            lines.append(f"- {label}: `{path}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_markdown_summary(
    markdown: str,
    summaries_dir: str | Path = "reports/summaries",
    *,
    stem: str = "signal_analysis_summary",
    run_date: date | str | None = None,
) -> Path:
    """Save Markdown text under the report summaries directory."""
    if run_date is None:
        date_part = date.today().isoformat()
    elif isinstance(run_date, date):
        date_part = run_date.isoformat()
    else:
        date_part = run_date

    output_dir = Path(summaries_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{date_part}_{_safe_filename(stem)}.md"
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def dataset_overview_from_records(
    records_count: int,
    *,
    features: pd.DataFrame | None = None,
    labels: pd.Series | None = None,
) -> str:
    """Build a factual dataset overview sentence for a report."""
    parts = [f"Dataset contains {records_count} signal record(s)."]
    if features is not None:
        parts.append(f"Feature table has {features.shape[0]} row(s) and {features.shape[1]} column(s).")
    if labels is not None:
        label_counts = labels.dropna().astype(str).value_counts().to_dict()
        parts.append(f"Observed label counts: {label_counts}." if label_counts else "No labels were provided.")
    return " ".join(parts)


def _safe_filename(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    safe = safe.strip("_")
    if not safe:
        raise ValueError("stem must contain at least one filename-safe character.")
    return safe

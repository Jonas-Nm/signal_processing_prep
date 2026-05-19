"""Tests for Markdown reporting helpers."""

import pandas as pd

from signal_processing_prep.reporting import (
    MarkdownReport,
    dataset_overview_from_records,
    generate_markdown_summary,
    save_markdown_summary,
)


def test_generate_markdown_summary_includes_standard_sections() -> None:
    """Generated summaries follow the roadmap report structure."""
    markdown = generate_markdown_summary(
        MarkdownReport(
            dataset_overview="Two synthetic records.",
            figures={"FFT": "reports/figures/2026-05-19/fft.png"},
        )
    )

    assert "# Signal Analysis Summary" in markdown
    assert "## Dataset overview" in markdown
    assert "## Signal quality observations" in markdown
    assert "## Feature/model findings" in markdown
    assert "FFT" in markdown


def test_save_markdown_summary_writes_file(tmp_path) -> None:
    """Markdown summaries are saved to reports/summaries-style paths."""
    path = save_markdown_summary(
        "# Demo\n",
        tmp_path,
        stem="Demo Report",
        run_date="2026-05-19",
    )

    assert path.exists()
    assert path.name == "2026-05-19_Demo_Report.md"
    assert path.read_text(encoding="utf-8") == "# Demo\n"


def test_dataset_overview_from_records_is_factual() -> None:
    """Dataset overview text reports counts without interpretation."""
    features = pd.DataFrame({"rms": [1.0, 2.0], "label": ["a", "b"]})

    overview = dataset_overview_from_records(
        2,
        features=features,
        labels=features["label"],
    )

    assert "2 signal record" in overview
    assert "2 row" in overview
    assert "Observed label counts" in overview

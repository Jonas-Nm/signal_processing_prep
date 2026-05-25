"""Feature, model-result, and figure-output plotting helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_feature_distribution(
    features: pd.DataFrame,
    feature: str,
    *,
    label_column: str | None = "label",
    ax: Axes | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot one feature distribution, optionally grouped by label."""
    if feature not in features.columns:
        raise ValueError(f"Feature column not found: {feature}")
    if label_column is not None and label_column not in features.columns:
        raise ValueError(f"Label column not found: {label_column}")
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    if label_column is None or features[label_column].isna().all():
        values = pd.to_numeric(features[feature], errors="coerce").dropna()
        ax.hist(values, bins=min(20, max(5, values.size)), alpha=0.75)
    else:
        for label, group in features.groupby(label_column, dropna=True):
            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            if not values.empty:
                ax.hist(values, bins=min(20, max(5, values.size)), alpha=0.55, label=str(label))
        ax.legend(loc="best")
    ax.set_title(f"{feature} distribution")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    *,
    ax: Axes | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot a labeled confusion matrix."""
    matrix = np.asarray(matrix)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("matrix shape must match the number of labels.")
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    threshold = float(np.max(matrix)) / 2.0 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > threshold else "black"
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_feature_importance(
    importances: pd.Series | Mapping[str, float],
    *,
    top_n: int = 20,
    ax: Axes | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot feature importances sorted by absolute importance."""
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    series = pd.Series(importances, dtype=float).dropna()
    if series.empty:
        raise ValueError("At least one feature importance value is required.")
    series = series.reindex(series.abs().sort_values(ascending=False).index).head(top_n).sort_values()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(series))))
    else:
        fig = ax.figure
    ax.barh(series.index.astype(str), series.values)
    ax.set_title("Feature importance")
    ax.set_xlabel("Importance")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_anomaly_scores(
    predictions: pd.DataFrame,
    *,
    score_column: str = "anomaly_score",
    name_column: str = "record_name",
    top_n: int | None = 10,
    ax: Axes | None = None,
    show: bool = False,
) -> tuple[Figure, Axes]:
    """Plot anomaly scores sorted from most to least anomalous."""
    if score_column not in predictions.columns:
        raise ValueError(f"Score column not found: {score_column}")
    if top_n is not None and top_n <= 0:
        raise ValueError("top_n must be positive.")
    ranked = predictions.sort_values(score_column, ascending=False)
    if top_n is not None:
        ranked = ranked.head(top_n)
    scores = pd.to_numeric(ranked[score_column], errors="coerce")
    if scores.isna().all():
        raise ValueError("Anomaly scores must contain at least one numeric value.")
    if name_column in ranked.columns:
        fallback = ranked["row_index"] if "row_index" in ranked.columns else ranked.index
        labels = ranked[name_column].fillna(pd.Series(fallback, index=ranked.index)).astype(str)
    elif "row_index" in ranked.columns:
        labels = ranked["row_index"].astype(str)
    else:
        labels = ranked.index.astype(str)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(ranked))))
    else:
        fig = ax.figure
    positions = np.arange(len(ranked))
    ax.barh(positions, scores.to_numpy())
    ax.set_yticks(positions, labels=labels)
    ax.invert_yaxis()
    ax.set_title("Top anomaly scores")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Record or row")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def save_figure(
    fig: Figure,
    figures_dir: str | Path = "reports/figures",
    *,
    stem: str,
    run_date: date | str | None = None,
    dpi: int = 150,
) -> Path:
    """Save a figure under a date-stamped reports/figures directory."""
    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    if run_date is None:
        date_part = date.today().isoformat()
    elif isinstance(run_date, date):
        date_part = run_date.isoformat()
    else:
        date_part = run_date
    output_dir = Path(figures_dir) / date_part
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_safe_filename(stem)}.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


def _safe_filename(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    safe = safe.strip("_")
    if not safe:
        raise ValueError("stem must contain at least one filename-safe character.")
    return safe

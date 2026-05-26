"""Candidate-region summarization utilities for exploratory anomaly scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from signal_processing_prep._modeling_common import ModelEvaluation
from signal_processing_prep.records import AnnotationInterval


REGION_IDENTITY_COLUMNS = (
    "record_name",
    "source_name",
    "source_path",
    "channel_name",
    "channel_index",
)

REGION_COLUMNS = (
    "region_rank",
    *REGION_IDENTITY_COLUMNS,
    "region_start_seconds",
    "region_end_seconds",
    "duration_seconds",
    "window_count",
    "peak_anomaly_score",
    "best_window_rank",
    "peak_window_center_seconds",
)

CONSENSUS_COLUMNS = (
    "region_rank",
    *REGION_IDENTITY_COLUMNS,
    "region_start_seconds",
    "region_end_seconds",
    "duration_seconds",
    "method_count",
    "methods",
    "contributing_region_count",
    "best_window_rank",
    "mean_best_method_rank",
)


def group_candidate_regions(
    evaluation: ModelEvaluation,
    *,
    top_n: int = 20,
    merge_gap_seconds: float = 0.0,
) -> pd.DataFrame:
    """Merge top-scoring overlapping windows into inspection-oriented regions."""
    _validate_region_parameters(top_n=top_n, merge_gap_seconds=merge_gap_seconds)
    if evaluation.task_type != "unsupervised_anomaly_score":
        raise ValueError("group_candidate_regions requires an anomaly-score evaluation.")

    windows = _validated_windows(evaluation.prediction_frame)
    selected = (
        windows.sort_values(
            ["anomaly_score", "window_start_seconds"],
            ascending=[False, True],
            kind="stable",
        )
        .head(top_n)
        .sort_values(["_region_identity", "window_start_seconds", "window_end_seconds"], kind="stable")
    )
    grouped = _merge_windows(selected, merge_gap_seconds=merge_gap_seconds)
    if not grouped:
        return pd.DataFrame(columns=REGION_COLUMNS)

    regions = pd.DataFrame(grouped).sort_values(
        ["peak_anomaly_score", "region_start_seconds"],
        ascending=[False, True],
        kind="stable",
    )
    regions.insert(0, "region_rank", np.arange(1, len(regions) + 1, dtype=int))
    return regions.loc[:, list(REGION_COLUMNS)].reset_index(drop=True)


def consensus_candidate_regions(
    evaluations: Mapping[str, ModelEvaluation],
    *,
    top_n_per_method: int = 20,
    merge_gap_seconds: float = 0.0,
) -> pd.DataFrame:
    """Merge candidate regions across methods and summarize method agreement."""
    _validate_region_parameters(
        top_n=top_n_per_method,
        merge_gap_seconds=merge_gap_seconds,
    )
    if not evaluations:
        raise ValueError("At least one anomaly-score evaluation is required.")

    rows: list[dict[str, object]] = []
    for method, evaluation in evaluations.items():
        regions = group_candidate_regions(
            evaluation,
            top_n=top_n_per_method,
            merge_gap_seconds=merge_gap_seconds,
        )
        for row in regions.itertuples(index=False):
            rows.append(
                {
                    "method": str(method),
                    **{
                        column: getattr(row, column)
                        for column in REGION_IDENTITY_COLUMNS
                    },
                    "_region_identity": _identity_key(row),
                    "region_start_seconds": float(row.region_start_seconds),
                    "region_end_seconds": float(row.region_end_seconds),
                    "best_window_rank": int(row.best_window_rank),
                }
            )

    if not rows:
        return pd.DataFrame(columns=CONSENSUS_COLUMNS)

    ordered = pd.DataFrame(rows).sort_values(
        ["_region_identity", "region_start_seconds", "region_end_seconds"],
        kind="stable",
    )
    merged: list[dict[str, object]] = []
    contributors: list[dict[str, object]] = []
    current_end: float | None = None
    current_start: float | None = None
    current_identity: tuple[object, ...] | None = None
    for row in ordered.to_dict(orient="records"):
        start = float(row["region_start_seconds"])
        end = float(row["region_end_seconds"])
        identity = row["_region_identity"]
        if (
            current_end is None
            or identity != current_identity
            or start > current_end + merge_gap_seconds
        ):
            if contributors:
                merged.append(_consensus_region(current_start, current_end, contributors))
            current_start = start
            current_end = end
            current_identity = identity
            contributors = [row]
        else:
            current_end = max(current_end, end)
            contributors.append(row)
    if contributors:
        merged.append(_consensus_region(current_start, current_end, contributors))

    regions = pd.DataFrame(merged).sort_values(
        ["method_count", "mean_best_method_rank", "region_start_seconds"],
        ascending=[False, True, True],
        kind="stable",
    )
    regions.insert(0, "region_rank", np.arange(1, len(regions) + 1, dtype=int))
    return regions.loc[:, list(CONSENSUS_COLUMNS)].reset_index(drop=True)


def evaluate_candidate_region_overlap(
    regions: pd.DataFrame,
    known_intervals: Sequence[AnnotationInterval],
) -> pd.DataFrame:
    """Compare candidate regions with withheld known intervals for synthetic demos."""
    _validate_region_frame(regions)
    output = regions.copy(deep=True)
    if len(_identified_region_keys(output)) > 1:
        raise ValueError(
            "known_intervals are not record-scoped; evaluate candidate regions "
            "from one record at a time."
        )
    overlap_rows: list[dict[str, object]] = []
    for row in output.itertuples(index=False):
        start = float(getattr(row, "region_start_seconds"))
        end = float(getattr(row, "region_end_seconds"))
        best_interval: AnnotationInterval | None = None
        best_overlap = 0.0
        for interval in known_intervals:
            overlap = max(
                0.0,
                min(end, interval.end_seconds) - max(start, interval.start_seconds),
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_interval = interval
        duration = end - start
        interval_duration = (
            best_interval.end_seconds - best_interval.start_seconds
            if best_interval is not None
            else None
        )
        overlap_rows.append(
            {
                "overlap_seconds": best_overlap,
                "candidate_overlap_fraction": best_overlap / duration,
                "known_interval_capture_fraction": (
                    best_overlap / interval_duration if interval_duration is not None else 0.0
                ),
                "overlaps_known_interval": best_overlap > 0.0,
                "matched_interval_kind": (
                    best_interval.kind if best_interval is not None else None
                ),
            }
        )
    overlap_frame = pd.DataFrame(
        overlap_rows,
        columns=[
            "overlap_seconds",
            "candidate_overlap_fraction",
            "known_interval_capture_fraction",
            "overlaps_known_interval",
            "matched_interval_kind",
        ],
    )
    return pd.concat([output.reset_index(drop=True), overlap_frame], axis=1)


def _validated_windows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "window_start_seconds",
        "window_end_seconds",
        "window_center_seconds",
        "anomaly_score",
        "anomaly_rank",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Anomaly predictions require window timing columns: {missing}.")
    windows = frame.copy(deep=True)
    for column in required:
        windows[column] = pd.to_numeric(windows[column], errors="coerce")
        if not np.isfinite(windows[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Candidate-region column '{column}' must be finite numeric data.")
    if (windows["window_end_seconds"] <= windows["window_start_seconds"]).any():
        raise ValueError("Window end times must exceed start times.")
    for column in REGION_IDENTITY_COLUMNS:
        if column not in windows.columns:
            windows[column] = None
    windows["_region_identity"] = [_identity_key(row) for row in windows.itertuples(index=False)]
    return windows


def _merge_windows(
    windows: pd.DataFrame,
    *,
    merge_gap_seconds: float,
) -> list[dict[str, float | int]]:
    grouped: list[dict[str, float | int]] = []
    rows: list[pd.Series] = []
    current_end: float | None = None
    current_identity: tuple[object, ...] | None = None
    for _, row in windows.iterrows():
        start = float(row["window_start_seconds"])
        end = float(row["window_end_seconds"])
        identity = row["_region_identity"]
        if (
            current_end is None
            or identity != current_identity
            or start > current_end + merge_gap_seconds
        ):
            if rows:
                grouped.append(_candidate_region(rows))
            rows = [row]
            current_end = end
            current_identity = identity
        else:
            rows.append(row)
            current_end = max(current_end, end)
    if rows:
        grouped.append(_candidate_region(rows))
    return grouped


def _candidate_region(rows: list[pd.Series]) -> dict[str, float | int]:
    ordered_by_peak = sorted(
        rows,
        key=lambda row: (-float(row["anomaly_score"]), float(row["window_start_seconds"])),
    )
    peak = ordered_by_peak[0]
    start = min(float(row["window_start_seconds"]) for row in rows)
    end = max(float(row["window_end_seconds"]) for row in rows)
    return {
        **{column: peak[column] for column in REGION_IDENTITY_COLUMNS},
        "region_start_seconds": start,
        "region_end_seconds": end,
        "duration_seconds": end - start,
        "window_count": len(rows),
        "peak_anomaly_score": float(peak["anomaly_score"]),
        "best_window_rank": int(min(float(row["anomaly_rank"]) for row in rows)),
        "peak_window_center_seconds": float(peak["window_center_seconds"]),
    }


def _consensus_region(
    start: float | None,
    end: float | None,
    contributors: list[dict[str, object]],
) -> dict[str, object]:
    if start is None or end is None:
        raise ValueError("Consensus regions require valid start and end times.")
    best_ranks_by_method: dict[str, int] = {}
    for contributor in contributors:
        method = str(contributor["method"])
        rank = int(contributor["best_window_rank"])
        best_ranks_by_method[method] = min(best_ranks_by_method.get(method, rank), rank)
    methods = tuple(sorted(best_ranks_by_method))
    ranks = list(best_ranks_by_method.values())
    return {
        **{
            column: contributors[0][column]
            for column in REGION_IDENTITY_COLUMNS
        },
        "region_start_seconds": start,
        "region_end_seconds": end,
        "duration_seconds": end - start,
        "method_count": len(methods),
        "methods": methods,
        "contributing_region_count": len(contributors),
        "best_window_rank": min(ranks),
        "mean_best_method_rank": float(np.mean(ranks)),
    }


def _identity_key(row: object) -> tuple[object, ...]:
    values: list[str] = []
    for column in REGION_IDENTITY_COLUMNS:
        value = getattr(row, column) if hasattr(row, column) else row[column]  # type: ignore[index]
        values.append("" if pd.isna(value) else str(value))
    return tuple(values)


def _validate_region_parameters(*, top_n: int, merge_gap_seconds: float) -> None:
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    if not np.isfinite(merge_gap_seconds) or merge_gap_seconds < 0:
        raise ValueError("merge_gap_seconds must be a finite non-negative value.")


def _identified_region_keys(regions: pd.DataFrame) -> set[tuple[str, ...]]:
    identity_columns = [
        column for column in REGION_IDENTITY_COLUMNS if column in regions.columns
    ]
    if not identity_columns:
        return set()
    keys: set[tuple[str, ...]] = set()
    for values in regions.loc[:, identity_columns].itertuples(index=False, name=None):
        key = tuple("" if pd.isna(value) else str(value) for value in values)
        if any(key):
            keys.add(key)
    return keys


def _validate_region_frame(regions: pd.DataFrame) -> None:
    required = {"region_start_seconds", "region_end_seconds"}
    missing = sorted(required - set(regions.columns))
    if missing:
        raise ValueError(f"Candidate region columns not found: {missing}.")
    for column in required:
        values = pd.to_numeric(regions[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Candidate-region column '{column}' must be finite numeric data.")
    if (
        pd.to_numeric(regions["region_end_seconds"])
        <= pd.to_numeric(regions["region_start_seconds"])
    ).any():
        raise ValueError("Region end times must exceed start times.")

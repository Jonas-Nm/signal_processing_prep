"""Validated analysis artifacts that isolate pandas from core workflow policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


FEATURE_METADATA_COLUMNS = (
    "record_name",
    "source_name",
    "source_path",
    "source_format",
    "signal_column",
    "original_dtype",
    "channel_name",
    "channel_index",
    "label",
    "split",
    "sensor",
    "is_anomalous",
    "anomaly_kind",
    "anomaly_start_seconds",
    "anomaly_end_seconds",
    "known_anomaly_overlap",
    "is_synthetic_anomaly_record",
    "sampling_rate_hz",
    "n_samples",
    "duration_seconds",
    "frequency_window",
    "window_start_seconds",
    "window_end_seconds",
    "window_center_seconds",
    "window_n_samples",
    "_row_index",
)


@dataclass(frozen=True)
class FeatureSchema:
    """Column roles owned by a feature artifact."""

    feature_columns: tuple[str, ...]
    label_column: str = "label"
    group_candidates: tuple[str, ...] = ("source_name", "record_name")
    metadata_columns: tuple[str, ...] = FEATURE_METADATA_COLUMNS


class FeatureTable:
    """Validated feature rows with owned feature and identity roles."""

    def __init__(self, frame: pd.DataFrame, schema: FeatureSchema) -> None:
        missing = [column for column in schema.feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Feature columns not found in table: {missing}.")
        reserved = sorted(set(schema.feature_columns) & set(schema.metadata_columns))
        if reserved:
            raise ValueError(
                f"Feature columns must not contain reserved context columns: {reserved}."
            )
        self._frame = frame.copy(deep=True).reset_index(drop=True)
        self.schema = schema

    @classmethod
    def from_dataframe(
        cls,
        frame: pd.DataFrame,
        *,
        feature_columns: tuple[str, ...] | list[str] | None = None,
    ) -> FeatureTable:
        """Construct a feature artifact from calculated and presentation rows."""
        if feature_columns is None:
            raise ValueError(
                "feature_columns must be explicitly declared for a FeatureTable."
            )
        return cls(frame, FeatureSchema(tuple(feature_columns)))

    @classmethod
    def concat(cls, tables: list[FeatureTable]) -> FeatureTable:
        """Combine compatible feature artifacts into one artifact."""
        if not tables:
            return cls.from_dataframe(pd.DataFrame(), feature_columns=())
        schema = tables[0].schema
        if any(table.schema != schema for table in tables[1:]):
            raise ValueError("Cannot combine feature tables having different schemas.")
        return cls(pd.concat([table._frame for table in tables], ignore_index=True), schema)

    def __len__(self) -> int:
        """Return feature row count."""
        return len(self._frame)

    @property
    def empty(self) -> bool:
        """Return whether the artifact has no feature rows."""
        return self._frame.empty

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return declared calculated feature columns."""
        return self.schema.feature_columns

    def to_dataframe(self) -> pd.DataFrame:
        """Return an independent DataFrame view for inspection and plotting."""
        return self._frame.copy(deep=True)

    def numeric_matrix(
        self,
        *,
        feature_columns: tuple[str, ...] | list[str] | None = None,
    ) -> tuple[pd.DataFrame, tuple[str, ...]]:
        """Select usable feature values and retain source row positions."""
        selected = tuple(feature_columns) if feature_columns is not None else self.feature_columns
        missing = [column for column in selected if column not in self._frame.columns]
        if missing:
            raise ValueError(f"Requested feature column(s) not found: {missing}.")
        numeric = (
            self._frame.loc[:, list(selected)]
            .select_dtypes(include=[np.number])
            .replace([np.inf, -np.inf], np.nan)
            .dropna(axis=1, how="all")
        )
        valid_rows = numeric.notna().any(axis=1) if not numeric.empty else pd.Series(False, index=numeric.index)
        source_positions = np.flatnonzero(valid_rows.to_numpy()).tolist()
        numeric = numeric.loc[valid_rows]
        numeric.attrs["source_positions"] = source_positions
        if numeric.empty:
            return numeric, ()
        return numeric.fillna(numeric.median(numeric_only=True)), tuple(str(c) for c in numeric.columns)

    def labels(self) -> pd.Series | None:
        """Return label values when represented."""
        column = self.schema.label_column
        return self._frame[column].copy() if column in self._frame.columns else None

    def resolve_group_column(self, requested: str | None = None) -> str | None:
        """Resolve source grouping used to prevent split leakage."""
        if requested is not None:
            if requested not in self._frame.columns:
                raise ValueError(f"Group column not found: {requested}")
            return requested
        for candidate in self.schema.group_candidates:
            if candidate in self._frame.columns and not self._frame[candidate].isna().all():
                return candidate
        return None


class QualityTable:
    """Validated per-record quality observations."""

    def __init__(self, frame: pd.DataFrame) -> None:
        required = {"record_name", "issues"}
        missing = required - set(frame.columns)
        if missing and not frame.empty:
            raise ValueError(f"Quality columns not found: {sorted(missing)}.")
        self._frame = frame.copy(deep=True)

    def to_dataframe(self) -> pd.DataFrame:
        """Return an independent DataFrame view."""
        return self._frame.copy(deep=True)

    @property
    def empty(self) -> bool:
        """Return whether no quality rows exist."""
        return self._frame.empty


class PredictionTable:
    """Validated prediction or anomaly-score rows."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame.copy(deep=True)

    @classmethod
    def anomaly_scores(cls, frame: pd.DataFrame) -> PredictionTable:
        """Validate ranked anomaly-score rows."""
        required = {"row_index", "anomaly_score", "anomaly_rank"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Anomaly prediction columns not found: {sorted(missing)}.")
        _validate_row_index(frame)
        scores = pd.to_numeric(frame["anomaly_score"], errors="coerce")
        if not np.isfinite(scores.to_numpy(dtype=float)).all():
            raise ValueError("Anomaly scores must be finite numeric values.")
        ranks = pd.to_numeric(frame["anomaly_rank"], errors="coerce")
        expected = scores.rank(method="first", ascending=False).astype(int)
        if not np.array_equal(ranks.to_numpy(), expected.to_numpy()):
            raise ValueError("Anomaly ranks must order anomaly scores from highest to lowest.")
        return cls(frame)

    @classmethod
    def classification(cls, frame: pd.DataFrame) -> PredictionTable:
        """Validate labeled classification prediction rows."""
        required = {"row_index", "true_label", "predicted_label"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Classification prediction columns not found: {sorted(missing)}.")
        _validate_row_index(frame)
        return cls(frame)

    def to_dataframe(self) -> pd.DataFrame:
        """Return an independent DataFrame view."""
        return self._frame.copy(deep=True)

    def validate_alignment(self, features: FeatureTable) -> None:
        """Validate represented prediction rows against source feature identities."""
        source = features.to_dataframe()
        indices = self._frame["row_index"].astype(int).to_numpy()
        if np.any(indices < 0) or np.any(indices >= len(source)):
            raise ValueError("Prediction row_index values are outside the feature table.")
        expected_rows = source.iloc[indices].reset_index(drop=True)
        for column in FEATURE_METADATA_COLUMNS:
            if column not in self._frame.columns or column not in expected_rows.columns:
                continue
            actual = self._frame[column].reset_index(drop=True)
            expected = expected_rows[column].reset_index(drop=True)
            if not actual.equals(expected):
                raise ValueError(
                    f"Prediction column '{column}' is not aligned with its feature row_index."
                )


def _validate_row_index(frame: pd.DataFrame) -> None:
    values = pd.to_numeric(frame["row_index"], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Prediction row_index values must be finite integers.")
    if not np.equal(values.to_numpy(), np.floor(values.to_numpy())).all():
        raise ValueError("Prediction row_index values must be finite integers.")
    if values.duplicated().any():
        raise ValueError("Prediction row_index values must be unique.")

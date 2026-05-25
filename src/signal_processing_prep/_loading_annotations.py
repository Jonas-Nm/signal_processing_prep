"""Internal metadata-table joining into canonical record annotations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from signal_processing_prep.errors import ConfigurationError, RecordDataError
from signal_processing_prep.records import (
    AnnotationInterval,
    RecordAnnotations,
    SignalRecord,
)

_KEY_COLUMNS = ("record_name", "name", "file_name", "filename", "stem", "source_path")
_TYPED_COLUMNS = {
    *_KEY_COLUMNS,
    "label",
    "condition",
    "class",
    "target",
    "split",
    "sensor",
    "is_anomalous",
    "anomaly_kind",
    "anomaly_start_seconds",
    "anomaly_end_seconds",
}


@dataclass(frozen=True)
class DatasetAnnotationJoiner:
    """Join tabular labels and annotations without leaking raw metadata into records."""

    lookup: Mapping[str, Mapping[str, Any]]
    labels_by_name: Mapping[str, str]

    @classmethod
    def from_inputs(
        cls,
        metadata_table: str | Path | pd.DataFrame | None,
        labels_by_name: Mapping[str, str] | None,
    ) -> DatasetAnnotationJoiner:
        """Build a joiner from optional metadata and explicit label mapping."""
        return cls(_metadata_lookup(metadata_table), dict(labels_by_name or {}))

    def annotate(self, record: SignalRecord, path: Path) -> SignalRecord:
        """Return a record with canonical typed labels and annotations."""
        keys = tuple(
            dict.fromkeys(
                key
                for key in (record.name, path.name, path.stem, str(path))
                if key is not None
            )
        )
        row = next((self.lookup[key] for key in keys if key in self.lookup), {})
        mapped_label = next((self.labels_by_name[key] for key in keys if key in self.labels_by_name), None)
        label = record.label or mapped_label or _external_label(row)
        if not row and label == record.label:
            return record
        return record.derive(record.values, label=label, annotations=_annotations(record, row))


def _metadata_lookup(metadata_table: str | Path | pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    if metadata_table is None:
        return {}
    frame = pd.read_csv(metadata_table) if isinstance(metadata_table, str | Path) else metadata_table.copy()
    key_columns = [column for column in _KEY_COLUMNS if column in frame.columns]
    if not key_columns:
        raise ConfigurationError(
            "metadata_table must contain at least one key column: " + ", ".join(_KEY_COLUMNS)
        )
    lookup: dict[str, dict[str, Any]] = {}
    for _, item in frame.iterrows():
        row = {str(key): value for key, value in item.to_dict().items() if not pd.isna(value)}
        for column in key_columns:
            value = item.get(column)
            if pd.isna(value):
                continue
            key = str(value)
            lookup[key] = row
            if column == "source_path":
                lookup[Path(key).name] = row
                lookup[Path(key).stem] = row
    return lookup


def _external_label(row: Mapping[str, Any]) -> str | None:
    for column in ("label", "condition", "class", "target"):
        if column in row and not pd.isna(row[column]):
            return str(row[column])
    return None


def _annotations(record: SignalRecord, row: Mapping[str, Any]) -> RecordAnnotations:
    if not row:
        return record.annotations
    split = _optional_text(row.get("split")) or record.annotations.split
    sensor = _optional_text(row.get("sensor")) or record.annotations.sensor
    kind = _optional_text(row.get("anomaly_kind")) or record.annotations.anomaly_kind
    anomalous = (
        _as_bool(row["is_anomalous"])
        if "is_anomalous" in row and not pd.isna(row["is_anomalous"])
        else record.annotations.is_anomalous
    )
    interval = _interval(row)
    extras = {key: value for key, value in row.items() if key not in _TYPED_COLUMNS}
    return RecordAnnotations(
        split=split,
        sensor=sensor,
        is_anomalous=anomalous,
        anomaly_kind=kind,
        intervals=(interval,) if interval is not None else record.annotations.intervals,
        attributes={**dict(record.annotations.attributes), **extras},
    )


def _interval(row: Mapping[str, Any]) -> AnnotationInterval | None:
    start = row.get("anomaly_start_seconds")
    end = row.get("anomaly_end_seconds")
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return None
    if float(start) < 0.0 or float(end) <= float(start):
        raise RecordDataError("Annotation interval end must exceed start.")
    try:
        return AnnotationInterval(
            start_seconds=float(start),
            end_seconds=float(end),
            kind=_optional_text(row.get("anomaly_kind")),
        )
    except ValueError as error:
        raise RecordDataError("Annotation interval bounds are invalid.") from error


def _optional_text(value: object) -> str | None:
    return None if value is None or pd.isna(value) else str(value)


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)

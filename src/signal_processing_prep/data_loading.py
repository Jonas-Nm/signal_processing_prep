"""Load common signal file formats into `SignalRecord` objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.io import wavfile

from signal_processing_prep.config import ProjectConfig, load_config
from signal_processing_prep.records import (
    AcquisitionDiagnostics,
    SignalProvenance,
    SignalRecord,
)

_COMMON_TIME_COLUMNS = {"time", "timestamp", "t", "seconds", "time_s", "time_seconds"}
_METADATA_KEY_COLUMNS = ("record_name", "name", "file_name", "filename", "stem", "source_path")


def load_signal_file(
    path: str | Path,
    *,
    sampling_rate_hz: float | None = None,
    label: str | None = None,
    signal_column: str | None = None,
    label_column: str | None = None,
    channel: int | None = None,
) -> SignalRecord:
    """Load one supported signal file into a `SignalRecord`."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        values, metadata, detected_label, inferred_sampling_rate = _load_csv(
            file_path, signal_column=signal_column, label_column=label_column
        )
        record_sampling_rate = _resolve_sampling_rate(
            sampling_rate_hz,
            inferred_sampling_rate,
            file_path,
        )
        metadata.update(
            _sampling_rate_metadata(
                provided_sampling_rate_hz=sampling_rate_hz,
                inferred_sampling_rate_hz=inferred_sampling_rate,
            )
        )
        acquisition = _acquisition_diagnostics(metadata)
    elif suffix == ".txt":
        values = _load_text(file_path, channel=channel)
        metadata = _channel_metadata(channel)
        detected_label = None
        record_sampling_rate = _require_sampling_rate(sampling_rate_hz, file_path)
        acquisition = AcquisitionDiagnostics(
            sampling_rate_source="provided",
            provided_sampling_rate_hz=record_sampling_rate,
        )
    elif suffix == ".npy":
        values = _load_npy(file_path, channel=channel)
        metadata = _channel_metadata(channel)
        detected_label = None
        record_sampling_rate = _require_sampling_rate(sampling_rate_hz, file_path)
        acquisition = AcquisitionDiagnostics(
            sampling_rate_source="provided",
            provided_sampling_rate_hz=record_sampling_rate,
        )
    elif suffix == ".wav":
        record_sampling_rate, values, metadata = _load_wav(file_path, channel=channel)
        detected_label = None
        if sampling_rate_hz is not None and float(sampling_rate_hz) != record_sampling_rate:
            raise ValueError(
                "Provided sampling_rate_hz does not match the WAV file sampling rate."
            )
        acquisition = AcquisitionDiagnostics(sampling_rate_source="wav_header")
    else:
        raise ValueError(f"Unsupported signal file extension: {suffix}")

    return SignalRecord(
        values=values,
        sampling_rate_hz=record_sampling_rate,
        label=label if label is not None else detected_label,
        name=file_path.stem,
        metadata={"source_path": str(file_path), **metadata},
        provenance=SignalProvenance(
            source_name=file_path.stem,
            source_path=str(file_path),
            channel_index=channel,
        ),
        acquisition=acquisition,
    )


def load_signal_file_channels(
    path: str | Path,
    *,
    sampling_rate_hz: float | None = None,
    label: str | None = None,
    signal_columns: Sequence[str] | None = None,
    label_column: str | None = None,
) -> list[SignalRecord]:
    """Load all channels or signal columns from one supported file.

    One-dimensional files return a single record. Multi-channel WAV, NPY, TXT,
    and multi-signal CSV files are split into one `SignalRecord` per channel so
    downstream analysis can remain explicit and single-channel.
    """
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        dataframe = pd.read_csv(file_path)
        columns = (
            list(signal_columns)
            if signal_columns is not None
            else _signal_columns(dataframe, label_column=label_column)
        )
        if len(columns) == 0:
            raise ValueError("CSV file does not contain numeric signal columns.")
        records = [
            load_signal_file(
                file_path,
                sampling_rate_hz=sampling_rate_hz,
                label=label,
                signal_column=column,
                label_column=label_column,
            )
            for column in columns
        ]
        return [_rename_channel_record(record, column, index) for index, (record, column) in enumerate(zip(records, columns))]

    if suffix == ".wav":
        raw_sampling_rate, raw_values = wavfile.read(file_path)
        if raw_values.ndim == 1:
            return [load_signal_file(file_path, sampling_rate_hz=sampling_rate_hz, label=label)]
        records = [
            load_signal_file(
                file_path,
                sampling_rate_hz=sampling_rate_hz or float(raw_sampling_rate),
                label=label,
                channel=channel_index,
            )
            for channel_index in range(raw_values.shape[1])
        ]
        return [
            _rename_channel_record(record, str(channel_index), channel_index)
            for channel_index, record in enumerate(records)
        ]

    if suffix in {".txt", ".npy"}:
        values = np.loadtxt(file_path, dtype=np.float64) if suffix == ".txt" else np.load(file_path)
        array = np.asarray(values)
        squeezed = np.squeeze(array)
        if squeezed.ndim == 1:
            return [load_signal_file(file_path, sampling_rate_hz=sampling_rate_hz, label=label)]
        if squeezed.ndim != 2:
            raise ValueError(f"Signal file must contain one- or two-dimensional data: {file_path}")
        records = [
            load_signal_file(
                file_path,
                sampling_rate_hz=sampling_rate_hz,
                label=label,
                channel=channel_index,
            )
            for channel_index in range(squeezed.shape[1])
        ]
        return [
            _rename_channel_record(record, str(channel_index), channel_index)
            for channel_index, record in enumerate(records)
        ]

    raise ValueError(f"Unsupported signal file extension: {suffix}")


def load_signal_dataset(
    config: ProjectConfig | str | Path,
    *,
    metadata_table: str | Path | pd.DataFrame | None = None,
    labels_by_name: Mapping[str, str] | None = None,
    split_channels: bool = True,
) -> list[SignalRecord]:
    """Load all files matching the configured patterns into signal records.

    Labels can come from file-internal label columns, a metadata table, or an
    explicit name-to-label mapping. Metadata-table rows are matched by common
    key columns such as ``record_name``, ``file_name``, ``filename``, ``stem``,
    or ``source_path``.
    """
    project_config = load_config(config) if isinstance(config, str | Path) else config
    data_dir = project_config.paths.data_dir
    paths = _matching_paths(data_dir, project_config.loading.file_patterns)
    metadata_lookup = _metadata_lookup(metadata_table)
    loaded_records: list[SignalRecord] = []

    for path in paths:
        if split_channels:
            records = load_signal_file_channels(
                path,
                sampling_rate_hz=project_config.loading.sampling_rate_hz,
                signal_columns=(
                    [project_config.loading.signal_column]
                    if project_config.loading.signal_column is not None
                    else None
                ),
                label_column=project_config.loading.label_column,
            )
        else:
            records = [
                load_signal_file(
                    path,
                    sampling_rate_hz=project_config.loading.sampling_rate_hz,
                    signal_column=project_config.loading.signal_column,
                    label_column=project_config.loading.label_column,
                )
            ]
        for record in records:
            loaded_records.append(
                _apply_external_metadata(
                    record,
                    path,
                    metadata_lookup=metadata_lookup,
                    labels_by_name=labels_by_name or {},
                )
            )

    return loaded_records


def _load_csv(
    path: Path,
    *,
    signal_column: str | None,
    label_column: str | None,
) -> tuple[NDArray[np.float64], dict[str, Any], str | None, float | None]:
    dataframe = pd.read_csv(path)
    if dataframe.empty:
        raise ValueError(f"CSV file contains no rows: {path}")

    selected_signal_column = signal_column or _first_numeric_column(
        dataframe,
        label_column=label_column,
    )
    if selected_signal_column not in dataframe.columns:
        raise ValueError(f"Signal column not found in CSV: {selected_signal_column}")

    values = dataframe[selected_signal_column].to_numpy(dtype=np.float64)
    detected_label = _single_label(dataframe, label_column)
    time_metadata = _time_axis_metadata(dataframe)
    metadata = {
        "format": "csv",
        "signal_column": selected_signal_column,
        "columns": list(dataframe.columns),
        **time_metadata,
    }
    if label_column is not None:
        metadata["label_column"] = label_column
    return values, metadata, detected_label, _optional_metadata_float(time_metadata, "inferred_sampling_rate_hz")


def _load_text(path: Path, *, channel: int | None = None) -> NDArray[np.float64]:
    values = np.loadtxt(path, dtype=np.float64)
    return _as_1d_or_channel(values, source=path, channel=channel)


def _load_npy(path: Path, *, channel: int | None = None) -> NDArray[np.float64]:
    values = np.load(path)
    return _as_1d_or_channel(values, source=path, channel=channel)


def _load_wav(path: Path, *, channel: int | None = None) -> tuple[float, NDArray[np.float64], dict[str, Any]]:
    sampling_rate_hz, values = wavfile.read(path)
    metadata = {"format": "wav", "original_dtype": str(values.dtype)}
    if values.ndim == 1:
        if channel not in {None, 0}:
            raise ValueError("Mono WAV files only contain channel 0.")
        return float(sampling_rate_hz), _normalize_wav_values(values), metadata
    if values.ndim != 2:
        raise ValueError("WAV data must be one- or two-dimensional.")
    if channel is None:
        raise ValueError("Multi-channel WAV files require channel or load_signal_file_channels.")
    selected = _select_channel(values, channel, source=path)
    return (
        float(sampling_rate_hz),
        _normalize_wav_values(selected),
        {**metadata, "channel_index": channel, "n_channels": int(values.shape[1])},
    )


def _first_numeric_column(dataframe: pd.DataFrame, *, label_column: str | None = None) -> str:
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    if numeric_columns.empty:
        raise ValueError("CSV file does not contain a numeric signal column.")

    excluded_columns = set(_COMMON_TIME_COLUMNS)
    if label_column is not None:
        excluded_columns.add(label_column.lower())
    signal_candidates = [
        column for column in numeric_columns if str(column).lower() not in excluded_columns
    ]
    if len(signal_candidates) == 1:
        return str(signal_candidates[0])
    if len(signal_candidates) > 1:
        raise ValueError(
            "CSV file contains multiple numeric signal candidates; provide signal_column."
        )
    raise ValueError("CSV file contains only time-like numeric columns; provide signal_column.")


def _signal_columns(dataframe: pd.DataFrame, *, label_column: str | None = None) -> list[str]:
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    excluded_columns = set(_COMMON_TIME_COLUMNS)
    if label_column is not None:
        excluded_columns.add(label_column.lower())
    return [str(column) for column in numeric_columns if str(column).lower() not in excluded_columns]


def _single_label(dataframe: pd.DataFrame, label_column: str | None) -> str | None:
    if label_column is None:
        return None
    if label_column not in dataframe.columns:
        raise ValueError(f"Label column not found in CSV: {label_column}")

    labels = dataframe[label_column].dropna().unique()
    if len(labels) == 0:
        return None
    if len(labels) > 1:
        raise ValueError("A single SignalRecord cannot represent multiple CSV labels.")
    return str(labels[0])


def _require_sampling_rate(sampling_rate_hz: float | None, path: Path) -> float:
    if sampling_rate_hz is None:
        raise ValueError(f"sampling_rate_hz is required for {path.suffix} files.")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")
    return float(sampling_rate_hz)


def _resolve_sampling_rate(
    provided_sampling_rate_hz: float | None,
    inferred_sampling_rate_hz: float | None,
    path: Path,
) -> float:
    if provided_sampling_rate_hz is not None:
        return _require_sampling_rate(provided_sampling_rate_hz, path)
    if inferred_sampling_rate_hz is None:
        raise ValueError(f"sampling_rate_hz is required for {path.suffix} files without a usable time column.")
    if inferred_sampling_rate_hz <= 0:
        raise ValueError("Inferred sampling rate must be positive.")
    return float(inferred_sampling_rate_hz)


def _sampling_rate_metadata(
    *,
    provided_sampling_rate_hz: float | None,
    inferred_sampling_rate_hz: float | None,
) -> dict[str, Any]:
    if provided_sampling_rate_hz is None:
        return {"sampling_rate_source": "time_column"} if inferred_sampling_rate_hz is not None else {}
    metadata: dict[str, Any] = {
        "sampling_rate_source": "provided",
        "provided_sampling_rate_hz": float(provided_sampling_rate_hz),
    }
    if inferred_sampling_rate_hz is not None:
        mismatch_fraction = abs(float(provided_sampling_rate_hz) - inferred_sampling_rate_hz) / inferred_sampling_rate_hz
        metadata["sampling_rate_mismatch_fraction"] = mismatch_fraction
    return metadata


def _as_1d(values: NDArray[Any], *, source: Path) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError(f"Signal file contains a scalar, not a time series: {source}")
    squeezed = np.squeeze(array)
    if squeezed.ndim != 1:
        raise ValueError(f"Signal file must contain one-dimensional data: {source}")
    return np.asarray(squeezed, dtype=np.float64)


def _as_1d_or_channel(
    values: NDArray[Any],
    *,
    source: Path,
    channel: int | None,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError(f"Signal file contains a scalar, not a time series: {source}")
    squeezed = np.squeeze(array)
    if squeezed.ndim == 1:
        if channel not in {None, 0}:
            raise ValueError("One-dimensional signal files only contain channel 0.")
        return np.asarray(squeezed, dtype=np.float64)
    if squeezed.ndim == 2:
        if channel is None:
            raise ValueError("Multi-channel files require channel or load_signal_file_channels.")
        return np.asarray(_select_channel(squeezed, channel, source=source), dtype=np.float64)
    raise ValueError(f"Signal file must contain one- or two-dimensional data: {source}")


def _select_channel(values: NDArray[Any], channel: int, *, source: Path) -> NDArray[Any]:
    if channel < 0:
        raise ValueError("channel must be non-negative.")
    if values.ndim != 2:
        raise ValueError(f"Channel selection requires two-dimensional data: {source}")
    if channel >= values.shape[1]:
        raise ValueError(f"channel {channel} is out of range for {values.shape[1]} channel(s).")
    return values[:, channel]


def _channel_metadata(channel: int | None) -> dict[str, Any]:
    return {} if channel is None else {"channel_index": channel}


def _rename_channel_record(record: SignalRecord, channel_name: str, channel_index: int) -> SignalRecord:
    return SignalRecord(
        values=record.values,
        sampling_rate_hz=record.sampling_rate_hz,
        label=record.label,
        name=f"{record.name}_{channel_name}",
        metadata={
            **record.metadata,
            "channel_name": channel_name,
            "channel_index": channel_index,
        },
        provenance=SignalProvenance(
            source_name=record.provenance.source_name or record.name,
            source_path=record.provenance.source_path,
            channel_name=channel_name,
            channel_index=channel_index,
        ),
        acquisition=record.acquisition,
    )


def _matching_paths(data_dir: Path, file_patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in file_patterns:
        paths.extend(data_dir.glob(pattern))
    return sorted({path for path in paths if path.is_file()})


def _metadata_lookup(
    metadata_table: str | Path | pd.DataFrame | None,
) -> dict[str, dict[str, Any]]:
    if metadata_table is None:
        return {}
    dataframe = (
        pd.read_csv(metadata_table)
        if isinstance(metadata_table, str | Path)
        else metadata_table.copy()
    )
    lookup: dict[str, dict[str, Any]] = {}
    key_columns = [column for column in _METADATA_KEY_COLUMNS if column in dataframe.columns]
    if not key_columns:
        raise ValueError(
            "metadata_table must contain at least one key column: "
            + ", ".join(_METADATA_KEY_COLUMNS)
        )
    for _, row in dataframe.iterrows():
        row_dict = {
            str(key): value for key, value in row.to_dict().items() if not pd.isna(value)
        }
        for key_column in key_columns:
            value = row.get(key_column)
            if pd.isna(value):
                continue
            key = str(value)
            lookup[key] = row_dict
            if key_column == "source_path":
                path = Path(key)
                lookup[path.name] = row_dict
                lookup[path.stem] = row_dict
    return lookup


def _apply_external_metadata(
    record: SignalRecord,
    path: Path,
    *,
    metadata_lookup: Mapping[str, dict[str, Any]],
    labels_by_name: Mapping[str, str],
) -> SignalRecord:
    keys = _record_lookup_keys(record, path)
    external_metadata = next(
        (metadata_lookup[key] for key in keys if key in metadata_lookup),
        {},
    )
    external_label = _external_label(external_metadata)
    mapped_label = next((labels_by_name[key] for key in keys if key in labels_by_name), None)
    label = record.label if record.label is not None else mapped_label or external_label
    metadata = dict(record.metadata)
    if external_metadata:
        metadata["external_metadata"] = external_metadata
    return SignalRecord(
        values=record.values,
        sampling_rate_hz=record.sampling_rate_hz,
        label=label,
        name=record.name,
        metadata=metadata,
        provenance=record.provenance,
        acquisition=record.acquisition,
    )


def _record_lookup_keys(record: SignalRecord, path: Path) -> tuple[str, ...]:
    keys = [key for key in (record.name, path.name, path.stem, str(path)) if key]
    return tuple(dict.fromkeys(keys))


def _external_label(metadata: Mapping[str, Any]) -> str | None:
    for column in ("label", "condition", "class", "target"):
        value = metadata.get(column)
        if value is not None and not pd.isna(value):
            return str(value)
    return None


def _time_axis_metadata(dataframe: pd.DataFrame) -> dict[str, Any]:
    time_column = _first_time_column(dataframe)
    if time_column is None:
        return {}
    time_values = pd.to_numeric(dataframe[time_column], errors="coerce").to_numpy(dtype=np.float64)
    finite_time = time_values[np.isfinite(time_values)]
    metadata: dict[str, Any] = {
        "time_column": time_column,
        "time_start_seconds": float(finite_time[0]) if finite_time.size else float("nan"),
        "time_end_seconds": float(finite_time[-1]) if finite_time.size else float("nan"),
    }
    if finite_time.size < 2:
        return {**metadata, "time_axis_valid": False}
    diffs = np.diff(finite_time)
    positive_diffs = diffs[diffs > 0.0]
    if positive_diffs.size != diffs.size:
        return {
            **metadata,
            "time_axis_valid": False,
            "time_axis_nonmonotonic_count": int(diffs.size - positive_diffs.size),
        }
    median_step = float(np.median(positive_diffs))
    if median_step <= 0.0:
        return {**metadata, "time_axis_valid": False}
    max_relative_deviation = float(np.max(np.abs(positive_diffs - median_step)) / median_step)
    gap_count = int(np.count_nonzero(positive_diffs > 1.5 * median_step))
    return {
        **metadata,
        "time_axis_valid": True,
        "time_step_median_seconds": median_step,
        "time_step_jitter_fraction": max_relative_deviation,
        "time_gap_count": gap_count,
        "inferred_sampling_rate_hz": 1.0 / median_step,
    }


def _first_time_column(dataframe: pd.DataFrame) -> str | None:
    for column in dataframe.columns:
        if str(column).lower() in _COMMON_TIME_COLUMNS:
            return str(column)
    return None


def _optional_metadata_float(metadata: Mapping[str, Any], key: str) -> float | None:
    value = metadata.get(key)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_metadata_int(metadata: Mapping[str, Any], key: str) -> int | None:
    value = metadata.get(key)
    if value is None or pd.isna(value):
        return None
    return int(value)


def _optional_metadata_bool(metadata: Mapping[str, Any], key: str) -> bool | None:
    value = metadata.get(key)
    if value is None or pd.isna(value):
        return None
    return bool(value)


def _acquisition_diagnostics(metadata: Mapping[str, Any]) -> AcquisitionDiagnostics:
    """Build typed acquisition diagnostics while retaining legacy metadata."""
    time_column = metadata.get("time_column")
    sampling_rate_source = metadata.get("sampling_rate_source")
    return AcquisitionDiagnostics(
        sampling_rate_source=str(sampling_rate_source) if sampling_rate_source is not None else None,
        provided_sampling_rate_hz=_optional_metadata_float(metadata, "provided_sampling_rate_hz"),
        inferred_sampling_rate_hz=_optional_metadata_float(metadata, "inferred_sampling_rate_hz"),
        time_column=str(time_column) if time_column is not None else None,
        time_axis_valid=_optional_metadata_bool(metadata, "time_axis_valid"),
        time_start_seconds=_optional_metadata_float(metadata, "time_start_seconds"),
        time_end_seconds=_optional_metadata_float(metadata, "time_end_seconds"),
        time_step_median_seconds=_optional_metadata_float(metadata, "time_step_median_seconds"),
        time_step_jitter_fraction=_optional_metadata_float(metadata, "time_step_jitter_fraction"),
        time_gap_count=_optional_metadata_int(metadata, "time_gap_count"),
        sampling_rate_mismatch_fraction=_optional_metadata_float(
            metadata, "sampling_rate_mismatch_fraction"
        ),
    )


def _normalize_wav_values(values: NDArray[Any]) -> NDArray[np.float64]:
    if np.issubdtype(values.dtype, np.unsignedinteger):
        info = np.iinfo(values.dtype)
        midpoint = (info.max + info.min + 1) / 2.0
        return (values.astype(np.float64) - midpoint) / midpoint
    if np.issubdtype(values.dtype, np.signedinteger):
        info = np.iinfo(values.dtype)
        scale = max(abs(info.min), abs(info.max))
        return values.astype(np.float64) / scale
    return values.astype(np.float64)

"""Load common signal file formats into `SignalRecord` objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.io import wavfile

from signal_processing_prep.records import SignalRecord

_COMMON_TIME_COLUMNS = {"time", "timestamp", "t", "seconds", "time_s", "time_seconds"}


def load_signal_file(
    path: str | Path,
    *,
    sampling_rate_hz: float | None = None,
    label: str | None = None,
    signal_column: str | None = None,
    label_column: str | None = None,
) -> SignalRecord:
    """Load one supported signal file into a `SignalRecord`."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        values, metadata, detected_label = _load_csv(
            file_path, signal_column=signal_column, label_column=label_column
        )
        record_sampling_rate = _require_sampling_rate(sampling_rate_hz, file_path)
    elif suffix == ".txt":
        values = _load_text(file_path)
        metadata = {}
        detected_label = None
        record_sampling_rate = _require_sampling_rate(sampling_rate_hz, file_path)
    elif suffix == ".npy":
        values = _load_npy(file_path)
        metadata = {}
        detected_label = None
        record_sampling_rate = _require_sampling_rate(sampling_rate_hz, file_path)
    elif suffix == ".wav":
        record_sampling_rate, values, metadata = _load_wav(file_path)
        detected_label = None
        if sampling_rate_hz is not None and float(sampling_rate_hz) != record_sampling_rate:
            raise ValueError(
                "Provided sampling_rate_hz does not match the WAV file sampling rate."
            )
    else:
        raise ValueError(f"Unsupported signal file extension: {suffix}")

    return SignalRecord(
        values=values,
        sampling_rate_hz=record_sampling_rate,
        label=label if label is not None else detected_label,
        name=file_path.stem,
        metadata={"source_path": str(file_path), **metadata},
    )


def _load_csv(
    path: Path,
    *,
    signal_column: str | None,
    label_column: str | None,
) -> tuple[NDArray[np.float64], dict[str, Any], str | None]:
    dataframe = pd.read_csv(path)
    if dataframe.empty:
        raise ValueError(f"CSV file contains no rows: {path}")

    selected_signal_column = signal_column or _first_numeric_column(dataframe)
    if selected_signal_column not in dataframe.columns:
        raise ValueError(f"Signal column not found in CSV: {selected_signal_column}")

    values = dataframe[selected_signal_column].to_numpy(dtype=np.float64)
    detected_label = _single_label(dataframe, label_column)
    metadata = {
        "format": "csv",
        "signal_column": selected_signal_column,
        "columns": list(dataframe.columns),
    }
    if label_column is not None:
        metadata["label_column"] = label_column
    return values, metadata, detected_label


def _load_text(path: Path) -> NDArray[np.float64]:
    values = np.loadtxt(path, dtype=np.float64)
    return _as_1d(values, source=path)


def _load_npy(path: Path) -> NDArray[np.float64]:
    values = np.load(path)
    return _as_1d(values, source=path)


def _load_wav(path: Path) -> tuple[float, NDArray[np.float64], dict[str, Any]]:
    sampling_rate_hz, values = wavfile.read(path)
    metadata = {"format": "wav", "original_dtype": str(values.dtype)}
    if values.ndim != 1:
        raise ValueError("Only mono WAV files are supported in the basic loader.")
    return float(sampling_rate_hz), _normalize_wav_values(values), metadata


def _first_numeric_column(dataframe: pd.DataFrame) -> str:
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    if numeric_columns.empty:
        raise ValueError("CSV file does not contain a numeric signal column.")

    signal_candidates = [
        column for column in numeric_columns if str(column).lower() not in _COMMON_TIME_COLUMNS
    ]
    if len(signal_candidates) == 1:
        return str(signal_candidates[0])
    if len(signal_candidates) > 1:
        raise ValueError(
            "CSV file contains multiple numeric signal candidates; provide signal_column."
        )
    raise ValueError("CSV file contains only time-like numeric columns; provide signal_column.")


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


def _as_1d(values: NDArray[Any], *, source: Path) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError(f"Signal file contains a scalar, not a time series: {source}")
    squeezed = np.squeeze(array)
    if squeezed.ndim != 1:
        raise ValueError(f"Signal file must contain one-dimensional data: {source}")
    return np.asarray(squeezed, dtype=np.float64)


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

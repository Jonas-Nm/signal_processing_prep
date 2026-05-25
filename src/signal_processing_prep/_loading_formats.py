"""Internal format parsers producing canonical signal records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.io import wavfile

from signal_processing_prep.errors import RecordDataError
from signal_processing_prep.records import AcquisitionDiagnostics, SignalProvenance, SignalRecord

_COMMON_TIME_COLUMNS = {"time", "timestamp", "t", "seconds", "time_s", "time_seconds"}


class SignalFileLoader(Protocol):
    """Protocol implemented by file-format parsing strategies."""

    suffix: str

    def load_record(
        self,
        path: Path,
        *,
        sampling_rate_hz: float | None = None,
        label: str | None = None,
        signal_column: str | None = None,
        label_column: str | None = None,
        channel: int | None = None,
    ) -> SignalRecord:
        """Load one explicitly selected signal channel."""
        ...

    def load_channels(
        self,
        path: Path,
        *,
        sampling_rate_hz: float | None = None,
        label: str | None = None,
        signal_columns: Sequence[str] | None = None,
        label_column: str | None = None,
    ) -> list[SignalRecord]:
        """Load each represented channel as a record."""
        ...


@dataclass(frozen=True)
class _FormatSignalFileLoader:
    suffix: str

    def _check_path(self, path: Path) -> None:
        if path.suffix.lower() != self.suffix:
            raise RecordDataError(f"{type(self).__name__} only handles {self.suffix} files.")

    def _record(
        self,
        path: Path,
        *,
        values: NDArray[np.float64],
        sampling_rate_hz: float,
        label: str | None = None,
        acquisition: AcquisitionDiagnostics | None = None,
        signal_column: str | None = None,
        original_dtype: str | None = None,
        channel_name: str | None = None,
        channel_index: int | None = None,
    ) -> SignalRecord:
        return SignalRecord(
            values=values,
            sampling_rate_hz=sampling_rate_hz,
            label=label,
            name=path.stem if channel_name is None else f"{path.stem}_{channel_name}",
            provenance=SignalProvenance(
                source_name=path.stem,
                source_path=str(path),
                source_format=self.suffix.removeprefix("."),
                signal_column=signal_column,
                original_dtype=original_dtype,
                channel_name=channel_name,
                channel_index=channel_index,
            ),
            acquisition=acquisition or AcquisitionDiagnostics(),
        )


class CsvSignalFileLoader(_FormatSignalFileLoader):
    """Parse CSV columns and derive acquisition timing observations."""

    def __init__(self) -> None:
        super().__init__(".csv")

    def load_record(
        self,
        path: Path,
        *,
        sampling_rate_hz: float | None = None,
        label: str | None = None,
        signal_column: str | None = None,
        label_column: str | None = None,
        channel: int | None = None,
    ) -> SignalRecord:
        self._check_path(path)
        frame = _read_csv(path)
        if frame.empty:
            raise RecordDataError(f"CSV file contains no rows: {path}")
        selected = signal_column or _first_numeric_column(frame, label_column=label_column)
        if selected not in frame.columns:
            raise RecordDataError(f"Signal column not found in CSV: {selected}")
        if channel not in {None, 0}:
            raise RecordDataError("A selected CSV signal column only contains channel 0.")
        detected_label = _single_label(frame, label_column)
        acquisition = _csv_acquisition(frame, sampling_rate_hz)
        resolved_rate = _resolve_sampling_rate(sampling_rate_hz, acquisition, path)
        try:
            values = frame[selected].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise RecordDataError(
                f"CSV signal column '{selected}' cannot be converted to numeric values: {path}"
            ) from error
        return self._record(
            path,
            values=values,
            sampling_rate_hz=resolved_rate,
            label=label if label is not None else detected_label,
            acquisition=acquisition,
            signal_column=selected,
            original_dtype=str(frame[selected].dtype),
        )

    def load_channels(
        self,
        path: Path,
        *,
        sampling_rate_hz: float | None = None,
        label: str | None = None,
        signal_columns: Sequence[str] | None = None,
        label_column: str | None = None,
    ) -> list[SignalRecord]:
        self._check_path(path)
        frame = _read_csv(path)
        columns = list(signal_columns) if signal_columns is not None else _signal_columns(
            frame, label_column=label_column
        )
        if not columns:
            raise RecordDataError("CSV file does not contain numeric signal columns.")
        records: list[SignalRecord] = []
        for index, column in enumerate(columns):
            record = self.load_record(
                path,
                sampling_rate_hz=sampling_rate_hz,
                label=label,
                signal_column=column,
                label_column=label_column,
            )
            records.append(_channel_record(record, column, index))
        return records


class _ArraySignalFileLoader(_FormatSignalFileLoader):
    def _read_array(self, path: Path) -> NDArray[Any]:
        raise NotImplementedError

    def load_record(
        self,
        path: Path,
        *,
        sampling_rate_hz: float | None = None,
        label: str | None = None,
        signal_column: str | None = None,
        label_column: str | None = None,
        channel: int | None = None,
    ) -> SignalRecord:
        del signal_column, label_column
        self._check_path(path)
        rate = _require_sampling_rate(sampling_rate_hz, path)
        raw_values = self._read_array(path)
        return self._record(
            path,
            values=_as_1d_or_channel(raw_values, path=path, channel=channel),
            sampling_rate_hz=rate,
            label=label,
            acquisition=AcquisitionDiagnostics(
                sampling_rate_source="provided", provided_sampling_rate_hz=rate
            ),
            original_dtype=str(np.asarray(raw_values).dtype),
            channel_index=channel,
        )

    def load_channels(
        self,
        path: Path,
        *,
        sampling_rate_hz: float | None = None,
        label: str | None = None,
        signal_columns: Sequence[str] | None = None,
        label_column: str | None = None,
    ) -> list[SignalRecord]:
        del signal_columns, label_column
        self._check_path(path)
        array = np.squeeze(np.asarray(self._read_array(path)))
        if array.ndim == 1:
            return [self.load_record(path, sampling_rate_hz=sampling_rate_hz, label=label)]
        if array.ndim != 2:
            raise RecordDataError(f"Signal file must contain one- or two-dimensional data: {path}")
        return [
            _channel_record(
                self.load_record(path, sampling_rate_hz=sampling_rate_hz, label=label, channel=index),
                str(index),
                index,
            )
            for index in range(array.shape[1])
        ]


class TxtSignalFileLoader(_ArraySignalFileLoader):
    """Parse numeric plain-text signals."""

    def __init__(self) -> None:
        super().__init__(".txt")

    def _read_array(self, path: Path) -> NDArray[Any]:
        try:
            return np.loadtxt(path, dtype=np.float64)
        except (OSError, ValueError) as error:
            raise RecordDataError(f"TXT signal data could not be parsed: {path}") from error


class NpySignalFileLoader(_ArraySignalFileLoader):
    """Parse NumPy arrays as one or more channels."""

    def __init__(self) -> None:
        super().__init__(".npy")

    def _read_array(self, path: Path) -> NDArray[Any]:
        try:
            return np.asarray(np.load(path))
        except (OSError, ValueError) as error:
            raise RecordDataError(f"NPY signal data could not be parsed: {path}") from error


class WavSignalFileLoader(_FormatSignalFileLoader):
    """Parse WAV signals using encoded sampling-rate facts."""

    def __init__(self) -> None:
        super().__init__(".wav")

    def load_record(
        self,
        path: Path,
        *,
        sampling_rate_hz: float | None = None,
        label: str | None = None,
        signal_column: str | None = None,
        label_column: str | None = None,
        channel: int | None = None,
    ) -> SignalRecord:
        del signal_column, label_column
        self._check_path(path)
        raw_rate, raw_values = _read_wav(path)
        rate = float(raw_rate)
        if sampling_rate_hz is not None and float(sampling_rate_hz) != rate:
            raise RecordDataError(
                "Provided sampling_rate_hz does not match the WAV file sampling rate."
            )
        if raw_values.ndim == 1:
            if channel not in {None, 0}:
                raise RecordDataError("Mono WAV files only contain channel 0.")
            values = raw_values
        elif raw_values.ndim == 2:
            if channel is None:
                raise RecordDataError(
                    "Multi-channel WAV files require channel selection or load_channels()."
                )
            values = _select_channel(raw_values, channel, path)
        else:
            raise RecordDataError("WAV data must be one- or two-dimensional.")
        return self._record(
            path,
            values=_normalize_wav_values(values),
            sampling_rate_hz=rate,
            label=label,
            acquisition=AcquisitionDiagnostics(sampling_rate_source="wav_header"),
            original_dtype=str(raw_values.dtype),
            channel_index=channel,
        )

    def load_channels(
        self,
        path: Path,
        *,
        sampling_rate_hz: float | None = None,
        label: str | None = None,
        signal_columns: Sequence[str] | None = None,
        label_column: str | None = None,
    ) -> list[SignalRecord]:
        del signal_columns, label_column
        _, values = _read_wav(path)
        if values.ndim == 1:
            return [self.load_record(path, sampling_rate_hz=sampling_rate_hz, label=label)]
        if values.ndim != 2:
            raise RecordDataError("WAV data must be one- or two-dimensional.")
        return [
            _channel_record(
                self.load_record(path, sampling_rate_hz=sampling_rate_hz, label=label, channel=index),
                str(index),
                index,
            )
            for index in range(values.shape[1])
        ]


def default_loaders() -> dict[str, SignalFileLoader]:
    """Return default registered file-format strategies."""
    return {
        ".csv": CsvSignalFileLoader(),
        ".txt": TxtSignalFileLoader(),
        ".npy": NpySignalFileLoader(),
        ".wav": WavSignalFileLoader(),
    }


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, UnicodeError, ValueError) as error:
        raise RecordDataError(f"CSV signal data could not be parsed: {path}") from error


def _read_wav(path: Path) -> tuple[int, NDArray[Any]]:
    try:
        return wavfile.read(path)
    except (OSError, ValueError) as error:
        raise RecordDataError(f"WAV signal data could not be parsed: {path}") from error


def _channel_record(record: SignalRecord, channel_name: str, channel_index: int) -> SignalRecord:
    provenance = record.provenance
    return SignalRecord(
        values=record.values,
        sampling_rate_hz=record.sampling_rate_hz,
        label=record.label,
        name=f"{provenance.source_name or record.name}_{channel_name}",
        attributes=record.attributes,
        provenance=SignalProvenance(
            source_name=provenance.source_name,
            source_path=provenance.source_path,
            source_format=provenance.source_format,
            signal_column=provenance.signal_column,
            original_dtype=provenance.original_dtype,
            channel_name=channel_name,
            channel_index=channel_index,
        ),
        acquisition=record.acquisition,
        annotations=record.annotations,
        processing_history=record.processing_history,
    )


def _first_numeric_column(frame: pd.DataFrame, *, label_column: str | None) -> str:
    candidates = _signal_columns(frame, label_column=label_column)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RecordDataError("CSV contains multiple numeric signal candidates; provide signal_column.")
    raise RecordDataError("CSV does not contain a numeric non-time signal column.")


def _signal_columns(frame: pd.DataFrame, *, label_column: str | None) -> list[str]:
    excluded = set(_COMMON_TIME_COLUMNS)
    if label_column is not None:
        excluded.add(label_column.lower())
    return [
        str(column)
        for column in frame.select_dtypes(include=[np.number]).columns
        if str(column).lower() not in excluded
    ]


def _single_label(frame: pd.DataFrame, label_column: str | None) -> str | None:
    if label_column is None:
        return None
    if label_column not in frame.columns:
        raise RecordDataError(f"Label column not found in CSV: {label_column}")
    labels = frame[label_column].dropna().unique()
    if len(labels) > 1:
        raise RecordDataError("A single SignalRecord cannot represent multiple CSV labels.")
    return str(labels[0]) if len(labels) else None


def _csv_acquisition(frame: pd.DataFrame, provided_rate: float | None) -> AcquisitionDiagnostics:
    time_column = next(
        (str(column) for column in frame.columns if str(column).lower() in _COMMON_TIME_COLUMNS),
        None,
    )
    if time_column is None:
        return AcquisitionDiagnostics(
            sampling_rate_source="provided" if provided_rate is not None else None,
            provided_sampling_rate_hz=provided_rate,
        )
    time = pd.to_numeric(frame[time_column], errors="coerce").to_numpy(dtype=np.float64)
    finite = time[np.isfinite(time)]
    if finite.size < 2:
        return AcquisitionDiagnostics(time_column=time_column, time_axis_valid=False)
    diffs = np.diff(finite)
    if np.any(diffs <= 0):
        return AcquisitionDiagnostics(
            time_column=time_column,
            time_axis_valid=False,
            time_start_seconds=float(finite[0]),
            time_end_seconds=float(finite[-1]),
        )
    median = float(np.median(diffs))
    inferred = 1.0 / median
    mismatch = None if provided_rate is None else abs(float(provided_rate) - inferred) / inferred
    return AcquisitionDiagnostics(
        sampling_rate_source="provided" if provided_rate is not None else "time_column",
        provided_sampling_rate_hz=provided_rate,
        inferred_sampling_rate_hz=inferred,
        time_column=time_column,
        time_axis_valid=True,
        time_start_seconds=float(finite[0]),
        time_end_seconds=float(finite[-1]),
        time_step_median_seconds=median,
        time_step_jitter_fraction=float(np.max(np.abs(diffs - median)) / median),
        time_gap_count=int(np.count_nonzero(diffs > 1.5 * median)),
        sampling_rate_mismatch_fraction=mismatch,
    )


def _resolve_sampling_rate(
    provided_rate: float | None, acquisition: AcquisitionDiagnostics, path: Path
) -> float:
    if provided_rate is not None:
        return _require_sampling_rate(provided_rate, path)
    if acquisition.inferred_sampling_rate_hz is None:
        raise RecordDataError(
            f"sampling_rate_hz is required for {path.suffix} files without a usable time column."
        )
    return acquisition.inferred_sampling_rate_hz


def _require_sampling_rate(rate: float | None, path: Path) -> float:
    if rate is None:
        raise RecordDataError(f"sampling_rate_hz is required for {path.suffix} files.")
    if rate <= 0:
        raise RecordDataError("sampling_rate_hz must be positive.")
    return float(rate)


def _as_1d_or_channel(values: NDArray[Any], *, path: Path, channel: int | None) -> NDArray[np.float64]:
    array = np.squeeze(np.asarray(values, dtype=np.float64))
    if array.ndim == 1:
        if channel not in {None, 0}:
            raise RecordDataError("One-dimensional signal files only contain channel 0.")
        return array
    if array.ndim == 2 and channel is not None:
        return np.asarray(_select_channel(array, channel, path), dtype=np.float64)
    if array.ndim == 2:
        raise RecordDataError("Multi-channel files require channel selection or load_channels().")
    raise RecordDataError(f"Signal file must contain one- or two-dimensional data: {path}")


def _select_channel(values: NDArray[Any], channel: int, path: Path) -> NDArray[Any]:
    if channel < 0 or channel >= values.shape[1]:
        raise RecordDataError(f"channel {channel} is out of range for {values.shape[1]} channel(s).")
    return values[:, channel]


def _normalize_wav_values(values: NDArray[Any]) -> NDArray[np.float64]:
    if np.issubdtype(values.dtype, np.unsignedinteger):
        info = np.iinfo(values.dtype)
        midpoint = (info.max + info.min + 1) / 2.0
        return (values.astype(np.float64) - midpoint) / midpoint
    if np.issubdtype(values.dtype, np.signedinteger):
        info = np.iinfo(values.dtype)
        return values.astype(np.float64) / max(abs(info.min), abs(info.max))
    return values.astype(np.float64)

"""Canonical immutable domain structures for sampled signal analysis."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

_UNSET = object()


def _freeze_value(value: Any) -> Any:
    """Recursively freeze values retained by domain records."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True)
        array.setflags(write=False)
        return array
    return value


@dataclass(frozen=True)
class SignalProvenance:
    """Stable source identity and source-format facts for one signal."""

    source_name: str | None = None
    source_path: str | None = None
    source_format: str | None = None
    signal_column: str | None = None
    original_dtype: str | None = None
    channel_name: str | None = None
    channel_index: int | None = None


@dataclass(frozen=True)
class AcquisitionDiagnostics:
    """Loader-derived sampling-time observations."""

    sampling_rate_source: str | None = None
    provided_sampling_rate_hz: float | None = None
    inferred_sampling_rate_hz: float | None = None
    time_column: str | None = None
    time_axis_valid: bool | None = None
    time_start_seconds: float | None = None
    time_end_seconds: float | None = None
    time_step_median_seconds: float | None = None
    time_step_jitter_fraction: float | None = None
    time_gap_count: int | None = None
    sampling_rate_mismatch_fraction: float | None = None


@dataclass(frozen=True)
class AnnotationInterval:
    """A known labeled interval within a signal record."""

    start_seconds: float
    end_seconds: float
    kind: str | None = None

    def __post_init__(self) -> None:
        """Validate interval bounds."""
        if self.start_seconds < 0:
            raise ValueError("Annotation interval start_seconds must be non-negative.")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("Annotation interval end_seconds must exceed start_seconds.")


@dataclass(frozen=True)
class RecordAnnotations:
    """Dataset annotations attached to a record after metadata joining."""

    split: str | None = None
    sensor: str | None = None
    is_anomalous: bool | None = None
    anomaly_kind: str | None = None
    intervals: tuple[AnnotationInterval, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze additional annotation values."""
        object.__setattr__(self, "intervals", tuple(self.intervals))
        object.__setattr__(self, "attributes", _freeze_value(self.attributes))


@dataclass(frozen=True)
class SegmentSpan:
    """Position of a derived record within its source sampled signal."""

    index: int
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    source_name: str | None = None

    def __post_init__(self) -> None:
        """Validate segment boundaries."""
        if self.index < 0 or self.start_sample < 0:
            raise ValueError("Segment index and start_sample must be non-negative.")
        if self.end_sample <= self.start_sample:
            raise ValueError("Segment end_sample must exceed start_sample.")
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("Segment end_seconds must exceed non-negative start_seconds.")


@dataclass(frozen=True)
class ProcessingStep:
    """One explicit transformation used to derive a signal record."""

    operation: str
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and freeze transformation parameters."""
        if not self.operation:
            raise ValueError("Processing step operation must not be empty.")
        object.__setattr__(self, "parameters", _freeze_value(self.parameters))


@dataclass(frozen=True)
class SignalRecord:
    """Immutable sampled signal with typed identity, observations, and history."""

    values: NDArray[np.float64]
    sampling_rate_hz: float
    label: str | None = None
    name: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    provenance: SignalProvenance = field(default_factory=SignalProvenance)
    acquisition: AcquisitionDiagnostics = field(default_factory=AcquisitionDiagnostics)
    annotations: RecordAnnotations = field(default_factory=RecordAnnotations)
    segment_span: SegmentSpan | None = None
    processing_history: tuple[ProcessingStep, ...] = ()

    def __post_init__(self) -> None:
        """Validate and protect record-owned storage from in-place mutation."""
        values = np.array(self.values, dtype=np.float64, copy=True)
        if values.ndim != 1:
            raise ValueError("SignalRecord values must be a one-dimensional array.")
        if values.size == 0:
            raise ValueError("SignalRecord values must not be empty.")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive.")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "attributes", _freeze_value(self.attributes))
        object.__setattr__(self, "processing_history", tuple(self.processing_history))
        if self.provenance.source_name is None and self.name is not None:
            object.__setattr__(
                self,
                "provenance",
                SignalProvenance(
                    source_name=self.name,
                    source_path=self.provenance.source_path,
                    source_format=self.provenance.source_format,
                    signal_column=self.provenance.signal_column,
                    original_dtype=self.provenance.original_dtype,
                    channel_name=self.provenance.channel_name,
                    channel_index=self.provenance.channel_index,
                ),
            )

    @property
    def n_samples(self) -> int:
        """Return the number of samples in the signal."""
        return int(self.values.size)

    @property
    def duration_seconds(self) -> float:
        """Return signal duration in seconds."""
        return self.n_samples / self.sampling_rate_hz

    @property
    def time_seconds(self) -> NDArray[np.float64]:
        """Return a read-only regular sample time axis."""
        values = np.arange(self.n_samples, dtype=np.float64) / self.sampling_rate_hz
        values.setflags(write=False)
        return values

    def derive(
        self,
        values: NDArray[np.float64],
        *,
        name: str | None | object = _UNSET,
        label: str | None | object = _UNSET,
        attribute_updates: Mapping[str, Any] | None = None,
        processing_step: ProcessingStep | None = None,
        segment_span: SegmentSpan | None = None,
        annotations: RecordAnnotations | None = None,
    ) -> SignalRecord:
        """Create a derived immutable record while preserving source identity."""
        attributes = dict(self.attributes)
        if attribute_updates:
            attributes.update(attribute_updates)
        history = self.processing_history
        if processing_step is not None:
            history = (*history, processing_step)
        return SignalRecord(
            values=values,
            sampling_rate_hz=self.sampling_rate_hz,
            label=self.label if label is _UNSET else label,  # type: ignore[arg-type]
            name=self.name if name is _UNSET else name,  # type: ignore[arg-type]
            attributes=attributes,
            provenance=self.provenance,
            acquisition=self.acquisition,
            annotations=self.annotations if annotations is None else annotations,
            segment_span=segment_span,
            processing_history=history,
        )

    def with_values(
        self,
        values: NDArray[np.float64],
        *,
        attribute_updates: Mapping[str, Any] | None = None,
        processing_step: ProcessingStep | None = None,
    ) -> SignalRecord:
        """Replace values as an explicit derived processing result."""
        return self.derive(
            values,
            attribute_updates=attribute_updates,
            processing_step=processing_step,
            segment_span=self.segment_span,
        )

    def segment(self, start_sample: int, end_sample: int, index: int) -> SignalRecord:
        """Return a typed fixed interval segment of this record."""
        if start_sample < 0 or end_sample > self.n_samples or start_sample >= end_sample:
            raise ValueError(
                "Segment bounds must satisfy 0 <= start_sample < end_sample <= n_samples."
            )
        span = SegmentSpan(
            index=index,
            start_sample=start_sample,
            end_sample=end_sample,
            start_seconds=start_sample / self.sampling_rate_hz,
            end_seconds=end_sample / self.sampling_rate_hz,
            source_name=self.provenance.source_name or self.name,
        )
        return self.derive(
            self.values[start_sample:end_sample],
            name=f"{self.name or 'record'}_window_{index}",
            segment_span=span,
        )


@dataclass(frozen=True)
class SignalDataset:
    """Immutable signal-record collection with domain-level selections."""

    records: tuple[SignalRecord, ...]

    def __post_init__(self) -> None:
        """Keep record storage immutable even when constructed directly."""
        object.__setattr__(self, "records", tuple(self.records))

    @classmethod
    def from_records(cls, records: Iterable[SignalRecord]) -> SignalDataset:
        """Create a dataset from an iterable of records."""
        return cls(tuple(records))

    def __iter__(self) -> Iterator[SignalRecord]:
        """Iterate over records in discovery order."""
        return iter(self.records)

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self.records)

    def by_split(self, split: str) -> SignalDataset:
        """Return records having the requested split annotation."""
        expected = split.lower()
        return SignalDataset(
            tuple(
                record
                for record in self.records
                if record.annotations.split is not None
                and record.annotations.split.lower() == expected
            )
        )

    def by_label(self, label: str) -> SignalDataset:
        """Return records having the requested record label."""
        return SignalDataset(tuple(record for record in self.records if record.label == label))

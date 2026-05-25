"""Shared data structures for signal analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SignalProvenance:
    """Stable source identity for one signal or derived signal segment."""

    source_name: str | None = None
    source_path: str | None = None
    channel_name: str | None = None
    channel_index: int | None = None


@dataclass(frozen=True)
class AcquisitionDiagnostics:
    """Loader-derived acquisition and sampling-time diagnostics."""

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
class SignalRecord:
    """Container for one sampled time-series signal and its metadata."""

    values: NDArray[np.float64]
    sampling_rate_hz: float
    label: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: SignalProvenance = field(default_factory=SignalProvenance)
    acquisition: AcquisitionDiagnostics = field(default_factory=AcquisitionDiagnostics)

    def __post_init__(self) -> None:
        """Validate and normalize signal values after initialization."""
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("SignalRecord values must be a one-dimensional array.")
        if values.size == 0:
            raise ValueError("SignalRecord values must not be empty.")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive.")

        object.__setattr__(self, "values", values)
        if self.provenance.source_name is None and self.name is not None:
            object.__setattr__(
                self,
                "provenance",
                SignalProvenance(
                    source_name=self.name,
                    source_path=self.provenance.source_path,
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
        """Return the signal duration in seconds."""
        return self.n_samples / self.sampling_rate_hz

    @property
    def time_seconds(self) -> NDArray[np.float64]:
        """Return a time axis in seconds for each sample."""
        return np.arange(self.n_samples, dtype=np.float64) / self.sampling_rate_hz

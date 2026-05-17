"""Shared data structures for signal analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SignalRecord:
    """Container for one sampled time-series signal and its metadata."""

    values: NDArray[np.float64]
    sampling_rate_hz: float
    label: str | None = None
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

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

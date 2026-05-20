"""Shared validation helpers for signal-processing routines."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def validate_finite_signal_values(values: NDArray[np.float64]) -> None:
    """Reject non-finite samples before DSP routines produce misleading output."""
    if not np.isfinite(values).all():
        raise ValueError(
            "Signal contains non-finite samples; run quality checks and clean, "
            "skip, segment, or impute the record before FFT, PSD, STFT, or other "
            "frequency-domain analysis."
        )

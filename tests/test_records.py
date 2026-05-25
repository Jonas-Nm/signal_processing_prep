"""Tests for immutable canonical domain records."""

import numpy as np
import pytest

from signal_processing_prep.records import (
    AcquisitionDiagnostics,
    ProcessingStep,
    RecordAnnotations,
    SegmentSpan,
    SignalProvenance,
    SignalDataset,
    SignalRecord,
)


def test_signal_record_exposes_typed_identity_and_immutable_attributes() -> None:
    record = SignalRecord(
        values=[0, 1, 0, -1],
        sampling_rate_hz=4.0,
        label="normal",
        name="example",
        attributes={"sensor_note": "temporary", "nested": {"x": [1, 2]}},
        provenance=SignalProvenance(source_format="csv", signal_column="accel"),
    )

    assert record.values.dtype == np.float64
    assert record.duration_seconds == 1.0
    assert record.provenance.source_name == "example"
    assert record.provenance.signal_column == "accel"
    assert record.attributes["nested"]["x"] == (1, 2)
    with pytest.raises(ValueError):
        record.values[0] = 5.0
    with pytest.raises(TypeError):
        record.attributes["new"] = "value"  # type: ignore[index]


def test_signal_record_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        SignalRecord(values=[], sampling_rate_hz=1.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        SignalRecord(values=[[1.0, 2.0]], sampling_rate_hz=1.0)
    with pytest.raises(ValueError, match="positive"):
        SignalRecord(values=[1.0], sampling_rate_hz=0.0)


def test_derivation_preserves_typed_state_and_adds_processing_history() -> None:
    record = SignalRecord(
        values=[0.0, 1.0, 0.0],
        sampling_rate_hz=10.0,
        name="capture",
        acquisition=AcquisitionDiagnostics(time_axis_valid=True),
        annotations=RecordAnnotations(sensor="accel"),
    )
    derived = record.with_values(
        np.array([0.0, 0.5, 0.0]),
        processing_step=ProcessingStep("filter", {"kind": "lowpass"}),
    )

    assert derived.provenance == record.provenance
    assert derived.acquisition == record.acquisition
    assert derived.annotations == record.annotations
    assert derived.processing_history[0].parameters["kind"] == "lowpass"
    assert record.derive(record.values, label=None).label is None


def test_segment_position_is_typed_without_metadata_mirror() -> None:
    record = SignalRecord(values=np.arange(10), sampling_rate_hz=10.0, name="capture")

    segment = record.segment(2, 6, 1)

    assert segment.segment_span == SegmentSpan(1, 2, 6, 0.2, 0.6, "capture")
    assert "window_start_seconds" not in segment.attributes
    with pytest.raises(ValueError, match="Segment bounds"):
        record.segment(0, 20, 0)


def test_dataset_copies_record_sequences_and_selects_typed_annotations() -> None:
    records = [
        SignalRecord([0.0], 1.0, name="train", annotations=RecordAnnotations(split="train")),
        SignalRecord([1.0], 1.0, name="test", annotations=RecordAnnotations(split="test")),
    ]
    dataset = SignalDataset(records)  # type: ignore[arg-type]
    records.clear()

    assert len(dataset) == 2
    assert dataset.by_split("test").records[0].name == "test"

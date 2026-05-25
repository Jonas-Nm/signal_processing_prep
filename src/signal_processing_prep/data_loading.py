"""Public dataset-loading facade for canonical signal records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from signal_processing_prep._loading_annotations import DatasetAnnotationJoiner
from signal_processing_prep._loading_formats import (
    CsvSignalFileLoader,
    NpySignalFileLoader,
    SignalFileLoader,
    TxtSignalFileLoader,
    WavSignalFileLoader,
    default_loaders,
)
from signal_processing_prep.config import ProjectConfig, load_config
from signal_processing_prep.errors import ConfigurationError, RecordDataError
from signal_processing_prep.records import SignalDataset, SignalRecord


@dataclass(frozen=True)
class DatasetLoadingResult:
    """Loaded records and explicit diagnostics for skipped malformed inputs."""

    dataset: SignalDataset
    notes: tuple[str, ...] = ()


@dataclass
class SignalDatasetLoader:
    """Load files and annotations through replaceable parser strategies."""

    loaders: Mapping[str, SignalFileLoader] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Install default strategies or normalize registered suffixes."""
        self.loaders = (
            default_loaders()
            if not self.loaders
            else {suffix.lower(): loader for suffix, loader in self.loaders.items()}
        )

    def _loader_for(self, path: Path) -> SignalFileLoader:
        loader = self.loaders.get(path.suffix.lower())
        if loader is None:
            raise RecordDataError(f"Unsupported signal file extension: {path.suffix.lower()}")
        return loader

    def load_file(self, path: str | Path, **kwargs: Any) -> SignalRecord:
        """Load one selected channel through its format strategy."""
        file_path = Path(path)
        return self._loader_for(file_path).load_record(file_path, **kwargs)

    def load_file_channels(self, path: str | Path, **kwargs: Any) -> SignalDataset:
        """Load all represented channels through their format strategy."""
        file_path = Path(path)
        return SignalDataset.from_records(
            self._loader_for(file_path).load_channels(file_path, **kwargs)
        )

    def load_dataset(
        self,
        config: ProjectConfig | str | Path,
        *,
        metadata_table: str | Path | pd.DataFrame | None = None,
        labels_by_name: Mapping[str, str] | None = None,
        split_channels: bool = True,
    ) -> SignalDataset:
        """Load configured files and attach canonical typed annotations."""
        return self.load_dataset_result(
            config,
            metadata_table=metadata_table,
            labels_by_name=labels_by_name,
            split_channels=split_channels,
        ).dataset

    def load_dataset_result(
        self,
        config: ProjectConfig | str | Path,
        *,
        metadata_table: str | Path | pd.DataFrame | None = None,
        labels_by_name: Mapping[str, str] | None = None,
        split_channels: bool = True,
        invalid_record_policy: str = "raise",
    ) -> DatasetLoadingResult:
        """Load configured records while optionally skipping malformed files."""
        if invalid_record_policy not in {"skip", "raise"}:
            raise ConfigurationError("invalid_record_policy must be 'skip' or 'raise'.")
        project = load_config(config) if isinstance(config, str | Path) else config
        joiner = DatasetAnnotationJoiner.from_inputs(metadata_table, labels_by_name)
        records: list[SignalRecord] = []
        notes: list[str] = []
        for path in _matching_paths(project.paths.data_dir, project.loading.file_patterns):
            kwargs: dict[str, Any] = {
                "sampling_rate_hz": project.loading.sampling_rate_hz,
                "label_column": project.loading.label_column,
            }
            try:
                if split_channels:
                    kwargs["signal_columns"] = (
                        [project.loading.signal_column]
                        if project.loading.signal_column is not None
                        else None
                    )
                    loaded = self.load_file_channels(path, **kwargs).records
                else:
                    kwargs["signal_column"] = project.loading.signal_column
                    loaded = (self.load_file(path, **kwargs),)
                records.extend(joiner.annotate(record, path) for record in loaded)
            except RecordDataError as error:
                if invalid_record_policy == "raise":
                    raise
                notes.append(f"Loading skipped {path.name}: {error}")
        return DatasetLoadingResult(SignalDataset.from_records(records), tuple(notes))


def _matching_paths(data_dir: Path, patterns: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(data_dir.glob(pattern))
    return sorted({path for path in paths if path.is_file()})


__all__ = [
    "SignalDatasetLoader",
    "DatasetLoadingResult",
    "SignalFileLoader",
    "CsvSignalFileLoader",
    "TxtSignalFileLoader",
    "NpySignalFileLoader",
    "WavSignalFileLoader",
]

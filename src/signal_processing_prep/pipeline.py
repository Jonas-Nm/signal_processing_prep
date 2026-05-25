"""High-level orchestration for exploratory signal-analysis workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from signal_processing_prep._modeling_common import ModelEvaluation
from signal_processing_prep.artifacts import FeatureTable, PredictionTable, QualityTable
from signal_processing_prep.config import AnalysisConfig, ProjectConfig, load_config
from signal_processing_prep.data_loading import SignalDatasetLoader
from signal_processing_prep.errors import ConfigurationError, RecordDataError
from signal_processing_prep.features import FeatureExtractor
from signal_processing_prep.modeling import (
    AnomalyScorer,
    IsolationForestScorer,
    SupervisedBaselineSuite,
)
from signal_processing_prep.preprocessing import SignalPreprocessor
from signal_processing_prep.quality import QualityAssessor
from signal_processing_prep.records import SignalDataset, SignalRecord
from signal_processing_prep.reporting import AnalysisReport, AnalysisReportBuilder
from signal_processing_prep.synthetic import make_synthetic_dataset


@dataclass(frozen=True)
class AnalysisResult:
    """Typed workflow artifacts with convenient notebook-facing views."""

    dataset: SignalDataset
    quality: QualityTable
    features: FeatureTable
    supervised_models: dict[str, ModelEvaluation]
    anomaly_model: ModelEvaluation | None
    report: AnalysisReport
    modeling_notes: tuple[str, ...]
    processing_notes: tuple[str, ...] = ()

    @property
    def records(self) -> tuple[SignalRecord, ...]:
        """Return processed records represented by this result."""
        return self.dataset.records

    @property
    def quality_frame(self) -> pd.DataFrame:
        """Return quality observations as a DataFrame copy."""
        return self.quality.to_dataframe()

    @property
    def feature_frame(self) -> pd.DataFrame:
        """Return feature values as a DataFrame copy."""
        return self.features.to_dataframe()

    @property
    def markdown_summary(self) -> str:
        """Render report Markdown for notebook or presentation output."""
        return self.report.render_markdown()


class SignalAnalysisPipeline:
    """Configured workflow orchestrating deep, replaceable collaborators."""

    def __init__(
        self,
        analysis_config: AnalysisConfig | None = None,
        *,
        project_config: ProjectConfig | None = None,
        dataset_loader: SignalDatasetLoader | None = None,
        feature_extractor: FeatureExtractor | None = None,
        anomaly_scorer: AnomalyScorer | None = None,
        supervised_suite: SupervisedBaselineSuite | None = None,
        report_builder: AnalysisReportBuilder | None = None,
        quality_assessor: QualityAssessor | None = None,
        preprocessor: SignalPreprocessor | None = None,
    ) -> None:
        """Construct a workflow and resolve configured stage strategies once."""
        self.analysis_config = analysis_config or (
            project_config.analysis if project_config is not None else AnalysisConfig()
        )
        self.project_config = project_config
        modeling = self.analysis_config.modeling
        self.dataset_loader = dataset_loader or SignalDatasetLoader()
        self.feature_extractor = feature_extractor or FeatureExtractor.from_analysis_config(
            self.analysis_config
        )
        self.anomaly_scorer = anomaly_scorer or IsolationForestScorer(
            contamination=modeling.anomaly_contamination,
            random_state=modeling.random_state,
        )
        self.supervised_suite = supervised_suite or SupervisedBaselineSuite(
            random_state=modeling.random_state
        )
        self.report_builder = report_builder or AnalysisReportBuilder()
        self.quality_assessor = quality_assessor or QualityAssessor(self.analysis_config.quality)
        self.preprocessor = preprocessor or SignalPreprocessor(self.analysis_config.filtering)

    @classmethod
    def from_config(
        cls,
        config: ProjectConfig | str | Path,
        **collaborators: object,
    ) -> SignalAnalysisPipeline:
        """Create a pipeline from YAML or a validated project configuration."""
        project = load_config(config) if isinstance(config, str | Path) else config
        return cls(
            project.analysis,
            project_config=project,
            **collaborators,  # type: ignore[arg-type]
        )

    def run_dataset(
        self,
        *,
        metadata_table: str | Path | pd.DataFrame | None = None,
        labels_by_name: Mapping[str, str] | None = None,
        split_channels: bool = True,
    ) -> AnalysisResult:
        """Load and analyze the dataset attached to this configured pipeline."""
        if self.project_config is None:
            raise ValueError("run_dataset() requires construction through from_config().")
        loading = self.dataset_loader.load_dataset_result(
            self.project_config,
            metadata_table=metadata_table,
            labels_by_name=labels_by_name,
            split_channels=split_channels,
            invalid_record_policy=self.analysis_config.invalid_record_policy,
        )
        if len(loading.dataset) == 0 and loading.notes:
            raise RecordDataError(
                "No valid records remained after loading. " + " ".join(loading.notes)
            )
        return self._run_records(loading.dataset, list(loading.notes))

    def run_records(self, records: SignalDataset | Sequence[SignalRecord]) -> AnalysisResult:
        """Analyze already available records under configured failure policy."""
        raw_dataset = records if isinstance(records, SignalDataset) else SignalDataset.from_records(records)
        return self._run_records(raw_dataset, [])

    def _run_records(
        self, raw_dataset: SignalDataset, initial_processing_notes: list[str]
    ) -> AnalysisResult:
        """Run stages after loading while retaining input-boundary diagnostics."""
        if len(raw_dataset) == 0:
            raise ValueError("At least one SignalRecord is required.")
        quality = self.quality_assessor.assess(raw_dataset.records)
        processed, processing_notes = self._prepare_analysis_records(raw_dataset.records)
        processing_notes = [*initial_processing_notes, *processing_notes]
        analyzed, features, feature_notes = self._extract_feature_table(processed.records)
        processing_notes.extend(feature_notes)
        supervised: dict[str, ModelEvaluation] = {}
        anomaly: ModelEvaluation | None = None
        modeling_notes: list[str] = []
        if self.analysis_config.modeling.enabled and not features.empty:
            supervised, anomaly, modeling_notes = self._run_optional_modeling(features)
        elif self.analysis_config.modeling.enabled:
            modeling_notes.append("Modeling skipped: no feature rows were available.")
        report = self.report_builder.build(
            record_count=len(raw_dataset),
            quality=quality,
            features=features,
            supervised_models=supervised,
            anomaly_model=anomaly,
            modeling_notes=modeling_notes,
            processing_notes=processing_notes,
        )
        return AnalysisResult(
            dataset=analyzed,
            quality=quality,
            features=features,
            supervised_models=supervised,
            anomaly_model=anomaly,
            report=report,
            modeling_notes=tuple(modeling_notes),
            processing_notes=tuple(processing_notes),
        )

    def run_synthetic(self) -> AnalysisResult:
        """Analyze the built-in synthetic demonstration dataset."""
        return self.run_records(SignalDataset.from_records(make_synthetic_dataset()))

    def _prepare_analysis_records(
        self, records: Sequence[SignalRecord]
    ) -> tuple[SignalDataset, list[str]]:
        kept: list[SignalRecord] = []
        notes: list[str] = []
        for record in records:
            try:
                kept.append(self.preprocessor.process(record))
            except RecordDataError as error:
                if self.analysis_config.invalid_record_policy == "raise":
                    raise
                notes.append(f"Preprocessing skipped {record.name or '<unnamed>'}: {error}")
        return SignalDataset.from_records(kept), notes

    def _extract_feature_table(
        self, records: Sequence[SignalRecord]
    ) -> tuple[SignalDataset, FeatureTable, list[str]]:
        kept: list[SignalRecord] = []
        tables: list[FeatureTable] = []
        notes: list[str] = []
        for record in records:
            try:
                tables.append(self.feature_extractor.extract_records([record]))
                kept.append(record)
            except RecordDataError as error:
                if self.analysis_config.invalid_record_policy == "raise":
                    raise
                notes.append(f"Feature extraction skipped {record.name or '<unnamed>'}: {error}")
        return SignalDataset.from_records(kept), FeatureTable.concat(tables), notes

    def _run_optional_modeling(
        self, features: FeatureTable
    ) -> tuple[dict[str, ModelEvaluation], ModelEvaluation | None, list[str]]:
        notes: list[str] = []
        labels = features.labels()
        labeled = labels is not None and not labels.isna().all()
        has_unlabeled = labels is not None and labels.isna().any()
        if labeled:
            try:
                supervised = self.supervised_suite.evaluate(features)
            except ConfigurationError:
                raise
            except ValueError as error:
                notes.append(f"Supervised baselines skipped: {error}")
            else:
                for name, evaluation in supervised.items():
                    if not isinstance(evaluation.predictions, PredictionTable):
                        raise TypeError(
                            f"Supervised model '{name}' must return a PredictionTable artifact."
                        )
                    evaluation.predictions.validate_alignment(features)
                notes.append("Supervised baselines were trained because labels were available.")
                if not (
                    self.analysis_config.modeling.anomaly_when_unlabeled and has_unlabeled
                ):
                    return supervised, None, notes
                anomaly = self._optional_anomaly(features, notes)
                return supervised, anomaly, notes
        anomaly = self._optional_anomaly(features, notes)
        return {}, anomaly, notes

    def _optional_anomaly(
        self, features: FeatureTable, notes: list[str]
    ) -> ModelEvaluation | None:
        try:
            evaluation = self.anomaly_scorer.score(features)
        except ConfigurationError:
            raise
        except ValueError as error:
            notes.append(f"Anomaly scoring skipped: {error}")
            return None
        if evaluation.task_type != "unsupervised_anomaly_score":
            raise ValueError(
                "Anomaly scorer must return task_type='unsupervised_anomaly_score'."
            )
        if not isinstance(evaluation.predictions, PredictionTable):
            raise TypeError("Anomaly scorer must return a PredictionTable artifact.")
        evaluation.predictions.validate_alignment(features)
        notes.append(
            f"{evaluation.model_name.replace('_', ' ').title()} anomaly scoring was run "
            "as an exploratory baseline."
        )
        return evaluation


def run_synthetic_analysis(config: AnalysisConfig | None = None) -> AnalysisResult:
    """Run the preferred workflow against built-in synthetic records."""
    return SignalAnalysisPipeline(config).run_synthetic()

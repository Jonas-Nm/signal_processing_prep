"""Supervised baseline classification for extracted signal features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from signal_processing_prep._modeling_common import (
    SUPERVISED_METADATA_COLUMNS,
    ModelEvaluation,
    attach_available_metadata,
)
from signal_processing_prep.artifacts import FeatureTable, PredictionTable


def run_supervised_baselines(
    features: FeatureTable,
    *,
    label_column: str = "label",
    group_column: str | None = None,
    test_size: float = 0.3,
    random_state: int = 0,
) -> dict[str, ModelEvaluation]:
    """Train interpretable baselines using source-grouped evaluation where possible."""
    labeled, y = _supervised_labeled_rows(features.to_dataframe(), label_column)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in the interval (0, 1).")
    if y.nunique() < 2:
        raise ValueError("At least two labels are required for supervised modeling.")
    if len(y) < 4:
        raise ValueError("At least four labeled rows are required for train/test evaluation.")
    if y.value_counts().min() < 2:
        raise ValueError("Each label must have at least two rows for train/test evaluation.")

    labels = tuple(sorted(y.astype(str).unique()))
    train_index, test_index, split_strategy = _supervised_split(
        labeled,
        y,
        labeled,
        group_column=group_column,
        test_size=test_size,
        random_state=random_state,
    )
    x_train, x_test, y_train, y_test, test_metadata, feature_columns = (
        _supervised_feature_matrices(
            labeled,
            y,
            train_index,
            test_index,
            feature_columns=features.feature_columns,
        )
    )
    models: dict[str, Any] = {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }
    return {
        name: _fit_supervised_model(
            name,
            estimator,
            x_train,
            x_test,
            y_train,
            y_test,
            test_metadata,
            feature_columns,
            split_strategy,
            labels,
        )
        for name, estimator in models.items()
    }


def _fit_supervised_model(
    name: str,
    estimator: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    test_metadata: pd.DataFrame,
    feature_columns: tuple[str, ...],
    split_strategy: str,
    labels: tuple[str, ...],
) -> ModelEvaluation:
    estimator.fit(x_train, y_train)
    predicted = estimator.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision_weighted": float(
            precision_score(y_test, predicted, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_test, predicted, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_test, predicted, average="weighted", zero_division=0)),
        "n_train": float(len(y_train)),
        "n_test": float(len(y_test)),
    }
    prediction_frame = pd.DataFrame(
        {
            "row_index": _prediction_row_index(x_test, test_metadata),
            "true_label": y_test.to_numpy(),
            "predicted_label": predicted,
        }
    )
    prediction_frame = attach_available_metadata(
        prediction_frame,
        test_metadata.reset_index(drop=True),
        SUPERVISED_METADATA_COLUMNS,
    )
    return ModelEvaluation(
        model_name=name,
        task_type="supervised_classification",
        estimator=estimator,
        feature_columns=feature_columns,
        metrics=metrics,
        predictions=PredictionTable.classification(prediction_frame),
        split_strategy=split_strategy,
        labels=labels,
        confusion_matrix=confusion_matrix(y_test, predicted, labels=list(labels)),
        feature_importances=_feature_importances(name, estimator, feature_columns),
    )


def _supervised_labeled_rows(features: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, pd.Series]:
    if label_column not in features.columns:
        raise ValueError(f"Label column not found: {label_column}")
    labeled = features.dropna(subset=[label_column]).copy()
    if labeled.empty:
        raise ValueError("No labeled rows are available for supervised modeling.")
    labeled["_row_index"] = labeled.index
    labeled = labeled.reset_index(drop=True)
    return labeled, labeled[label_column].astype(str)


def _supervised_feature_matrices(
    labeled: pd.DataFrame,
    y: pd.Series,
    train_index: pd.Index,
    test_index: pd.Index,
    *,
    feature_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, tuple[str, ...]]:
    training = labeled.loc[train_index, list(feature_columns)]
    training = training.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    training = training.dropna(axis=1, how="all")
    if training.empty:
        raise ValueError("At least one numeric feature column is required for supervised modeling.")
    feature_columns = tuple(str(column) for column in training.columns)
    testing = labeled.loc[test_index, list(feature_columns)].replace([np.inf, -np.inf], np.nan)
    x_train = training.loc[training.notna().any(axis=1)]
    x_test = testing.loc[testing.notna().any(axis=1)]
    y_train = y.loc[x_train.index]
    y_test = y.loc[x_test.index]
    if x_train.empty or x_test.empty or not _split_preserves_labels(y_train, y_test):
        raise ValueError(
            "Train/test split does not contain usable numeric feature rows for every label."
        )
    return x_train, x_test, y_train, y_test, labeled.loc[x_test.index], feature_columns


def _supervised_split(
    x: pd.DataFrame,
    y: pd.Series,
    labeled: pd.DataFrame,
    *,
    group_column: str | None,
    test_size: float,
    random_state: int,
) -> tuple[pd.Index, pd.Index, str]:
    resolved_group_column = _resolve_group_column(labeled, group_column)
    if resolved_group_column is None:
        train_index, test_index = train_test_split(
            x.index, test_size=test_size, random_state=random_state, stratify=y
        )
        return (
            pd.Index(train_index),
            pd.Index(test_index),
            f"train_test_split test_size={test_size:g}, stratified=True, random_state={random_state}",
        )
    groups = labeled[resolved_group_column].astype(str)
    _validate_group_labels(groups, y, resolved_group_column)
    splitter = GroupShuffleSplit(n_splits=50, test_size=test_size, random_state=random_state)
    for train_positions, test_positions in splitter.split(x, y, groups):
        train_index = x.index[train_positions]
        test_index = x.index[test_positions]
        if _split_preserves_labels(y.loc[train_index], y.loc[test_index]):
            return (
                train_index,
                test_index,
                (
                    f"group_shuffle_split group_column={resolved_group_column}, "
                    f"test_size={test_size:g}, random_state={random_state}"
                ),
            )
    raise ValueError(
        "Grouped train/test split could not preserve every label in both train and test sets. "
        "Use more source records per label or explicitly pass group_column=None only if row-level "
        "leakage is acceptable for the analysis."
    )


def _resolve_group_column(labeled: pd.DataFrame, group_column: str | None) -> str | None:
    if group_column is not None:
        if group_column not in labeled.columns:
            raise ValueError(f"Group column not found: {group_column}")
        return group_column
    for candidate in ("source_name", "record_name"):
        if candidate in labeled.columns and not labeled[candidate].isna().all():
            return candidate
    return None


def _validate_group_labels(groups: pd.Series, y: pd.Series, group_column: str) -> None:
    if (y.groupby(groups).nunique() > 1).any():
        raise ValueError(f"Each {group_column} group must contain only one label.")
    if (groups.groupby(y).nunique() < 2).any():
        raise ValueError(f"Each label must have at least two {group_column} groups for grouped evaluation.")


def _split_preserves_labels(y_train: pd.Series, y_test: pd.Series) -> bool:
    labels = set(y_train.astype(str).unique()) | set(y_test.astype(str).unique())
    return labels == set(y_train.astype(str).unique()) == set(y_test.astype(str).unique())


def _prediction_row_index(x_test: pd.DataFrame, test_metadata: pd.DataFrame) -> pd.Series | pd.Index:
    if "_row_index" in test_metadata.columns:
        return test_metadata["_row_index"].reset_index(drop=True)
    return x_test.index


def _feature_importances(name: str, estimator: Any, feature_columns: tuple[str, ...]) -> pd.Series | None:
    classifier = estimator.named_steps["classifier"]
    if name == "random_forest":
        return pd.Series(classifier.feature_importances_, index=feature_columns).sort_values(ascending=False)
    if name == "logistic_regression":
        coefficients = np.mean(np.abs(classifier.coef_), axis=0)
        return pd.Series(coefficients, index=feature_columns).sort_values(ascending=False)
    return None

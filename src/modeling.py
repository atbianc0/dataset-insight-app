"""Leakage-resistant modeling on raw tabular columns.

All learned feature decisions and mappings live inside ``RawFeatureTransformer``.
Because that transformer is the first step of every cross-validation pipeline,
frequency maps, imputers, encoders, and schema conversions are fitted only on
the rows available to that fold.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.contracts import AnalysisConfig, ModelBundle

MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null", "unknown", "?", "-", "--"}
HIGH_CARDINALITY_LIMIT = 80


def dtype_family(series: pd.Series) -> str:
    """Return a stable raw-schema family, including pandas extension dtypes."""

    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        return "categorical"
    return "categorical"


def _normalise_missing(series: pd.Series) -> pd.Series:
    series = pd.Series(series, index=series.index, name=series.name)
    if dtype_family(series) == "categorical":
        values = series.astype("string").str.strip()
        return values.where(~values.str.lower().isin(MISSING_TOKENS), pd.NA)
    if dtype_family(series) == "boolean":
        return series.astype("boolean")
    return series.replace([np.inf, -np.inf], np.nan)


def _name_tokens(name: str) -> list[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [token.lower() for token in re.split(r"[^A-Za-z0-9]+", spaced) if token]


def _has_identifier_token(name: str) -> bool:
    """Match id as a token, avoiding false positives such as ``paid``."""

    return "id" in _name_tokens(name)


def is_identifier_column(series: pd.Series, name: str) -> bool:
    values = _normalise_missing(series).dropna()
    if values.empty:
        return False
    unique_ratio = float(values.nunique(dropna=True) / len(values))
    if _has_identifier_token(name) and unique_ratio > 0.75:
        return True
    if dtype_family(values) == "numeric" and unique_ratio > 0.995 and len(values) > 25:
        numeric = pd.to_numeric(values, errors="coerce").sort_values()
        differences = numeric.diff().dropna()
        return bool(not differences.empty and np.isclose(differences, 1).mean() > 0.9)
    return False


def _looks_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if dtype_family(series) == "numeric":
        return False
    sample = _normalise_missing(series).dropna().head(200).astype(str)
    if sample.empty:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.85)


def _numeric_ratio(series: pd.Series) -> float:
    values = _normalise_missing(series).dropna()
    if values.empty:
        return 0.0
    return float(pd.to_numeric(values, errors="coerce").notna().mean())


def _word_counts(series: pd.Series) -> pd.Series:
    return _normalise_missing(series).fillna("").astype(str).str.split().str.len()


def _unit_parts(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    cleaned = _normalise_missing(series).astype("string")
    valid = cleaned.where(cleaned.str.match(r"^\s*[-+]?\d*\.?\d+\s*[A-Za-z]+\s*$", na=False))
    numbers = pd.to_numeric(valid.str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce")
    units = valid.str.extract(r"[-+]?\d*\.?\d+\s*([A-Za-z]+)")[0].astype("string").str.lower()
    return numbers, units


def _multi_parts(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    cleaned = _normalise_missing(series).fillna("").astype(str)
    split = cleaned.str.split(",")
    first = split.str[0].str.strip().replace("", np.nan)
    counts = split.map(lambda items: sum(bool(str(item).strip()) for item in items))
    return first, counts.astype(float)


def _series_equal(left: pd.Series, right: pd.Series) -> bool:
    paired = pd.concat([_normalise_missing(left), _normalise_missing(right)], axis=1).dropna()
    if len(paired) < max(20, int(0.5 * min(len(left), len(right)))):
        return False
    return bool((paired.iloc[:, 0].astype(str) == paired.iloc[:, 1].astype(str)).mean() >= 0.999)


def detect_leakage(
    X: pd.DataFrame,
    y: pd.Series | None,
    target_name: str | None,
    problem_type: str | None,
) -> tuple[set[str], list[str]]:
    """Find only high-confidence leakage; return columns to exclude and warnings."""

    if y is None:
        return set(), []

    y = pd.Series(y, index=X.index)
    excluded: set[str] = set()
    warnings_found: list[str] = []
    target_tokens = set(_name_tokens(target_name or ""))

    for column in X.columns:
        series = X[column]
        if _series_equal(series, y):
            excluded.add(column)
            warnings_found.append(
                f"Excluded {column}: it duplicates or directly derives the selected target."
            )
            continue

        column_tokens = set(_name_tokens(column))
        if target_tokens and target_tokens & column_tokens and column_tokens & {
            "after", "final", "outcome", "reason", "result", "status"
        }:
            excluded.add(column)
            warnings_found.append(
                f"Excluded {column}: its name suggests it may be recorded after the outcome."
            )
            continue

        if _looks_datetime(series):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(_normalise_missing(series), errors="coerce").dropna()
            if len(parsed) >= 20 and (
                parsed.is_monotonic_increasing or parsed.is_monotonic_decreasing
            ):
                warnings_found.append(
                    f"Review temporal ordering for {column}: rows are time-ordered, so a random "
                    "holdout may overstate performance compared with a future-period validation."
                )

        if problem_type == "regression" and _numeric_ratio(series) >= 0.95:
            paired = pd.concat(
                [pd.to_numeric(series, errors="coerce"), pd.to_numeric(y, errors="coerce")],
                axis=1,
            ).dropna()
            if len(paired) >= 20 and abs(float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))) >= 0.9995:
                excluded.add(column)
                warnings_found.append(
                    f"Excluded {column}: it is nearly a deterministic numeric copy of the target."
                )
                continue

        if problem_type == "classification":
            paired = pd.DataFrame(
                {"feature": _normalise_missing(series), "target": y}
            ).dropna()
            feature_levels = int(paired["feature"].nunique(dropna=True))
            target_levels = int(paired["target"].nunique(dropna=True))
            suspicious_derivation_name = bool(
                column_tokens
                & {"after", "derived", "final", "label", "outcome", "reason", "result", "target"}
            )
            if (
                len(paired) >= 20
                and target_levels >= 2
                and target_levels <= feature_levels <= min(50, max(2, len(paired) // 5))
                and suspicious_derivation_name
            ):
                table = pd.crosstab(paired["feature"].astype(str), paired["target"])
                weighted_purity = float(table.max(axis=1).sum() / table.to_numpy().sum())
                if weighted_purity >= 0.999:
                    excluded.add(column)
                    warnings_found.append(
                        f"Excluded {column}: its values almost deterministically derive the selected target."
                    )
                    continue

        # Unit labels can directly reveal a class (for example min vs Seasons -> type).
        if problem_type == "classification" and dtype_family(series) == "categorical":
            numbers, units = _unit_parts(series)
            coverage = float(numbers.notna().mean())
            if coverage >= 0.8 and 1 < units.nunique(dropna=True) <= 20:
                paired = pd.DataFrame({"unit": units, "target": y}).dropna()
                if len(paired) >= 20:
                    table = pd.crosstab(paired["unit"], paired["target"])
                    weighted_purity = float(table.max(axis=1).sum() / table.to_numpy().sum())
                    if weighted_purity >= 0.995:
                        excluded.add(column)
                        warnings_found.append(
                            f"Excluded {column}: its value units almost perfectly reveal {target_name}."
                        )

    return excluded, warnings_found


class RawFeatureTransformer(BaseEstimator, TransformerMixin):
    """Learn raw-column feature rules and apply them consistently at scoring time."""

    def __init__(
        self,
        *,
        drop_identifier_columns: bool = True,
        high_cardinality_limit: int = HIGH_CARDINALITY_LIMIT,
        target_name: str | None = None,
        problem_type: str | None = None,
    ) -> None:
        self.drop_identifier_columns = drop_identifier_columns
        self.high_cardinality_limit = high_cardinality_limit
        self.target_name = target_name
        self.problem_type = problem_type

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "RawFeatureTransformer":
        frame = self._frame(X)
        excluded_leakage, leakage_warnings = detect_leakage(
            frame, y, self.target_name, self.problem_type
        )
        self.input_columns_ = frame.columns.tolist()
        self.input_schema_ = {column: dtype_family(frame[column]) for column in frame.columns}
        self.identifier_columns_ = []
        self.dropped_columns_ = []
        self.leakage_warnings_ = leakage_warnings
        self.specs_: list[dict[str, Any]] = []

        for column in frame.columns:
            series = _normalise_missing(frame[column])
            if column in excluded_leakage:
                self.dropped_columns_.append((column, "high-confidence target leakage"))
                continue
            if series.dropna().empty:
                self.dropped_columns_.append((column, "all values missing"))
                continue
            if series.nunique(dropna=True) <= 1:
                self.dropped_columns_.append((column, "constant in training rows"))
                continue
            if self.drop_identifier_columns and is_identifier_column(series, column):
                self.identifier_columns_.append(column)
                self.dropped_columns_.append((column, "identifier-like"))
                continue

            if _looks_datetime(series):
                self.specs_.append(
                    {
                        "source": column,
                        "kind": "datetime",
                        "outputs": [
                            f"{column}__year",
                            f"{column}__month",
                            f"{column}__day",
                            f"{column}__dayofweek",
                        ],
                    }
                )
                continue

            family = dtype_family(series)
            if family == "numeric":
                self.specs_.append({"source": column, "kind": "numeric", "outputs": [column]})
                continue
            if family == "boolean":
                self.specs_.append({"source": column, "kind": "categorical", "outputs": [column]})
                continue

            if _numeric_ratio(series) >= 0.95:
                self.specs_.append({"source": column, "kind": "numeric", "outputs": [column]})
                continue

            comma_share = float(series.fillna("").astype(str).str.contains(",", regex=False).mean())
            if comma_share >= 0.25:
                first, _ = _multi_parts(series)
                first_cardinality = int(first.nunique(dropna=True))
                spec: dict[str, Any] = {
                    "source": column,
                    "kind": "multi",
                    "outputs": [f"{column}__item_count"],
                }
                if first_cardinality <= self.high_cardinality_limit:
                    spec["first_kind"] = "categorical"
                    spec["outputs"].append(f"{column}__first_item")
                else:
                    keys = first.fillna("__missing__").astype(str)
                    spec["first_kind"] = "frequency"
                    spec["frequency_map"] = keys.value_counts(normalize=True).to_dict()
                    spec["outputs"].append(f"{column}__first_item_frequency")
                self.specs_.append(spec)
                continue

            numbers, units = _unit_parts(series)
            if float(numbers.notna().mean()) >= 0.8:
                outputs = [f"{column}__number"]
                include_unit = 0 < units.nunique(dropna=True) <= self.high_cardinality_limit
                if include_unit:
                    outputs.append(f"{column}__unit")
                self.specs_.append(
                    {
                        "source": column,
                        "kind": "numeric_unit",
                        "include_unit": include_unit,
                        "outputs": outputs,
                    }
                )
                continue

            sample = series.dropna().head(200).astype(str)
            if not sample.empty and (
                float(sample.str.split().str.len().mean()) >= 4
                or float(sample.str.len().mean()) >= 30
            ):
                self.specs_.append(
                    {
                        "source": column,
                        "kind": "word_count",
                        "outputs": [f"{column}__word_count"],
                    }
                )
                continue

            cardinality = int(series.nunique(dropna=True))
            if cardinality > self.high_cardinality_limit:
                keys = series.fillna("__missing__").astype(str)
                self.specs_.append(
                    {
                        "source": column,
                        "kind": "frequency",
                        "frequency_map": keys.value_counts(normalize=True).to_dict(),
                        "outputs": [f"{column}__frequency"],
                    }
                )
            else:
                self.specs_.append({"source": column, "kind": "categorical", "outputs": [column]})

        self.required_source_columns_ = list(dict.fromkeys(spec["source"] for spec in self.specs_))
        self.output_columns_ = [output for spec in self.specs_ for output in spec["outputs"]]
        if not self.output_columns_:
            raise ValueError("No usable feature columns remained after train-only preparation.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._frame(X)
        missing = [column for column in self.required_source_columns_ if column not in frame.columns]
        if missing:
            raise ValueError("Prediction dataset is missing required columns: " + ", ".join(missing))

        output = pd.DataFrame(index=frame.index)
        for spec in self.specs_:
            column = spec["source"]
            series = _normalise_missing(frame[column])
            kind = spec["kind"]
            if kind == "numeric":
                output[spec["outputs"][0]] = pd.to_numeric(series, errors="coerce").astype(float)
            elif kind == "categorical":
                output[spec["outputs"][0]] = series.astype("string").astype(object).where(series.notna(), np.nan)
            elif kind == "datetime":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    parsed = pd.to_datetime(series, errors="coerce")
                parts = [parsed.dt.year, parsed.dt.month, parsed.dt.day, parsed.dt.dayofweek]
                for name, values in zip(spec["outputs"], parts, strict=True):
                    output[name] = values.astype(float)
            elif kind == "multi":
                first, count = _multi_parts(series)
                output[spec["outputs"][0]] = count
                if spec["first_kind"] == "categorical":
                    output[spec["outputs"][1]] = first.astype(object).where(first.notna(), np.nan)
                else:
                    keys = first.fillna("__missing__").astype(str)
                    output[spec["outputs"][1]] = keys.map(spec["frequency_map"]).fillna(0.0).astype(float)
            elif kind == "numeric_unit":
                numbers, units = _unit_parts(series)
                output[spec["outputs"][0]] = numbers.astype(float)
                if spec["include_unit"]:
                    output[spec["outputs"][1]] = units.astype(object).where(units.notna(), np.nan)
            elif kind == "word_count":
                output[spec["outputs"][0]] = _word_counts(series).astype(float)
            elif kind == "frequency":
                keys = series.fillna("__missing__").astype(str)
                output[spec["outputs"][0]] = keys.map(spec["frequency_map"]).fillna(0.0).astype(float)
        return output[self.output_columns_]

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        del input_features
        return np.asarray(self.output_columns_, dtype=object)

    @staticmethod
    def _frame(X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Raw modeling inputs must be provided as a pandas DataFrame.")
        frame = X.copy()
        frame.columns = [str(column).strip() for column in frame.columns]
        return frame


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]


def _categorical_columns(frame: pd.DataFrame) -> list[str]:
    numeric = set(_numeric_columns(frame))
    return [column for column in frame.columns if column not in numeric]


def build_model_pipeline(
    estimator: Any,
    *,
    categorical_strategy: str,
    drop_identifier_columns: bool,
    target_name: str,
    problem_type: str,
) -> Pipeline:
    if categorical_strategy == "onehot":
        categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    else:
        categorical_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                _numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", categorical_encoder),
                    ]
                ),
                _categorical_columns,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0 if categorical_strategy == "ordinal" else 0.3,
    )
    return Pipeline(
        [
            (
                "raw_features",
                RawFeatureTransformer(
                    drop_identifier_columns=drop_identifier_columns,
                    target_name=target_name,
                    problem_type=problem_type,
                ),
            ),
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def _candidate_models(problem_type: str, effort: str, random_seed: int) -> dict[str, dict[str, Any]]:
    if problem_type == "classification":
        candidates: dict[str, dict[str, Any]] = {
            "Gradient Boosting": {
                "model": HistGradientBoostingClassifier(random_state=random_seed, max_iter=120),
                "categorical_strategy": "ordinal",
            },
            "Logistic Regression": {
                "model": LogisticRegression(max_iter=1500, solver="lbfgs"),
                "categorical_strategy": "onehot",
            },
        }
        if effort == "expanded":
            candidates["Extra Trees"] = {
                "model": ExtraTreesClassifier(
                    n_estimators=180,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=random_seed,
                    n_jobs=1,
                ),
                "categorical_strategy": "ordinal",
            }
        return candidates

    candidates = {
        "Gradient Boosting": {
            "model": HistGradientBoostingRegressor(random_state=random_seed, max_iter=120),
            "categorical_strategy": "ordinal",
        },
        "Ridge Regression": {
            "model": Ridge(alpha=1.0),
            "categorical_strategy": "onehot",
        },
    }
    if effort == "expanded":
        candidates["Extra Trees"] = {
            "model": ExtraTreesRegressor(
                n_estimators=160,
                min_samples_leaf=2,
                random_state=random_seed,
                n_jobs=1,
            ),
            "categorical_strategy": "ordinal",
        }
    return candidates


def normalise_target(y: pd.Series, problem_type: str) -> pd.Series:
    series = _normalise_missing(pd.Series(y)).dropna()
    if problem_type == "regression":
        return pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if _numeric_ratio(series) >= 0.95:
        numeric = pd.to_numeric(series, errors="coerce")
        if np.allclose(numeric, np.round(numeric), equal_nan=True):
            return numeric.astype(int)
        return numeric.astype(float)
    return series.astype(str).astype(object)


def infer_positive_label(labels: list[Any], y_train: pd.Series, requested: Any | None = None) -> Any | None:
    if len(labels) != 2:
        if requested is not None:
            raise ValueError("A positive label override is only valid for binary classification.")
        return None
    if requested is not None:
        for label in labels:
            if label == requested or str(label) == str(requested):
                return label
        raise ValueError(f"Positive label {requested!r} is not present in the training target.")

    preferred = {"1", "true", "yes", "y", "positive", "churn", "churned", "failed", "fraud", "default"}
    for label in labels:
        if str(label).strip().lower() in preferred:
            return label
    counts = y_train.value_counts(dropna=False)
    return counts.idxmin()


def classification_metrics(
    y_true: pd.Series,
    y_pred: Iterable[Any],
    *,
    probabilities: np.ndarray | None = None,
    class_labels: list[Any] | None = None,
    positive_label: Any | None = None,
) -> dict[str, Any]:
    true = pd.Series(y_true).reset_index(drop=True)
    predicted = pd.Series(y_pred).reset_index(drop=True)
    labels = class_labels or list(pd.Index(pd.concat([true, predicted]).unique()))
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "precision": float(precision_score(true, predicted, average="weighted", zero_division=0)),
        "recall": float(recall_score(true, predicted, average="weighted", zero_division=0)),
        "f1": float(f1_score(true, predicted, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(true, predicted, average="weighted", zero_division=0)),
        "support": {str(label): int((true == label).sum()) for label in labels},
        "confusion_labels": labels,
        "confusion_matrix": confusion_matrix(true, predicted, labels=labels).tolist(),
    }
    if probabilities is not None and len(labels) == 2 and positive_label is not None:
        try:
            positive_index = labels.index(positive_label)
            binary_true = true.eq(positive_label).astype(int)
            positive_scores = np.asarray(probabilities)[:, positive_index]
            metrics["average_precision"] = float(average_precision_score(binary_true, positive_scores))
            metrics["roc_auc"] = float(roc_auc_score(binary_true, positive_scores))
        except (ValueError, IndexError):
            pass
    return metrics


def regression_metrics(y_true: pd.Series, y_pred: Iterable[float]) -> dict[str, float]:
    true = pd.to_numeric(pd.Series(y_true), errors="coerce")
    predicted = pd.to_numeric(pd.Series(y_pred), errors="coerce")
    return {
        "rmse": float(np.sqrt(mean_squared_error(true, predicted))),
        "mae": float(mean_absolute_error(true, predicted)),
        "r2": float(r2_score(true, predicted)),
    }


def evaluate_model_predictions(
    problem_type: str,
    y_true: pd.Series,
    y_pred: Iterable[Any],
    *,
    probabilities: np.ndarray | None = None,
    class_labels: list[Any] | None = None,
    positive_label: Any | None = None,
) -> dict[str, Any]:
    if problem_type == "classification":
        return classification_metrics(
            y_true,
            y_pred,
            probabilities=probabilities,
            class_labels=class_labels,
            positive_label=positive_label,
        )
    return regression_metrics(y_true, y_pred)


def baseline_predictions(bundle: ModelBundle, row_count: int) -> tuple[np.ndarray, np.ndarray | None]:
    strategy = bundle.baseline_strategy
    if bundle.problem_type == "classification":
        predictions = np.repeat(np.asarray([strategy["majority_label"]]), row_count)
        probabilities = None
        if len(bundle.class_labels) == 2:
            positive_rate = float(strategy["positive_rate"])
            positive_index = bundle.class_labels.index(bundle.positive_label)
            probabilities = np.empty((row_count, 2), dtype=float)
            probabilities[:, positive_index] = positive_rate
            probabilities[:, 1 - positive_index] = 1.0 - positive_rate
        return predictions, probabilities
    return np.full(row_count, float(strategy["mean"]), dtype=float), None


def _training_reference(frame: pd.DataFrame, required_columns: list[str]) -> dict[str, Any]:
    reference: dict[str, Any] = {}
    for column in required_columns:
        series = _normalise_missing(frame[column])
        family = dtype_family(series)
        payload: dict[str, Any] = {
            "family": family,
            "missing_rate": float(series.isna().mean()),
        }
        if family == "numeric" or _numeric_ratio(series) >= 0.95:
            numeric = pd.to_numeric(series, errors="coerce")
            payload.update(
                {
                    "kind": "numeric",
                    "mean": float(numeric.mean()) if numeric.notna().any() else None,
                    "std": float(numeric.std(ddof=0)) if numeric.notna().any() else None,
                }
            )
        else:
            values = series.fillna("__missing__").astype(str)
            frequencies = values.value_counts(normalize=True)
            payload.update(
                {
                    "kind": "categorical",
                    "frequencies": {str(key): float(value) for key, value in frequencies.items()},
                }
            )
        reference[column] = payload
    return reference


def _transformed_feature_names(pipeline: Pipeline) -> list[str]:
    try:
        return pipeline.named_steps["preprocessor"].get_feature_names_out().astype(str).tolist()
    except Exception:
        raw = pipeline.named_steps["raw_features"].get_feature_names_out().astype(str).tolist()
        transformed_count = int(getattr(pipeline.named_steps["model"], "n_features_in_", len(raw)))
        return raw if len(raw) == transformed_count else [f"feature_{index}" for index in range(transformed_count)]


def _raw_permutation_importance(
    pipeline: Pipeline,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    problem_type: str,
    random_seed: int,
) -> pd.DataFrame:
    if X_holdout.empty:
        return pd.DataFrame(columns=["feature", "importance", "label"])
    sample_size = min(2000, len(X_holdout))
    sample = X_holdout.sample(n=sample_size, random_state=random_seed) if len(X_holdout) > sample_size else X_holdout
    target = y_holdout.loc[sample.index]
    scoring = "balanced_accuracy" if problem_type == "classification" else "neg_root_mean_squared_error"
    try:
        measured = permutation_importance(
            pipeline,
            sample,
            target,
            scoring=scoring,
            n_repeats=3,
            random_state=random_seed,
            n_jobs=1,
        )
    except (TypeError, ValueError):
        return pd.DataFrame(columns=["feature", "importance", "label"])
    return (
        pd.DataFrame(
            {
                "feature": sample.columns,
                "importance": measured.importances_mean,
                "label": "predictive association",
            }
        )
        .sort_values("importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: AnalysisConfig,
    *,
    drop_identifier_columns: bool = True,
) -> dict[str, Any]:
    """Select with CV on training rows, then evaluate once on untouched holdout."""

    frame = pd.DataFrame(X).copy()
    raw_target = pd.Series(y, index=frame.index)
    valid = raw_target.notna()
    frame = frame.loc[valid].copy()
    raw_target = raw_target.loc[valid]
    problem_type = config.problem_type
    if problem_type == "auto":
        from src.pipeline import detect_problem_type  # local import avoids a module cycle

        problem_type = detect_problem_type(raw_target)
    target = normalise_target(raw_target, problem_type)
    frame = frame.loc[target.index].copy()
    if len(frame) < 20:
        raise ValueError("Please provide at least 20 usable target rows for modeling.")

    stratify = target if problem_type == "classification" and target.nunique() > 1 else None
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        frame,
        target,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=stratify,
    )

    if problem_type == "classification":
        minimum_class = int(y_train.value_counts().min())
        requested_folds = 3 if config.effort == "standard" else 5
        folds = min(requested_folds, minimum_class)
        if folds < 2:
            raise ValueError("Each target class needs at least two training rows for cross-validation.")
        splitter: Any = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.random_seed)
        selection_scoring = "f1_macro"
    else:
        folds = min(3 if config.effort == "standard" else 5, max(2, len(y_train) // 10))
        splitter = KFold(n_splits=folds, shuffle=True, random_state=config.random_seed)
        selection_scoring = "neg_root_mean_squared_error"

    candidates = _candidate_models(problem_type, config.effort, config.random_seed)
    fitted_specs: dict[str, Pipeline] = {}
    cv_results: dict[str, dict[str, Any]] = {}
    for name, spec in candidates.items():
        candidate = build_model_pipeline(
            clone(spec["model"]),
            categorical_strategy=spec["categorical_strategy"],
            drop_identifier_columns=drop_identifier_columns,
            target_name=config.target,
            problem_type=problem_type,
        )
        scores = cross_val_score(
            candidate,
            X_train,
            y_train,
            cv=splitter,
            scoring=selection_scoring,
            n_jobs=1,
            error_score="raise",
        )
        display_scores = -scores if problem_type == "regression" else scores
        cv_results[name] = {
            "selection_metric": "rmse" if problem_type == "regression" else "f1_macro",
            "cv_mean": float(display_scores.mean()),
            "cv_std": float(display_scores.std(ddof=0)),
            "cv_scores": [float(value) for value in display_scores],
            "folds": folds,
        }
        fitted_specs[name] = candidate

    if problem_type == "classification":
        best_name = max(cv_results, key=lambda name: cv_results[name]["cv_mean"])
    else:
        best_name = min(cv_results, key=lambda name: cv_results[name]["cv_mean"])
    best_pipeline = fitted_specs[best_name]

    class_labels: list[Any] = []
    positive_label: Any | None = None
    negative_label: Any | None = None
    threshold: float | None = None
    if problem_type == "classification":
        class_labels = pd.Index(y_train.unique()).sort_values().tolist()
        positive_label = infer_positive_label(class_labels, y_train, config.positive_label)
        if len(class_labels) == 2:
            negative_label = next(label for label in class_labels if label != positive_label)
            minority_share = float(y_train.value_counts(normalize=True).min())
            if minority_share < 0.2:
                oof_probabilities = cross_val_predict(
                    clone(best_pipeline),
                    X_train,
                    y_train,
                    cv=splitter,
                    method="predict_proba",
                    n_jobs=1,
                )
                positive_index = class_labels.index(positive_label)
                binary_true = y_train.reset_index(drop=True).eq(positive_label).astype(int)
                precision, recall, thresholds = precision_recall_curve(
                    binary_true, oof_probabilities[:, positive_index]
                )
                if len(thresholds):
                    f1_values = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)
                    threshold = float(thresholds[int(np.nanargmax(f1_values[:-1]))])

    best_pipeline.fit(X_train, y_train)
    if problem_type == "classification":
        class_labels = best_pipeline.named_steps["model"].classes_.tolist()
    probabilities = None
    if problem_type == "classification" and hasattr(best_pipeline, "predict_proba"):
        probabilities = best_pipeline.predict_proba(X_holdout)
    if threshold is not None and probabilities is not None:
        positive_index = class_labels.index(positive_label)
        holdout_pred = np.where(
            probabilities[:, positive_index] >= threshold, positive_label, negative_label
        )
    else:
        holdout_pred = best_pipeline.predict(X_holdout)

    holdout_metrics = evaluate_model_predictions(
        problem_type,
        y_holdout,
        holdout_pred,
        probabilities=probabilities,
        class_labels=class_labels,
        positive_label=positive_label,
    )

    if problem_type == "classification":
        majority_label = y_train.mode(dropna=False).iloc[0]
        positive_rate = float(y_train.eq(positive_label).mean()) if positive_label is not None else None
        baseline_strategy = {
            "kind": "training-majority",
            "majority_label": majority_label,
            "positive_rate": positive_rate,
        }
        baseline_pred = np.repeat(np.asarray([majority_label]), len(y_holdout))
        baseline_probabilities = None
        if len(class_labels) == 2:
            positive_index = class_labels.index(positive_label)
            baseline_probabilities = np.empty((len(y_holdout), 2), dtype=float)
            baseline_probabilities[:, positive_index] = positive_rate
            baseline_probabilities[:, 1 - positive_index] = 1.0 - positive_rate
        baseline_metrics = evaluate_model_predictions(
            problem_type,
            y_holdout,
            baseline_pred,
            probabilities=baseline_probabilities,
            class_labels=class_labels,
            positive_label=positive_label,
        )
        baseline_metrics["baseline_strategy"] = f"Predict training majority class ({majority_label})"
        primary_metric = "average_precision" if len(class_labels) == 2 else "f1_macro"
    else:
        training_mean = float(y_train.mean())
        baseline_strategy = {"kind": "training-mean", "mean": training_mean}
        baseline_pred = np.full(len(y_holdout), training_mean)
        baseline_metrics = evaluate_model_predictions(problem_type, y_holdout, baseline_pred)
        baseline_metrics["baseline_strategy"] = f"Predict training target mean ({training_mean:.3f})"
        primary_metric = "rmse"

    raw_transformer: RawFeatureTransformer = best_pipeline.named_steps["raw_features"]
    feature_names = _transformed_feature_names(best_pipeline)
    required_columns = raw_transformer.required_source_columns_
    identifier_columns = raw_transformer.identifier_columns_
    identifier_reference = {
        column: set(frame[column].dropna().tolist()) for column in identifier_columns
    }
    bundle = ModelBundle(
        pipeline=best_pipeline,
        target_column=config.target,
        problem_type=problem_type,
        raw_schema={column: dtype_family(frame[column]) for column in frame.columns},
        required_feature_columns=required_columns,
        optional_identifier_columns=identifier_columns,
        feature_names=feature_names,
        class_labels=class_labels,
        positive_label=positive_label,
        negative_label=negative_label,
        decision_threshold=threshold,
        baseline_strategy=baseline_strategy,
        baseline_metrics=baseline_metrics,
        holdout_metrics=holdout_metrics,
        cv_results=cv_results,
        primary_metric=primary_metric,
        training_reference=_training_reference(X_train, required_columns),
        identifier_reference=identifier_reference,
        leakage_warnings=raw_transformer.leakage_warnings_,
        training_rows=len(X_train),
        holdout_rows=len(X_holdout),
        random_seed=config.random_seed,
    )

    feature_importance = _raw_permutation_importance(
        best_pipeline, X_holdout, y_holdout, problem_type, config.random_seed
    )
    result_metrics = {
        name: {
            ("rmse" if problem_type == "regression" else "f1_macro"): values["cv_mean"],
            "cv_std": values["cv_std"],
        }
        for name, values in cv_results.items()
    }
    class_distribution = y_train.value_counts(normalize=True)
    return {
        "results": result_metrics,
        "cv_results": cv_results,
        "best_model_name": best_name,
        "best_model": bundle,
        "fitted_pipeline": best_pipeline,
        "model_bundle": bundle,
        "best_metrics": holdout_metrics,
        "baseline_metrics": baseline_metrics,
        "metric_name": primary_metric if primary_metric in holdout_metrics else ("f1" if problem_type == "classification" else "rmse"),
        "X_test": X_holdout,
        "y_test": y_holdout,
        "preds": holdout_pred,
        "best_probabilities": probabilities,
        "imbalance_ratio": float(class_distribution.min()) if problem_type == "classification" else None,
        "positive_label": positive_label,
        "decision_threshold": threshold,
        "feature_names": feature_names,
        "feature_importance": feature_importance,
        "leakage_warnings": raw_transformer.leakage_warnings_,
        "dropped_columns": raw_transformer.dropped_columns_,
        "dropped_target_rows": {"before_split": int((~valid).sum()), "train_split": 0, "test_split": 0},
    }

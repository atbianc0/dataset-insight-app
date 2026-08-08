"""Raw-file scoring, external evaluation, drift, and readiness decisions."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.contracts import ModelBundle, ScoringResult
from src.modeling import (
    baseline_predictions,
    evaluate_model_predictions,
    normalise_target,
)


def _normalised_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    original_to_normalised = {column: str(column).strip() for column in frame.columns}
    normalised = list(original_to_normalised.values())
    duplicates = pd.Index(normalised)[pd.Index(normalised).duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(
            "Scoring dataset contains duplicate columns after trimming: " + ", ".join(duplicates)
        )
    return frame.rename(columns=original_to_normalised).copy(), original_to_normalised


def _safe_probability_name(label: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(label)).strip("_").lower() or "class"
    return f"probability_{token}"


def _coerce_schema(bundle: ModelBundle, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    coerced = frame.copy()
    warnings: list[str] = []
    missing = [column for column in bundle.required_feature_columns if column not in coerced.columns]
    if missing:
        raise ValueError("Prediction dataset is missing required columns: " + ", ".join(missing))

    for column in bundle.required_feature_columns:
        reference = bundle.training_reference.get(column, {})
        if reference.get("kind") != "numeric":
            continue
        source = coerced[column]
        converted = pd.to_numeric(source, errors="coerce")
        lost = int((source.notna() & converted.isna()).sum())
        if lost:
            warnings.append(
                f"{column}: {lost:,} value(s) could not be converted to the numeric training schema and were treated as missing."
            )
        coerced[column] = converted
    return coerced, warnings


def _categorical_drift(series: pd.Series, reference: dict[str, Any]) -> dict[str, float]:
    values = series.astype("string").fillna("__missing__").astype(str)
    train_frequencies = dict(reference.get("frequencies", {}))
    known = set(train_frequencies)
    external_counts = values.value_counts(normalize=True)
    unseen_rate = float((~values.isin(known)).mean()) if known else 0.0
    train_other = max(0.0, 1.0 - sum(train_frequencies.values()))
    external_known = {key: float(external_counts.get(key, 0.0)) for key in known}
    external_other = max(0.0, 1.0 - sum(external_known.values()))
    total_variation = 0.5 * (
        sum(abs(train_frequencies[key] - external_known[key]) for key in known)
        + abs(train_other - external_other)
    )
    return {
        "total_variation_distance": float(total_variation),
        "unseen_category_rate": unseen_rate,
    }


def compare_distributions(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    target: pd.Series | None = None,
) -> dict[str, Any]:
    per_column: dict[str, dict[str, Any]] = {}
    max_smd = 0.0
    max_tv = 0.0
    max_missing_change = 0.0
    max_unseen = 0.0

    for column in bundle.required_feature_columns:
        reference = bundle.training_reference.get(column, {})
        series = frame[column]
        missing_rate = float(series.isna().mean())
        missing_change = abs(missing_rate - float(reference.get("missing_rate", 0.0)))
        payload: dict[str, Any] = {
            "training_missing_rate": float(reference.get("missing_rate", 0.0)),
            "external_missing_rate": missing_rate,
            "missingness_change": missing_change,
        }
        max_missing_change = max(max_missing_change, missing_change)
        if reference.get("kind") == "numeric":
            numeric = pd.to_numeric(series, errors="coerce")
            train_mean = reference.get("mean")
            train_std = reference.get("std")
            external_mean = float(numeric.mean()) if numeric.notna().any() else None
            if train_mean is not None and external_mean is not None:
                denominator = max(float(train_std or 0.0), 1e-9)
                smd = abs(external_mean - float(train_mean)) / denominator
            else:
                smd = 0.0
            payload.update(
                {
                    "training_mean": train_mean,
                    "external_mean": external_mean,
                    "standardized_mean_difference": float(smd),
                }
            )
            max_smd = max(max_smd, float(smd))
        else:
            categorical = _categorical_drift(series, reference)
            payload.update(categorical)
            max_tv = max(max_tv, categorical["total_variation_distance"])
            max_unseen = max(max_unseen, categorical["unseen_category_rate"])
        per_column[column] = payload

    identifier_overlap: dict[str, int] = {}
    for column, training_values in bundle.identifier_reference.items():
        if column in frame.columns:
            identifier_overlap[column] = int(frame[column].dropna().isin(training_values).sum())

    target_prevalence_change = None
    if (
        target is not None
        and bundle.problem_type == "classification"
        and bundle.positive_label is not None
        and len(target)
    ):
        external_prevalence = float(pd.Series(target).eq(bundle.positive_label).mean())
        training_prevalence = float(bundle.baseline_strategy.get("positive_rate", 0.0))
        target_prevalence_change = external_prevalence - training_prevalence

    if max_smd >= 0.5 or max_tv >= 0.2 or max_missing_change >= 0.2 or max_unseen >= 0.2:
        level = "high"
    elif max_smd >= 0.2 or max_tv >= 0.1 or max_missing_change >= 0.1 or max_unseen >= 0.05:
        level = "moderate"
    else:
        level = "low"

    return {
        "level": level,
        "per_column": per_column,
        "max_standardized_mean_difference": float(max_smd),
        "max_total_variation_distance": float(max_tv),
        "max_missingness_change": float(max_missing_change),
        "max_unseen_category_rate": float(max_unseen),
        "target_prevalence_change": target_prevalence_change,
        "identifier_overlap": identifier_overlap,
        "identifier_overlap_total": int(sum(identifier_overlap.values())),
    }


def _readiness(
    bundle: ModelBundle,
    external_metrics: dict[str, Any] | None,
    external_baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    if external_metrics is None or external_baseline is None:
        return {
            "status": "provisional",
            "summary": "Internal holdout results are provisional until a labeled external dataset is evaluated.",
        }

    if bundle.problem_type == "classification":
        metric = bundle.primary_metric
        if metric not in external_metrics or metric not in bundle.holdout_metrics:
            metric = "balanced_accuracy"
        internal_value = float(bundle.holdout_metrics[metric])
        external_value = float(external_metrics[metric])
        baseline_value = float(external_baseline[metric])
        performance_drop = internal_value - external_value
        close_to_baseline = external_value <= baseline_value + 0.05
    else:
        metric = "r2"
        internal_value = float(bundle.holdout_metrics[metric])
        external_value = float(external_metrics[metric])
        baseline_value = float(external_baseline[metric])
        performance_drop = internal_value - external_value
        close_to_baseline = external_value <= baseline_value + 0.05

    if performance_drop >= 0.10 or close_to_baseline:
        return {
            "status": "not deployment-ready",
            "summary": (
                f"External {metric} is {external_value:.3f} versus {internal_value:.3f} internally "
                f"and {baseline_value:.3f} for the train-derived baseline. Generalization is not reliable enough for deployment."
            ),
            "metric": metric,
            "internal_value": internal_value,
            "external_value": external_value,
            "external_baseline_value": baseline_value,
            "performance_drop": performance_drop,
        }
    return {
        "status": "externally validated",
        "summary": (
            f"External {metric} is {external_value:.3f}; it remains meaningfully above the "
            "train-derived baseline without a material internal-to-external drop."
        ),
        "metric": metric,
        "internal_value": internal_value,
        "external_value": external_value,
        "external_baseline_value": baseline_value,
        "performance_drop": performance_drop,
    }


def score_or_evaluate(bundle: ModelBundle, raw_df: pd.DataFrame) -> ScoringResult:
    """Score raw rows and evaluate automatically when the target is present."""

    if not isinstance(bundle, ModelBundle):
        raise TypeError("score_or_evaluate requires a fitted ModelBundle.")
    if not isinstance(raw_df, pd.DataFrame):
        raise TypeError("The scoring input must be a pandas DataFrame.")
    if raw_df.empty:
        raise ValueError("The scoring dataset contains no rows.")

    original = raw_df.copy()
    working, _ = _normalised_columns(raw_df)
    working, schema_warnings = _coerce_schema(bundle, working)
    allowed = set(bundle.required_feature_columns + bundle.optional_identifier_columns + [bundle.target_column])
    extras = [column for column in working.columns if column not in allowed]
    if extras:
        schema_warnings.append(
            "Ignored extra scoring columns not used by the model: " + ", ".join(extras[:12])
        )

    predictions = bundle.predict(working)
    scored = original.copy()
    prediction_column = "prediction" if "prediction" not in scored.columns else "model_prediction"
    scored[prediction_column] = predictions

    probabilities = None
    if bundle.problem_type == "classification" and hasattr(bundle.pipeline, "predict_proba"):
        probabilities = bundle.predict_proba(working)
        for index, label in enumerate(bundle.class_labels):
            name = _safe_probability_name(label)
            if name in scored.columns:
                name = "model_" + name
            scored[name] = probabilities[:, index]

    external_metrics = None
    external_baseline = None
    valid_target = None
    if bundle.target_column in working.columns:
        raw_target = working[bundle.target_column]
        valid_mask = raw_target.notna()
        if valid_mask.any():
            normalised = normalise_target(raw_target.loc[valid_mask], bundle.problem_type)
            valid_index = normalised.index
            valid_target = normalised
            prediction_series = pd.Series(predictions, index=working.index).loc[valid_index]
            probability_subset = None
            if probabilities is not None:
                probability_subset = probabilities[working.index.get_indexer(valid_index)]
            external_metrics = evaluate_model_predictions(
                bundle.problem_type,
                valid_target,
                prediction_series,
                probabilities=probability_subset,
                class_labels=bundle.class_labels,
                positive_label=bundle.positive_label,
            )
            baseline_pred, baseline_probabilities = baseline_predictions(bundle, len(valid_target))
            external_baseline = evaluate_model_predictions(
                bundle.problem_type,
                valid_target,
                baseline_pred,
                probabilities=baseline_probabilities,
                class_labels=bundle.class_labels,
                positive_label=bundle.positive_label,
            )
            external_metrics["evaluated_rows"] = int(len(valid_target))
            external_metrics["baseline_metrics"] = external_baseline

    drift = compare_distributions(bundle, working, valid_target)
    for column, payload in drift["per_column"].items():
        unseen_rate = float(payload.get("unseen_category_rate", 0.0))
        if unseen_rate > 0:
            schema_warnings.append(
                f"{column}: {unseen_rate:.1%} of rows contain categories not observed in "
                "the model-training rows; they were handled without refitting."
            )
    readiness = _readiness(bundle, external_metrics, external_baseline)
    return ScoringResult(
        scored_rows=scored,
        schema_warnings=schema_warnings,
        drift_summary=drift,
        external_metrics=external_metrics,
        readiness=readiness,
    )

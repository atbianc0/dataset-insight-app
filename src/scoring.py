"""Raw-file scoring, external evaluation, drift, and readiness decisions."""

from __future__ import annotations

import math
import re
from numbers import Integral
from typing import Any

import pandas as pd

from src.contracts import ModelBundle, ScoringResult
from src.modeling import (
    baseline_predictions,
    evaluate_model_predictions,
    normalise_target,
)

MIN_EXTERNAL_EVALUATION_ROWS = 20
MIN_EXTERNAL_CLASS_SUPPORT = 20
LOWER_IS_BETTER_METRICS = {
    "mae",
    "mean_absolute_error",
    "mean_squared_error",
    "mse",
    "rmse",
    "root_mean_squared_error",
}


def _normalised_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    original_columns = list(frame.columns)
    normalised = [str(column).strip() for column in original_columns]
    duplicates = pd.Index(normalised)[pd.Index(normalised).duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(
            "Scoring dataset contains duplicate columns after trimming: " + ", ".join(duplicates)
        )
    normalized_frame = frame.copy()
    normalized_frame.columns = normalised
    return normalized_frame, dict(zip(original_columns, normalised, strict=True))


def _safe_probability_name(label: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(label)).strip("_").lower() or "class"
    return f"probability_{token}"


def _collision_free_output_name(base: str, occupied: set[Any]) -> str:
    """Return a deterministic model-output name without replacing an input column."""

    if base not in occupied:
        occupied.add(base)
        return base

    model_base = f"model_{base}"
    if model_base not in occupied:
        occupied.add(model_base)
        return model_base

    suffix = 2
    while f"{model_base}_{suffix}" in occupied:
        suffix += 1
    name = f"{model_base}_{suffix}"
    occupied.add(name)
    return name


def _labels_equal(left: Any, right: Any) -> bool:
    try:
        result = left == right
        if result is pd.NA or pd.isna(result):
            return False
        return bool(result)
    except (TypeError, ValueError):
        return False


def _canonical_class_label(value: Any, class_labels: list[Any]) -> Any | None:
    exact = [label for label in class_labels if _labels_equal(value, label)]
    if len(exact) == 1:
        return exact[0]

    value_text = str(value).strip()
    text_matches = [label for label in class_labels if value_text == str(label).strip()]
    if len(text_matches) == 1:
        return text_matches[0]

    folded_matches = [
        label for label in class_labels if value_text.casefold() == str(label).strip().casefold()
    ]
    return folded_matches[0] if len(folded_matches) == 1 else None


def _normalise_external_target(
    bundle: ModelBundle,
    raw_target: pd.Series,
) -> tuple[pd.Series | None, str | None]:
    """Normalize a complete external target or explain why evaluation is unsafe."""

    provided = raw_target.loc[raw_target.notna()]
    if provided.empty:
        return None, f"{bundle.target_column}: no non-missing target labels are available."

    try:
        normalised = normalise_target(provided, bundle.problem_type)
    except (TypeError, ValueError, OverflowError) as exc:
        return None, f"{bundle.target_column}: target values could not be normalized ({exc})."

    omitted_count = len(provided) - len(normalised)
    if omitted_count:
        return None, (
            f"{bundle.target_column}: {omitted_count:,} provided target value(s) became missing or "
            "invalid during normalization; external evaluation was not run."
        )

    if bundle.problem_type == "regression":
        return normalised, None
    if not bundle.class_labels:
        return None, (
            f"{bundle.target_column}: the model bundle has no training labels, so external "
            "evaluation was not run."
        )

    canonical_values: list[Any] = []
    unknown_values: list[Any] = []
    for value in normalised.tolist():
        canonical = _canonical_class_label(value, bundle.class_labels)
        if canonical is None:
            unknown_values.append(value)
        else:
            canonical_values.append(canonical)

    if unknown_values:
        examples = list(dict.fromkeys(str(value) for value in unknown_values))[:5]
        return None, (
            f"{bundle.target_column}: {len(unknown_values):,} target value(s) use labels not "
            f"present in the model's training labels ({', '.join(examples)}); external "
            "evaluation was not run."
        )

    return pd.Series(canonical_values, index=normalised.index, name=raw_target.name), None


def _external_target_warning(bundle: ModelBundle, target: pd.Series) -> str | None:
    if len(target) < MIN_EXTERNAL_EVALUATION_ROWS:
        return (
            f"{bundle.target_column}: external evaluation requires at least "
            f"{MIN_EXTERNAL_EVALUATION_ROWS:,} labeled rows; found {len(target):,}. "
            "Predictions were produced, but readiness remains provisional."
        )
    if bundle.problem_type == "classification" and target.nunique(dropna=False) < 2:
        return (
            f"{bundle.target_column}: the external target contains only one class. At least two "
            "observed classes are required for comparative classification metrics, so readiness "
            "remains provisional."
        )
    if bundle.problem_type == "classification":
        observed = target.unique().tolist()
        missing_labels = [
            label
            for label in bundle.class_labels
            if not any(_labels_equal(label, value) for value in observed)
        ]
        if missing_labels:
            missing_text = ", ".join(str(label) for label in missing_labels[:5])
            return (
                f"{bundle.target_column}: the external target has zero support for fitted "
                f"class(es): {missing_text}. Every fitted class needs external support before "
                "comparative classification metrics can establish readiness."
            )
        support = {
            label: int(target.map(lambda value: _labels_equal(label, value)).sum())
            for label in bundle.class_labels
        }
        low_support = {
            label: count
            for label, count in support.items()
            if count < MIN_EXTERNAL_CLASS_SUPPORT
        }
        if low_support:
            rendered = ", ".join(
                f"{label}={count:,}" for label, count in list(low_support.items())[:5]
            )
            return (
                f"{bundle.target_column}: stable external estimates require at least "
                f"{MIN_EXTERNAL_CLASS_SUPPORT:,} observations per fitted class; low support "
                f"was found for {rendered}. Predictions were produced, but uncertainty is too "
                "high for an external validation claim."
            )
    return None


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
    identifier_overlap_rate: dict[str, float] = {}
    for column, training_values in bundle.identifier_reference.items():
        if column in frame.columns:
            overlap = int(frame[column].dropna().isin(training_values).sum())
            identifier_overlap[column] = overlap
            identifier_overlap_rate[column] = float(overlap / max(len(frame), 1))

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
        "identifier_overlap_rate": identifier_overlap_rate,
        "max_identifier_overlap_rate": float(max(identifier_overlap_rate.values(), default=0.0)),
        "identifier_overlap_total": int(sum(identifier_overlap.values())),
    }


def _readiness(
    bundle: ModelBundle,
    external_metrics: dict[str, Any] | None,
    external_baseline: dict[str, Any] | None,
    drift_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if external_metrics is None or external_baseline is None:
        return {
            "status": "provisional",
            "summary": "Internal holdout results are provisional until a labeled external dataset is evaluated.",
        }

    evaluated_rows = external_metrics.get("evaluated_rows")
    if not isinstance(evaluated_rows, Integral) or evaluated_rows < MIN_EXTERNAL_EVALUATION_ROWS:
        return {
            "status": "provisional",
            "summary": (
                f"At least {MIN_EXTERNAL_EVALUATION_ROWS:,} labeled external rows are required "
                "before making a validation claim."
            ),
        }

    metric = bundle.primary_metric
    if (
        not metric
        or metric not in external_metrics
        or metric not in external_baseline
        or metric not in bundle.holdout_metrics
    ):
        return {
            "status": "provisional",
            "summary": (
                f"The configured primary metric ({metric or 'unspecified'}) is unavailable or "
                "incomplete, so no external validation claim was made."
            ),
        }

    try:
        internal_value = float(bundle.holdout_metrics[metric])
        external_value = float(external_metrics[metric])
        baseline_value = float(external_baseline[metric])
    except (TypeError, ValueError):
        return {
            "status": "provisional",
            "summary": (
                f"The configured primary metric ({metric}) is not numeric, so no external "
                "validation claim was made."
            ),
        }

    if not all(math.isfinite(value) for value in (internal_value, external_value, baseline_value)):
        return {
            "status": "provisional",
            "summary": (
                f"The configured primary metric ({metric}) contains a non-finite value, so no "
                "external validation claim was made."
            ),
        }

    lower_is_better = metric.lower() in LOWER_IS_BETTER_METRICS
    if lower_is_better:
        performance_drop = external_value - internal_value
        close_to_baseline = external_value >= baseline_value - 0.05
        comparison = "below"
    else:
        performance_drop = internal_value - external_value
        close_to_baseline = external_value <= baseline_value + 0.05
        comparison = "above"

    overlap_total = int((drift_summary or {}).get("identifier_overlap_total", 0) or 0)
    if performance_drop >= 0.10 or close_to_baseline or overlap_total:
        if performance_drop >= 0.10 or close_to_baseline:
            summary = (
                f"External {metric} is {external_value:.3f} versus {internal_value:.3f} "
                f"internally and {baseline_value:.3f} for the train-derived baseline. "
                "Generalization is not reliable enough for deployment."
            )
        else:
            summary = (
                f"External {metric} is {external_value:.3f}, but external rows overlap "
                f"model-development entities ({overlap_total:,} matched identifier row(s)); "
                "this file cannot establish independent validation."
            )
        payload = {
            "status": "not deployment-ready",
            "summary": summary,
            "metric": metric,
            "internal_value": internal_value,
            "external_value": external_value,
            "external_baseline_value": baseline_value,
            "performance_drop": performance_drop,
            "metric_direction": "lower is better" if lower_is_better else "higher is better",
        }
        if overlap_total:
            payload["identifier_overlap_total"] = overlap_total
        return payload
    return {
        "status": "externally validated",
        "summary": (
            f"External {metric} is {external_value:.3f}; it remains meaningfully {comparison} the "
            "train-derived baseline without a material internal-to-external drop."
        ),
        "metric": metric,
        "internal_value": internal_value,
        "external_value": external_value,
        "external_baseline_value": baseline_value,
        "performance_drop": performance_drop,
        "metric_direction": "lower is better" if lower_is_better else "higher is better",
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
    positional_input = raw_df.reset_index(drop=True)
    working, _ = _normalised_columns(positional_input)
    working, schema_warnings = _coerce_schema(bundle, working)
    allowed = set(bundle.required_feature_columns + bundle.optional_identifier_columns + [bundle.target_column])
    extras = [column for column in working.columns if column not in allowed]
    if extras:
        schema_warnings.append(
            "Ignored extra scoring columns not used by the model: " + ", ".join(extras[:12])
        )

    predictions = bundle.predict(working)
    scored = original.copy()
    occupied_columns = set(scored.columns)
    prediction_column = _collision_free_output_name("prediction", occupied_columns)
    scored[prediction_column] = predictions

    probabilities = None
    probability_columns: dict[str, str] = {}
    if bundle.problem_type == "classification" and hasattr(bundle.pipeline, "predict_proba"):
        probabilities = bundle.predict_proba(working)
        for index, label in enumerate(bundle.class_labels):
            name = _collision_free_output_name(_safe_probability_name(label), occupied_columns)
            scored[name] = probabilities[:, index]
            probability_columns[str(label)] = name

    external_metrics = None
    external_baseline = None
    valid_target = None
    evaluation_status = "not_requested"
    evaluation_warning = None
    if bundle.target_column in working.columns:
        raw_target = working[bundle.target_column]
        valid_target, evaluation_warning = _normalise_external_target(bundle, raw_target)
        if evaluation_warning is not None:
            evaluation_status = "blocked"
            schema_warnings.append(evaluation_warning)
        elif valid_target is not None:
            evaluation_warning = _external_target_warning(bundle, valid_target)
            if evaluation_warning is not None:
                evaluation_status = "blocked"
                schema_warnings.append(evaluation_warning)
            else:
                evaluation_status = "completed"
                valid_index = valid_target.index
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
                baseline_pred, baseline_probabilities = baseline_predictions(
                    bundle, len(valid_target)
                )
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
    for column, overlap in drift["identifier_overlap"].items():
        if overlap:
            rate = float(drift["identifier_overlap_rate"].get(column, 0.0))
            schema_warnings.append(
                f"{column}: {overlap:,} row(s) ({rate:.1%}) overlap model-development "
                "entities. External metrics are not independent validation evidence."
            )
    readiness = _readiness(bundle, external_metrics, external_baseline, drift)
    if evaluation_status == "blocked":
        readiness = {
            "status": "provisional",
            "summary": (
                "External evaluation was blocked because the supplied target could not support "
                "a safe comparison with the fitted model. Predictions were produced, but no "
                "external validation claim was made."
            ),
        }
    return ScoringResult(
        scored_rows=scored,
        schema_warnings=schema_warnings,
        drift_summary=drift,
        external_metrics=external_metrics,
        readiness=readiness,
        evaluation_status=evaluation_status,
        prediction_column=prediction_column,
        probability_columns=probability_columns,
    )

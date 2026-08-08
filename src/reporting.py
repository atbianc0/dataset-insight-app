from dataclasses import fields, is_dataclass
from numbers import Real
from typing import Any, Mapping, Optional

import pandas as pd

from src.ai_assistant import extract_stage_payload

APP_VERSION = "1.0.0"


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return a shallow mapping for dictionaries and the public data contracts."""

    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    if hasattr(value, "to_dict") and not isinstance(value, pd.DataFrame):
        converted = value.to_dict()
        return converted if isinstance(converted, Mapping) else {}
    return {}


def _first_value(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _first_mapping(*values: Any) -> Mapping[str, Any]:
    for value in values:
        mapped = _mapping(value)
        if mapped:
            return mapped
    return {}


def _bullets(items: Any) -> list[str]:
    if items is None:
        return []
    if isinstance(items, str):
        items = [items]
    return [f"- {item}" for item in items if item]


def _metric_lines(metrics: Any) -> list[str]:
    lines = []
    for name, value in _mapping(metrics).items():
        if isinstance(value, bool) or not isinstance(value, Real):
            continue
        label = name.replace("_", " ").title()
        if isinstance(value, int) and (name.endswith("rows") or name.endswith("support")):
            lines.append(f"- {label}: {value:,}")
        else:
            lines.append(f"- {label}: {float(value):.4f}")
    return lines


def _support_lines(metrics: Any) -> list[str]:
    values = _mapping(metrics)
    support = dict(_mapping(values.get("support")))
    for container_name in ("per_class", "classification_report"):
        for label, raw_evidence in _mapping(values.get(container_name)).items():
            evidence = _mapping(raw_evidence)
            if label not in support and isinstance(evidence.get("support"), Real):
                support[label] = evidence["support"]
    lines = []
    for label, count in support.items():
        if isinstance(count, Real) and not isinstance(count, bool):
            escaped_label = str(label).replace("`", "\\`")
            lines.append(f"- Class `{escaped_label}`: {int(count):,}")
    return lines


def _insight_lines(items: Any) -> list[str]:
    lines = []
    for item in items or []:
        if not isinstance(item, Mapping):
            lines.append(str(item))
            continue
        title = _first_value(item.get("title"), item.get("headline"), item.get("name"))
        detail = _first_value(
            item.get("summary"), item.get("finding"), item.get("description"), item.get("text")
        )
        if title and detail and str(title) != str(detail):
            lines.append(f"{title}: {detail}")
        elif title or detail:
            lines.append(str(title or detail))
    return lines


def _strategy_text(strategy: Any) -> str | None:
    if isinstance(strategy, str):
        return strategy
    values = _mapping(strategy)
    if not values:
        return None
    parts = []
    for name, value in values.items():
        if isinstance(value, Real) and not isinstance(value, bool):
            rendered = f"{float(value):.4f}"
        else:
            rendered = str(value)
        parts.append(f"{name.replace('_', ' ')}={rendered}")
    return ", ".join(parts)


def _cross_validation_lines(cv_results: Any, default_metric: str | None) -> list[str]:
    lines = []
    for model_name, raw_evidence in _mapping(cv_results).items():
        evidence = _mapping(raw_evidence)
        mean = evidence.get("cv_mean")
        std = evidence.get("cv_std")
        folds = evidence.get("folds")
        if not isinstance(mean, Real):
            continue
        metric = evidence.get("selection_metric") or default_metric or "selection metric"
        text = f"- {model_name}: {metric} mean {float(mean):.4f}"
        if isinstance(std, Real):
            text += f", std {float(std):.4f}"
        if isinstance(folds, Real):
            text += f", {int(folds)} folds"
        lines.append(text)
    return lines


def _feature_schema_lines(bundle: Mapping[str, Any]) -> list[str]:
    required = list(bundle.get("required_feature_columns") or [])
    identifiers = list(bundle.get("optional_identifier_columns") or [])
    fitted_features = list(bundle.get("feature_names") or [])
    raw_schema = _mapping(bundle.get("raw_schema"))
    if not required and not identifiers and not fitted_features and not raw_schema:
        return []

    def code(value: Any) -> str:
        escaped_value = str(value).replace("`", "\\`")
        return f"`{escaped_value}`"

    lines = [
        "- Method: imputation, categorical encoding/frequency mapping, numeric-string and unit extraction, datetime expansion, and text-derived transformations were fitted on training rows only and reused unchanged for holdout and external rows."
    ]
    if required:
        rendered = ", ".join(
            f"{code(column)} ({raw_schema.get(column, 'unknown')})" for column in required
        )
        lines.append(f"- Required raw model features ({len(required):,}): {rendered}")
    if identifiers:
        lines.append(
            f"- Preserved identifier columns excluded from model fitting ({len(identifiers):,}): "
            + ", ".join(code(column) for column in identifiers)
        )
    if fitted_features:
        lines.append(
            f"- Fitted transformation outputs ({len(fitted_features):,}): "
            + ", ".join(code(feature) for feature in fitted_features)
        )
    leakage_warnings = bundle.get("leakage_warnings") or []
    if leakage_warnings:
        lines.append("- Leakage screening exclusions/warnings:")
        lines.extend(f"  - {warning}" for warning in leakage_warnings)
    return lines


def _drift_lines(drift: Mapping[str, Any]) -> list[str]:
    lines = []
    if drift.get("level"):
        lines.append(f"- Overall level: {drift['level']}")
    maxima = (
        ("max_standardized_mean_difference", "Maximum numeric SMD"),
        ("max_total_variation_distance", "Maximum categorical TVD"),
        ("max_missingness_change", "Maximum missingness change"),
        ("max_unseen_category_rate", "Maximum unseen-category rate"),
    )
    for key, label in maxima:
        value = drift.get(key)
        if isinstance(value, Real):
            lines.append(f"- {label}: {float(value):.4f}")
    prevalence_change = drift.get("target_prevalence_change")
    if isinstance(prevalence_change, Real):
        lines.append(f"- Target-prevalence change: {float(prevalence_change):+.4f}")

    overlap = _mapping(drift.get("identifier_overlap"))
    if overlap:
        rendered = ", ".join(f"{name}={int(count):,}" for name, count in overlap.items())
        total = drift.get("identifier_overlap_total")
        suffix = f" (total {int(total):,})" if isinstance(total, Real) else ""
        lines.append(f"- Identifier overlap: {rendered}{suffix}")

    per_column = _mapping(drift.get("per_column"))
    if per_column:
        lines.extend(["", "#### Per-column drift"])
        for column, raw_evidence in per_column.items():
            evidence = _mapping(raw_evidence)
            parts = []
            smd = evidence.get("standardized_mean_difference")
            tvd = evidence.get("total_variation_distance")
            missing = evidence.get("missingness_change")
            unseen = evidence.get("unseen_category_rate")
            if isinstance(smd, Real):
                parts.append(f"numeric SMD={float(smd):.4f}")
            if isinstance(tvd, Real):
                parts.append(f"categorical TVD={float(tvd):.4f}")
            if isinstance(missing, Real):
                parts.append(f"missingness change={float(missing):.4f}")
            if isinstance(unseen, Real):
                parts.append(f"unseen-category rate={float(unseen):.4f}")
            if parts:
                lines.append(f"- {column}: {', '.join(parts)}")
    return lines


def build_report_markdown(
    result: Mapping[str, Any] | Any,
    dataset_name: str,
    scoring_result: Optional[Any] = None,
    app_version: str = APP_VERSION,
) -> str:
    """Build a portable report from either typed contracts or legacy payloads."""

    result_data = _mapping(result)
    workflow = _mapping(result_data.get("dataset_recommendation"))
    decision = _first_mapping(result_data.get("decision"), result_data.get("workflow_decision"))
    internal = _mapping(result_data.get("internal_validation"))
    predictive_attempt = _mapping(result_data.get("predictive_attempt"))
    bundle = _first_mapping(
        result_data.get("model_bundle"),
        internal.get("model_bundle"),
        predictive_attempt.get("model_bundle"),
    )
    insight_analysis = _mapping(result_data.get("insight_analysis"))
    overview = _mapping(insight_analysis.get("overview"))
    target_assessment = _mapping(result_data.get("target_assessment"))
    quality = _first_mapping(
        result_data.get("quality"), internal.get("quality"), predictive_attempt.get("quality")
    )
    config = _first_mapping(
        result_data.get("analysis_config"),
        internal.get("analysis_config"),
        predictive_attempt.get("analysis_config"),
        result_data.get("config"),
        internal.get("config"),
        predictive_attempt.get("config"),
    )

    scoring = _mapping(scoring_result)
    external_metrics = _first_mapping(
        scoring.get("external_metrics"), scoring.get("metrics"), result_data.get("external_metrics")
    )
    drift = _first_mapping(scoring.get("drift_summary"), result_data.get("drift_summary"))

    selected_target = _first_value(
        result_data.get("selected_target"), config.get("target"), bundle.get("target_column")
    )
    mode = _first_value(result_data.get("mode"), decision.get("selected_mode"))
    if mode is None:
        mode = "prediction" if bundle or internal else "analysis"
    row_count = _first_value(overview.get("rows"), result_data.get("original_rows"))
    if row_count is None and bundle:
        row_count = int(bundle.get("training_rows", 0)) + int(bundle.get("holdout_rows", 0))
    column_count = _first_value(overview.get("columns"), result_data.get("original_columns"))
    if column_count is None:
        column_count = len(_mapping(bundle.get("raw_schema")))

    task_ai = extract_stage_payload(result_data.get("assistant_extensions"), "task_understanding")
    report_ai = extract_stage_payload(result_data.get("assistant_extensions"), "report_generation")

    lines = [
        "# DataLens Analysis Report",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- DataLens version: `{app_version}`",
        f"- Workflow: `{mode}`",
        f"- Selected target: `{selected_target or 'None'}`",
        f"- Rows: `{int(row_count or 0):,}`",
        f"- Columns: `{int(column_count or 0):,}`",
        "",
        "## Decision",
        decision.get("summary") or workflow.get("summary") or "No workflow summary was available.",
    ]
    lines.extend(_bullets(decision.get("details", [])))

    if target_assessment:
        lines.extend(
            [
                "",
                "## Target Review",
                f"- Problem type: {target_assessment.get('problem_type', 'unknown')}",
                f"- Usable target rows: {int(target_assessment.get('usable_rows', 0)):,}",
                f"- Unique target values: {int(target_assessment.get('unique_count', 0)):,}",
                f"- Usable raw feature columns: {int(target_assessment.get('usable_feature_count', 0)):,}",
            ]
        )
        lines.extend(_bullets(target_assessment.get("blockers", [])))
        lines.extend(_bullets(target_assessment.get("reasons_against_prediction", [])))

    headlines = insight_analysis.get("headlines", [])
    if not headlines:
        headlines = _insight_lines(result_data.get("ranked_insights", []))
    if headlines:
        lines.extend(["", "## Key Insights"])
        lines.extend(_bullets(headlines))

    quality_notes = insight_analysis.get("data_quality_summary", [])
    if quality_notes:
        lines.extend(["", "## Data Quality"])
        lines.extend(_bullets(quality_notes))

    responsible_use = insight_analysis.get("responsible_use_notes", [])
    if responsible_use:
        lines.extend(["", "## Responsible Use"])
        lines.extend(_bullets(responsible_use))

    if mode == "prediction" or bundle or internal or predictive_attempt:
        best_metrics = _first_mapping(
            result_data.get("best_metrics"),
            internal.get("best_metrics"),
            internal.get("holdout_metrics"),
            internal.get("metrics"),
            predictive_attempt.get("best_metrics"),
            predictive_attempt.get("holdout_metrics"),
            bundle.get("holdout_metrics"),
        )
        baseline = _first_mapping(
            result_data.get("baseline_metrics"),
            internal.get("baseline_metrics"),
            predictive_attempt.get("baseline_metrics"),
            bundle.get("baseline_metrics"),
        )
        cv_results = _first_mapping(
            result_data.get("cv_results"),
            internal.get("cv_results"),
            predictive_attempt.get("cv_results"),
            bundle.get("cv_results"),
        )
        primary_metric = _first_value(
            result_data.get("metric_name"),
            internal.get("primary_metric"),
            predictive_attempt.get("primary_metric"),
            bundle.get("primary_metric"),
        )
        positive_label = _first_value(
            result_data.get("positive_label"),
            config.get("positive_label"),
            predictive_attempt.get("positive_label"),
            bundle.get("positive_label"),
        )
        if predictive_attempt:
            used_rows = predictive_attempt.get("used_rows")
            if used_rows is None and bundle:
                used_rows = int(bundle.get("training_rows", 0)) + int(bundle.get("holdout_rows", 0))
        else:
            used_rows = _first_value(result_data.get("used_rows"), internal.get("used_rows"))
        if used_rows is None and bundle:
            used_rows = int(bundle.get("training_rows", 0)) + int(bundle.get("holdout_rows", 0))
        validation_strategy = _first_value(
            config.get("validation_strategy"),
            result_data.get("validation_strategy"),
            internal.get("validation_strategy"),
            predictive_attempt.get("validation_strategy"),
            result_data.get("split_strategy"),
            internal.get("split_strategy"),
            predictive_attempt.get("split_strategy"),
        )
        if validation_strategy is None and bundle:
            validation_strategy = "holdout_cv"
        effort = _first_value(
            config.get("effort"),
            result_data.get("training_effort"),
            internal.get("effort"),
            predictive_attempt.get("effort"),
        )
        test_size = _first_value(
            config.get("test_size"),
            result_data.get("test_size"),
            internal.get("test_size"),
            predictive_attempt.get("test_size"),
        )
        if test_size is None and bundle:
            training_rows = int(bundle.get("training_rows", 0))
            holdout_rows = int(bundle.get("holdout_rows", 0))
            if training_rows + holdout_rows:
                test_size = holdout_rows / (training_rows + holdout_rows)
        random_seed = _first_value(
            config.get("random_seed"),
            result_data.get("random_seed"),
            result_data.get("random_state"),
            internal.get("random_seed"),
            predictive_attempt.get("random_seed"),
            bundle.get("random_seed"),
        )
        internal_summary = _first_value(
            quality.get("summary"),
            internal.get("conclusion"),
            internal.get("summary"),
            predictive_attempt.get("conclusion"),
            predictive_attempt.get("summary"),
        )
        internal_status = _first_value(
            result_data.get("validation_status"),
            internal.get("validation_status"),
            internal.get("status"),
            predictive_attempt.get("validation_status"),
            quality.get("verdict"),
            "provisional",
        )

        lines.extend(
            [
                "",
                "## Internal Model Validation",
                f"- Model: {_first_value(result_data.get('best_model_name'), internal.get('best_model_name'), internal.get('model_name'), predictive_attempt.get('best_model_name'), predictive_attempt.get('model_name'), 'Unknown')}",
                f"- Status: {internal_status}",
                f"- Positive label: {positive_label if positive_label is not None else 'not applicable'}",
                f"- Rows used for bounded modeling: {int(used_rows or 0):,}",
            ]
        )
        if validation_strategy is not None:
            lines.append(f"- Validation strategy: {validation_strategy}")
        if effort is not None:
            lines.append(f"- Effort: {effort}")
        if isinstance(test_size, Real):
            lines.append(f"- Holdout fraction: {float(test_size):.3f}")
        if random_seed is not None:
            lines.append(f"- Random seed: {random_seed}")
        if internal_summary:
            lines.append(f"- Internal conclusion: {internal_summary}")
        else:
            lines.append(
                "- Internal conclusion: Internal holdout evidence is provisional until a separate representative dataset is evaluated."
            )
        lines.extend(_metric_lines(best_metrics))
        support_lines = _support_lines(best_metrics)
        if support_lines:
            lines.extend(["", "### Per-class holdout support"])
            lines.extend(support_lines)

        if baseline or bundle.get("baseline_strategy"):
            lines.extend(["", "### Baseline"])
            strategy = _first_value(baseline.get("baseline_strategy"), bundle.get("baseline_strategy"))
            rendered_strategy = _strategy_text(strategy)
            if rendered_strategy:
                lines.append(f"- Strategy: {rendered_strategy}")
            lines.extend(_metric_lines(baseline))

        cv_lines = _cross_validation_lines(cv_results, str(primary_metric) if primary_metric else None)
        if cv_lines:
            lines.extend(["", "### Cross-validation model selection"])
            lines.extend(cv_lines)
        else:
            cv_summary = _mapping(result_data.get("cross_validation"))
            if cv_summary:
                lines.extend(["", "### Model-selection evidence"])
                lines.extend(_metric_lines(cv_summary))
        feature_schema_lines = _feature_schema_lines(bundle)
        if feature_schema_lines:
            lines.extend(["", "### Fitted feature and transformation schema"])
            lines.extend(feature_schema_lines)
        lines.append(
            "Internal holdout evidence is provisional until a separate representative dataset is evaluated."
        )

    if scoring:
        lines.extend(["", "## External Validation or Scoring"])
        evaluation_status = scoring.get("evaluation_status")
        if evaluation_status == "blocked":
            scoring_mode = "evaluation blocked"
        elif evaluation_status == "completed":
            scoring_mode = "evaluation"
        else:
            scoring_mode = scoring.get("mode") or ("evaluation" if external_metrics else "scoring")
        lines.append(f"- Mode: {scoring_mode}")
        if evaluation_status:
            lines.append(f"- Evaluation status: {evaluation_status}")
        raw_readiness = scoring.get("readiness")
        readiness = _mapping(raw_readiness)
        readiness_status = readiness.get("status") or (
            raw_readiness if isinstance(raw_readiness, str) else None
        )
        readiness_summary = readiness.get("summary") or scoring.get("summary")
        if readiness_status:
            lines.append(f"- Readiness: {readiness_status}")
        if readiness_status or readiness_summary:
            conclusion = str(readiness_status or "undetermined")
            if readiness_summary:
                conclusion += f" — {readiness_summary}"
            lines.append(f"- External conclusion: {conclusion}")
        lines.extend(_metric_lines(external_metrics))
        external_positive_label = _first_value(
            bundle.get("positive_label"), result_data.get("positive_label")
        )
        external_support = _mapping(external_metrics.get("support"))
        evaluated_rows = external_metrics.get("evaluated_rows")
        positive_support = (
            external_support.get(str(external_positive_label))
            if external_positive_label is not None
            else None
        )
        if isinstance(positive_support, Real) and isinstance(evaluated_rows, Real) and evaluated_rows:
            prevalence = float(positive_support) / float(evaluated_rows)
            lines.append(
                f"- External positive-label prevalence (`{external_positive_label}`): "
                f"{int(positive_support):,} of {int(evaluated_rows):,} ({prevalence:.2%})"
            )
        external_support_lines = _support_lines(external_metrics)
        if external_support_lines:
            lines.extend(["", "### External per-class support"])
            lines.extend(external_support_lines)

        schema_warnings = scoring.get("schema_warnings") or []
        if schema_warnings:
            lines.extend(["", "### Schema and compatibility warnings"])
            lines.extend(_bullets(schema_warnings))

        if drift:
            lines.extend(["", "### Distribution shift"])
            rendered_drift = _drift_lines(drift)
            if rendered_drift:
                lines.extend(rendered_drift)
            else:
                drift_items = drift.get("warnings") or drift.get("top_shifts") or []
                lines.extend(_bullets([str(item) for item in drift_items]))

    ai_summary = task_ai.get("ai_dataset_summary") or report_ai.get("ai_report_summary")
    if ai_summary:
        lines.extend(
            [
                "",
                "## Optional AI Interpretation",
                str(ai_summary),
                "AI interpretation is advisory; deterministic checks and validation results remain authoritative.",
            ]
        )

    notes = list(dict.fromkeys(result_data.get("notes", [])))
    if notes:
        lines.extend(["", "## Preparation and Method Notes"])
        lines.extend(_bullets(notes[:20]))

    lines.extend(
        [
            "",
            "## Limitations",
            "- Associations and model importance do not establish causation.",
            "- Heuristic target recommendations require domain review.",
            "- Performance on an internal holdout does not guarantee real-world generalization.",
            "- Sensitive or protected attributes require legal, ethical, and domain-specific review before use.",
        ]
    )
    return "\n".join(lines)

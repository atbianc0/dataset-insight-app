from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Optional

import pandas as pd

from src.ai_assistant import extract_stage_payload

APP_VERSION = "1.0.0"


def _mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict") and not isinstance(value, pd.DataFrame):
        converted = value.to_dict()
        return converted if isinstance(converted, Mapping) else {}
    return {}


def _bullets(items):
    return [f"- {item}" for item in items if item]


def _metric_lines(metrics):
    lines = []
    for name, value in _mapping(metrics).items():
        if isinstance(value, (int, float)):
            lines.append(f"- {name.replace('_', ' ').title()}: {value:.4f}")
    return lines


def build_report_markdown(
    result: Mapping[str, Any],
    dataset_name: str,
    scoring_result: Optional[Any] = None,
    app_version: str = APP_VERSION,
) -> str:
    """Build a portable report without depending on Streamlit rendering state."""
    workflow = _mapping(result.get("dataset_recommendation"))
    insight_analysis = _mapping(result.get("insight_analysis"))
    overview = _mapping(insight_analysis.get("overview"))
    decision = _mapping(result.get("decision"))
    quality = _mapping(result.get("quality"))
    target_assessment = _mapping(result.get("target_assessment"))
    scoring = _mapping(scoring_result)
    external_metrics = _mapping(
        scoring.get("external_metrics")
        or scoring.get("metrics")
        or result.get("external_metrics")
    )
    drift = scoring.get("drift_summary") or result.get("drift_summary") or []

    task_ai = extract_stage_payload(result.get("assistant_extensions"), "task_understanding")
    report_ai = extract_stage_payload(result.get("assistant_extensions"), "report_generation")

    lines = [
        "# DataLens Analysis Report",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- DataLens version: `{app_version}`",
        f"- Workflow: `{result.get('mode', 'analysis')}`",
        f"- Selected target: `{result.get('selected_target') or 'None'}`",
        f"- Rows: `{int(overview.get('rows', result.get('original_rows', 0))):,}`",
        f"- Columns: `{int(overview.get('columns', result.get('original_columns', 0))):,}`",
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

    if result.get("mode") == "prediction":
        lines.extend(
            [
                "",
                "## Internal Model Validation",
                f"- Model: {result.get('best_model_name', 'Unknown')}",
                f"- Status: {quality.get('verdict', 'provisional')}",
                f"- Positive label: {result.get('positive_label', 'not applicable')}",
                f"- Rows used for bounded modeling: {int(result.get('used_rows', 0)):,}",
                f"- Split strategy: {result.get('split_strategy', 'random holdout with training-only model selection')}",
                f"- Random seed: {result.get('random_state', 42)}",
            ]
        )
        if quality.get("summary"):
            lines.append(quality["summary"])
        lines.extend(_metric_lines(result.get("best_metrics")))

        baseline = _mapping(result.get("baseline_metrics"))
        if baseline:
            lines.extend(["", "### Baseline"])
            if baseline.get("baseline_strategy"):
                lines.append(f"- Strategy: {baseline['baseline_strategy']}")
            lines.extend(_metric_lines(baseline))

        cv_summary = _mapping(result.get("cross_validation"))
        if cv_summary:
            lines.extend(["", "### Model-selection evidence"])
            lines.extend(_metric_lines(cv_summary))
        lines.append(
            "Internal holdout evidence is provisional until a separate representative dataset is evaluated."
        )

    if scoring:
        lines.extend(["", "## External Validation or Scoring"])
        lines.append(
            f"- Mode: {scoring.get('mode', 'evaluation' if external_metrics else 'scoring')}"
        )
        raw_readiness = scoring.get("readiness")
        readiness = _mapping(raw_readiness)
        if readiness.get("status"):
            lines.append(f"- Readiness: {readiness['status']}")
        elif raw_readiness:
            lines.append(f"- Readiness: {raw_readiness}")
        if readiness.get("summary"):
            lines.append(str(readiness["summary"]))
        elif scoring.get("summary"):
            lines.append(str(scoring["summary"]))
        lines.extend(_metric_lines(external_metrics))
        if drift:
            lines.extend(["", "### Distribution shift"])
            if isinstance(drift, Mapping):
                drift_items = drift.get("warnings") or drift.get("top_shifts") or []
            else:
                drift_items = drift
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

    notes = list(dict.fromkeys(result.get("notes", [])))
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

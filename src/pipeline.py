import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.contracts import AnalysisConfig, DatasetProfile
from src.extensions import DEFAULT_EXTENSION_REGISTRY, AnalysisContext
from src.heuristics import (
    classify_general_column_role as _classify_general_column_role,
)
from src.heuristics import (
    is_identifier_like as _is_identifier_like,
)
from src.heuristics import (
    is_integer_like,
    normalize_missing_tokens,
    numeric_conversion_ratio,
    safe_ratio,
)
from src.heuristics import (
    is_text_heavy as _is_text_heavy_target,
)
from src.heuristics import (
    tokenize_column_name as _tokenize_column_name,
)
from src.insights import run_insight_analysis
from src.modeling import train_model

MAX_PREVIEW_ROWS = 25
MAX_CHART_ROWS = 5000
BENCHMARK_ROWS = 20000
MAX_TRAIN_ROWS = int(os.getenv("DATALENS_MODEL_SAMPLE_ROWS", "60000"))
HIGH_CARDINALITY_LIMIT = 80
MIN_PREDICTION_ROWS = 30
POSITIVE_TARGET_KEYWORDS = {
    "amount",
    "approved",
    "churn",
    "class",
    "conversion",
    "default",
    "defect",
    "demand",
    "failed",
    "fraud",
    "grade",
    "label",
    "loss",
    "outcome",
    "passed",
    "price",
    "probability",
    "result",
    "retained",
    "revenue",
    "risk",
    "sales",
    "score",
    "segment",
    "status",
    "success",
    "survived",
    "target",
    "value",
}
HIGH_CONFIDENCE_AUTO_TARGET_KEYWORDS = {
    "attrition",
    "churn",
    "converted",
    "conversion",
    "default",
    "fraud",
    "label",
    "outcome",
    "response",
    "target",
}
NEGATIVE_TARGET_KEYWORDS = {
    "address",
    "cast",
    "comment",
    "customer",
    "description",
    "director",
    "email",
    "first",
    "last",
    "latitude",
    "longitude",
    "message",
    "name",
    "notes",
    "phone",
    "text",
    "title",
    "url",
    "uuid",
}
GROUP_TOKEN_STOPWORDS = {
    "actual",
    "class",
    "label",
    "level",
    "result",
    "score",
    "status",
    "target",
    "type",
    "value",
}
TEMPORAL_TARGET_KEYWORDS = {
    "created",
    "date",
    "day",
    "month",
    "released",
    "time",
    "timestamp",
    "updated",
    "year",
}


def _collect_assistant_extensions(
    extension_registry,
    df,
    target_col=None,
    artifacts=None,
    metadata=None,
):
    registry = extension_registry or DEFAULT_EXTENSION_REGISTRY
    context = AnalysisContext(
        df=df,
        target_col=target_col,
        artifacts=artifacts or {},
        metadata=metadata or {},
    )
    return registry.collect(context)


def sanitize_dataframe(df):
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    duplicates = (
        pd.Index(cleaned.columns)[pd.Index(cleaned.columns).duplicated()].unique().tolist()
    )
    if duplicates:
        raise ValueError(
            "Column names must be unique after trimming whitespace. Duplicate columns: "
            + ", ".join(duplicates[:5])
        )
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    for column in cleaned.columns:
        series = normalize_missing_tokens(cleaned[column])
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            cleaned[column] = series
            numeric_candidate = pd.to_numeric(series, errors="coerce")
            if numeric_candidate.notna().mean() >= 0.95 and numeric_candidate.notna().sum() > 0:
                cleaned[column] = numeric_candidate
        else:
            cleaned[column] = series

    return cleaned


def sanitize_target_series(series):
    target = normalize_missing_tokens(series).replace([np.inf, -np.inf], np.nan)

    if pd.api.types.is_object_dtype(target) or pd.api.types.is_string_dtype(target):
        target = target.astype("string").str.strip()
        numeric_candidate = pd.to_numeric(target, errors="coerce")
        if numeric_candidate.notna().mean() >= 0.95 and numeric_candidate.notna().sum() > 0:
            target = numeric_candidate

    return target


def filter_valid_target_rows(X, y, label, min_rows=20):
    cleaned_y = sanitize_target_series(pd.Series(y))
    valid_mask = ~cleaned_y.isna()
    dropped_count = int((~valid_mask).sum())

    if X is None:
        filtered_X = None
    else:
        filtered_X = pd.DataFrame(X).loc[valid_mask].copy()

    filtered_y = cleaned_y.loc[valid_mask].copy()

    if len(filtered_y) < min_rows:
        raise ValueError(
            f"After cleaning invalid target values ({label}), only {len(filtered_y)} usable rows remained. "
            "Please choose a cleaner target column or upload more complete data."
        )

    return filtered_X, filtered_y, dropped_count


def detect_problem_type(y):
    if (
        pd.api.types.is_object_dtype(y)
        or pd.api.types.is_string_dtype(y)
        or isinstance(y.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(y)
    ):
        if numeric_conversion_ratio(y) >= 0.95:
            y = pd.to_numeric(y, errors="coerce")
        else:
            return "classification"

    if pd.api.types.is_bool_dtype(y):
        return "classification"

    unique_count = y.nunique(dropna=False)
    unique_ratio = unique_count / max(len(y), 1)
    if is_integer_like(y) and unique_count <= 5 and unique_ratio <= 0.05:
        return "classification"
    return "regression"


def _categorical_column_names(frame):
    return [
        column
        for column in frame.columns
        if (
            pd.api.types.is_object_dtype(frame[column])
            or pd.api.types.is_string_dtype(frame[column])
            or isinstance(frame[column].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(frame[column])
        )
    ]


def summarize_target_style(y, problem_type):
    unique_count = int(y.nunique(dropna=False))
    if problem_type == "classification":
        label = "categorical classification"
    elif is_integer_like(y) and y.min() >= 0:
        label = "count-style regression"
    else:
        label = "numeric regression"
    return {"label": label, "unique_count": unique_count}


def _describe_target_shape(series, problem_type):
    unique_count = int(series.nunique(dropna=True))
    if problem_type == "classification":
        if unique_count == 2:
            return "Binary label"
        return "Categorical outcome"
    if problem_type == "regression":
        if is_integer_like(series):
            return "Numeric count / amount"
        return "Numeric outcome"
    return "Unclear target shape"

def _build_feature_subset_summary(df):
    likely_useful = []
    risky = []
    avoid = []

    for column in df.columns:
        series = df[column]
        role = _classify_general_column_role(series, column)
        missing_ratio = float(series.isna().mean())
        unique_count = int(series.nunique(dropna=True))

        if series.isna().all() or unique_count <= 1:
            avoid.append(
                {
                    "column": column,
                    "role": "Empty / constant",
                    "guidance": "Exclude from modeling",
                    "reason": "All values are missing or constant, so the column cannot add signal.",
                }
            )
        elif role == "identifier":
            avoid.append(
                {
                    "column": column,
                    "role": "Identifier-like",
                    "guidance": "Exclude from modeling",
                    "reason": "Identifier-style values usually memorize rows instead of generalizing.",
                }
            )
        elif role == "text":
            avoid.append(
                {
                    "column": column,
                    "role": "Long text",
                    "guidance": "Summarize instead",
                    "reason": "Long-form text is better summarized than directly modeled in this app.",
                }
            )
        elif missing_ratio >= 0.45:
            risky.append(
                {
                    "column": column,
                    "role": role.replace("_", " ").title(),
                    "guidance": "Use carefully",
                    "reason": f"High missingness ({missing_ratio:.1%}) could weaken both analysis and prediction.",
                }
            )
        elif role == "datetime":
            likely_useful.append(
                {
                    "column": column,
                    "role": "Date / time",
                    "guidance": "Keep for trends or derived features",
                    "reason": "Date-like columns can support trend summaries or derived time parts.",
                }
            )
        elif role in {"numeric", "boolean"}:
            likely_useful.append(
                {
                    "column": column,
                    "role": role.replace("_", " ").title(),
                    "guidance": "Strong general-purpose input",
                    "reason": "Structured numeric values are useful for summaries, correlations, and models.",
                }
            )
        elif role == "categorical":
            likely_useful.append(
                {
                    "column": column,
                    "role": "Categorical",
                    "guidance": "Usually useful",
                    "reason": "Low or medium-cardinality categories often help grouping and classification.",
                }
            )
        else:
            risky.append(
                {
                    "column": column,
                    "role": "High-cardinality category",
                    "guidance": "Check before modeling",
                    "reason": "High-cardinality categories may need encoding and can be noisy.",
                }
            )

    return {
        "likely_useful": likely_useful[:12],
        "risky": risky[:12],
        "avoid": avoid[:12],
        "counts": {
            "likely_useful": len(likely_useful),
            "risky": len(risky),
            "avoid": len(avoid),
        },
    }


def _detect_multi_target_groups(candidates):
    viable = [
        candidate for candidate in candidates
        if candidate["status"] in {"recommended", "possible"} and candidate["problem_type"] in {"classification", "regression"}
    ]
    groups = {}

    def add_group(label, columns, problem_type, reason):
        key = tuple(sorted(columns))
        if len(key) < 2:
            return
        average_score = round(
            float(np.mean([candidate["score"] for candidate in viable if candidate["column"] in key])),
            2,
        )
        existing = groups.get(key)
        payload = {
            "group_label": label,
            "columns": list(key),
            "problem_type": problem_type,
            "reason": reason,
            "average_score": average_score,
        }
        if existing is None or payload["average_score"] > existing["average_score"]:
            groups[key] = payload

    prefixes = {}
    suffixes = {}
    token_groups = {}

    for candidate in viable:
        tokens = _tokenize_column_name(candidate["column"])
        if not tokens:
            continue
        prefix = tokens[0]
        suffix = tokens[-1]
        if len(prefix) > 2:
            prefixes.setdefault((candidate["problem_type"], prefix), []).append(candidate["column"])
        if len(suffix) > 2:
            suffixes.setdefault((candidate["problem_type"], suffix), []).append(candidate["column"])
        for token in tokens:
            if len(token) > 2 and token not in GROUP_TOKEN_STOPWORDS:
                token_groups.setdefault((candidate["problem_type"], token), []).append(candidate["column"])

    for (problem_type, token), columns in prefixes.items():
        if len(columns) >= 2:
            add_group(
                f"Shared prefix: {token}",
                columns,
                problem_type,
                f"These columns share the '{token}' prefix and may represent related outcomes worth modeling together.",
            )
    for (problem_type, token), columns in suffixes.items():
        if len(columns) >= 2:
            add_group(
                f"Shared suffix: {token}",
                columns,
                problem_type,
                f"These columns share the '{token}' outcome style and may support multi-target prediction.",
            )
    for (problem_type, token), columns in token_groups.items():
        if len(columns) >= 2:
            add_group(
                f"Shared token: {token}",
                columns,
                problem_type,
                f"These targets all reference '{token}', which suggests a related family of outcomes.",
            )

    frame = pd.DataFrame(groups.values())
    if frame.empty:
        return []
    frame = frame.sort_values(["average_score", "group_label"], ascending=[False, True])
    return frame.to_dict(orient="records")


def _build_training_column_metadata(df):
    """Profile modeling suitability once instead of once per possible target."""

    metadata = {}
    for column in df.columns:
        series = df[column]
        unique_count = int(series.nunique(dropna=True))
        if series.isna().all():
            status, reason = "drop", "all values missing"
        elif unique_count <= 1:
            status, reason = "drop", "constant values"
        elif _is_identifier_like(series, column):
            status, reason = "drop", "identifier-like"
        elif _is_text_heavy_target(series):
            status, reason = "derived", "long text converted to a length feature"
        else:
            status, reason = "usable", "supported raw feature"
        metadata[column] = {
            "status": status,
            "reason": reason,
            "role": _classify_general_column_role(series, column),
        }
    return metadata


def evaluate_target_candidate(
    df,
    column,
    drop_identifier_columns=True,
    column_metadata=None,
):
    if column not in df.columns:
        raise ValueError(f"Target column '{column}' was not found.")

    raw_series = df[column]
    series = sanitize_target_series(raw_series)
    non_null = series.dropna()
    usable_rows = int(len(non_null))
    missing_ratio = 1 - safe_ratio(usable_rows, len(df))
    unique_count = int(non_null.nunique(dropna=True))
    unique_ratio = safe_ratio(unique_count, usable_rows)
    name_tokens = set(_tokenize_column_name(column))
    positive_keywords = sorted(name_tokens & POSITIVE_TARGET_KEYWORDS)
    negative_keywords = sorted(name_tokens & NEGATIVE_TARGET_KEYWORDS)
    temporal_keywords = sorted(name_tokens & TEMPORAL_TARGET_KEYWORDS)
    problem_type = detect_problem_type(non_null) if usable_rows else None
    role = _classify_general_column_role(non_null if usable_rows else raw_series, column)
    pros = []
    cautions = []
    blockers = []
    score = 0.0
    usable_feature_count = 0
    prepared_columns = []
    rejected_feature_columns = []

    if positive_keywords:
        pros.append(
            "Column name suggests an outcome field"
            + f" ({', '.join(positive_keywords[:3])})."
        )
        score += 2.5

    if negative_keywords:
        cautions.append(
            "Column name looks more descriptive than outcome-focused"
            + f" ({', '.join(negative_keywords[:3])})."
        )
        score -= 1.0

    if temporal_keywords and not positive_keywords:
        blockers.append(
            "Column name looks time-based"
            + f" ({', '.join(temporal_keywords[:3])}), which is usually better for trend analysis than prediction."
        )
        score -= 2.0

    if usable_rows == 0:
        blockers.append("No usable values remain after cleaning.")
    elif usable_rows < MIN_PREDICTION_ROWS:
        blockers.append(f"Only {usable_rows} usable rows are available for this target.")
    elif usable_rows < 80:
        cautions.append(f"Only {usable_rows} usable rows remain, so prediction may be unstable.")
        score += 0.5
    else:
        pros.append(f"{usable_rows} usable rows remain after cleaning.")
        score += 1.5

    if missing_ratio >= 0.45:
        blockers.append(f"Missingness is too high ({missing_ratio:.1%}).")
    elif missing_ratio >= 0.2:
        cautions.append(f"Missingness is noticeable ({missing_ratio:.1%}).")
        score -= 0.5
    else:
        pros.append("Coverage is reasonably complete.")
        score += 1.0

    if unique_count <= 1:
        blockers.append("Only one distinct target value remains after cleaning.")

    if role == "identifier":
        blockers.append("Looks identifier-like or nearly unique, so it is not a meaningful prediction target.")
        score -= 4.0
    elif role == "datetime":
        blockers.append("Looks like a timestamp or date field, which is usually better as a trend axis or feature.")
        score -= 2.5
    elif role == "text":
        blockers.append("Looks like free text or long-form text, which is better for summarization than direct prediction here.")
        score -= 3.0

    if problem_type == "classification":
        class_count = unique_count
        min_class_count = int(non_null.value_counts(dropna=False).min()) if usable_rows else 0
        if class_count == 2:
            pros.append("Binary label structure is a strong fit for classification.")
            score += 2.5
        elif 3 <= class_count <= 12:
            pros.append(f"{class_count} classes is a practical classification range.")
            score += 2.0
        elif class_count <= 30 and unique_ratio <= 0.25:
            cautions.append(f"{class_count} classes may still work, but the classification task is getting fragmented.")
            score += 0.5
        elif class_count > min(40, max(10, usable_rows // 3)):
            blockers.append(f"{class_count} classes is too fragmented for a practical classification workflow here.")
            score -= 2.0

        if usable_rows and min_class_count < 3:
            blockers.append("At least one class has fewer than 3 rows.")
    elif problem_type == "regression":
        if pd.api.types.is_numeric_dtype(non_null) or numeric_conversion_ratio(non_null) >= 0.95:
            score += 1.5
        if unique_count >= 15:
            pros.append("Target behaves like a continuous or count-style numeric outcome.")
            score += 1.5
        elif unique_count < 8:
            cautions.append("Very few distinct numeric values are present, so descriptive analysis may be just as useful.")
            score -= 0.5

    if unique_ratio > 0.98 and unique_count > 50 and role not in {"numeric"}:
        blockers.append("Values are almost entirely unique, which is not realistic for supervised prediction.")

    needs_deep_assessment = not blockers
    if needs_deep_assessment:
        metadata = column_metadata or _build_training_column_metadata(df)
        assessment = assess_target_for_prediction(
            df,
            column,
            drop_identifier_columns=drop_identifier_columns,
            column_metadata=metadata,
            prepare_features=False,
            already_sanitized=True,
        )
        pros.extend([reason for reason in assessment["reasons_for_prediction"] if reason not in pros])
        cautions.extend([reason for reason in assessment["reasons_against_prediction"] if reason not in cautions])
        blockers.extend([reason for reason in assessment["blockers"] if reason not in blockers])
        usable_feature_count = int(assessment["usable_feature_count"])

        if not blockers and usable_feature_count >= 4:
            score += 1.5
        elif usable_feature_count >= 2:
            score += 0.75
        elif usable_feature_count == 1:
            cautions.append("Only one usable feature column remains after preparation.")

        prepared_columns = [
            name
            for name, payload in metadata.items()
            if name != column and payload["status"] != "drop"
        ][:12]
        rejected_feature_columns = [
            name
            for name, payload in metadata.items()
            if name != column and payload["status"] == "drop"
        ][:12]

    if blockers:
        status = "rejected"
        recommended_use = "Insights / conclusions"
    elif score >= 5.0:
        status = "recommended"
        recommended_use = problem_type or "prediction"
    else:
        status = "possible"
        recommended_use = problem_type or "manual review"

    summary = (
        f"{column} is a {status} target candidate for "
        f"{(problem_type or 'analysis').replace('_', ' ')}."
    )
    if status == "rejected":
        summary = f"{column} is not a practical prediction target for this dataset."

    return {
        "column": column,
        "status": status,
        "score": round(score, 2),
        "problem_type": problem_type,
        "target_shape": _describe_target_shape(non_null, problem_type) if usable_rows else "No usable values",
        "recommended_use": recommended_use,
        "summary": summary,
        "pros": pros[:5],
        "cautions": cautions[:5],
        "blockers": blockers[:5],
        "usable_rows": usable_rows,
        "missing_pct": round(missing_ratio * 100, 1),
        "unique_count": unique_count,
        "unique_ratio": round(unique_ratio, 3),
        "usable_feature_count": usable_feature_count,
        "suggested_feature_subset": prepared_columns,
        "rejected_feature_columns": rejected_feature_columns,
        "positive_name_signals": positive_keywords,
        "negative_name_signals": negative_keywords,
    }


def recommend_dataset_workflow(
    df,
    drop_identifier_columns=True,
    top_n=8,
    extension_registry=None,
):
    sanitized = sanitize_dataframe(df)
    column_metadata = _build_training_column_metadata(sanitized)
    candidates = [
        evaluate_target_candidate(
            sanitized,
            column,
            drop_identifier_columns=drop_identifier_columns,
            column_metadata=column_metadata,
        )
        for column in sanitized.columns
    ]

    candidates = sorted(
        candidates,
        key=lambda candidate: (
            {"recommended": 2, "possible": 1, "rejected": 0}[candidate["status"]],
            candidate["score"],
            candidate["column"].lower(),
        ),
        reverse=True,
    )
    recommended_candidates = [candidate for candidate in candidates if candidate["status"] == "recommended"]
    possible_candidates = [candidate for candidate in candidates if candidate["status"] == "possible"]
    rejected_candidates = [candidate for candidate in candidates if candidate["status"] == "rejected"]
    visible_candidates = (recommended_candidates + possible_candidates)[:top_n]
    primary_candidate = visible_candidates[0] if visible_candidates else None
    credible_recommended_candidates = [
        candidate
        for candidate in recommended_candidates
        if set(candidate.get("positive_name_signals", []))
        & HIGH_CONFIDENCE_AUTO_TARGET_KEYWORDS
    ]
    strong_primary = (
        credible_recommended_candidates[0] if credible_recommended_candidates else None
    )
    second_candidate = visible_candidates[1] if len(visible_candidates) > 1 else None
    multi_target_candidates = _detect_multi_target_groups(recommended_candidates + possible_candidates)
    feature_subset_summary = _build_feature_subset_summary(sanitized)

    if strong_primary is None:
        recommended_workflow = "insights"
        recommended_task_type = "descriptive analysis / conclusions"
        if primary_candidate is None:
            summary = "No strong prediction target stands out, so insight-focused analysis is the safest recommendation."
        else:
            recommended_task_type = primary_candidate["problem_type"] or recommended_task_type
            summary = (
                f"No strong prediction target stands out. '{primary_candidate['column']}' is only a tentative option, "
                "so insight-focused analysis is still the safer default."
            )
    else:
        recommended_workflow = "prediction"
        recommended_task_type = strong_primary["problem_type"] or "prediction"
        clear_primary = (
            second_candidate is None
            or strong_primary["score"] - second_candidate["score"] >= 1.25
        )
        if clear_primary:
            summary = (
                f"The dataset appears suitable for {recommended_task_type}, and "
                f"'{strong_primary['column']}' is the strongest default target."
            )
        else:
            summary = (
                f"The dataset can support prediction, but the best target is not completely obvious. "
                f"'{strong_primary['column']}' is the strongest current option."
            )

    if multi_target_candidates:
        best_group = multi_target_candidates[0]
        if recommended_workflow == "prediction":
            summary += (
                f" It may also support multi-target {best_group['problem_type']} across "
                f"{', '.join(best_group['columns'][:3])}."
            )

    task_recommendations = []
    if primary_candidate is not None:
        task_recommendations.append(
            {
                "task_type": primary_candidate["problem_type"].title() if primary_candidate["problem_type"] else "Prediction",
                "fit": "Best fit" if primary_candidate["status"] == "recommended" else "Possible but weak",
                "targets": primary_candidate["column"],
                "reason": primary_candidate["summary"],
            }
        )
    if len(visible_candidates) >= 2:
        task_recommendations.append(
            {
                "task_type": "Multiple possible targets",
                "fit": "Possible",
                "targets": ", ".join([candidate["column"] for candidate in visible_candidates[:3]]),
                "reason": "Several columns look plausible, so the user goal should guide the final target choice.",
            }
        )
    if multi_target_candidates:
        top_group = multi_target_candidates[0]
        task_recommendations.append(
            {
                "task_type": "Multi-target prediction",
                "fit": "Possible",
                "targets": ", ".join(top_group["columns"]),
                "reason": top_group["reason"],
            }
        )

    insight_analysis = run_insight_analysis(sanitized)
    best_analysis_path = (
        insight_analysis["analysis_recommendations"].iloc[0].to_dict()
        if not insight_analysis["analysis_recommendations"].empty
        else None
    )
    for row in insight_analysis["analysis_recommendations"].head(4).to_dict(orient="records"):
        task_recommendations.append(
            {
                "task_type": row["analysis_type"],
                "fit": "Useful companion" if primary_candidate is not None else "Best fit",
                "targets": "",
                "reason": row["reason"],
            }
        )

    workflow_result = {
        "summary": summary,
        "recommended_workflow": recommended_workflow,
        "recommended_task_type": recommended_task_type,
        "recommended_primary_target": strong_primary["column"] if strong_primary else None,
        "recommended_target_columns": [candidate["column"] for candidate in recommended_candidates[:top_n]],
        "candidate_targets": visible_candidates,
        "rejected_target_candidates": rejected_candidates[:top_n],
        "multi_target_candidates": multi_target_candidates[:5],
        "task_recommendations": task_recommendations[:8],
        "feature_subset_summary": feature_subset_summary,
        "insight_analysis": insight_analysis,
        "best_analysis_path": best_analysis_path,
        "candidate_lookup": {candidate["column"]: candidate for candidate in candidates},
        "clear_primary_target": bool(
            strong_primary is not None
            and (
                second_candidate is None
                or strong_primary["score"] - second_candidate["score"] >= 1.25
            )
        ),
    }

    workflow_result["assistant_extensions"] = _collect_assistant_extensions(
        extension_registry,
        sanitized,
        artifacts={
            "workflow": workflow_result,
            "insight_analysis": insight_analysis,
            "column_inspection": insight_analysis.get("column_inspection"),
            "feature_subset_summary": feature_subset_summary,
        },
    )

    return workflow_result


def recommend_target_columns(df, top_n=5, extension_registry=None):
    workflow = recommend_dataset_workflow(
        df,
        drop_identifier_columns=True,
        top_n=top_n,
        extension_registry=extension_registry,
    )
    rows = []
    for candidate in workflow["candidate_targets"][:top_n]:
        rows.append(
            {
                "column": candidate["column"],
                "score": candidate["score"],
                "workflow_fit": candidate["status"].title(),
                "suggested_task": candidate["problem_type"] or "Insights",
                "reasons": "; ".join((candidate["pros"] + candidate["cautions"])[:3]) or candidate["summary"],
            }
        )
    return pd.DataFrame(rows)


def assess_target_for_prediction(
    df,
    target_col,
    drop_identifier_columns=True,
    column_metadata=None,
    prepare_features=True,
    already_sanitized=False,
):
    sanitized = df if already_sanitized else sanitize_dataframe(df)
    if target_col not in sanitized.columns:
        raise ValueError(f"Target column '{target_col}' was not found.")

    target = sanitize_target_series(sanitized[target_col])
    non_null_target = target.dropna()
    usable_rows = int(len(non_null_target))
    missing_ratio = 1 - safe_ratio(usable_rows, len(sanitized))
    unique_count = int(non_null_target.nunique(dropna=True))
    unique_ratio = safe_ratio(unique_count, usable_rows)
    problem_type = detect_problem_type(non_null_target) if usable_rows else None

    reasons_for = []
    reasons_against = []
    blockers = []
    usable_feature_count = 0

    if usable_rows == 0:
        blockers.append("The selected target has no usable values after cleaning.")
    elif usable_rows < MIN_PREDICTION_ROWS:
        blockers.append(
            f"Only {usable_rows} usable target rows remain after cleaning, which is too small for a reliable model."
        )
    elif usable_rows < 80:
        reasons_against.append(
            f"Only {usable_rows} usable target rows remain, so predictive results may be unstable."
        )
    else:
        reasons_for.append(f"There are {usable_rows} usable rows for supervised modeling.")

    if missing_ratio >= 0.45:
        blockers.append(
            f"The target is {missing_ratio:.1%} missing, which is too incomplete for a practical model."
        )
    elif missing_ratio >= 0.2:
        reasons_against.append(
            f"The target is {missing_ratio:.1%} missing, which weakens training coverage."
        )
    else:
        reasons_for.append("Target coverage is reasonably complete.")

    if unique_count <= 1:
        blockers.append("The target has only one distinct value after cleaning.")

    if usable_rows and _is_identifier_like(non_null_target, target_col):
        blockers.append("The target looks identifier-like or sequential rather than meaningfully predictable.")

    if usable_rows and _is_text_heavy_target(non_null_target):
        blockers.append("The target looks like long-form text, which this app should summarize rather than predict directly.")

    if problem_type == "classification":
        class_counts = non_null_target.value_counts(dropna=False)
        min_class_count = int(class_counts.min()) if not class_counts.empty else 0
        if len(class_counts) < 2:
            blockers.append("Classification needs at least two target classes.")
        elif unique_count > min(40, max(10, usable_rows // 3)):
            blockers.append(
                f"The target creates {unique_count} classes, which is too fragmented for a practical classification workflow here."
            )
        elif min_class_count < 3:
            blockers.append("At least one target class has fewer than 3 rows.")
        else:
            reasons_for.append(f"Target looks like a classification problem with {unique_count} classes.")

        if unique_ratio > 0.35 and unique_count > 15:
            reasons_against.append("The target has many classes relative to dataset size, so the model may generalize poorly.")

    elif problem_type == "regression":
        if not pd.api.types.is_numeric_dtype(non_null_target) and numeric_conversion_ratio(non_null_target) < 0.95:
            blockers.append("The target does not behave like a clean numeric outcome for regression.")
        elif unique_count < 8:
            reasons_against.append("The target has very few distinct values, so a descriptive summary may be more meaningful than regression.")
        else:
            reasons_for.append("Target behaves like a numeric outcome suited to regression.")

    try:
        if prepare_features:
            prepared = prepare_training_frame(
                sanitized,
                target_col,
                drop_identifier_columns=drop_identifier_columns,
            )
            usable_feature_count = int(prepared["X"].shape[1])
        else:
            metadata = column_metadata or _build_training_column_metadata(sanitized)
            usable_feature_count = sum(
                payload["status"] != "drop"
                for column, payload in metadata.items()
                if column != target_col
            )
        if usable_feature_count < 1:
            blockers.append("No usable feature columns remain after preparation.")
        elif usable_feature_count == 1:
            reasons_against.append("Only one usable feature column remains after preparation.")
        else:
            reasons_for.append(
                f"{usable_feature_count} usable feature columns remain after preparation."
            )
    except Exception as exc:
        blockers.append(str(exc))

    mode_recommendation = "analysis" if blockers else "prediction"
    if mode_recommendation == "prediction":
        summary = (
            f"Prediction mode is reasonable for '{target_col}' because the target shape and dataset coverage look workable."
        )
    else:
        summary = (
            f"Insight mode is safer for '{target_col}' because the target or dataset does not support a trustworthy predictive workflow."
        )

    return {
        "selected_target": target_col,
        "problem_type": problem_type,
        "mode_recommendation": mode_recommendation,
        "summary": summary,
        "reasons_for_prediction": reasons_for,
        "reasons_against_prediction": reasons_against,
        "blockers": blockers,
        "usable_rows": usable_rows,
        "missing_ratio": float(missing_ratio),
        "unique_count": unique_count,
        "unique_ratio": float(unique_ratio),
        "usable_feature_count": usable_feature_count,
    }


def prepare_training_frame(df, target_col, drop_identifier_columns=True):
    """Return cleaned *raw* features; fitted pipelines perform all feature learning.

    The ``drop_identifier_columns`` argument remains for API compatibility.  ID
    detection and exclusion now occur inside each fitted CV pipeline, so the raw
    columns remain available for later scoring and row preservation.
    """

    del drop_identifier_columns
    sanitized = sanitize_dataframe(df)

    if target_col not in sanitized.columns:
        raise ValueError(f"Target column '{target_col}' was not found.")

    sanitized[target_col] = sanitize_target_series(sanitized[target_col])
    initial_row_count = len(sanitized)
    cleaned = sanitized.dropna(subset=[target_col]).copy()
    if len(cleaned) < 20:
        raise ValueError("Please upload at least 20 rows with a target value.")

    X = cleaned.drop(columns=[target_col]).copy()
    X, y, dropped_target_rows = filter_valid_target_rows(
        X,
        cleaned[target_col],
        "initial target preparation",
    )
    if dropped_target_rows:
        cleaned = cleaned.loc[y.index].copy()

    notes = []
    removed_target_rows = initial_row_count - len(cleaned)
    if removed_target_rows:
        notes.append(
            f"Removed {removed_target_rows} rows with missing or invalid target values in {target_col}."
        )

    if X.empty:
        raise ValueError("The dataset needs at least one feature column besides the target.")

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = _categorical_column_names(X)
    if not numeric_cols and not categorical_cols:
        raise ValueError("No supported numeric or categorical features were found.")

    return {
        "X": X,
        "y": y,
        "notes": notes,
        "dropped_columns": [],
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def sample_training_data(X, y, problem_type, max_rows=MAX_TRAIN_ROWS, random_state=42):
    if len(X) <= max_rows:
        return X, y, False

    if problem_type == "classification":
        class_counts = pd.Series(y).value_counts(dropna=False)
        can_stratify = (
            len(class_counts) > 1
            and int(class_counts.min()) >= 2
            and max_rows >= len(class_counts)
        )
        if can_stratify:
            sampled_X, _, sampled_y, _ = train_test_split(
                X,
                y,
                train_size=max_rows,
                random_state=random_state,
                stratify=y,
            )
            return sampled_X, sampled_y, True

        # Preserve every class once when stratification is mathematically
        # impossible (for example a class with a single row), then sample the
        # remaining rows without a deprecated groupby-apply operation.
        combined = pd.concat([X, pd.Series(y, index=X.index, name="__target__")], axis=1)
        representative_indices = combined.groupby("__target__", dropna=False).head(1).index
        representative = combined.loc[representative_indices]
        remaining_count = max_rows - len(representative)
        remainder = combined.drop(index=representative_indices).sample(
            n=max(0, min(remaining_count, len(combined) - len(representative))),
            random_state=random_state,
        )
        sampled = pd.concat([representative, remainder])
        return sampled.drop(columns="__target__"), sampled["__target__"], True

    sampled_idx = X.sample(n=max_rows, random_state=random_state).index
    return X.loc[sampled_idx], y.loc[sampled_idx], True


def assess_model_quality(problem_type, best_metrics, baseline_metrics):
    if problem_type == "classification":
        f1_gain = best_metrics["f1"] - baseline_metrics["f1"]
        accuracy_gain = best_metrics["accuracy"] - baseline_metrics["accuracy"]
        pr_gain = best_metrics.get("average_precision", best_metrics["f1"]) - baseline_metrics.get("average_precision", baseline_metrics["f1"])
        best_primary = best_metrics.get("average_precision", best_metrics["f1"])
        if best_primary >= 0.8 and (f1_gain >= 0.15 or pr_gain >= 0.15):
            verdict = "strong"
        elif best_primary >= 0.6 and (f1_gain >= 0.05 or pr_gain >= 0.05):
            verdict = "useful"
        else:
            verdict = "weak"
        if "average_precision" in best_metrics and "average_precision" in baseline_metrics:
            summary = (
                f"Model worth: {verdict}. Average precision improved by {pr_gain:.3f}, "
                f"weighted F1 changed by {f1_gain:.3f}, and accuracy changed by {accuracy_gain:.3f} "
                f"versus the baseline."
            )
        else:
            summary = (
                f"Model worth: {verdict}. Weighted F1 improved by {f1_gain:.3f} and "
                f"accuracy improved by {accuracy_gain:.3f} over the baseline."
            )
        return {
            "verdict": verdict,
            "summary": summary,
            "primary_delta": f1_gain,
            "baseline_metric": baseline_metrics["f1"],
            "best_metric": best_metrics["f1"],
        }

    rmse_improvement = baseline_metrics["rmse"] - best_metrics["rmse"]
    r2_value = best_metrics["r2"]
    relative_rmse_gain = rmse_improvement / max(baseline_metrics["rmse"], 1e-9)
    if r2_value >= 0.7 and relative_rmse_gain >= 0.15:
        verdict = "strong"
    elif r2_value >= 0.45 and relative_rmse_gain >= 0.08:
        verdict = "useful"
    elif r2_value >= 0.25 and relative_rmse_gain >= 0.05:
        verdict = "limited"
    else:
        verdict = "weak"
    summary = (
        f"Model worth: {verdict}. RMSE improved by {rmse_improvement:.3f} "
        f"({relative_rmse_gain:.1%}) over the baseline and R^2 is {r2_value:.3f}."
    )
    return {
        "verdict": verdict,
        "summary": summary,
        "primary_delta": relative_rmse_gain,
        "baseline_metric": baseline_metrics["rmse"],
        "best_metric": best_metrics["rmse"],
    }


def build_chart_context(problem_type, X_sample, y_sample, holdout_actual, holdout_pred, feature_importance):
    numeric_cols = X_sample.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = _categorical_column_names(X_sample)
    top_features = feature_importance["feature"].tolist() if not feature_importance.empty else []

    resolved_top_features = []
    for feature in top_features:
        raw_name = feature.split("__", 1)[-1]
        if raw_name in X_sample.columns and raw_name not in resolved_top_features:
            resolved_top_features.append(raw_name)

    top_numeric = [col for col in resolved_top_features if col in numeric_cols][:2]
    top_categorical = [col for col in resolved_top_features if col in categorical_cols][:2]

    if not top_numeric and numeric_cols:
        if problem_type == "regression":
            correlations = []
            for column in numeric_cols:
                joined = pd.concat([X_sample[column], y_sample], axis=1).dropna()
                if len(joined) > 2:
                    corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
                    correlations.append((column, abs(corr) if pd.notna(corr) else 0))
            top_numeric = [name for name, _ in sorted(correlations, key=lambda item: item[1], reverse=True)[:2]]
        else:
            top_numeric = numeric_cols[:2]

    if not top_categorical and categorical_cols:
        filtered = [
            col for col in categorical_cols
            if 1 < X_sample[col].nunique(dropna=True) <= 20
        ]
        top_categorical = filtered[:2]

    context = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "top_numeric": top_numeric,
        "top_categorical": top_categorical,
        "target_distribution": y_sample.head(MAX_CHART_ROWS),
        "holdout_actual": holdout_actual.head(MAX_CHART_ROWS),
        "holdout_pred": holdout_pred.head(MAX_CHART_ROWS),
    }

    if problem_type == "classification":
        unique_classes = pd.Index(sorted(y_sample.dropna().unique().tolist()))
        if len(unique_classes) <= 20:
            matrix = confusion_matrix(
                holdout_actual,
                holdout_pred,
                labels=unique_classes,
            )
            context["confusion_matrix"] = {
                "labels": unique_classes.tolist(),
                "matrix": matrix.tolist(),
            }

    relationship_frames = []
    for column in top_numeric + top_categorical:
        frame = pd.DataFrame(
            {
                "feature": X_sample[column].head(MAX_CHART_ROWS).reset_index(drop=True),
                "target": y_sample.head(MAX_CHART_ROWS).reset_index(drop=True),
            }
        )
        relationship_frames.append({"column": column, "frame": frame})

    context["relationships"] = relationship_frames
    return context


def train_best_model(
    X,
    y,
    problem_type,
    numeric_cols,
    categorical_cols,
    target_style_label=None,
    effort="standard",
    test_size=0.2,
    random_state=42,
    positive_label=None,
):
    del numeric_cols, categorical_cols, target_style_label
    target_name = str(getattr(y, "name", None) or "target")
    config = AnalysisConfig(
        target=target_name,
        problem_type=problem_type,
        positive_label=positive_label,
        effort=effort,
        test_size=test_size,
        random_seed=random_state,
    )
    return train_model(pd.DataFrame(X), pd.Series(y, name=target_name), config)


def _build_analysis_result(
    df,
    insight_analysis,
    target_col=None,
    target_assessment=None,
    predictive_attempt=None,
    extra_notes=None,
    dataset_recommendation=None,
):
    details = []
    summary = "The dataset is better suited to insight-focused analysis than predictive modeling."

    if target_col is None:
        summary = "No target was selected, so the app switched to insight mode automatically."
        details.append("A supervised model needs a clearly defined outcome column.")
    elif target_assessment is not None:
        summary = target_assessment["summary"]
        details.extend(target_assessment["blockers"])
        details.extend(target_assessment["reasons_against_prediction"])

    if predictive_attempt is not None:
        summary = (
            f"Prediction was tested for '{target_col}', but the holdout results were too weak to present as a useful model."
        )
        details.append(predictive_attempt["quality"]["summary"])
        details.append(
            f"Best model tested: {predictive_attempt['best_model_name']}."
        )

    notes = list(extra_notes or [])
    if target_assessment is not None:
        notes.extend(target_assessment["reasons_for_prediction"])
        notes.extend(target_assessment["reasons_against_prediction"])
        notes.extend(target_assessment["blockers"])
    if predictive_attempt is not None:
        notes.append(predictive_attempt["quality"]["summary"])

    return {
        "mode": "analysis",
        "dataset_recommendation": dataset_recommendation,
        "decision": {
            "selected_mode": "analysis",
            "summary": summary,
            "details": details[:6],
        },
        "target_assessment": target_assessment,
        "predictive_attempt": predictive_attempt,
        "insight_analysis": insight_analysis,
        "analysis_recommendations": insight_analysis["analysis_recommendations"],
        "notes": notes,
        "used_rows": int(df.shape[0]),
        "original_rows": int(df.shape[0]),
        "original_columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "selected_target": target_col,
    }


def run_analysis(
    df,
    target_col,
    problem_type_mode="Auto Detect",
    test_size=0.2,
    drop_identifier_columns=True,
    training_effort="standard",
    extension_registry=None,
    positive_label=None,
    random_state=42,
    dataset_profile=None,
    precomputed_workflow=None,
):
    if isinstance(dataset_profile, DatasetProfile):
        sanitized_df = dataset_profile.sanitized_frame
    else:
        sanitized_df = sanitize_dataframe(df)
    selected_target = target_col if target_col in sanitized_df.columns else None
    dataset_recommendation = precomputed_workflow or recommend_dataset_workflow(
        sanitized_df,
        drop_identifier_columns=drop_identifier_columns,
        top_n=min(8, len(sanitized_df.columns)),
        extension_registry=extension_registry,
    )
    insight_analysis = dataset_recommendation["insight_analysis"]
    insight_analysis["selected_target"] = selected_target

    if not selected_target:
        result = _build_analysis_result(
            sanitized_df,
            insight_analysis,
            target_col=None,
            extra_notes=["No target selected. Defaulted to descriptive dataset analysis."],
            dataset_recommendation=dataset_recommendation,
        )
        result["assistant_extensions"] = _collect_assistant_extensions(
            extension_registry,
            sanitized_df,
            artifacts={
                "workflow": dataset_recommendation,
                "insight_analysis": insight_analysis,
                "column_inspection": insight_analysis.get("column_inspection"),
                "feature_subset_summary": dataset_recommendation.get("feature_subset_summary"),
                "result": result,
            },
        )
        return result

    target_candidate = dataset_recommendation["candidate_lookup"].get(selected_target)
    if target_candidate is not None and target_candidate["status"] != "rejected":
        target_assessment = assess_target_for_prediction(
            sanitized_df,
            selected_target,
            drop_identifier_columns=drop_identifier_columns,
            prepare_features=False,
        )
    elif target_candidate is not None:
        target_assessment = {
            "selected_target": selected_target,
            "problem_type": target_candidate["problem_type"],
            "mode_recommendation": "analysis",
            "summary": (
                f"Insight mode is safer for '{selected_target}' because it was rejected as a practical prediction target."
            ),
            "reasons_for_prediction": target_candidate["pros"],
            "reasons_against_prediction": target_candidate["cautions"],
            "blockers": target_candidate["blockers"],
            "usable_rows": target_candidate["usable_rows"],
            "missing_ratio": float(target_candidate["missing_pct"]) / 100.0,
            "unique_count": target_candidate["unique_count"],
            "unique_ratio": float(target_candidate["unique_ratio"]),
            "usable_feature_count": target_candidate["usable_feature_count"],
        }
    else:
        target_assessment = assess_target_for_prediction(
            sanitized_df,
            selected_target,
            drop_identifier_columns=drop_identifier_columns,
            prepare_features=False,
        )

    if target_assessment["mode_recommendation"] != "prediction":
        result = _build_analysis_result(
            sanitized_df,
            insight_analysis,
            target_col=selected_target,
            target_assessment=target_assessment,
            dataset_recommendation=dataset_recommendation,
        )
        result["assistant_extensions"] = _collect_assistant_extensions(
            extension_registry,
            sanitized_df,
            target_col=selected_target,
            artifacts={
                "workflow": dataset_recommendation,
                "insight_analysis": insight_analysis,
                "column_inspection": insight_analysis.get("column_inspection"),
                "feature_subset_summary": dataset_recommendation.get("feature_subset_summary"),
                "target_candidate": target_candidate,
                "target_assessment": target_assessment,
                "result": result,
            },
        )
        return result

    prepared = prepare_training_frame(
        sanitized_df,
        selected_target,
        drop_identifier_columns=drop_identifier_columns,
    )
    X = prepared["X"]
    y = prepared["y"]

    problem_type = detect_problem_type(y) if problem_type_mode == "Auto Detect" else problem_type_mode.lower()
    target_style = summarize_target_style(y, problem_type)

    sampled_X, sampled_y, sampled = sample_training_data(X, y, problem_type)
    sampled_y.name = selected_target
    sampled_numeric = [col for col in prepared["numeric_cols"] if col in sampled_X.columns]
    sampled_categorical = [col for col in prepared["categorical_cols"] if col in sampled_X.columns]

    training_kwargs = {
        "target_style_label": target_style["label"],
        "effort": training_effort,
        "test_size": test_size,
    }
    if positive_label is not None:
        training_kwargs["positive_label"] = positive_label
    if random_state != 42:
        training_kwargs["random_state"] = random_state
    trained = train_best_model(
        sampled_X,
        sampled_y,
        problem_type,
        sampled_numeric,
        sampled_categorical,
        **training_kwargs,
    )
    if trained.get("model_bundle") is not None:
        trained["model_bundle"].identifier_reference = {
            column: set(X[column].dropna().tolist())
            for column in trained["model_bundle"].optional_identifier_columns
            if column in X.columns
        }

    feature_importance = trained["feature_importance"]
    baseline_metrics = trained["baseline_metrics"]
    quality = assess_model_quality(problem_type, trained["best_metrics"], baseline_metrics)

    notes = list(prepared["notes"])
    if problem_type_mode == "Auto Detect":
        notes.insert(0, f"Auto-detected {problem_type} ({target_style['label']}).")
    if sampled:
        notes.insert(0, f"Sampled {len(sampled_X):,} rows from {len(X):,} for reliable training speed.")
    extra_target_drops = trained.get("dropped_target_rows", {})
    late_drops = sum(extra_target_drops.values()) if extra_target_drops else 0
    if late_drops:
        notes.insert(
            0,
            f"Removed {late_drops} additional target rows during final validation before model fitting.",
        )
    if trained["imbalance_ratio"] is not None and trained["imbalance_ratio"] < 0.1:
        notes.insert(
            0,
            f"Detected an imbalanced classification target (minority class share {trained['imbalance_ratio']:.3%}); ranking models by average precision.",
        )
    notes.extend(target_assessment["reasons_for_prediction"])
    notes.extend(target_assessment["reasons_against_prediction"])
    notes.extend(trained.get("leakage_warnings", []))
    notes.append("Internal validation is provisional until a labeled external dataset is evaluated.")

    prediction_preview = trained["X_test"].copy().reset_index(drop=True)
    prediction_preview["actual"] = pd.Series(trained["y_test"]).reset_index(drop=True)
    prediction_preview["prediction"] = pd.Series(trained["preds"]).reset_index(drop=True)
    if problem_type == "regression" and is_integer_like(y):
        prediction_preview["rounded_prediction"] = prediction_preview["prediction"].round().astype(int)

    chart_context = build_chart_context(
        problem_type,
        sampled_X,
        sampled_y.reset_index(drop=True),
        trained["y_test"].reset_index(drop=True),
        pd.Series(trained["preds"]).reset_index(drop=True),
        feature_importance,
    )

    if quality["verdict"] == "weak":
        result = _build_analysis_result(
            sanitized_df,
            insight_analysis,
            target_col=selected_target,
            target_assessment=target_assessment,
            predictive_attempt={
                "problem_type": problem_type,
                "best_model_name": trained["best_model_name"],
                "best_metrics": trained["best_metrics"],
                "baseline_metrics": baseline_metrics,
                "quality": quality,
                "model_bundle": trained.get("model_bundle"),
            },
            extra_notes=notes,
            dataset_recommendation=dataset_recommendation,
        )
        result["assistant_extensions"] = _collect_assistant_extensions(
            extension_registry,
            sanitized_df,
            target_col=selected_target,
            artifacts={
                "workflow": dataset_recommendation,
                "insight_analysis": insight_analysis,
                "column_inspection": insight_analysis.get("column_inspection"),
                "feature_subset_summary": dataset_recommendation.get("feature_subset_summary"),
                "target_candidate": target_candidate,
                "target_assessment": target_assessment,
                "prepared_frame": prepared,
                "result": result,
            },
        )
        return result

    result = {
        "mode": "prediction",
        "dataset_recommendation": dataset_recommendation,
        "decision": {
            "selected_mode": "prediction",
            "summary": (
                f"Prediction mode stayed active for '{selected_target}' because the target looked workable "
                f"and the best holdout model was rated {quality['verdict']}."
            ),
            "details": (
                target_assessment["reasons_for_prediction"]
                + target_assessment["reasons_against_prediction"]
                + [quality["summary"]]
            )[:6],
        },
        "target_assessment": target_assessment,
        "analysis_recommendations": insight_analysis["analysis_recommendations"],
        "insight_analysis": insight_analysis,
        "problem_type": problem_type,
        "target_style": target_style,
        "results": trained["results"],
        "best_model_name": trained["best_model_name"],
        "best_model": trained["best_model"],
        "model_bundle": trained.get("model_bundle"),
        "best_metrics": trained["best_metrics"],
        "baseline_metrics": baseline_metrics,
        "cv_results": trained.get("cv_results", {}),
        "positive_label": trained.get("positive_label"),
        "decision_threshold": trained.get("decision_threshold"),
        "validation_status": "provisional",
        "quality": quality,
        "metric_name": trained["metric_name"],
        "feature_columns": (
            trained["model_bundle"].required_feature_columns
            if trained.get("model_bundle") is not None
            else sampled_X.columns.tolist()
        ),
        "feature_importance": feature_importance,
        "chart_context": chart_context,
        "prediction_preview": prediction_preview.head(MAX_PREVIEW_ROWS),
        "notes": notes,
        "training_effort": training_effort,
        "used_rows": int(len(sampled_X)),
        "original_rows": int(len(X)),
        "original_columns": int(sanitized_df.shape[1]),
        "missing_cells": int(sanitized_df.isna().sum().sum()),
        "target_series": y.reset_index(drop=True),
        "holdout_actual": trained["y_test"].reset_index(drop=True),
        "holdout_pred": pd.Series(trained["preds"]).reset_index(drop=True),
        "selected_target": selected_target,
    }
    result["assistant_extensions"] = _collect_assistant_extensions(
        extension_registry,
        sanitized_df,
        target_col=selected_target,
        artifacts={
            "workflow": dataset_recommendation,
            "insight_analysis": insight_analysis,
            "column_inspection": insight_analysis.get("column_inspection"),
            "feature_subset_summary": dataset_recommendation.get("feature_subset_summary"),
            "target_candidate": target_candidate,
            "target_assessment": target_assessment,
            "prepared_frame": prepared,
            "result": result,
        },
    )
    return result

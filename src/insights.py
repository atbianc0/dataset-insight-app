import warnings

import numpy as np
import pandas as pd

from src.heuristics import (
    classify_general_column_role,
    is_datetime_candidate as _is_datetime_candidate,
    is_identifier_like as _is_identifier_like,
    is_text_heavy as _is_long_text,
    safe_ratio as _safe_ratio,
    sample_values as _sample_values,
)


MAX_SAMPLE_VALUES = 3
MAX_CORRELATIONS = 10
MAX_GROUP_ROWS = 12
MAX_ANOMALIES = 10


def build_column_inspection(df):
    rows = []
    row_count = max(len(df), 1)

    for column in df.columns:
        series = df[column]
        non_null = int(series.notna().sum())
        missing_pct = series.isna().mean() * 100
        unique_values = int(series.nunique(dropna=True))
        role_key = classify_general_column_role(series, column)
        if role_key == "identifier":
            role = "Identifier-like"
            recommendation = "Avoid as a prediction target"
        elif role_key == "datetime":
            role = "Date / time"
            recommendation = "Useful for trend analysis"
        elif role_key == "numeric":
            role = "Numeric measure"
            recommendation = "Useful for stats, trends, correlations"
        elif role_key == "text":
            role = "Long text"
            recommendation = "Better for summaries than direct ML targets"
        elif role_key == "categorical":
            role = "Categorical label"
            recommendation = "Useful for grouping or classification"
        else:
            role = "High-cardinality category"
            recommendation = "Use carefully; often better as a feature"

        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "role_hint": role,
                "non_null": non_null,
                "missing_pct": round(missing_pct, 1),
                "unique_values": unique_values,
                "coverage_pct": round((_safe_ratio(non_null, row_count)) * 100, 1),
                "sample_values": _sample_values(series),
                "recommendation": recommendation,
            }
        )

    return pd.DataFrame(rows)


def recommend_analysis_paths(df):
    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    datetime_cols = [column for column in df.columns if _is_datetime_candidate(df[column])]
    categorical_cols = [
        column
        for column in df.columns
        if column not in numeric_cols and column not in datetime_cols
    ]
    low_cardinality = [
        column for column in categorical_cols if 2 <= df[column].nunique(dropna=True) <= 15
    ]

    recommendations = [
        {
            "analysis_type": "Exploratory data analysis",
            "reason": "Always useful for understanding column coverage, data quality, and distribution shape.",
        }
    ]

    if numeric_cols:
        recommendations.append(
            {
                "analysis_type": "Descriptive statistics",
                "reason": f"The dataset has {len(numeric_cols)} numeric column(s) that can be summarized directly.",
            }
        )

    if len(numeric_cols) >= 2:
        recommendations.append(
            {
                "analysis_type": "Correlation analysis",
                "reason": "Multiple numeric fields are available, so relationships can be ranked and compared.",
            }
        )

    if datetime_cols:
        recommendations.append(
            {
                "analysis_type": "Trend analysis",
                "reason": f"Detected date-like column(s): {', '.join(datetime_cols[:2])}.",
            }
        )

    if low_cardinality and numeric_cols:
        recommendations.append(
            {
                "analysis_type": "Grouping analysis",
                "reason": "Low-cardinality categories can be compared against numeric measures.",
            }
        )

    if len(numeric_cols) >= 2 and len(df) >= 30:
        recommendations.append(
            {
                "analysis_type": "Anomaly detection",
                "reason": "There are enough numeric observations to flag unusually extreme rows.",
            }
        )

    if len(numeric_cols) >= 3 and len(df) >= 50:
        recommendations.append(
            {
                "analysis_type": "Clustering or segmentation",
                "reason": "Several numeric dimensions are present, which makes grouping similar records plausible.",
            }
        )

    return pd.DataFrame(recommendations)


def _build_numeric_summary(df):
    numeric_cols = [
        column for column in df.select_dtypes(include=["number", "bool"]).columns
        if not _is_identifier_like(df[column], column)
    ]
    if not numeric_cols:
        return pd.DataFrame()

    summary = df[numeric_cols].describe().transpose().reset_index().rename(columns={"index": "column"})
    summary["missing_pct"] = [round(df[column].isna().mean() * 100, 1) for column in summary["column"]]
    return summary.round(3)


def _build_categorical_summary(df):
    rows = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series) or _is_datetime_candidate(series):
            continue

        top_values = series.dropna().astype(str).value_counts().head(3)
        rows.append(
            {
                "column": column,
                "unique_values": int(series.nunique(dropna=True)),
                "missing_pct": round(series.isna().mean() * 100, 1),
                "top_values": ", ".join(
                    [f"{value} ({count})" for value, count in top_values.items()]
                ) or "None",
            }
        )

    return pd.DataFrame(rows)


def _build_correlations(df):
    numeric_cols = [
        column for column in df.select_dtypes(include=["number", "bool"]).columns
        if not _is_identifier_like(df[column], column)
    ]
    if len(numeric_cols) < 2:
        return pd.DataFrame()

    corr = df[numeric_cols].corr(numeric_only=True)
    pairs = []
    for i, left in enumerate(numeric_cols):
        for right in numeric_cols[i + 1:]:
            value = corr.loc[left, right]
            if pd.isna(value):
                continue
            pairs.append(
                {
                    "column_a": left,
                    "column_b": right,
                    "correlation": round(float(value), 3),
                    "abs_correlation": round(abs(float(value)), 3),
                }
            )

    if not pairs:
        return pd.DataFrame()

    frame = pd.DataFrame(pairs).sort_values(
        ["abs_correlation", "column_a", "column_b"],
        ascending=[False, True, True],
    )
    return frame.head(MAX_CORRELATIONS).reset_index(drop=True)


def _build_group_summary(df):
    numeric_cols = [
        column for column in df.select_dtypes(include=["number", "bool"]).columns
        if df[column].nunique(dropna=True) > 1 and not _is_identifier_like(df[column], column)
    ]
    categorical_cols = [
        column for column in df.columns
        if column not in numeric_cols and 2 <= df[column].nunique(dropna=True) <= 12
    ]

    best = None
    best_score = None

    for category_column in categorical_cols:
        for metric_column in numeric_cols:
            frame = df[[category_column, metric_column]].dropna()
            if len(frame) < 10:
                continue

            grouped = frame.groupby(category_column)[metric_column].agg(["mean", "count"])
            grouped = grouped[grouped["count"] >= 3]
            if len(grouped) < 2:
                continue

            spread = grouped["mean"].std()
            scale = frame[metric_column].std()
            score = float(spread / scale) if scale and not np.isnan(scale) else 0.0
            if best_score is None or score > best_score:
                best_score = score
                best = (category_column, metric_column, grouped)

    if best is None:
        return pd.DataFrame()

    category_column, metric_column, grouped = best
    summary = grouped.sort_values("mean", ascending=False).head(MAX_GROUP_ROWS).reset_index()
    summary = summary.rename(
        columns={
            category_column: "group",
            "mean": "average_value",
            "count": "row_count",
        }
    )
    summary.insert(0, "category_column", category_column)
    summary.insert(1, "metric_column", metric_column)
    return summary.round(3)


def _select_preferred_datetime_column(df, datetime_cols):
    ranked = sorted(
        datetime_cols,
        key=lambda column: (
            0 if any(token in column.lower() for token in ["date", "time", "month", "year"]) else 1,
            column.lower(),
        ),
    )
    return ranked[0] if ranked else None


def _build_trend_summary(df):
    datetime_cols = [column for column in df.columns if _is_datetime_candidate(df[column])]
    if not datetime_cols:
        return None

    date_column = _select_preferred_datetime_column(df, datetime_cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed_dates = pd.to_datetime(df[date_column], errors="coerce")

    trend_frame = pd.DataFrame({"date": parsed_dates}).dropna()
    if trend_frame.empty:
        return None

    numeric_candidates = [
        column
        for column in df.select_dtypes(include=["number", "bool"]).columns
        if column != date_column and not _is_identifier_like(df[column], column) and df[column].nunique(dropna=True) > 1
    ]

    span_days = (trend_frame["date"].max() - trend_frame["date"].min()).days
    frequency = "M" if span_days >= 120 else "D"
    trend_frame["period"] = trend_frame["date"].dt.to_period(frequency).dt.to_timestamp()

    if numeric_candidates:
        metric_column = sorted(
            numeric_candidates,
            key=lambda column: df[column].notna().sum(),
            reverse=True,
        )[0]
        trend_frame["value"] = df.loc[trend_frame.index, metric_column]
        trend_frame = trend_frame.dropna(subset=["value"])
        aggregated = trend_frame.groupby("period")["value"].mean().reset_index()
        label = f"Average {metric_column} over time"
    else:
        metric_column = None
        aggregated = trend_frame.groupby("period").size().reset_index(name="value")
        label = "Row count over time"

    if len(aggregated) < 2:
        return None

    change = aggregated["value"].iloc[-1] - aggregated["value"].iloc[0]
    baseline = aggregated["value"].iloc[0]
    change_pct = _safe_ratio(change, baseline) if baseline not in (0, np.nan) else 0.0

    return {
        "date_column": date_column,
        "metric_column": metric_column,
        "description": (
            f"{label}. Change from first to last period: {change:.3f} "
            f"({change_pct:.1%})."
        ),
        "frame": aggregated,
    }


def _build_anomaly_summary(df):
    numeric_cols = [
        column for column in df.select_dtypes(include=["number", "bool"]).columns
        if df[column].nunique(dropna=True) > 1 and not _is_identifier_like(df[column], column)
    ]
    if len(numeric_cols) < 2 or len(df) < 20:
        return pd.DataFrame()

    numeric_frame = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    numeric_frame = numeric_frame.dropna(thresh=max(2, len(numeric_cols) // 2))
    if numeric_frame.empty:
        return pd.DataFrame()

    std = numeric_frame.std(ddof=0).replace(0, np.nan)
    z_scores = ((numeric_frame - numeric_frame.mean()) / std).abs()
    score = z_scores.mean(axis=1, skipna=True)
    flagged = score[score >= 2.0].sort_values(ascending=False).head(MAX_ANOMALIES)
    if flagged.empty:
        flagged = score.sort_values(ascending=False).head(min(MAX_ANOMALIES, len(score)))

    summary = df.loc[flagged.index].copy()
    summary.insert(0, "row_index", summary.index.astype(int))
    summary.insert(1, "anomaly_score", flagged.round(3))
    return summary.reset_index(drop=True)


def _build_headline_insights(overview, correlations, group_summary, trend_summary, anomaly_summary):
    insights = [
        (
            f"Dataset contains {overview['rows']:,} rows and {overview['columns']:,} columns "
            f"with {overview['missing_cells']:,} missing cells."
        )
    ]

    if overview["top_missing_column"]:
        insights.append(
            f"Most incomplete column: {overview['top_missing_column']} "
            f"({overview['top_missing_pct']:.1f}% missing)."
        )

    if not correlations.empty:
        top_corr = correlations.iloc[0]
        insights.append(
            f"Strongest numeric relationship found: {top_corr['column_a']} vs {top_corr['column_b']} "
            f"(correlation {top_corr['correlation']})."
        )

    if not group_summary.empty:
        first = group_summary.iloc[0]
        insights.append(
            f"Grouping signal: {first['metric_column']} changes meaningfully across {first['category_column']} groups."
        )

    if trend_summary is not None:
        insights.append(trend_summary["description"])

    if not anomaly_summary.empty:
        insights.append(
            f"Top anomaly score observed: {anomaly_summary.iloc[0]['anomaly_score']:.3f}. "
            "Review the anomaly table for unusually extreme rows."
        )

    return insights[:5]


def _build_data_quality_summary(column_inspection, overview):
    summaries = []
    role_counts = column_inspection["role_hint"].value_counts().to_dict()
    if role_counts:
        summaries.append(
            "Column mix: "
            + ", ".join([f"{count} {role.lower()}" for role, count in role_counts.items()][:4])
            + "."
        )

    if overview["top_missing_column"]:
        summaries.append(
            f"The biggest coverage risk is '{overview['top_missing_column']}' at "
            f"{overview['top_missing_pct']:.1f}% missing."
        )

    avoid_target_count = int((column_inspection["recommendation"] == "Avoid as a prediction target").sum())
    if avoid_target_count:
        summaries.append(
            f"{avoid_target_count} column(s) look more like identifiers than meaningful prediction targets."
        )

    long_text_count = int((column_inspection["role_hint"] == "Long text").sum())
    if long_text_count:
        summaries.append(
            f"{long_text_count} long-text column(s) are better suited for summaries than direct modeling."
        )

    return summaries[:4]


def _build_grouping_highlight(group_summary):
    if group_summary.empty:
        return None

    top = group_summary.iloc[0]
    highest_group = group_summary.iloc[0]["group"]
    lowest_group = group_summary.iloc[-1]["group"]
    return (
        f"{top['metric_column']} varies across {top['category_column']}. "
        f"Highest shown group: {highest_group}; lowest shown group: {lowest_group}."
    )


def _build_anomaly_highlight(anomaly_summary):
    if anomaly_summary.empty:
        return None

    top = anomaly_summary.iloc[0]
    return (
        f"Most unusual row in the sample is row {int(top['row_index'])} "
        f"with anomaly score {float(top['anomaly_score']):.3f}."
    )


def run_insight_analysis(df, target_col=None):
    overview = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "numeric_columns": int(len(df.select_dtypes(include=["number", "bool"]).columns)),
        "top_missing_column": None,
        "top_missing_pct": 0.0,
    }

    if not df.empty:
        missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
        if not missing_pct.empty and missing_pct.iloc[0] > 0:
            overview["top_missing_column"] = missing_pct.index[0]
            overview["top_missing_pct"] = float(missing_pct.iloc[0])

    column_inspection = build_column_inspection(df)
    analysis_recommendations = recommend_analysis_paths(df)
    numeric_summary = _build_numeric_summary(df)
    categorical_summary = _build_categorical_summary(df)
    correlations = _build_correlations(df)
    group_summary = _build_group_summary(df)
    trend_summary = _build_trend_summary(df)
    anomaly_summary = _build_anomaly_summary(df)
    data_quality_summary = _build_data_quality_summary(column_inspection, overview)
    grouping_highlight = _build_grouping_highlight(group_summary)
    anomaly_highlight = _build_anomaly_highlight(anomaly_summary)
    headlines = _build_headline_insights(
        overview,
        correlations,
        group_summary,
        trend_summary,
        anomaly_summary,
    )

    return {
        "overview": overview,
        "column_inspection": column_inspection,
        "analysis_recommendations": analysis_recommendations,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "correlations": correlations,
        "group_summary": group_summary,
        "trend_summary": trend_summary,
        "anomaly_summary": anomaly_summary,
        "data_quality_summary": data_quality_summary,
        "grouping_highlight": grouping_highlight,
        "anomaly_highlight": anomaly_highlight,
        "best_analysis_path": (
            analysis_recommendations.iloc[0].to_dict()
            if not analysis_recommendations.empty
            else None
        ),
        "headlines": headlines,
        "selected_target": target_col,
    }

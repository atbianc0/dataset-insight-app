"""Trustworthy, general-purpose summaries for tabular datasets.

The functions in this module deliberately separate descriptive evidence from
statistical claims.  Dataset-level counts are exact.  Potentially expensive
pairwise calculations use a deterministic bounded sample and say so in their
output.
"""

import math
import re
import warnings

import numpy as np
import pandas as pd

from src.heuristics import (
    classify_general_column_role,
)
from src.heuristics import (
    is_datetime_candidate as _is_datetime_candidate,
)
from src.heuristics import (
    is_identifier_like as _is_identifier_like,
)
from src.heuristics import (
    safe_ratio as _safe_ratio,
)
from src.heuristics import (
    sample_values as _sample_values,
)

MAX_SAMPLE_VALUES = 3
MAX_CORRELATIONS = 10
MAX_GROUP_ROWS = 12
MAX_ANOMALIES = 10
MAX_ANALYSIS_ROWS = 50_000
MAX_MULTI_VALUE_ROWS_PER_COLUMN = 8
RANDOM_STATE = 42
CORRELATION_MIN_PAIRED_ROWS = 30
CORRELATION_MIN_COVERAGE = 0.60
CORRELATION_MIN_EFFECT = 0.30
CORRELATION_MAX_ADJUSTED_P = 0.05
ANOMALY_THRESHOLD = 4.5

_OUTCOME_NAMES = {
    "attrition",
    "churn",
    "churned",
    "converted",
    "conversion",
    "default",
    "fraud",
    "label",
    "outcome",
    "response",
    "target",
}
_SENSITIVE_NAMES = {
    "age",
    "disability",
    "ethnicity",
    "gender",
    "nationality",
    "race",
    "religion",
    "sex",
}


def _bounded_frame(df, columns=None, max_rows=MAX_ANALYSIS_ROWS):
    frame = df if columns is None else df.loc[:, columns]
    if len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=RANDOM_STATE).sort_index()


def _normalise_name(value):
    return " ".join(part for part in re.split(r"[^a-z0-9]+", str(value).lower()) if part)


def _is_string_like(series):
    dtype = series.dtype
    return (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def _conversion_hint(series):
    if not _is_string_like(series):
        return None, 0.0

    sample = pd.Series(series).dropna().head(1_000).astype("string").str.strip()
    if sample.empty:
        return None, 0.0

    numeric_ratio = float(pd.to_numeric(sample, errors="coerce").notna().mean())
    if numeric_ratio >= 0.95:
        return "Convert to numeric after validating units and missing tokens", numeric_ratio

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        datetime_ratio = float(pd.to_datetime(sample, errors="coerce").notna().mean())
    if datetime_ratio >= 0.85:
        return "Parse as date/time after confirming the source format", datetime_ratio
    return None, max(numeric_ratio, datetime_ratio)


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
            recommendation = "Useful for count-based trend analysis"
        elif role_key == "numeric":
            role = "Numeric measure"
            recommendation = "Useful for distributions and supported associations"
        elif role_key == "text":
            role = "Long text"
            recommendation = "Better for summaries than direct ML targets"
        elif role_key in {"categorical", "boolean"}:
            role = "Categorical label"
            recommendation = "Useful for grouping or classification"
        else:
            role = "High-cardinality category"
            recommendation = "Use carefully; often better as a feature"

        conversion, confidence = _conversion_hint(series)
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "role_hint": role,
                "non_null": non_null,
                "missing_pct": round(missing_pct, 1),
                "unique_values": unique_values,
                "uniqueness_pct": round(_safe_ratio(unique_values, non_null) * 100, 1),
                "coverage_pct": round((_safe_ratio(non_null, row_count)) * 100, 1),
                "sample_values": _sample_values(
                    series,
                    max_values=min(MAX_SAMPLE_VALUES, unique_values),
                ),
                "type_conversion": conversion,
                "conversion_confidence": round(confidence, 3) if conversion else None,
                "recommendation": recommendation,
            }
        )

    return pd.DataFrame(rows)


def _numeric_columns(df, excluded=None):
    excluded = set(excluded or [])
    return [
        column
        for column in df.select_dtypes(include=["number", "bool"]).columns
        if column not in excluded
        and not _is_identifier_like(df[column], column)
        and df[column].nunique(dropna=True) > 1
    ]


def _build_numeric_summary(df):
    numeric_cols = _numeric_columns(df)
    if not numeric_cols:
        return pd.DataFrame()

    summary = (
        df[numeric_cols]
        .describe()
        .transpose()
        .reset_index()
        .rename(columns={"index": "column"})
    )
    summary["missing_pct"] = [
        round(df[column].isna().mean() * 100, 1) for column in summary["column"]
    ]
    return summary.round(3)


def _build_categorical_summary(df):
    rows = []
    for column in df.columns:
        series = df[column]
        if (pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)) or _is_datetime_candidate(series):
            continue

        top_values = series.dropna().astype(str).value_counts().head(3)
        top_value = top_values.index[0] if not top_values.empty else None
        top_count = int(top_values.iloc[0]) if not top_values.empty else 0
        rows.append(
            {
                "column": column,
                "unique_values": int(series.nunique(dropna=True)),
                "missing_pct": round(series.isna().mean() * 100, 1),
                "top_value": top_value,
                "top_value_count": top_count,
                "top_value_pct": round(_safe_ratio(top_count, len(df)) * 100, 1),
                "top_values": ", ".join(
                    [f"{value} ({count:,})" for value, count in top_values.items()]
                )
                or "None",
            }
        )

    return pd.DataFrame(rows)


def _approximate_correlation_p_value(correlation, paired_n):
    """Return a conservative normal approximation based on Fisher's z."""
    if paired_n < 4 or not np.isfinite(correlation):
        return 1.0
    if abs(correlation) >= 1:
        return 0.0
    fisher_z = math.atanh(float(correlation)) * math.sqrt(paired_n - 3)
    return float(math.erfc(abs(fisher_z) / math.sqrt(2)))


def _benjamini_hochberg(p_values):
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for rank_index in range(len(values) - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        candidate = values[original_index] * len(values) / rank
        running = min(running, candidate)
        adjusted[original_index] = min(1.0, running)
    return adjusted.tolist()


def _build_correlations(df):
    numeric_cols = _numeric_columns(df)
    if len(numeric_cols) < 2:
        return pd.DataFrame()

    analysis_frame = _bounded_frame(df, numeric_cols)
    analysis_rows = len(analysis_frame)
    pairs = []
    for index, left in enumerate(numeric_cols):
        for right in numeric_cols[index + 1 :]:
            pair = analysis_frame[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
            paired_n = len(pair)
            if paired_n < 2 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
                continue
            value = float(pair[left].corr(pair[right]))
            if not np.isfinite(value):
                continue
            pairs.append(
                {
                    "column_a": left,
                    "column_b": right,
                    "correlation": value,
                    "abs_correlation": abs(value),
                    "paired_n": paired_n,
                    "coverage_pct": _safe_ratio(paired_n, analysis_rows) * 100,
                    "analysis_rows": analysis_rows,
                    "sampled": len(df) > analysis_rows,
                    "p_value": _approximate_correlation_p_value(value, paired_n),
                }
            )

    if not pairs:
        return pd.DataFrame()

    adjusted = _benjamini_hochberg([row["p_value"] for row in pairs])
    for row, adjusted_p in zip(pairs, adjusted):
        row["adjusted_p_value"] = adjusted_p
        row["headline_eligible"] = bool(
            row["paired_n"] >= CORRELATION_MIN_PAIRED_ROWS
            and row["coverage_pct"] >= CORRELATION_MIN_COVERAGE * 100
            and row["abs_correlation"] >= CORRELATION_MIN_EFFECT
            and adjusted_p <= CORRELATION_MAX_ADJUSTED_P
        )

    frame = pd.DataFrame(pairs).sort_values(
        ["headline_eligible", "abs_correlation", "paired_n", "column_a", "column_b"],
        ascending=[False, False, False, True, True],
    )
    numeric_rounding = {
        "correlation": 3,
        "abs_correlation": 3,
        "coverage_pct": 1,
        "p_value": 6,
        "adjusted_p_value": 6,
    }
    return frame.head(MAX_CORRELATIONS).reset_index(drop=True).round(numeric_rounding)


def _build_group_summary(df, target_col=None):
    numeric_cols = _numeric_columns(df, excluded=[target_col] if target_col else None)
    categorical_cols = [
        column
        for column in df.columns
        if column != target_col
        and column not in numeric_cols
        and not _is_identifier_like(df[column], column)
        and 2 <= df[column].nunique(dropna=True) <= 12
        and not _is_datetime_candidate(df[column])
    ]
    if not numeric_cols or not categorical_cols:
        return pd.DataFrame()

    columns = list(dict.fromkeys(categorical_cols + numeric_cols))
    analysis_frame = _bounded_frame(df, columns)
    best = None
    best_score = -1.0

    for category_column in categorical_cols:
        for metric_column in numeric_cols:
            frame = analysis_frame[[category_column, metric_column]].dropna()
            if len(frame) < 30:
                continue
            minimum_support = max(5, int(math.ceil(len(frame) * 0.01)))
            grouped = frame.groupby(category_column, observed=True)[metric_column].agg(["mean", "count"])
            grouped = grouped[grouped["count"] >= minimum_support]
            if len(grouped) < 2:
                continue
            scale = float(frame[metric_column].std())
            spread = float(grouped["mean"].max() - grouped["mean"].min())
            score = abs(spread / scale) if scale and np.isfinite(scale) else 0.0
            if score > best_score:
                best_score = score
                best = (category_column, metric_column, grouped, float(frame[metric_column].mean()))

    if best is None:
        return pd.DataFrame()

    category_column, metric_column, grouped, overall_average = best
    summary = grouped.sort_values("mean", ascending=False).head(MAX_GROUP_ROWS).reset_index()
    summary = summary.rename(
        columns={category_column: "group", "mean": "average_value", "count": "row_count"}
    )
    summary.insert(0, "category_column", category_column)
    summary.insert(1, "metric_column", metric_column)
    summary["overall_average"] = overall_average
    summary["difference_from_overall"] = summary["average_value"] - overall_average
    summary["standardized_spread"] = best_score
    summary["analysis_rows"] = len(analysis_frame)
    summary["sampled"] = len(df) > len(analysis_frame)
    summary["interpretation"] = "Descriptive association only; this comparison does not establish causation."
    return summary.round(3)


def _select_preferred_datetime_column(df, datetime_cols):
    ranked = sorted(
        datetime_cols,
        key=lambda column: (
            0 if any(token in str(column).lower() for token in ["date", "time", "month", "year"]) else 1,
            str(column).lower(),
        ),
    )
    return ranked[0] if ranked else None


def _format_period(period, frequency):
    return str(period.year) if frequency == "Y" else period.strftime("%Y-%m-%d")


def _build_trend_summary(df):
    datetime_cols = [column for column in df.columns if _is_datetime_candidate(df[column])]
    if not datetime_cols:
        return None

    date_column = _select_preferred_datetime_column(df, datetime_cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed_dates = pd.to_datetime(df[date_column], errors="coerce")

    parsed_dates = parsed_dates.dropna()
    if parsed_dates.empty:
        return None

    span_days = int((parsed_dates.max() - parsed_dates.min()).days)
    frequency = "Y" if span_days >= 730 else ("M" if span_days >= 120 else "D")
    periods = parsed_dates.dt.to_period(frequency)
    all_periods = periods.value_counts().sort_index().rename_axis("period_key").reset_index(name="value")
    all_periods["period"] = all_periods["period_key"].dt.to_timestamp()

    first_period = periods.min()
    last_period = periods.max()
    partial_keys = []
    if parsed_dates.min().normalize() > first_period.start_time.normalize():
        partial_keys.append(first_period)
    if parsed_dates.max().normalize() < last_period.end_time.normalize() and last_period not in partial_keys:
        partial_keys.append(last_period)

    complete = all_periods[~all_periods["period_key"].isin(partial_keys)].copy()
    complete = complete[["period", "value"]].reset_index(drop=True)
    all_frame = all_periods[["period", "value"]].reset_index(drop=True)
    partial_labels = [_format_period(period, frequency) for period in partial_keys]
    frequency_label = {"Y": "year", "M": "month", "D": "day"}[frequency]

    warnings_list = []
    if partial_labels:
        warnings_list.append(
            "Excluded partial boundary period(s) from comparisons: " + ", ".join(partial_labels) + "."
        )

    if len(complete) >= 2:
        change = float(complete["value"].iloc[-1] - complete["value"].iloc[0])
        baseline = float(complete["value"].iloc[0])
        final_value = float(complete["value"].iloc[-1])
        if baseline >= 30:
            change_pct = _safe_ratio(change, baseline)
            description = (
                f"Row count by {frequency_label} across complete periods. "
                f"Change from first to last complete period: {change:,.0f} ({change_pct:.1%})."
            )
        else:
            description = (
                f"Row count by {frequency_label} across complete periods. First and last complete-period "
                f"counts were {baseline:,.0f} and {final_value:,.0f}; the small starting count makes a "
                "percentage comparison unstable."
            )
    else:
        description = f"Row count by {frequency_label}; too few complete periods for a first-to-last comparison."
    if warnings_list:
        description += " " + " ".join(warnings_list)

    return {
        "date_column": date_column,
        "metric_column": None,
        "aggregation": "row_count",
        "frequency": frequency_label,
        "description": description,
        "frame": complete,
        "all_periods_frame": all_frame,
        "partial_periods": partial_labels,
        "warnings": warnings_list,
        "date_coverage_pct": round(_safe_ratio(len(parsed_dates), len(df)) * 100, 1),
    }


def _build_anomaly_summary(df):
    numeric_cols = _numeric_columns(df)
    if len(numeric_cols) < 2 or len(df) < 20:
        return pd.DataFrame()

    analysis_frame = _bounded_frame(df, numeric_cols)
    numeric_frame = analysis_frame.apply(pd.to_numeric, errors="coerce")
    numeric_frame = numeric_frame.dropna(thresh=max(2, int(math.ceil(len(numeric_cols) / 2))))
    if numeric_frame.empty:
        return pd.DataFrame()

    median = numeric_frame.median()
    median_absolute_deviation = (numeric_frame - median).abs().median().replace(0, np.nan)
    robust_z = 0.6745 * (numeric_frame - median).abs().divide(median_absolute_deviation)
    valid_dimensions = robust_z.notna().sum(axis=1)
    score = robust_z.max(axis=1, skipna=True).where(valid_dimensions >= 2)
    flagged = score[score >= ANOMALY_THRESHOLD].sort_values(ascending=False).head(MAX_ANOMALIES)
    if flagged.empty:
        return pd.DataFrame()

    summary = df.loc[flagged.index].copy()
    summary.insert(0, "row_index", flagged.index)
    summary.insert(1, "anomaly_score", flagged.round(3))
    summary.insert(2, "threshold", ANOMALY_THRESHOLD)
    return summary.reset_index(drop=True)


def _build_multi_value_summary(df):
    rows = []
    for column in df.columns:
        series = df[column]
        if not _is_string_like(series) or _is_datetime_candidate(series):
            continue
        sample = _bounded_frame(pd.DataFrame({"value": series.dropna()}), ["value"], max_rows=5_000)["value"]
        sample = sample.astype("string")
        if sample.empty:
            continue
        comma_fraction = float(sample.str.contains(",", regex=False, na=False).mean())
        if comma_fraction < 0.08:
            continue
        token_counts = sample.str.count(",") + 1
        if float(token_counts.quantile(0.95)) > 30:
            continue
        sample_fragments = sample.str.split(",").explode().str.strip()
        fragment_word_counts = sample_fragments.str.split().str.len()
        if fragment_word_counts.empty or float(fragment_word_counts.quantile(0.75)) > 6:
            # Commas in prose are punctuation, not evidence of a multi-value field.
            continue

        values = series.astype("string").reset_index(drop=True)
        exploded = values.str.split(",").explode().str.strip()
        exploded = exploded[exploded.notna() & exploded.ne("")]
        if exploded.empty:
            continue
        token_frame = exploded.rename("value").reset_index().rename(columns={"index": "row_number"})
        token_frame = token_frame.drop_duplicates(["row_number", "value"])
        counts = token_frame["value"].value_counts()
        unique_tokens = int(token_frame["value"].nunique())
        represented_rows = int(values.notna().sum())
        for value, count in counts.head(MAX_MULTI_VALUE_ROWS_PER_COLUMN).items():
            rows.append(
                {
                    "column": column,
                    "value": value,
                    "row_count": int(count),
                    "row_pct": round(_safe_ratio(count, len(df)) * 100, 1),
                    "represented_rows": represented_rows,
                    "unique_tokens": unique_tokens,
                    "delimiter_coverage_pct": round(comma_fraction * 100, 1),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _normalise_unit(unit):
    compact = re.sub(r"[._-]+", " ", str(unit).strip().lower())
    aliases = {
        "min": "minutes",
        "mins": "minutes",
        "minute": "minutes",
        "minute s": "minutes",
        "season": "seasons",
        "season s": "seasons",
        "hr": "hours",
        "hrs": "hours",
        "hour": "hours",
    }
    return aliases.get(compact, compact if compact.endswith("s") else compact + "s")


def _build_unit_summary(df):
    rows = []
    value_pattern = r"^\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*([A-Za-z][A-Za-z ._-]*)\s*$"
    for column in df.columns:
        series = df[column]
        if not _is_string_like(series) or _is_datetime_candidate(series):
            continue
        non_null = series.dropna().astype("string")
        if non_null.empty:
            continue
        sample = non_null.head(2_000).str.extract(value_pattern)
        parse_ratio = float(sample[0].notna().mean())
        if parse_ratio < 0.70:
            continue
        extracted = non_null.str.extract(value_pattern)
        parsed = pd.DataFrame(
            {
                "value": pd.to_numeric(extracted[0], errors="coerce"),
                "unit": extracted[1].map(_normalise_unit),
            }
        ).dropna()
        if parsed.empty or parsed["unit"].nunique() > 8:
            continue
        for unit, values in parsed.groupby("unit", observed=True)["value"]:
            rows.append(
                {
                    "column": column,
                    "unit": unit,
                    "row_count": int(values.count()),
                    "coverage_pct": round(_safe_ratio(values.count(), len(df)) * 100, 1),
                    "mean": round(float(values.mean()), 2),
                    "median": round(float(values.median()), 2),
                    "min": round(float(values.min()), 2),
                    "max": round(float(values.max()), 2),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["column", "row_count"], ascending=[True, False]).reset_index(drop=True)


def _infer_association_target(df, target_col=None):
    if target_col in df.columns:
        return target_col, False
    candidates = []
    for column in df.columns:
        normalised = _normalise_name(column)
        unique_count = int(df[column].nunique(dropna=True))
        if normalised in _OUTCOME_NAMES and 2 <= unique_count <= 20:
            candidates.append(column)
    return (candidates[0], True) if len(candidates) == 1 else (None, False)


def _infer_positive_label(series):
    values = pd.Series(series).dropna().unique().tolist()
    if not values:
        return None
    if pd.api.types.is_bool_dtype(series):
        return True if True in values else values[-1]
    preferred = {"1", "yes", "true", "positive", "churn", "churned", "fraud", "default"}
    for value in values:
        if str(value).strip().lower() in preferred:
            return value
    if pd.api.types.is_numeric_dtype(series):
        return max(values)
    return sorted(values, key=lambda value: str(value))[-1]


def _build_target_associations(df, target_col):
    empty_overview = None
    if target_col not in df.columns:
        return empty_overview, pd.DataFrame(), []
    target = df[target_col]
    usable_target = target.dropna()
    unique_values = usable_target.unique().tolist()
    if not 2 <= len(unique_values) <= 20:
        return empty_overview, pd.DataFrame(), []

    positive_label = _infer_positive_label(target)
    positive = target.eq(positive_label)
    usable_mask = target.notna()
    usable_rows = int(usable_mask.sum())
    positive_rows = int((positive & usable_mask).sum())
    overview = {
        "target_column": target_col,
        "positive_label": positive_label,
        "usable_rows": usable_rows,
        "missing_rows": int(target.isna().sum()),
        "positive_rows": positive_rows,
        "positive_rate": round(_safe_ratio(positive_rows, usable_rows), 6),
    }
    if len(unique_values) != 2:
        return overview, pd.DataFrame(), []

    rows = []
    minimum_support = max(10, int(math.ceil(usable_rows * 0.002)))
    for column in df.columns:
        if column == target_col or _is_identifier_like(df[column], column) or _is_datetime_candidate(df[column]):
            continue
        feature = df[column]
        unique_count = int(feature.nunique(dropna=True))
        categorical = _is_string_like(feature) or pd.api.types.is_bool_dtype(feature) or unique_count <= 12
        if categorical and 2 <= unique_count <= 20:
            frame = pd.DataFrame({"feature": feature, "positive": positive, "usable": usable_mask})
            frame = frame[frame["usable"]].dropna(subset=["feature"])
            grouped = frame.groupby("feature", observed=True)["positive"].agg(["count", "sum", "mean"])
            grouped = grouped[grouped["count"] >= minimum_support]
            if len(grouped) < 2:
                continue
            effect = float(grouped["mean"].max() - grouped["mean"].min())
            for level, values in grouped.iterrows():
                rows.append(
                    {
                        "association_kind": "categorical_rate",
                        "feature": column,
                        "level": level,
                        "row_count": int(values["count"]),
                        "target_count": int(values["sum"]),
                        "target_rate": float(values["mean"]),
                        "overall_target_rate": _safe_ratio(positive_rows, usable_rows),
                        "rate_difference": float(values["mean"]) - _safe_ratio(positive_rows, usable_rows),
                        "group_mean": np.nan,
                        "comparison_mean": np.nan,
                        "signed_effect": np.nan,
                        "effect_size": effect,
                        "interpretation": "Descriptive association; this does not establish causation.",
                    }
                )
            continue

        if not pd.api.types.is_numeric_dtype(feature):
            continue
        frame = pd.DataFrame(
            {"feature": pd.to_numeric(feature, errors="coerce"), "positive": positive, "usable": usable_mask}
        )
        frame = frame[frame["usable"]].dropna(subset=["feature"])
        positive_values = frame.loc[frame["positive"], "feature"]
        comparison_values = frame.loc[~frame["positive"], "feature"]
        if len(positive_values) < 20 or len(comparison_values) < 20:
            continue
        pooled_variance = (
            (len(positive_values) - 1) * positive_values.var()
            + (len(comparison_values) - 1) * comparison_values.var()
        ) / max(len(frame) - 2, 1)
        pooled_scale = math.sqrt(pooled_variance) if pooled_variance > 0 else 0.0
        signed_effect = (
            float((positive_values.mean() - comparison_values.mean()) / pooled_scale)
            if pooled_scale
            else 0.0
        )
        rows.append(
            {
                "association_kind": "numeric_difference",
                "feature": column,
                "level": f"{positive_label} versus other target rows",
                "row_count": int(len(frame)),
                "target_count": int(len(positive_values)),
                "target_rate": _safe_ratio(len(positive_values), len(frame)),
                "overall_target_rate": _safe_ratio(positive_rows, usable_rows),
                "rate_difference": np.nan,
                "group_mean": float(positive_values.mean()),
                "comparison_mean": float(comparison_values.mean()),
                "signed_effect": signed_effect,
                "effect_size": abs(signed_effect),
                "interpretation": "Standardized descriptive difference; this does not establish causation.",
            }
        )

    if not rows:
        return overview, pd.DataFrame(), []
    associations = pd.DataFrame(rows).sort_values(
        ["effect_size", "association_kind", "feature", "target_rate"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)
    associations = associations.round(
        {
            "target_rate": 4,
            "overall_target_rate": 4,
            "rate_difference": 4,
            "group_mean": 3,
            "comparison_mean": 3,
            "signed_effect": 3,
            "effect_size": 3,
        }
    )

    highlights = []
    categorical_rows = associations[associations["association_kind"] == "categorical_rate"]
    if not categorical_rows.empty:
        perfect_levels = categorical_rows[
            categorical_rows["target_rate"].le(0.001)
            | categorical_rows["target_rate"].ge(0.999)
        ].sort_values(["row_count", "effect_size"], ascending=[False, False])
        if not perfect_levels.empty:
            perfect = perfect_levels.iloc[0]
            highlights.append(
                f"Potential leakage or distribution-shift warning: {perfect['feature']}={perfect['level']} has "
                f"{int(perfect['target_count']):,} of {int(perfect['row_count']):,} positive {target_col} rows "
                f"({float(perfect['target_rate']):.1%}). Investigate this perfect descriptive association "
                "before relying on it."
            )
        feature_effects = categorical_rows.groupby("feature")["effect_size"].max().sort_values(ascending=False)
        top_feature = feature_effects.index[0]
        top_level = categorical_rows[categorical_rows["feature"] == top_feature].sort_values(
            ["target_rate", "row_count"], ascending=[False, False]
        ).iloc[0]
        highlights.append(
            f"{target_col} is associated with {top_feature} in this dataset: {top_level['level']} has "
            f"{int(top_level['target_count']):,} of {int(top_level['row_count']):,} positive rows "
            f"({float(top_level['target_rate']):.1%}) versus {overview['positive_rate']:.1%} overall. "
            "This is descriptive, not causal."
        )
    else:
        top = associations.iloc[0]
        highlights.append(
            f"{target_col} is associated with {top['feature']} in this dataset "
            f"(standardized difference {float(top['signed_effect']):.2f}). This is descriptive, not causal."
        )
    return overview, associations, highlights


def _build_responsible_use_notices(df, target_col=None):
    sensitive = [
        column
        for column in df.columns
        if any(token in _SENSITIVE_NAMES for token in _normalise_name(column).split())
    ]
    if not sensitive:
        return []
    columns = ", ".join(map(str, sensitive))
    target_context = f" in relation to {target_col}" if target_col else ""
    return [
        f"Sensitive or demographic field(s) detected: {columns}. Associations{target_context} should not be "
        "treated as causal or used for consequential decisions without fairness, legal, and domain review."
    ]


def _build_distribution_headline(categorical_summary, total_rows):
    if categorical_summary.empty:
        return None
    candidates = categorical_summary[
        categorical_summary["unique_values"].between(2, 12)
        & categorical_summary["top_value_count"].gt(0)
    ]
    if candidates.empty:
        return None
    top = candidates.sort_values(["missing_pct", "unique_values", "column"]).iloc[0]
    return (
        f"Most common {top['column']} value: {top['top_value']} "
        f"({int(top['top_value_count']):,} of {total_rows:,} rows, {float(top['top_value_pct']):.1f}%)."
    )


def _build_multi_value_headlines(summary):
    if summary.empty:
        return []
    column_rank = (
        summary.groupby("column", as_index=False)
        .agg(unique_tokens=("unique_tokens", "max"), represented_rows=("represented_rows", "max"))
        .sort_values(["unique_tokens", "represented_rows", "column"], ascending=[True, False, True])
    )
    headlines = []
    for column in column_rank["column"].head(2):
        top = summary[summary["column"] == column].iloc[0]
        headlines.append(
            f"Leading {column} entry after splitting comma-separated values: {top['value']} "
            f"({int(top['row_count']):,} rows)."
        )
    return headlines


def _build_unit_headline(unit_summary):
    if unit_summary.empty:
        return None
    column_counts = unit_summary.groupby("column")["row_count"].sum().sort_values(ascending=False)
    column = column_counts.index[0]
    rows = unit_summary[unit_summary["column"] == column]
    details = [
        f"{int(row.row_count):,} {row.unit} rows (median {float(row.median):g})"
        for row in rows.itertuples()
    ]
    return f"{column} contains separate numeric units: " + "; ".join(details) + "."


def _build_headline_insights(
    overview,
    categorical_summary,
    correlations,
    group_summary,
    trend_summary,
    anomaly_summary,
    multi_value_summary,
    unit_summary,
    target_highlights,
):
    insights = [
        f"Dataset contains {overview['rows']:,} rows and {overview['columns']:,} columns "
        f"with {overview['missing_cells']:,} missing cells."
    ]
    if overview["top_missing_column"]:
        insights.append(
            f"Most incomplete column: {overview['top_missing_column']} "
            f"({overview['top_missing_pct']:.1f}% missing)."
        )
    insights.extend(target_highlights[:1])

    distribution = _build_distribution_headline(categorical_summary, overview["rows"])
    if distribution:
        insights.append(distribution)
    insights.extend(_build_multi_value_headlines(multi_value_summary))
    unit_headline = _build_unit_headline(unit_summary)
    if unit_headline:
        insights.append(unit_headline)

    eligible = correlations[correlations["headline_eligible"]] if not correlations.empty else correlations
    if not eligible.empty:
        top_corr = eligible.iloc[0]
        insights.append(
            f"Supported numeric association: {top_corr['column_a']} and {top_corr['column_b']} "
            f"(correlation {top_corr['correlation']:.3f}, paired n={int(top_corr['paired_n']):,}, "
            f"coverage {float(top_corr['coverage_pct']):.1f}%, adjusted p={float(top_corr['adjusted_p_value']):.3g})."
        )
    if trend_summary is not None:
        insights.append(trend_summary["description"])
    if not anomaly_summary.empty:
        insights.append(
            f"{len(anomaly_summary):,} sampled row(s) crossed the documented robust anomaly threshold "
            f"of {ANOMALY_THRESHOLD:g}."
        )
    return insights[:9]


def _build_data_quality_summary(column_inspection, overview):
    if column_inspection.empty:
        return ["No columns are available for data-quality inspection."]
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
    if overview["duplicate_rows"]:
        summaries.append(f"Found {overview['duplicate_rows']:,} exact duplicate row(s).")
    if overview["constant_columns"]:
        summaries.append(
            f"{overview['constant_columns']:,} column(s) are constant among their non-missing values."
        )
    avoid_target_count = int((column_inspection["recommendation"] == "Avoid as a prediction target").sum())
    if avoid_target_count:
        summaries.append(
            f"{avoid_target_count} column(s) look more like identifiers than meaningful prediction targets."
        )
    return summaries[:5]


def _build_grouping_highlight(group_summary):
    if group_summary.empty:
        return None
    top = group_summary.iloc[0]
    highest_group = group_summary.iloc[0]["group"]
    lowest_group = group_summary.iloc[-1]["group"]
    return (
        f"Descriptive comparison: average {top['metric_column']} differs across the shown "
        f"{top['category_column']} groups (highest: {highest_group}; lowest: {lowest_group}). "
        "This association is not evidence of causation or statistical significance."
    )


def _build_anomaly_highlight(anomaly_summary):
    if anomaly_summary.empty:
        return None
    top = anomaly_summary.iloc[0]
    return (
        f"Row {top['row_index']} crossed the robust anomaly threshold of {ANOMALY_THRESHOLD:g} "
        f"with score {float(top['anomaly_score']):.3f}; review it in source context before acting."
    )


def recommend_analysis_paths(df, target_col=None, analysis_artifacts=None):
    artifacts = analysis_artifacts or {}
    correlations = artifacts.get("correlations")
    trend_summary = artifacts.get("trend_summary")
    group_summary = artifacts.get("group_summary")
    anomaly_summary = artifacts.get("anomaly_summary")
    multi_value_summary = artifacts.get("multi_value_summary")
    unit_summary = artifacts.get("unit_summary")
    target_associations = artifacts.get("target_associations")

    if correlations is None:
        correlations = _build_correlations(df)
    if trend_summary is None and "trend_summary" not in artifacts:
        trend_summary = _build_trend_summary(df)
    if group_summary is None:
        group_summary = _build_group_summary(df, target_col=target_col)
    if anomaly_summary is None:
        anomaly_summary = _build_anomaly_summary(df)
    if multi_value_summary is None:
        multi_value_summary = _build_multi_value_summary(df)
    if unit_summary is None:
        unit_summary = _build_unit_summary(df)
    if target_associations is None:
        target_associations = pd.DataFrame()

    paths = []
    missing_pct = _safe_ratio(int(df.isna().sum().sum()), max(int(df.size), 1)) * 100
    quality_score = 45 + min(missing_pct, 25)
    paths.append(
        {
            "analysis_type": "Data quality and coverage",
            "reason": "Exact missingness, uniqueness, duplicate, constant-column, and type-conversion checks are available.",
            "score": quality_score,
        }
    )

    if not target_associations.empty:
        top_effect = float(target_associations["effect_size"].max())
        paths.append(
            {
                "analysis_type": "Target-aware association analysis",
                "reason": f"Supported descriptive comparisons are available for {target_col}; wording remains non-causal.",
                "score": 85 + min(top_effect * 10, 10),
            }
        )
    if not multi_value_summary.empty:
        paths.append(
            {
                "analysis_type": "Multi-value category analysis",
                "reason": "Comma-separated fields can be split to count the categories represented across rows.",
                "score": 78,
            }
        )
    if not unit_summary.empty:
        paths.append(
            {
                "analysis_type": "Unit-aware numeric analysis",
                "reason": "Numeric values with different units are summarized separately to avoid invalid averages.",
                "score": 77,
            }
        )
    if trend_summary is not None:
        paths.append(
            {
                "analysis_type": "Trend analysis",
                "reason": f"{trend_summary['date_column']} supports count-based complete-period comparisons.",
                "score": 74 if len(trend_summary["frame"]) >= 2 else 55,
            }
        )

    eligible_correlations = (
        correlations[correlations["headline_eligible"]] if not correlations.empty else pd.DataFrame()
    )
    if not eligible_correlations.empty:
        paths.append(
            {
                "analysis_type": "Correlation analysis",
                "reason": "At least one numeric pair meets the effect-size, paired-support, coverage, and adjusted-significance gates.",
                "score": 70 + min(float(eligible_correlations["abs_correlation"].max()) * 10, 10),
            }
        )
    if not group_summary.empty:
        paths.append(
            {
                "analysis_type": "Descriptive grouping analysis",
                "reason": "Supported category groups can be compared descriptively without implying significance or causation.",
                "score": 58 + min(float(group_summary["standardized_spread"].iloc[0]) * 5, 8),
            }
        )
    if not anomaly_summary.empty:
        paths.append(
            {
                "analysis_type": "Robust anomaly review",
                "reason": f"At least one sampled row crosses the documented robust-z threshold of {ANOMALY_THRESHOLD:g}.",
                "score": 56,
            }
        )
    if _numeric_columns(df):
        paths.append(
            {
                "analysis_type": "Descriptive statistics",
                "reason": "Numeric distributions can be summarized with counts, quantiles, and missingness.",
                "score": 48,
            }
        )

    recommendations = pd.DataFrame(paths).sort_values(
        ["score", "analysis_type"], ascending=[False, True]
    ).reset_index(drop=True)
    recommendations.insert(0, "rank", np.arange(1, len(recommendations) + 1))
    return recommendations


def run_insight_analysis(df, target_col=None):
    overview = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "numeric_columns": int(len(df.select_dtypes(include=["number", "bool"]).columns)),
        "duplicate_rows": int(df.duplicated().sum()) if len(df.columns) else 0,
        "constant_columns": int(sum(df[column].nunique(dropna=True) <= 1 for column in df.columns)),
        "top_missing_column": None,
        "top_missing_pct": 0.0,
    }
    if not df.empty:
        missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
        if not missing_pct.empty and missing_pct.iloc[0] > 0:
            overview["top_missing_column"] = missing_pct.index[0]
            overview["top_missing_pct"] = float(missing_pct.iloc[0])

    association_target, target_was_inferred = _infer_association_target(df, target_col)
    column_inspection = build_column_inspection(df)
    numeric_summary = _build_numeric_summary(df)
    categorical_summary = _build_categorical_summary(df)
    correlations = _build_correlations(df)
    group_summary = _build_group_summary(df, target_col=association_target)
    trend_summary = _build_trend_summary(df)
    anomaly_summary = _build_anomaly_summary(df)
    multi_value_summary = _build_multi_value_summary(df)
    unit_summary = _build_unit_summary(df)
    target_overview, target_associations, target_highlights = _build_target_associations(
        df, association_target
    ) if association_target else (None, pd.DataFrame(), [])
    responsible_use_notices = _build_responsible_use_notices(df, association_target)

    artifacts = {
        "correlations": correlations,
        "group_summary": group_summary,
        "trend_summary": trend_summary,
        "anomaly_summary": anomaly_summary,
        "multi_value_summary": multi_value_summary,
        "unit_summary": unit_summary,
        "target_associations": target_associations,
    }
    analysis_recommendations = recommend_analysis_paths(
        df, target_col=association_target, analysis_artifacts=artifacts
    )
    data_quality_summary = _build_data_quality_summary(column_inspection, overview)
    grouping_highlight = _build_grouping_highlight(group_summary)
    anomaly_highlight = _build_anomaly_highlight(anomaly_summary)
    headlines = _build_headline_insights(
        overview,
        categorical_summary,
        correlations,
        group_summary,
        trend_summary,
        anomaly_summary,
        multi_value_summary,
        unit_summary,
        target_highlights,
    )

    return {
        "overview": overview,
        "column_inspection": column_inspection,
        "analysis_recommendations": analysis_recommendations,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "correlations": correlations,
        "correlation_methodology": (
            f"Pearson correlations use at most {MAX_ANALYSIS_ROWS:,} deterministic rows. Headlines require "
            f"paired n >= {CORRELATION_MIN_PAIRED_ROWS}, coverage >= {CORRELATION_MIN_COVERAGE:.0%}, "
            f"|r| >= {CORRELATION_MIN_EFFECT:.2f}, and Benjamini-Hochberg adjusted p <= "
            f"{CORRELATION_MAX_ADJUSTED_P:.2f}."
        ),
        "group_summary": group_summary,
        "trend_summary": trend_summary,
        "anomaly_summary": anomaly_summary,
        "anomaly_methodology": (
            f"Rows are shown only when their maximum median/MAD robust-z score is >= {ANOMALY_THRESHOLD:g} "
            f"across at least two usable numeric dimensions; at most {MAX_ANALYSIS_ROWS:,} deterministic rows are scanned."
        ),
        "multi_value_summary": multi_value_summary,
        "unit_summary": unit_summary,
        "target_overview": target_overview,
        "target_associations": target_associations,
        "target_association_highlights": target_highlights,
        "target_association_warnings": [
            message for message in target_highlights if message.startswith("Potential leakage")
        ],
        "association_target": association_target,
        "association_target_inferred": target_was_inferred,
        "responsible_use_notices": responsible_use_notices,
        "responsible_use_notes": responsible_use_notices,
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

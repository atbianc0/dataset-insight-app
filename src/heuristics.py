import re
import warnings

import numpy as np
import pandas as pd


COMMON_MISSING_TOKENS = {
    "",
    " ",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "unknown",
    "?",
    "-",
    "--",
}


def safe_ratio(numerator, denominator):
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def is_integer_like(series, tolerance=1e-9):
    non_null = pd.Series(series).dropna()
    if non_null.empty:
        return False

    numeric_values = pd.to_numeric(non_null, errors="coerce")
    if numeric_values.isna().any():
        return False

    numeric_array = numeric_values.to_numpy(dtype=float, na_value=np.nan)
    return np.all(np.isclose(numeric_array, np.round(numeric_array), atol=tolerance))


def numeric_conversion_ratio(series):
    non_null = pd.Series(series).dropna()
    if non_null.empty:
        return 0.0
    numeric_values = pd.to_numeric(non_null, errors="coerce")
    return float(numeric_values.notna().mean())


def normalize_missing_tokens(series):
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        normalized = series.astype("string").str.strip()
        return normalized.where(~normalized.str.lower().isin(COMMON_MISSING_TOKENS), pd.NA)
    return series


def is_datetime_candidate(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False

    sample = pd.Series(series).dropna().astype(str).head(150)
    if sample.empty:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.85


def is_identifier_like(series, name):
    non_null = pd.Series(series).dropna()
    if non_null.empty:
        return False

    unique_ratio = safe_ratio(non_null.nunique(), len(non_null))
    lower_name = str(name).lower()
    if "id" in lower_name and unique_ratio > 0.75:
        return True

    if pd.api.types.is_numeric_dtype(non_null):
        if is_integer_like(non_null):
            diffs = pd.Series(non_null).sort_values().diff().dropna()
            mostly_step_one = not diffs.empty and (diffs == 1).mean() > 0.9
            return unique_ratio > 0.98 and non_null.nunique() > 25 and mostly_step_one
        return False

    return unique_ratio > 0.995 and non_null.nunique() > 100


def word_count(series):
    return pd.Series(series).fillna("").astype(str).str.split().str.len()


def is_text_heavy(series):
    sample = pd.Series(series).dropna().astype(str).head(200)
    if sample.empty:
        return False

    return word_count(sample).mean() >= 4 or sample.str.len().mean() >= 30


def tokenize_column_name(name):
    parts = [part.lower() for part in re.split(r"[^A-Za-z0-9]+", str(name)) if part]
    if len(parts) == 1:
        parts = [part.lower() for part in re.findall(r"[A-Z]?[a-z]+|\d+", str(name)) if part] or parts
    return parts


def classify_general_column_role(series, column_name):
    if is_identifier_like(series, column_name):
        return "identifier"
    if is_datetime_candidate(series):
        return "datetime"
    if is_text_heavy(series):
        return "text"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    unique_count = int(pd.Series(series).nunique(dropna=True))
    if 2 <= unique_count <= 20:
        return "categorical"
    return "high_cardinality"


def sample_values(series, max_values=3):
    values = pd.Series(series).dropna().astype(str).unique().tolist()[:max_values]
    return ", ".join(values) if values else ""

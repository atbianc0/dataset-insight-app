import re
import warnings
from typing import Any

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

PREFERRED_POSITIVE_LABELS = (
    "1",
    "true",
    "yes",
    "y",
    "positive",
    "churn",
    "churned",
    "failed",
    "fraud",
    "default",
)


def _positive_label_token(label: Any) -> str:
    token = str(label).strip().lower()
    try:
        if float(token) == 1.0:
            return "1"
    except ValueError:
        pass
    return token


def safe_ratio(numerator, denominator):
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def infer_binary_positive_label(
    series,
    requested_label: Any | None = None,
    *,
    labels: list[Any] | None = None,
) -> Any | None:
    """Resolve one stable positive label for binary targets only.

    Recognizable outcome labels take precedence. Otherwise, the less common
    class is positive; tied classes use a deterministic textual ordering.
    """

    values = pd.Series(series).dropna()
    available = list(labels) if labels is not None else values.unique().tolist()
    if len(available) != 2:
        if requested_label is not None:
            raise ValueError("A positive label override is only valid for binary classification.")
        return None

    if requested_label is not None:
        exact_matches = [label for label in available if label == requested_label]
        if exact_matches:
            return exact_matches[0]
        text_matches = [label for label in available if str(label) == str(requested_label)]
        if len(text_matches) == 1:
            return text_matches[0]
        raise ValueError(f"Positive label {requested_label!r} is not present in the target column.")

    normalized = {
        _positive_label_token(label): label
        for label in sorted(available, key=lambda value: (type(value).__name__, str(value)))
    }
    for preferred in PREFERRED_POSITIVE_LABELS:
        if preferred in normalized:
            return normalized[preferred]

    counts = values.value_counts(dropna=False)
    minimum_count = min(int(counts.get(label, 0)) for label in available)
    least_common = [label for label in available if int(counts.get(label, 0)) == minimum_count]
    return sorted(
        least_common,
        key=lambda value: (type(value).__name__, str(value)),
    )[-1]


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

    sample = pd.Series(series).dropna().head(150).astype(str)
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
    lower_name = str(name).strip().lower()
    compact_name = re.sub(r"[^a-z0-9]", "", lower_name)
    name_tokens = set(tokenize_column_name(name))
    compact_entity_names = {
        "account",
        "customer",
        "employee",
        "event",
        "order",
        "product",
        "record",
        "row",
        "session",
        "show",
        "transaction",
        "user",
    }
    identifier_name = (
        lower_name == "id"
        or lower_name.startswith("id_")
        or lower_name.endswith("_id")
        or (compact_name.startswith("id") and compact_name[2:] in compact_entity_names)
        or (compact_name.endswith("id") and compact_name[:-2] in compact_entity_names)
        or bool(name_tokens & {"id", "uuid", "guid", "identifier"})
    )
    # Explicit ID tokens describe entity keys even when each entity appears on
    # many rows. Requiring near-uniqueness would misclassify repeated customer,
    # account, or session IDs as ordinary model features.
    if identifier_name and non_null.nunique() > 1:
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
    sample = pd.Series(series).dropna().head(200).astype(str)
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
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    unique_count = int(pd.Series(series).nunique(dropna=True))
    if 2 <= unique_count <= 20:
        return "categorical"
    return "high_cardinality"


def sample_values(series, max_values=3):
    if max_values <= 0:
        return ""

    values = []
    seen = set()
    for value in pd.Series(series).array:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            continue
        rendered = str(value)
        if rendered in seen:
            continue
        seen.add(rendered)
        values.append(rendered)
        if len(values) >= max_values:
            break
    return ", ".join(values)

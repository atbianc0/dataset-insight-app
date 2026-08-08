"""Deterministic, reusable dataset profiling.

Dataset-level counts are always exact.  Only work whose cost grows quickly with
row count (relationships, anomalies, and model selection) should use the
bounded ``analysis_frame`` carried by :class:`DatasetProfile`.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np
import pandas as pd

from src.contracts import DatasetProfile
from src.heuristics import (
    classify_general_column_role,
    normalize_missing_tokens,
    numeric_conversion_ratio,
)

DEFAULT_ANALYSIS_ROWS = 10_000
HIGH_CONFIDENCE_OUTCOME_TOKENS = {
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


def _name_tokens(name: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return {
        token.lower()
        for token in re.split(r"[^A-Za-z0-9]+", spaced)
        if token
    }


def sanitize_for_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw table without silently discarding columns."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Dataset profiling requires a pandas DataFrame.")

    cleaned = frame.copy()
    normalized_columns = [str(column).strip() for column in cleaned.columns]
    duplicates = (
        pd.Index(normalized_columns)[pd.Index(normalized_columns).duplicated()]
        .unique()
        .tolist()
    )
    if duplicates:
        raise ValueError(
            "Column names must be unique after trimming whitespace. Duplicate columns: "
            + ", ".join(duplicates[:5])
        )
    cleaned.columns = normalized_columns
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    for column in cleaned.columns:
        series = normalize_missing_tokens(cleaned[column])
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.notna().any() and numeric.notna().mean() >= 0.95:
                series = numeric
        cleaned[column] = series
    return cleaned


def fingerprint_dataframe(frame: pd.DataFrame) -> str:
    """Return a stable content-and-schema fingerprint for cache invalidation."""

    digest = hashlib.sha256()
    digest.update(str(frame.shape).encode("utf-8"))
    digest.update("\x1f".join(map(str, frame.columns)).encode("utf-8"))
    digest.update("\x1f".join(map(str, frame.dtypes)).encode("utf-8"))
    hashed = pd.util.hash_pandas_object(frame, index=True, categorize=True)
    digest.update(hashed.to_numpy(dtype="uint64", copy=False).tobytes())
    return digest.hexdigest()


def _problem_type(series: pd.Series) -> str | None:
    values = series.dropna()
    if len(values) < 20 or values.nunique(dropna=True) < 2:
        return None
    if (
        pd.api.types.is_bool_dtype(values)
        or isinstance(values.dtype, pd.CategoricalDtype)
        or (
            (pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values))
            and numeric_conversion_ratio(values) < 0.95
        )
    ):
        return "classification"
    numeric = pd.to_numeric(values, errors="coerce")
    unique = int(numeric.nunique(dropna=True))
    if unique <= 20 and unique / len(values) <= 0.05:
        return "classification"
    return "regression"


def _target_candidates(frame: pd.DataFrame, roles: dict[str, str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for column in frame.columns:
        values = frame[column].dropna()
        problem_type = _problem_type(frame[column])
        tokens = _name_tokens(column)
        credible_outcome = bool(tokens & HIGH_CONFIDENCE_OUTCOME_TOKENS)
        technically_modelable = (
            problem_type is not None
            and roles[column] not in {"identifier", "datetime", "text"}
            and len(values) >= 20
        )
        candidates.append(
            {
                "column": column,
                "problem_type": problem_type,
                "technically_modelable": technically_modelable,
                "credible_business_outcome": credible_outcome and technically_modelable,
                "auto_select": credible_outcome and technically_modelable,
                "usable_rows": int(len(values)),
                "missing_cells": int(frame[column].isna().sum()),
                "unique_values": int(values.nunique(dropna=True)),
            }
        )
    return sorted(
        candidates,
        key=lambda item: (
            item["auto_select"],
            item["technically_modelable"],
            item["usable_rows"],
        ),
        reverse=True,
    )


def build_dataset_profile(
    frame: pd.DataFrame,
    *,
    fingerprint: str | None = None,
    max_analysis_rows: int = DEFAULT_ANALYSIS_ROWS,
    random_seed: int = 42,
) -> DatasetProfile:
    """Sanitize and profile a table once for reuse across the application."""

    if max_analysis_rows < 1:
        raise ValueError("max_analysis_rows must be positive.")
    sanitized = sanitize_for_profile(frame)
    row_count, column_count = sanitized.shape
    analysis_frame = (
        sanitized
        if row_count <= max_analysis_rows
        else sanitized.sample(n=max_analysis_rows, random_state=random_seed).sort_index()
    )

    roles = {
        column: classify_general_column_role(sanitized[column], column)
        for column in sanitized.columns
    }
    column_profiles: list[dict[str, Any]] = []
    for column in sanitized.columns:
        series = sanitized[column]
        non_null = int(series.notna().sum())
        column_profiles.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "role": roles[column],
                "non_null": non_null,
                "missing_cells": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean() * 100),
                "unique_values": int(series.nunique(dropna=True)),
                "coverage_pct": float(non_null / max(row_count, 1) * 100),
            }
        )

    exact_overview = {
        "rows": int(row_count),
        "columns": int(column_count),
        "cells": int(row_count * column_count),
        "missing_cells": int(sanitized.isna().sum().sum()),
        "duplicate_rows": int(sanitized.duplicated().sum()) if column_count else 0,
        "constant_columns": int(
            sum(sanitized[column].nunique(dropna=True) <= 1 for column in sanitized.columns)
        ),
    }
    warnings: list[str] = []
    if row_count > max_analysis_rows:
        warnings.append(
            f"Expensive analyses use a deterministic {max_analysis_rows:,}-row sample; "
            "dataset-level counts remain exact."
        )

    return DatasetProfile(
        fingerprint=fingerprint or fingerprint_dataframe(sanitized),
        sanitized_frame=sanitized,
        schema={column: str(dtype) for column, dtype in sanitized.dtypes.items()},
        exact_overview=exact_overview,
        column_profiles=column_profiles,
        column_roles=roles,
        analysis_frame=analysis_frame,
        target_candidates=_target_candidates(sanitized, roles),
        warnings=warnings,
    )

import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _is_datetime_candidate(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if pd.api.types.is_numeric_dtype(series):
        return False

    non_null = series.dropna()
    if non_null.empty:
        return False

    sample = non_null.astype(str).head(200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.8


def _expand_datetime_column(series, prefix):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series, errors="coerce")
    return pd.DataFrame(
        {
            f"{prefix}__year": parsed.dt.year,
            f"{prefix}__month": parsed.dt.month,
            f"{prefix}__day": parsed.dt.day,
            f"{prefix}__dayofweek": parsed.dt.dayofweek,
        }
    )


def _is_identifier_like(series, column_name):
    non_null = series.dropna()
    if non_null.empty:
        return False

    unique_ratio = non_null.nunique() / len(non_null)
    lower_name = column_name.lower()

    if "id" in lower_name and unique_ratio > 0.8:
        return True

    if unique_ratio > 0.98 and non_null.nunique() > 20:
        return True

    return False


def prepare_features(
    X,
    target_name=None,
    max_categories=40,
    drop_identifier_columns=True,
):
    prepared = X.copy()
    dropped_columns = []
    transformed_columns = []
    unsupported_columns = []

    for column in list(prepared.columns):
        series = prepared[column]

        if series.isna().all():
            dropped_columns.append((column, "all values missing"))
            prepared = prepared.drop(columns=[column])
            continue

        if series.nunique(dropna=True) <= 1:
            dropped_columns.append((column, "constant values"))
            prepared = prepared.drop(columns=[column])
            continue

        if target_name and column == target_name:
            continue

        if drop_identifier_columns and _is_identifier_like(series, column):
            dropped_columns.append((column, "identifier-like"))
            prepared = prepared.drop(columns=[column])
            continue

        if _is_datetime_candidate(series):
            expanded = _expand_datetime_column(series, column)
            prepared = prepared.drop(columns=[column])
            prepared = pd.concat([prepared, expanded], axis=1)
            transformed_columns.append((column, "expanded datetime parts"))
            continue

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            unique_count = series.nunique(dropna=True)
            if unique_count > max_categories:
                dropped_columns.append((column, f"high cardinality ({unique_count} values)"))
                prepared = prepared.drop(columns=[column])
                unsupported_columns.append(column)

    numeric_cols = prepared.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = prepared.select_dtypes(include=["object", "category"]).columns.tolist()

    return {
        "X": prepared,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "dropped_columns": dropped_columns,
        "transformed_columns": transformed_columns,
        "unsupported_columns": unsupported_columns,
    }


def build_preprocessor(X, numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_transformer, numeric_cols))

    if categorical_cols:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor

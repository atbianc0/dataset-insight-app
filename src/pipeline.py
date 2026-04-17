import os
import re
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    precision_recall_curve,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


MAX_PREVIEW_ROWS = 25
MAX_CHART_ROWS = 5000
BENCHMARK_ROWS = 20000
MAX_TRAIN_ROWS = 60000
HIGH_CARDINALITY_LIMIT = 80
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


def is_integer_like(series, tolerance=1e-9):
    non_null = series.dropna()
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


def sanitize_dataframe(df):
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    cleaned = cleaned.loc[:, ~pd.Index(cleaned.columns).duplicated()].copy()
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


def assert_no_missing_target(y, label):
    missing_count = int(pd.Series(y).isna().sum())
    if missing_count:
        raise ValueError(
            f"The target column still contains {missing_count} missing values after cleaning ({label}). "
            "Please remove or correct blank/invalid target values."
        )


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
    if str(y.dtype) in ["object", "category", "bool"]:
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


def summarize_target_style(y, problem_type):
    unique_count = int(y.nunique(dropna=False))
    if problem_type == "classification":
        label = "categorical classification"
    elif is_integer_like(y) and y.min() >= 0:
        label = "count-style regression"
    else:
        label = "numeric regression"
    return {"label": label, "unique_count": unique_count}


def is_count_regression_target(y, problem_type):
    return (
        problem_type == "regression"
        and is_integer_like(y)
        and y.min() >= 0
    )


def _is_datetime_candidate(series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False

    sample = series.dropna().astype(str).head(150)
    if sample.empty:
        return False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce")
    return parsed.notna().mean() >= 0.85


def _expand_datetime_column(series, name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series, errors="coerce")
    return pd.DataFrame(
        {
            f"{name}__year": parsed.dt.year,
            f"{name}__month": parsed.dt.month,
            f"{name}__day": parsed.dt.day,
            f"{name}__dayofweek": parsed.dt.dayofweek,
        }
    )


def _is_identifier_like(series, name):
    non_null = series.dropna()
    if non_null.empty:
        return False
    unique_ratio = non_null.nunique() / len(non_null)
    lower = name.lower()
    if "id" in lower and unique_ratio > 0.75:
        return True

    if pd.api.types.is_numeric_dtype(non_null):
        if is_integer_like(non_null):
            diffs = pd.Series(non_null).sort_values().diff().dropna()
            mostly_step_one = not diffs.empty and (diffs == 1).mean() > 0.9
            return unique_ratio > 0.98 and non_null.nunique() > 25 and mostly_step_one
        return False

    return unique_ratio > 0.995 and non_null.nunique() > 100


def _split_multi_value_text(series):
    cleaned = series.fillna("").astype(str)
    tokenized = cleaned.str.split(",")
    first_item = tokenized.str[0].str.strip().replace("", np.nan)
    item_count = tokenized.apply(
        lambda values: sum(1 for value in values if str(value).strip()) if isinstance(values, list) else 0
    )
    return first_item, item_count


def _extract_numeric_text_parts(series):
    cleaned = series.fillna("").astype(str).str.strip()
    numeric_text = cleaned.where(
        cleaned.str.match(r"^\s*[-+]?\d*\.?\d+\s*[A-Za-z]+\s*$", na=False)
    )
    numbers = pd.to_numeric(numeric_text.str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce")
    units = numeric_text.str.extract(r"[-+]?\d*\.?\d+\s*([A-Za-z]+)")[0]
    return numbers, units


def _word_count(series):
    return series.fillna("").astype(str).str.split().str.len()


def recommend_target_columns(df, top_n=5):
    recommendations = []

    for column in df.columns:
        series = df[column]
        non_null_ratio = 1 - (series.isna().mean())
        unique_count = series.nunique(dropna=True)
        unique_ratio = unique_count / max(series.dropna().shape[0], 1)
        lower = column.lower()

        score = 0.0
        reasons = []

        if _is_identifier_like(series, column):
            score -= 3
            reasons.append("identifier-like")

        if non_null_ratio < 0.6:
            score -= 1
            reasons.append("many missing values")
        else:
            score += 0.5

        if any(keyword in lower for keyword in ["target", "label", "class", "fraud", "churn", "rating", "score", "amount", "price", "revenue", "sales", "type"]):
            score += 2.5
            reasons.append("name suggests a prediction target")

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series) or str(series.dtype) == "bool":
            if 2 <= unique_count <= 20:
                score += 2
                reasons.append("good classification candidate")
            elif unique_count > 100:
                score -= 1.5
                reasons.append("too many categories")
        elif pd.api.types.is_numeric_dtype(series):
            if is_integer_like(series) and 2 <= unique_count <= 8 and unique_ratio < 0.2:
                score += 1.5
                reasons.append("good classification candidate")
            elif unique_count >= 10:
                score += 2
                reasons.append("good regression candidate")
            elif unique_count <= 1:
                score -= 2
                reasons.append("constant or nearly constant")

        if any(keyword in lower for keyword in ["description", "title", "name", "cast", "director"]):
            score -= 1
            reasons.append("often better as a feature than a target")

        recommendations.append(
            {
                "column": column,
                "score": score,
                "reasons": ", ".join(reasons[:3]) if reasons else "general-purpose target candidate",
            }
        )

    frame = pd.DataFrame(recommendations).sort_values(["score", "column"], ascending=[False, True])
    return frame.head(top_n).reset_index(drop=True)


def prepare_training_frame(df, target_col, drop_identifier_columns=True):
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
    dropped_columns = []
    removed_target_rows = initial_row_count - len(cleaned)
    if removed_target_rows:
        notes.append(
            f"Removed {removed_target_rows} rows with missing or invalid target values in {target_col}."
        )

    for column in list(X.columns):
        series = X[column]
        if series.isna().all():
            X = X.drop(columns=[column])
            dropped_columns.append(column)
            notes.append(f"Dropped {column}: all values missing.")
            continue

        if series.nunique(dropna=True) <= 1:
            X = X.drop(columns=[column])
            dropped_columns.append(column)
            notes.append(f"Dropped {column}: constant values.")
            continue

        if drop_identifier_columns and _is_identifier_like(series, column):
            X = X.drop(columns=[column])
            dropped_columns.append(column)
            notes.append(f"Dropped {column}: identifier-like.")
            continue

        if _is_datetime_candidate(series):
            expanded = _expand_datetime_column(series, column)
            X = X.drop(columns=[column])
            X = pd.concat([X, expanded], axis=1)
            notes.append(f"Expanded {column} into datetime features.")
            continue

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            cleaned = series.fillna("").astype(str)

            if cleaned.str.contains(",").mean() >= 0.25:
                first_item, item_count = _split_multi_value_text(series)
                X[f"{column}__item_count"] = item_count
                if first_item.nunique(dropna=True) <= HIGH_CARDINALITY_LIMIT:
                    X[f"{column}__first_item"] = first_item
                else:
                    X[f"{column}__first_item_frequency"] = first_item.map(
                        first_item.value_counts(dropna=False, normalize=True)
                    )
                X = X.drop(columns=[column])
                notes.append(f"Expanded multi-value text column {column}.")
                continue

            numbers, units = _extract_numeric_text_parts(series)
            if numbers.notna().mean() >= 0.8:
                X[f"{column}__number"] = numbers
                if units.nunique(dropna=True) and units.nunique(dropna=True) <= HIGH_CARDINALITY_LIMIT:
                    X[f"{column}__unit"] = units
                X = X.drop(columns=[column])
                notes.append(f"Extracted numeric text features from {column}.")
                continue

            if _word_count(series).mean() >= 4:
                X[f"{column}__word_count"] = _word_count(series)
                X = X.drop(columns=[column])
                notes.append(f"Converted long text column {column} into a length feature.")
                continue

            cardinality = int(series.nunique(dropna=True))
            if cardinality > HIGH_CARDINALITY_LIMIT:
                frequency = series.map(series.value_counts(dropna=False, normalize=True))
                encoded_name = f"{column}__frequency"
                X[encoded_name] = frequency.fillna(0.0)
                X = X.drop(columns=[column])
                notes.append(
                    f"Encoded {column} as frequency feature because cardinality was {cardinality}."
                )

    X = X.replace([np.inf, -np.inf], np.nan)
    empty_after_processing = [column for column in X.columns if X[column].isna().all()]
    if empty_after_processing:
        X = X.drop(columns=empty_after_processing)
        dropped_columns.extend(empty_after_processing)
        notes.extend(
            [f"Dropped {column}: empty after cleaning or feature extraction." for column in empty_after_processing]
        )

    if X.empty:
        raise ValueError("No usable feature columns remained after preparation.")

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if not numeric_cols and not categorical_cols:
        raise ValueError("No supported numeric or categorical features were found.")

    return {
        "X": X,
        "y": y,
        "notes": notes,
        "dropped_columns": dropped_columns,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }


def sample_training_data(X, y, problem_type, max_rows=MAX_TRAIN_ROWS, random_state=42):
    if len(X) <= max_rows:
        return X, y, False

    if problem_type == "classification":
        combined = pd.concat([X, y.rename("__target__")], axis=1)
        base = combined.groupby("__target__", group_keys=False).apply(
            lambda frame: frame.sample(n=1, random_state=random_state)
        )
        remaining = max_rows - len(base)
        if remaining <= 0:
            sampled = base
        else:
            rest = combined.drop(index=base.index)
            stratify = rest["__target__"] if rest["__target__"].nunique() > 1 else None
            sampled_rest = rest.sample(
                n=min(remaining, len(rest)),
                random_state=random_state,
            ) if stratify is None else train_test_split(
                rest,
                train_size=min(remaining, len(rest)),
                random_state=random_state,
                stratify=stratify,
            )[0]
            sampled = pd.concat([base, sampled_rest], axis=0)

        sampled_y = sampled["__target__"]
        sampled_X = sampled.drop(columns=["__target__"])
        return sampled_X, sampled_y, True

    sampled_idx = X.sample(n=max_rows, random_state=random_state).index
    return X.loc[sampled_idx], y.loc[sampled_idx], True


def build_preprocessor(numeric_cols, categorical_cols, categorical_strategy="ordinal"):
    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0))]),
                numeric_cols,
            )
        )

    if categorical_cols:
        encoder = (
            OneHotEncoder(handle_unknown="ignore")
            if categorical_strategy == "onehot"
            else OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
        )
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                        ("encoder", encoder),
                    ]
                ),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers)


def get_candidate_models(problem_type, target_style_label=None, effort="standard", imbalance_ratio=None):
    if problem_type == "classification":
        models = {
            "Gradient Boosting": {
                "model": HistGradientBoostingClassifier(
                    random_state=42,
                    max_leaf_nodes=31,
                ),
                "categorical_strategy": "ordinal",
            },
            "Random Forest": {
                "model": RandomForestClassifier(
                    n_estimators=120,
                    random_state=42,
                    n_jobs=1,
                    class_weight="balanced_subsample",
                ),
                "categorical_strategy": "ordinal",
            },
        }
        if imbalance_ratio is not None and imbalance_ratio < 0.1:
            models["Balanced Random Forest"] = {
                "model": RandomForestClassifier(
                    n_estimators=220,
                    random_state=42,
                    n_jobs=1,
                    class_weight="balanced",
                    min_samples_leaf=2,
                ),
                "categorical_strategy": "ordinal",
            }
        if effort == "expanded":
            models["Gradient Boosting Deep"] = {
                "model": HistGradientBoostingClassifier(
                    random_state=42,
                    max_leaf_nodes=63,
                    learning_rate=0.06,
                    min_samples_leaf=30,
                ),
                "categorical_strategy": "ordinal",
            }
            models["Extra Trees"] = {
                "model": ExtraTreesClassifier(
                    n_estimators=160,
                    random_state=42,
                    n_jobs=1,
                    class_weight="balanced_subsample",
                ),
                "categorical_strategy": "ordinal",
            }
            models["Logistic Regression"] = {
                "model": LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight="balanced" if imbalance_ratio is not None and imbalance_ratio < 0.1 else None,
                ),
                "categorical_strategy": "onehot",
            }
        return models
    models = {
        "Gradient Boosting": {
            "model": HistGradientBoostingRegressor(
                random_state=42,
                max_leaf_nodes=31,
            ),
            "categorical_strategy": "ordinal",
        },
        "Random Forest": {
            "model": RandomForestRegressor(
                n_estimators=160,
                random_state=42,
                n_jobs=1,
            ),
            "categorical_strategy": "ordinal",
        },
        "Ridge Regression": {
            "model": Ridge(alpha=1.0),
            "categorical_strategy": "onehot",
        },
    }
    if effort == "expanded":
        models["Gradient Boosting Deep"] = {
            "model": HistGradientBoostingRegressor(
                random_state=42,
                max_leaf_nodes=63,
                learning_rate=0.05,
                min_samples_leaf=20,
            ),
            "categorical_strategy": "ordinal",
        }
        models["Extra Trees"] = {
            "model": ExtraTreesRegressor(
                n_estimators=120,
                random_state=42,
                n_jobs=1,
            ),
            "categorical_strategy": "ordinal",
        }
        if target_style_label == "count-style regression":
            models["Poisson Gradient Boosting"] = {
                "model": HistGradientBoostingRegressor(
                    random_state=42,
                    loss="poisson",
                    max_leaf_nodes=63,
                    learning_rate=0.05,
                    min_samples_leaf=20,
                ),
                "categorical_strategy": "ordinal",
            }
    return models


def evaluate_predictions(problem_type, y_true, y_pred):
    if problem_type == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def add_probability_metrics(problem_type, metrics, y_true, probabilities):
    if problem_type != "classification" or probabilities is None:
        return metrics

    unique_classes = pd.Series(y_true).dropna().unique()
    if len(unique_classes) != 2:
        return metrics

    try:
        positive_scores = probabilities[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_true, positive_scores))
        metrics["average_precision"] = float(average_precision_score(y_true, positive_scores))
    except Exception:
        return metrics
    return metrics


def build_baseline_metrics(problem_type, y_true):
    if problem_type == "classification":
        majority_class = y_true.mode(dropna=False).iloc[0]
        baseline_pred = pd.Series([majority_class] * len(y_true), index=y_true.index)
        metrics = evaluate_predictions(problem_type, y_true, baseline_pred)
        if y_true.nunique(dropna=False) == 2:
            class_counts = y_true.value_counts(normalize=True, dropna=False)
            positive_label = class_counts.idxmin()
            positive_rate = float(class_counts.loc[positive_label])
            binary_true = (
                pd.Series(y_true).reset_index(drop=True).eq(positive_label).fillna(False).astype(int)
            )
            baseline_scores = np.full(len(binary_true), positive_rate)
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, baseline_pred))
            metrics["average_precision"] = float(average_precision_score(binary_true, baseline_scores))
            metrics["roc_auc"] = 0.5
        metrics["baseline_strategy"] = f"Predict majority class ({majority_class})"
        return metrics

    mean_value = float(y_true.mean())
    baseline_pred = pd.Series([mean_value] * len(y_true), index=y_true.index)
    metrics = evaluate_predictions(problem_type, y_true, baseline_pred)
    metrics["baseline_strategy"] = f"Predict target mean ({mean_value:.3f})"
    return metrics


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
    categorical_cols = X_sample.select_dtypes(include=["object", "category"]).columns.tolist()
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


def rank_metric(problem_type):
    return "f1" if problem_type == "classification" else "rmse"


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
):
    X = pd.DataFrame(X).reset_index(drop=True)
    _, y, dropped_before_split = filter_valid_target_rows(
        None,
        pd.Series(y).reset_index(drop=True),
        "before train/test split",
    )
    X = X.loc[y.index].reset_index(drop=True)
    y = y.reset_index(drop=True)

    stratify = y if problem_type == "classification" and y.nunique(dropna=False) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    X_train, y_train, dropped_train = filter_valid_target_rows(
        X_train,
        y_train,
        "training split",
        min_rows=max(10, min(20, len(y_train))),
    )
    X_test, y_test, dropped_test = filter_valid_target_rows(
        X_test,
        y_test,
        "test split",
        min_rows=max(5, min(10, len(y_test))),
    )

    imbalance_ratio = None
    if problem_type == "classification":
        class_distribution = y.value_counts(normalize=True, dropna=False)
        if not class_distribution.empty:
            imbalance_ratio = float(class_distribution.min())

    candidates = get_candidate_models(
        problem_type,
        target_style_label=target_style_label,
        effort=effort,
        imbalance_ratio=imbalance_ratio,
    )
    results = {}
    fitted = {}
    thresholded_predictions = {}
    probability_store = {}

    for name, spec in candidates.items():
        estimator = spec["model"]
        categorical_strategy = spec.get("categorical_strategy", "ordinal")
        pipe = Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor(
                        numeric_cols,
                        categorical_cols,
                        categorical_strategy=categorical_strategy,
                    ),
                ),
                ("model", estimator),
            ]
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probabilities = None

        if problem_type == "classification" and hasattr(pipe, "predict_proba"):
            try:
                probabilities = pipe.predict_proba(X_test)
                if probabilities.shape[1] == 2 and imbalance_ratio is not None and imbalance_ratio < 0.1:
                    class_counts = y_train.value_counts(dropna=False)
                    positive_label = class_counts.idxmin()
                    class_labels = pipe.named_steps["model"].classes_
                    positive_index = int(np.where(class_labels == positive_label)[0][0])
                    positive_scores = probabilities[:, positive_index]
                    binary_true = (
                        pd.Series(y_test).reset_index(drop=True).eq(positive_label).fillna(False).astype(int)
                    )
                    precision, recall, thresholds = precision_recall_curve(binary_true, positive_scores)
                    f1_scores = (2 * precision * recall) / np.clip(precision + recall, 1e-9, None)
                    if len(thresholds):
                        best_index = int(np.nanargmax(f1_scores[:-1]))
                        threshold = thresholds[best_index]
                        negative_label = next(label for label in class_labels if label != positive_label)
                        tuned_preds = np.where(positive_scores >= threshold, positive_label, negative_label)
                        preds = pd.Series(tuned_preds, index=X_test.index)
            except Exception:
                probabilities = None

        metrics = evaluate_predictions(problem_type, y_test, preds)
        metrics = add_probability_metrics(problem_type, metrics, y_test, probabilities)
        if problem_type == "classification" and y_test.nunique(dropna=False) == 2:
            metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_test, preds))
        results[name] = metrics
        fitted[name] = pipe
        thresholded_predictions[name] = preds
        probability_store[name] = probabilities

    metric = "average_precision" if problem_type == "classification" and imbalance_ratio is not None and imbalance_ratio < 0.1 else rank_metric(problem_type)
    if metric not in next(iter(results.values())):
        metric = rank_metric(problem_type)
    if problem_type == "classification":
        best_name = max(results, key=lambda model_name: results[model_name][metric])
    else:
        best_name = min(results, key=lambda model_name: results[model_name][metric])

    best_model = fitted[best_name]
    best_preds = thresholded_predictions[best_name]
    return {
        "results": results,
        "best_model_name": best_name,
        "best_model": best_model,
        "best_metrics": results[best_name],
        "metric_name": metric,
        "X_test": X_test,
        "y_test": y_test,
        "preds": best_preds,
        "best_probabilities": probability_store[best_name],
        "imbalance_ratio": imbalance_ratio,
        "dropped_target_rows": {
            "before_split": dropped_before_split,
            "train_split": dropped_train,
            "test_split": dropped_test,
        },
    }


def build_feature_importance(best_model, feature_names):
    estimator = best_model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    frame = pd.DataFrame(
        {"feature": feature_names, "importance": estimator.feature_importances_}
    )
    return frame.sort_values("importance", ascending=False).head(15)


def get_feature_names(numeric_cols, categorical_cols):
    return [f"num__{col}" for col in numeric_cols] + [f"cat__{col}" for col in categorical_cols]


def align_prediction_frame(prediction_df, feature_columns):
    missing = [col for col in feature_columns if col not in prediction_df.columns]
    if missing:
        raise ValueError("Prediction dataset is missing required columns: " + ", ".join(missing))
    return prediction_df[feature_columns].copy()


def run_analysis(
    df,
    target_col,
    problem_type_mode="Auto Detect",
    test_size=0.2,
    drop_identifier_columns=True,
    training_effort="standard",
):
    prepared = prepare_training_frame(df, target_col, drop_identifier_columns=drop_identifier_columns)
    X = prepared["X"]
    y = prepared["y"]

    problem_type = detect_problem_type(y) if problem_type_mode == "Auto Detect" else problem_type_mode.lower()
    target_style = summarize_target_style(y, problem_type)

    sampled_X, sampled_y, sampled = sample_training_data(X, y, problem_type)
    sampled_numeric = [col for col in prepared["numeric_cols"] if col in sampled_X.columns]
    sampled_categorical = [col for col in prepared["categorical_cols"] if col in sampled_X.columns]

    trained = train_best_model(
        sampled_X,
        sampled_y,
        problem_type,
        sampled_numeric,
        sampled_categorical,
        target_style_label=target_style["label"],
        effort=training_effort,
        test_size=test_size,
    )

    feature_names = get_feature_names(sampled_numeric, sampled_categorical)
    feature_importance = build_feature_importance(trained["best_model"], feature_names)
    baseline_metrics = build_baseline_metrics(problem_type, trained["y_test"])
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

    return {
        "problem_type": problem_type,
        "target_style": target_style,
        "results": trained["results"],
        "best_model_name": trained["best_model_name"],
        "best_model": trained["best_model"],
        "best_metrics": trained["best_metrics"],
        "baseline_metrics": baseline_metrics,
        "quality": quality,
        "metric_name": trained["metric_name"],
        "feature_columns": sampled_X.columns.tolist(),
        "feature_importance": feature_importance,
        "chart_context": chart_context,
        "prediction_preview": prediction_preview.head(MAX_PREVIEW_ROWS),
        "notes": notes,
        "training_effort": training_effort,
        "used_rows": int(len(sampled_X)),
        "original_rows": int(len(X)),
        "original_columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "target_series": y.reset_index(drop=True),
        "holdout_actual": trained["y_test"].reset_index(drop=True),
        "holdout_pred": pd.Series(trained["preds"]).reset_index(drop=True),
    }

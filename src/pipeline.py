import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


MAX_PREVIEW_ROWS = 25
MAX_CHART_ROWS = 5000
BENCHMARK_ROWS = 20000
MAX_TRAIN_ROWS = 60000
HIGH_CARDINALITY_LIMIT = 80


def is_integer_like(series, tolerance=1e-9):
    non_null = series.dropna()
    if non_null.empty or not pd.api.types.is_numeric_dtype(non_null):
        return False
    return np.all(np.isclose(non_null, np.round(non_null), atol=tolerance))


def detect_problem_type(y):
    if str(y.dtype) in ["object", "category", "bool"]:
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
    return ("id" in lower and unique_ratio > 0.75) or (unique_ratio > 0.98 and non_null.nunique() > 25)


def prepare_training_frame(df, target_col, drop_identifier_columns=True):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found.")

    cleaned = df.dropna(subset=[target_col]).copy()
    if len(cleaned) < 20:
        raise ValueError("Please upload at least 20 rows with a target value.")

    X = cleaned.drop(columns=[target_col]).copy()
    y = cleaned[target_col].copy()

    notes = []
    dropped_columns = []

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
            cardinality = int(series.nunique(dropna=True))
            if cardinality > HIGH_CARDINALITY_LIMIT:
                X = X.drop(columns=[column])
                dropped_columns.append(column)
                notes.append(f"Dropped {column}: high cardinality ({cardinality}).")

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


def build_preprocessor(numeric_cols, categorical_cols):
    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers)


def get_candidate_models(problem_type):
    if problem_type == "classification":
        return {
            "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=120,
                random_state=42,
                n_jobs=1,
                class_weight="balanced_subsample",
            ),
        }
    return {
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=120,
            random_state=42,
            n_jobs=1,
        ),
    }


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


def build_baseline_metrics(problem_type, y_true):
    if problem_type == "classification":
        majority_class = y_true.mode(dropna=False).iloc[0]
        baseline_pred = pd.Series([majority_class] * len(y_true), index=y_true.index)
        metrics = evaluate_predictions(problem_type, y_true, baseline_pred)
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
        if f1_gain >= 0.15:
            verdict = "strong"
        elif f1_gain >= 0.05:
            verdict = "useful"
        else:
            verdict = "weak"
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
    if r2_value >= 0.5 or rmse_improvement > 1.0:
        verdict = "strong"
    elif r2_value >= 0.15 or rmse_improvement > 0.25:
        verdict = "useful"
    else:
        verdict = "weak"
    summary = (
        f"Model worth: {verdict}. RMSE improved by {rmse_improvement:.3f} over the "
        f"baseline and R^2 is {r2_value:.3f}."
    )
    return {
        "verdict": verdict,
        "summary": summary,
        "primary_delta": rmse_improvement,
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


def train_best_model(X, y, problem_type, numeric_cols, categorical_cols, test_size=0.2, random_state=42):
    stratify = y if problem_type == "classification" and y.nunique(dropna=False) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    candidates = get_candidate_models(problem_type)
    results = {}
    fitted = {}

    for name, estimator in candidates.items():
        pipe = Pipeline(
            [
                ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
                ("model", estimator),
            ]
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results[name] = evaluate_predictions(problem_type, y_test, preds)
        fitted[name] = pipe

    metric = rank_metric(problem_type)
    if problem_type == "classification":
        best_name = max(results, key=lambda model_name: results[model_name][metric])
    else:
        best_name = min(results, key=lambda model_name: results[model_name][metric])

    best_model = fitted[best_name]
    best_preds = best_model.predict(X_test)
    return {
        "results": results,
        "best_model_name": best_name,
        "best_model": best_model,
        "best_metrics": results[best_name],
        "metric_name": metric,
        "X_test": X_test,
        "y_test": y_test,
        "preds": best_preds,
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

    prediction_preview = sampled_X.loc[trained["X_test"].index].copy().reset_index(drop=True)
    prediction_preview["actual"] = trained["y_test"].reset_index(drop=True)
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
        "used_rows": int(len(sampled_X)),
        "original_rows": int(len(X)),
        "original_columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "target_series": y.reset_index(drop=True),
        "holdout_actual": trained["y_test"].reset_index(drop=True),
        "holdout_pred": pd.Series(trained["preds"]).reset_index(drop=True),
    }

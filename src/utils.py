import pandas as pd
from sklearn.pipeline import Pipeline

from src.evaluate import build_diagnostic_artifacts, evaluate_model, summarize_target
from src.preprocess import build_preprocessor, prepare_features
from src.train import detect_problem_type, get_models, split_data


def choose_best_model(problem_type, results, ranking_metric=None):
    if ranking_metric is None:
        ranking_metric = "f1" if problem_type == "classification" else "rmse"
    elif ranking_metric not in next(iter(results.values())):
        ranking_metric = "f1" if problem_type == "classification" else "rmse"

    best_name = None
    best_score = None

    for name, metrics in results.items():
        score = metrics[ranking_metric]

        if problem_type == "classification":
            if best_score is None or score > best_score:
                best_score = score
                best_name = name
        else:
            if ranking_metric in {"rmse", "mae"}:
                if best_score is None or score < best_score:
                    best_score = score
                    best_name = name
            elif best_score is None or score > best_score:
                best_score = score
                best_name = name

    return best_name


def validate_training_frame(df, target_col):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' was not found in the dataset.")

    cleaned_df = df.dropna(subset=[target_col]).copy()
    if cleaned_df.empty:
        raise ValueError("The selected target column only contains missing values.")

    if len(cleaned_df) < 20:
        raise ValueError("Please upload at least 20 complete rows for a stable training run.")

    X = cleaned_df.drop(columns=[target_col])
    if X.empty:
        raise ValueError("The dataset needs at least one feature column besides the target.")

    return X, cleaned_df[target_col], cleaned_df


def align_prediction_frame(prediction_df, feature_columns):
    aligned = prediction_df.copy()
    missing_cols = [col for col in feature_columns if col not in aligned.columns]
    if missing_cols:
        raise ValueError(
            "Prediction dataset is missing required feature columns: "
            + ", ".join(missing_cols)
        )

    extra_cols = [col for col in aligned.columns if col not in feature_columns]
    if extra_cols:
        aligned = aligned.drop(columns=extra_cols)

    return aligned[feature_columns]


def build_feature_importance_frame(model, feature_names):
    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        importance_values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        importance_values = abs(coef[0]) if getattr(coef, "ndim", 1) > 1 else abs(coef)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance_values}
    )
    return importance_df.sort_values("importance", ascending=False).head(20)


def profile_dataset(df, target_col=None):
    missing_by_column = df.isna().sum().sort_values(ascending=False)
    notes = []

    if target_col is not None:
        target_missing = int(df[target_col].isna().sum())
        if target_missing:
            notes.append(f"Removed {target_missing} rows with missing target values.")

    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_by_column": missing_by_column,
        "notes": notes,
        "dropped_column_names": [],
    }


def run_experiment(
    df,
    target_col,
    problem_type_mode="Auto Detect",
    ranking_metric=None,
    test_size=0.2,
    random_state=42,
    drop_identifier_columns=True,
    max_categories=40,
):
    X_raw, y, cleaned_df = validate_training_frame(df, target_col)
    problem_type = (
        detect_problem_type(y)
        if problem_type_mode == "Auto Detect"
        else problem_type_mode.lower()
    )

    if problem_type == "classification" and y.nunique(dropna=False) < 2:
        raise ValueError("Classification needs at least two target classes to train a model.")

    prepared = prepare_features(
        X_raw,
        target_name=target_col,
        max_categories=max_categories,
        drop_identifier_columns=drop_identifier_columns,
    )
    X = prepared["X"]
    if X.empty:
        raise ValueError("No usable feature columns remained after preprocessing.")

    if not prepared["numeric_cols"] and not prepared["categorical_cols"]:
        raise ValueError("No supported feature columns were found in the uploaded dataset.")

    dataset_profile = profile_dataset(cleaned_df, target_col=target_col)
    dataset_profile["notes"].extend(
        [f"Dropped {col}: {reason}" for col, reason in prepared["dropped_columns"]]
    )
    dataset_profile["notes"].extend(
        [f"Transformed {col}: {reason}" for col, reason in prepared["transformed_columns"]]
    )
    dataset_profile["dropped_column_names"] = [col for col, _ in prepared["dropped_columns"]]

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        problem_type,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(X, prepared["numeric_cols"], prepared["categorical_cols"])
    models = get_models(problem_type)

    results = {}
    fitted_models = {}

    for name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results[name] = evaluate_model(problem_type, y_test, preds)
        fitted_models[name] = pipe

    best_model_name = choose_best_model(problem_type, results, ranking_metric=ranking_metric)
    best_model = fitted_models[best_model_name]
    transformed_feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
    holdout_preds = best_model.predict(X_test)

    probability_frame = pd.DataFrame()
    if problem_type == "classification" and hasattr(best_model, "predict_proba"):
        proba = best_model.predict_proba(X_test)
        probability_columns = [
            f"probability_{class_name}" for class_name in best_model.named_steps["model"].classes_
        ]
        probability_frame = pd.DataFrame(proba, columns=probability_columns, index=X_test.index)

    return {
        "problem_type": problem_type,
        "results": results,
        "models": fitted_models,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "best_metrics": results[best_model_name],
        "X_test": X_test,
        "y_test": y_test,
        "holdout_predictions": pd.Series(holdout_preds, index=X_test.index),
        "probability_frame": probability_frame,
        "feature_columns": X.columns.tolist(),
        "feature_importance": build_feature_importance_frame(best_model, transformed_feature_names),
        "profile": dataset_profile,
        "target_summary": summarize_target(y, problem_type),
        "diagnostics": build_diagnostic_artifacts(problem_type, y_test, holdout_preds),
    }

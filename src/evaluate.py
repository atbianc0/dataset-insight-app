import numpy as np
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


def evaluate_model(problem_type, y_true, y_pred):
    if problem_type == "classification":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def summarize_target(y, problem_type):
    if problem_type == "classification":
        counts = y.value_counts(dropna=False).to_dict()
        return {"class_counts": counts}

    return {
        "mean": float(y.mean()),
        "median": float(y.median()),
        "std": float(y.std()),
    }


def build_diagnostic_artifacts(problem_type, y_true, y_pred):
    if problem_type == "classification":
        labels = sorted(np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])))
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        return {
            "confusion_labels": labels,
            "confusion_matrix": matrix.tolist(),
        }

    residuals = np.asarray(y_true) - np.asarray(y_pred)
    return {
        "residual_min": float(residuals.min()),
        "residual_max": float(residuals.max()),
        "residual_mean": float(residuals.mean()),
    }

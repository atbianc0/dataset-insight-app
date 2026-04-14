from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split


def detect_problem_type(y):
    if str(y.dtype) in ["object", "category", "bool"] or y.nunique(dropna=False) <= 12:
        return "classification"
    return "regression"


def get_models(problem_type):
    if problem_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=2000, solver="liblinear"),
            "Random Forest": RandomForestClassifier(
                n_estimators=250,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            ),
            "Extra Trees": ExtraTreesClassifier(
                n_estimators=250,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            ),
        }

    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=250,
            random_state=42,
            n_jobs=-1,
        ),
    }


def split_data(X, y, problem_type, test_size=0.2, random_state=42):
    if problem_type == "classification":
        class_counts = y.value_counts(dropna=False)
        should_stratify = class_counts.min() >= 2 and y.nunique(dropna=False) > 1
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if should_stratify else None,
        )

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

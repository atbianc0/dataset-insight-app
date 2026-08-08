import string

import numpy as np
import pandas as pd
import pytest

from src.contracts import AnalysisConfig
from src.modeling import (
    RawFeatureTransformer,
    baseline_predictions,
    classification_metrics,
    detect_leakage,
    dtype_family,
    infer_positive_label,
    normalise_target,
    train_model,
)
from src.scoring import score_or_evaluate


def test_raw_transformer_covers_every_supported_feature_family():
    rows = 24
    unit_names = [f"u{letter}" for letter in string.ascii_lowercase[:rows]]
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{index:03d}" for index in range(rows)],
            "all_missing": [None] * rows,
            "constant": ["same"] * rows,
            "numeric": np.square(np.arange(rows)) + 0.25,
            "numeric_text": pd.Series([str(index % 7) for index in range(rows)], dtype="string"),
            "boolean": pd.Series(([True, False] * 12), dtype="boolean"),
            "event_date": pd.date_range("2024-01-01", periods=rows, freq="D").astype(str),
            "multi_low": ["A, B" if index % 2 else "B, C" for index in range(rows)],
            "multi_high": [f"topic_{index}, shared" for index in range(rows)],
            "unit_low": [f"{index + 1} kg" if index % 2 else f"{index + 1} lb" for index in range(rows)],
            "unit_high": [f"{index + 1} {unit_names[index]}" for index in range(rows)],
            "prose": [
                f"This is a sufficiently long sentence with useful descriptive context number {index}."
                for index in range(rows)
            ],
            "high_category": [f"category_{index}" for index in range(rows)],
            "ordinary_category": ["north", "south", "west"] * 8,
        }
    )
    transformer = RawFeatureTransformer(high_cardinality_limit=3).fit(frame)
    by_source = {spec["source"]: spec for spec in transformer.specs_}

    assert dtype_family(frame["boolean"]) == "boolean"
    assert dtype_family(pd.Series(pd.Categorical(["a", "b"]))) == "categorical"
    assert {"customer_id"} == set(transformer.identifier_columns_)
    assert {name for name, _ in transformer.dropped_columns_} >= {
        "customer_id",
        "all_missing",
        "constant",
    }
    assert by_source["numeric"]["kind"] == "numeric"
    assert by_source["numeric_text"]["kind"] == "numeric"
    assert by_source["boolean"]["kind"] == "categorical"
    assert by_source["event_date"]["kind"] == "datetime"
    assert by_source["multi_low"]["first_kind"] == "categorical"
    assert by_source["multi_high"]["first_kind"] == "frequency"
    assert by_source["unit_low"]["include_unit"] is True
    assert by_source["unit_high"]["include_unit"] is False
    assert by_source["prose"]["kind"] == "word_count"
    assert by_source["high_category"]["kind"] == "frequency"
    assert by_source["ordinary_category"]["kind"] == "categorical"

    transformed = transformer.transform(frame)
    assert transformed.columns.tolist() == transformer.get_feature_names_out().tolist()
    assert transformed["multi_low__item_count"].eq(2).all()
    assert transformed["unit_low__number"].notna().all()
    assert transformed["prose__word_count"].ge(10).all()
    assert transformed["event_date__year"].eq(2024).all()

    with pytest.raises(ValueError, match="missing required columns"):
        transformer.transform(frame.drop(columns="numeric"))
    with pytest.raises(TypeError, match="pandas DataFrame"):
        transformer.transform([[1, 2]])
    with pytest.raises(ValueError, match="No usable feature columns"):
        RawFeatureTransformer().fit(pd.DataFrame({"empty": [None] * 20}))


def test_regression_expanded_training_scoring_and_baseline_paths():
    rng = np.random.default_rng(44)
    rows = 180
    feature = rng.normal(size=rows)
    frame = pd.DataFrame(
        {
            "account_id": [f"A{index:04d}" for index in range(rows)],
            "measure": feature,
            "region": pd.Series(rng.choice(["north", "south", "west"], rows), dtype="string"),
            "event_date": pd.date_range("2023-01-01", periods=rows, freq="D"),
        }
    )
    target = pd.Series(
        2.2 * feature + frame["region"].eq("north").astype(float) + rng.normal(0, 0.35, rows),
        name="Revenue",
    )
    trained = train_model(
        frame,
        target,
        AnalysisConfig(
            target="Revenue",
            problem_type="regression",
            effort="expanded",
            test_size=0.2,
            random_seed=9,
        ),
    )
    bundle = trained["model_bundle"]

    assert set(trained["cv_results"]) == {
        "Gradient Boosting",
        "Ridge Regression",
        "Extra Trees",
    }
    assert trained["best_metrics"]["r2"] > 0.8
    assert bundle.baseline_strategy["kind"] == "training-mean"
    baseline, probabilities = baseline_predictions(bundle, 4)
    assert len(baseline) == 4 and probabilities is None
    assert np.allclose(bundle.predict(frame.head(3)), bundle.pipeline.predict(frame.head(3)))

    external = frame.head(40).copy()
    external["Revenue"] = target.head(40).to_numpy()
    evaluated = score_or_evaluate(bundle, external)
    assert evaluated.external_metrics["evaluated_rows"] == 40
    assert "rmse" in evaluated.external_metrics
    assert not any(column.startswith("probability_") for column in evaluated.scored_rows)

    scored_only = score_or_evaluate(bundle, frame.tail(10))
    assert scored_only.external_metrics is None
    assert scored_only.readiness["status"] == "provisional"


def test_multiclass_and_target_normalization_edge_cases():
    rows = 120
    frame = pd.DataFrame(
        {
            "segment": pd.Series((["A", "B", "C"] * 40), dtype="category"),
            "measure": np.sin(np.arange(rows)),
        }
    )
    target = pd.Series((["low", "medium", "high"] * 40), name="Outcome", dtype="string")
    trained = train_model(
        frame,
        target,
        AnalysisConfig(target="Outcome", problem_type="classification", random_seed=2),
    )

    assert trained["model_bundle"].positive_label is None
    assert trained["metric_name"] == "f1_macro"
    assert "average_precision" not in trained["best_metrics"]
    assert normalise_target(pd.Series([True, False, True]), "classification").dtype == bool
    assert normalise_target(pd.Series(["1.5", "2.5"]), "classification").dtype == float
    assert detect_leakage(frame, None, "Outcome", "classification") == (set(), [])
    with pytest.raises(ValueError, match="only valid for binary"):
        infer_positive_label(["a", "b", "c"], target, requested="a")
    with pytest.raises(ValueError, match="not present"):
        infer_positive_label(["a", "b"], pd.Series(["a", "b"]), requested="missing")

    malformed_probabilities = np.ones((3, 1))
    metrics = classification_metrics(
        pd.Series([0, 1, 0]),
        [0, 1, 0],
        probabilities=malformed_probabilities,
        class_labels=[0, 1],
        positive_label=1,
    )
    assert "average_precision" not in metrics


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target": ""}, "non-empty"),
        ({"target": "y", "problem_type": "forecast"}, "problem_type"),
        ({"target": "y", "validation_strategy": "time"}, "holdout_cv"),
        ({"target": "y", "effort": "huge"}, "effort"),
        ({"target": "y", "test_size": 0.01}, "test_size"),
    ],
)
def test_analysis_config_rejects_invalid_contracts(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AnalysisConfig(**kwargs)


def test_scoring_rejects_wrong_types_empty_tables_and_duplicate_normalized_headers():
    frame = pd.DataFrame({"x": np.arange(40), "group": ["a", "b"] * 20})
    target = pd.Series([0, 1] * 20, name="Churn")
    bundle = train_model(
        frame,
        target,
        AnalysisConfig(target="Churn", problem_type="classification"),
    )["model_bundle"]

    with pytest.raises(TypeError, match="ModelBundle"):
        score_or_evaluate(object(), frame)
    with pytest.raises(TypeError, match="pandas DataFrame"):
        score_or_evaluate(bundle, [[1, "a"]])
    with pytest.raises(ValueError, match="no rows"):
        score_or_evaluate(bundle, frame.iloc[0:0])
    duplicate = pd.DataFrame([[1, 2, "a"]], columns=["x", " x ", "group"])
    with pytest.raises(ValueError, match="duplicate columns"):
        score_or_evaluate(bundle, duplicate)

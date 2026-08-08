from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src import pipeline
from src.contracts import AnalysisConfig, ModelBundle
from src.modeling import RawFeatureTransformer, detect_leakage, train_model
from src.scoring import score_or_evaluate


def _mixed_dtype_frame(rows=150):
    category = pd.Series((["monthly", "annual", "two_year"] * rows)[:rows], dtype="string")
    flag = pd.Series(([True, False, False] * rows)[:rows], dtype="boolean")
    frame = pd.DataFrame(
        {
            "CustomerID": [f"C{i:04d}" for i in range(rows)],
            "plan": category,
            "active": flag,
            "spend": np.linspace(10, 500, rows),
            "visits_text": pd.Series([str(i % 12) for i in range(rows)], dtype="string"),
        }
    )
    target = pd.Series(((category == "monthly") | flag.fillna(False)).astype(int), name="Churn")
    return frame, target


def test_string_category_bool_are_classification_features_and_are_fitted():
    frame, target = _mixed_dtype_frame()

    assert pipeline.detect_problem_type(pd.Series(["yes", "no"] * 20, dtype="string")) == "classification"
    trained = train_model(
        frame,
        target,
        AnalysisConfig(target="Churn", problem_type="classification", random_seed=7),
    )
    bundle = trained["model_bundle"]

    assert isinstance(bundle, ModelBundle)
    assert {"plan", "active", "spend", "visits_text"}.issubset(bundle.required_feature_columns)
    assert "CustomerID" in bundle.optional_identifier_columns
    assert any("plan" in feature for feature in bundle.feature_names)
    assert any("active" in feature for feature in bundle.feature_names)
    assert bundle.positive_label == 1
    assert trained["cv_results"]
    assert bundle.baseline_strategy["kind"] == "training-majority"


def test_frequency_mapping_is_learned_from_fit_rows_and_unseen_values_do_not_refit():
    train = pd.DataFrame(
        {
            "high_cardinality": [f"value_{i}" for i in range(100)],
            "amount": np.arange(100),
        }
    )
    transformer = RawFeatureTransformer(drop_identifier_columns=False).fit(train)
    frequency_spec = next(spec for spec in transformer.specs_ if spec["source"] == "high_cardinality")

    transformed = transformer.transform(
        pd.DataFrame({"high_cardinality": ["value_0", "never_seen"], "amount": [1, 2]})
    )

    assert frequency_spec["frequency_map"]["value_0"] == 0.01
    assert transformed["high_cardinality__frequency"].tolist() == [0.01, 0.0]
    assert "never_seen" not in frequency_spec["frequency_map"]


def test_raw_scoring_preserves_rows_coerces_numeric_dtype_and_evaluates():
    frame, target = _mixed_dtype_frame()
    trained = train_model(
        frame,
        target,
        AnalysisConfig(target="Churn", problem_type="classification"),
    )
    scoring = frame.head(25).copy()
    scoring["spend"] = scoring["spend"].round().astype("int64")
    scoring["plan"] = scoring["plan"].astype("category")
    scoring["Churn"] = target.head(25).to_numpy()
    scoring["unused_external_note"] = "kept"

    result = score_or_evaluate(trained["model_bundle"], scoring)

    assert result.scored_rows["CustomerID"].tolist() == scoring["CustomerID"].tolist()
    assert result.scored_rows["unused_external_note"].eq("kept").all()
    assert "prediction" in result.scored_rows
    assert {"probability_0", "probability_1"}.issubset(result.scored_rows.columns)
    assert result.external_metrics["evaluated_rows"] == 25
    assert "baseline_metrics" in result.external_metrics
    assert result.readiness["status"] in {"externally validated", "not deployment-ready"}
    assert any("extra scoring columns" in warning.lower() for warning in result.schema_warnings)
    assert result.drift_summary["identifier_overlap"]["CustomerID"] == 25


def test_bounded_churn_train_test_scoring_keeps_categoricals_and_surfaces_overlap():
    root = Path(__file__).resolve().parents[1]
    training = pd.read_csv(root / "sample_data/customer_churn_dataset-training-master.csv", nrows=1000)
    external = pd.read_csv(root / "sample_data/customer_churn_dataset-testing-master.csv", nrows=500)
    trained = train_model(
        training.drop(columns=["Churn"]),
        training["Churn"],
        AnalysisConfig(target="Churn", problem_type="classification", random_seed=11),
    )
    bundle = trained["model_bundle"]

    assert {"Gender", "Subscription Type", "Contract Length"}.issubset(bundle.required_feature_columns)
    scored = score_or_evaluate(bundle, external)

    assert len(scored.scored_rows) == 500
    assert scored.scored_rows["CustomerID"].equals(external["CustomerID"])
    assert scored.external_metrics["evaluated_rows"] == 500
    expected_overlap = int(external["CustomerID"].isin(set(training["CustomerID"])).sum())
    assert scored.drift_summary["identifier_overlap"]["CustomerID"] == expected_overlap
    assert scored.readiness["status"] == "not deployment-ready"


def test_run_analysis_keeps_legacy_payload_and_exposes_model_bundle():
    frame, target = _mixed_dtype_frame()
    # The insight subsystem receives ordinary bools here; extension-dtype
    # coverage is asserted directly against the modeling core above.
    frame["active"] = frame["active"].astype(int)
    frame["visits_text"] = frame["visits_text"].astype(int)
    data = frame.copy()
    data["Churn"] = target

    result = pipeline.run_analysis(data, "Churn")

    assert result["mode"] == "prediction"
    assert isinstance(result["model_bundle"], ModelBundle)
    assert result["best_model"] is result["model_bundle"]
    assert result["validation_status"] == "provisional"
    assert result["feature_columns"] == result["model_bundle"].required_feature_columns


def test_positive_label_and_baseline_are_fitted_from_training_rows_only():
    frame, numeric_target = _mixed_dtype_frame(rows=180)
    target = numeric_target.map({0: "stay", 1: "leave"}).astype("string")
    target.name = "Churn"
    config = AnalysisConfig(
        target="Churn",
        problem_type="classification",
        positive_label="leave",
        test_size=0.25,
        random_seed=19,
    )
    trained = train_model(frame, target, config)
    bundle = trained["model_bundle"]
    _, _, y_train, _ = train_test_split(
        frame,
        target.astype(object),
        test_size=0.25,
        random_state=19,
        stratify=target,
    )

    assert bundle.positive_label == "leave"
    assert bundle.negative_label == "stay"
    assert bundle.baseline_strategy["positive_rate"] == pytest.approx(
        float(y_train.eq("leave").mean())
    )
    assert bundle.baseline_strategy["majority_label"] == y_train.mode().iloc[0]
    assert "average_precision" in bundle.holdout_metrics
    assert "roc_auc" in bundle.holdout_metrics


def test_leakage_checks_flag_derived_targets_post_outcome_fields_and_temporal_order():
    rows = 120
    target = pd.Series(([0, 1] * 60), name="Churn")
    frame = pd.DataFrame(
        {
            "inverse_label": 1 - target,
            "churn_reason": ["none", "cancelled"] * 60,
            "event_date": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "ordinary_measure": np.linspace(0, 1, rows),
        }
    )

    excluded, warnings = detect_leakage(frame, target, "Churn", "classification")
    warning_text = " ".join(warnings).lower()

    assert {"inverse_label", "churn_reason"}.issubset(excluded)
    assert "after the outcome" in warning_text
    assert "temporal ordering" in warning_text
    assert "ordinary_measure" not in excluded


def test_missing_required_fields_block_but_unseen_categories_are_allowed_with_warning():
    frame, target = _mixed_dtype_frame()
    bundle = train_model(
        frame,
        target,
        AnalysisConfig(target="Churn", problem_type="classification"),
    )["model_bundle"]

    with pytest.raises(ValueError, match="missing required columns: plan"):
        score_or_evaluate(bundle, frame.drop(columns="plan").head(5))

    external = frame.head(12).copy()
    external["plan"] = external["plan"].astype("string")
    external.loc[external.index[:3], "plan"] = "never_seen"
    scored = score_or_evaluate(bundle, external)

    assert len(scored.scored_rows) == 12
    assert any("categories not observed" in warning for warning in scored.schema_warnings)
    assert scored.drift_summary["per_column"]["plan"]["unseen_category_rate"] == 0.25


def test_rare_binary_class_uses_oof_threshold_without_touching_final_holdout():
    rng = np.random.default_rng(8)
    rows = 300
    frame = pd.DataFrame(
        {
            "segment": pd.Series(rng.choice(["A", "B", "C"], rows), dtype="string"),
            "measure": rng.normal(size=rows),
        }
    )
    target = pd.Series([1] * 15 + [0] * (rows - 15), name="Churn")
    trained = train_model(
        frame,
        target,
        AnalysisConfig(target="Churn", problem_type="classification", random_seed=3),
    )

    assert trained["positive_label"] == 1
    assert trained["decision_threshold"] is not None
    assert trained["model_bundle"].holdout_rows == 60
    assert trained["cv_results"][trained["best_model_name"]]["folds"] == 3


def test_pure_noise_target_falls_back_to_insights_instead_of_claiming_model_value():
    rng = np.random.default_rng(991)
    rows = 500
    frame = pd.DataFrame(
        {f"noise_{index}": rng.normal(size=rows) for index in range(5)}
    )
    frame["Churn"] = rng.integers(0, 2, size=rows)

    result = pipeline.run_analysis(frame, "Churn", random_state=17)

    assert result["mode"] == "analysis"
    assert result["predictive_attempt"] is not None
    assert result["predictive_attempt"]["quality"]["verdict"] == "weak"

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src import pipeline
from src.contracts import AnalysisConfig, ModelBundle
from src.modeling import RawFeatureTransformer, detect_leakage, train_model
from src.scoring import _readiness, score_or_evaluate


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


def test_numeric_string_feature_schema_is_learned_from_outer_training_rows_only():
    rows = 100
    random_seed = 23
    target = pd.Series([0, 1] * 50, name="Churn")
    frame = pd.DataFrame(
        {
            "numeric_text": pd.Series([str(index) for index in range(rows)], dtype="string"),
            "signal": np.cos(np.arange(rows)),
        }
    )
    training_index, _ = train_test_split(
        frame.index,
        test_size=0.2,
        random_state=random_seed,
        stratify=target,
    )
    frame.loc[list(training_index[:5]), "numeric_text"] = "not-numeric"
    assert pd.to_numeric(frame["numeric_text"], errors="coerce").notna().mean() == 0.95

    trained = train_model(
        frame,
        target,
        AnalysisConfig(
            target="Churn",
            problem_type="classification",
            random_seed=random_seed,
        ),
    )
    raw_transformer = trained["model_bundle"].named_steps["raw_features"]
    spec = next(item for item in raw_transformer.specs_ if item["source"] == "numeric_text")

    assert spec["kind"] != "numeric"


def test_raw_scoring_preserves_rows_coerces_numeric_dtype_and_evaluates():
    frame, target = _mixed_dtype_frame()
    trained = train_model(
        frame,
        target,
        AnalysisConfig(target="Churn", problem_type="classification"),
    )
    scoring = frame.head(60).copy()
    scoring["spend"] = scoring["spend"].round().astype("int64")
    scoring["plan"] = scoring["plan"].astype("category")
    scoring["Churn"] = target.head(60).to_numpy()
    scoring["unused_external_note"] = "kept"

    result = score_or_evaluate(trained["model_bundle"], scoring)

    assert result.scored_rows["CustomerID"].tolist() == scoring["CustomerID"].tolist()
    assert result.scored_rows["unused_external_note"].eq("kept").all()
    assert "prediction" in result.scored_rows
    assert {"probability_0", "probability_1"}.issubset(result.scored_rows.columns)
    assert result.external_metrics["evaluated_rows"] == 60
    assert "baseline_metrics" in result.external_metrics
    assert result.readiness["status"] in {"externally validated", "not deployment-ready"}
    assert any("extra scoring columns" in warning.lower() for warning in result.schema_warnings)
    assert result.drift_summary["identifier_overlap"]["CustomerID"] == 60


def test_external_target_labels_are_canonicalized_or_block_evaluation_safely():
    frame, target = _mixed_dtype_frame()
    bundle = train_model(
        frame,
        target,
        AnalysisConfig(target="Churn", problem_type="classification"),
    )["model_bundle"]

    compatible = frame.head(60).copy()
    compatible["Churn"] = target.head(60).astype(str).to_numpy()
    evaluated = score_or_evaluate(bundle, compatible)

    assert evaluated.evaluation_status == "completed"
    assert evaluated.external_metrics["evaluated_rows"] == 60
    assert evaluated.readiness["status"] in {"externally validated", "not deployment-ready"}

    incompatible = frame.head(60).copy()
    incompatible["Churn"] = target.head(60).astype(object).to_numpy()
    incompatible.loc[incompatible.index[3], "Churn"] = "not-a-training-class"
    blocked = score_or_evaluate(bundle, incompatible)

    assert len(blocked.scored_rows) == len(incompatible)
    assert blocked.external_metrics is None
    assert blocked.evaluation_status == "blocked"
    assert blocked.readiness["status"] == "provisional"
    assert any(
        "not present in the model's training labels" in warning
        for warning in blocked.schema_warnings
    )


def test_scoring_uses_collision_free_output_columns_without_overwriting_input():
    frame, target = _mixed_dtype_frame()
    bundle = train_model(
        frame,
        target,
        AnalysisConfig(target="Churn", problem_type="classification"),
    )["model_bundle"]
    external = frame.head(10).copy()
    protected = {
        "prediction": "original prediction",
        "model_prediction": "original model prediction",
        "model_prediction_2": "original second model prediction",
        "probability_0": "original probability zero",
        "model_probability_0": "original model probability zero",
        "model_probability_0_2": "original second model probability zero",
        "probability_1": "original probability one",
        "model_probability_1": "original model probability one",
    }
    for column, value in protected.items():
        external[column] = value

    scored = score_or_evaluate(bundle, external)

    for column, value in protected.items():
        assert scored.scored_rows[column].eq(value).all()
    assert scored.prediction_column == "model_prediction_3"
    assert scored.probability_columns == {
        "0": "model_probability_0_3",
        "1": "model_probability_1_2",
    }
    assert scored.scored_rows[scored.prediction_column].notna().all()
    assert all(
        scored.scored_rows[column].between(0.0, 1.0).all()
        for column in scored.probability_columns.values()
    )


def test_scoring_preserves_duplicate_index_rows_and_evaluates_positionally():
    frame, target = _mixed_dtype_frame()
    bundle = train_model(
        frame,
        target,
        AnalysisConfig(target="Churn", problem_type="classification"),
    )["model_bundle"]
    external = frame.head(60).copy()
    external["Churn"] = target.head(60).to_numpy()
    external["row_marker"] = np.arange(len(external))
    external.index = pd.Index(([7, 7, 2] * 20), name="source_index")

    scored = score_or_evaluate(bundle, external)

    assert scored.scored_rows.index.equals(external.index)
    assert scored.scored_rows["row_marker"].tolist() == list(range(60))
    assert len(scored.scored_rows) == 60
    assert scored.external_metrics["evaluated_rows"] == 60


def test_duplicate_headers_are_rejected_by_position_after_normalization():
    frame, target = _mixed_dtype_frame()
    bundle = train_model(
        frame,
        target,
        AnalysisConfig(target="Churn", problem_type="classification"),
    )["model_bundle"]
    duplicate = pd.DataFrame(
        [["C0001", "monthly", "shadow", True, 20.0, "2"]],
        columns=["CustomerID", "plan", "plan", "active", "spend", "visits_text"],
    )

    with pytest.raises(ValueError, match="duplicate columns after trimming: plan"):
        score_or_evaluate(bundle, duplicate)


def test_single_class_and_tiny_external_targets_cannot_validate_readiness():
    frame, target = _mixed_dtype_frame()
    bundle = train_model(
        frame,
        target,
        AnalysisConfig(target="Churn", problem_type="classification"),
    )["model_bundle"]

    single_class = frame.head(25).copy()
    single_class["Churn"] = 1
    single_result = score_or_evaluate(bundle, single_class)
    assert single_result.external_metrics is None
    assert single_result.evaluation_status == "blocked"
    assert single_result.readiness["status"] == "provisional"
    assert any("only one class" in warning for warning in single_result.schema_warnings)

    tiny = frame.head(4).copy()
    tiny["Churn"] = [0, 1, 0, 1]
    tiny_result = score_or_evaluate(bundle, tiny)
    assert tiny_result.external_metrics is None
    assert tiny_result.evaluation_status == "blocked"
    assert tiny_result.readiness["status"] == "provisional"
    assert any("at least 20" in warning for warning in tiny_result.schema_warnings)


def test_low_per_class_external_support_cannot_claim_validation():
    frame, target = _mixed_dtype_frame()
    bundle = train_model(
        frame,
        target,
        AnalysisConfig(target="Churn", problem_type="classification"),
    )["model_bundle"]
    external = frame.head(40).copy()
    external["Churn"] = [0] * 39 + [1]

    scored = score_or_evaluate(bundle, external)

    assert scored.external_metrics is None
    assert scored.evaluation_status == "blocked"
    assert scored.readiness["status"] == "provisional"
    assert any("per fitted class" in warning for warning in scored.schema_warnings)


def test_external_identifier_overlap_blocks_independent_validation_claim():
    frame, target = _mixed_dtype_frame()
    bundle = train_model(
        frame,
        target,
        AnalysisConfig(target="Churn", problem_type="classification"),
    )["model_bundle"]
    external = frame.head(60).copy()
    external["Churn"] = target.head(60).to_numpy()

    scored = score_or_evaluate(bundle, external)

    assert scored.evaluation_status == "completed"
    assert scored.drift_summary["identifier_overlap_total"] == 60
    assert scored.readiness["status"] == "not deployment-ready"
    assert any("overlap" in warning.lower() for warning in scored.schema_warnings)


def test_identifier_overlap_is_tracked_when_identifier_exclusion_is_disabled():
    frame, target = _mixed_dtype_frame()
    trained = train_model(
        frame,
        target,
        AnalysisConfig(target="Churn", problem_type="classification", random_seed=19),
        drop_identifier_columns=False,
    )
    bundle = trained["model_bundle"]

    assert bundle.optional_identifier_columns == []
    assert "CustomerID" in bundle.required_feature_columns
    assert bundle.identifier_reference["CustomerID"] == set(frame["CustomerID"])
    assert "CustomerID" in bundle.named_steps["raw_features"].detected_identifier_columns_
    assert any(
        "identifier exclusion was disabled" in warning
        for warning in bundle.leakage_warnings
    )

    external = frame.head(60).copy()
    external["Churn"] = target.head(60).to_numpy()
    scored = score_or_evaluate(bundle, external)

    assert scored.evaluation_status == "completed"
    assert scored.drift_summary["identifier_overlap"]["CustomerID"] == 60
    assert scored.readiness["status"] == "not deployment-ready"


def test_one_row_multiclass_and_regression_targets_are_scored_but_not_validated():
    rows = 90
    frame = pd.DataFrame(
        {
            "measure": np.linspace(-1.0, 1.0, rows),
            "segment": ["north", "south", "west"] * 30,
        }
    )
    multiclass_target = pd.Series(["low", "medium", "high"] * 30, name="Outcome")
    classification_bundle = train_model(
        frame,
        multiclass_target,
        AnalysisConfig(target="Outcome", problem_type="classification", random_seed=8),
    )["model_bundle"]
    one_classification_row = frame.head(1).copy()
    one_classification_row["Outcome"] = "low"

    classified = score_or_evaluate(classification_bundle, one_classification_row)

    assert len(classified.scored_rows) == 1
    assert classified.external_metrics is None
    assert classified.evaluation_status == "blocked"
    assert classified.readiness["status"] == "provisional"

    two_of_three_classes = frame.head(30).copy()
    two_of_three_classes["Outcome"] = ["low", "medium"] * 15
    partial = score_or_evaluate(classification_bundle, two_of_three_classes)
    assert partial.external_metrics is None
    assert partial.evaluation_status == "blocked"
    assert partial.readiness["status"] == "provisional"
    assert any("zero support" in warning for warning in partial.schema_warnings)

    regression_target = pd.Series(2.0 * frame["measure"] + 0.5, name="Revenue")
    regression_bundle = train_model(
        frame,
        regression_target,
        AnalysisConfig(target="Revenue", problem_type="regression", random_seed=8),
    )["model_bundle"]
    one_regression_row = frame.head(1).copy()
    one_regression_row["Revenue"] = regression_target.iloc[0]

    regressed = score_or_evaluate(regression_bundle, one_regression_row)

    assert len(regressed.scored_rows) == 1
    assert regressed.external_metrics is None
    assert regressed.evaluation_status == "blocked"
    assert regressed.readiness["status"] == "provisional"


def test_regression_readiness_uses_primary_rmse_direction_and_rejects_weak_evidence():
    bundle = ModelBundle(
        pipeline=None,
        target_column="Revenue",
        problem_type="regression",
        raw_schema={},
        required_feature_columns=[],
        optional_identifier_columns=[],
        feature_names=[],
        holdout_metrics={"rmse": 0.50, "r2": 0.20},
        primary_metric="rmse",
    )

    degraded = _readiness(
        bundle,
        {"rmse": 0.75, "r2": 0.99, "evaluated_rows": 50},
        {"rmse": 0.80, "r2": -0.10},
    )
    assert degraded["status"] == "not deployment-ready"
    assert degraded["metric"] == "rmse"
    assert degraded["performance_drop"] == pytest.approx(0.25)

    improved = _readiness(
        bundle,
        {"rmse": 0.30, "r2": 0.30, "evaluated_rows": 50},
        {"rmse": 0.80, "r2": -0.10},
    )
    assert improved["status"] == "externally validated"
    assert improved["metric"] == "rmse"

    for metrics in (
        {"rmse": float("nan"), "evaluated_rows": 50},
        {"rmse": float("inf"), "evaluated_rows": 50},
        {"rmse": 0.30, "evaluated_rows": 3},
    ):
        provisional = _readiness(bundle, metrics, {"rmse": 0.80})
        assert provisional["status"] == "provisional"


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


def test_numeric_post_outcome_probability_is_excluded_as_classification_leakage():
    rng = np.random.default_rng(31)
    rows = 180
    target = pd.Series(rng.integers(0, 2, size=rows), name="Churn")
    frame = pd.DataFrame(
        {
            "ChurnProbability": target * 0.98 + rng.normal(0, 0.001, size=rows),
            "ordinary_measure": rng.normal(size=rows),
        }
    )

    excluded, warnings = detect_leakage(frame, target, "Churn", "classification")
    trained = train_model(
        frame,
        target,
        AnalysisConfig(target="Churn", problem_type="classification", random_seed=7),
    )

    assert "ChurnProbability" in excluded
    assert "ChurnProbability" not in trained["model_bundle"].required_feature_columns
    assert any("near-deterministic" in warning for warning in warnings)


def test_repeated_token_pattern_identifier_is_excluded_and_overlap_is_reported():
    rng = np.random.default_rng(12)
    entity_count = 40
    repetitions = 10
    customer_ids = np.repeat(
        [f"C{index:03d}" for index in range(entity_count)], repetitions
    )
    entity_labels = rng.integers(0, 2, size=entity_count)
    target = pd.Series(np.repeat(entity_labels, repetitions), name="Churn")
    frame = pd.DataFrame(
        {
            "CustomerID": customer_ids,
            "noise": rng.normal(size=len(customer_ids)),
        }
    )

    trained = train_model(
        frame,
        target,
        AnalysisConfig(target="Churn", problem_type="classification", random_seed=4),
    )
    bundle = trained["model_bundle"]

    assert "CustomerID" in bundle.optional_identifier_columns
    assert "CustomerID" not in bundle.required_feature_columns
    assert any("holdout rows share" in warning for warning in bundle.leakage_warnings)
    assert trained["best_metrics"]["balanced_accuracy"] < 0.8


def test_generated_feature_names_are_collision_safe():
    rng = np.random.default_rng(2)
    rows = 150
    target = pd.Series(rng.integers(0, 2, size=rows), name="Churn")
    frame = pd.DataFrame(
        {
            "duration": [f"{80 + (index % 20)} min" for index in range(rows)],
            "duration__number": np.linspace(1, 2, rows),
            "noise": rng.normal(size=rows),
        }
    )

    trained = train_model(
        frame,
        target,
        AnalysisConfig(target="Churn", problem_type="classification", random_seed=9),
    )
    transformer = trained["model_bundle"].named_steps["raw_features"]

    assert len(transformer.output_columns_) == len(set(transformer.output_columns_))
    assert {"duration", "duration__number"}.issubset(
        trained["model_bundle"].required_feature_columns
    )
    assert trained["model_bundle"].predict(frame.head(5)).shape == (5,)


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
    target = pd.Series(
        ["case"] * 15 + ["control"] * (rows - 15),
        name="Churn",
        dtype="string",
    )
    trained = train_model(
        frame,
        target,
        AnalysisConfig(
            target="Churn",
            problem_type="classification",
            positive_label="case",
            random_seed=3,
        ),
    )

    assert trained["positive_label"] == "case"
    assert trained["decision_threshold"] is not None
    assert trained["model_bundle"].holdout_rows == 60
    assert trained["cv_results"][trained["best_model_name"]]["folds"] == 3
    assert trained["selection_metric"] == "average_precision"
    assert {
        result["selection_metric"] for result in trained["cv_results"].values()
    } == {"average_precision"}


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

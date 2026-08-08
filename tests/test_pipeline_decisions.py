from pathlib import Path

import pandas as pd
import pytest

from src import pipeline


def make_prediction_ready_frame(rows=120):
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(rows)],
            "tenure_months": [6 + (i % 24) for i in range(rows)],
            "monthly_spend": [55 + (i % 10) * 7 for i in range(rows)],
            "contract_type": (["monthly", "annual", "two_year"] * ((rows // 3) + 1))[:rows],
            "churn_status": [i % 2 for i in range(rows)],
        }
    )


def make_insight_first_frame(rows=60):
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(rows)],
            "signup_date": pd.date_range("2024-01-01", periods=rows).astype(str),
            "notes": [
                f"Customer note {i} includes several words for descriptive review"
                for i in range(rows)
            ],
            "region_name": (["north", "south", "east", "west"] * ((rows // 4) + 1))[:rows],
        }
    )


def test_text_heavy_target_is_rejected_for_prediction():
    df = pd.DataFrame(
        {
            "age": list(range(60)),
            "segment": ["A", "B"] * 30,
            "notes": [
                f"This is a long free text note number {i} with several descriptive words"
                for i in range(60)
            ],
        }
    )

    candidate = pipeline.evaluate_target_candidate(df, "notes")

    assert candidate["status"] == "rejected"
    assert any("free text" in blocker.lower() for blocker in candidate["blockers"])


def test_identifier_like_target_is_rejected_for_prediction():
    df = make_prediction_ready_frame().rename(columns={"customer_id": "target_id"})

    candidate = pipeline.evaluate_target_candidate(df, "target_id")
    assessment = pipeline.assess_target_for_prediction(df, "target_id")

    assert candidate["status"] == "rejected"
    assert assessment["mode_recommendation"] == "analysis"
    assert any("identifier-like" in blocker.lower() for blocker in assessment["blockers"])


def test_dataset_workflow_prefers_prediction_for_clear_target():
    workflow = pipeline.recommend_dataset_workflow(make_prediction_ready_frame())

    assert workflow["recommended_workflow"] == "prediction"
    assert workflow["recommended_primary_target"] == "churn_status"
    assert workflow["clear_primary_target"] is True


def test_dataset_workflow_prefers_insights_without_strong_target():
    workflow = pipeline.recommend_dataset_workflow(make_insight_first_frame())

    assert workflow["recommended_workflow"] == "insights"
    assert workflow["recommended_primary_target"] is None
    assert "insight-focused analysis" in workflow["summary"].lower()


def test_netflix_type_is_modelable_but_never_auto_selected_as_an_outcome():
    fixture = Path(__file__).resolve().parents[1] / "sample_data/netflix_titles.csv"

    workflow = pipeline.recommend_dataset_workflow(pd.read_csv(fixture))

    assert workflow["recommended_workflow"] == "insights"
    assert workflow["recommended_primary_target"] is None
    assert "type" in workflow["candidate_lookup"]


def test_pipeline_sanitizer_rejects_normalized_duplicate_headers():
    frame = pd.DataFrame([[1, 2]], columns=["value", " value "])

    with pytest.raises(ValueError, match="unique after trimming"):
        pipeline.sanitize_dataframe(frame)


def test_pipeline_sanitizer_leaves_feature_type_learning_to_training_rows():
    frame = pd.DataFrame(
        {
            "numeric_text": pd.Series([str(index) for index in range(95)] + ["bad"] * 5),
            "Churn": [0, 1] * 50,
        }
    )

    prepared = pipeline.prepare_training_frame(frame, "Churn")

    assert pd.api.types.is_string_dtype(prepared["X"]["numeric_text"])
    assert prepared["X"]["numeric_text"].iloc[-1] == "bad"


def test_run_analysis_forwards_selected_seed_to_bounded_sampling(monkeypatch):
    observed = {}
    original = pipeline.sample_training_data

    def capture_seed(X, y, problem_type, max_rows=pipeline.MAX_TRAIN_ROWS, random_state=42):
        observed["random_state"] = random_state
        return original(
            X,
            y,
            problem_type,
            max_rows=max_rows,
            random_state=random_state,
        )

    monkeypatch.setattr(pipeline, "sample_training_data", capture_seed)

    result = pipeline.run_analysis(
        make_prediction_ready_frame(),
        "churn_status",
        random_state=731,
    )

    assert observed["random_state"] == 731
    assert result["model_bundle"].random_seed == 731


def test_run_analysis_retargets_cached_insights_and_positive_label():
    rows = 180
    outcome = pd.Series(["approve", "deny"] * (rows // 2), dtype="string")
    frame = pd.DataFrame(
        {
            "signal_measure": [
                (0 if label == "approve" else 10) + (index % 7) / 100
                for index, label in enumerate(outcome)
            ],
            "segment": ["A", "B", "C"] * (rows // 3),
            "Churn": [0, 0, 1] * (rows // 3),
            "Outcome2": outcome,
        }
    )
    workflow = pipeline.recommend_dataset_workflow(frame)
    assert workflow["insight_analysis"]["association_target"] == "Churn"

    result = pipeline.run_analysis(
        frame,
        "Outcome2",
        positive_label="approve",
        precomputed_workflow=workflow,
    )
    insights = result["insight_analysis"]

    assert result["mode"] == "prediction"
    assert result["positive_label"] == "approve"
    assert insights["association_target"] == "Outcome2"
    assert insights["association_target_inferred"] is False
    assert insights["target_overview"]["positive_label"] == "approve"
    assert all(not item.startswith("Churn is associated") for item in insights["headlines"])
    assert workflow["insight_analysis"]["association_target"] == "Churn"


def test_run_analysis_uses_the_fitted_binary_positive_label_in_insights():
    rows = 240
    outcome = pd.Series(["approve"] * 60 + ["deny"] * 180, dtype="string")
    frame = pd.DataFrame(
        {
            "signal_group": [
                "eligible" if label == "approve" else "ineligible" for label in outcome
            ],
            "review_score": [index % 17 for index in range(rows)],
            "Outcome": outcome,
        }
    )

    result = pipeline.run_analysis(frame, "Outcome", random_state=17)

    assert result["mode"] == "prediction"
    assert result["positive_label"] == "approve"
    assert result["model_bundle"].positive_label == "approve"
    assert result["insight_analysis"]["target_overview"]["positive_label"] == "approve"
    assert result["dataset_recommendation"]["insight_analysis"]["target_overview"][
        "positive_label"
    ] == "approve"


def test_multi_target_grouping_surfaces_related_targets():
    df = pd.DataFrame(
        {
            "age": [20 + (i % 30) for i in range(120)],
            "region": (["north", "south", "east", "west"] * 30),
            "sales_q1": [100 + (i % 20) for i in range(120)],
            "sales_q2": [120 + (i % 20) for i in range(120)],
            "sales_q3": [130 + (i % 20) for i in range(120)],
        }
    )

    workflow = pipeline.recommend_dataset_workflow(df)

    assert workflow["multi_target_candidates"]
    top_group = workflow["multi_target_candidates"][0]
    assert top_group["group_label"] == "Shared prefix: sales"
    assert top_group["problem_type"] == "regression"
    assert set(top_group["columns"]) == {"sales_q1", "sales_q2", "sales_q3"}


def test_run_analysis_falls_back_to_insights_when_model_quality_is_weak(monkeypatch):
    df = make_prediction_ready_frame()

    class DummyBestModel:
        named_steps = {"model": object()}

    def fake_train_best_model(
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
        y_test = pd.Series([0, 1, 0, 1])
        preds = pd.Series([0, 0, 0, 0])
        return {
            "results": {"Dummy Model": {"accuracy": 0.5, "precision": 0.25, "recall": 0.5, "f1": 0.333}},
            "best_model_name": "Dummy Model",
            "best_model": DummyBestModel(),
            "best_metrics": {"accuracy": 0.5, "precision": 0.25, "recall": 0.5, "f1": 0.333},
            "baseline_metrics": {"f1": 0.333},
            "feature_importance": pd.DataFrame(columns=["feature", "importance"]),
            "metric_name": "f1",
            "X_test": X.head(4).reset_index(drop=True),
            "y_test": y_test,
            "preds": preds,
            "best_probabilities": None,
            "imbalance_ratio": None,
            "dropped_target_rows": {"before_split": 0, "train_split": 0, "test_split": 0},
        }

    monkeypatch.setattr(pipeline, "train_best_model", fake_train_best_model)
    monkeypatch.setattr(
        pipeline,
        "build_chart_context",
        lambda problem_type, X_sample, y_sample, holdout_actual, holdout_pred, feature_importance: {},
    )
    monkeypatch.setattr(
        pipeline,
        "assess_model_quality",
        lambda problem_type, best_metrics, baseline_metrics: {
            "verdict": "weak",
            "summary": "Model worth: weak. Holdout quality did not clearly beat the baseline.",
            "primary_delta": 0.0,
            "baseline_metric": baseline_metrics["f1"],
            "best_metric": best_metrics["f1"],
        },
    )

    result = pipeline.run_analysis(df, "churn_status")

    assert result["mode"] == "analysis"
    assert result["predictive_attempt"] is not None
    assert result["predictive_attempt"]["best_model_name"] == "Dummy Model"
    assert "too weak" in result["decision"]["summary"].lower()

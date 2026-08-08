import pandas as pd

from src.contracts import AnalysisConfig, AnalysisResult, ModelBundle, ScoringResult
from src.reporting import build_report_markdown


def test_report_separates_internal_and_external_evidence():
    result = {
        "mode": "prediction",
        "selected_target": "Churn",
        "best_model_name": "Demo model",
        "best_metrics": {"balanced_accuracy": 0.8},
        "baseline_metrics": {
            "balanced_accuracy": 0.5,
            "baseline_strategy": "Training majority class",
        },
        "quality": {"verdict": "useful", "summary": "Internal result is useful."},
        "decision": {"summary": "Prediction was evaluated.", "details": []},
        "insight_analysis": {
            "overview": {"rows": 100, "columns": 4},
            "headlines": ["One supported association was found."],
            "data_quality_summary": [],
        },
        "notes": [],
        "used_rows": 80,
    }
    external = {
        "mode": "evaluation",
        "readiness": "not deployment-ready",
        "external_metrics": {"balanced_accuracy": 0.52},
        "drift_summary": {"warnings": ["Payment Delay shifted materially."]},
    }

    report = build_report_markdown(result, "churn.csv", external)

    assert "## Internal Model Validation" in report
    assert "## External Validation or Scoring" in report
    assert "not deployment-ready" in report
    assert "Payment Delay shifted materially" in report
    assert "do not establish causation" in report


def test_report_formats_structured_readiness_without_dumping_a_dictionary():
    result = {
        "mode": "analysis",
        "selected_target": None,
        "decision": {"summary": "Insight-first."},
        "insight_analysis": {"overview": {"rows": 5, "columns": 2}},
        "notes": [],
    }
    scoring = {
        "external_metrics": {"balanced_accuracy": 0.51},
        "readiness": {
            "status": "not deployment-ready",
            "summary": "External performance remained too close to baseline.",
        },
    }

    report = build_report_markdown(result, "example.csv", scoring)

    assert "Readiness: not deployment-ready" in report
    assert "External performance remained too close to baseline." in report
    assert "{'status':" not in report


def test_report_uses_typed_contracts_real_config_cv_and_drift_evidence():
    bundle = ModelBundle(
        pipeline=None,
        target_column="Churn",
        problem_type="classification",
        raw_schema={"Plan": "categorical", "Churn": "numeric"},
        required_feature_columns=["Plan"],
        optional_identifier_columns=[],
        feature_names=["Plan__frequency"],
        class_labels=[0, 1],
        positive_label=1,
        baseline_strategy={"kind": "majority", "positive_rate": 0.57},
        baseline_metrics={"balanced_accuracy": 0.5},
        holdout_metrics={"balanced_accuracy": 0.74, "f1_macro": 0.72},
        cv_results={
            "Linear model": {"cv_mean": 0.71, "cv_std": 0.03, "folds": 5},
            "Tree model": {"cv_mean": 0.69, "cv_std": 0.04, "folds": 5},
        },
        training_rows=70,
        holdout_rows=30,
        random_seed=731,
    )
    result = AnalysisResult(
        workflow_decision={
            "selected_mode": "prediction",
            "summary": "Prediction was evaluated with an untouched holdout.",
        },
        ranked_insights=[
            {
                "title": "Supported contract association",
                "summary": "Plan was associated with the outcome with adequate support.",
            }
        ],
        internal_validation={
            "best_model_name": "Linear model",
            "quality": {
                "verdict": "provisional",
                "summary": "Internal evidence is promising but provisional.",
            },
            "analysis_config": AnalysisConfig(
                target="Churn",
                problem_type="classification",
                validation_strategy="holdout_cv",
                effort="expanded",
                test_size=0.3,
                random_seed=731,
            ),
        },
        notes=["Transformations were fitted on training rows only."],
        model_bundle=bundle,
    )
    scoring = ScoringResult(
        scored_rows=pd.DataFrame({"Plan": ["monthly"], "prediction": [1]}),
        schema_warnings=["One unseen category was handled without refitting."],
        external_metrics={
            "balanced_accuracy": 0.51,
            "evaluated_rows": 64_374,
            "support": {"0": 33_881, "1": 30_493},
        },
        readiness={
            "status": "not deployment-ready",
            "summary": "External performance dropped materially.",
        },
        drift_summary={
            "level": "high",
            "max_standardized_mean_difference": 0.75,
            "max_total_variation_distance": 0.22,
            "max_missingness_change": 0.10,
            "max_unseen_category_rate": 0.12,
            "target_prevalence_change": -0.09,
            "identifier_overlap": {"CustomerID": 25},
            "identifier_overlap_total": 25,
            "per_column": {
                "Age": {
                    "standardized_mean_difference": 0.75,
                    "missingness_change": 0.02,
                },
                "Plan": {
                    "total_variation_distance": 0.22,
                    "unseen_category_rate": 0.12,
                    "missingness_change": 0.10,
                },
            },
        },
    )

    report = build_report_markdown(result, "typed.csv", scoring)

    assert "Prediction was evaluated with an untouched holdout." in report
    assert "Supported contract association" in report
    assert "Random seed: 731" in report
    assert "Validation strategy: holdout_cv" in report
    assert "Effort: expanded" in report
    assert "Holdout fraction: 0.300" in report
    assert "Linear model" in report and "0.7100" in report and "5 folds" in report
    assert "Internal conclusion: Internal evidence is promising but provisional." in report
    assert "External conclusion: not deployment-ready" in report
    assert "External performance dropped materially." in report
    assert "Maximum numeric SMD: 0.7500" in report
    assert "Maximum categorical TVD: 0.2200" in report
    assert "Age: numeric SMD=0.7500" in report
    assert "Plan: categorical TVD=0.2200" in report
    assert "CustomerID=25" in report
    assert "One unseen category was handled without refitting." in report
    assert "External positive-label prevalence (`1`): 30,493 of 64,374 (47.37%)" in report


def test_report_renders_weak_predictive_attempt_evidence_and_fitted_schema():
    bundle = ModelBundle(
        pipeline=None,
        target_column="Churn",
        problem_type="classification",
        raw_schema={"CustomerID": "identifier", "Plan": "categorical", "Churn": "numeric"},
        required_feature_columns=["Plan"],
        optional_identifier_columns=["CustomerID"],
        feature_names=["Plan__frequency"],
        class_labels=[0, 1],
        positive_label=1,
        baseline_strategy={"kind": "training-majority", "majority_label": 1},
        baseline_metrics={"balanced_accuracy": 0.5},
        holdout_metrics={"balanced_accuracy": 0.53, "f1_macro": 0.51},
        cv_results={
            "Weak linear model": {
                "selection_metric": "f1_macro",
                "cv_mean": 0.52,
                "cv_std": 0.06,
                "folds": 3,
            }
        },
        leakage_warnings=["A post-outcome field was excluded."],
        training_rows=60,
        holdout_rows=20,
        random_seed=909,
    )
    result = {
        "mode": "analysis",
        "selected_target": "Churn",
        "decision": {
            "selected_mode": "analysis",
            "summary": "Prediction was tested, then the app fell back to insights.",
        },
        "insight_analysis": {"overview": {"rows": 200, "columns": 3}},
        "predictive_attempt": {
            "best_model_name": "Weak linear model",
            "best_metrics": {"balanced_accuracy": 0.53, "f1_macro": 0.51},
            "baseline_metrics": {"balanced_accuracy": 0.5},
            "quality": {
                "verdict": "weak",
                "summary": "The holdout stayed too close to baseline.",
            },
            "model_bundle": bundle,
        },
        "used_rows": 200,
        "notes": [],
    }

    report = build_report_markdown(result, "weak.csv")

    assert "Workflow: `analysis`" in report
    assert "## Internal Model Validation" in report
    assert "Model: Weak linear model" in report
    assert "Status: weak" in report
    assert "Rows used for bounded modeling: 80" in report
    assert "Balanced Accuracy: 0.5300" in report
    assert "Balanced Accuracy: 0.5000" in report
    assert "Weak linear model: f1_macro mean 0.5200, std 0.0600, 3 folds" in report
    assert "Random seed: 909" in report
    assert "### Fitted feature and transformation schema" in report
    assert "`Plan` (categorical)" in report
    assert "`Plan__frequency`" in report
    assert "fitted on training rows only" in report
    assert "A post-outcome field was excluded." in report


def test_validation_status_precedes_quality_verdict_and_support_is_rendered():
    result = {
        "mode": "prediction",
        "selected_target": "Outcome",
        "validation_status": "provisional",
        "best_model_name": "Useful model",
        "best_metrics": {
            "balanced_accuracy": 0.81,
            "support": {"stay": 31, "leave": 29},
        },
        "quality": {
            "verdict": "useful",
            "summary": "The model exceeded its baseline internally.",
        },
        "decision": {"summary": "Prediction remained active."},
        "insight_analysis": {"overview": {"rows": 60, "columns": 2}},
        "notes": [],
    }

    report = build_report_markdown(result, "provisional.csv")

    assert "Status: provisional" in report
    assert "Status: useful" not in report
    assert "### Per-class holdout support" in report
    assert "Class `stay`: 31" in report
    assert "Class `leave`: 29" in report


def test_blocked_labeled_evaluation_is_not_described_as_plain_scoring():
    result = {
        "mode": "analysis",
        "decision": {"summary": "Insight-first."},
        "insight_analysis": {"overview": {"rows": 4, "columns": 2}},
        "notes": [],
    }
    scoring = ScoringResult(
        scored_rows=pd.DataFrame({"feature": [1], "prediction": [0]}),
        schema_warnings=["Outcome contains labels not seen during training."],
        readiness={
            "status": "provisional",
            "summary": "External evaluation was blocked because target labels were incompatible.",
        },
        evaluation_status="blocked",
    )

    report = build_report_markdown(result, "blocked.csv", scoring)

    assert "Mode: evaluation blocked" in report
    assert "Evaluation status: blocked" in report
    assert "Mode: scoring" not in report
    assert "External evaluation was blocked" in report

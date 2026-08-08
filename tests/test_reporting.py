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

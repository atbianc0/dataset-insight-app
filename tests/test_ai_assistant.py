import json

import pandas as pd

from src.ai_assistant import build_runtime_extension_registry
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


class FakeResponse:
    def __init__(self, output_text):
        self.output_text = output_text


class FakeResponsesAPI:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(json.dumps(self.payload))


class FakeClient:
    def __init__(self, payload):
        self.responses = FakeResponsesAPI(payload)


def test_ai_assistant_adds_advisory_workflow_context_without_replacing_heuristics():
    fake_client = FakeClient(
        {
            "dataset_summary": "This looks like a customer retention dataset with a plausible churn target.",
            "prediction_explanation": "Prediction seems reasonable because the heuristic workflow already found a workable churn label.",
            "report_summary": "The dataset is prediction-capable, but the heuristic path should remain authoritative.",
            "semantic_column_notes": [
                {
                    "column": "monthly_spend",
                    "meaning": "Likely recurring customer revenue or billing amount.",
                    "confidence": "high",
                }
            ],
            "target_suggestions": [
                {
                    "column": "churn_status",
                    "advice": "Best used as the primary prediction target because it already behaves like a compact binary outcome.",
                }
            ],
            "feature_suggestions": [
                {
                    "column": "tenure_months",
                    "advice": "Likely useful because tenure often carries retention signal.",
                }
            ],
            "cautions": [
                "These are advisory suggestions and should not override heuristic rejection or fallback logic.",
            ],
        }
    )
    registry = build_runtime_extension_registry(enable_ai=True, client=fake_client)

    workflow = pipeline.recommend_dataset_workflow(
        make_prediction_ready_frame(),
        extension_registry=registry,
    )

    semantic_payload = workflow["assistant_extensions"]["stage_outputs"][
        "semantic_column_interpretation"
    ]["merged_payload"]
    task_payload = workflow["assistant_extensions"]["stage_outputs"]["task_understanding"][
        "merged_payload"
    ]

    assert workflow["recommended_primary_target"] == "churn_status"
    assert semantic_payload["ai_semantic_column_notes"][0]["column"] == "monthly_spend"
    assert task_payload["ai_target_suggestions"][0]["column"] == "churn_status"
    assert fake_client.responses.calls
    assert len(fake_client.responses.calls) == 1


def test_ai_assistant_outputs_are_available_in_final_analysis_report_context():
    fake_client = FakeClient(
        {
            "dataset_summary": "This dataset is better understood as a churn-style retention problem.",
            "prediction_explanation": "The heuristic recommendation remains authoritative, and AI is only clarifying the rationale.",
            "report_summary": "Use the heuristic workflow and treat these notes as explanatory support.",
            "semantic_column_notes": [],
            "target_suggestions": [],
            "feature_suggestions": [
                {
                    "column": "contract_type",
                    "advice": "Useful categorical input because plan structure may correlate with churn.",
                }
            ],
            "cautions": [],
        }
    )
    registry = build_runtime_extension_registry(enable_ai=True, client=fake_client)

    result = pipeline.run_analysis(
        make_prediction_ready_frame(),
        None,
        extension_registry=registry,
    )

    report_payload = result["assistant_extensions"]["stage_outputs"]["report_generation"][
        "merged_payload"
    ]
    task_payload = result["assistant_extensions"]["stage_outputs"]["task_understanding"][
        "merged_payload"
    ]

    assert result["mode"] == "analysis"
    assert report_payload["ai_report_summary"].startswith("Use the heuristic workflow")
    assert task_payload["ai_feature_suggestions"][0]["column"] == "contract_type"

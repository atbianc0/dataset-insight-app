import json

import pandas as pd

from src import pipeline
from src.ai_assistant import (
    OpenAIDatasetInterpretationAssistant,
    build_runtime_extension_registry,
)
from src.extensions import AnalysisContext


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
    request_text = fake_client.responses.calls[0]["input"]
    assert "sample_values" not in request_text
    assert "C0000" not in request_text


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


def test_ai_request_contains_only_aggregate_schema_not_rows_or_categorical_values():
    """Privacy consent must match the exact payload sent to the optional provider."""

    fake_client = FakeClient(
        {
            "dataset_summary": "Aggregate-only summary.",
            "prediction_explanation": "Aggregate-only explanation.",
            "report_summary": "Aggregate-only report.",
            "semantic_column_notes": [],
            "target_suggestions": [],
            "feature_suggestions": [],
            "cautions": [],
        }
    )
    assistant = OpenAIDatasetInterpretationAssistant(client=fake_client)
    context = AnalysisContext(
        df=pd.DataFrame(
            {
                "CustomerName": ["ROW_VALUE_ALICE", "ROW_VALUE_BOB"],
                "Plan": ["CATEGORY_VALUE_ULTRA", "CATEGORY_VALUE_BASIC"],
                "Churn": [1, 0],
            }
        ),
        target_col="Churn",
        artifacts={
            "workflow": {
                "recommended_workflow": "prediction",
                "recommended_task_type": "classification",
                "recommended_primary_target": "Churn",
                "summary": "WORKFLOW_FREE_TEXT_WITH_CATEGORY_VALUE_ULTRA",
                "best_analysis_path": "PATH_WITH_CATEGORY_VALUE_BASIC",
                "candidate_targets": [
                    {
                        "column": "Churn",
                        "status": "recommended",
                        "problem_type": "classification",
                        "score": 9.0,
                        "target_shape": {"top_values": ["CATEGORY_VALUE_ULTRA"]},
                        "pros": ["PRO_WITH_ROW_VALUE_ALICE"],
                        "cautions": ["CAUTION_WITH_CATEGORY_VALUE_BASIC"],
                    }
                ],
                "multi_target_candidates": ["MULTI_WITH_ROW_VALUE_BOB"],
            },
            "insight_analysis": {
                "overview": {
                    "rows": 2,
                    "columns": 3,
                    "missing_cells": 0,
                    "numeric_columns": 1,
                    "top_missing_column": "Plan",
                    "top_missing_pct": 0.0,
                },
                "headlines": ["HEADLINE_WITH_CATEGORY_VALUE_ULTRA"],
            },
            "column_inspection": pd.DataFrame(
                [
                    {
                        "column": "Plan",
                        "dtype": "string",
                        "role_hint": "categorical",
                        "coverage_pct": 100.0,
                        "missing_pct": 0.0,
                        "unique_values": 2,
                        "sample_values": ["CATEGORY_VALUE_ULTRA"],
                        "recommendation": "RECOMMENDATION_WITH_CATEGORY_VALUE_BASIC",
                    }
                ]
            ),
            "feature_subset_summary": {
                "likely_useful": [
                    {
                        "column": "Plan",
                        "role": "categorical",
                        "guidance": "GUIDANCE_WITH_CATEGORY_VALUE_ULTRA",
                        "reason": "REASON_WITH_ROW_VALUE_ALICE",
                    }
                ]
            },
            "target_assessment": {
                "mode_recommendation": "prediction",
                "summary": "ASSESSMENT_WITH_CATEGORY_VALUE_BASIC",
                "reasons_for_prediction": ["REASON_WITH_ROW_VALUE_BOB"],
                "reasons_against_prediction": ["AGAINST_WITH_CATEGORY_VALUE_ULTRA"],
                "blockers": ["BLOCKER_WITH_ROW_VALUE_ALICE"],
                "usable_rows": 2,
                "unique_count": 2,
            },
        },
    )

    assistant.task_understanding(context)

    request_text = fake_client.responses.calls[0]["input"]
    forbidden_values = {
        "ROW_VALUE_ALICE",
        "ROW_VALUE_BOB",
        "CATEGORY_VALUE_ULTRA",
        "CATEGORY_VALUE_BASIC",
    }
    assert all(value not in request_text for value in forbidden_values)
    assert '"rows": 2' in request_text
    assert '"columns": 3' in request_text
    assert '"column": "Plan"' in request_text
    assert '"unique_values": 2' in request_text

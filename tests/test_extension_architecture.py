import pandas as pd

from src import pipeline
from src.extensions import (
    HeuristicPipelineExtension,
    PipelineExtension,
    PipelineExtensionRegistry,
)


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


class DemoAssistant(PipelineExtension):
    name = "demo_assistant"

    def semantic_column_interpretation(self, context):
        return {
            "semantic_aliases": {
                context.df.columns[0]: "demo semantic label",
            }
        }

    def task_understanding(self, context):
        workflow = context.artifacts["workflow"]
        return {
            "assistant_task_note": f"Preferred workflow: {workflow['recommended_workflow']}",
        }

    def feature_suggestion(self, context):
        return {
            "assistant_feature_shortlist": context.df.columns[:2].tolist(),
        }

    def report_generation(self, context):
        return {
            "generated_narrative": "Demo assistant summary",
        }


class BrokenAssistant(PipelineExtension):
    name = "broken_assistant"

    def task_understanding(self, context):
        raise RuntimeError("synthetic extension failure")


def test_workflow_and_analysis_collect_extension_outputs():
    df = make_prediction_ready_frame()
    registry = PipelineExtensionRegistry(
        [HeuristicPipelineExtension(), DemoAssistant()]
    )

    workflow = pipeline.recommend_dataset_workflow(
        df,
        extension_registry=registry,
    )

    semantic_payload = workflow["assistant_extensions"]["stage_outputs"][
        "semantic_column_interpretation"
    ]["merged_payload"]
    assert workflow["assistant_extensions"]["providers"] == [
        "heuristic_core",
        "demo_assistant",
    ]
    assert "column_roles" in semantic_payload
    assert semantic_payload["semantic_aliases"]["customer_id"] == "demo semantic label"

    task_payload = workflow["assistant_extensions"]["stage_outputs"]["task_understanding"][
        "merged_payload"
    ]
    assert task_payload["recommended_workflow"] == workflow["recommended_workflow"]
    assert task_payload["assistant_task_note"] == (
        f"Preferred workflow: {workflow['recommended_workflow']}"
    )

    result = pipeline.run_analysis(
        df,
        None,
        extension_registry=registry,
    )

    feature_payload = result["assistant_extensions"]["stage_outputs"]["feature_suggestion"][
        "merged_payload"
    ]
    report_payload = result["assistant_extensions"]["stage_outputs"]["report_generation"][
        "merged_payload"
    ]
    assert feature_payload["assistant_feature_shortlist"] == [
        "customer_id",
        "tenure_months",
    ]
    assert report_payload["generated_narrative"] == "Demo assistant summary"
    assert report_payload["decision_summary"] == result["decision"]["summary"]


def test_extension_failures_are_captured_without_breaking_workflow():
    df = make_prediction_ready_frame()
    registry = PipelineExtensionRegistry(
        [HeuristicPipelineExtension(), BrokenAssistant()]
    )

    workflow = pipeline.recommend_dataset_workflow(
        df,
        extension_registry=registry,
    )

    task_stage = workflow["assistant_extensions"]["stage_outputs"]["task_understanding"]
    assert workflow["summary"]
    assert task_stage["errors"] == [
        {
            "provider": "broken_assistant",
            "stage": "task_understanding",
            "message": "synthetic extension failure",
        }
    ]

import json
import os
from typing import Any, Dict, List, Optional

from src.extensions import (
    HeuristicPipelineExtension,
    PipelineExtension,
    PipelineExtensionRegistry,
    StageOutput,
)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency at runtime
    OpenAI = None


DEFAULT_AI_MODEL = os.getenv("DATASET_AI_MODEL", "gpt-4o-mini")
MAX_COLUMNS_FOR_AI = 12
MAX_TARGETS_FOR_AI = 5
MAX_FEATURES_PER_BUCKET = 5


def ai_assistant_is_available():
    return bool(os.getenv("OPENAI_API_KEY")) and OpenAI is not None


def build_runtime_extension_registry(enable_ai=False, model=None, client=None):
    extensions = [HeuristicPipelineExtension()]
    if enable_ai:
        extensions.append(OpenAIDatasetInterpretationAssistant(model=model or DEFAULT_AI_MODEL, client=client))
    return PipelineExtensionRegistry(extensions)


def extract_stage_payload(assistant_extensions, stage):
    if not assistant_extensions:
        return {}
    return assistant_extensions.get("stage_outputs", {}).get(stage, {}).get("merged_payload", {})


def extract_stage_errors(assistant_extensions, stage):
    if not assistant_extensions:
        return []
    return assistant_extensions.get("stage_outputs", {}).get(stage, {}).get("errors", [])


class OpenAIDatasetInterpretationAssistant(PipelineExtension):
    name = "openai_dataset_assistant"

    def __init__(self, model=DEFAULT_AI_MODEL, client=None):
        self.model = model
        self._client = client
        self._cache = {}

    def semantic_column_interpretation(self, context):
        analysis = self._analyze_context(context)
        semantic_notes = analysis.get("semantic_column_notes", [])
        if not semantic_notes:
            return None

        return StageOutput(
            provider=self.name,
            stage="semantic_column_interpretation",
            payload={
                "ai_semantic_column_notes": semantic_notes,
            },
            summary="AI-assisted semantic notes were generated from column names and sampled values.",
        )

    def task_understanding(self, context):
        analysis = self._analyze_context(context)
        return StageOutput(
            provider=self.name,
            stage="task_understanding",
            payload={
                "ai_dataset_summary": analysis.get("dataset_summary"),
                "ai_prediction_explanation": analysis.get("prediction_explanation"),
                "ai_target_suggestions": analysis.get("target_suggestions", []),
                "ai_feature_suggestions": analysis.get("feature_suggestions", []),
                "ai_cautions": analysis.get("cautions", []),
            },
            summary=analysis.get("dataset_summary"),
        )

    def report_generation(self, context):
        analysis = self._analyze_context(context)
        return StageOutput(
            provider=self.name,
            stage="report_generation",
            payload={
                "ai_report_summary": analysis.get("report_summary"),
            },
            summary="AI-generated report summary was prepared from the heuristic workflow context.",
        )

    def _analyze_context(self, context):
        payload = self._build_ai_payload(context)
        cache_key = json.dumps(payload, sort_keys=True)
        if cache_key not in self._cache:
            self._cache[cache_key] = self._request_analysis(payload)
        return self._cache[cache_key]

    def _build_ai_payload(self, context):
        workflow = context.artifacts.get("workflow") or {}
        insight_analysis = context.artifacts.get("insight_analysis") or {}
        feature_subset = context.artifacts.get("feature_subset_summary") or {}
        target_assessment = context.artifacts.get("target_assessment") or {}
        column_inspection = context.artifacts.get("column_inspection")

        column_rows = []
        if column_inspection is not None:
            for row in column_inspection.to_dict(orient="records")[:MAX_COLUMNS_FOR_AI]:
                column_rows.append(
                    {
                        "column": row.get("column"),
                        "dtype": row.get("dtype"),
                        "role_hint": row.get("role_hint"),
                        "coverage_pct": row.get("coverage_pct"),
                        "missing_pct": row.get("missing_pct"),
                        "unique_values": row.get("unique_values"),
                        "sample_values": row.get("sample_values"),
                        "heuristic_recommendation": row.get("recommendation"),
                    }
                )

        candidate_targets = []
        for candidate in workflow.get("candidate_targets", [])[:MAX_TARGETS_FOR_AI]:
            candidate_targets.append(
                {
                    "column": candidate.get("column"),
                    "status": candidate.get("status"),
                    "problem_type": candidate.get("problem_type"),
                    "target_shape": candidate.get("target_shape"),
                    "score": candidate.get("score"),
                    "pros": candidate.get("pros", [])[:2],
                    "cautions": candidate.get("cautions", [])[:2],
                    "blockers": candidate.get("blockers", [])[:2],
                }
            )

        feature_buckets = {}
        for bucket in ("likely_useful", "risky", "avoid"):
            feature_buckets[bucket] = [
                {
                    "column": item.get("column"),
                    "role": item.get("role"),
                    "guidance": item.get("guidance"),
                    "reason": item.get("reason"),
                }
                for item in feature_subset.get(bucket, [])[:MAX_FEATURES_PER_BUCKET]
            ]

        overview = insight_analysis.get("overview", {})
        return {
            "dataset_overview": {
                "rows": overview.get("rows"),
                "columns": overview.get("columns"),
                "missing_cells": overview.get("missing_cells"),
                "numeric_columns": overview.get("numeric_columns"),
                "top_missing_column": overview.get("top_missing_column"),
                "top_missing_pct": overview.get("top_missing_pct"),
            },
            "heuristic_recommendation": {
                "recommended_workflow": workflow.get("recommended_workflow"),
                "recommended_task_type": workflow.get("recommended_task_type"),
                "recommended_primary_target": workflow.get("recommended_primary_target"),
                "summary": workflow.get("summary"),
                "best_analysis_path": workflow.get("best_analysis_path"),
            },
            "selected_target": context.target_col,
            "candidate_targets": candidate_targets,
            "multi_target_candidates": workflow.get("multi_target_candidates", [])[:3],
            "feature_guidance": feature_buckets,
            "column_inspection": column_rows,
            "analysis_headlines": insight_analysis.get("headlines", [])[:5],
            "target_assessment": {
                "mode_recommendation": target_assessment.get("mode_recommendation"),
                "summary": target_assessment.get("summary"),
                "reasons_for_prediction": target_assessment.get("reasons_for_prediction", [])[:3],
                "reasons_against_prediction": target_assessment.get("reasons_against_prediction", [])[:3],
                "blockers": target_assessment.get("blockers", [])[:3],
            },
        }

    def _request_analysis(self, payload):
        client = self._get_client()
        instructions = (
            "You are an AI assistant for a tabular dataset analysis app. "
            "The heuristic workflow recommendation is the source of truth. "
            "Do not override it. Your job is to improve clarity, semantic interpretation, "
            "and user guidance. Be cautious, practical, and explicit about uncertainty."
        )
        prompt = (
            "Use the dataset summary below to produce a concise assistant layer. "
            "Treat heuristic recommendations as authoritative and frame all AI suggestions as advisory. "
            "Prefer explaining why prediction is or is not reasonable over encouraging modeling.\n\n"
            + json.dumps(payload, indent=2)
        )
        schema = {
            "name": "dataset_assistant_summary",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "dataset_summary": {"type": "string"},
                    "prediction_explanation": {"type": "string"},
                    "report_summary": {"type": "string"},
                    "semantic_column_notes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "column": {"type": "string"},
                                "meaning": {"type": "string"},
                                "confidence": {"type": "string"},
                            },
                            "required": ["column", "meaning", "confidence"],
                        },
                    },
                    "target_suggestions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "column": {"type": "string"},
                                "advice": {"type": "string"},
                            },
                            "required": ["column", "advice"],
                        },
                    },
                    "feature_suggestions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "column": {"type": "string"},
                                "advice": {"type": "string"},
                            },
                            "required": ["column", "advice"],
                        },
                    },
                    "cautions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "dataset_summary",
                    "prediction_explanation",
                    "report_summary",
                    "semantic_column_notes",
                    "target_suggestions",
                    "feature_suggestions",
                    "cautions",
                ],
            },
        }

        response = client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
            max_output_tokens=1200,
            text={"format": {"type": "json_schema", **schema}},
        )
        content = getattr(response, "output_text", None)
        if not content:
            raise ValueError("The AI assistant returned no text output.")
        parsed = json.loads(content)
        return parsed

    def _get_client(self):
        if self._client is not None:
            return self._client
        if OpenAI is None:
            raise RuntimeError(
                "The optional 'openai' package is not installed, so the AI assistant layer is unavailable."
            )
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set, so the AI assistant layer is unavailable."
            )
        self._client = OpenAI()
        return self._client

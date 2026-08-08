from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence

EXTENSION_STAGES = (
    "semantic_column_interpretation",
    "task_understanding",
    "feature_suggestion",
    "external_enrichment",
    "report_generation",
)


@dataclass
class AnalysisContext:
    df: Any
    target_col: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageOutput:
    provider: str
    stage: str
    payload: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None
    confidence: Optional[float] = None

    def to_dict(self):
        result = {
            "provider": self.provider,
            "stage": self.stage,
            "payload": self.payload,
        }
        if self.summary is not None:
            result["summary"] = self.summary
        if self.confidence is not None:
            result["confidence"] = self.confidence
        return result


class PipelineExtension:
    name = "extension"

    def semantic_column_interpretation(self, context):
        return None

    def task_understanding(self, context):
        return None

    def feature_suggestion(self, context):
        return None

    def external_enrichment(self, context):
        return None

    def report_generation(self, context):
        return None


class HeuristicPipelineExtension(PipelineExtension):
    name = "heuristic_core"

    def semantic_column_interpretation(self, context):
        column_inspection = context.artifacts.get("column_inspection")
        if column_inspection is None or column_inspection.empty:
            return None

        column_roles = {}
        for row in column_inspection.to_dict(orient="records"):
            column_roles[row["column"]] = {
                "role_hint": row.get("role_hint"),
                "recommendation": row.get("recommendation"),
                "coverage_pct": row.get("coverage_pct"),
                "sample_values": row.get("sample_values"),
            }

        return StageOutput(
            provider=self.name,
            stage="semantic_column_interpretation",
            payload={
                "column_roles": column_roles,
                "column_role_rows": column_inspection.to_dict(orient="records"),
            },
            summary="Heuristic column-role interpretation is available for every column.",
        )

    def task_understanding(self, context):
        workflow = context.artifacts.get("workflow")
        if not workflow:
            return None

        payload = {
            "recommended_workflow": workflow.get("recommended_workflow"),
            "recommended_task_type": workflow.get("recommended_task_type"),
            "recommended_primary_target": workflow.get("recommended_primary_target"),
            "candidate_targets": workflow.get("candidate_targets", []),
            "multi_target_candidates": workflow.get("multi_target_candidates", []),
            "task_recommendations": workflow.get("task_recommendations", []),
        }
        best_analysis_path = workflow.get("best_analysis_path")
        if best_analysis_path is not None:
            payload["best_analysis_path"] = best_analysis_path

        return StageOutput(
            provider=self.name,
            stage="task_understanding",
            payload=payload,
            summary=workflow.get("summary"),
        )

    def feature_suggestion(self, context):
        payload = {}

        feature_subset_summary = context.artifacts.get("feature_subset_summary")
        if feature_subset_summary:
            payload["feature_subset_summary"] = feature_subset_summary

        target_candidate = context.artifacts.get("target_candidate")
        if target_candidate:
            payload["target_feature_shortlist"] = target_candidate.get("suggested_feature_subset", [])
            payload["target_feature_exclusions"] = target_candidate.get("rejected_feature_columns", [])

        target_assessment = context.artifacts.get("target_assessment")
        if target_assessment:
            payload["usable_feature_count"] = target_assessment.get("usable_feature_count")
            payload["mode_recommendation"] = target_assessment.get("mode_recommendation")

        prepared_frame = context.artifacts.get("prepared_frame")
        if prepared_frame:
            payload["prepared_numeric_columns"] = prepared_frame.get("numeric_cols", [])
            payload["prepared_categorical_columns"] = prepared_frame.get("categorical_cols", [])
            payload["preparation_notes"] = prepared_frame.get("notes", [])

        if not payload:
            return None

        return StageOutput(
            provider=self.name,
            stage="feature_suggestion",
            payload=payload,
            summary="Heuristic feature guidance is available from the current preparation logic.",
        )

    def report_generation(self, context):
        result = context.artifacts.get("result")
        workflow = context.artifacts.get("workflow") or {}
        analysis = context.artifacts.get("insight_analysis") or {}
        if result is None and not workflow:
            return None

        suggested_sections = ["Dataset Recommendation", "Workflow Decision"]
        if workflow.get("candidate_targets"):
            suggested_sections.append("Target Review")
        if workflow.get("feature_subset_summary"):
            suggested_sections.append("Feature Guidance")
        if result is not None and result.get("mode") == "prediction":
            suggested_sections.append("Predictive Model Summary")
        if analysis.get("analysis_recommendations") is not None:
            suggested_sections.append("Recommended Analysis Paths")

        payload = {
            "decision_summary": (
                result.get("decision", {}).get("summary")
                if result is not None
                else workflow.get("summary")
            ),
            "suggested_sections": suggested_sections,
            "headline_points": analysis.get("headlines", []),
        }

        return StageOutput(
            provider=self.name,
            stage="report_generation",
            payload=payload,
            summary="The heuristic pipeline can already produce report-ready structure and summary text.",
        )


def _normalize_stage_output(provider, stage, result):
    if result is None:
        return None
    if isinstance(result, StageOutput):
        return result
    if isinstance(result, dict):
        return StageOutput(
            provider=provider,
            stage=stage,
            payload=result.get("payload", result),
            summary=result.get("summary"),
            confidence=result.get("confidence"),
        )
    raise TypeError(f"Unsupported stage output type for {provider}.{stage}: {type(result)!r}")


class PipelineExtensionRegistry:
    def __init__(self, extensions: Optional[Sequence[PipelineExtension]] = None):
        self.extensions = list(extensions or [HeuristicPipelineExtension()])

    def collect(self, context: AnalysisContext, stages: Optional[Iterable[str]] = None):
        stage_names = list(stages or EXTENSION_STAGES)
        providers = [extension.name for extension in self.extensions]
        stage_outputs = {}

        for stage in stage_names:
            merged_payload = {}
            outputs = []
            errors = []

            for extension in self.extensions:
                hook = getattr(extension, stage, None)
                if hook is None:
                    continue
                try:
                    normalized = _normalize_stage_output(extension.name, stage, hook(context))
                except Exception as exc:
                    errors.append(
                        {
                            "provider": extension.name,
                            "stage": stage,
                            "message": str(exc),
                        }
                    )
                    continue

                if normalized is None:
                    continue

                merged_payload.update(normalized.payload)
                outputs.append(normalized.to_dict())

            stage_outputs[stage] = {
                "merged_payload": merged_payload,
                "outputs": outputs,
                "errors": errors,
            }

        return {
            "providers": providers,
            "stage_order": stage_names,
            "stage_outputs": stage_outputs,
        }


DEFAULT_EXTENSION_REGISTRY = PipelineExtensionRegistry()

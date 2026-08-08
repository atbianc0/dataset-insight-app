"""Stable data contracts shared by modeling, scoring, and the UI.

The public application still exposes dictionaries for backwards compatibility,
but new code should pass these explicit contracts between subsystems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

ProblemType = Literal["classification", "regression"]


@dataclass
class DatasetProfile:
    """One sanitized/profiled view reused throughout an analysis session."""

    fingerprint: str
    sanitized_frame: pd.DataFrame
    schema: dict[str, str]
    exact_overview: dict[str, int]
    column_profiles: list[dict[str, Any]] = field(default_factory=list)
    column_roles: dict[str, str] = field(default_factory=dict)
    analysis_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_candidates: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AnalysisConfig:
    """User-controlled settings for one supervised analysis."""

    target: str
    problem_type: ProblemType | Literal["auto"] = "auto"
    validation_strategy: Literal["holdout_cv"] = "holdout_cv"
    positive_label: Any | None = None
    effort: Literal["standard", "expanded"] = "standard"
    test_size: float = 0.2
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("AnalysisConfig.target must be a non-empty column name.")
        if self.problem_type not in {"auto", "classification", "regression"}:
            raise ValueError("problem_type must be auto, classification, or regression.")
        if self.validation_strategy != "holdout_cv":
            raise ValueError("Only holdout_cv validation is currently supported.")
        if self.effort not in {"standard", "expanded"}:
            raise ValueError("effort must be standard or expanded.")
        if not 0.1 <= self.test_size <= 0.4:
            raise ValueError("test_size must be between 0.1 and 0.4.")


@dataclass
class ModelBundle:
    """A fitted model plus everything required for repeatable raw-file scoring."""

    pipeline: Any
    target_column: str
    problem_type: ProblemType
    raw_schema: dict[str, str]
    required_feature_columns: list[str]
    optional_identifier_columns: list[str]
    feature_names: list[str]
    class_labels: list[Any] = field(default_factory=list)
    positive_label: Any | None = None
    negative_label: Any | None = None
    decision_threshold: float | None = None
    baseline_strategy: dict[str, Any] = field(default_factory=dict)
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    holdout_metrics: dict[str, Any] = field(default_factory=dict)
    cv_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    primary_metric: str = "f1_macro"
    training_reference: dict[str, Any] = field(default_factory=dict)
    identifier_reference: dict[str, set[Any]] = field(default_factory=dict)
    leakage_warnings: list[str] = field(default_factory=list)
    training_rows: int = 0
    holdout_rows: int = 0
    random_seed: int = 42
    version: str = "1.0.0"

    @property
    def named_steps(self) -> Any:
        """Keep compatibility with callers that previously received Pipeline."""

        return self.pipeline.named_steps

    def predict_proba(self, raw_frame: pd.DataFrame) -> Any:
        if not hasattr(self.pipeline, "predict_proba"):
            raise AttributeError("This model does not provide class probabilities.")
        return self.pipeline.predict_proba(raw_frame)

    def predict(self, raw_frame: pd.DataFrame) -> Any:
        if (
            self.problem_type == "classification"
            and self.decision_threshold is not None
            and len(self.class_labels) == 2
            and hasattr(self.pipeline, "predict_proba")
        ):
            import numpy as np

            probabilities = self.pipeline.predict_proba(raw_frame)
            positive_index = self.class_labels.index(self.positive_label)
            return np.where(
                probabilities[:, positive_index] >= self.decision_threshold,
                self.positive_label,
                self.negative_label,
            )
        return self.pipeline.predict(raw_frame)


@dataclass
class AnalysisResult:
    """Typed core result; ``to_payload`` supports the existing Streamlit UI."""

    workflow_decision: dict[str, Any]
    ranked_insights: list[dict[str, Any]] = field(default_factory=list)
    internal_validation: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    model_bundle: ModelBundle | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "decision": self.workflow_decision,
            "ranked_insights": self.ranked_insights,
            "internal_validation": self.internal_validation,
            "notes": self.notes,
            "model_bundle": self.model_bundle,
        }


@dataclass
class ScoringResult:
    """Predictions, compatibility warnings, drift, and optional evaluation."""

    scored_rows: pd.DataFrame
    schema_warnings: list[str] = field(default_factory=list)
    drift_summary: dict[str, Any] = field(default_factory=dict)
    external_metrics: dict[str, Any] | None = None
    readiness: dict[str, Any] = field(default_factory=dict)
    evaluation_status: Literal["not_requested", "completed", "blocked"] = "not_requested"
    prediction_column: str | None = None
    probability_columns: dict[str, str] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "scored_rows": self.scored_rows,
            "schema_warnings": self.schema_warnings,
            "drift_summary": self.drift_summary,
            "external_metrics": self.external_metrics,
            "readiness": self.readiness,
            "evaluation_status": self.evaluation_status,
            "prediction_column": self.prediction_column,
            "probability_columns": self.probability_columns,
        }

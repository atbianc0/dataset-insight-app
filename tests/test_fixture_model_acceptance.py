import os
import time
from pathlib import Path

import pandas as pd
import pytest

from src.contracts import AnalysisConfig
from src.modeling import train_model
from src.pipeline import sample_training_data
from src.profiling import build_dataset_profile
from src.scoring import score_or_evaluate

SAMPLE_DATA = Path(__file__).resolve().parents[1] / "sample_data"


def test_pr_fixture_gate_profiles_every_row_and_models_ten_thousand_rows():
    """Fast CI uses exact full profiles plus a bounded, representative model fit."""

    training = pd.read_csv(SAMPLE_DATA / "customer_churn_dataset-training-master.csv")
    profile = build_dataset_profile(training)
    usable = profile.sanitized_frame.dropna(subset=["Churn"])
    configured_rows = int(os.getenv("DATALENS_MODEL_SAMPLE_ROWS", "10000"))
    fast_rows = min(configured_rows, 10_000)
    sampled_X, sampled_y, sampled = sample_training_data(
        usable.drop(columns="Churn"),
        usable["Churn"],
        "classification",
        max_rows=fast_rows,
    )
    sampled_y.name = "Churn"

    trained = train_model(
        sampled_X,
        sampled_y,
        AnalysisConfig(target="Churn", problem_type="classification", random_seed=42),
    )
    bundle = trained["model_bundle"]

    assert profile.exact_overview["rows"] == 440_833
    assert profile.exact_overview["missing_cells"] == 12
    assert sampled is True
    assert len(sampled_X) == fast_rows
    assert {"Gender", "Subscription Type", "Contract Length"}.issubset(
        bundle.required_feature_columns
    )
    assert "CustomerID" in bundle.optional_identifier_columns
    assert bundle.positive_label == 1
    assert bundle.training_rows + bundle.holdout_rows == fast_rows


@pytest.mark.full
def test_release_fixture_gate_runs_sixty_thousand_rows_and_complete_external_validation():
    """Release evidence: full fixture, bounded model, and every external row."""

    training = pd.read_csv(SAMPLE_DATA / "customer_churn_dataset-training-master.csv")
    external = pd.read_csv(SAMPLE_DATA / "customer_churn_dataset-testing-master.csv")
    model_rows = int(os.getenv("DATALENS_MODEL_SAMPLE_ROWS", "60000"))
    external_rows = int(os.getenv("DATALENS_EXTERNAL_VALIDATION_ROWS", "64374"))
    assert model_rows == 60_000
    assert external_rows == 64_374

    profile = build_dataset_profile(training)
    usable = profile.sanitized_frame.dropna(subset=["Churn"])
    sampled_X, sampled_y, sampled = sample_training_data(
        usable.drop(columns="Churn"),
        usable["Churn"],
        "classification",
        max_rows=model_rows,
    )
    sampled_y.name = "Churn"
    training_started = time.perf_counter()
    trained = train_model(
        sampled_X,
        sampled_y,
        AnalysisConfig(target="Churn", problem_type="classification", random_seed=42),
    )
    bundle = trained["model_bundle"]
    # Entity-overlap checks intentionally retain the entire training identity
    # reference even though model fitting is bounded.
    bundle.identifier_reference = {
        "CustomerID": set(profile.sanitized_frame["CustomerID"].dropna().tolist())
    }
    scoring = score_or_evaluate(bundle, external.head(external_rows))
    training_and_scoring_seconds = time.perf_counter() - training_started

    assert sampled is True
    assert bundle.training_rows + bundle.holdout_rows == 60_000
    assert len(scoring.scored_rows) == 64_374
    assert scoring.scored_rows["CustomerID"].equals(external["CustomerID"])
    assert {"probability_0", "probability_1"}.issubset(scoring.scored_rows.columns)
    assert scoring.external_metrics["evaluated_rows"] == 64_374
    assert scoring.drift_summary["identifier_overlap"]["CustomerID"] == 62_995
    assert scoring.drift_summary["level"] in {"moderate", "high"}
    assert scoring.readiness["status"] == "not deployment-ready"
    assert scoring.readiness["performance_drop"] >= 0.10
    assert training_and_scoring_seconds < 60

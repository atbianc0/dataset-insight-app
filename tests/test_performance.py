import time
from pathlib import Path

import pandas as pd
import pytest

from src.profiling import build_dataset_profile

SAMPLE_DATA = Path(__file__).resolve().parents[1] / "sample_data"


@pytest.mark.full
@pytest.mark.parametrize(
    ("fixture_name", "maximum_seconds"),
    [
        ("netflix_titles.csv", 3),
        ("customer_churn_dataset-testing-master.csv", 8),
        ("customer_churn_dataset-training-master.csv", 20),
    ],
)
def test_reference_profile_performance_targets(fixture_name, maximum_seconds):
    frame = pd.read_csv(SAMPLE_DATA / fixture_name)

    started = time.perf_counter()
    profile = build_dataset_profile(frame)
    elapsed = time.perf_counter() - started

    assert profile.exact_overview["rows"] == len(frame)
    assert elapsed < maximum_seconds

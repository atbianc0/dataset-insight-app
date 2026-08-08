import pandas as pd
import pytest

from src.profiling import build_dataset_profile, fingerprint_dataframe


def test_profile_keeps_exact_counts_and_uses_a_deterministic_analysis_sample():
    frame = pd.DataFrame(
        {
            "CustomerID": [f"C{index:03d}" for index in range(100)],
            "value": list(range(99)) + [None],
            "Churn": ([0, 1] * 50),
        }
    )

    first = build_dataset_profile(frame, max_analysis_rows=20)
    second = build_dataset_profile(frame, max_analysis_rows=20)

    assert first.exact_overview["rows"] == 100
    assert first.exact_overview["missing_cells"] == 1
    assert len(first.analysis_frame) == 20
    pd.testing.assert_frame_equal(first.analysis_frame, second.analysis_frame)
    assert first.fingerprint == second.fingerprint
    assert first.column_roles["CustomerID"] == "identifier"


def test_only_credible_outcome_names_are_auto_selected():
    frame = pd.DataFrame(
        {
            "Age": range(20, 80),
            "type": ["A", "B"] * 30,
            "Churn": [0, 1] * 30,
            "measure": range(60),
        }
    )

    profile = build_dataset_profile(frame)
    by_name = {item["column"]: item for item in profile.target_candidates}

    assert by_name["Churn"]["auto_select"] is True
    assert by_name["Age"]["auto_select"] is False
    assert by_name["type"]["auto_select"] is False
    assert by_name["measure"]["auto_select"] is False


def test_profile_rejects_normalized_duplicate_headers():
    frame = pd.DataFrame([[1, 2]], columns=["value", " value "])

    with pytest.raises(ValueError, match="unique after trimming"):
        build_dataset_profile(frame)


def test_fingerprint_changes_with_values_or_schema():
    frame = pd.DataFrame({"value": [1, 2]})

    assert fingerprint_dataframe(frame) != fingerprint_dataframe(pd.DataFrame({"value": [1, 3]}))
    assert fingerprint_dataframe(frame) != fingerprint_dataframe(pd.DataFrame({"other": [1, 2]}))

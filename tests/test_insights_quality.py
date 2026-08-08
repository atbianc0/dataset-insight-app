from pathlib import Path

import numpy as np
import pandas as pd

from src.insights import run_insight_analysis

SAMPLE_DATA = Path(__file__).resolve().parents[1] / "sample_data"


def _row(frame, **conditions):
    selected = frame
    for column, value in conditions.items():
        selected = selected[selected[column] == value]
    assert not selected.empty, f"No row matched {conditions}"
    return selected.iloc[0]


def test_correlations_require_effect_support_coverage_and_adjusted_significance():
    rng = np.random.default_rng(2026)
    rows = 2_000
    signal = rng.normal(size=rows)
    frame = pd.DataFrame(
        {
            "signal": signal,
            "strong_pair": signal + rng.normal(scale=0.08, size=rows),
            "independent_noise": rng.normal(size=rows),
            "low_support_a": [*range(8), *([np.nan] * (rows - 8))],
            "low_support_b": [*range(8), *([np.nan] * (rows - 8))],
        }
    )

    result = run_insight_analysis(frame)
    correlations = result["correlations"]
    strong = _row(correlations, column_a="signal", column_b="strong_pair")
    low_support = _row(correlations, column_a="low_support_a", column_b="low_support_b")

    assert bool(strong["headline_eligible"]) is True
    assert strong["paired_n"] == rows
    assert strong["coverage_pct"] == 100.0
    assert "adjusted_p_value" in correlations.columns
    assert bool(low_support["headline_eligible"]) is False
    assert low_support["paired_n"] == 8
    assert low_support["coverage_pct"] == 0.4
    promoted_relationships = [
        headline for headline in result["headlines"] if headline.startswith("Supported numeric association")
    ]
    assert not any("low_support" in headline for headline in promoted_relationships)


def test_independent_noise_is_not_promoted_as_a_relationship():
    rng = np.random.default_rng(71)
    frame = pd.DataFrame({f"noise_{index}": rng.normal(size=4_000) for index in range(5)})

    result = run_insight_analysis(frame)

    assert not result["correlations"]["headline_eligible"].any()
    assert not any("Supported numeric association" in text for text in result["headlines"])
    assert "Correlation analysis" not in result["analysis_recommendations"]["analysis_type"].tolist()


def test_anomalies_are_not_forced_when_no_row_crosses_the_threshold():
    frame = pd.DataFrame(
        {
            "x": np.linspace(-1, 1, 500),
            "y": np.linspace(1, -1, 500),
            "z": np.sin(np.linspace(0, 2 * np.pi, 500)),
        }
    )

    result = run_insight_analysis(frame)

    assert result["anomaly_summary"].empty
    assert result["anomaly_highlight"] is None
    assert "Robust anomaly review" not in result["analysis_recommendations"]["analysis_type"].tolist()


def test_trends_default_to_counts_and_exclude_partial_boundary_periods():
    complete_year = pd.date_range("2021-01-01", "2022-12-31", freq="D")
    partial_year = pd.date_range("2023-01-01", "2023-03-10", freq="D")
    dates = complete_year.append(partial_year)
    frame = pd.DataFrame({"event_date": dates.astype(str), "arbitrary_metric": np.arange(len(dates))})

    trend = run_insight_analysis(frame)["trend_summary"]

    assert trend["aggregation"] == "row_count"
    assert trend["metric_column"] is None
    assert trend["partial_periods"] == ["2023"]
    assert trend["frame"]["value"].tolist() == [365, 365]
    assert "Excluded partial boundary" in trend["description"]


def test_multi_value_and_numeric_unit_fields_are_summarized_separately():
    frame = pd.DataFrame(
        {
            "topics": ["Analytics, Finance", "Finance", "Analytics, Health", "Analytics"] * 20,
            "duration": ["90 min", "110 min", "1 Season", "2 Seasons"] * 20,
            "description": [
                "A sentence with punctuation, context, and several clauses that should remain prose."
            ]
            * 80,
        }
    )

    result = run_insight_analysis(frame)
    multi = result["multi_value_summary"]
    units = result["unit_summary"]

    analytics = _row(multi, column="topics", value="Analytics")
    minutes = _row(units, column="duration", unit="minutes")
    seasons = _row(units, column="duration", unit="seasons")
    assert analytics["row_count"] == 60
    assert "description" not in multi["column"].tolist()
    assert minutes["row_count"] == 40
    assert minutes["median"] == 100
    assert seasons["row_count"] == 40
    assert seasons["median"] == 1.5
    assert result["best_analysis_path"]["analysis_type"] == "Multi-value category analysis"


def test_target_associations_use_counts_noncausal_wording_and_responsible_use_notice():
    rows = 120
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{index:04d}" for index in range(rows)],
            "Age": [20 + index % 40 for index in range(rows)],
            "Gender": ["Female", "Male"] * 60,
            "Contract Length": ["Monthly"] * 40 + ["Annual"] * 80,
            "Churn": [1] * 40 + ([0, 1] * 40),
        }
    )

    result = run_insight_analysis(frame)
    monthly = _row(
        result["target_associations"],
        association_kind="categorical_rate",
        feature="Contract Length",
        level="Monthly",
    )

    assert result["association_target"] == "Churn"
    assert result["association_target_inferred"] is True
    assert result["target_overview"]["usable_rows"] == rows
    assert monthly["row_count"] == 40
    assert monthly["target_count"] == 40
    assert monthly["target_rate"] == 1.0
    assert "Potential leakage or distribution-shift warning" in result["target_association_warnings"][0]
    assert "not causal" in " ".join(result["target_association_highlights"]).lower()
    notice = result["responsible_use_notes"][0]
    assert "Age" in notice and "Gender" in notice
    assert "fairness" in notice
    assert result["best_analysis_path"]["analysis_type"] == "Target-aware association analysis"


def test_pandas_string_target_and_category_dtypes_are_supported():
    frame = pd.DataFrame(
        {
            "segment": pd.Series(["A", "B"] * 40, dtype="category"),
            "spend": np.linspace(10.25, 93.75, 80),
            "Outcome": pd.Series(["Yes"] * 20 + ["No"] * 60, dtype="string"),
        }
    )

    result = run_insight_analysis(frame, target_col="Outcome")

    assert result["target_overview"]["positive_label"] == "Yes"
    assert result["target_overview"]["positive_rows"] == 20
    assert set(result["target_associations"]["feature"]) == {"segment", "spend"}
    assert result["association_target_inferred"] is False


def test_group_comparison_does_not_claim_meaning_or_significance():
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        {
            "segment": np.repeat(["A", "B", "C"], 100),
            "measure": rng.normal(size=300),
            "other_measure": rng.normal(size=300),
        }
    )

    highlight = run_insight_analysis(frame)["grouping_highlight"]

    assert highlight is not None
    assert "meaningful" not in highlight.lower()
    assert "is not evidence" in highlight.lower()
    assert "causation" in highlight.lower()


def test_netflix_fixture_surfaces_exact_and_format_aware_facts_without_a_false_target():
    frame = pd.read_csv(SAMPLE_DATA / "netflix_titles.csv")

    result = run_insight_analysis(frame)
    type_summary = _row(result["categorical_summary"], column="type")
    top_country = _row(result["multi_value_summary"], column="country", value="United States")
    top_genre = _row(
        result["multi_value_summary"], column="listed_in", value="International Movies"
    )
    minutes = _row(result["unit_summary"], column="duration", unit="minutes")
    seasons = _row(result["unit_summary"], column="duration", unit="seasons")

    assert result["overview"]["rows"] == 8_807
    assert result["overview"]["columns"] == 12
    assert result["overview"]["missing_cells"] == 4_307
    assert result["overview"]["top_missing_column"] == "director"
    assert round(result["overview"]["top_missing_pct"], 1) == 29.9
    assert type_summary["top_value"] == "Movie"
    assert type_summary["top_value_count"] == 6_131
    assert "TV Show (2,676)" in type_summary["top_values"]
    assert top_country["row_count"] == 3_690
    assert top_genre["row_count"] == 2_752
    assert minutes["row_count"] == 6_128 and minutes["median"] == 98
    assert seasons["row_count"] == 2_676 and seasons["median"] == 1
    assert result["trend_summary"]["partial_periods"] == ["2021"]
    assert result["association_target"] is None
    assert result["anomaly_summary"].empty
    assert result["best_analysis_path"]["analysis_type"] != "Exploratory data analysis"


def test_churn_training_fixture_reports_exact_target_support_and_perfect_monthly_association():
    frame = pd.read_csv(SAMPLE_DATA / "customer_churn_dataset-training-master.csv")

    result = run_insight_analysis(frame)
    monthly = _row(
        result["target_associations"],
        association_kind="categorical_rate",
        feature="Contract Length",
        level="Monthly",
    )

    assert result["overview"]["rows"] == 440_833
    assert result["overview"]["columns"] == 12
    assert result["overview"]["missing_cells"] == 12
    assert result["target_overview"]["usable_rows"] == 440_832
    assert result["target_overview"]["missing_rows"] == 1
    assert result["target_overview"]["positive_rows"] == 249_999
    assert result["target_overview"]["positive_rate"] == 0.567107
    assert monthly["row_count"] == 87_104
    assert monthly["target_count"] == 87_104
    assert monthly["target_rate"] == 1.0
    assert "Contract Length=Monthly" in result["target_association_warnings"][0]
    assert "Age" in result["responsible_use_notes"][0]
    assert "Gender" in result["responsible_use_notes"][0]


def test_churn_validation_fixture_reports_exact_counts_without_reusing_training_claims():
    frame = pd.read_csv(SAMPLE_DATA / "customer_churn_dataset-testing-master.csv")

    result = run_insight_analysis(frame)
    monthly = _row(
        result["target_associations"],
        association_kind="categorical_rate",
        feature="Contract Length",
        level="Monthly",
    )

    assert result["overview"]["rows"] == 64_374
    assert result["overview"]["columns"] == 12
    assert result["overview"]["missing_cells"] == 0
    assert result["target_overview"]["usable_rows"] == 64_374
    assert result["target_overview"]["positive_rows"] == 30_493
    assert result["target_overview"]["positive_rate"] == 0.473685
    assert monthly["row_count"] == 22_130
    assert monthly["target_count"] == 11_421
    assert monthly["target_rate"] == 0.5161
    assert not any("Contract Length=Monthly" in warning for warning in result["target_association_warnings"])

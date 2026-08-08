import os
import time as stdlib_time

import pytest
from streamlit.testing.v1 import AppTest, local_script_runner


class _MonotonicClock:
    """Clock facade that keeps AppTest timeouts immune to wall-clock adjustments."""

    time = staticmethod(stdlib_time.monotonic)
    sleep = staticmethod(stdlib_time.sleep)


@pytest.fixture(autouse=True)
def _use_monotonic_apptest_clock(monkeypatch):
    monkeypatch.setattr(local_script_runner, "time", _MonotonicClock)


def _app(timeout=30):
    return AppTest.from_file("app.py", default_timeout=timeout).run()


def _button(app, label):
    return next(button for button in app.button if button.label == label)


def _metric_value(app, label, occurrence=0):
    matches = [metric.value for metric in app.metric if metric.label == label]
    return matches[occurrence]


def test_blank_state_has_three_clear_sources_and_no_ai_requirement(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    app = _app()

    assert not app.exception
    assert not app.error
    assert [button.label for button in app.button] == [
        "Choose Upload",
        "Choose Netflix example",
        "Choose Churn example",
    ]
    assert any("No API key is required" in message.value for message in app.info)
    assert any("AI is off" in caption.value for caption in app.caption)
    assert all("XLS file" not in markdown.value for markdown in app.markdown)


def test_netflix_example_stays_insight_first_and_runs_without_ai(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _app()

    _button(app, "Choose Netflix example").click().run()

    assert not app.exception
    assert not app.error
    assert _metric_value(app, "Rows") == "8,807"
    assert _metric_value(app, "Columns") == "12"
    assert _metric_value(app, "Missing cells") == "4,308"
    assert _metric_value(app, "Suggested path") == "Insights"
    assert app.selectbox[0].label == "Outcome column"
    assert app.selectbox[0].value == "Insights only (no target)"

    _button(app, "Run analysis").click().run()

    assert not app.exception
    assert not app.error
    assert [tab.label for tab in app.tabs] == [
        "Decision",
        "Key insights",
        "Model validation",
        "Technical details",
        "Export",
    ]
    assert any(metric.label == "Workflow" and metric.value == "Insights" for metric in app.metric)
    assert any("6,131" in item.value and "Movie" in item.value for item in app.markdown)
    assert len(app.get("imgs")) == 1


def test_switching_sources_invalidates_the_prepared_dataset():
    app = _app()
    _button(app, "Choose Netflix example").click().run()
    assert _metric_value(app, "Rows") == "8,807"

    _button(app, "Choose Upload").click().run()

    assert not app.exception
    assert len(app.file_uploader) == 1
    assert app.file_uploader[0].label == "Upload a table"
    assert not app.metric
    assert "dataset_profile" not in app.session_state


def test_upload_duplicate_headers_is_a_recoverable_error():
    app = _app()
    _button(app, "Choose Upload").click().run()

    app.file_uploader[0].upload(
        "duplicate.csv",
        b"customer_id, customer_id\n1,2\n",
        "text/csv",
    ).run()

    assert not app.exception
    assert len(app.error) == 1
    assert "Duplicate columns: customer_id" in app.error[0].value


@pytest.mark.full
def test_churn_example_pairs_validation_data_and_selects_churn(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _app(timeout=60)

    _button(app, "Choose Churn example").click().run()

    assert not app.exception
    assert not app.error
    assert _metric_value(app, "Rows") == "440,833"
    assert _metric_value(app, "Columns") == "12"
    assert _metric_value(app, "Missing cells") == "12"
    assert app.selectbox[0].label == "Outcome column"
    assert app.selectbox[0].value == "Churn"
    assert len(app.session_state["validation_payload"]["dataframe"]) == 64_374
    assert app.session_state["validation_profile"].exact_overview["missing_cells"] == 0
    assert os.getenv("OPENAI_API_KEY") is None

    _button(app, "Run analysis").click().run(timeout=60)

    assert not app.exception
    assert any("not deployment-ready" in message.value.lower() for message in app.error)
    scoring = app.session_state["scoring_result"]
    assert len(scoring.scored_rows) == 64_374
    assert scoring.external_metrics["evaluated_rows"] == 64_374
    assert scoring.drift_summary["identifier_overlap_total"] == 62_995
    assert scoring.readiness["status"] == "not deployment-ready"
    assert {"probability_0", "probability_1"}.issubset(scoring.scored_rows.columns)

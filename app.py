"""DataLens Streamlit product surface.

The UI deliberately keeps uploaded data in ``st.session_state`` instead of a
process-wide cache.  Only the bundled, public example bytes use Streamlit's
cache.  Profiling and workflow results are keyed by an input fingerprint so a
dataset is prepared once and downstream state is invalidated predictably.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.ai_assistant import (
    ai_assistant_is_available,
    build_runtime_extension_registry,
    extract_stage_errors,
    extract_stage_payload,
)
from src.data_io import dataframe_to_csv_bytes, read_uploaded_table_details
from src.pipeline import recommend_dataset_workflow, run_analysis
from src.profiling import build_dataset_profile
from src.reporting import APP_VERSION, build_report_markdown
from src.scoring import score_or_evaluate

APP_ROOT = Path(__file__).resolve().parent
EXAMPLE_DIR = APP_ROOT / "sample_data"
NO_TARGET = "Insights only (no target)"
DOWNSTREAM_KEYS = (
    "analysis_result",
    "analysis_signature",
    "scoring_result",
    "scoring_signature",
    "report_markdown",
    "stage_timings",
)
INPUT_KEYS = (
    "data_input_key",
    "dataset_name",
    "upload_payload",
    "dataset_profile",
    "validation_payload",
    "validation_profile",
    "ingestion_timings",
    "workflow",
    "workflow_signature",
    *DOWNSTREAM_KEYS,
)


st.set_page_config(
    page_title="DataLens | Trustworthy Dataset Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#102a43; --muted:#627d98; --teal:#0b7285; --surface:#ffffff; }
    [data-testid="stAppViewContainer"] { background:#f6f8fb; }
    [data-testid="stHeader"] { background:rgba(246,248,251,.88); }
    [data-testid="stSidebar"] { background:#102a43; }
    [data-testid="stSidebar"] * { color:#f0f4f8; }
    [data-testid="stSidebar"] div[data-baseweb="select"] * {
        color:#102a43;
    }
    .block-container { max-width:1280px; padding-top:1.7rem; padding-bottom:4rem; }
    .hero { padding:2rem 2.2rem; border-radius:22px; color:white;
      background:linear-gradient(125deg,#102a43 0%,#16697a 72%,#168aad 100%);
      box-shadow:0 18px 45px rgba(16,42,67,.16); margin-bottom:1.25rem; }
    .hero h1 { color:white; font-size:clamp(2.1rem,5vw,3.6rem); line-height:1.02;
      letter-spacing:-.045em; margin:.4rem 0 .7rem; }
    .hero p { color:#d9eaf0; max-width:780px; font-size:1.06rem; margin:0; }
    .eyebrow { color:#90e0ef; font-weight:750; letter-spacing:.12em;
      text-transform:uppercase; font-size:.77rem; }
    div[data-testid="stMetric"], div[data-testid="stVerticalBlockBorderWrapper"] {
      background:var(--surface); border-radius:14px; }
    div[data-testid="stMetric"] { border:1px solid #d9e2ec; padding:1rem; }
    div[data-testid="stFileUploader"] { background:white; border:1px dashed #9fb3c8;
      border-radius:14px; padding:.4rem .8rem; }
    .stButton > button, .stDownloadButton > button { border-radius:10px; font-weight:650; }
    h2, h3 { color:var(--ink); letter-spacing:-.02em; }
    </style>
    """,
    unsafe_allow_html=True,
)


class UploadedFileShim:
    """Minimal upload protocol accepted by ``read_uploaded_table_details``."""

    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


@st.cache_data(show_spinner=False)
def load_builtin_example_bytes(file_name: str) -> bytes:
    """Cache only immutable, public example data bundled with the app."""

    allowed = {
        "netflix_titles.csv",
        "customer_churn_dataset-training-master.csv",
        "customer_churn_dataset-testing-master.csv",
    }
    if file_name not in allowed:
        raise ValueError("Unknown built-in example.")
    return (EXAMPLE_DIR / file_name).read_bytes()


def _fingerprint(file_name: str, content: bytes) -> str:
    return f"{file_name}:{hashlib.sha256(content).hexdigest()}"


def _clear_state(keys: tuple[str, ...]) -> None:
    for key in keys:
        st.session_state.pop(key, None)


def _select_source(source: str) -> None:
    if st.session_state.get("source_choice") != source:
        _clear_state(INPUT_KEYS)
        st.session_state["source_choice"] = source


def _timed(callable_, *args, **kwargs):
    started = time.perf_counter()
    value = callable_(*args, **kwargs)
    return value, time.perf_counter() - started


def _read_payload(file_name: str, content: bytes) -> dict[str, Any]:
    return read_uploaded_table_details(UploadedFileShim(file_name, content))


def _prepare_input(
    dataset_name: str,
    dataset_bytes: bytes,
    *,
    validation_name: str | None = None,
    validation_bytes: bytes | None = None,
) -> None:
    input_key = _fingerprint(dataset_name, dataset_bytes)
    if validation_name and validation_bytes is not None:
        input_key += ":" + _fingerprint(validation_name, validation_bytes)
    if st.session_state.get("data_input_key") == input_key:
        return

    _clear_state(INPUT_KEYS)
    timings: dict[str, float] = {}
    with st.status("Preparing the dataset", expanded=False) as status:
        status.write("Parsing and validating the source file…")
        payload, timings["parsing"] = _timed(_read_payload, dataset_name, dataset_bytes)

        status.write("Profiling schema, coverage, and exact dataset counts…")
        profile, timings["profiling"] = _timed(
            build_dataset_profile,
            payload["dataframe"],
            fingerprint=input_key,
        )

        validation_payload = None
        validation_profile = None
        if validation_name and validation_bytes is not None:
            status.write("Preparing the paired external-validation dataset…")
            started = time.perf_counter()
            validation_payload = _read_payload(validation_name, validation_bytes)
            validation_profile = build_dataset_profile(validation_payload["dataframe"])
            timings["validation profiling"] = time.perf_counter() - started

        elapsed = sum(timings.values())
        status.update(label=f"Dataset ready in {elapsed:.2f} seconds", state="complete")

    st.session_state.update(
        {
            "data_input_key": input_key,
            "dataset_name": dataset_name,
            "upload_payload": payload,
            "dataset_profile": profile,
            "validation_payload": validation_payload,
            "validation_profile": validation_profile,
            "ingestion_timings": timings,
        }
    )


def _format_seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f} s"


def _dataframe(frame: pd.DataFrame, *, height: int | None = None) -> None:
    display_frame = frame.copy()
    for column in display_frame.select_dtypes(include=["object"]).columns:
        non_null = display_frame[column].dropna()
        if non_null.map(type).nunique() > 1:
            display_frame[column] = display_frame[column].astype("string")
    if height is None:
        st.dataframe(display_frame, width="stretch", hide_index=True)
    else:
        st.dataframe(display_frame, width="stretch", hide_index=True, height=height)


def _render_ai_panel(extensions: dict[str, Any] | None) -> None:
    semantic = extract_stage_payload(extensions, "semantic_column_interpretation")
    task = extract_stage_payload(extensions, "task_understanding")
    report = extract_stage_payload(extensions, "report_generation")
    errors = (
        extract_stage_errors(extensions, "semantic_column_interpretation")
        + extract_stage_errors(extensions, "task_understanding")
        + extract_stage_errors(extensions, "report_generation")
    )
    if task.get("ai_dataset_summary"):
        st.info(task["ai_dataset_summary"])
    if task.get("ai_prediction_explanation"):
        st.write(task["ai_prediction_explanation"])
    notes = semantic.get("ai_semantic_column_notes")
    if notes:
        _dataframe(pd.DataFrame(notes))
    if report.get("ai_report_summary"):
        st.write(report["ai_report_summary"])
    for error in errors:
        st.warning(f"Optional AI interpretation was unavailable: {error['message']}")


def _numeric_metrics(metrics: dict[str, Any] | None) -> pd.DataFrame:
    rows = []
    for name, value in (metrics or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            rows.append({"metric": name.replace("_", " ").title(), "value": value})
    return pd.DataFrame(rows)


def _render_decision(result: dict[str, Any], scoring_result: Any | None) -> None:
    decision = result.get("decision", {})
    if result.get("mode") == "prediction":
        st.success(decision.get("summary", "Prediction workflow completed."))
    else:
        st.info(decision.get("summary", "Insight-first workflow completed."))

    details = decision.get("details") or []
    if details:
        st.markdown("\n".join(f"- {item}" for item in details))

    if result.get("mode") == "prediction":
        positive_label = result.get("positive_label")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Workflow", "Prediction")
        c2.metric("Target", result.get("selected_target") or "—")
        c3.metric(
            "Positive label",
            "Not applicable" if positive_label is None else str(positive_label),
        )
        c4.metric("Internal status", "Provisional")
        st.warning(
            "Internal validation is provisional. It does not establish deployment readiness without "
            "representative external validation."
        )
    else:
        overview = result["insight_analysis"]["overview"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Workflow", "Insights")
        c2.metric("Rows", f"{overview['rows']:,}")
        c3.metric("Columns", f"{overview['columns']:,}")

    if scoring_result is not None:
        readiness = scoring_result.readiness or {}
        status = readiness.get("status", "provisional")
        summary = readiness.get("summary", "")
        if status == "not deployment-ready":
            st.error(f"External verdict: {status}. {summary}")
        elif status == "externally validated":
            st.success(f"External verdict: {status}. {summary}")
        else:
            st.warning(f"External verdict: {status}. {summary}")


def _render_key_insights(result: dict[str, Any]) -> None:
    analysis = result["insight_analysis"]
    overview = analysis["overview"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{overview['rows']:,}")
    c2.metric("Columns", f"{overview['columns']:,}")
    c3.metric("Missing cells", f"{overview['missing_cells']:,}")
    c4.metric("Duplicate rows", f"{overview.get('duplicate_rows', 0):,}")

    headlines = analysis.get("headlines") or []
    st.subheader("Evidence-backed findings")
    if headlines:
        st.markdown("\n".join(f"- {line}" for line in headlines))
    else:
        st.info("No relationship met the support and effect thresholds for a headline conclusion.")

    responsible = analysis.get("responsible_use_notices") or []
    for notice in responsible:
        st.warning(notice)

    quality = analysis.get("data_quality_summary") or []
    if quality:
        st.subheader("Data quality")
        st.markdown("\n".join(f"- {line}" for line in quality))

    target_overview = analysis.get("target_overview")
    if target_overview:
        c1, c2, c3 = st.columns(3)
        c1.metric("Association target", target_overview["target_column"])
        c2.metric("Positive label", str(target_overview["positive_label"]))
        c3.metric("Positive rate", f"{target_overview['positive_rate']:.1%}")

    target_highlights = analysis.get("target_association_highlights") or []
    if target_highlights:
        st.subheader("Target-aware associations")
        st.markdown("\n".join(f"- {line}" for line in target_highlights))
    target_associations = analysis.get("target_associations")
    if isinstance(target_associations, pd.DataFrame) and not target_associations.empty:
        with st.expander("Association evidence and support counts", expanded=False):
            _dataframe(target_associations.head(100))

    multi_value = analysis.get("multi_value_summary")
    if isinstance(multi_value, pd.DataFrame) and not multi_value.empty:
        st.subheader("Multi-value fields")
        st.caption("Comma-separated fields are counted by represented row, not as one combined label.")
        _dataframe(multi_value)

    units = analysis.get("unit_summary")
    if isinstance(units, pd.DataFrame) and not units.empty:
        st.subheader("Values with units")
        st.caption("Different units are summarized separately so unlike quantities are not averaged together.")
        _dataframe(units)

    trend = analysis.get("trend_summary")
    if trend is not None:
        st.subheader("Time coverage and row-count trend")
        st.caption(trend["description"])
        trend_frame = trend.get("frame")
        if isinstance(trend_frame, pd.DataFrame) and not trend_frame.empty:
            chart_frame = trend_frame[["period", "value"]].copy()
            chart_frame["period"] = pd.to_datetime(chart_frame["period"], errors="coerce")
            chart_frame["value"] = pd.to_numeric(chart_frame["value"], errors="coerce")
            chart_frame = chart_frame.dropna()
            if not chart_frame.empty:
                fig, ax = plt.subplots(figsize=(9, 3.2))
                try:
                    ax.plot(chart_frame["period"], chart_frame["value"], marker="o", linewidth=2)
                    ax.set(xlabel="Period", ylabel="Rows")
                    ax.grid(axis="y", alpha=0.25)
                    fig.autofmt_xdate()
                    fig.tight_layout()
                    st.pyplot(fig)
                finally:
                    plt.close(fig)

    anomalies = analysis.get("anomaly_summary")
    if isinstance(anomalies, pd.DataFrame) and not anomalies.empty:
        st.subheader("Rows crossing the robust anomaly threshold")
        st.caption(analysis.get("anomaly_methodology", ""))
        _dataframe(anomalies)

    with st.expander("Distribution tables", expanded=False):
        numeric = analysis.get("numeric_summary")
        categorical = analysis.get("categorical_summary")
        if isinstance(numeric, pd.DataFrame) and not numeric.empty:
            st.caption("Numeric ranges and quantiles")
            _dataframe(numeric)
        if isinstance(categorical, pd.DataFrame) and not categorical.empty:
            st.caption("Categorical counts and proportions")
            _dataframe(categorical)


def _render_confusion_matrix(payload: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    try:
        matrix = payload["matrix"]
        labels = payload["labels"]
        image = ax.imshow(matrix, cmap="Blues")
        ax.set(title="Holdout confusion matrix", xlabel="Predicted", ylabel="Actual")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                ax.text(column_index, row_index, value, ha="center", va="center")
        fig.colorbar(image, ax=ax)
        st.pyplot(fig)
    finally:
        plt.close(fig)


def _render_model_validation(result: dict[str, Any], scoring_result: Any | None) -> None:
    if result.get("mode") != "prediction":
        attempt = result.get("predictive_attempt")
        if attempt:
            st.warning(attempt["quality"]["summary"])
            _dataframe(_numeric_metrics(attempt.get("best_metrics")))
        else:
            st.info("No predictive model was selected for this analysis.")
        return

    st.subheader("Internal validation — provisional")
    positive_label = result.get("positive_label")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", result.get("best_model_name", "—"))
    c2.metric("Rows modeled", f"{result.get('used_rows', 0):,}")
    c3.metric("Positive label", "N/A" if positive_label is None else str(positive_label))
    threshold = result.get("decision_threshold")
    c4.metric("Decision threshold", "Default" if threshold is None else f"{threshold:.3f}")

    internal = _numeric_metrics(result.get("best_metrics"))
    baseline = _numeric_metrics(result.get("baseline_metrics"))
    left, right = st.columns(2)
    with left:
        st.caption("Untouched internal holdout")
        _dataframe(internal)
    with right:
        st.caption("Train-derived baseline on the same holdout")
        _dataframe(baseline)

    cv_results = result.get("cv_results") or {}
    if cv_results:
        with st.expander("Cross-validation model selection", expanded=False):
            rows = []
            for model, metrics in cv_results.items():
                row = {"model": model}
                row.update(metrics)
                rows.append(row)
            _dataframe(pd.DataFrame(rows))

    chart = result.get("chart_context", {})
    if "confusion_matrix" in chart:
        _render_confusion_matrix(chart["confusion_matrix"])

    importance = result.get("feature_importance")
    if isinstance(importance, pd.DataFrame) and not importance.empty:
        st.subheader("Predictive associations on the holdout")
        st.caption(
            "Permutation importance describes predictive association in this dataset; it is not a causal driver analysis."
        )
        chart_frame = importance[["feature", "importance"]].copy()
        chart_frame["importance"] = pd.to_numeric(chart_frame["importance"], errors="coerce")
        chart_frame = chart_frame.dropna().head(20)
        if not chart_frame.empty:
            chart_frame = chart_frame.sort_values("importance")
            fig, ax = plt.subplots(figsize=(9, max(3.2, min(7.2, len(chart_frame) * 0.38))))
            try:
                ax.barh(chart_frame["feature"], chart_frame["importance"], color="#168aad")
                ax.set(xlabel="Permutation importance", ylabel="")
                ax.grid(axis="x", alpha=0.25)
                fig.tight_layout()
                st.pyplot(fig)
            finally:
                plt.close(fig)
        _dataframe(importance)

    if scoring_result is None:
        st.warning("No external validation result is available yet.")
        return

    st.subheader("External validation and distribution shift")
    readiness = scoring_result.readiness or {}
    status = readiness.get("status", "provisional")
    summary = readiness.get("summary", "")
    if status == "not deployment-ready":
        st.error(f"{status.title()}: {summary}")
    elif status == "externally validated":
        st.success(f"{status.title()}: {summary}")
    else:
        st.warning(f"{status.title()}: {summary}")

    for warning in scoring_result.schema_warnings:
        st.warning(warning)
    external = _numeric_metrics(scoring_result.external_metrics)
    if not external.empty:
        _dataframe(external)

    drift = scoring_result.drift_summary or {}
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Drift level", str(drift.get("level", "unknown")).title())
    d2.metric("Max numeric SMD", f"{drift.get('max_standardized_mean_difference', 0):.3f}")
    d3.metric("Max categorical TVD", f"{drift.get('max_total_variation_distance', 0):.3f}")
    d4.metric("Overlapping IDs", f"{drift.get('identifier_overlap_total', 0):,}")


def _render_technical(
    result: dict[str, Any],
    profile: Any,
    payload: dict[str, Any],
    workflow: dict[str, Any],
    scoring_result: Any | None,
) -> None:
    st.subheader("Schema and exact profile")
    overview = profile.exact_overview
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{overview['rows']:,}")
    c2.metric("Columns", f"{overview['columns']:,}")
    c3.metric("Cells", f"{overview['cells']:,}")
    c4.metric("Missing cells", f"{overview['missing_cells']:,}")
    _dataframe(pd.DataFrame(profile.column_profiles))

    for warning in payload.get("warnings", []):
        st.warning(warning)
    for warning in profile.warnings:
        st.info(warning)

    validation_profile = st.session_state.get("validation_profile")
    if validation_profile is not None:
        st.subheader("Paired external-validation profile")
        external = validation_profile.exact_overview
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Validation rows", f"{external['rows']:,}")
        e2.metric("Validation columns", f"{external['columns']:,}")
        e3.metric("Validation missing", f"{external['missing_cells']:,}")
        e4.metric("Validation duplicates", f"{external['duplicate_rows']:,}")

    st.subheader("Stage timings")
    timings = dict(st.session_state.get("ingestion_timings", {}))
    timings.update(st.session_state.get("stage_timings", {}))
    timing_frame = pd.DataFrame(
        [{"stage": name.title(), "elapsed": _format_seconds(seconds)} for name, seconds in timings.items()]
    )
    _dataframe(timing_frame)

    with st.expander("Workflow and target recommendations", expanded=False):
        st.write(workflow.get("summary", ""))
        recommendations = workflow.get("task_recommendations") or []
        if recommendations:
            _dataframe(pd.DataFrame(recommendations))
        targets = workflow.get("candidate_targets") or []
        if targets:
            columns = [
                "column",
                "status",
                "problem_type",
                "score",
                "usable_rows",
                "missing_pct",
                "unique_count",
                "summary",
            ]
            target_frame = pd.DataFrame(targets)
            _dataframe(target_frame[[column for column in columns if column in target_frame]])

    analysis = result["insight_analysis"]
    correlations = analysis.get("correlations")
    if isinstance(correlations, pd.DataFrame) and not correlations.empty:
        with st.expander("Correlation evidence", expanded=False):
            st.caption(analysis.get("correlation_methodology", ""))
            _dataframe(correlations)

    st.subheader("Raw-data preview")
    _dataframe(profile.sanitized_frame.head(25))
    st.caption(payload.get("source_summary", ""))

    if scoring_result is not None:
        per_column = scoring_result.drift_summary.get("per_column", {})
        if per_column:
            with st.expander("Per-column drift detail", expanded=False):
                rows = [{"column": name, **values} for name, values in per_column.items()]
                _dataframe(pd.DataFrame(rows))


def _render_export(result: dict[str, Any], dataset_name: str, scoring_result: Any | None) -> None:
    report = st.session_state.get("report_markdown")
    if not report:
        report = build_report_markdown(result, dataset_name, scoring_result=scoring_result)
    st.download_button(
        "Download methodology report",
        data=report.encode("utf-8"),
        file_name=f"{Path(dataset_name).stem}_datalens_report.md",
        mime="text/markdown",
        width="stretch",
    )
    with st.expander("Preview report", expanded=False):
        st.markdown(report)

    if scoring_result is not None:
        st.download_button(
            "Download scored rows",
            data=dataframe_to_csv_bytes(scoring_result.scored_rows),
            file_name="scored_predictions.csv",
            mime="text/csv",
            width="stretch",
        )
        st.caption(
            "The scored file preserves row order, identifiers, and original columns, then appends predictions and class probabilities."
        )


def _analysis_bundle(result: dict[str, Any]) -> Any | None:
    if result.get("model_bundle") is not None:
        return result["model_bundle"]
    attempt = result.get("predictive_attempt") or {}
    return attempt.get("model_bundle")


def _run_analysis_and_optional_validation(
    profile: Any,
    workflow: dict[str, Any],
    target: str | None,
    *,
    problem_type: str,
    effort: str,
    test_size: float,
    drop_identifiers: bool,
    positive_label: Any | None,
    random_seed: int,
    extension_registry: Any,
) -> None:
    timings: dict[str, float] = {}
    with st.status("Running the analysis", expanded=True) as status:
        status.write("Training and selecting the workflow with train-only transformations…")
        started = time.perf_counter()
        result = run_analysis(
            profile.sanitized_frame,
            target,
            problem_type_mode=problem_type,
            test_size=test_size,
            drop_identifier_columns=drop_identifiers,
            training_effort=effort.lower(),
            extension_registry=extension_registry,
            positive_label=positive_label,
            random_state=random_seed,
            dataset_profile=profile,
            precomputed_workflow=workflow,
        )
        timings["analysis and training"] = time.perf_counter() - started

        scoring_result = None
        bundle = _analysis_bundle(result)
        validation_payload = st.session_state.get("validation_payload")
        if bundle is not None and validation_payload is not None:
            status.write("Scoring every paired validation row and measuring drift…")
            started = time.perf_counter()
            scoring_result = score_or_evaluate(bundle, validation_payload["dataframe"])
            timings["external validation"] = time.perf_counter() - started

        status.write("Generating a reproducible methodology report…")
        started = time.perf_counter()
        report = build_report_markdown(
            result,
            st.session_state["dataset_name"],
            scoring_result=scoring_result,
            app_version=APP_VERSION,
        )
        timings["report generation"] = time.perf_counter() - started
        status.update(
            label=f"Analysis completed in {sum(timings.values()):.2f} seconds",
            state="complete",
            expanded=False,
        )

    st.session_state["analysis_result"] = result
    st.session_state["scoring_result"] = scoring_result
    st.session_state["report_markdown"] = report
    st.session_state["stage_timings"] = timings


def _score_uploaded_file(bundle: Any, uploaded_file: Any, result: dict[str, Any]) -> None:
    try:
        content = uploaded_file.getvalue()
        scoring_key = _fingerprint(uploaded_file.name, content)
        with st.status("Scoring the validation file", expanded=True) as status:
            status.write("Parsing and checking the scoring schema…")
            payload, parse_seconds = _timed(_read_payload, uploaded_file.name, content)
            status.write("Generating predictions, external metrics, and drift checks…")
            scoring_result, score_seconds = _timed(
                score_or_evaluate, bundle, payload["dataframe"]
            )
            status.write("Refreshing the report…")
            report, report_seconds = _timed(
                build_report_markdown,
                result,
                st.session_state["dataset_name"],
                scoring_result=scoring_result,
                app_version=APP_VERSION,
            )
            status.update(
                label=f"Scored {len(scoring_result.scored_rows):,} rows in {parse_seconds + score_seconds:.2f} seconds",
                state="complete",
                expanded=False,
            )
        st.session_state["scoring_result"] = scoring_result
        st.session_state["scoring_signature"] = scoring_key
        st.session_state["report_markdown"] = report
        timings = st.session_state.setdefault("stage_timings", {})
        timings["scoring file parsing"] = parse_seconds
        timings["external validation"] = score_seconds
        timings["report generation"] = report_seconds
    except Exception as exc:  # Streamlit must keep recoverable input errors on the page.
        st.error(f"Could not score this file: {exc}")


st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">DataLens v1.0</div>
      <h1>Evidence before conclusions.</h1>
      <p>Profile a tabular dataset, surface defensible insights, and validate a predictive model without hiding weak generalization.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## DataLens")
    st.caption(f"Version {APP_VERSION} · general-purpose tabular analysis")
    st.divider()
    with st.expander("Advanced modeling controls", expanded=False):
        problem_type_mode = st.selectbox(
            "Prediction task",
            ["Auto Detect", "Classification", "Regression"],
            help="Only applies when a credible target is selected.",
        )
        training_effort = st.selectbox(
            "Training effort",
            ["Standard", "Expanded"],
            help="Standard uses 3-fold model selection; Expanded uses 5 folds.",
        )
        test_size = st.slider("Untouched holdout size", 0.10, 0.35, 0.20, 0.05)
        drop_identifier_columns = st.checkbox("Exclude identifier columns", value=True)
        random_seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42)

    with st.expander("Optional AI and privacy", expanded=False):
        ai_available = ai_assistant_is_available()
        ai_requested = st.checkbox(
            "Enable optional AI interpretation",
            value=False,
            disabled=not ai_available,
            help="Deterministic analysis remains authoritative.",
        )
        ai_consent = False
        if ai_requested and ai_available:
            ai_consent = st.checkbox(
                "I consent to sending aggregate schema and statistics",
                value=False,
            )
        if not ai_available:
            st.caption("AI is off. No OpenAI secret is configured, and the app works fully without one.")
        st.caption(
            "AI receives aggregate schema/statistics only—never uploaded rows, sample values, identifier values, or name values. "
            "Uploaded data stays in this browser session and is not put in a shared cache."
        )
    ai_enabled = bool(ai_available and ai_requested and ai_consent)

st.markdown("## Choose a source")
source_columns = st.columns(3)
source_cards = [
    (
        "upload",
        "Upload",
        "Analyze your CSV, TSV, TXT, or XLSX in this session.",
        "Choose Upload",
    ),
    (
        "netflix",
        "Netflix example",
        "Explore titles, coverage, countries, genres, ratings, and duration units.",
        "Choose Netflix example",
    ),
    (
        "churn",
        "Churn + validation example",
        "Train on the large churn table and automatically evaluate the paired test table.",
        "Choose Churn example",
    ),
]
for column, (source_key, title, description, button_label) in zip(source_columns, source_cards):
    with column, st.container(border=True):
        st.markdown(f"### {title}")
        st.caption(description)
        st.button(
            button_label,
            key=f"source_button_{source_key}",
            on_click=_select_source,
            args=(source_key,),
            width="stretch",
            type="primary" if st.session_state.get("source_choice") == source_key else "secondary",
        )

source_choice = st.session_state.get("source_choice")
if source_choice is None:
    st.info("Choose one of the three sources above to begin. No API key is required.")
    st.stop()

dataset_name: str | None = None
dataset_bytes: bytes | None = None
validation_name: str | None = None
validation_bytes: bytes | None = None

if source_choice == "upload":
    uploaded = st.file_uploader(
        "Upload a table",
        type=["csv", "tsv", "txt", "xlsx"],
        help="Limits: 50 MB, 1,000,000 rows, 500 columns, and 20 million cells.",
    )
    if uploaded is None:
        st.info("Select a CSV, TSV, TXT, or XLSX file to continue.")
        st.stop()
    dataset_name = uploaded.name
    dataset_bytes = uploaded.getvalue()
elif source_choice == "netflix":
    dataset_name = "netflix_titles.csv"
    dataset_bytes = load_builtin_example_bytes(dataset_name)
    st.info("Netflix example selected. This path stays insight-first unless you explicitly choose a target.")
else:
    dataset_name = "customer_churn_dataset-training-master.csv"
    dataset_bytes = load_builtin_example_bytes(dataset_name)
    validation_name = "customer_churn_dataset-testing-master.csv"
    validation_bytes = load_builtin_example_bytes(validation_name)
    st.info("Churn training and testing files are paired. External evaluation will run automatically after training.")

try:
    _prepare_input(
        dataset_name,
        dataset_bytes,
        validation_name=validation_name,
        validation_bytes=validation_bytes,
    )
except Exception as exc:
    st.error(f"Could not prepare the dataset: {exc}")
    st.stop()

profile = st.session_state["dataset_profile"]
payload = st.session_state["upload_payload"]
input_key = st.session_state["data_input_key"]

extension_registry = build_runtime_extension_registry(enable_ai=ai_enabled)
workflow_signature = (input_key, drop_identifier_columns, ai_enabled)
if st.session_state.get("workflow_signature") != workflow_signature:
    _clear_state(DOWNSTREAM_KEYS)
    with st.status("Ranking analysis paths", expanded=False) as status:
        workflow, workflow_seconds = _timed(
            recommend_dataset_workflow,
            profile.sanitized_frame,
            drop_identifier_columns=drop_identifier_columns,
            top_n=min(8, len(profile.sanitized_frame.columns)),
            extension_registry=extension_registry,
        )
        status.update(
            label=f"Analysis paths ranked in {workflow_seconds:.2f} seconds",
            state="complete",
        )
    st.session_state["workflow"] = workflow
    st.session_state["workflow_signature"] = workflow_signature
    timings = st.session_state.setdefault("ingestion_timings", {})
    timings["insight profiling"] = workflow_seconds

workflow = st.session_state["workflow"]

# The profiler distinguishes columns that are merely modelable from credible
# outcome fields.  Preserve all candidates for deliberate user selection, but
# never advertise prediction as the default when no credible outcome exists.
credible_auto_targets = [
    item["column"] for item in profile.target_candidates if item.get("auto_select")
]
if not credible_auto_targets and workflow.get("recommended_workflow") == "prediction":
    workflow = dict(workflow)
    workflow["recommended_workflow"] = "insights"
    workflow["recommended_primary_target"] = None
    workflow["summary"] = (
        "No high-confidence business outcome stands out, so insight-first analysis is the safe default. "
        "Technically modelable columns remain available as intentional user choices."
    )
    st.session_state["workflow"] = workflow

overview = profile.exact_overview
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{overview['rows']:,}")
c2.metric("Columns", f"{overview['columns']:,}")
c3.metric("Missing cells", f"{overview['missing_cells']:,}")
c4.metric("Suggested path", workflow["recommended_workflow"].title())

st.markdown("## Set the analysis goal")
default_target = credible_auto_targets[0] if credible_auto_targets else NO_TARGET
target_options = [NO_TARGET, *profile.sanitized_frame.columns.tolist()]
target_key = f"target_selection_{input_key[:24]}"
selected_target_option = st.selectbox(
    "Outcome column",
    target_options,
    index=target_options.index(default_target) if default_target in target_options else 0,
    key=target_key,
    help="Only high-confidence outcome names are selected automatically. Any column remains available for intentional analysis.",
)
selected_target = None if selected_target_option == NO_TARGET else selected_target_option

positive_label_override = None
with st.expander("Advanced target semantics", expanded=False):
    if selected_target is None:
        st.caption("Choose an outcome column to review positive-label semantics.")
    else:
        labels = profile.sanitized_frame[selected_target].dropna().unique().tolist()
        if 1 < len(labels) <= 20:
            label_options: list[Any] = ["Automatic inference", *labels]
            chosen_label = st.selectbox(
                "Positive label",
                label_options,
                key=f"positive_label_{input_key[:16]}_{selected_target}",
                format_func=str,
                help="Used consistently for probabilities, AP, ROC-AUC, confusion matrices, and exports.",
            )
            if chosen_label != "Automatic inference":
                positive_label_override = chosen_label
        else:
            st.caption("Positive-label override is available for classification targets with 2–20 labels.")

analysis_signature = (
    input_key,
    selected_target,
    problem_type_mode,
    training_effort,
    test_size,
    drop_identifier_columns,
    positive_label_override,
    int(random_seed),
    ai_enabled,
)
if (
    st.session_state.get("analysis_result") is not None
    and st.session_state.get("analysis_signature") != analysis_signature
):
    _clear_state(DOWNSTREAM_KEYS)
    st.info("Analysis inputs changed. Run the analysis to produce a fresh result.")

if st.button("Run analysis", type="primary", width="stretch"):
    try:
        _run_analysis_and_optional_validation(
            profile,
            workflow,
            selected_target,
            problem_type=problem_type_mode,
            effort=training_effort,
            test_size=test_size,
            drop_identifiers=drop_identifier_columns,
            positive_label=positive_label_override,
            random_seed=int(random_seed),
            extension_registry=extension_registry,
        )
        st.session_state["analysis_signature"] = analysis_signature
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

result = st.session_state.get("analysis_result")
if result is None:
    st.caption(workflow.get("summary", ""))
    with st.expander("Preview the profile before running", expanded=False):
        _dataframe(pd.DataFrame(profile.column_profiles))
    st.stop()

bundle = _analysis_bundle(result)
if bundle is not None and st.session_state.get("validation_payload") is None:
    st.markdown("### Optional scoring or external validation")
    scoring_file = st.file_uploader(
        "Upload compatible raw rows",
        type=["csv", "tsv", "txt", "xlsx"],
        key=f"scoring_file_{input_key[:16]}",
        help="Extra columns and unseen categories are allowed. Only genuinely missing required fields block scoring.",
    )
    if scoring_file is not None and st.button("Score or evaluate file", width="stretch"):
        _score_uploaded_file(bundle, scoring_file, result)

scoring_result = st.session_state.get("scoring_result")
st.divider()
st.markdown("## Results")
decision_tab, insights_tab, validation_tab, technical_tab, export_tab = st.tabs(
    ["Decision", "Key insights", "Model validation", "Technical details", "Export"]
)
with decision_tab:
    _render_decision(result, scoring_result)
    _render_ai_panel(result.get("assistant_extensions"))
with insights_tab:
    _render_key_insights(result)
with validation_tab:
    _render_model_validation(result, scoring_result)
with technical_tab:
    _render_technical(result, profile, payload, workflow, scoring_result)
with export_tab:
    _render_export(result, st.session_state["dataset_name"], scoring_result)

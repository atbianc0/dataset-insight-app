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
from src.pipeline import (
    align_prediction_frame,
    recommend_dataset_workflow,
    run_analysis,
    sanitize_dataframe,
)


st.set_page_config(page_title="Dataset Insight App", layout="wide")


@st.cache_data(show_spinner=False)
def load_uploaded_table_payload(file_name, file_bytes):
    class UploadedFileShim:
        def __init__(self, name, content):
            self.name = name
            self._content = content

        def getvalue(self):
            return self._content

    return read_uploaded_table_details(UploadedFileShim(file_name, file_bytes))


def build_upload_key(file_name, file_bytes):
    return f"{file_name}:{len(file_bytes)}"


def render_ingestion_details(upload_payload):
    st.subheader("Ingestion")
    st.caption(upload_payload["source_summary"])

    if upload_payload["warnings"]:
        for warning in upload_payload["warnings"]:
            st.warning(warning)

    with st.expander("Schema Preview", expanded=False):
        st.dataframe(upload_payload["schema_preview"], use_container_width=True)


def render_dataset_recommendation(workflow):
    st.subheader("Dataset Recommendation")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Suggested Workflow", workflow["recommended_workflow"].title())
    col2.metric("Suggested Task", workflow["recommended_task_type"].title())
    col3.metric("Primary Target", workflow["recommended_primary_target"] or "None")
    col4.metric("Viable Targets", str(len(workflow["candidate_targets"])))

    if workflow["recommended_workflow"] == "prediction":
        st.success(workflow["summary"])
    else:
        st.warning(workflow["summary"])

    best_analysis_path = workflow.get("best_analysis_path")
    if best_analysis_path is not None:
        st.caption(
            f"Best no-target path: {best_analysis_path['analysis_type']}. "
            f"{best_analysis_path['reason']}"
        )

    st.dataframe(pd.DataFrame(workflow["task_recommendations"]), use_container_width=True)


def render_ai_assistant_panel(assistant_extensions):
    semantic_payload = extract_stage_payload(
        assistant_extensions,
        "semantic_column_interpretation",
    )
    task_payload = extract_stage_payload(
        assistant_extensions,
        "task_understanding",
    )
    report_payload = extract_stage_payload(
        assistant_extensions,
        "report_generation",
    )
    errors = (
        extract_stage_errors(assistant_extensions, "semantic_column_interpretation")
        + extract_stage_errors(assistant_extensions, "task_understanding")
        + extract_stage_errors(assistant_extensions, "report_generation")
    )

    has_ai_content = any(
        [
            semantic_payload.get("ai_semantic_column_notes"),
            task_payload.get("ai_dataset_summary"),
            task_payload.get("ai_target_suggestions"),
            task_payload.get("ai_feature_suggestions"),
            report_payload.get("ai_report_summary"),
        ]
    )
    if not has_ai_content and not errors:
        return

    st.subheader("AI Assistant Layer")
    st.caption(
        "Optional AI interpretation. Heuristic recommendations still control workflow decisions."
    )

    if task_payload.get("ai_dataset_summary"):
        st.write(task_payload["ai_dataset_summary"])

    if task_payload.get("ai_prediction_explanation"):
        st.info(task_payload["ai_prediction_explanation"])

    if semantic_payload.get("ai_semantic_column_notes"):
        st.caption("AI semantic column notes")
        st.dataframe(
            pd.DataFrame(semantic_payload["ai_semantic_column_notes"]),
            use_container_width=True,
        )

    if task_payload.get("ai_target_suggestions"):
        st.caption("AI target guidance")
        st.dataframe(pd.DataFrame(task_payload["ai_target_suggestions"]), use_container_width=True)

    if task_payload.get("ai_feature_suggestions"):
        st.caption("AI feature guidance")
        st.dataframe(pd.DataFrame(task_payload["ai_feature_suggestions"]), use_container_width=True)

    if task_payload.get("ai_cautions"):
        st.caption("AI cautions")
        st.markdown("\n".join([f"- {item}" for item in task_payload["ai_cautions"]]))

    if report_payload.get("ai_report_summary"):
        st.caption("AI report summary")
        st.write(report_payload["ai_report_summary"])

    if errors:
        for error in errors:
            st.warning(f"AI assistant unavailable: {error['message']}")


def render_target_candidate_tables(workflow):
    st.subheader("Ranked Target Candidates")
    if workflow["candidate_targets"]:
        candidate_rows = []
        for candidate in workflow["candidate_targets"]:
            candidate_rows.append(
                {
                    "column": candidate["column"],
                    "status": candidate["status"].title(),
                    "suggested_task": candidate["problem_type"] or "Insights",
                    "target_shape": candidate["target_shape"],
                    "score": candidate["score"],
                    "usable_rows": candidate["usable_rows"],
                    "missing_pct": candidate["missing_pct"],
                    "unique_count": candidate["unique_count"],
                    "why_it_fits": "; ".join(candidate["pros"][:2]) or candidate["summary"],
                    "watch_out_for": "; ".join(candidate["cautions"][:2]),
                }
            )
        st.dataframe(pd.DataFrame(candidate_rows), use_container_width=True)
    else:
        st.info("No strong supervised target stands out. The dataset looks better suited for insight-focused analysis.")

    if workflow["multi_target_candidates"]:
        st.subheader("Potential Multi-Target Groups")
        multi_rows = []
        for group in workflow["multi_target_candidates"]:
            multi_rows.append(
                {
                    "group": group["group_label"],
                    "problem_type": group["problem_type"],
                    "columns": ", ".join(group["columns"]),
                    "reason": group["reason"],
                }
            )
        st.dataframe(pd.DataFrame(multi_rows), use_container_width=True)
        st.caption(
            "These groups suggest related outcomes that may be worth modeling together. "
            "The current app still trains one selected target at a time."
        )

    with st.expander("Rejected Target Candidates", expanded=False):
        if workflow["rejected_target_candidates"]:
            rejected_rows = []
            for candidate in workflow["rejected_target_candidates"]:
                rejected_rows.append(
                    {
                        "column": candidate["column"],
                        "reason": "; ".join(candidate["blockers"][:2]) or candidate["summary"],
                        "cautions": "; ".join(candidate["cautions"][:2]),
                    }
                )
            st.dataframe(pd.DataFrame(rejected_rows), use_container_width=True)
        else:
            st.write("No columns were strongly rejected by the heuristic review.")


def render_target_review(workflow, selected_target_option):
    st.subheader("Current Selection Review")
    if selected_target_option == "No target / insights only":
        best_analysis_path = workflow.get("best_analysis_path")
        if best_analysis_path is not None:
            st.info(
                f"No target selected. The strongest next step is "
                f"{best_analysis_path['analysis_type'].lower()} because {best_analysis_path['reason'].lower()}"
            )
        else:
            st.info("No target selected. The app will stay in insights mode and summarize the dataset.")
        return

    candidate = workflow["candidate_lookup"].get(selected_target_option)
    if candidate is None:
        st.info("No recommendation details are available for the selected column.")
        return

    status = candidate["status"]
    if status == "recommended":
        st.success(candidate["summary"])
    elif status == "possible":
        st.warning(candidate["summary"])
    else:
        st.error(candidate["summary"])

    info1, info2, info3, info4 = st.columns(4)
    info1.metric("Target Status", status.title())
    info2.metric("Suggested Task", candidate["problem_type"] or "Insights")
    info3.metric("Target Shape", candidate["target_shape"])
    info4.metric("Usable Rows", f"{candidate['usable_rows']:,}")

    if candidate["pros"]:
        st.caption("Why this target may be useful")
        st.markdown("\n".join([f"- {item}" for item in candidate["pros"]]))

    if candidate["cautions"]:
        st.caption("Things to watch")
        st.markdown("\n".join([f"- {item}" for item in candidate["cautions"]]))

    if candidate["blockers"]:
        st.caption("Why prediction may not be appropriate")
        st.markdown("\n".join([f"- {item}" for item in candidate["blockers"]]))

    if candidate["suggested_feature_subset"] or candidate["rejected_feature_columns"]:
        col1, col2 = st.columns(2)
        with col1:
            if candidate["suggested_feature_subset"]:
                st.caption("Likely modeling inputs for this target")
                st.write(", ".join(candidate["suggested_feature_subset"]))
        with col2:
            if candidate["rejected_feature_columns"]:
                st.caption("Columns usually excluded for this target")
                st.write(", ".join(candidate["rejected_feature_columns"]))


def render_modeling_column_notes(workflow):
    subset = workflow["feature_subset_summary"]
    st.subheader("Modeling Column Notes")
    counts = subset.get("counts", {})

    col1, col2, col3 = st.columns(3)
    col1.metric("Likely Useful", counts.get("likely_useful", len(subset["likely_useful"])))
    col2.metric("Risky", counts.get("risky", len(subset["risky"])))
    col3.metric("Avoid", counts.get("avoid", len(subset["avoid"])))

    if subset["likely_useful"]:
        st.caption("Likely useful inputs")
        st.dataframe(pd.DataFrame(subset["likely_useful"]), use_container_width=True)

    if subset["risky"]:
        st.caption("Columns to use carefully")
        st.dataframe(pd.DataFrame(subset["risky"]), use_container_width=True)

    if subset["avoid"]:
        st.caption("Columns that are usually poor modeling inputs")
        st.dataframe(pd.DataFrame(subset["avoid"]), use_container_width=True)


def render_prediction_metric_cards(result):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Workflow", "Prediction")
    col2.metric("Problem Type", result["problem_type"].title())
    col3.metric("Best Model", result["best_model_name"])
    col4.metric("Rows Used", f"{result['used_rows']:,}")


def render_quality_summary(result):
    st.subheader("Model Quality")
    quality = result["quality"]
    baseline = result["baseline_metrics"]
    best = result["best_metrics"]

    q1, q2, q3 = st.columns(3)
    q1.metric("Model Worth", quality["verdict"].title())
    q2.metric(
        f"Best {result['metric_name'].upper()}",
        f"{best[result['metric_name']]:.3f}",
    )
    q3.metric(
        f"Baseline {result['metric_name'].upper()}",
        f"{baseline[result['metric_name']]:.3f}",
    )

    st.write(quality["summary"])

    metric_frame = pd.DataFrame(
        {"best_model": best, "baseline": {k: v for k, v in baseline.items() if k in best}}
    )
    st.dataframe(metric_frame)
    st.caption(baseline["baseline_strategy"])


def render_confusion_matrix(confusion_payload):
    fig, ax = plt.subplots(figsize=(6, 4))
    matrix = confusion_payload["matrix"]
    labels = confusion_payload["labels"]
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, value, ha="center", va="center")

    fig.colorbar(image, ax=ax)
    st.pyplot(fig)


def render_relevant_prediction_charts(result):
    st.subheader("Relevant Figures")
    chart_context = result["chart_context"]
    rendered_any = False

    if result["problem_type"] == "classification":
        target_counts = result["target_series"].value_counts(dropna=False).sort_index()
        if len(target_counts) <= 25:
            st.caption("Target class balance")
            st.bar_chart(target_counts)
            rendered_any = True

        if "confusion_matrix" in chart_context:
            st.caption("Where the model predicts correctly vs incorrectly")
            render_confusion_matrix(chart_context["confusion_matrix"])
            rendered_any = True

        for relationship in chart_context["relationships"]:
            column = relationship["column"]
            frame = relationship["frame"].dropna()
            if frame.empty:
                continue

            if frame["feature"].dtype == "O" or str(frame["feature"].dtype).startswith("category"):
                grouped = pd.crosstab(frame["feature"], frame["target"], normalize="index")
                if 1 < len(grouped) <= 20:
                    st.caption(f"Class mix by {column}")
                    st.bar_chart(grouped)
                    rendered_any = True
                    break
            else:
                grouped = frame.groupby("target")["feature"].median()
                if 1 < len(grouped) <= 20:
                    st.caption(f"Median {column} by class")
                    st.bar_chart(grouped)
                    rendered_any = True
                    break
    else:
        compare_df = pd.DataFrame(
            {
                "actual": chart_context["holdout_actual"],
                "predicted": chart_context["holdout_pred"],
            }
        )
        st.caption("Actual vs predicted holdout values")
        st.scatter_chart(compare_df, x="actual", y="predicted")
        rendered_any = True

        residual_df = pd.DataFrame(
            {"residual": chart_context["holdout_actual"] - chart_context["holdout_pred"]}
        )
        st.caption("Residual distribution")
        st.bar_chart(
            residual_df["residual"]
            .round(1)
            .value_counts()
            .sort_index()
        )
        rendered_any = True

        for relationship in chart_context["relationships"]:
            column = relationship["column"]
            frame = relationship["frame"].dropna()
            if frame.empty:
                continue

            if frame["feature"].dtype == "O" or str(frame["feature"].dtype).startswith("category"):
                grouped = frame.groupby("feature")["target"].mean().sort_values(ascending=False)
                if 1 < len(grouped) <= 20:
                    st.caption(f"Average target by {column}")
                    st.bar_chart(grouped)
                    rendered_any = True
                    break
            else:
                st.caption(f"Target relationship with {column}")
                st.scatter_chart(frame, x="feature", y="target")
                rendered_any = True
                break

    if not rendered_any and not result["feature_importance"].empty:
        st.caption("Top feature importance")
        st.bar_chart(result["feature_importance"].set_index("feature"))
        rendered_any = True

    if not rendered_any:
        st.info("No high-signal predictive chart was available for this dataset shape.")


def render_decision_summary(result):
    st.subheader("Workflow Decision")
    decision = result["decision"]
    mode_label = "Prediction mode" if result["mode"] == "prediction" else "Insights mode"

    if result["mode"] == "prediction":
        st.success(f"{mode_label}: {decision['summary']}")
    else:
        st.warning(f"{mode_label}: {decision['summary']}")

    if decision.get("details"):
        st.markdown("\n".join([f"- {detail}" for detail in decision["details"]]))

    assessment = result.get("target_assessment")
    if assessment is not None:
        st.caption(
            f"Target review: {assessment['usable_rows']:,} usable rows, "
            f"{assessment['unique_count']:,} unique values, "
            f"{assessment['usable_feature_count']:,} usable feature columns after preparation."
        )


def render_headlines(analysis):
    if analysis["headlines"]:
        st.subheader("Key Conclusions")
        st.markdown("\n".join([f"- {line}" for line in analysis["headlines"]]))


def render_analysis_mode(result):
    analysis = result["insight_analysis"]
    overview = analysis["overview"]

    top_focus = (
        analysis["analysis_recommendations"].iloc[0]["analysis_type"]
        if not analysis["analysis_recommendations"].empty
        else "Exploratory data analysis"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Workflow", "Insights")
    col2.metric("Rows", f"{overview['rows']:,}")
    col3.metric("Columns", f"{overview['columns']:,}")
    col4.metric("Recommended Focus", top_focus)

    render_headlines(analysis)

    if analysis["data_quality_summary"]:
        st.subheader("Data Quality Summary")
        st.markdown("\n".join([f"- {line}" for line in analysis["data_quality_summary"]]))

    st.subheader("Recommended Analysis Paths")
    st.dataframe(analysis["analysis_recommendations"], use_container_width=True)

    if not analysis["numeric_summary"].empty:
        st.subheader("Numeric Summary")
        st.dataframe(analysis["numeric_summary"], use_container_width=True)

    if not analysis["categorical_summary"].empty:
        st.subheader("Categorical Summary")
        st.dataframe(analysis["categorical_summary"], use_container_width=True)

    if not analysis["correlations"].empty:
        st.subheader("Top Correlations")
        corr_chart = analysis["correlations"].copy()
        corr_chart["pair"] = corr_chart["column_a"] + " vs " + corr_chart["column_b"]
        st.bar_chart(corr_chart.set_index("pair")["abs_correlation"])
        st.dataframe(analysis["correlations"], use_container_width=True)

    trend_summary = analysis["trend_summary"]
    if trend_summary is not None:
        st.subheader("Trend View")
        st.caption(trend_summary["description"])
        trend_frame = trend_summary["frame"].copy().set_index("period")
        st.line_chart(trend_frame["value"])

    if not analysis["group_summary"].empty:
        st.subheader("Grouping Insight")
        if analysis["grouping_highlight"]:
            st.caption(analysis["grouping_highlight"])
        st.caption(
            f"Average {analysis['group_summary'].iloc[0]['metric_column']} by "
            f"{analysis['group_summary'].iloc[0]['category_column']}"
        )
        st.bar_chart(analysis["group_summary"].set_index("group")["average_value"])
        st.dataframe(analysis["group_summary"], use_container_width=True)

    if not analysis["anomaly_summary"].empty:
        st.subheader("Potential Anomalies")
        if analysis["anomaly_highlight"]:
            st.caption(analysis["anomaly_highlight"])
        st.dataframe(analysis["anomaly_summary"], use_container_width=True)

    predictive_attempt = result.get("predictive_attempt")
    if predictive_attempt is not None:
        st.subheader("Why Prediction Was Not Used")
        st.write(predictive_attempt["quality"]["summary"])
        attempt_frame = pd.DataFrame(
            {
                "best_model": predictive_attempt["best_metrics"],
                "baseline": {
                    key: value
                    for key, value in predictive_attempt["baseline_metrics"].items()
                    if key in predictive_attempt["best_metrics"]
                },
            }
        )
        st.dataframe(attempt_frame, use_container_width=True)


def render_additional_insights(result):
    analysis = result["insight_analysis"]
    render_headlines(analysis)

    if not analysis["correlations"].empty:
        st.subheader("Additional Insight Signals")
        st.dataframe(analysis["correlations"], use_container_width=True)

    if analysis["trend_summary"] is not None:
        st.caption(analysis["trend_summary"]["description"])
        st.line_chart(analysis["trend_summary"]["frame"].set_index("period")["value"])

    if not analysis["group_summary"].empty:
        st.caption(
            f"Grouping highlight: average {analysis['group_summary'].iloc[0]['metric_column']} "
            f"by {analysis['group_summary'].iloc[0]['category_column']}."
        )
        st.dataframe(analysis["group_summary"], use_container_width=True)


def _markdown_bullets(items, fallback=None):
    if not items:
        return [f"- {fallback}"] if fallback else []
    return [f"- {item}" for item in items]


def _format_feature_guidance(items):
    return [
        f"- `{item['column']}`: {item['guidance']}. {item['reason']}"
        for item in items
    ]


def build_report_markdown(result, dataset_name):
    workflow = result.get("dataset_recommendation") or {}
    analysis = result["insight_analysis"]
    overview = analysis["overview"]
    task_ai_payload = extract_stage_payload(result.get("assistant_extensions"), "task_understanding")
    report_ai_payload = extract_stage_payload(result.get("assistant_extensions"), "report_generation")
    lines = [
        f"# Dataset Insight Report",
        "",
        f"- Dataset: `{dataset_name}`",
        f"- Workflow selected: `{result['mode']}`",
        f"- Selected target: `{result.get('selected_target') or 'None'}`",
        f"- Rows: `{overview['rows']:,}`",
        f"- Columns: `{overview['columns']:,}`",
        "",
        "## Dataset Recommendation",
        workflow.get("summary", "No dataset-level recommendation summary available."),
    ]

    if workflow.get("recommended_primary_target"):
        lines.append(f"Suggested primary target: {workflow['recommended_primary_target']}")
    if workflow.get("recommended_target_columns")[1:]:
        lines.append(
            "Other viable targets: "
            + ", ".join(workflow["recommended_target_columns"][1:5])
        )
    if workflow.get("best_analysis_path") is not None:
        lines.append(
            f"Best no-target path: {workflow['best_analysis_path']['analysis_type']} "
            f"because {workflow['best_analysis_path']['reason'].lower()}"
        )

    lines.extend(
        [
            "",
            "## Workflow Decision",
            result["decision"]["summary"],
        ]
    )

    if task_ai_payload.get("ai_dataset_summary") or report_ai_payload.get("ai_report_summary"):
        lines.extend(
            [
                "",
                "## AI Assistant Summary",
                task_ai_payload.get("ai_dataset_summary")
                or report_ai_payload.get("ai_report_summary"),
            ]
        )
        if task_ai_payload.get("ai_prediction_explanation"):
            lines.append(task_ai_payload["ai_prediction_explanation"])

    lines.extend(_markdown_bullets(result["decision"].get("details", [])))

    target_assessment = result.get("target_assessment")
    if target_assessment is not None:
        lines.extend(
            [
                "",
                "## Target Review",
                f"- Usable rows: {target_assessment['usable_rows']:,}",
                f"- Unique target values: {target_assessment['unique_count']:,}",
                f"- Usable feature columns after preparation: {target_assessment['usable_feature_count']:,}",
            ]
        )

    if workflow.get("multi_target_candidates"):
        lines.extend(["", "## Potential Multi-Target Options"])
        for group in workflow["multi_target_candidates"][:3]:
            lines.append(f"- {', '.join(group['columns'])}: {group['reason']}")

    if analysis["headlines"]:
        lines.extend(["", "## Key Conclusions"])
        lines.extend(_markdown_bullets(analysis["headlines"]))

    if analysis["data_quality_summary"]:
        lines.extend(["", "## Data Quality Summary"])
        lines.extend(_markdown_bullets(analysis["data_quality_summary"]))

    feature_subset = workflow.get("feature_subset_summary") or {}
    if feature_subset:
        lines.extend(["", "## Feature Guidance"])
        if feature_subset.get("likely_useful"):
            lines.append("Likely useful columns:")
            lines.extend(_format_feature_guidance(feature_subset["likely_useful"][:5]))
        if feature_subset.get("risky"):
            lines.append("Risky columns:")
            lines.extend(_format_feature_guidance(feature_subset["risky"][:5]))
        if feature_subset.get("avoid"):
            lines.append("Avoid columns:")
            lines.extend(_format_feature_guidance(feature_subset["avoid"][:5]))

    if result["mode"] == "prediction":
        lines.extend(
            [
                "",
                "## Predictive Model Summary",
                f"Problem type: {result['problem_type']}",
                f"Target style: {result['target_style']['label']}",
                f"Best model: {result['best_model_name']}",
                f"Model worth: {result['quality']['verdict']}",
                "",
                "### Metrics",
            ]
        )
        for name, value in result["best_metrics"].items():
            lines.append(f"- {name}: {value:.4f}")
        lines.extend(
            [
                "",
                "### Scoring Guidance",
                "- A second file can be scored when it provides the same modeling-ready columns used during training.",
                "- If scoring fails, compare the scoring file schema against the training feature columns shown in the app.",
            ]
        )
    else:
        if not analysis["analysis_recommendations"].empty:
            lines.extend(["", "## Recommended Analysis Paths"])
            for _, row in analysis["analysis_recommendations"].iterrows():
                lines.append(f"- {row['analysis_type']}: {row['reason']}")

        if analysis["grouping_highlight"]:
            lines.extend(["", "## Grouping Insight", analysis["grouping_highlight"]])

        if analysis["trend_summary"] is not None:
            lines.extend(["", "## Trend Summary", analysis["trend_summary"]["description"]])

        if analysis["anomaly_highlight"]:
            lines.extend(["", "## Anomaly Summary", analysis["anomaly_highlight"]])

        predictive_attempt = result.get("predictive_attempt")
        if predictive_attempt is not None:
            lines.extend(
                [
                    "",
                    "## Prediction Attempt Summary",
                    f"Best model tested: {predictive_attempt['best_model_name']}",
                    predictive_attempt["quality"]["summary"],
                ]
            )

    if result["notes"]:
        lines.extend(["", "## Preparation Notes"])
        lines.extend(_markdown_bullets(result["notes"][:12]))

    return "\n".join(lines)


st.title("Dataset Insight App")
st.write(
    "Upload a tabular dataset and the app will inspect the columns, decide whether "
    "prediction actually makes sense, and either train a model or switch into a more "
    "useful insight-focused analysis flow."
)

with st.sidebar:
    st.header("Settings")
    problem_type_mode = st.selectbox(
        "Prediction type override",
        ["Auto Detect", "Classification", "Regression"],
        help="Only used when the app decides prediction mode is appropriate.",
    )
    training_effort = st.selectbox(
        "Training effort",
        ["Standard", "Expanded"],
        help="Expanded tries more complex models and usually takes longer.",
    )
    test_size = st.slider("Holdout test size", 0.1, 0.35, 0.2, 0.05)
    drop_identifier_columns = st.checkbox("Drop identifier-like columns", value=True)
    ai_assistant_enabled = False
    if ai_assistant_is_available():
        ai_assistant_enabled = st.checkbox(
            "Enable AI assistant layer",
            value=False,
            help="Adds optional semantic interpretation and plain-language guidance. It does not override heuristic workflow decisions.",
        )
    else:
        st.caption("Set `OPENAI_API_KEY` and install `openai` to enable the optional AI assistant layer.")

training_file = st.file_uploader("Upload a dataset", type=["csv", "tsv", "txt", "xlsx", "xls"])
prediction_file = st.file_uploader(
    "Optional: upload a second dataset for scoring",
    type=["csv", "tsv", "txt", "xlsx", "xls"],
    help="This is only used when the app stays in prediction mode.",
)

if training_file is not None:
    upload_key = build_upload_key(training_file.name, training_file.getvalue())
    previous_upload_key = st.session_state.get("preliminary_report_upload_key")
    if previous_upload_key != upload_key:
        st.session_state["preliminary_report_upload_key"] = upload_key
        st.session_state["preliminary_report_ready"] = False

    try:
        upload_payload = load_uploaded_table_payload(training_file.name, training_file.getvalue())
        df = sanitize_dataframe(upload_payload["dataframe"])
    except Exception as exc:
        st.error(f"Could not read the dataset: {exc}")
        st.stop()

    if df.empty:
        st.error("The uploaded dataset has no rows.")
        st.stop()

    st.subheader("Start Here")
    st.write(
        "Upload complete. Start the preliminary report to inspect the dataset and get "
        "recommendations before choosing a modeling target."
    )

    if st.button("Start Preliminary Report", type="primary"):
        st.session_state["preliminary_report_ready"] = True

    if not st.session_state.get("preliminary_report_ready", False):
        st.stop()

    render_ingestion_details(upload_payload)

    st.subheader("Dataset Snapshot")
    st.dataframe(df.head(10), use_container_width=True)

    info1, info2, info3 = st.columns(3)
    info1.metric("Rows", f"{df.shape[0]:,}")
    info2.metric("Columns", f"{df.shape[1]:,}")
    info3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")

    extension_registry = build_runtime_extension_registry(enable_ai=ai_assistant_enabled)

    workflow = recommend_dataset_workflow(
        df,
        drop_identifier_columns=drop_identifier_columns,
        top_n=min(8, len(df.columns)),
        extension_registry=extension_registry,
    )

    render_dataset_recommendation(workflow)
    render_ai_assistant_panel(workflow.get("assistant_extensions"))
    render_target_candidate_tables(workflow)
    render_modeling_column_notes(workflow)

    st.subheader("Column Inspection")
    st.dataframe(workflow["insight_analysis"]["column_inspection"], use_container_width=True)

    target_options = ["No target / insights only"] + df.columns.tolist()
    default_target_index = 0
    if (
        workflow["recommended_workflow"] == "prediction"
        and workflow["recommended_primary_target"] in df.columns
    ):
        default_target_index = target_options.index(workflow["recommended_primary_target"])
    selected_target_option = st.selectbox(
        "Choose a target column for modeling",
        target_options,
        index=default_target_index,
        help="Leave this on the first option when the dataset is better suited for general analysis than prediction.",
    )

    render_target_review(workflow, selected_target_option)

    if st.button("Analyze Dataset", type="primary"):
        target_col = None if selected_target_option == target_options[0] else selected_target_option

        with st.spinner("Inspecting the dataset and choosing the right workflow..."):
            try:
                result = run_analysis(
                    df,
                    target_col,
                    problem_type_mode=problem_type_mode,
                    test_size=test_size,
                    drop_identifier_columns=drop_identifier_columns,
                    training_effort=training_effort.lower(),
                    extension_registry=extension_registry,
                )
            except Exception as exc:
                message = str(exc)
                if "contains NaN" in message:
                    message = (
                        "The selected target column still contains invalid or missing values after cleaning. "
                        "Try re-uploading the file, checking the target column for blanks like 'unknown' or "
                        "'null', or choosing a different target."
                    )
                st.error(f"Analysis failed: {message}")
                st.stop()

        render_decision_summary(result)
        render_ai_assistant_panel(result.get("assistant_extensions"))

        if result["notes"]:
            st.info("\n".join(result["notes"][:12]))

        if result["mode"] == "prediction":
            render_prediction_metric_cards(result)
            st.caption(
                f"Detected target style: {result['target_style']['label']} "
                f"with {result['target_style']['unique_count']} unique values."
            )

            render_quality_summary(result)

            st.subheader("Leaderboard")
            leaderboard = pd.DataFrame(result["results"]).T.sort_values(
                result["metric_name"],
                ascending=result["metric_name"] in {"rmse", "mae"},
            )
            st.dataframe(leaderboard, use_container_width=True)

            if not result["feature_importance"].empty:
                st.subheader("Top Drivers")
                st.bar_chart(result["feature_importance"].set_index("feature"))
                st.dataframe(result["feature_importance"], use_container_width=True)

            render_relevant_prediction_charts(result)

            st.subheader("Prediction Preview")
            st.dataframe(result["prediction_preview"], use_container_width=True)

            st.subheader("Additional Dataset Insights")
            render_additional_insights(result)

            if prediction_file is not None:
                try:
                    prediction_upload = load_uploaded_table_payload(
                        prediction_file.name,
                        prediction_file.getvalue(),
                    )
                    prediction_df = sanitize_dataframe(prediction_upload["dataframe"])
                    aligned = align_prediction_frame(prediction_df, result["feature_columns"])
                    scored = aligned.copy()
                    scored["prediction"] = result["best_model"].predict(aligned)
                    if result["problem_type"] == "regression":
                        target_series = df[result["selected_target"]]
                        if target_series.dropna().shape[0] and (
                            pd.api.types.is_numeric_dtype(target_series)
                            and (target_series.dropna().round() == target_series.dropna()).all()
                        ):
                            scored["rounded_prediction"] = scored["prediction"].round().astype(int)
                    st.subheader("Scored File Preview")
                    st.dataframe(scored.head(25), use_container_width=True)
                    st.download_button(
                        "Download Scored CSV",
                        data=dataframe_to_csv_bytes(scored),
                        file_name="scored_predictions.csv",
                        mime="text/csv",
                    )
                except Exception as exc:
                    message = str(exc)
                    if "missing required columns" in message.lower():
                        feature_preview = ", ".join(result["feature_columns"][:10])
                        message += (
                            " The current scoring flow expects the same modeling-ready columns used during "
                            f"training, such as: {feature_preview}."
                        )
                    st.warning(f"Could not score the second dataset: {message}")
        else:
            render_analysis_mode(result)

        report_markdown = build_report_markdown(result, training_file.name)
        st.subheader("Report")
        st.code(report_markdown, language="markdown")
        st.download_button(
            "Download Report",
            data=report_markdown.encode("utf-8"),
            file_name="analysis_report.md",
            mime="text/markdown",
        )

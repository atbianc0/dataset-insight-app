import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_io import dataframe_to_csv_bytes, read_uploaded_table
from src.pipeline import align_prediction_frame, run_analysis


st.set_page_config(page_title="Dataset Insight App", layout="wide")


@st.cache_data(show_spinner=False)
def load_uploaded_table(file_name, file_bytes):
    class UploadedFileShim:
        def __init__(self, name, content):
            self.name = name
            self._content = content

        def getvalue(self):
            return self._content

    return read_uploaded_table(UploadedFileShim(file_name, file_bytes))


def render_metric_cards(result):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Problem Type", result["problem_type"].title())
    col2.metric("Best Model", result["best_model_name"])
    col3.metric("Rows Used", f"{result['used_rows']:,}")
    col4.metric(result["metric_name"].upper(), f"{result['best_metrics'][result['metric_name']]:.3f}")


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


def render_relevant_charts(result):
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
        st.info("No high-signal figure was available for this dataset shape.")


def build_report_text(result, dataset_name, target_col):
    lines = [
        f"Dataset Insight Report: {dataset_name}",
        f"Target column: {target_col}",
        f"Problem type: {result['problem_type']}",
        f"Target style: {result['target_style']['label']}",
        f"Best model: {result['best_model_name']}",
        f"Model worth: {result['quality']['verdict']}",
        "",
        "Metrics",
    ]

    for name, value in result["best_metrics"].items():
        lines.append(f"{name}: {value:.4f}")

    lines.extend(["", "Preparation Notes"])
    lines.extend(result["notes"] or ["None"])
    return "\n".join(lines)


st.title("Dataset Insight App")
st.write(
    "A faster, more reliable dataset modeling app. Upload a table, choose a target, "
    "and it will train a strong default model without freezing on large files."
)

with st.sidebar:
    st.header("Settings")
    problem_type_mode = st.selectbox(
        "Prediction type",
        ["Auto Detect", "Classification", "Regression"],
    )
    test_size = st.slider("Holdout test size", 0.1, 0.35, 0.2, 0.05)
    drop_identifier_columns = st.checkbox("Drop identifier-like columns", value=True)

training_file = st.file_uploader("Upload a training dataset", type=["csv", "tsv", "txt"])
prediction_file = st.file_uploader(
    "Optional: upload a second dataset for scoring",
    type=["csv", "tsv", "txt"],
)

if training_file is not None:
    try:
        df = load_uploaded_table(training_file.name, training_file.getvalue())
    except Exception as exc:
        st.error(f"Could not read the training file: {exc}")
        st.stop()

    if df.empty:
        st.error("The uploaded dataset has no rows.")
        st.stop()

    st.subheader("Dataset Snapshot")
    st.dataframe(df.head(10))

    info1, info2, info3 = st.columns(3)
    info1.metric("Rows", f"{df.shape[0]:,}")
    info2.metric("Columns", f"{df.shape[1]:,}")
    info3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")

    target_col = st.selectbox("Choose the target column", df.columns)

    if st.button("Train Model", type="primary"):
        with st.spinner("Preparing data and training a reliable model..."):
            try:
                result = run_analysis(
                    df,
                    target_col,
                    problem_type_mode=problem_type_mode,
                    test_size=test_size,
                    drop_identifier_columns=drop_identifier_columns,
                )
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.stop()

        render_metric_cards(result)
        st.caption(
            f"Detected target style: {result['target_style']['label']} "
            f"with {result['target_style']['unique_count']} unique values."
        )

        if result["notes"]:
            st.info("\n".join(result["notes"]))

        render_quality_summary(result)

        st.subheader("Leaderboard")
        leaderboard = pd.DataFrame(result["results"]).T.sort_values(
            result["metric_name"],
            ascending=result["metric_name"] in {"rmse", "mae"},
        )
        st.dataframe(leaderboard)

        if not result["feature_importance"].empty:
            st.subheader("Top Drivers")
            st.bar_chart(result["feature_importance"].set_index("feature"))
            st.dataframe(result["feature_importance"])

        render_relevant_charts(result)

        st.subheader("Prediction Preview")
        st.dataframe(result["prediction_preview"])

        report_text = build_report_text(result, training_file.name, target_col)
        st.subheader("Report")
        st.text_area("Analysis report", report_text, height=220)

        if prediction_file is not None:
            try:
                prediction_df = load_uploaded_table(prediction_file.name, prediction_file.getvalue())
                aligned = align_prediction_frame(prediction_df, result["feature_columns"])
                scored = aligned.copy()
                scored["prediction"] = result["best_model"].predict(aligned)
                if result["problem_type"] == "regression":
                    target_series = df[target_col]
                    if target_series.dropna().shape[0] and (
                        pd.api.types.is_numeric_dtype(target_series) and
                        (target_series.dropna().round() == target_series.dropna()).all()
                    ):
                        scored["rounded_prediction"] = scored["prediction"].round().astype(int)
                st.subheader("Scored File Preview")
                st.dataframe(scored.head(25))
                st.download_button(
                    "Download Scored CSV",
                    data=dataframe_to_csv_bytes(scored),
                    file_name="scored_predictions.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.warning(f"Could not score the second dataset: {exc}")

        st.download_button(
            "Download Report",
            data=report_text.encode("utf-8"),
            file_name="analysis_report.txt",
            mime="text/plain",
        )

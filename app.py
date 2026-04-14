import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data_io import dataframe_to_csv_bytes, read_uploaded_table
from src.evaluate import evaluate_model
from src.explain import generate_report_text, generate_summary
from src.utils import align_prediction_frame, run_experiment


DEFAULT_OUTPUTS = [
    "Executive Summary",
    "Charts",
    "Model Leaderboard",
    "Predictions Table",
    "Feature Importance",
    "Data Quality Report",
    "Technical Metrics",
    "Recommendations",
]


def render_confusion_matrix(diagnostics):
    labels = diagnostics["confusion_labels"]
    matrix = diagnostics["confusion_matrix"]

    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, value, ha="center", va="center")

    fig.colorbar(image, ax=ax)
    st.pyplot(fig)


def render_regression_plot(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)


st.set_page_config(page_title="Dataset Insight App", layout="wide")

st.title("Dataset Insight App")
st.write(
    "Upload a delimited dataset, choose the kind of output you want, and generate "
    "user-friendly predictions, charts, and modeling conclusions."
)

with st.sidebar:
    st.header("Configuration")
    output_preferences = st.multiselect(
        "Output sections",
        DEFAULT_OUTPUTS,
        default=DEFAULT_OUTPUTS,
    )
    problem_type_mode = st.selectbox(
        "Prediction type",
        ["Auto Detect", "Classification", "Regression"],
    )
    ranking_metric_label = st.selectbox(
        "Best model metric",
        ["Auto", "F1 / RMSE", "Accuracy", "R²", "MAE"],
    )
    test_size = st.slider("Holdout test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    max_categories = st.slider(
        "Max categories before a text column is dropped",
        min_value=10,
        max_value=100,
        value=40,
        step=5,
    )
    drop_identifier_columns = st.checkbox("Drop identifier-like columns", value=True)


ranking_metric_map = {
    "Auto": None,
    "F1 / RMSE": None,
    "Accuracy": "accuracy",
    "R²": "r2",
    "MAE": "mae",
}


training_file = st.file_uploader("Upload a training dataset", type=["csv", "tsv", "txt"])
prediction_file = st.file_uploader(
    "Optional: upload a second dataset to score with the trained model",
    type=["csv", "tsv", "txt"],
)

if training_file is not None:
    try:
        df = read_uploaded_table(training_file)
    except Exception as exc:
        st.error(f"Could not read the uploaded training file: {exc}")
        st.stop()

    if df.empty:
        st.error("The uploaded training file has no rows.")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Rows", f"{df.shape[0]:,}")
    metric_2.metric("Columns", f"{df.shape[1]:,}")
    metric_3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")

    target_col = st.selectbox("Choose the target column", df.columns)
    run_button = st.button("Run Analysis", type="primary")

    if run_button:
        try:
            output = run_experiment(
                df,
                target_col,
                problem_type_mode=problem_type_mode,
                ranking_metric=ranking_metric_map[ranking_metric_label],
                test_size=test_size,
                max_categories=max_categories,
                drop_identifier_columns=drop_identifier_columns,
            )

            prediction_source_name = "holdout test split"
            prediction_frame = output["X_test"].copy()
            actual_series = output["y_test"].reset_index(drop=True)
            prediction_metrics = output["best_metrics"]

            if prediction_file is not None:
                prediction_df = read_uploaded_table(prediction_file)
                prediction_source_name = "uploaded scoring file"
                actual_series = None
                if target_col in prediction_df.columns:
                    actual_series = prediction_df[target_col].reset_index(drop=True)
                    prediction_df = prediction_df.drop(columns=[target_col])
                prediction_frame = align_prediction_frame(
                    prediction_df,
                    output["feature_columns"],
                )

            preds = output["best_model"].predict(prediction_frame)
            pred_df = prediction_frame.copy().reset_index(drop=True)
            if actual_series is not None:
                pred_df["actual"] = actual_series.values
            pred_df["prediction"] = preds

            if output["problem_type"] == "classification" and hasattr(output["best_model"], "predict_proba"):
                probability_frame = output["best_model"].predict_proba(prediction_frame)
                probability_columns = [
                    f"probability_{class_name}"
                    for class_name in output["best_model"].named_steps["model"].classes_
                ]
                proba_df = pd.DataFrame(probability_frame, columns=probability_columns)
                pred_df = pd.concat([pred_df, proba_df], axis=1)

            if actual_series is not None and prediction_file is not None:
                prediction_metrics = evaluate_model(output["problem_type"], actual_series, preds)

            st.subheader("Analysis Summary")
            st.write(
                generate_summary(
                    output["problem_type"],
                    output["best_model_name"],
                    prediction_metrics,
                    output["profile"]["notes"],
                )
            )

            if "Model Leaderboard" in output_preferences:
                st.subheader("Model Leaderboard")
                leaderboard = pd.DataFrame(output["results"]).T.sort_values(
                    "f1" if output["problem_type"] == "classification" else "rmse",
                    ascending=output["problem_type"] != "classification",
                )
                st.dataframe(leaderboard)

            if "Technical Metrics" in output_preferences:
                st.subheader("Selected Output Metrics")
                st.json(prediction_metrics)

            if "Data Quality Report" in output_preferences:
                st.subheader("Data Quality")
                missing_df = output["profile"]["missing_by_column"].reset_index()
                missing_df.columns = ["column", "missing_values"]
                st.dataframe(missing_df)
                if output["profile"]["notes"]:
                    st.info("\n".join(output["profile"]["notes"]))

            if "Charts" in output_preferences:
                st.subheader("Charts")
                if output["problem_type"] == "classification":
                    target_counts = df[target_col].value_counts(dropna=False)
                    st.bar_chart(target_counts)
                    render_confusion_matrix(output["diagnostics"])
                else:
                    st.line_chart(df[target_col].dropna().reset_index(drop=True))
                    render_regression_plot(output["y_test"], output["holdout_predictions"])

            if "Feature Importance" in output_preferences and not output["feature_importance"].empty:
                st.subheader("Top Drivers")
                st.bar_chart(output["feature_importance"].set_index("feature"))
                st.dataframe(output["feature_importance"])

            if "Predictions Table" in output_preferences:
                st.subheader(f"Predictions Preview ({prediction_source_name})")
                st.dataframe(pred_df.head(25))

            if "Recommendations" in output_preferences:
                st.subheader("Recommended Next Steps")
                recommendations = [
                    "Review dropped identifier or high-cardinality columns before production use.",
                    "Validate with a truly separate scoring dataset before trusting business conclusions.",
                    "Use the prediction download below to inspect individual row-level outputs.",
                ]
                for recommendation in recommendations:
                    st.write(f"- {recommendation}")

            report_text = generate_report_text(
                dataset_name=training_file.name,
                problem_type=output["problem_type"],
                best_model_name=output["best_model_name"],
                metrics=prediction_metrics,
                profile=output["profile"],
                output_preferences=output_preferences,
            )

            if "Executive Summary" in output_preferences:
                st.subheader("Executive Report")
                st.text_area("Report", report_text, height=320)

            st.download_button(
                "Download Predictions CSV",
                data=dataframe_to_csv_bytes(pred_df),
                file_name="predictions.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download Analysis Report",
                data=report_text.encode("utf-8"),
                file_name="analysis_report.txt",
                mime="text/plain",
            )
        except Exception as exc:
            st.error(f"Error while running analysis: {exc}")

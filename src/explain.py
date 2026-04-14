def generate_summary(problem_type, best_model_name, metrics, profile_notes=None):
    profile_notes = profile_notes or []

    if problem_type == "classification":
        summary = (
            f"The strongest model was {best_model_name}. "
            f"It achieved accuracy {metrics.get('accuracy', 0):.3f}, "
            f"precision {metrics.get('precision', 0):.3f}, "
            f"recall {metrics.get('recall', 0):.3f}, "
            f"and weighted F1 {metrics.get('f1', 0):.3f} on the holdout split."
        )
    else:
        summary = (
            f"The strongest model was {best_model_name}. "
            f"It produced RMSE {metrics.get('rmse', 0):.3f}, "
            f"MAE {metrics.get('mae', 0):.3f}, "
            f"and R^2 {metrics.get('r2', 0):.3f} on the holdout split."
        )

    if profile_notes:
        summary += " Data preparation notes: " + "; ".join(profile_notes[:3]) + "."

    return summary


def generate_report_text(
    dataset_name,
    problem_type,
    best_model_name,
    metrics,
    profile,
    output_preferences,
):
    lines = [
        f"Dataset Insight Report: {dataset_name}",
        "",
        f"Problem type: {problem_type}",
        f"Best model: {best_model_name}",
        "",
    ]

    if "Executive Summary" in output_preferences:
        lines.extend(
            [
                "Executive Summary",
                generate_summary(
                    problem_type,
                    best_model_name,
                    metrics,
                    profile.get("notes", []),
                ),
                "",
            ]
        )

    if "Data Quality Report" in output_preferences:
        lines.extend(
            [
                "Data Quality",
                f"Rows: {profile['row_count']}",
                f"Columns: {profile['column_count']}",
                f"Missing cells: {profile['missing_cells']}",
                f"Dropped columns: {', '.join(profile['dropped_column_names']) or 'None'}",
                "",
            ]
        )

    if "Technical Metrics" in output_preferences:
        lines.append("Model Metrics")
        for metric_name, metric_value in metrics.items():
            lines.append(f"{metric_name}: {metric_value:.4f}")
        lines.append("")

    if "Recommendations" in output_preferences:
        lines.extend(
            [
                "Recommendations",
                "Validate results on a truly representative dataset before deployment.",
                "Review dropped columns to decide whether any should be engineered instead of excluded.",
                "Use prediction-file scoring when you have a separate evaluation dataset.",
                "",
            ]
        )

    return "\n".join(lines).strip()

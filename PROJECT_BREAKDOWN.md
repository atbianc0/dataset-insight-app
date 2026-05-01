# Project Breakdown

## Overview

This project is a Streamlit application for tabular dataset analysis with one core design constraint:

`Do not force machine learning when the dataset or target does not justify it.`

The app behaves like a lightweight dataset triage system. It tries to determine:

- whether the dataset is suitable for supervised prediction
- which columns are plausible targets
- whether multiple targets may be relevant
- whether the data is better suited for exploratory or descriptive analysis instead

That makes the project fundamentally different from a typical “upload CSV -> train model” app. Prediction is one possible output path, not the default truth.

## Technical Goals

From an implementation standpoint, the app is optimized for:

- simple and maintainable heuristics
- readable data preparation logic
- transparent user-facing recommendations
- practical fallback behavior
- low dependency complexity

It is not trying to be:

- a full AutoML platform
- a feature-store system
- a general NLP or text-modeling system
- a production-grade MLOps framework

## Active Architecture

The currently active execution path is centered in four files:

- [app.py](/Users/abinghambianco/Developer/dataset-insight-app/app.py)
- [src/pipeline.py](/Users/abinghambianco/Developer/dataset-insight-app/src/pipeline.py)
- [src/insights.py](/Users/abinghambianco/Developer/dataset-insight-app/src/insights.py)
- [src/data_io.py](/Users/abinghambianco/Developer/dataset-insight-app/src/data_io.py)
- [src/extensions.py](/Users/abinghambianco/Developer/dataset-insight-app/src/extensions.py)

There are older modules in `src/` that look like earlier development iterations:

- [src/train.py](/Users/abinghambianco/Developer/dataset-insight-app/src/train.py)
- [src/evaluate.py](/Users/abinghambianco/Developer/dataset-insight-app/src/evaluate.py)
- [src/preprocess.py](/Users/abinghambianco/Developer/dataset-insight-app/src/preprocess.py)
- [src/utils.py](/Users/abinghambianco/Developer/dataset-insight-app/src/utils.py)
- [src/explain.py](/Users/abinghambianco/Developer/dataset-insight-app/src/explain.py)

They are not part of the current primary UI flow.

## AI-Extension Seams

The pipeline now exposes explicit stage-level extension hooks so future AI-assisted behavior can be added without replacing the current heuristic flow.

Defined stages:

- `semantic_column_interpretation`
- `task_understanding`
- `feature_suggestion`
- `external_enrichment`
- `report_generation`

Current behavior:

- the default provider is still heuristic and local-only
- `recommend_dataset_workflow(...)` and `run_analysis(...)` attach an `assistant_extensions` bundle
- extension outputs are captured per stage, merged in provider order, and isolated from pipeline failures

First AI implementation:

- `src/ai_assistant.py` adds an optional OpenAI-backed dataset interpretation assistant
- it reads heuristic artifacts instead of raw rows whenever possible
- it provides advisory semantic notes, target/feature suggestions, and clearer summaries
- it does not choose the final workflow or override heuristic rejection logic

This gives the project a clean path for later additions like LLM-based schema interpretation, feature ideation, dataset-goal understanding, external lookups, and richer narrative report generation while keeping today’s lightweight behavior intact.

## Runtime Flow

The app’s live execution path is:

1. User uploads dataset.
2. File is parsed and sanitized.
3. User clicks `Start Preliminary Report`.
4. App builds a preliminary dataset recommendation.
5. User chooses a target or chooses no target.
6. User clicks `Analyze Dataset`.
7. App either:
   - returns a prediction-mode result
   - returns an analysis-mode result

## Front-End Flow In `app.py`

`app.py` is the orchestration layer and Streamlit UI.

### Page initialization

The app begins with:

- `st.set_page_config(page_title="Dataset Insight App", layout="wide")`

This configures a wide-layout dashboard style.

### Cached upload reader

`load_uploaded_table(file_name, file_bytes)`

Responsibilities:

- wraps raw upload bytes in a small shim object
- uses `@st.cache_data(show_spinner=False)`
- delegates actual parsing to `read_uploaded_table(...)`

Reason for this pattern:

- Streamlit’s uploader object is not always ideal for cached direct reuse
- caching based on `(file_name, file_bytes)` avoids repeated parsing on reruns

### Preliminary report gating

The flow control for “upload first, inspect next, choose target later” is implemented using:

- `build_upload_key(file_name, file_bytes)`
- `st.session_state["preliminary_report_upload_key"]`
- `st.session_state["preliminary_report_ready"]`

Behavior:

- each new upload resets `preliminary_report_ready` to `False`
- the user must click `Start Preliminary Report`
- until that button is clicked, the app stops before showing target selection

This is important because it changes the app from a reactive immediate-analysis UI into an intentional two-stage workflow.

### Preliminary report rendering functions

Important UI helper functions:

- `render_dataset_recommendation(workflow)`
- `render_target_candidate_tables(workflow)`
- `render_modeling_column_notes(workflow)`

These functions render outputs from `recommend_dataset_workflow(...)`.

### Final result rendering functions

- `render_decision_summary(result)`
- `render_prediction_metric_cards(result)`
- `render_quality_summary(result)`
- `render_relevant_prediction_charts(result)`
- `render_analysis_mode(result)`
- `render_additional_insights(result)`
- `build_report_text(result, dataset_name)`

The UI does not compute substantial logic on its own. It mostly renders structured payloads returned from the pipeline.

## File Parsing And IO In `src/data_io.py`

This module is intentionally minimal.

### `SUPPORTED_FILE_TYPES`

Currently:

- `csv`
- `tsv`
- `txt`

### `_detect_delimiter(sample_text)`

Uses `csv.Sniffer().sniff(...)` with delimiters:

- `,`
- `\t`
- `;`
- `|`

Fallback:

- `,`

### `read_uploaded_table(uploaded_file)`

Responsibilities:

- validate extension
- read raw bytes
- decode as UTF-8 with ignored decoding errors
- detect delimiter
- call `pd.read_csv(...)`

Important note:

- this assumes flat delimited text files only
- there is no Excel, Parquet, or schema-aware ingestion

### `dataframe_to_csv_bytes(df)`

Simple export helper:

- `df.to_csv(index=False).encode("utf-8")`

## Core Decision Engine In `src/pipeline.py`

This is the most important file in the project. It contains:

- sanitation
- target heuristics
- task recommendation logic
- feature preparation
- model training
- model evaluation
- workflow branching

## Constants In `src/pipeline.py`

Several constants shape behavior:

- `MAX_PREVIEW_ROWS = 25`
- `MAX_CHART_ROWS = 5000`
- `BENCHMARK_ROWS = 20000`
- `MAX_TRAIN_ROWS = 60000`
- `HIGH_CARDINALITY_LIMIT = 80`
- `MIN_PREDICTION_ROWS = 30`

### Semantic constants

- `COMMON_MISSING_TOKENS`
- `POSITIVE_TARGET_KEYWORDS`
- `NEGATIVE_TARGET_KEYWORDS`
- `GROUP_TOKEN_STOPWORDS`
- `TEMPORAL_TARGET_KEYWORDS`

These keyword sets are used to make the target recommendation system more human-readable and domain-reasonable.

## Data Sanitation Layer

### `normalize_missing_tokens(series)`

For string-like columns:

- strips whitespace
- lowercases when checking
- converts common placeholders like `unknown`, `null`, `n/a`, `-` into missing values

For non-string columns:

- returns the series unchanged

### `sanitize_dataframe(df)`

Responsibilities:

- copies the dataframe
- strips column names
- removes duplicated columns
- replaces `inf` and `-inf` with `NaN`
- normalizes missing tokens column by column
- converts object/string columns to numeric if at least 95% of non-null values parse as numbers

Why this matters:

- many CSV datasets come with pseudo-numeric text columns
- recommendation quality is much better if the app can treat `"123"` as numeric instead of categorical noise

### `sanitize_target_series(series)`

Target-focused variant of sanitation.

Differences from `sanitize_dataframe(...)`:

- explicitly strips string targets
- separately attempts numeric coercion on the target

This helps avoid situations where a target appears categorical only because it was stored as text.

### `filter_valid_target_rows(X, y, label, min_rows=20)`

Responsibilities:

- sanitize target
- drop rows where target remains invalid
- optionally filter `X` to match
- enforce a minimum remaining row count

Used in multiple places:

- initial target prep
- before train/test split
- after split

This repeated validation is important because the app intentionally prefers stopping over silently modeling broken target data.

## Type And Shape Heuristics

### `is_integer_like(series, tolerance=1e-9)`

Checks whether a numeric-like series is effectively integer-valued.

Used for:

- count-style regression detection
- small-integer classification detection
- identifier checks

### `numeric_conversion_ratio(series)`

Measures the fraction of non-null values that can be parsed as numeric.

Used in:

- auto-conversion logic
- target-type detection

### `detect_problem_type(y)`

Heuristic classifier for:

- `classification`
- `regression`

Rules:

- object/category/bool -> classification unless it is almost entirely numeric text
- small integer targets with low unique ratios may be treated as classification
- otherwise regression

This is not a full statistical test. It is a pragmatic guesser optimized for common tabular data cases.

### `summarize_target_style(y, problem_type)`

Returns a lightweight target summary object:

- label
- unique count

Possible labels include:

- `categorical classification`
- `count-style regression`
- `numeric regression`

## Column Classification Heuristics

### `_is_datetime_candidate(series)`

Returns `True` if:

- dtype is datetime already
- or string samples parse as datetime at a high enough rate

Important restriction:

- numeric columns are not treated as datetimes here

This avoids accidental date detection on generic integer columns.

### `_is_identifier_like(series, name)`

Heuristic ID detection using:

- name contains `id`
- very high uniqueness ratio
- mostly step-one numeric sequences
- large unique string spaces

This is central to preventing fake targets and weak features.

### `_is_text_heavy_target(series)`

Classifies a series as long-form text if:

- average word count is high
- or average string length is high

This explicitly pushes the app away from pretending to model raw long text targets.

### `_tokenize_column_name(name)`

Splits column names for:

- target keyword detection
- multi-target grouping
- temporal keyword detection

Works across:

- snake_case
- camelCase
- mixed punctuation

## Target Recommendation System

The target recommendation system is the key technical feature of the project.

It has two levels:

- per-column evaluation
- whole-dataset synthesis

## `evaluate_target_candidate(df, column, drop_identifier_columns=True)`

This function is the single-column evaluator.

### Inputs

- full sanitized dataframe
- candidate column name
- identifier-dropping preference

### High-level stages

1. Extract and sanitize the candidate target.
2. Compute target statistics.
3. Compute name-based signals.
4. Compute structural blockers.
5. Estimate target type.
6. Optionally run deeper feasibility checks.
7. Return a structured recommendation object.

### Statistics computed

- `usable_rows`
- `missing_ratio`
- `unique_count`
- `unique_ratio`

### Name-based signals

The function tokenizes the column name and checks membership in:

- `POSITIVE_TARGET_KEYWORDS`
- `NEGATIVE_TARGET_KEYWORDS`
- `TEMPORAL_TARGET_KEYWORDS`

Examples:

- `churn`, `label`, `price`, `status`, `score` increase plausibility
- `name`, `description`, `notes`, `title` reduce plausibility
- `date`, `year`, `timestamp` may directly block target use

### Structural blockers

Examples of hard blockers:

- no usable values
- fewer than `MIN_PREDICTION_ROWS`
- single unique value
- identifier-like target
- time-based target
- text-heavy target
- too many classes
- nearly all unique non-numeric values

### Soft cautions

Examples:

- moderate missingness
- few usable rows
- weak feature set after preparation
- moderately fragmented class structure

### Problem-type-specific logic

For classification:

- binary labels score highly
- medium-cardinality categories score reasonably
- too many classes become a blocker
- tiny classes become a blocker

For regression:

- numeric/coercible targets score positively
- continuous-looking outcomes score positively
- very low distinct-value counts trigger caution

### Deep feasibility pass

If no early blockers exist, the function delegates to:

- `assess_target_for_prediction(...)`

This adds:

- more target viability details
- usable feature count after prep
- additional caution or blocker messages

### Return shape

It returns a dictionary with fields including:

- `column`
- `status`
- `score`
- `problem_type`
- `target_shape`
- `recommended_use`
- `summary`
- `pros`
- `cautions`
- `blockers`
- `usable_rows`
- `missing_pct`
- `unique_count`
- `unique_ratio`
- `usable_feature_count`
- `suggested_feature_subset`
- `rejected_feature_columns`
- `positive_name_signals`
- `negative_name_signals`

This object is what the UI uses to explain the app’s reasoning.

## `assess_target_for_prediction(df, target_col, drop_identifier_columns=True)`

This is the second-stage validator used after the user has chosen a target.

Where `evaluate_target_candidate(...)` is recommendation-oriented, `assess_target_for_prediction(...)` is execution-gating oriented.

### Responsibilities

- validate usable target rows
- validate missingness
- reject identifier-like targets
- reject long-form text targets
- check classification fragmentation
- check regression realism
- call `prepare_training_frame(...)` to ensure usable features remain

### Output shape

Returns:

- `selected_target`
- `problem_type`
- `mode_recommendation`
- `summary`
- `reasons_for_prediction`
- `reasons_against_prediction`
- `blockers`
- `usable_rows`
- `missing_ratio`
- `unique_count`
- `unique_ratio`
- `usable_feature_count`

This object is later embedded in both prediction-mode and analysis-mode result payloads.

## Whole-Dataset Recommendation

## `recommend_dataset_workflow(df, drop_identifier_columns=True, top_n=8)`

This is the dataset-level synthesizer.

### Internal flow

1. Sanitize dataframe.
2. Evaluate every column via `evaluate_target_candidate(...)`.
3. Sort candidates by:
   - status tier
   - score
   - column name
4. Partition into:
   - recommended candidates
   - possible candidates
   - rejected candidates
5. Detect multi-target groups.
6. Build feature-subset notes.
7. Build insight-analysis recommendation list.
8. Produce a unified recommendation object.

### Recommended workflow logic

Important detail:

- the app does not recommend prediction merely because there is a “possible” target
- it requires at least one `recommended` target to prefer prediction at the dataset level

This prevents weak metadata datasets from being mislabeled as prediction-ready.

### Multi-target grouping

`_detect_multi_target_groups(candidates)` groups plausible targets by:

- shared prefix
- shared suffix
- shared meaningful tokens

It produces group objects with:

- `group_label`
- `columns`
- `problem_type`
- `reason`
- `average_score`

### Feature subset summary

`_build_feature_subset_summary(df)` classifies columns into:

- `likely_useful`
- `risky`
- `avoid`

This supports the UI’s “modeling column notes” section.

### Output shape

Returns a dictionary including:

- `summary`
- `recommended_workflow`
- `recommended_task_type`
- `recommended_primary_target`
- `recommended_target_columns`
- `candidate_targets`
- `rejected_target_candidates`
- `multi_target_candidates`
- `task_recommendations`
- `feature_subset_summary`
- `insight_analysis`
- `candidate_lookup`
- `clear_primary_target`

This is one of the app’s most important interface contracts.

## Feature Preparation Pipeline

## `prepare_training_frame(df, target_col, drop_identifier_columns=True)`

This is the main feature engineering and cleanup function for supervised mode.

### Responsibilities

- sanitize full dataframe
- ensure target exists
- sanitize target
- remove rows with invalid targets
- split into `X` and `y`
- transform columns
- produce lists of numeric and categorical columns

### Column-level transformations

#### All-missing columns

- dropped

#### Constant columns

- dropped

#### Identifier-like columns

- optionally dropped

#### Datetime-like columns

- expanded into:
  - `year`
  - `month`
  - `day`
  - `dayofweek`

#### Multi-value text columns

If comma-separated values appear frequently:

- derive first item
- derive item count
- optionally use first-item frequency encoding

#### Numeric-text columns

If values look like patterns such as `"12 kg"`:

- extract numeric part
- optionally extract unit token

#### Long text columns

- convert to word-count feature

#### Very high-cardinality categorical columns

- frequency encode

### Output shape

Returns:

- `X`
- `y`
- `notes`
- `dropped_columns`
- `numeric_cols`
- `categorical_cols`

This function is a major bridge between recommendation and actual training.

## Sampling, Preprocessing, And Model Training

## `sample_training_data(X, y, problem_type, max_rows=MAX_TRAIN_ROWS, random_state=42)`

Used to cap training volume.

Behavior:

- if dataset is below cap, return unchanged
- if classification, preserve class representation as best as possible
- if regression, sample rows uniformly

This is mainly a practical latency and stability measure.

## `build_preprocessor(numeric_cols, categorical_cols, categorical_strategy="ordinal")`

Builds a `ColumnTransformer`.

### Numeric path

- `SimpleImputer(strategy="constant", fill_value=0.0)`

### Categorical path

- `SimpleImputer(strategy="constant", fill_value="__missing__")`
- `OneHotEncoder` or `OrdinalEncoder`

Choice depends on model family:

- linear models tend to use one-hot
- tree models generally use ordinal encoding in this implementation

## Candidate models

### `get_candidate_models(problem_type, target_style_label=None, effort="standard", imbalance_ratio=None)`

Returns a dictionary of model specs:

- estimator instance
- categorical encoding strategy

Classification models include:

- `HistGradientBoostingClassifier`
- `RandomForestClassifier`
- optionally `Balanced Random Forest`
- optionally `ExtraTreesClassifier`
- optionally `LogisticRegression`

Regression models include:

- `HistGradientBoostingRegressor`
- `RandomForestRegressor`
- `Ridge`
- optionally `ExtraTreesRegressor`
- optionally Poisson gradient boosting for count-style regression

### Notes on model selection philosophy

- small set of defaults
- no extensive hyperparameter search
- practical runtime over exhaustive search
- enough variety to avoid a single-model app

## Training execution

## `train_best_model(...)`

Responsibilities:

- revalidate target after reset
- perform train/test split
- stratify classification when possible
- fit all candidate models
- compute evaluation metrics
- optionally adjust binary decision threshold for imbalanced classification
- pick best model using ranking metric

### Classification threshold tuning

If classification is binary and highly imbalanced:

- compute probabilities
- identify minority class
- compute precision-recall curve
- choose threshold maximizing F1 on precision-recall curve

This is a notable piece of sophistication in an otherwise lightweight pipeline.

### Training output shape

Returns:

- `results`
- `best_model_name`
- `best_model`
- `best_metrics`
- `metric_name`
- `X_test`
- `y_test`
- `preds`
- `best_probabilities`
- `imbalance_ratio`
- `dropped_target_rows`

## Evaluation Logic

## `evaluate_predictions(problem_type, y_true, y_pred)`

Classification metrics:

- accuracy
- weighted precision
- weighted recall
- weighted F1

Regression metrics:

- RMSE
- MAE
- R²

## `add_probability_metrics(problem_type, metrics, y_true, probabilities)`

For binary classification when probabilities exist:

- ROC AUC
- average precision

## `build_baseline_metrics(problem_type, y_true)`

Baseline strategies:

- classification: always predict majority class
- regression: always predict mean target value

This is important because the app does not want to surface a model that merely looks good in isolation.

## `assess_model_quality(problem_type, best_metrics, baseline_metrics)`

Assigns one of several verdicts:

- `strong`
- `useful`
- `limited`
- `weak`

Classification quality considers:

- F1 gain
- accuracy gain
- average precision gain when available

Regression quality considers:

- RMSE improvement vs baseline
- relative RMSE gain
- R² magnitude

This quality gate is the final protection against low-value predictions.

## Prediction Chart Support

## `build_chart_context(problem_type, X_sample, y_sample, holdout_actual, holdout_pred, feature_importance)`

Produces a render-friendly chart payload for the UI.

Includes:

- numeric and categorical columns
- top numeric and categorical relationships
- holdout actual values
- holdout predictions
- optional confusion matrix
- relationship frames for charts

This separates chart computation from rendering logic in `app.py`.

## Feature Importance

## `build_feature_importance(best_model, feature_names)`

Uses:

- `feature_importances_` for tree models

It does not currently support linear-model coefficient rendering unless feature importances are present through the model object being used.

The returned frame is limited to top rows for UI readability.

## Final Workflow Branching

## `_build_analysis_result(...)`

Helper to construct the analysis-mode payload.

This wraps:

- decision summary
- target assessment details
- predictive attempt summary when applicable
- insight analysis payload
- metadata such as row and missing counts

## `run_analysis(...)`

This is the central top-level function in the pipeline.

### Execution order

1. Sanitize dataframe.
2. Build dataset-level recommendation.
3. Reuse `insight_analysis` from that recommendation.
4. If no target selected:
   - return analysis-mode result immediately.
5. If target is selected:
   - fetch or compute target assessment.
6. If target assessment says prediction is unsafe:
   - return analysis-mode result.
7. Prepare supervised training frame.
8. Detect problem type.
9. Sample training data if needed.
10. Train models.
11. Build feature importance.
12. Compute baseline.
13. Assess quality.
14. If quality is weak:
   - return analysis-mode result with predictive-attempt details.
15. Otherwise:
   - return prediction-mode result.

### Prediction-mode payload shape

Returned dictionary includes:

- `mode`
- `dataset_recommendation`
- `decision`
- `target_assessment`
- `analysis_recommendations`
- `insight_analysis`
- `problem_type`
- `target_style`
- `results`
- `best_model_name`
- `best_model`
- `best_metrics`
- `baseline_metrics`
- `quality`
- `metric_name`
- `feature_columns`
- `feature_importance`
- `chart_context`
- `prediction_preview`
- `notes`
- `training_effort`
- `used_rows`
- `original_rows`
- `original_columns`
- `missing_cells`
- `target_series`
- `holdout_actual`
- `holdout_pred`
- `selected_target`

### Analysis-mode payload shape

Returned dictionary includes:

- `mode`
- `dataset_recommendation`
- `decision`
- `target_assessment`
- `predictive_attempt`
- `insight_analysis`
- `analysis_recommendations`
- `notes`
- `used_rows`
- `original_rows`
- `original_columns`
- `missing_cells`
- `selected_target`

The UI depends heavily on this payload contract.

## Insight Engine In `src/insights.py`

This module provides the app’s non-predictive analysis layer.

## Constants

- `MAX_SAMPLE_VALUES = 3`
- `MAX_CORRELATIONS = 10`
- `MAX_GROUP_ROWS = 12`
- `MAX_ANOMALIES = 10`

These keep outputs short enough for dashboards and reports.

## Column inspection

## `build_column_inspection(df)`

For each column it computes:

- dtype
- role hint
- non-null count
- missing percentage
- unique count
- coverage percentage
- sample values
- human-readable recommendation

Role hints include:

- identifier-like
- date/time
- numeric measure
- long text
- categorical label
- high-cardinality category

This is used directly in the UI preliminary report.

## Analysis path recommendation

## `recommend_analysis_paths(df)`

Looks at:

- presence of numeric columns
- presence of datetime-like columns
- presence of low-cardinality categories
- dataset row count

Can recommend:

- exploratory data analysis
- descriptive statistics
- correlation analysis
- trend analysis
- grouping analysis
- anomaly detection
- clustering or segmentation

These are broad task recommendations, not model outputs.

## Numeric and categorical summaries

### `_build_numeric_summary(df)`

Uses `describe()` on numeric and boolean columns after excluding identifier-like columns.

Also adds:

- `missing_pct`

### `_build_categorical_summary(df)`

For non-numeric, non-datetime columns it computes:

- unique values
- missing percentage
- top 3 values with counts

## Correlation analysis

## `_build_correlations(df)`

Behavior:

- select numeric columns
- exclude identifier-like columns
- compute pairwise correlations
- flatten upper-triangle pairs
- sort by absolute correlation magnitude
- keep top rows

This is used both in insight mode and as extra context in prediction mode.

## Grouping analysis

## `_build_group_summary(df)`

Purpose:

- find a categorical column and numeric metric pair where group means differ meaningfully

Method:

- iterate low-cardinality categorical columns
- pair against numeric columns
- compute group means and counts
- measure normalized between-group spread
- keep strongest candidate

This is a very lightweight grouping-signal detector.

## Trend analysis

## `_build_trend_summary(df)`

Purpose:

- detect if the dataset supports time-based summarization

Method:

- identify datetime-like columns
- choose a preferred date column
- aggregate by period
- choose row count or mean numeric metric over time
- compute first-to-last change

Important limitations:

- only one date column is chosen
- only one primary metric is shown
- no seasonality or forecasting logic is present

## Anomaly analysis

## `_build_anomaly_summary(df)`

Method:

- select numeric columns
- exclude identifiers
- compute standardized absolute z-scores
- average across available columns
- surface top anomaly rows

This is intentionally simple and easy to explain.

## Headline generation

## `_build_headline_insights(...)`

Produces a small list of user-facing conclusions such as:

- row/column summary
- most incomplete column
- strongest correlation
- grouping signal
- trend summary
- top anomaly score

This is the closest thing the app currently has to a narrative summarizer.

## `run_insight_analysis(df, target_col=None)`

This is the main public entry point for the module.

It builds:

- dataset overview
- column inspection
- analysis recommendations
- numeric summary
- categorical summary
- correlations
- group summary
- trend summary
- anomaly summary
- headlines

Return shape:

- `overview`
- `column_inspection`
- `analysis_recommendations`
- `numeric_summary`
- `categorical_summary`
- `correlations`
- `group_summary`
- `trend_summary`
- `anomaly_summary`
- `headlines`
- `selected_target`

## UI Rendering Contract

The Streamlit layer assumes structured dictionaries and dataframes rather than opaque text blobs.

That has several benefits:

- easier debugging
- easier extension
- easier unit testing in the future
- clearer separation between logic and presentation

The main rendering contracts are:

- dataset recommendation object
- analysis result object
- prediction result object
- chart context object
- insight analysis object

If these contracts change, `app.py` will need coordinated updates.

## Reporting Layer

## `build_report_text(result, dataset_name)`

This generates a downloadable plain-text report.

It includes:

- dataset recommendation summary
- suggested primary target
- alternate viable targets
- workflow decision
- multi-target options if present
- key conclusions
- predictive metrics or analysis paths
- preparation notes

This is currently a plain text artifact rather than a richer HTML or PDF report.

## Second-Dataset Scoring Path

Prediction mode optionally supports scoring a second uploaded dataset.

Flow:

1. User uploads prediction file.
2. `load_uploaded_table(...)` parses it.
3. `align_prediction_frame(...)` checks that required feature columns exist.
4. Best model predicts on aligned frame.
5. Output preview and downloadable scored CSV are shown.

Limitations:

- this only works well when the second dataset matches the training schema closely
- feature engineering transformations are not persisted as an explicit reusable schema object

## Error-Handling Strategy

The project uses exceptions for hard validation failures and user-friendly UI messages at the Streamlit layer.

Examples of hard failures:

- unsupported upload type
- empty dataset
- no valid target rows
- no usable feature columns
- incompatible scoring dataset

The UI layer catches exceptions and converts them into:

- `st.error(...)`
- `st.warning(...)`

This is acceptable for a small app, though a larger codebase would probably want typed custom exceptions.

## Heuristic Philosophy

The project consistently favors:

- readable rules over opaque learned meta-modeling
- honest fallback behavior over overconfident outputs
- shallow but robust analysis over ambitious brittle automation

This is visible everywhere:

- target recommendation uses explicit heuristics
- model families are simple and interpretable
- insight mode is rule-based
- weak predictive results are rejected

## Performance Characteristics

The app is designed for moderate tabular datasets, not huge distributed workloads.

Performance-related design choices include:

- Streamlit data caching for uploads
- row sampling with `MAX_TRAIN_ROWS`
- limited model set
- narrow chart payloads
- truncated tables and summaries

Potential bottlenecks:

- evaluating every column as a target can get expensive on very wide datasets
- repeated dataframe copies during sanitation and preparation
- Python-level loops in heuristic routines

## Current Technical Debt

### 1. Legacy modules remain in repo

There are old modules whose presence can confuse onboarding.

### 2. No formal test suite

The recommendation logic is now complex enough that tests would add significant value.

### 3. Some duplicated heuristics

Datetime, identifier, and text-detection logic exists in both `src/pipeline.py` and `src/insights.py`.

This is manageable now, but centralizing shared heuristics would reduce drift risk.

### 4. Multi-target is advisory only

Recommendation exists, execution does not.

### 5. Report generation is still basic

The text report is useful, but it is not yet a full technical or business analysis document.

## Suggested Refactor Directions

If the project keeps growing, a reasonable next architecture could be:

### Option 1: Keep current structure, extract shared heuristics

Create a shared module for:

- identifier detection
- datetime detection
- column tokenization
- text heaviness detection

This is the smallest useful refactor.

### Option 2: Split recommendation contracts into typed dataclasses

Possible typed objects:

- `TargetCandidate`
- `DatasetRecommendation`
- `PredictionResult`
- `AnalysisResult`

Benefits:

- stronger internal contracts
- easier tests
- less key-name fragility

### Option 3: Add dedicated tests before structural refactors

Probably the safest near-term step.

Recommended test areas:

- target rejection logic
- dataset-level workflow recommendation
- weak-model fallback behavior
- multi-target grouping
- scoring-file alignment

## Practical Debugging Guide

If a future contributor wants to debug behavior, the most useful checkpoints are:

### Why did the app reject a target?

Inspect:

- `evaluate_target_candidate(...)`
- `assess_target_for_prediction(...)`
- UI table from `render_target_candidate_tables(...)`

### Why did the app choose insights mode?

Inspect:

- `run_analysis(...)`
- `result["decision"]`
- `result["target_assessment"]`
- `result["predictive_attempt"]`

### Why did a scoring file fail?

Inspect:

- `align_prediction_frame(...)`
- `result["feature_columns"]`

### Why is a dataset classified as regression vs classification?

Inspect:

- `detect_problem_type(...)`
- `summarize_target_style(...)`

## Example Mental Trace Of A Full Run

For a dataset with columns like:

- `customer_id`
- `tenure`
- `monthly_charges`
- `contract_type`
- `churn`

The system likely does this:

1. Sanitize dataframe.
2. Mark `customer_id` as identifier-like.
3. Evaluate `churn` as binary classification target.
4. Recommend dataset workflow = prediction.
5. User selects `churn`.
6. `prepare_training_frame(...)` drops `customer_id`.
7. Train classification models.
8. Compare to majority-class baseline.
9. If quality is good, return prediction-mode result.
10. Also surface extra correlations and grouping summaries from `insight_analysis`.

For a dataset like entertainment metadata:

- `show_id`
- `title`
- `cast`
- `country`
- `date_added`
- `release_year`

The system likely does this:

1. Sanitize dataframe.
2. Reject `show_id` as identifier-like.
3. Reject `title` and `cast` as text-heavy/descriptive.
4. Reject `release_year` as temporal-style target.
5. Find no strong recommended target.
6. Recommend insights workflow.
7. Surface trend analysis, category summaries, and descriptive outputs.

## Bottom-Line Technical Summary

The project is a rule-driven tabular-analysis app with three main technical pillars:

1. A dataset and target recommendation engine that tries to avoid bad modeling choices.
2. A lightweight but practical supervised learning pipeline for viable targets.
3. A fallback exploratory-analysis engine that keeps the app useful even when machine learning is not appropriate.

If you want a one-line technical mental model:

`This codebase is a heuristic dataset triage layer wrapped around a compact scikit-learn tabular modeling pipeline and a rule-based insight engine.`

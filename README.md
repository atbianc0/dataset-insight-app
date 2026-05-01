# Dataset Insight App

Dataset Insight App is a Streamlit app for practical tabular dataset triage.

Its core idea is simple:

`Prediction is one possible path, not the default truth.`

Instead of forcing every uploaded file into a machine learning workflow, the app inspects the dataset first and decides whether:

- supervised prediction actually makes sense
- a selected target is plausible
- multiple targets may be relevant
- the dataset is better suited for descriptive or exploratory analysis instead

If prediction looks justified, the app trains a small set of models and checks whether the result is genuinely useful. If prediction looks weak, unrealistic, or misleading, the app falls back to an insights-first path.

## Project Philosophy

This project is intentionally **not** trying to be:

- a full AutoML platform
- a “train a model no matter what” app
- a generic NLP pipeline for long text fields
- a heavy abstraction layer over every data science step

The design goal is trust and practicality:

- prefer readable heuristics over opaque automation
- explain why the app recommends prediction, caution, or rejection
- surface strong no-target analysis paths when ML is not the right answer
- keep the workflow lightweight and maintainable

## What The App Does

For an uploaded dataset, the app can:

- read CSV, TSV, TXT, XLSX, and XLS files
- show ingestion warnings and a schema preview
- sanitize common missing-value tokens
- inspect column roles such as identifier-like, numeric, date/time, categorical, and long text
- rank target candidates as recommended, possible, or rejected
- flag related target families that may support multi-target work
- explain the best no-target analysis path
- highlight likely useful, risky, and avoid-for-modeling columns
- run prediction mode only when the target and dataset support it
- fall back to insights mode when prediction is weak or not appropriate
- generate a markdown report summarizing the workflow decision and findings

## AI-Ready Extension Architecture

The current implementation stays heuristic-first, but the pipeline is now organized so future AI-assisted components can plug into stable stages instead of rewriting the main flow.

The active extension stages are:

- semantic column interpretation
- task and dataset understanding
- feature suggestion
- external enrichment
- report generation

Today, those stages are backed by the built-in heuristic provider. Future providers can be added through `src/extensions.py` and passed into `recommend_dataset_workflow(...)` or `run_analysis(...)`.

The first practical AI feature is an optional OpenAI-backed dataset interpretation assistant in `src/ai_assistant.py`.
It plugs into the existing extension stages and focuses on:

- semantic interpretation of columns
- plain-language explanation of whether prediction is sensible
- advisory target suggestions
- advisory feature suggestions
- clearer report summaries

It does not replace heuristic decisions, target rejection rules, or model-quality gating.
To enable it locally, install dependencies from `requirements.txt`, set `OPENAI_API_KEY`, and turn on `Enable AI assistant layer` in the sidebar.

That means later additions such as:

- LLM-based semantic column interpretation
- AI feature suggestions for a selected target
- dataset/task understanding from natural-language goals
- external enrichment lookups
- natural-language report generation

can be layered onto the existing pipeline contract rather than replacing the pipeline itself.

## Active Files

The current main app flow is centered on:

- [app.py](/Users/abinghambianco/Developer/dataset-insight-app/app.py)
- [src/pipeline.py](/Users/abinghambianco/Developer/dataset-insight-app/src/pipeline.py)
- [src/insights.py](/Users/abinghambianco/Developer/dataset-insight-app/src/insights.py)
- [src/data_io.py](/Users/abinghambianco/Developer/dataset-insight-app/src/data_io.py)
- [src/heuristics.py](/Users/abinghambianco/Developer/dataset-insight-app/src/heuristics.py)

There are also older modules in `src/` from earlier iterations:

- [src/train.py](/Users/abinghambianco/Developer/dataset-insight-app/src/train.py)
- [src/evaluate.py](/Users/abinghambianco/Developer/dataset-insight-app/src/evaluate.py)
- [src/preprocess.py](/Users/abinghambianco/Developer/dataset-insight-app/src/preprocess.py)
- [src/utils.py](/Users/abinghambianco/Developer/dataset-insight-app/src/utils.py)
- [src/explain.py](/Users/abinghambianco/Developer/dataset-insight-app/src/explain.py)

Those files are not part of the primary Streamlit flow today and should be treated as legacy unless they are deliberately reintegrated.

## Workflow

### 1. Upload and inspection

After upload, the app shows:

- ingestion details and warnings
- a schema preview
- a dataset snapshot
- column inspection
- ranked target candidates
- rejected targets and reasons
- dataset-level workflow recommendation
- suggested no-target analysis paths

### 2. Decision stage

When the user chooses a target, the app explains:

- whether the target is recommended, possible, or rejected
- why the target fits or does not fit prediction
- how many usable rows remain
- what the likely modeling inputs are
- which columns are typically excluded

### 3. Prediction mode

If the dataset is prediction-ready, the app:

- prepares features
- trains a small set of default models
- compares holdout performance against a simple baseline
- keeps prediction mode only when the result is useful enough to trust

### 4. Insights mode

If prediction is not justified, the app emphasizes:

- headline summaries
- descriptive statistics
- correlations
- grouping insights
- anomaly review
- trend summaries when date-like columns exist

## Setup

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Testing

The project now includes a lightweight pytest suite around decision logic and ingestion.

Run tests with:

```bash
python -m pytest -q
```

Current test coverage is focused on:

- target rejection logic
- dataset-level workflow recommendation
- weak-model fallback behavior
- multi-target grouping
- scoring-file alignment checks
- ingestion behavior for text and Excel uploads
- extension hook integration and failure isolation

## Sample Data

Example files are available in [sample_data](/Users/abinghambianco/Developer/dataset-insight-app/sample_data).

## Known Limitations

- The workflow still uses heuristics, so there will always be edge cases where human judgment should override the default recommendation.
- Scoring a second file works best when its schema matches the modeling-ready columns expected by the trained model.
- Insight mode is intentionally lightweight and readable; it is not meant to replace deeper domain-specific analysis.
- Multi-target suggestions are surfaced, but the current UI still trains one selected target at a time.

## Status

This is a practical learning project that became genuinely useful. AI tools were part of the development process, but the project direction is intentionally opinionated:

- do not force ML
- be transparent about uncertainty
- prefer useful analysis over automatic modeling

# Dataset Insight App

A Streamlit app that helps users upload tabular data, train predictive models, inspect charts, review data quality, and download predictions plus a plain-language analysis report.

## Features
- Upload CSV, TSV, or TXT tabular datasets
- Choose a target column and optionally override classification vs regression
- Drop identifier-like columns and expand datetime columns automatically
- Train multiple machine learning models and rank them by the selected metric
- Generate charts, feature-importance views, prediction previews, and evaluation metrics
- Score a second dataset with the trained model
- Download prediction output and a text report

## Tech Stack
- Python
- Pandas
- scikit-learn
- Streamlit
- Matplotlib

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.profiling import build_dataset_profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "sample_data" / "fixtures.json"


def test_fixture_manifest_locks_hash_source_license_schema_and_expected_facts():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["hash_algorithm"] == "sha256"
    assert len(manifest["fixtures"]) == 3
    for fixture in manifest["fixtures"]:
        path = PROJECT_ROOT / fixture["path"]
        content = path.read_bytes()
        frame = pd.read_csv(path)

        assert hashlib.sha256(content).hexdigest() == fixture["sha256"]
        assert len(content) == fixture["byte_size"]
        assert fixture["source"]["url"].startswith("https://www.kaggle.com/datasets/")
        assert fixture["license"]["spdx"] in {"CC0-1.0", "GPL-2.0-only"}
        license_text = (PROJECT_ROOT / fixture["license"]["text_path"]).read_text(
            encoding="utf-8"
        )
        expected_license_heading = {
            "CC0-1.0": "CC0 1.0 Universal",
            "GPL-2.0-only": "GNU GENERAL PUBLIC LICENSE",
        }[fixture["license"]["spdx"]]
        assert expected_license_heading in license_text
        assert list(frame.columns) == fixture["schema"]["columns"]
        assert frame.shape == (
            fixture["expected_facts"]["row_count"],
            fixture["schema"]["column_count"],
        )


def test_fixture_profiles_match_sanitized_acceptance_counts_and_auto_targets():
    netflix = build_dataset_profile(pd.read_csv(PROJECT_ROOT / "sample_data/netflix_titles.csv"))
    training = build_dataset_profile(
        pd.read_csv(PROJECT_ROOT / "sample_data/customer_churn_dataset-training-master.csv")
    )
    testing = build_dataset_profile(
        pd.read_csv(PROJECT_ROOT / "sample_data/customer_churn_dataset-testing-master.csv")
    )

    assert netflix.exact_overview == {
        "rows": 8_807,
        "columns": 12,
        "cells": 105_684,
        "missing_cells": 4_308,
        "duplicate_rows": 0,
        "constant_columns": 0,
    }
    assert training.exact_overview["rows"] == 440_833
    assert training.exact_overview["missing_cells"] == 12
    assert testing.exact_overview["rows"] == 64_374
    assert testing.exact_overview["missing_cells"] == 0
    assert not [item for item in netflix.target_candidates if item["auto_select"]]
    assert [item["column"] for item in training.target_candidates if item["auto_select"]] == [
        "Churn"
    ]
    assert [item["column"] for item in testing.target_candidates if item["auto_select"]] == [
        "Churn"
    ]

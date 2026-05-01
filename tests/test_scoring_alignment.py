import pandas as pd
import pytest

from src.pipeline import align_prediction_frame


def test_align_prediction_frame_reorders_to_training_features():
    prediction_df = pd.DataFrame({"b": [3, 4], "a": [1, 2], "extra": [8, 9]})

    aligned = align_prediction_frame(prediction_df, ["a", "b"])

    assert aligned.columns.tolist() == ["a", "b"]
    assert aligned.to_dict(orient="list") == {"a": [1, 2], "b": [3, 4]}


def test_align_prediction_frame_raises_on_missing_columns():
    prediction_df = pd.DataFrame({"a": [1], "b": [2]})

    with pytest.raises(ValueError, match="missing required columns: c"):
        align_prediction_frame(prediction_df, ["a", "c"])

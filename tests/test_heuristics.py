import pandas as pd

from src.heuristics import is_identifier_like, sample_values


def test_identifier_detection_uses_tokens_not_arbitrary_id_substrings():
    values = pd.Series([f"value-{index}" for index in range(100)])

    assert is_identifier_like(values, "customer_id") is True
    assert is_identifier_like(values, "CustomerID") is True
    assert is_identifier_like(values, "paid_amount") is False
    assert is_identifier_like(values, "valid_result") is False
    assert is_identifier_like(values, "video_score") is False
    assert is_identifier_like(values, "width") is False


def test_sample_values_stops_after_collecting_the_requested_unique_values():
    class RenderGuard:
        def __init__(self, value, may_render=True):
            self.value = value
            self.may_render = may_render

        def __str__(self):
            if not self.may_render:
                raise AssertionError("sample_values rendered rows beyond its bounded sample")
            return self.value

    values = pd.Series(
        [
            RenderGuard("alpha"),
            RenderGuard("beta"),
            RenderGuard("gamma"),
            *[RenderGuard("unused", may_render=False) for _ in range(10_000)],
        ],
        dtype="object",
    )

    assert sample_values(values, max_values=3) == "alpha, beta, gamma"

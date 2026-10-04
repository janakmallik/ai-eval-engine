from aieval.tracing.usage import ModelUsage


def test_model_usage_stores_token_counts():
    usage = ModelUsage(
        input_tokens=100,
        output_tokens=50,
    )

    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 150


import pytest

from aieval.tracing.usage import ModelUsage


def test_model_usage_stores_token_counts():
    usage = ModelUsage(
        input_tokens=100,
        output_tokens=50,
    )

    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 150


def test_model_usage_rejects_negative_input_tokens():
    with pytest.raises(ValueError):
        ModelUsage(
            input_tokens=-1,
            output_tokens=50,
        )


def test_model_usage_rejects_negative_output_tokens():
    with pytest.raises(ValueError):
        ModelUsage(
            input_tokens=100,
            output_tokens=-1,
        )


def test_model_usage_stores_cost():
    usage = ModelUsage(
        input_tokens=100,
        output_tokens=50,
        input_cost=0.001,
        output_cost=0.002,
    )

    assert usage.input_cost == 0.001
    assert usage.output_cost == 0.002
    assert usage.total_cost == 0.003


def test_model_usage_to_dict():
    usage = ModelUsage(
        input_tokens=100,
        output_tokens=50,
        input_cost=0.001,
        output_cost=0.002,
    )

    assert usage.to_dict() == {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "input_cost": 0.001,
        "output_cost": 0.002,
        "total_cost": 0.003,
    }

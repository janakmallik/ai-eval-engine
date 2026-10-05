from aieval.tracing.response import ModelResponse
from aieval.tracing.usage import ModelUsage


def test_model_response_stores_output():
    response = ModelResponse(output="Paris")

    assert response.output == "Paris"
    assert response.usage is None


def test_model_response_stores_usage():
    usage = ModelUsage(
        input_tokens=100,
        output_tokens=20,
        input_cost=0.001,
        output_cost=0.002,
    )

    response = ModelResponse(
        output="Paris",
        usage=usage,
    )

    assert response.output == "Paris"
    assert response.usage is usage


def test_model_response_stores_finish_reason():
    response = ModelResponse(
        output="Paris",
        finish_reason="stop",
    )

    assert response.finish_reason == "stop"


def test_model_response_stores_response_id():
    response = ModelResponse(
        output="Paris",
        response_id="response-123",
    )

    assert response.response_id == "response-123"


def test_model_response_to_dict():
    response = ModelResponse(
        output="Paris",
        usage=ModelUsage(
            input_tokens=100,
            output_tokens=20,
            input_cost=0.001,
            output_cost=0.002,
        ),
        finish_reason="stop",
        response_id="response-123",
    )

    assert response.to_dict() == {
        "output": "Paris",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_cost": 0.001,
            "output_cost": 0.002,
            "total_cost": 0.003,
        },
        "finish_reason": "stop",
        "response_id": "response-123",
        "temperature": None,
        "max_tokens": None,
    }


def test_model_response_round_trip():
    response = ModelResponse(
        output="Paris",
        usage=ModelUsage(
            input_tokens=100,
            output_tokens=20,
            input_cost=0.001,
            output_cost=0.002,
        ),
        finish_reason="stop",
        response_id="response-123",
    )

    restored = ModelResponse.from_dict(response.to_dict())

    assert restored.output == "Paris"
    assert restored.usage is not None
    assert restored.usage.input_tokens == 100
    assert restored.usage.output_tokens == 20
    assert restored.usage.input_cost == 0.001
    assert restored.usage.output_cost == 0.002
    assert restored.finish_reason == "stop"
    assert restored.response_id == "response-123"


# ModelResponse without usage, explicitly verify that a normal response with no
# usage still serializes and restores correctly.
def test_model_response_round_trip_without_usage():
    response = ModelResponse(
        output="Paris",
        finish_reason="stop",
        response_id="response-123",
    )

    restored = ModelResponse.from_dict(response.to_dict())

    assert restored.output == "Paris"
    assert restored.usage is None
    assert restored.finish_reason == "stop"
    assert restored.response_id == "response-123"


def test_model_response_to_dict_without_usage():
    response = ModelResponse(
        output="Paris",
        finish_reason="stop",
        response_id="response-123",
    )

    assert response.to_dict() == {
        "output": "Paris",
        "usage": None,
        "finish_reason": "stop",
        "response_id": "response-123",
        "temperature": None,
        "max_tokens": None,
    }


def test_model_response_stores_temperature():
    response = ModelResponse(
        output="Paris",
        temperature=0.2,
    )

    assert response.temperature == 0.2


def test_model_response_to_dict_includes_temperature():
    response = ModelResponse(
        output="Paris",
        temperature=0.2,
    )

    assert response.to_dict() == {
        "output": "Paris",
        "usage": None,
        "finish_reason": None,
        "response_id": None,
        "temperature": 0.2,
        "max_tokens": None,
    }


def test_model_response_round_trip_preserves_temperature():
    response = ModelResponse(
        output="Paris",
        temperature=0.2,
    )

    restored = ModelResponse.from_dict(response.to_dict())

    assert restored.temperature == 0.2


def test_model_response_full_round_trip():
    response = ModelResponse(
        output="Paris",
        usage=ModelUsage(
            input_tokens=100,
            output_tokens=20,
            input_cost=0.001,
            output_cost=0.002,
        ),
        finish_reason="stop",
        response_id="response-123",
        temperature=0.2,
    )

    restored = ModelResponse.from_dict(response.to_dict())

    assert restored.output == "Paris"
    assert restored.usage is not None
    assert restored.usage.input_tokens == 100
    assert restored.usage.output_tokens == 20
    assert restored.usage.total_tokens == 120
    assert restored.usage.input_cost == 0.001
    assert restored.usage.output_cost == 0.002
    assert restored.usage.total_cost == 0.003
    assert restored.finish_reason == "stop"
    assert restored.response_id == "response-123"
    assert restored.temperature == 0.2


def test_model_response_stores_max_tokens():
    response = ModelResponse(
        output="Paris",
        max_tokens=100,
    )

    assert response.max_tokens == 100


def test_model_response_to_dict_includes_max_tokens():
    response = ModelResponse(
        output="Paris",
        max_tokens=100,
    )

    assert response.to_dict() == {
        "output": "Paris",
        "usage": None,
        "finish_reason": None,
        "response_id": None,
        "temperature": None,
        "max_tokens": 100,
    }


def test_model_response_round_trip_preserves_max_tokens():
    response = ModelResponse(
        output="Paris",
        max_tokens=100,
    )

    restored = ModelResponse.from_dict(response.to_dict())

    assert restored.max_tokens == 100

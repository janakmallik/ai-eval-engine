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

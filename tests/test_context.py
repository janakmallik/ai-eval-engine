import pytest
from aieval.context import EvaluationContext
from aieval.dataset import EvalCase


def test_evaluation_context():

    case = EvalCase(
        id="001",
        input="What is 2 + 2?",
        expected="4",
    )

    context = EvaluationContext(
        case=case,
        actual="4",
    )

    assert context.case.id == "001"
    assert context.case.input == "What is 2 + 2?"
    assert context.case.expected == "4"
    assert context.actual == "4"


def test_evaluation_context_supports_metadata():

    case = EvalCase(
        id="001",
        input="What is 2 + 2?",
        expected="4",
    )

    context = EvaluationContext(
        case=case,
        actual="4",
        metadata={
            "model": "test-model",
            "latency_ms": 120,
        },
    )

    assert context.metadata["model"] == "test-model"
    assert context.metadata["latency_ms"] == 120


def test_evaluation_context_metadata_defaults_to_empty_dict():

    case = EvalCase(
        id="001",
        input="What is 2 + 2?",
        expected="4",
    )

    context = EvaluationContext(
        case=case,
        actual="4",
    )

    assert context.metadata == {}


def test_evaluation_context_is_immutable():

    context = EvaluationContext(
        case=EvalCase(
            id="001",
            input="What is 2 + 2?",
            expected="4",
        ),
        actual="4",
    )

    with pytest.raises(AttributeError):
        context.actual = "5"

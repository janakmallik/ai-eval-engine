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

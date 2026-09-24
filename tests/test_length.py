from aieval.context import EvaluationContext
from aieval.dataset import EvalCase
from aieval.evaluators.length import LengthEvaluator


def test_length_evaluator_passes():

    evaluator = LengthEvaluator(max_length=100)

    context = EvaluationContext(
        case=EvalCase(
            id="007",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="Paris is the capital of France.",
    )

    result = evaluator.evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_length_evaluator_rejects_long_output():

    evaluator = LengthEvaluator(max_length=10)

    context = EvaluationContext(
        case=EvalCase(
            id="008",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="Paris is the capital of France.",
    )

    result = evaluator.evaluate(context)

    assert result.score == 0.0
    assert result.passed is False


def test_length_evaluator_records_actual_length():

    evaluator = LengthEvaluator(max_length=100)

    context = EvaluationContext(
        case=EvalCase(
            id="009",
            input="Say hello.",
            expected="Hello",
        ),
        actual="Hello",
    )

    result = evaluator.evaluate(context)

    assert result.actual == "5"
    assert result.expected == "100"

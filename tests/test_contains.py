from aieval.context import EvaluationContext
from aieval.dataset import EvalCase
from aieval.evaluators.contains import ContainsEvaluator


def test_contains_evaluator():

    evaluator = ContainsEvaluator()

    context = EvaluationContext(
        case=EvalCase(
            id="001",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="Paris is the capital of France.",
    )

    result = evaluator.evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_contains_evaluator_fails():

    evaluator = ContainsEvaluator()

    context = EvaluationContext(
        case=EvalCase(
            id="002",
            input="What is the capital of France?",
            expected="London",
        ),
        actual="Paris is the capital of France.",
    )

    result = evaluator.evaluate(context)

    assert result.score == 0.0
    assert result.passed is False

from aieval.context import EvaluationContext
from aieval.dataset import EvalCase
from aieval.evaluators.base import Evaluator
from aieval.evaluators.exact_match import ExactMatchEvaluator


def test_exact_match_is_an_evaluator():

    evaluator: Evaluator = ExactMatchEvaluator()

    context = EvaluationContext(
        case=EvalCase(
            id="001",
            input="What is 2 + 2?",
            expected="4",
        ),
        actual="4",
    )

    result = evaluator.evaluate(context)

    assert result.passed is True
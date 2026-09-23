from aieval.evaluators.base import Evaluator
from aieval.evaluators.exact_match import ExactMatchEvaluator


def test_exact_match_is_an_evaluator():

    evaluator: Evaluator = ExactMatchEvaluator()

    result = evaluator.evaluate(
        case_id="001",
        expected="4",
        actual="4",
    )

    assert result.passed is True
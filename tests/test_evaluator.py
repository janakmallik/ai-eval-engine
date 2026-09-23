from aieval.evaluators.exact_match import ExactMatchEvaluator


def test_exact_match_passes():

    evaluator = ExactMatchEvaluator()

    result = evaluator.evaluate(
        case_id="001",
        expected="4",
        actual="4",
    )

    assert result.score == 1.0
    assert result.passed is True


def test_exact_match_fails():

    evaluator = ExactMatchEvaluator()

    result = evaluator.evaluate(
        case_id="002",
        expected="4",
        actual="5",
    )

    assert result.score == 0.0
    assert result.passed is False

from aieval.evaluators.contains import ContainsEvaluator


def test_contains_evaluator():

    evaluator = ContainsEvaluator()

    result = evaluator.evaluate(
        case_id="001",
        expected="Paris",
        actual="Paris is the capital of France.",
    )

    assert result.score == 1.0
    assert result.passed is True


def test_contains_evaluator_fails():

    evaluator = ContainsEvaluator()

    result = evaluator.evaluate(
        case_id="002",
        expected="London",
        actual="Paris is the capital of France.",
    )

    assert result.score == 0.0
    assert result.passed is False

from aieval.result import EvaluationResult


def test_evaluation_result():
    result = EvaluationResult(
        case_id="001",
        evaluator_name="exact_match",
        expected="4",
        actual="4",
        score=1.0,
        passed=True,
    )

    assert result.case_id == "001"
    assert result.expected == "4"
    assert result.actual == "4"
    assert result.score == 1.0
    assert result.passed is True


def test_evaluation_result_to_dict():

    result = EvaluationResult(
        case_id="001",
        evaluator_name="exact_match",
        expected="4",
        actual="4",
        score=1.0,
        passed=True,
    )

    assert result.to_dict() == {
        "case_id": "001",
        "evaluator_name": "exact_match",
        "expected": "4",
        "actual": "4",
        "score": 1.0,
        "passed": True,
    }

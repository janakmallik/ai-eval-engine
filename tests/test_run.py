from aieval.result import EvaluationResult
from aieval.run import EvaluationRun


def test_evaluation_run():

    results = [
        EvaluationResult(
            case_id="001",
            expected="4",
            actual="4",
            score=1.0,
            passed=True,
        ),
        EvaluationResult(
            case_id="002",
            expected="Paris",
            actual="Paris",
            score=1.0,
            passed=True,
        ),
        EvaluationResult(
            case_id="003",
            expected="5",
            actual="6",
            score=0.0,
            passed=False,
        ),
    ]

    run = EvaluationRun(results)

    assert run.total == 3
    assert run.passed == 2
    assert run.failed == 1
    assert run.score == 2 / 3

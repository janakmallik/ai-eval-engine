from aieval.result import EvaluationResult
from aieval.run import EvaluationRun
from aieval.summary import EvaluationSummary

def test_evaluation_run_summary():
    results = [
        EvaluationResult(
            case_id="001",
            evaluator_name="exact_match",
            expected="4",
            actual="4",
            score=1.0,
            passed=True,
        ),
        EvaluationResult(
            case_id="002",
            evaluator_name="exact_match",
            expected="5",
            actual="5",
            score=1.0,
            passed=True,
        ),
        EvaluationResult(
            case_id="003",
            evaluator_name="exact_match",
            expected="6",
            actual="7",
            score=0.0,
            passed=False,
        ),
    ]

    run = EvaluationRun(results)

    summary = run.summary("exact_match")

    assert summary.evaluator_name == "exact_match"
    assert summary.total == 3
    assert summary.passed == 2
    assert summary.failed == 1
    assert summary.score == 2 / 3
    assert summary.pass_rate == 2 / 3


def test_evaluation_run_summary_ignores_other_evaluators():
    results = [
        EvaluationResult(
            case_id="001",
            evaluator_name="exact_match",
            expected="4",
            actual="4",
            score=1.0,
            passed=True,
        ),
        EvaluationResult(
            case_id="001",
            evaluator_name="length",
            expected="4",
            actual="4",
            score=0.5,
            passed=False,
        ),
    ]

    run = EvaluationRun(results)

    summary = run.summary("exact_match")

    assert summary.total == 1
    assert summary.passed == 1
    assert summary.failed == 0
    assert summary.score == 1.0
    assert summary.pass_rate == 1.0

def test_evaluation_summary_to_dict():

    summary = EvaluationSummary(
        evaluator_name="exact_match",
        total=2,
        passed=1,
        failed=1,
        score=0.5,
        pass_rate=0.5,
    )

    assert summary.to_dict() == {
        "evaluator_name": "exact_match",
        "total": 2,
        "passed": 1,
        "failed": 1,
        "score": 0.5,
        "pass_rate": 0.5,
    }
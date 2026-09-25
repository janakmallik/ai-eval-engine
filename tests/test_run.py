from aieval.result import EvaluationResult
from aieval.run import EvaluationRun


def test_evaluation_run():

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
            expected="Paris",
            actual="Paris",
            score=1.0,
            passed=True,
        ),
        EvaluationResult(
            case_id="003",
            evaluator_name="exact_match",
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


def test_evaluation_run_pass_rate():

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
            actual="6",
            score=0.0,
            passed=False,
        ),
    ]

    run = EvaluationRun(results)

    assert run.pass_rate == 0.5


def test_evaluation_run_filters_by_evaluator():

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
            score=1.0,
            passed=True,
        ),
        EvaluationResult(
            case_id="002",
            evaluator_name="exact_match",
            expected="5",
            actual="6",
            score=0.0,
            passed=False,
        ),
    ]

    run = EvaluationRun(results)

    exact_match_results = run.by_evaluator("exact_match")

    assert len(exact_match_results) == 2
    assert all(result.evaluator_name == "exact_match"
               for result in exact_match_results)


def test_evaluation_run_summaries():

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
            actual="6",
            score=0.0,
            passed=False,
        ),
        EvaluationResult(
            case_id="003",
            evaluator_name="contains",
            expected="Paris",
            actual="Paris is a city.",
            score=1.0,
            passed=True,
        ),
    ]

    run = EvaluationRun(results)

    summaries = run.summaries()

    assert set(summaries) == {"exact_match", "contains"}

    assert summaries["exact_match"].total == 2
    assert summaries["exact_match"].passed == 1
    assert summaries["exact_match"].failed == 1

    assert summaries["contains"].total == 1
    assert summaries["contains"].passed == 1
    assert summaries["contains"].failed == 0

# Verifies that EvaluationRun.to_dict() serializes the run's aggregate stats (total, passed, failed, score, pass_rate) along with a nested per-evaluator summaries dict, where each summary is itself serialized via EvaluationSummary.to_dict().
def test_evaluation_run_to_dict():

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
            actual="6",
            score=0.0,
            passed=False,
        ),
    ]

    run = EvaluationRun(results)

    assert run.to_dict() == {
        "total": 2,
        "passed": 1,
        "failed": 1,
        "score": 0.5,
        "pass_rate": 0.5,
        "summaries": {
            "exact_match": {
                "evaluator_name": "exact_match",
                "total": 2,
                "passed": 1,
                "failed": 1,
                "score": 0.5,
                "pass_rate": 0.5,
            }
        },
    }

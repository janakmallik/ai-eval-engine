from aieval.comparison import compare_runs
from aieval.result import EvaluationResult
from aieval.run import EvaluationRun


def test_compare_runs():
    baseline = EvaluationRun(
        results=[
            EvaluationResult(
                case_id="001",
                evaluator_name="exact_match",
                expected="Paris",
                actual="Paris",
                score=1.0,
                passed=True,
            ),
            EvaluationResult(
                case_id="002",
                evaluator_name="exact_match",
                expected="London",
                actual="London",
                score=1.0,
                passed=True,
            ),
        ]
    )

    current = EvaluationRun(
        results=[
            EvaluationResult(
                case_id="001",
                evaluator_name="exact_match",
                expected="Paris",
                actual="Paris",
                score=1.0,
                passed=True,
            ),
            EvaluationResult(
                case_id="002",
                evaluator_name="exact_match",
                expected="London",
                actual="Paris",
                score=0.0,
                passed=False,
            ),
        ]
    )

    comparison = compare_runs(baseline, current)

    assert comparison.baseline_score == 1.0
    assert comparison.current_score == 0.5
    assert comparison.score_delta == -0.5

    assert comparison.baseline_pass_rate == 1.0
    assert comparison.current_pass_rate == 0.5
    assert comparison.pass_rate_delta == -0.5

def test_compare_runs_by_evaluator():
    baseline = EvaluationRun(
        results=[
            EvaluationResult(
                case_id="001",
                evaluator_name="exact_match",
                expected="Paris",
                actual="Paris",
                score=1.0,
                passed=True,
            ),
            EvaluationResult(
                case_id="001",
                evaluator_name="similarity",
                expected="Paris",
                actual="Paris",
                score=1.0,
                passed=True,
            ),
        ]
    )

    current = EvaluationRun(
        results=[
            EvaluationResult(
                case_id="001",
                evaluator_name="exact_match",
                expected="Paris",
                actual="London",
                score=0.0,
                passed=False,
            ),
            EvaluationResult(
                case_id="001",
                evaluator_name="similarity",
                expected="Paris",
                actual="Paris",
                score=1.0,
                passed=True,
            ),
        ]
    )

    comparison = compare_runs(baseline, current)

    assert comparison.evaluator_deltas["exact_match"] == -1.0
    assert comparison.evaluator_deltas["similarity"] == 0.0
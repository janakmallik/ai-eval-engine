from aieval.comparison import ComparisonResult
from aieval.regression import RegressionDetector
from aieval.gate import RegressionGate


def test_regression_gate_passes_when_no_regression():
    comparison = ComparisonResult(
        baseline_score=0.92,
        current_score=0.91,
        score_delta=-0.01,
        baseline_pass_rate=0.92,
        current_pass_rate=0.91,
        pass_rate_delta=-0.01,
        evaluator_deltas={
            "exact_match": -0.01,
            "similarity": 0.01,
        },
    )

    detector = RegressionDetector(threshold=0.05)
    result = detector.check(comparison)

    gate = RegressionGate()

    assert gate.passed(result) is True


def test_regression_gate_fails_when_regression_detected():
    comparison = ComparisonResult(
        baseline_score=0.92,
        current_score=0.86,
        score_delta=-0.06,
        baseline_pass_rate=0.92,
        current_pass_rate=0.86,
        pass_rate_delta=-0.06,
        evaluator_deltas={
            "exact_match": -0.10,
            "similarity": 0.01,
        },
    )

    detector = RegressionDetector(threshold=0.05)
    result = detector.check(comparison)

    gate = RegressionGate()

    assert gate.passed(result) is False
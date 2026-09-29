from aieval.comparison import ComparisonResult
from aieval.gate import GateResult, RegressionGate
from aieval.regression import RegressionConfig, RegressionDetector


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

    detector = RegressionDetector(
        config=RegressionConfig(threshold=0.05)
    )
    result = detector.check(comparison)

    gate = RegressionGate()
    gate_result = gate.check(result)

    assert isinstance(gate_result, GateResult)
    assert gate_result.passed is True
    assert gate_result.status == "passed"


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

    detector = RegressionDetector(
        config=RegressionConfig(threshold=0.05)
    )
    result = detector.check(comparison)

    gate = RegressionGate()
    gate_result = gate.check(result)

    assert isinstance(gate_result, GateResult)
    assert gate_result.passed is False
    assert gate_result.status == "failed"
    assert "regression" in gate_result.reason.lower()


def test_regression_gate_returns_zero_for_passed_gate():
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

    detector = RegressionDetector(
        config=RegressionConfig(threshold=0.05)
    )
    result = detector.check(comparison)

    gate = RegressionGate()
    gate_result = gate.check(result)

    assert gate.exit_code(gate_result) == 0


def test_regression_gate_returns_one_for_failed_gate():
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

    detector = RegressionDetector(
        config=RegressionConfig(threshold=0.05)
    )
    result = detector.check(comparison)

    gate = RegressionGate()
    gate_result = gate.check(result)

    assert gate.exit_code(gate_result) == 1
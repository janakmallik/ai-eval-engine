from aieval.cli import format_regression_report, main
from aieval.comparison import ComparisonResult
from aieval.gate import GateResult
from aieval.regression import RegressionResult


def test_cli_help():
    assert main(["--help"]) == 0


def test_format_regression_report():
    comparison = ComparisonResult(
        baseline_score=1.0,
        current_score=0.5,
        score_delta=-0.5,
        baseline_pass_rate=1.0,
        current_pass_rate=0.5,
        pass_rate_delta=-0.5,
        evaluator_deltas={
            "exact_match": -0.5,
            "similarity": 0.1,
        },
    )

    regression_result = RegressionResult(
        regressed=True,
        score_regression=True,
        evaluator_regressions=["exact_match"],
    )

    gate_result = GateResult(
        passed=False,
        status="failed",
        reason="Regression detected",
    )

    report = format_regression_report(
        comparison,
        regression_result,
        gate_result,
    )

    assert "Status: FAILED" in report
    assert "Reason: Regression detected" in report
    assert "baseline score: 1.000" in report
    assert "current score:  0.500" in report
    assert "score delta:    -0.500" in report
    assert "exact_match: -0.500" in report
    assert "similarity: +0.100" in report
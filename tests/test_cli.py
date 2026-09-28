from aieval.cli import format_regression_report, main
from aieval.comparison import ComparisonResult
from aieval.run import EvaluationRun
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

    baseline = EvaluationRun(
        results=[],
        metadata={
            "model": "model_v1",
            "dataset": "capitals-v1",
        },
    )

    current = EvaluationRun(
        results=[],
        metadata={
            "model": "model_v2",
            "dataset": "capitals-v1",
        },
    )

    report = format_regression_report(
        comparison,
        regression_result,
        gate_result,
        baseline,
        current,
    )

    assert "Status: FAILED" in report
    assert "Reason: Regression detected" in report
    assert "baseline score: 1.000" in report
    assert "current score:  0.500" in report
    assert "score delta:    -0.500" in report
    assert "exact_match: -0.500" in report
    assert "similarity: +0.100" in report
    assert "baseline model: model_v1" in report
    assert "current model:  model_v2" in report
    assert "dataset:        capitals-v1" in report

def test_cli_regression_returns_failure_exit_code(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"

    baseline.write_text(
        """
        {
            "metadata": {},
            "results": [
                {
                    "case_id": "001",
                    "evaluator_name": "exact_match",
                    "expected": "Paris",
                    "actual": "Paris",
                    "score": 1.0,
                    "passed": true
                }
            ]
        }
        """,
        encoding="utf-8",
    )

    current.write_text(
        """
        {
            "metadata": {},
            "results": [
                {
                    "case_id": "001",
                    "evaluator_name": "exact_match",
                    "expected": "Paris",
                    "actual": "London",
                    "score": 0.0,
                    "passed": false
                }
            ]
        }
        """,
        encoding="utf-8",
    )

    exit_code = main(
        [
            "regression",
            "--baseline",
            str(baseline),
            "--current",
            str(current),
        ]
    )

    assert exit_code == 1

def test_cli_regression_passes_when_threshold_allows_it(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"

    baseline.write_text(
        """
        {
            "metadata": {},
            "results": [
                {
                    "case_id": "001",
                    "evaluator_name": "exact_match",
                    "expected": "Paris",
                    "actual": "Paris",
                    "score": 1.0,
                    "passed": true
                }
            ]
        }
        """,
        encoding="utf-8",
    )

    current.write_text(
        """
        {
            "metadata": {},
            "results": [
                {
                    "case_id": "001",
                    "evaluator_name": "exact_match",
                    "expected": "Paris",
                    "actual": "London",
                    "score": 0.5,
                    "passed": false
                }
            ]
        }
        """,
        encoding="utf-8",
    )

    exit_code = main(
        [
            "regression",
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--threshold",
            "0.6",
        ]
    )

    assert exit_code == 0
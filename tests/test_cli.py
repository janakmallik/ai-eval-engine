from aieval.cli import format_regression_report, main
from aieval.comparison import ComparisonResult
from aieval.dataset import EvalCase
from aieval.evaluators.exact_match import ExactMatchEvaluator
from aieval.gate import GateResult
from aieval.regression import RegressionResult
from aieval.reporting.json import JsonReporter
from aieval.run import EvaluationRun
from aieval.runner import evaluate_dataset


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


def test_cli_accepts_evaluator_thresholds():
    from aieval.cli import build_parser

    parser = build_parser()

    args = parser.parse_args(
        [
            "regression",
            "--baseline",
            "baseline.json",
            "--current",
            "current.json",
            "--evaluator-threshold",
            "exact_match=0.01",
            "--evaluator-threshold",
            "similarity=0.10",
        ]
    )

    assert args.evaluator_threshold == [
        "exact_match=0.01",
        "similarity=0.10",
    ]


def test_parse_evaluator_thresholds():
    from aieval.cli import parse_evaluator_thresholds

    result = parse_evaluator_thresholds(
        [
            "exact_match=0.01",
            "similarity=0.10",
        ]
    )

    assert result == {
        "exact_match": 0.01,
        "similarity": 0.10,
    }


def test_cli_regression_works_with_real_json_reports(tmp_path):

    dataset = [
        EvalCase(
            id="1",
            input="What is the capital of France?",
            expected="Paris",
        )
    ]

    baseline_run = evaluate_dataset(
        model=lambda _: "Paris",
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        metadata={
            "model": "model_v1",
            "dataset": "capitals-v1",
        },
    )

    current_run = evaluate_dataset(
        model=lambda _: "London",
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        metadata={
            "model": "model_v2",
            "dataset": "capitals-v1",
        },
    )

    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"

    reporter = JsonReporter()
    reporter.write(baseline_run, baseline)
    reporter.write(current_run, current)

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


def test_cli_regression_accepts_real_json_reports_with_threshold(tmp_path):

    dataset = [
        EvalCase(
            id="1",
            input="What is the capital of France?",
            expected="Paris",
        )
    ]

    baseline_run = evaluate_dataset(
        model=lambda _: "Paris",
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
    )

    current_run = evaluate_dataset(
        model=lambda _: "London",
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
    )

    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"

    reporter = JsonReporter()
    reporter.write(baseline_run, baseline)
    reporter.write(current_run, current)

    exit_code = main(
        [
            "regression",
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--threshold",
            "1.0",
        ]
    )

    assert exit_code == 0


def test_cli_regression_passes_when_runs_are_identical(tmp_path):

    dataset = [
        EvalCase(
            id="1",
            input="What is the capital of France?",
            expected="Paris",
        )
    ]

    run = evaluate_dataset(
        model=lambda _: "Paris",
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
    )

    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"

    reporter = JsonReporter()
    reporter.write(run, baseline)
    reporter.write(run, current)

    exit_code = main(
        [
            "regression",
            "--baseline",
            str(baseline),
            "--current",
            str(current),
        ]
    )

    assert exit_code == 0


def test_cli_regression_fails_when_evaluator_regresses(tmp_path):

    dataset = [
        EvalCase(
            id="1",
            input="What is the capital of France?",
            expected="Paris",
        )
    ]

    baseline_run = evaluate_dataset(
        model=lambda _: "Paris",
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
    )

    current_run = evaluate_dataset(
        model=lambda _: "London",
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
    )

    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"

    reporter = JsonReporter()
    reporter.write(baseline_run, baseline)
    reporter.write(current_run, current)

    exit_code = main(
        [
            "regression",
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--evaluator-threshold",
            "exact_match=0.0",
        ]
    )

    assert exit_code == 1


# Together they verify: (1) the argument parser works, (2) the command renders
# a trace tree when tracing data exists, and (3) it prints a friendly message
# when there's no trace data.
# start
def test_cli_trace_command_accepts_report_path():
    from aieval.cli import build_parser

    parser = build_parser()

    args = parser.parse_args(
        [
            "trace",
            "--input",
            "run.json",
        ]
    )

    assert args.command == "trace"
    assert args.input == "run.json"


def test_cli_trace_command_renders_trace_tree(tmp_path, capsys):

    dataset = [
        EvalCase(
            id="1",
            input="What is the capital of France?",
            expected="Paris",
        )
    ]

    run = evaluate_dataset(
        model=lambda _: "Paris",
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        enable_tracing=True,
    )

    report = tmp_path / "run.json"
    JsonReporter().write(run, report)

    exit_code = main(
        [
            "trace",
            "--input",
            str(report),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "evaluation [" in captured.out
    assert "evaluation.case [" in captured.out
    assert "model [" in captured.out
    assert "evaluator.exact_match [" in captured.out


def test_cli_trace_command_works_without_trace(tmp_path, capsys):

    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    run = evaluate_dataset(
        model=lambda _: "hello",
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
    )

    report = tmp_path / "run.json"
    JsonReporter().write(run, report)

    exit_code = main(
        [
            "trace",
            "--input",
            str(report),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No trace data available." in captured.out


# end
def test_cli_trace_command_renders_span_attributes(tmp_path, capsys):

    dataset = [
        EvalCase(
            id="1",
            input="What is the capital of France?",
            expected="Paris",
        )
    ]

    run = evaluate_dataset(
        model=lambda _: "Paris",
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        metadata={
            "model": "test-model",
            "dataset": "capitals-v1",
        },
        enable_tracing=True,
    )

    report = tmp_path / "run.json"
    JsonReporter().write(run, report)

    exit_code = main(
        [
            "trace",
            "--input",
            str(report),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "model.input:" in captured.out
    assert "model.output:" in captured.out

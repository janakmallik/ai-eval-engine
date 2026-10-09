from aieval.cli import format_regression_report, main
from aieval.comparison import ComparisonResult
from aieval.dataset import EvalCase
from aieval.evaluators.exact_match import ExactMatchEvaluator
from aieval.gate import GateResult
from aieval.regression import RegressionResult
from aieval.reporting.json import JsonReporter
from aieval.run import EvaluationRun
from aieval.runner import evaluate_dataset
from aieval.tracing.trace import Trace


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


def test_cli_trace_command_renders_error_status_and_exception(
    tmp_path,
    capsys,
):

    trace = Trace()

    tool_span = trace.start_tool(
        tool="web_search",
        parent_span_id=None,
    )

    try:
        with tool_span:
            raise RuntimeError("search failed")
    except RuntimeError:
        pass

    from aieval.run import EvaluationRun

    run = EvaluationRun(
        results=[],
        trace=trace,
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
    assert "tool [error]" in captured.out
    assert "search failed" in captured.out


def test_format_regression_report_includes_performance_regressions():
    comparison = ComparisonResult(
        baseline_score=0.91,
        current_score=0.94,
        score_delta=0.03,
        baseline_pass_rate=0.91,
        current_pass_rate=0.94,
        pass_rate_delta=0.03,
        baseline_latency=0.80,
        current_latency=1.40,
        latency_delta=0.60,
        baseline_cost=0.010,
        current_cost=0.018,
        cost_delta=0.008,
        baseline_error_rate=0.02,
        current_error_rate=0.05,
        error_rate_delta=0.03,
        evaluator_deltas={
            "faithfulness": 0.03,
        },
    )

    regression_result = RegressionResult(
        regressed=True,
        score_regression=False,
        evaluator_regressions=[],
        latency_regression=True,
        cost_regression=True,
        error_rate_regression=True,
    )

    gate_result = GateResult(
        passed=False,
        status="failed",
        reason="Regression detected",
    )

    baseline = EvaluationRun(
        results=[],
        metadata={
            "model": "model-v1",
            "dataset": "test-set",
        },
    )

    current = EvaluationRun(
        results=[],
        metadata={
            "model": "model-v2",
            "dataset": "test-set",
        },
    )

    output = format_regression_report(
        comparison,
        regression_result,
        gate_result,
        baseline,
        current,
    )

    assert "Latency:" in output
    assert "baseline latency: 0.800s" in output
    assert "current latency:  1.400s" in output
    assert "latency regression: YES" in output

    assert "Cost:" in output
    assert "baseline cost: $0.010" in output
    assert "current cost:  $0.018" in output
    assert "cost regression: YES" in output

    assert "Reliability:" in output
    assert "baseline error rate: 0.020" in output
    assert "current error rate:  0.050" in output
    assert "error-rate regression: YES" in output


def test_cli_regression_fails_on_performance_regressions(tmp_path, capsys):
    reporter = JsonReporter()

    baseline = EvaluationRun(
        results=[],
        metadata={
            "model": "model-v1",
            "dataset": "test-set",
            "latency": 0.80,
            "cost": 0.010,
            "error_rate": 0.02,
        },
    )

    current = EvaluationRun(
        results=[],
        metadata={
            "model": "model-v2",
            "dataset": "test-set",
            "latency": 1.40,
            "cost": 0.018,
            "error_rate": 0.05,
        },
    )

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"

    reporter.write(baseline, baseline_path)
    reporter.write(current, current_path)

    exit_code = main(
        [
            "regression",
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--threshold",
            "0.0",
            "--latency-threshold",
            "0.10",
            "--cost-threshold",
            "0.002",
            "--error-rate-threshold",
            "0.01",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Status: FAILED" in captured.out
    assert "latency regression: YES" in captured.out
    assert "cost regression: YES" in captured.out
    assert "error-rate regression: YES" in captured.out


def test_format_regression_report_includes_change_summary():
    comparison = ComparisonResult(
        baseline_score=0.92,
        current_score=0.86,
        score_delta=-0.06,
        baseline_pass_rate=0.92,
        current_pass_rate=0.86,
        pass_rate_delta=-0.06,
        model_changed=True,
        prompt_changed=True,
        dataset_changed=False,
        evaluator_config_changed=True,
        model_version_changed=True,
        prompt_version_changed=False,
        dataset_version_changed=False,
    )

    regression_result = RegressionResult(
        regressed=True,
        score_regression=True,
        evaluator_regressions=[],
    )

    gate_result = GateResult(
        passed=False,
        status="failed",
        reason="Regression detected: quality",
    )

    baseline = EvaluationRun(results=[])
    current = EvaluationRun(results=[])

    output = format_regression_report(
        comparison,
        regression_result,
        gate_result,
        baseline,
        current,
    )

    assert "Changes:" in output
    assert "model: changed" in output
    assert "model version: changed" in output
    assert "prompt: changed" in output
    assert "prompt version: unchanged" in output
    assert "dataset: unchanged" in output
    assert "evaluator configuration: changed" in output


def test_cli_regression_uses_trace_metrics_from_json_reports(
    tmp_path,
    capsys,
):
    from aieval.tracing.usage import ModelUsage

    baseline_trace = Trace()
    baseline_model = baseline_trace.start_model(
        model="model",
        provider="test",
    )
    baseline_model.record_usage(
        ModelUsage(
            input_tokens=100,
            output_tokens=50,
            input_cost=0.001,
            output_cost=0.002,
        )
    )
    baseline_model.end()
    baseline_trace.end()

    current_trace = Trace()
    current_model = current_trace.start_model(
        model="model",
        provider="test",
    )
    current_model.record_usage(
        ModelUsage(
            input_tokens=200,
            output_tokens=100,
            input_cost=0.002,
            output_cost=0.004,
        )
    )
    current_model.end()
    current_trace.end()

    baseline_run = EvaluationRun(
        results=[],
        trace=baseline_trace,
    )

    current_run = EvaluationRun(
        results=[],
        trace=current_trace,
    )

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"

    reporter = JsonReporter()
    reporter.write(baseline_run, baseline_path)
    reporter.write(current_run, current_path)

    exit_code = main(
        [
            "regression",
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "baseline cost: $0.003" in captured.out
    assert "current cost:  $0.006" in captured.out
    assert "cost delta:    $+0.003" in captured.out


def test_cli_regression_uses_trace_error_rate_from_json_reports(
    tmp_path,
    capsys,
):
    baseline_trace = Trace()

    baseline_success_1 = baseline_trace.start_span("success")
    baseline_success_1.end()

    baseline_success_2 = baseline_trace.start_span("success")
    baseline_success_2.end()

    baseline_error = baseline_trace.start_span("error")
    baseline_error.status = "error"
    baseline_error.end()

    baseline_trace.end()

    current_trace = Trace()

    current_success = current_trace.start_span("success")
    current_success.end()

    current_error = current_trace.start_span("error")
    current_error.status = "error"
    current_error.end()

    current_trace.end()

    baseline_run = EvaluationRun(
        results=[],
        metadata={"error_rate": 0.99},
        trace=baseline_trace,
    )

    current_run = EvaluationRun(
        results=[],
        metadata={"error_rate": 0.99},
        trace=current_trace,
    )

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"

    reporter = JsonReporter()
    reporter.write(baseline_run, baseline_path)
    reporter.write(current_run, current_path)

    exit_code = main(
        [
            "regression",
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "baseline error rate: 0.333" in captured.out
    assert "current error rate:  0.500" in captured.out
    assert "error-rate delta:    +0.167" in captured.out


def test_cli_regression_handles_missing_report(tmp_path, capsys):
    missing = tmp_path / "missing.json"
    valid = tmp_path / "valid.json"

    JsonReporter().write(EvaluationRun(results=[]), valid)

    exit_code = main(
        [
            "regression",
            "--baseline",
            str(missing),
            "--current",
            str(valid),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error:" in captured.err
    assert "missing.json" in captured.err
    assert "Traceback" not in captured.err


def test_cli_regression_handles_invalid_json(tmp_path, capsys):
    invalid = tmp_path / "invalid.json"
    valid = tmp_path / "valid.json"

    invalid.write_text("{ invalid json", encoding="utf-8")
    JsonReporter().write(EvaluationRun(results=[]), valid)

    exit_code = main(
        [
            "regression",
            "--baseline",
            str(invalid),
            "--current",
            str(valid),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error:" in captured.err
    assert "Invalid JSON" in captured.err
    assert "Traceback" not in captured.err


def test_cli_trace_handles_missing_report(tmp_path, capsys):
    missing = tmp_path / "missing.json"

    exit_code = main(["trace", "--input", str(missing)])

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error:" in captured.err
    assert "missing.json" in captured.err
    assert "Traceback" not in captured.err


def test_cli_trace_handles_invalid_json(tmp_path, capsys):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{ invalid json", encoding="utf-8")

    exit_code = main(["trace", "--input", str(invalid)])

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error:" in captured.err
    assert "Invalid JSON" in captured.err
    assert "Traceback" not in captured.err


def test_cli_trace_reads_utf8_bom_report(tmp_path, capsys):
    report_path = tmp_path / "bom-report.json"
    report_path.write_text(
        JsonReporter().render(EvaluationRun(results=[])),
        encoding="utf-8-sig",
    )

    exit_code = main(["trace", "--input", str(report_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No trace data available." in captured.out
    assert "Traceback" not in captured.err


def test_cli_trace_handles_invalid_report_structure(tmp_path, capsys):
    report_path = tmp_path / "invalid-structure.json"
    report_path.write_text("{}", encoding="utf-8")

    exit_code = main(["trace", "--input", str(report_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error:" in captured.err
    assert "invalid-structure.json" in captured.err
    assert "Traceback" not in captured.err

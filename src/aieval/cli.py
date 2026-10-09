import argparse
import json
import sys

from aieval.comparison import ComparisonResult, compare_runs
from aieval.gate import GateResult, RegressionGate
from aieval.regression import (
    RegressionConfig,
    RegressionDetector,
    RegressionResult,
)
from aieval.reporting.json import JsonReporter
from aieval.run import EvaluationRun


def read_report(reporter: JsonReporter, path: str) -> EvaluationRun | None:
    try:
        return reporter.read(path)
    except FileNotFoundError:
        print(f"Error: Report file not found: {path}", file=sys.stderr)
    except json.JSONDecodeError as exc:
        print(
            f"Error: Invalid JSON in report file '{path}': "
            f"{exc.msg} (line {exc.lineno}, column {exc.colno})",
            file=sys.stderr,
        )
    except OSError as exc:
        print(
            f"Error: Could not read report file '{path}': {exc}",
            file=sys.stderr,
        )

    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aieval",
        description="AI evaluation and regression testing engine.",
    )

    subparsers = parser.add_subparsers(dest="command")

    regression_parser = subparsers.add_parser(
        "regression",
        help="Compare two evaluation runs.",
    )

    regression_parser.add_argument(
        "--baseline",
        required=True,
        help="Path to the baseline evaluation JSON.",
    )

    regression_parser.add_argument(
        "--current",
        required=True,
        help="Path to the current evaluation JSON.",
    )

    regression_parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Maximum allowed regression.",
    )

    regression_parser.add_argument(
        "--latency-threshold",
        type=float,
        default=0.0,
        help="Maximum allowed latency regression.",
    )

    regression_parser.add_argument(
        "--cost-threshold",
        type=float,
        default=0.0,
        help="Maximum allowed cost regression.",
    )

    regression_parser.add_argument(
        "--error-rate-threshold",
        type=float,
        default=0.0,
        help="Maximum allowed error-rate regression.",
    )

    regression_parser.add_argument(
        "--evaluator-threshold",
        action="append",
        default=[],
        help="Evaluator-specific threshold in the form NAME=VALUE.",
    )

    trace_parser = subparsers.add_parser(
        "trace",
        help="Display the trace tree from an evaluation JSON.",
    )

    trace_parser.add_argument(
        "--input",
        required=True,
        help="Path to the evaluation JSON.",
    )

    return parser


def parse_evaluator_thresholds(
    values: list[str],
) -> dict[str, float]:
    thresholds: dict[str, float] = {}

    for value in values:
        name, threshold = value.split("=", 1)
        thresholds[name] = float(threshold)

    return thresholds


def main(args: list[str] | None = None) -> int:
    parser = build_parser()

    try:
        parsed_args = parser.parse_args(args)
    except SystemExit as exc:
        return int(exc.code)

    if parsed_args.command == "regression":
        reporter = JsonReporter()

        baseline = read_report(reporter, parsed_args.baseline)
        if baseline is None:
            return 1

        current = read_report(reporter, parsed_args.current)
        if current is None:
            return 1

        comparison = compare_runs(baseline, current)

        config = RegressionConfig(
            threshold=parsed_args.threshold,
            evaluator_thresholds=parse_evaluator_thresholds(
                parsed_args.evaluator_threshold
            ),
            latency_threshold=parsed_args.latency_threshold,
            cost_threshold=parsed_args.cost_threshold,
            error_rate_threshold=parsed_args.error_rate_threshold,
        )

        detector = RegressionDetector(config=config)
        regression_result = detector.check(comparison)

        gate = RegressionGate()
        gate_result = gate.check(regression_result)

        print(
            format_regression_report(
                comparison,
                regression_result,
                gate_result,
                baseline,
                current,
            )
        )

        return gate.exit_code(gate_result)

    if parsed_args.command == "trace":
        reporter = JsonReporter()
        run = read_report(reporter, parsed_args.input)
        if run is None:
            return 1

        if run.trace is None:
            print("No trace data available.")
        else:
            print(run.trace.render_span_tree())

        return 0

    return 0


def format_regression_report(
    comparison: ComparisonResult,
    regression_result: RegressionResult,
    gate_result: GateResult,
    baseline: EvaluationRun,
    current: EvaluationRun,
) -> str:
    lines = [
        f"Status: {gate_result.status.upper()}",
        f"Reason: {gate_result.reason}",
        "",
        "Runs:",
        f"  baseline model: {baseline.metadata.get('model', 'unknown')}",
        f"  current model:  {current.metadata.get('model', 'unknown')}",
        f"  dataset:        {current.metadata.get('dataset', 'unknown')}",
        "",
        "Changes:",
        f"  model: {'changed' if comparison.model_changed else 'unchanged'}",
        f"  model version: "
        f"{'changed' if comparison.model_version_changed else 'unchanged'}",
        f"  prompt: {'changed' if comparison.prompt_changed else 'unchanged'}",
        f"  prompt version: "
        f"{'changed' if comparison.prompt_version_changed else 'unchanged'}",
        f"  dataset: {'changed' if comparison.dataset_changed else 'unchanged'}",
        f"  dataset version: "
        f"{'changed' if comparison.dataset_version_changed else 'unchanged'}",
        f"  evaluator configuration: "
        f"{'changed' if comparison.evaluator_config_changed else 'unchanged'}",
        "",
        "Overall:",
        f"  baseline score: {comparison.baseline_score:.3f}",
        f"  current score:  {comparison.current_score:.3f}",
        f"  score delta:    {comparison.score_delta:+.3f}",
        "",
        "Regression details:",
        f"  overall score regression: "
        f"{'YES' if regression_result.score_regression else 'NO'}",
        "  evaluator regressions:",
    ]

    for evaluator_name in sorted(regression_result.evaluator_regressions):
        delta = comparison.evaluator_deltas[evaluator_name]
        lines.append(f"    - {evaluator_name}: {delta:+.3f}")

    lines.extend(
        [
            "",
            "Latency:",
            f"  baseline latency: {comparison.baseline_latency:.3f}s",
            f"  current latency:  {comparison.current_latency:.3f}s",
            f"  latency delta:    {comparison.latency_delta:+.3f}s",
            f"  latency regression: "
            f"{'YES' if regression_result.latency_regression else 'NO'}",
            "",
            "Cost:",
            f"  baseline cost: ${comparison.baseline_cost:.3f}",
            f"  current cost:  ${comparison.current_cost:.3f}",
            f"  cost delta:    ${comparison.cost_delta:+.3f}",
            f"  cost regression: "
            f"{'YES' if regression_result.cost_regression else 'NO'}",
            "",
            "Reliability:",
            f"  baseline error rate: {comparison.baseline_error_rate:.3f}",
            f"  current error rate:  {comparison.current_error_rate:.3f}",
            f"  error-rate delta:    {comparison.error_rate_delta:+.3f}",
            f"  error-rate regression: "
            f"{'YES' if regression_result.error_rate_regression else 'NO'}",
            "",
            "Evaluator deltas:",
        ]
    )

    for evaluator_name, delta in sorted(comparison.evaluator_deltas.items()):
        lines.append(f"  {evaluator_name}: {delta:+.3f}")

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

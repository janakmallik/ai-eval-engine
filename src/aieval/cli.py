import argparse

from aieval.comparison import ComparisonResult, compare_runs
from aieval.gate import GateResult, RegressionGate
from aieval.regression import RegressionDetector, RegressionResult
from aieval.reporting.json import JsonReporter
from aieval.run import EvaluationRun


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

    return parser


def main(args: list[str] | None = None) -> int:
    parser = build_parser()

    try:
        parsed_args = parser.parse_args(args)
    except SystemExit as exc:
        return int(exc.code)

    if parsed_args.command == "regression":
        reporter = JsonReporter()

        baseline = reporter.read(parsed_args.baseline)
        current = reporter.read(parsed_args.current)

        comparison = compare_runs(baseline, current)

        detector = RegressionDetector(
            threshold=parsed_args.threshold
        )
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

    for evaluator_name in sorted(
        regression_result.evaluator_regressions
    ):
        delta = comparison.evaluator_deltas[evaluator_name]
        lines.append(f"    - {evaluator_name}: {delta:+.3f}")

    lines.append("")
    lines.append("Evaluator deltas:")

    for evaluator_name, delta in sorted(
        comparison.evaluator_deltas.items()
    ):
        lines.append(f"  {evaluator_name}: {delta:+.3f}")

    return "\n".join(lines)

if __name__ == "__main__":
    raise SystemExit(main())
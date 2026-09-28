import argparse

from aieval.gate import RegressionGate
from aieval.regression import RegressionDetector
from aieval.reporting.json import JsonReporter
from aieval.comparison import compare_runs


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

        print(gate_result.status)
        print(gate_result.reason)

        return gate.exit_code(gate_result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
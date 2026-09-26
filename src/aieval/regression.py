from dataclasses import dataclass

from aieval.comparison import ComparisonResult


@dataclass
class RegressionResult:
    regressed: bool
    score_regression: bool
    evaluator_regressions: list[str]


class RegressionDetector:
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def check(self, comparison: ComparisonResult) -> RegressionResult:
        score_regression = comparison.score_delta < -self.threshold

        evaluator_regressions = [
            evaluator_name
            for evaluator_name, delta in comparison.evaluator_deltas.items()
            if delta < -self.threshold
        ]

        return RegressionResult(
            regressed=score_regression or bool(evaluator_regressions),
            score_regression=score_regression,
            evaluator_regressions=evaluator_regressions,
        )
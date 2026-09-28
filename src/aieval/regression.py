from dataclasses import dataclass

from aieval.comparison import ComparisonResult


@dataclass
class RegressionResult:
    regressed: bool
    score_regression: bool
    evaluator_regressions: list[str]


class RegressionDetector:
    def __init__(
        self,
        threshold: float = 0.0,
        evaluator_thresholds: dict[str, float] | None = None,
    ):
        self.threshold = threshold
        self.evaluator_thresholds = evaluator_thresholds or {}

    def check(self, comparison: ComparisonResult) -> RegressionResult:
        score_regression = comparison.score_delta < -self.threshold

        evaluator_regressions = [
            evaluator_name
            for evaluator_name, delta in comparison.evaluator_deltas.items()
            if delta
            < -self.evaluator_thresholds.get(evaluator_name, self.threshold)
        ]

        return RegressionResult(
            regressed=score_regression or bool(evaluator_regressions),
            score_regression=score_regression,
            evaluator_regressions=evaluator_regressions,
        )
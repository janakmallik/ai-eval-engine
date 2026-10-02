from dataclasses import dataclass, field

from aieval.comparison import ComparisonResult


@dataclass
class RegressionResult:
    regressed: bool
    score_regression: bool
    evaluator_regressions: list[str]


@dataclass
class RegressionConfig:
    threshold: float = 0.0
    evaluator_thresholds: dict[str, float] = field(default_factory=dict)


class RegressionDetector:
    def __init__(self, config: RegressionConfig | None = None):
        self.config = config or RegressionConfig()

    def check(self, comparison: ComparisonResult) -> RegressionResult:
        score_regression = comparison.score_delta < -self.config.threshold

        evaluator_regressions = [
            evaluator_name
            for evaluator_name, delta in comparison.evaluator_deltas.items()
            if delta
            < -self.config.evaluator_thresholds.get(
                evaluator_name,
                self.config.threshold,
            )
        ]

        return RegressionResult(
            regressed=score_regression or bool(evaluator_regressions),
            score_regression=score_regression,
            evaluator_regressions=evaluator_regressions,
        )

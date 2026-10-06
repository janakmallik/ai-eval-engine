from dataclasses import dataclass, field

from aieval.comparison import ComparisonResult


@dataclass
class RegressionResult:
    regressed: bool
    score_regression: bool
    evaluator_regressions: list[str]
    latency_regression: bool = False
    cost_regression: bool = False
    error_rate_regression: bool = False


@dataclass
class RegressionConfig:
    threshold: float = 0.0
    evaluator_thresholds: dict[str, float] = field(default_factory=dict)
    latency_threshold: float = 0.0
    cost_threshold: float = 0.0
    error_rate_threshold: float = 0.0


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

        latency_regression = comparison.latency_delta > self.config.latency_threshold
        cost_regression = comparison.cost_delta > self.config.cost_threshold
        error_rate_regression = (
            comparison.error_rate_delta > self.config.error_rate_threshold
        )

        return RegressionResult(
            regressed=(
                score_regression
                or bool(evaluator_regressions)
                or latency_regression
                or cost_regression
                or error_rate_regression
            ),
            score_regression=score_regression,
            evaluator_regressions=evaluator_regressions,
            latency_regression=latency_regression,
            cost_regression=cost_regression,
            error_rate_regression=error_rate_regression,
        )

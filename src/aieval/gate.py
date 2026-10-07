from dataclasses import dataclass

from aieval.regression import RegressionResult


@dataclass
class GateResult:
    passed: bool
    status: str
    reason: str


class RegressionGate:
    def check(self, result: RegressionResult) -> GateResult:
        if result.regressed:
            reasons = []

            if result.score_regression:
                reasons.append("quality")

            if result.evaluator_regressions:
                reasons.append("evaluator")

            if result.latency_regression:
                reasons.append("latency")

            if result.cost_regression:
                reasons.append("cost")

            if result.error_rate_regression:
                reasons.append("error rate")

            return GateResult(
                passed=False,
                status="failed",
                reason=f"Regression detected: {', '.join(reasons)}",
            )

        return GateResult(
            passed=True,
            status="passed",
            reason="No regression detected",
        )

    def passed(self, result: RegressionResult) -> bool:
        return self.check(result).passed

    def exit_code(self, gate_result: GateResult) -> int:
        return 0 if gate_result.passed else 1

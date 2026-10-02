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
            return GateResult(
                passed=False,
                status="failed",
                reason="Regression detected",
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

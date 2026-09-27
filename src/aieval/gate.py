from aieval.regression import RegressionResult


class RegressionGate:
    def passed(self, result: RegressionResult) -> bool:
        return not result.regressed
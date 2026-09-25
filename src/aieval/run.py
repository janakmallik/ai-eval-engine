from dataclasses import dataclass

from aieval.result import EvaluationResult


@dataclass
class EvaluationRun:
    results: list[EvaluationResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def score(self) -> float:
        if self.total == 0:
            return 0.0

        return sum(result.score for result in self.results) / self.total

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0

        return self.passed / self.total

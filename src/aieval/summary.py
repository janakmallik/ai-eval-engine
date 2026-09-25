from dataclasses import dataclass


@dataclass
class EvaluationSummary:
    evaluator_name: str
    total: int
    passed: int
    failed: int
    score: float
    pass_rate: float

    def to_dict(self) -> dict:
        return {
            "evaluator_name": self.evaluator_name,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "score": self.score,
            "pass_rate": self.pass_rate,
        }

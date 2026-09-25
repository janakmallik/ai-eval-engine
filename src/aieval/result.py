from dataclasses import dataclass


@dataclass
class EvaluationResult:
    case_id: str
    evaluator_name: str
    expected: str
    actual: str
    score: float
    passed: bool

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "evaluator_name": self.evaluator_name,
            "expected": self.expected,
            "actual": self.actual,
            "score": self.score,
            "passed": self.passed,
        }

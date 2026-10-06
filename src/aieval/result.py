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

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationResult":
        return cls(
            case_id=data["case_id"],
            evaluator_name=data["evaluator_name"],
            expected=data["expected"],
            actual=data["actual"],
            score=data["score"],
            passed=data["passed"],
        )

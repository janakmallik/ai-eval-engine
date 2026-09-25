from dataclasses import dataclass


@dataclass
class EvaluationResult:
    case_id: str
    evaluator_name: str
    expected: str
    actual: str
    score: float
    passed: bool

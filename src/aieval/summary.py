from dataclasses import dataclass


@dataclass
class EvaluationSummary:
    evaluator_name: str
    total: int
    passed: int
    failed: int
    score: float
    pass_rate: float

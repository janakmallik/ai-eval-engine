from typing import Protocol

from aieval.result import EvaluationResult


class Evaluator(Protocol):

    def evaluate(
        self,
        case_id: str,
        expected: str,
        actual: str,
    ) -> EvaluationResult:
        ...

from typing import Protocol

from aieval.context import EvaluationContext
from aieval.result import EvaluationResult


class Evaluator(Protocol):

    def evaluate(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:
        ...

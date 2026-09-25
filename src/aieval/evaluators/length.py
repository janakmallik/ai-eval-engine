from aieval.context import EvaluationContext
from aieval.evaluators.base import Evaluator
from aieval.result import EvaluationResult


class LengthEvaluator(Evaluator):

    def __init__(
        self,
        max_length: int,
    ) -> None:
        self.max_length = max_length

    def evaluate(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:

        actual_length = len(context.actual)

        passed = actual_length <= self.max_length

        score = 1.0 if passed else 0.0

        return EvaluationResult(
            case_id=context.case.id,
            evaluator_name="length",
            expected=str(self.max_length),
            actual=str(actual_length),
            score=score,
            passed=passed,
        )

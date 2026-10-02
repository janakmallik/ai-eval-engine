from aieval.context import EvaluationContext
from aieval.result import EvaluationResult


class ContainsEvaluator:

    name = "contains"

    def evaluate(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:

        expected = context.case.expected
        actual = context.actual

        passed = expected in actual
        score = float(passed)

        return EvaluationResult(
            case_id=context.case.id,
            evaluator_name=self.name,
            expected=context.case.expected,
            actual=context.actual,
            score=score,
            passed=passed,
        )

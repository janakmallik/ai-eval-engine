from aieval.context import EvaluationContext
from aieval.result import EvaluationResult


class ContainsEvaluator:

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
            expected=context.case.expected,
            actual=context.actual,
            score=score,
            passed=passed,
        )

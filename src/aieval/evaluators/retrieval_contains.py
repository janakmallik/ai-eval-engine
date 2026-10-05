from aieval.context import EvaluationContext
from aieval.result import EvaluationResult


class RetrievalContainsEvaluator:
    name = "retrieval_contains"

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        expected = context.case.expected

        retrieved = context.retrieved or []

        passed = any(
            expected.lower() in str(document).lower() for document in retrieved
        )

        return EvaluationResult(
            case_id=context.case.id,
            evaluator_name=self.name,
            expected=expected,
            actual=retrieved,
            score=1.0 if passed else 0.0,
            passed=passed,
        )

from aieval.context import EvaluationContext
from aieval.result import EvaluationResult


class RetrievalRecallEvaluator:
    name = "retrieval_recall"

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        expected = context.case.expected
        retrieved = context.retrieved or []

        if not expected:
            score = 0.0
        else:
            matched = sum(
                any(
                    expected_document.lower() in str(document).lower()
                    for document in retrieved
                )
                for expected_document in expected
            )

            score = matched / len(expected)

        return EvaluationResult(
            case_id=context.case.id,
            evaluator_name=self.name,
            expected=expected,
            actual=retrieved,
            score=score,
            passed=score == 1.0,
        )

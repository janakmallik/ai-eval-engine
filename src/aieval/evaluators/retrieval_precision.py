import string

from aieval.context import EvaluationContext
from aieval.result import EvaluationResult


class RetrievalPrecisionEvaluator:
    name = "retrieval_precision"

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        expected = context.case.expected or []
        retrieved = context.retrieved or []

        if not retrieved:
            return EvaluationResult(
                case_id=context.case.id,
                evaluator_name=self.name,
                expected=expected,
                actual=retrieved,
                score=0.0,
                passed=False,
            )

        def normalize(value: object) -> str:
            return (
                str(value)
                .strip()
                .lower()
                .translate(str.maketrans("", "", string.punctuation))
            )

        expected_normalized = {normalize(item) for item in expected}

        relevant_count = sum(
            1 for document in retrieved if normalize(document) in expected_normalized
        )

        score = relevant_count / len(retrieved)

        return EvaluationResult(
            case_id=context.case.id,
            evaluator_name=self.name,
            expected=expected,
            actual=retrieved,
            score=score,
            passed=score == 1.0,
        )

from difflib import SequenceMatcher

from aieval.context import EvaluationContext
from aieval.normalizers import normalize_text
from aieval.result import EvaluationResult


class SimilarityEvaluator:

    def __init__(
        self,
        threshold: float = 0.8,
    ):
        self.threshold = threshold

    def evaluate(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:

        expected = context.case.expected
        actual = context.actual

        normalized_expected = normalize_text(expected)
        normalized_actual = normalize_text(actual)

        score = SequenceMatcher(
            None,
            normalized_expected,
            normalized_actual,
        ).ratio()

        return EvaluationResult(
            case_id=context.case.id,
            evaluator_name="similarity",
            expected=context.case.expected,
            actual=context.actual,
            score=score,
            passed=score >= self.threshold,
        )

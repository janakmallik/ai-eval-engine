from collections.abc import Callable

from aieval.context import EvaluationContext
from aieval.normalizers import normalize_text
from aieval.result import EvaluationResult


class ExactMatchEvaluator:

    def __init__(
        self,
        normalizer: Callable[[str], str] = normalize_text,
    ):
        self.normalizer = normalizer

    def evaluate(
        self,
        context: EvaluationContext,
    ) -> EvaluationResult:

        normalized_expected = self.normalizer(context.case.expected)

        normalized_actual = self.normalizer(context.actual)

        score = float(normalized_expected == normalized_actual)

        return EvaluationResult(
            case_id=context.case.id,
            expected=context.case.expected,
            actual=context.actual,
            score=score,
            passed=score == 1.0,
        )

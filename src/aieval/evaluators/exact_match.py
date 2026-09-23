from collections.abc import Callable

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
        case_id: str,
        expected: str,
        actual: str,
    ) -> EvaluationResult:

        normalized_expected = self.normalizer(expected)
        normalized_actual = self.normalizer(actual)

        score = float(normalized_expected == normalized_actual)

        return EvaluationResult(
            case_id=case_id,
            expected=expected,
            actual=actual,
            score=score,
            passed=score == 1.0,
        )

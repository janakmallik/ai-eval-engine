from difflib import SequenceMatcher

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
        case_id: str,
        expected: str,
        actual: str,
    ) -> EvaluationResult:

        normalized_expected = normalize_text(expected)
        normalized_actual = normalize_text(actual)

        score = SequenceMatcher(
            None,
            normalized_expected,
            normalized_actual,
        ).ratio()

        return EvaluationResult(
            case_id=case_id,
            expected=expected,
            actual=actual,
            score=score,
            passed=score >= self.threshold,
        )

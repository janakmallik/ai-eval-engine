from aieval.normalizers import normalize_text
from aieval.result import EvaluationResult


class ExactMatchEvaluator:

    def evaluate(
        self,
        case_id: str,
        expected: str,
        actual: str,
    ) -> EvaluationResult:

        normalized_expected = normalize_text(expected)
        normalized_actual = normalize_text(actual)

        score = float(normalized_expected == normalized_actual)

        return EvaluationResult(
            case_id=case_id,
            expected=expected,
            actual=actual,
            score=score,
            passed=score == 1.0,
        )

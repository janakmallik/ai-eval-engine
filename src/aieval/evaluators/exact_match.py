from aieval.result import EvaluationResult


class ExactMatchEvaluator:

    def evaluate(
        self,
        case_id: str,
        expected: str,
        actual: str,
    ) -> EvaluationResult:

        score = float(expected == actual)

        return EvaluationResult(
            case_id=case_id,
            expected=expected,
            actual=actual,
            score=score,
            passed=score == 1.0,
        )

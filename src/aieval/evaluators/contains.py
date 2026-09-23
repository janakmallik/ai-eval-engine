from aieval.result import EvaluationResult


class ContainsEvaluator:

    def evaluate(
        self,
        case_id: str,
        expected: str,
        actual: str,
    ) -> EvaluationResult:

        passed = expected in actual
        score = float(passed)

        return EvaluationResult(
            case_id=case_id,
            expected=expected,
            actual=actual,
            score=score,
            passed=passed,
        )

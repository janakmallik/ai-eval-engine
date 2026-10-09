import re

from aieval import EvaluationResult


class FaithfulnessEvaluator:
    """Estimate answer grounding using normalized word overlap."""

    name = "faithfulness"

    def evaluate(self, context) -> EvaluationResult:
        answer_words = set(re.findall(r"\b\w+\b", context.actual.lower()))
        context_text = " ".join(context.retrieved or [])
        context_words = set(re.findall(r"\b\w+\b", context_text.lower()))

        if not answer_words:
            score = 0.0
        else:
            supported_words = answer_words & context_words
            score = len(supported_words) / len(answer_words)

        return EvaluationResult(
            case_id=context.case.id,
            evaluator_name=self.name,
            expected=context.case.expected,
            actual=context.actual,
            score=score,
            passed=score >= 0.8,
        )

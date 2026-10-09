from aieval.context import EvaluationContext
from aieval.result import EvaluationResult


class AnswerRelevanceEvaluator:
    name = "answer_relevance"

    STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "what",
        "where",
        "which",
        "who",
        "why",
        "with",
    }

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        question_words = {
            word.strip(".,?!")
            for word in context.case.input.lower().split()
            if word.strip(".,?!") not in self.STOP_WORDS
        }

        answer_words = {
            word.strip(".,?!") for word in str(context.actual).lower().split()
        }

        if not question_words:
            score = 0.0
        else:
            matched_words = question_words.intersection(answer_words)
            score = len(matched_words) / len(question_words)

        return EvaluationResult(
            case_id=context.case.id,
            evaluator_name=self.name,
            expected=context.case.expected,
            actual=context.actual,
            score=score,
            passed=score == 1.0,
        )

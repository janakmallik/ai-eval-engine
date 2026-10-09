from aieval.context import EvaluationContext
from aieval.result import EvaluationResult


class ContextRelevanceEvaluator:
    name = "context_relevance"

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

        retrieved = context.retrieved or []

        relevant_count = 0

        for document in retrieved:
            document_words = {
                word.strip(".,?!") for word in str(document).lower().split()
            }

            if question_words.intersection(document_words):
                relevant_count += 1

        score = relevant_count / len(retrieved) if retrieved else 0.0

        return EvaluationResult(
            case_id=context.case.id,
            evaluator_name=self.name,
            expected=context.case.expected,
            actual=retrieved,
            score=score,
            passed=score == 1.0,
        )

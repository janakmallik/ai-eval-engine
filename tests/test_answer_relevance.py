from aieval import EvalCase, EvaluationContext
from aieval.evaluators.answer_relevance import AnswerRelevanceEvaluator


def test_answer_relevance_passes_when_answer_contains_question_topic():
    context = EvaluationContext(
        case=EvalCase(
            id="1",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="The capital of France is Paris.",
        metadata={},
    )

    result = AnswerRelevanceEvaluator().evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_answer_relevance_fails_when_answer_is_irrelevant():
    context = EvaluationContext(
        case=EvalCase(
            id="2",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="The Pacific Ocean is the largest ocean on Earth.",
        metadata={},
    )

    result = AnswerRelevanceEvaluator().evaluate(context)

    assert result.score == 0.0
    assert result.passed is False


def test_answer_relevance_returns_partial_score():
    context = EvaluationContext(
        case=EvalCase(
            id="3",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="France has many beautiful cities and a large coastline.",
        metadata={},
    )

    result = AnswerRelevanceEvaluator().evaluate(context)

    assert result.score == 0.5
    assert result.passed is False

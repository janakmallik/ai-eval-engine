from aieval import EvalCase, EvaluationContext
from aieval.evaluators.context_relevance import ContextRelevanceEvaluator


def test_context_relevance_passes_when_context_matches_question():
    context = EvaluationContext(
        case=EvalCase(
            id="1",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="Paris",
        metadata={},
        retrieved=[
            "Paris is the capital of France.",
            "France is located in Europe.",
        ],
    )

    result = ContextRelevanceEvaluator().evaluate(context)

    assert result.score > 0
    assert result.passed is True


def test_context_relevance_fails_when_context_is_irrelevant():
    context = EvaluationContext(
        case=EvalCase(
            id="2",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="Paris",
        metadata={},
        retrieved=[
            "The Pacific Ocean is the largest ocean on Earth.",
            "Mount Everest is the highest mountain above sea level.",
        ],
    )

    result = ContextRelevanceEvaluator().evaluate(context)

    assert result.score == 0.0
    assert result.passed is False


def test_context_relevance_returns_partial_score():
    context = EvaluationContext(
        case=EvalCase(
            id="3",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="Paris",
        metadata={},
        retrieved=[
            "Paris is the capital of France.",
            "The Pacific Ocean is the largest ocean on Earth.",
        ],
    )

    result = ContextRelevanceEvaluator().evaluate(context)

    assert result.score == 0.5
    assert result.passed is False


def test_context_relevance_returns_zero_when_retrieved_is_empty():
    context = EvaluationContext(
        case=EvalCase(
            id="4",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="Paris",
        metadata={},
        retrieved=[],
    )

    result = ContextRelevanceEvaluator().evaluate(context)

    assert result.score == 0.0
    assert result.passed is False

from aieval.context import EvaluationContext
from aieval.dataset import EvalCase
from aieval.evaluators.exact_match import ExactMatchEvaluator
from aieval.evaluators.similarity import SimilarityEvaluator


# test 1
def test_exact_match_passes():

    evaluator = ExactMatchEvaluator()

    context = EvaluationContext(
        case=EvalCase(
            id="001",
            input="What is 2 + 2?",
            expected="4",
        ),
        actual="4",
    )

    result = evaluator.evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


# test 2
def test_exact_match_fails():

    evaluator = ExactMatchEvaluator()

    context = EvaluationContext(
        case=EvalCase(
            id="002",
            input="What is 2 + 2?",
            expected="4",
        ),
        actual="5",
    )

    result = evaluator.evaluate(context)

    assert result.score == 0.0
    assert result.passed is False


def test_exact_match_uses_normalization():

    evaluator = ExactMatchEvaluator()

    context = EvaluationContext(
        case=EvalCase(
            id="003",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual=" PARIS. ",
    )

    result = evaluator.evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_exact_match_supports_custom_normalizer():

    def remove_prefix(text: str) -> str:
        return text.removeprefix("Answer: ").strip()

    evaluator = ExactMatchEvaluator(
        normalizer=remove_prefix,
    )

    context = EvaluationContext(
        case=EvalCase(
            id="004",
            input="What is 2 + 2?",
            expected="4",
        ),
        actual="Answer: 4",
    )

    result = evaluator.evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_similarity_evaluator():

    evaluator = SimilarityEvaluator(
        threshold=0.8,
    )

    context = EvaluationContext(
        case=EvalCase(
            id="005",
            input="What is the capital of France?",
            expected="Paris is the capital of France.",
        ),
        actual="Paris is the capital city of France.",
    )

    result = evaluator.evaluate(context)

    assert result.score > 0.8
    assert result.passed is True


def test_similarity_evaluator_rejects_dissimilar_text():

    evaluator = SimilarityEvaluator(
        threshold=0.8,
    )

    context = EvaluationContext(
        case=EvalCase(
            id="006",
            input="What is the capital of France?",
            expected="Paris is the capital of France.",
        ),
        actual="Bananas are yellow.",
    )

    result = evaluator.evaluate(context)

    assert result.score < 0.8
    assert result.passed is False

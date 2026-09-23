from aieval.evaluators.exact_match import ExactMatchEvaluator
from aieval.evaluators.similarity import SimilarityEvaluator


def test_exact_match_passes():

    evaluator = ExactMatchEvaluator()

    result = evaluator.evaluate(
        case_id="001",
        expected="4",
        actual="4",
    )

    assert result.score == 1.0
    assert result.passed is True


def test_exact_match_fails():

    evaluator = ExactMatchEvaluator()

    result = evaluator.evaluate(
        case_id="002",
        expected="4",
        actual="5",
    )

    assert result.score == 0.0
    assert result.passed is False


def test_exact_match_uses_normalization():

    evaluator = ExactMatchEvaluator()

    result = evaluator.evaluate(
        case_id="003",
        expected="Paris",
        actual=" PARIS. ",
    )

    assert result.score == 1.0
    assert result.passed is True


def test_exact_match_supports_custom_normalizer():

    def remove_prefix(text: str) -> str:
        return text.removeprefix("Answer: ").strip()

    evaluator = ExactMatchEvaluator(normalizer=remove_prefix, )

    result = evaluator.evaluate(
        case_id="004",
        expected="4",
        actual="Answer: 4",
    )

    assert result.score == 1.0
    assert result.passed is True


def test_similarity_evaluator():

    evaluator = SimilarityEvaluator(threshold=0.8, )

    result = evaluator.evaluate(
        case_id="005",
        expected="Paris is the capital of France.",
        actual="Paris is the capital city of France.",
    )

    assert result.score > 0.8
    assert result.passed is True


def test_similarity_evaluator_rejects_dissimilar_text():

    evaluator = SimilarityEvaluator(threshold=0.8, )

    result = evaluator.evaluate(
        case_id="006",
        expected="Paris is the capital of France.",
        actual="Bananas are yellow.",
    )

    assert result.score < 0.8
    assert result.passed is False

from aieval.context import EvaluationContext
from aieval.dataset import EvalCase
from aieval.evaluators.retrieval_contains import RetrievalContainsEvaluator


def test_retrieval_contains_evaluator_passes_when_expected_is_in_retrieved():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected="optimization algorithm",
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "Gradient descent is an optimization algorithm.",
            "It is commonly used to train machine learning models.",
        ],
    )

    evaluator = RetrievalContainsEvaluator()

    result = evaluator.evaluate(context)

    assert result.case_id == "1"
    assert result.evaluator_name == "retrieval_contains"
    assert result.score == 1.0
    assert result.passed is True


def test_retrieval_contains_evaluator_fails_when_expected_is_not_in_retrieved():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected="optimization algorithm",
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "Python is a programming language.",
            "Neural networks can learn complex patterns.",
        ],
    )

    evaluator = RetrievalContainsEvaluator()

    result = evaluator.evaluate(context)

    assert result.case_id == "1"
    assert result.evaluator_name == "retrieval_contains"
    assert result.score == 0.0
    assert result.passed is False


def test_retrieval_contains_evaluator_fails_when_retrieved_is_empty():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected="optimization algorithm",
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[],
    )

    evaluator = RetrievalContainsEvaluator()

    result = evaluator.evaluate(context)

    assert result.case_id == "1"
    assert result.evaluator_name == "retrieval_contains"
    assert result.score == 0.0
    assert result.passed is False


def test_retrieval_contains_evaluator_fails_when_retrieved_is_none():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected="optimization algorithm",
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=None,
    )

    evaluator = RetrievalContainsEvaluator()

    result = evaluator.evaluate(context)

    assert result.case_id == "1"
    assert result.evaluator_name == "retrieval_contains"
    assert result.score == 0.0
    assert result.passed is False


def test_retrieval_contains_evaluator_is_case_insensitive():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected="optimization algorithm",
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "Gradient descent is an Optimization Algorithm.",
        ],
    )

    evaluator = RetrievalContainsEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_retrieval_contains_evaluator_finds_match_in_any_document():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected="optimization algorithm",
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "Python is a programming language.",
            "Neural networks can learn complex patterns.",
            "Gradient descent is an Optimization Algorithm.",
        ],
    )

    evaluator = RetrievalContainsEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_retrieval_contains_evaluator_ignores_surrounding_whitespace():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected="optimization algorithm",
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "   Optimization Algorithm   ",
        ],
    )

    evaluator = RetrievalContainsEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_retrieval_contains_evaluator_ignores_punctuation():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected="optimization algorithm",
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "Optimization Algorithm!",
        ],
    )

    evaluator = RetrievalContainsEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 1.0
    assert result.passed is True

from aieval.context import EvaluationContext
from aieval.dataset import EvalCase
from aieval.evaluators.retrieval_contains import RetrievalContainsEvaluator
from aieval.evaluators.retrieval_recall import RetrievalRecallEvaluator
from aieval.result import EvaluationResult


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


def test_retrieval_recall_evaluator_returns_full_recall_when_all_expected_documents_are_retrieved():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected=[
            "Gradient descent is an optimization algorithm.",
            "It is used to optimize model parameters.",
        ],
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "Gradient descent is an optimization algorithm.",
            "It is used to optimize model parameters.",
        ],
    )

    evaluator = RetrievalRecallEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_retrieval_recall_evaluator_returns_partial_recall():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected=[
            "Gradient descent is an optimization algorithm.",
            "It is used to optimize model parameters.",
        ],
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "Gradient descent is an optimization algorithm.",
        ],
    )

    evaluator = RetrievalRecallEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 0.5
    assert result.passed is False


def test_retrieval_recall_evaluator_returns_zero_when_no_expected_documents_are_retrieved():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected=[
            "Gradient descent is an optimization algorithm.",
            "It is used to optimize model parameters.",
        ],
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "Python is a programming language.",
        ],
    )

    evaluator = RetrievalRecallEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 0.0
    assert result.passed is False


def test_retrieval_recall_evaluator_returns_zero_when_expected_documents_are_empty():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected=[],
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "Gradient descent is an optimization algorithm.",
        ],
    )

    evaluator = RetrievalRecallEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 0.0
    assert result.passed is False


def test_retrieval_recall_evaluator_is_order_independent():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected=[
            "Gradient descent is an optimization algorithm.",
            "It is used to optimize model parameters.",
        ],
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "It is used to optimize model parameters.",
            "Gradient descent is an optimization algorithm.",
        ],
    )

    evaluator = RetrievalRecallEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_retrieval_recall_evaluator_is_case_insensitive():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected=[
            "optimization algorithm",
            "model parameters",
        ],
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "Gradient descent is an Optimization Algorithm.",
            "It is used to optimize MODEL PARAMETERS.",
        ],
    )

    evaluator = RetrievalRecallEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_retrieval_recall_evaluator_returns_partial_score():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected=[
            "optimization algorithm",
            "model parameters",
        ],
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "Gradient descent is an optimization algorithm.",
        ],
    )

    evaluator = RetrievalRecallEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 0.5
    assert result.passed is False


from aieval.evaluators.retrieval_precision import RetrievalPrecisionEvaluator


def test_retrieval_precision_evaluator_returns_full_precision_when_all_retrieved_documents_are_relevant():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected=[
            "Gradient descent is an optimization algorithm.",
            "It is used to optimize model parameters.",
        ],
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "Gradient descent is an optimization algorithm.",
            "It is used to optimize model parameters.",
        ],
    )

    evaluator = RetrievalPrecisionEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_retrieval_precision_evaluator_returns_partial_precision_when_some_retrieved_documents_are_irrelevant():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected=[
            "Gradient descent is an optimization algorithm.",
            "It is used to optimize model parameters.",
        ],
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "Gradient descent is an optimization algorithm.",
            "Python is a programming language.",
        ],
    )

    evaluator = RetrievalPrecisionEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 0.5
    assert result.passed is False


def test_retrieval_precision_evaluator_returns_zero_when_no_retrieved_documents_are_relevant():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected=[
            "Gradient descent is an optimization algorithm.",
            "It is used to optimize model parameters.",
        ],
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "Python is a programming language.",
            "Neural networks can learn complex patterns.",
        ],
    )

    evaluator = RetrievalPrecisionEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 0.0
    assert result.passed is False


def test_retrieval_precision_evaluator_is_case_insensitive():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected=[
            "Gradient descent is an optimization algorithm.",
            "It is used to optimize model parameters.",
        ],
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "gradient descent is an OPTIMIZATION ALGORITHM.",
            "IT IS USED TO OPTIMIZE MODEL PARAMETERS.",
        ],
    )

    evaluator = RetrievalPrecisionEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_retrieval_precision_evaluator_ignores_surrounding_whitespace():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected=[
            "Gradient descent is an optimization algorithm.",
            "It is used to optimize model parameters.",
        ],
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "  Gradient descent is an optimization algorithm.  ",
            " It is used to optimize model parameters. ",
        ],
    )

    evaluator = RetrievalPrecisionEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_retrieval_precision_evaluator_fails_when_retrieved_is_empty():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected=[
            "Gradient descent is an optimization algorithm.",
            "It is used to optimize model parameters.",
        ],
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[],
    )

    evaluator = RetrievalPrecisionEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 0.0
    assert result.passed is False


def test_retrieval_precision_evaluator_fails_when_retrieved_is_none():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected=[
            "Gradient descent is an optimization algorithm.",
            "It is used to optimize model parameters.",
        ],
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=None,
    )

    evaluator = RetrievalPrecisionEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 0.0
    assert result.passed is False


def test_retrieval_precision_evaluator_counts_duplicate_retrieved_documents():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected=[
            "Gradient descent is an optimization algorithm.",
        ],
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "Gradient descent is an optimization algorithm.",
            "Gradient descent is an optimization algorithm.",
            "Python is a programming language.",
        ],
    )

    evaluator = RetrievalPrecisionEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 2 / 3
    assert result.passed is False


def test_retrieval_precision_evaluator_ignores_punctuation():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected=[
            "Gradient descent is an optimization algorithm.",
        ],
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "Gradient descent is an optimization algorithm!",
        ],
    )

    evaluator = RetrievalPrecisionEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_retrieval_precision_evaluator_returns_correct_fractional_score():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected=[
            "Gradient descent is an optimization algorithm.",
            "It is used to optimize model parameters.",
        ],
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "Gradient descent is an optimization algorithm.",
            "It is used to optimize model parameters.",
            "Python is a programming language.",
        ],
    )

    evaluator = RetrievalPrecisionEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 2 / 3
    assert result.passed is False


def test_retrieval_precision_evaluator_returns_zero_when_all_retrieved_documents_are_irrelevant():
    case = EvalCase(
        id="1",
        input="What is gradient descent?",
        expected=[
            "Gradient descent is an optimization algorithm.",
            "It is used to optimize model parameters.",
        ],
    )

    context = EvaluationContext(
        case=case,
        actual="Gradient descent",
        retrieved=[
            "Python is a programming language.",
            "Neural networks can learn complex patterns.",
            "Databases store structured information.",
        ],
    )

    evaluator = RetrievalPrecisionEvaluator()

    result = evaluator.evaluate(context)

    assert result.score == 0.0
    assert result.passed is False

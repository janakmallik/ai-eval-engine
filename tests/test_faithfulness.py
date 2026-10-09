from aieval import EvalCase, EvaluationContext
from aieval.evaluators.faithfulness import FaithfulnessEvaluator


def test_faithfulness_passes_when_answer_is_supported_by_context():
    context = EvaluationContext(
        case=EvalCase(
            id="1",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="Paris is the capital of France.",
        metadata={},
        retrieved=[
            "Paris is the capital and largest city of France.",
        ],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_faithfulness_fails_when_answer_is_not_supported_by_context():
    context = EvaluationContext(
        case=EvalCase(
            id="2",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="Paris is the capital of France. It has a population of 50 million.",
        metadata={},
        retrieved=[
            "Paris is the capital of France.",
        ],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score < 1.0
    assert result.passed is False


def test_faithfulness_fails_when_context_is_missing():
    context = EvaluationContext(
        case=EvalCase(
            id="3",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="Paris is the capital of France.",
        metadata={},
        retrieved=[],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score == 0.0
    assert result.passed is False


def test_faithfulness_handles_empty_answer():
    context = EvaluationContext(
        case=EvalCase(
            id="4",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="",
        metadata={},
        retrieved=["Paris is the capital of France."],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score == 0.0
    assert result.passed is False


def test_faithfulness_fails_when_answer_is_empty():
    context = EvaluationContext(
        case=EvalCase(
            id="4",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="",
        metadata={},
        retrieved=["Paris is the capital of France."],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score == 0.0
    assert result.passed is False


def test_faithfulness_fails_when_retrieved_context_is_none():
    context = EvaluationContext(
        case=EvalCase(
            id="6",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="Paris is the capital of France.",
        metadata={},
        retrieved=None,
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score == 0.0
    assert result.passed is False


def test_faithfulness_passes_when_all_answer_words_are_supported():
    context = EvaluationContext(
        case=EvalCase(
            id="7",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="Paris is the capital of France.",
        metadata={},
        retrieved=["Paris is the capital of France."],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_faithfulness_score_is_zero_when_no_answer_words_are_supported():
    context = EvaluationContext(
        case=EvalCase(
            id="8",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="Tokyo has a large population.",
        metadata={},
        retrieved=["Paris is the capital of France."],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score == 0.0
    assert result.passed is False


def test_faithfulness_passes_at_threshold():
    context = EvaluationContext(
        case=EvalCase(
            id="9",
            input="Give a short fact about Paris.",
            expected="Paris is a city.",
        ),
        actual="Paris is bright",
        metadata={},
        retrieved=["Paris is bright"],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_faithfulness_score_stays_between_zero_and_one():
    context = EvaluationContext(
        case=EvalCase(
            id="10",
            input="Describe Paris.",
            expected="Paris is a city.",
        ),
        actual="Paris is a beautiful city with many museums.",
        metadata={},
        retrieved=["Paris is a city."],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert 0.0 <= result.score <= 1.0


def test_faithfulness_fails_below_threshold():
    context = EvaluationContext(
        case=EvalCase(
            id="11",
            input="Describe Paris.",
            expected="Paris is a city.",
        ),
        actual="Paris is a city with museums.",
        metadata={},
        retrieved=["Paris is a city."],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score < 0.8
    assert result.passed is False


def test_faithfulness_passes_exactly_at_threshold():
    context = EvaluationContext(
        case=EvalCase(
            id="12",
            input="Describe Paris.",
            expected="Paris is a city.",
        ),
        actual="Paris is a city museums",
        metadata={},
        retrieved=["Paris is a city"],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score == 0.8
    assert result.passed is True


def test_faithfulness_passes_above_threshold():
    context = EvaluationContext(
        case=EvalCase(
            id="13",
            input="Describe Paris.",
            expected="Paris is a city.",
        ),
        actual="Paris is a city.",
        metadata={},
        retrieved=["Paris is a city."],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score > 0.8
    assert result.passed is True


def test_faithfulness_matching_is_case_insensitive():
    context = EvaluationContext(
        case=EvalCase(
            id="14",
            input="Describe Paris.",
            expected="Paris is a city.",
        ),
        actual="PARIS IS A CITY.",
        metadata={},
        retrieved=["Paris is a city."],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_faithfulness_handles_punctuation_differences():
    context = EvaluationContext(
        case=EvalCase(
            id="15",
            input="Describe Paris.",
            expected="Paris is a city.",
        ),
        actual="Paris is a city.",
        metadata={},
        retrieved=["Paris is a city"],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_faithfulness_handles_repeated_words():
    context = EvaluationContext(
        case=EvalCase(
            id="16",
            input="Describe Paris.",
            expected="Paris is a city.",
        ),
        actual="Paris Paris is a city",
        metadata={},
        retrieved=["Paris is a city"],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_faithfulness_handles_whitespace_only_answer():
    context = EvaluationContext(
        case=EvalCase(
            id="17",
            input="Describe Paris.",
            expected="Paris is a city.",
        ),
        actual="   \n\t ",
        metadata={},
        retrieved=["Paris is a city."],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score == 0.0
    assert result.passed is False


def test_faithfulness_has_expected_name():
    assert FaithfulnessEvaluator.name == "faithfulness"


def test_faithfulness_fails_with_empty_retrieved_document():
    context = EvaluationContext(
        case=EvalCase(
            id="19",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="Paris is the capital of France.",
        metadata={},
        retrieved=[""],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score == 0.0
    assert result.passed is False


def test_faithfulness_fails_with_irrelevant_context():
    context = EvaluationContext(
        case=EvalCase(
            id="20",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="Paris is the capital of France.",
        metadata={},
        retrieved=["Bananas grow in tropical climates."],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score < 0.8
    assert result.passed is False


def test_faithfulness_matches_mixed_case_and_punctuation():
    context = EvaluationContext(
        case=EvalCase(
            id="21",
            input="Describe Paris.",
            expected="Paris is a city.",
        ),
        actual="PARIS IS A CITY!",
        metadata={},
        retrieved=["Paris is a city."],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_faithfulness_fails_for_punctuation_only_answer():
    context = EvaluationContext(
        case=EvalCase(
            id="22",
            input="Describe Paris.",
            expected="Paris is a city.",
        ),
        actual="...!!!",
        metadata={},
        retrieved=["Paris is a city."],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score == 0.0
    assert result.passed is False


def test_faithfulness_score_is_bounded_with_repeated_words():
    context = EvaluationContext(
        case=EvalCase(
            id="23",
            input="Describe Paris.",
            expected="Paris is a city.",
        ),
        actual="Paris Paris Paris is a city.",
        metadata={},
        retrieved=["Paris is a city."],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert 0.0 <= result.score <= 1.0


def test_faithfulness_combines_multiple_retrieved_documents():
    context = EvaluationContext(
        case=EvalCase(
            id="24",
            input="Describe Paris.",
            expected="Paris is the capital of France.",
        ),
        actual="Paris is the capital of France.",
        metadata={},
        retrieved=[
            "Paris is the capital",
            "of France.",
        ],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_faithfulness_calculates_partial_support_correctly():
    context = EvaluationContext(
        case=EvalCase(
            id="25",
            input="Describe Paris.",
            expected="Paris is a city.",
        ),
        actual="Paris is a city museums",
        metadata={},
        retrieved=["Paris is a city"],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score == 0.8
    assert result.passed is True


def test_faithfulness_irrelevant_extra_document_does_not_reduce_score():
    context = EvaluationContext(
        case=EvalCase(
            id="26",
            input="What is the capital of France?",
            expected="Paris is the capital of France.",
        ),
        actual="Paris is the capital of France.",
        metadata={},
        retrieved=[
            "Paris is the capital of France.",
            "Bananas grow in tropical climates.",
        ],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score == 1.0
    assert result.passed is True


def test_faithfulness_fails_when_all_retrieved_documents_are_blank():
    context = EvaluationContext(
        case=EvalCase(
            id="27",
            input="What is the capital of France?",
            expected="Paris",
        ),
        actual="Paris is the capital of France.",
        metadata={},
        retrieved=["", "   "],
    )

    result = FaithfulnessEvaluator().evaluate(context)

    assert result.score == 0.0
    assert result.passed is False

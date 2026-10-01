import pytest

from aieval.dataset import EvalCase, EvalDataset
from aieval.evaluators.contains import ContainsEvaluator
from aieval.evaluators.exact_match import ExactMatchEvaluator
from aieval.runner import evaluate_dataset
from aieval.evaluators.length import LengthEvaluator

def test_evaluate_dataset():

    dataset = [
        EvalCase(
            id="001",
            input="What is 2 + 2?",
            expected="4",
        ),
        EvalCase(
            id="002",
            input="What is 3 + 3?",
            expected="6",
        ),
    ]

    def fake_model(question: str) -> str:
        answers = {
            "What is 2 + 2?": "4",
            "What is 3 + 3?": "6",
        }

        return answers[question]

    results = evaluate_dataset(
        model=fake_model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
    )

    assert results.total == 2
    assert results.passed == 2
    assert results.failed == 0
    assert results.score == 1.0

    assert results.results[0].score == 1.0
    assert results.results[0].passed is True

    assert results.results[1].score == 1.0
    assert results.results[1].passed is True


def test_runner_with_contains_evaluator():

    dataset = [
        EvalCase(
            id="001",
            input="What is the capital of France?",
            expected="Paris",
        ),
        EvalCase(
            id="002",
            input="Tell me about Paris.",
            expected="Paris",
        ),
    ]

    def fake_model(question: str) -> str:
        answers = {
            "What is the capital of France?": "Paris is the capital of France.",
            "Tell me about Paris.": "Paris is a city in France.",
        }

        return answers[question]

    run = evaluate_dataset(
        model=fake_model,
        dataset=dataset,
        evaluators=[ContainsEvaluator()],
    )

    assert run.total == 2
    assert run.passed == 2
    assert run.failed == 0
    assert run.score == 1.0


def test_runner_accepts_different_evaluator():

    dataset = [
        EvalCase(
            id="001",
            input="What is the capital of France?",
            expected="Paris",
        ),
    ]

    def fake_model(question: str) -> str:
        return "Paris is the capital of France."

    run = evaluate_dataset(
        model=fake_model,
        dataset=dataset,
        evaluators=[ContainsEvaluator()],
    )

    assert run.total == 1
    assert run.passed == 1
    assert run.failed == 0
    assert run.score == 1.0

def test_runner_accepts_multiple_evaluators():

    dataset = [
        EvalCase(
            id="011",
            input="What is the capital of France?",
            expected="Paris",
        ),
    ]

    def fake_model(question: str) -> str:
        return "Paris is the capital of France."

    run = evaluate_dataset(
        model=fake_model,
        dataset=dataset,
        evaluators=[
            ContainsEvaluator(),
            LengthEvaluator(max_length=100),
        ],
    )

    assert run.total == 2
    assert run.passed == 2
    assert run.failed == 0

def test_runner_attaches_metadata():

    dataset = [
        EvalCase(
            id="001",
            input="What is 2 + 2?",
            expected="4",
        ),
    ]

    def fake_model(question: str) -> str:
        return "4"

    metadata = {
        "model": "test-model",
        "dataset": "math-v1",
    }

    run = evaluate_dataset(
        model=fake_model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        metadata=metadata,
    )

    assert run.metadata == metadata

def test_runner_defaults_to_empty_metadata():

    dataset = [
        EvalCase(
            id="001",
            input="What is 2 + 2?",
            expected="4",
        ),
    ]

    def fake_model(question: str) -> str:
        return "4"

    run = evaluate_dataset(
        model=fake_model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
    )

    assert run.metadata == {}

def test_runner_accepts_eval_dataset():

    dataset = EvalDataset(
        cases=[
            EvalCase(
                id="001",
                input="What is 2 + 2?",
                expected="4",
            ),
            EvalCase(
                id="002",
                input="What is 3 + 3?",
                expected="6",
            ),
        ]
    )

    def fake_model(question: str) -> str:
        answers = {
            "What is 2 + 2?": "4",
            "What is 3 + 3?": "6",
        }

        return answers[question]

    run = evaluate_dataset(
        model=fake_model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
    )

    assert run.total == 2
    assert run.passed == 2
    assert run.failed == 0
    assert run.score == 1.0

# don't want to suddenly make every evaluation create traces. That's an architectural
# decision we'll test explicitly.
def test_evaluate_dataset_can_enable_tracing():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        return text

    evaluator = ExactMatchEvaluator()

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[evaluator],
        enable_tracing=True,
    )

    assert run.trace is not None

def test_evaluate_dataset_tracing_defaults_to_disabled():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        return text

    evaluator = ExactMatchEvaluator()

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[evaluator],
    )

    assert run.trace is None

def test_evaluate_dataset_creates_root_trace_span():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        return text

    evaluator = ExactMatchEvaluator()

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[evaluator],
        enable_tracing=True,
    )

    assert run.trace is not None
    root_span = run.trace.spans[0]

    assert root_span.name == "evaluation"
    assert root_span.parent_span_id is None

def test_evaluate_dataset_ends_root_trace_span():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        return text

    evaluator = ExactMatchEvaluator()

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[evaluator],
        enable_tracing=True,
    )

    assert run.trace is not None
    root_span = run.trace.spans[0]

    assert root_span.name == "evaluation"
    assert root_span.ended_at is not None

def test_evaluate_dataset_creates_case_span():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        return text

    evaluator = ExactMatchEvaluator()

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[evaluator],
        enable_tracing=True,
    )

    assert run.trace is not None
    case_span = run.trace.spans[1]
    root_span = run.trace.spans[0]

    assert case_span.name == "evaluation.case"
    assert case_span.parent_span_id == root_span.span_id

def test_evaluate_dataset_ends_case_span():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        return text

    evaluator = ExactMatchEvaluator()

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[evaluator],
        enable_tracing=True,
    )

    assert run.trace is not None
    case_span = run.trace.spans[1]

    assert case_span.ended_at is not None

def test_evaluate_dataset_creates_model_span():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        return text

    evaluator = ExactMatchEvaluator()

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[evaluator],
        enable_tracing=True,
    )

    assert run.trace is not None
    model_span = run.trace.spans[2]
    case_span = run.trace.spans[1]

    assert model_span.name == "model"
    assert model_span.parent_span_id == case_span.span_id

def test_evaluate_dataset_ends_model_span():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        return text

    evaluator = ExactMatchEvaluator()

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[evaluator],
        enable_tracing=True,
    )

    assert run.trace is not None
    model_span = run.trace.spans[2]

    assert model_span.name == "model"
    assert model_span.ended_at is not None

def test_evaluate_dataset_model_exception_marks_model_span_error():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        raise ValueError("model failed")

    evaluator = ExactMatchEvaluator()

    try:
        evaluate_dataset(
            model=model,
            dataset=dataset,
            evaluators=[evaluator],
            enable_tracing=True,
        )
    except ValueError:
        pass

    # The current implementation does not return the run after
    # the exception, so this test will initially need us to decide
    # how failed runs expose their trace.

def test_evaluate_dataset_ends_case_span_after_all_evaluators():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        return text

    evaluators = [
        ExactMatchEvaluator(),
        ExactMatchEvaluator(),
    ]

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=evaluators,
        enable_tracing=True,
    )

    assert run.trace is not None
    case_span = run.trace.spans[1]

    assert case_span.ended_at is not None

def test_evaluate_dataset_model_exception_records_exception_event():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        raise ValueError("model failed")

    evaluator = ExactMatchEvaluator()

    try:
        evaluate_dataset(
            model=model,
            dataset=dataset,
            evaluators=[evaluator],
            enable_tracing=True,
        )
    except ValueError:
        pass

    # We cannot inspect the run because the exception propagates.

def test_evaluate_dataset_propagates_model_exception():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        raise ValueError("model failed")

    evaluator = ExactMatchEvaluator()

    with pytest.raises(ValueError, match="model failed"):
        evaluate_dataset(
            model=model,
            dataset=dataset,
            evaluators=[evaluator],
            enable_tracing=True,
        )

def test_evaluate_dataset_creates_evaluator_span():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        return text

    evaluator = ExactMatchEvaluator()

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[evaluator],
        enable_tracing=True,
    )

    assert run.trace is not None
    assert len(run.trace.spans) == 4

def test_evaluate_dataset_evaluator_exception_ends_evaluator_span():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        return text

    class FailingEvaluator:
        def evaluate(self, context):
            raise ValueError("evaluation failed")

    with pytest.raises(ValueError, match="evaluation failed"):
        evaluate_dataset(
            model=model,
            dataset=dataset,
            evaluators=[FailingEvaluator()],
            enable_tracing=True,
        )
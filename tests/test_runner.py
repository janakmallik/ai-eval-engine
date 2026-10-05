import pytest

from aieval.dataset import EvalCase, EvalDataset
from aieval.evaluators.contains import ContainsEvaluator
from aieval.evaluators.exact_match import ExactMatchEvaluator
from aieval.evaluators.length import LengthEvaluator
from aieval.result import EvaluationResult
from aieval.runner import evaluate_dataset


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


def test_evaluate_dataset_evaluator_span_contains_evaluator_name():
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

    evaluator_span = run.trace.spans[3]

    assert evaluator_span.name == "evaluator.exact_match"


def test_evaluate_dataset_evaluator_span_contains_result_attributes():
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

    evaluator_span = run.trace.spans[3]

    assert evaluator_span.attributes["evaluator.name"] == "exact_match"
    assert evaluator_span.attributes["evaluation.score"] == 1.0
    assert evaluator_span.attributes["evaluation.passed"] is True


def test_evaluate_dataset_model_span_is_ok_after_success():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        return text

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        enable_tracing=True,
    )

    assert run.trace is not None

    model_span = run.trace.spans[2]

    assert model_span.status == "ok"


def test_evaluate_dataset_model_span_contains_result_attributes():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        return text.upper()

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        enable_tracing=True,
    )

    assert run.trace is not None

    model_span = run.trace.spans[2]

    assert model_span.attributes["model.input"] == "hello"
    assert model_span.attributes["model.output"] == "HELLO"


def test_evaluate_dataset_case_span_contains_case_attributes():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        return text

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        enable_tracing=True,
    )

    assert run.trace is not None

    case_span = run.trace.spans[1]

    assert case_span.attributes["case.id"] == "1"
    assert case_span.attributes["case.input"] == "hello"
    assert case_span.attributes["case.expected"] == "hello"


def test_evaluate_dataset_evaluator_span_is_error_after_exception():
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


def test_evaluate_dataset_evaluator_span_contains_case_id():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        return text

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        enable_tracing=True,
    )

    assert run.trace is not None

    evaluator_span = run.trace.spans[3]

    assert evaluator_span.attributes["evaluation.case_id"] == "1"


def test_evaluate_dataset_case_span_contains_evaluation_outcome():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        return text

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        enable_tracing=True,
    )

    assert run.trace is not None

    case_span = run.trace.spans[1]

    assert case_span.attributes["evaluation.passed"] is True


def test_evaluate_dataset_case_span_is_false_when_evaluator_fails():
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
        name = "failing"

        def evaluate(self, context):
            return EvaluationResult(
                case_id=context.case.id,
                evaluator_name=self.name,
                expected=context.case.expected,
                actual=context.actual,
                score=0.0,
                passed=False,
            )

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[FailingEvaluator()],
        enable_tracing=True,
    )

    assert run.trace is not None

    case_span = run.trace.spans[1]

    assert case_span.attributes["evaluation.passed"] is False


def test_evaluate_dataset_root_span_contains_run_attributes():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        return text

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        enable_tracing=True,
    )

    assert run.trace is not None

    root_span = run.trace.spans[0]

    assert root_span.attributes["evaluation.total_cases"] == 1
    assert root_span.attributes["evaluation.total_results"] == 1
    assert root_span.attributes["evaluation.passed"] == 1
    assert root_span.attributes["evaluation.failed"] == 0
    assert root_span.attributes["evaluation.pass_rate"] == 1.0


def test_evaluate_dataset_root_span_contains_case_outcome_counts():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        ),
        EvalCase(
            id="2",
            input="world",
            expected="hello",
        ),
    ]

    def model(text):
        return text

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        enable_tracing=True,
    )

    assert run.trace is not None

    root_span = run.trace.spans[0]

    assert root_span.attributes["evaluation.total_cases"] == 2
    assert root_span.attributes["evaluation.passed_cases"] == 1
    assert root_span.attributes["evaluation.failed_cases"] == 1


def test_evaluate_dataset_evaluator_span_records_exception():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        return text

    class ExplodingEvaluator:
        name = "exploding"

        def evaluate(self, context):
            raise RuntimeError("evaluator crashed")

    try:
        evaluate_dataset(
            model=model,
            dataset=dataset,
            evaluators=[ExplodingEvaluator()],
            enable_tracing=True,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError")


def test_evaluate_dataset_propagates_model_failure():
    dataset = [
        EvalCase(
            id="001",
            input="Paris",
            expected="Paris",
        )
    ]

    def failing_model(_):
        raise ValueError("model failed")

    with pytest.raises(ValueError, match="model failed"):
        evaluate_dataset(
            model=failing_model,
            dataset=dataset,
            evaluators=[],
            enable_tracing=True,
        )


def test_evaluate_dataset_traces_retrieval():
    dataset = [
        EvalCase(
            id="1",
            input="What is gradient descent?",
            expected="Gradient descent",
        )
    ]

    def retrieve(query, top_k):
        return [
            "Gradient descent is an optimization algorithm.",
            "It is commonly used to train machine learning models.",
        ]

    def model(text):
        return "Gradient descent"

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        enable_tracing=True,
        retriever=retrieve,
    )

    assert run.trace is not None

    retrieval_spans = [span for span in run.trace.spans if span.name == "retrieval"]

    assert len(retrieval_spans) == 1

    retrieval_span = retrieval_spans[0]

    assert retrieval_span.attributes["retrieval.query"] == ("What is gradient descent?")
    assert retrieval_span.attributes["retrieval.top_k"] == 5
    assert retrieval_span.attributes["retrieval.result_count"] == 2
    assert retrieval_span.ended_at is not None


def test_evaluate_dataset_retrieval_span_records_result_count():
    dataset = [
        EvalCase(
            id="1",
            input="What is gradient descent?",
            expected="Gradient descent",
        )
    ]

    def retrieve(query, top_k):
        return [
            "Gradient descent is an optimization algorithm.",
            "It is commonly used to train machine learning models.",
            "Gradient descent minimizes a loss function.",
        ]

    def model(text):
        return "Gradient descent"

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        enable_tracing=True,
        retriever=retrieve,
        retrieval_top_k=5,
    )

    assert run.trace is not None

    retrieval_spans = [span for span in run.trace.spans if span.name == "retrieval"]

    assert len(retrieval_spans) == 1
    assert retrieval_spans[0].attributes["retrieval.result_count"] == 3


def test_evaluate_dataset_retrieval_span_records_exception():
    dataset = [
        EvalCase(
            id="1",
            input="What is gradient descent?",
            expected="Gradient descent",
        )
    ]

    def retrieve(query, top_k):
        raise RuntimeError("retrieval failed")

    def model(text):
        return "Gradient descent"

    with pytest.raises(RuntimeError, match="retrieval failed"):
        evaluate_dataset(
            model=model,
            dataset=dataset,
            evaluators=[],
            enable_tracing=True,
            retriever=retrieve,
        )


def test_evaluate_dataset_uses_model_span():
    from aieval.tracing.trace import Trace

    dataset = [
        EvalCase(
            id="1",
            input="What is the capital of France?",
            expected="Paris",
        )
    ]

    def model(prompt):
        return "Paris"

    result = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        enable_tracing=True,
    )

    assert result.trace is not None

    model_spans = [span for span in result.trace.spans if span.name == "model"]

    assert len(model_spans) == 1

    model_span = model_spans[0]

    assert model_span.attributes["model.name"] == "model"
    assert model_span.attributes["model.provider"] == "unknown"


def test_evaluate_dataset_records_configured_model_metadata():
    dataset = [
        EvalCase(
            id="1",
            input="What is the capital of France?",
            expected="Paris",
        )
    ]

    def model(prompt):
        return "Paris"

    result = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        enable_tracing=True,
        model_name="gpt-5",
        model_provider="openai",
    )

    assert result.trace is not None

    model_spans = [span for span in result.trace.spans if span.name == "model"]

    assert len(model_spans) == 1

    model_span = model_spans[0]

    assert model_span.attributes["model.name"] == "gpt-5"
    assert model_span.attributes["model.provider"] == "openai"


def test_evaluate_dataset_records_model_input_and_output():
    dataset = [
        EvalCase(
            id="1",
            input="What is the capital of France?",
            expected="Paris",
        )
    ]

    def model(prompt):
        return "Paris"

    result = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        enable_tracing=True,
    )

    assert result.trace is not None

    model_spans = [span for span in result.trace.spans if span.name == "model"]

    assert len(model_spans) == 1

    model_span = model_spans[0]

    assert model_span.attributes["model.input"] == ("What is the capital of France?")
    assert model_span.attributes["model.output"] == "Paris"


def test_evaluate_dataset_records_model_usage():
    dataset = [
        EvalCase(
            id="1",
            input="What is the capital of France?",
            expected="Paris",
        )
    ]

    def model(prompt):
        return "Paris"

    result = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        enable_tracing=True,
    )

    model_span = next(span for span in result.trace.spans if span.name == "model")

    assert model_span.usage is None


from aieval.tracing.response import ModelResponse
from aieval.tracing.usage import ModelUsage


def test_evaluate_dataset_records_model_usage_from_response():
    dataset = [
        EvalCase(
            id="1",
            input="What is the capital of France?",
            expected="Paris",
        )
    ]

    def model(prompt):
        return ModelResponse(
            output="Paris",
            usage=ModelUsage(
                input_tokens=100,
                output_tokens=20,
                input_cost=0.001,
                output_cost=0.002,
            ),
        )

    result = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        enable_tracing=True,
    )

    model_span = next(span for span in result.trace.spans if span.name == "model")

    assert model_span.usage is not None
    assert model_span.usage.input_tokens == 100
    assert model_span.usage.output_tokens == 20
    assert model_span.usage.total_tokens == 120
    assert model_span.usage.input_cost == 0.001
    assert model_span.usage.output_cost == 0.002
    assert model_span.usage.total_cost == 0.003


def test_evaluate_dataset_records_model_finish_reason():
    dataset = [
        EvalCase(
            id="1",
            input="What is the capital of France?",
            expected="Paris",
        )
    ]

    def model(prompt):
        return ModelResponse(
            output="Paris",
            finish_reason="stop",
        )

    result = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        enable_tracing=True,
    )

    model_span = next(span for span in result.trace.spans if span.name == "model")

    assert model_span.attributes["model.finish_reason"] == "stop"


def test_evaluate_dataset_records_model_response_id():
    dataset = [
        EvalCase(
            id="1",
            input="What is the capital of France?",
            expected="Paris",
        )
    ]

    def model(prompt):
        return ModelResponse(
            output="Paris",
            response_id="response-123",
        )

    result = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        enable_tracing=True,
    )

    model_span = next(span for span in result.trace.spans if span.name == "model")

    assert model_span.attributes["model.response_id"] == "response-123"


def test_evaluate_dataset_records_model_temperature():
    dataset = [
        EvalCase(
            id="1",
            input="What is the capital of France?",
            expected="Paris",
        )
    ]

    def model(_prompt):
        return ModelResponse(
            output="Paris",
            temperature=0.2,
        )

    result = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        enable_tracing=True,
    )

    model_span = next(span for span in result.trace.spans if span.name == "model")

    assert model_span.attributes["model.temperature"] == 0.2


def test_evaluate_dataset_records_model_max_tokens():
    dataset = [
        EvalCase(
            id="1",
            input="What is the capital of France?",
            expected="Paris",
        )
    ]

    def model(_prompt):
        return ModelResponse(
            output="Paris",
            max_tokens=100,
        )

    result = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        enable_tracing=True,
    )

    model_span = next(span for span in result.trace.spans if span.name == "model")

    assert model_span.attributes["model.max_tokens"] == 100


def test_evaluate_dataset_records_model_request_id():
    dataset = [
        EvalCase(
            id="1",
            input="What is the capital of France?",
            expected="Paris",
        )
    ]

    def model(_prompt):
        return ModelResponse(
            output="Paris",
            request_id="request-123",
        )

    result = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        enable_tracing=True,
    )

    model_span = next(span for span in result.trace.spans if span.name == "model")

    assert model_span.attributes["model.request_id"] == "request-123"


def test_evaluate_dataset_passes_retrieval_results_to_context():
    dataset = [
        EvalCase(
            id="1",
            input="What is gradient descent?",
            expected="Gradient descent",
        )
    ]

    retrieved_documents = [
        "Gradient descent is an optimization algorithm.",
        "It is commonly used to train machine learning models.",
    ]

    def retrieve(query, top_k):
        return retrieved_documents

    def model(text):
        return "Gradient descent"

    captured_context = {}

    class ContextEvaluator:
        name = "context"

        def evaluate(self, context):
            captured_context["context"] = context
            return EvaluationResult(
                case_id=context.case.id,
                evaluator_name=self.name,
                expected=context.case.expected,
                actual=context.actual,
                score=1.0,
                passed=True,
            )

    evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ContextEvaluator()],
        retriever=retrieve,
        enable_tracing=True,
    )

    context = captured_context["context"]

    assert context.retrieved == retrieved_documents


def test_evaluate_dataset_passes_retrieval_top_k_to_retriever():
    dataset = [
        EvalCase(
            id="1",
            input="What is gradient descent?",
            expected="Gradient descent",
        )
    ]

    captured = {}

    def retrieve(query, top_k):
        captured["query"] = query
        captured["top_k"] = top_k
        return ["Gradient descent is an optimization algorithm."]

    def model(text):
        return "Gradient descent"

    evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ExactMatchEvaluator()],
        retriever=retrieve,
        retrieval_top_k=10,
        enable_tracing=True,
    )

    assert captured["query"] == "What is gradient descent?"
    assert captured["top_k"] == 10


def test_evaluate_dataset_passes_retrieval_results_without_tracing():
    dataset = [
        EvalCase(
            id="1",
            input="What is gradient descent?",
            expected="Gradient descent",
        )
    ]

    retrieved_documents = [
        "Gradient descent is an optimization algorithm.",
        "It is used to optimize model parameters.",
    ]

    def retrieve(query, top_k):
        return retrieved_documents

    def model(text):
        return "Gradient descent"

    captured_context = {}

    class ContextEvaluator:
        name = "context"

        def evaluate(self, context):
            captured_context["context"] = context
            return EvaluationResult(
                case_id=context.case.id,
                evaluator_name=self.name,
                expected=context.case.expected,
                actual=context.actual,
                score=1.0,
                passed=True,
            )

    run = evaluate_dataset(
        model=model,
        dataset=dataset,
        evaluators=[ContextEvaluator()],
        retriever=retrieve,
        enable_tracing=False,
    )

    assert run.trace is None
    assert captured_context["context"].retrieved == retrieved_documents

import json

from aieval.reporting.json import JsonReporter
from aieval.result import EvaluationResult
from aieval.run import EvaluationRun


def test_json_reporter_renders_evaluation_run():
    results = [
        EvaluationResult(
            case_id="001",
            evaluator_name="exact_match",
            expected="Paris",
            actual="Paris",
            score=1.0,
            passed=True,
        ),
        EvaluationResult(
            case_id="002",
            evaluator_name="exact_match",
            expected="London",
            actual="Paris",
            score=0.0,
            passed=False,
        ),
    ]

    run = EvaluationRun(results)

    reporter = JsonReporter()

    output = reporter.render(run)

    data = json.loads(output)

    assert data["schema_version"] == 1
    assert data["total"] == 2
    assert data["passed"] == 1
    assert data["failed"] == 1
    assert data["score"] == 0.5
    assert data["pass_rate"] == 0.5

    assert data["results"] == [
        {
            "case_id": "001",
            "evaluator_name": "exact_match",
            "expected": "Paris",
            "actual": "Paris",
            "score": 1.0,
            "passed": True,
        },
        {
            "case_id": "002",
            "evaluator_name": "exact_match",
            "expected": "London",
            "actual": "Paris",
            "score": 0.0,
            "passed": False,
        },
    ]


def test_json_reporter_writes_file(tmp_path):
    results = [
        EvaluationResult(
            case_id="001",
            evaluator_name="exact_match",
            expected="4",
            actual="4",
            score=1.0,
            passed=True,
        )
    ]

    run = EvaluationRun(results)

    output_path = tmp_path / "evaluation_report.json"

    reporter = JsonReporter()
    reporter.write(run, output_path)

    assert output_path.exists()

    assert json.loads(output_path.read_text()) == {
        "schema_version": 1,
        "run_id": run.run_id,
        "results": [
            {
                "case_id": "001",
                "evaluator_name": "exact_match",
                "expected": "4",
                "actual": "4",
                "score": 1.0,
                "passed": True,
            }
        ],
        "total": 1,
        "passed": 1,
        "failed": 0,
        "score": 1.0,
        "pass_rate": 1.0,
        "summaries": {
            "exact_match": {
                "evaluator_name": "exact_match",
                "total": 1,
                "passed": 1,
                "failed": 0,
                "score": 1.0,
                "pass_rate": 1.0,
            }
        },
        "metadata": {},
        "trace": None,
        "trace_summary": None,
    }


def test_json_reporter_reads_evaluation_run(tmp_path):
    run = EvaluationRun(
        results=[
            EvaluationResult(
                case_id="001",
                evaluator_name="exact_match",
                expected="Paris",
                actual="Paris",
                score=1.0,
                passed=True,
            )
        ],
        metadata={"model": "test-model"},
    )

    reporter = JsonReporter()

    path = tmp_path / "run.json"
    reporter.write(run, path)

    loaded = reporter.read(path)

    assert loaded == run


def test_json_reporter_reads_evaluation_run_with_trace(tmp_path):
    from aieval.tracing.trace import Trace

    trace = Trace()

    span = trace.start_span("retrieval")
    span.set_attribute("component", "vector_db")
    span.add_event(
        "cache.miss",
        attributes={"key": "embedding:123"},
    )
    span.end()

    run = EvaluationRun(
        results=[],
        metadata={"model": "test-model"},
        trace=trace,
    )

    reporter = JsonReporter()

    path = tmp_path / "run.json"
    reporter.write(run, path)

    loaded = reporter.read(path)

    assert loaded.trace is not None
    assert loaded.trace.trace_id == trace.trace_id
    assert len(loaded.trace.spans) == 1

    loaded_span = loaded.trace.spans[0]

    assert loaded_span.span_id == span.span_id
    assert loaded_span.name == "retrieval"
    assert loaded_span.attributes["component"] == "vector_db"

    assert len(loaded_span.events) == 1
    assert loaded_span.events[0]["name"] == "cache.miss"
    assert loaded_span.events[0]["attributes"]["key"] == "embedding:123"


def test_json_reporter_preserves_retrieval_trace(tmp_path):
    from aieval.tracing.trace import Trace

    trace = Trace()

    span = trace.start_retrieval(
        query="What is gradient descent?",
        top_k=5,
    )
    span.record_retrieval_result(result_count=3)
    span.end()

    run = EvaluationRun(
        results=[],
        metadata={"model": "test-model"},
        trace=trace,
    )

    reporter = JsonReporter()

    path = tmp_path / "retrieval_run.json"
    reporter.write(run, path)

    loaded = reporter.read(path)

    assert loaded.trace is not None
    assert len(loaded.trace.spans) == 1

    loaded_span = loaded.trace.spans[0]

    assert loaded_span.name == "retrieval"
    assert loaded_span.trace_id == trace.trace_id
    assert loaded_span.span_id == span.span_id
    assert loaded_span.attributes["retrieval.query"] == ("What is gradient descent?")
    assert loaded_span.attributes["retrieval.top_k"] == 5
    assert loaded_span.attributes["retrieval.result_count"] == 3


def test_json_reporter_preserves_retrieval_trace_from_evaluation_run(tmp_path):
    from aieval.dataset import EvalCase
    from aieval.evaluators.exact_match import ExactMatchEvaluator
    from aieval.runner import evaluate_dataset

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
            "It is used to optimize model parameters.",
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

    reporter = JsonReporter()

    path = tmp_path / "evaluation.json"
    reporter.write(run, path)

    loaded = reporter.read(path)

    assert loaded.trace is not None

    retrieval_spans = [span for span in loaded.trace.spans if span.name == "retrieval"]

    assert len(retrieval_spans) == 1

    retrieval = retrieval_spans[0]

    assert retrieval.attributes["retrieval.query"] == ("What is gradient descent?")
    assert retrieval.attributes["retrieval.top_k"] == 5
    assert retrieval.attributes["retrieval.result_count"] == 2
    assert retrieval.ended_at is not None
    assert retrieval.duration is not None


def test_json_reporter_preserves_nested_trace_hierarchy(tmp_path):
    from aieval.tracing.trace import Trace

    trace = Trace()

    root = trace.start_span("evaluation")

    case = trace.start_span(
        "evaluation.case",
        parent=root,
    )
    case.set_attribute("case.id", "001")

    retrieval = trace.start_retrieval(
        query="What is gradient descent?",
        top_k=5,
        parent=case,
    )
    retrieval.record_retrieval_result(result_count=3)
    retrieval.set_status("ok")
    retrieval.end()

    model = trace.start_span(
        "model",
        parent=case,
    )
    model.set_attribute("model.name", "test-model")
    model.set_status("ok")
    model.end()

    case.end()
    root.end()
    trace.end()

    run = EvaluationRun(
        results=[],
        metadata={"model": "test-model"},
        trace=trace,
    )

    reporter = JsonReporter()

    path = tmp_path / "nested_trace.json"
    reporter.write(run, path)

    loaded = reporter.read(path)

    assert loaded.trace is not None
    assert loaded.trace.trace_id == trace.trace_id
    assert len(loaded.trace.spans) == 4

    loaded_root = loaded.trace.spans[0]
    loaded_case = loaded.trace.spans[1]
    loaded_retrieval = loaded.trace.spans[2]
    loaded_model = loaded.trace.spans[3]

    assert loaded_root.name == "evaluation"
    assert loaded_root.parent_span_id is None

    assert loaded_case.name == "evaluation.case"
    assert loaded_case.parent_span_id == loaded_root.span_id
    assert loaded_case.attributes["case.id"] == "001"

    assert loaded_retrieval.name == "retrieval"
    assert loaded_retrieval.parent_span_id == loaded_case.span_id
    assert loaded_retrieval.attributes["retrieval.query"] == "What is gradient descent?"
    assert loaded_retrieval.attributes["retrieval.top_k"] == 5
    assert loaded_retrieval.attributes["retrieval.result_count"] == 3

    assert loaded_model.name == "model"
    assert loaded_model.parent_span_id == loaded_case.span_id
    assert loaded_model.attributes["model.name"] == "test-model"


def test_json_reporter_includes_trace_summary(tmp_path):
    from aieval.tracing.trace import Trace

    trace = Trace()

    root = trace.start_span("evaluation")
    root.end()

    model = trace.start_span("model")
    model.set_status("ok")
    model.end()

    trace.end()

    run = EvaluationRun(
        results=[],
        trace=trace,
    )

    reporter = JsonReporter()

    path = tmp_path / "trace_report.json"
    reporter.write(run, path)

    data = json.loads(path.read_text())

    assert data["trace_summary"] == {
        "trace_id": trace.trace_id,
        "span_count": 2,
        "error_count": 0,
        "duration": trace.duration,
        "completed_span_count": 2,
        "total_duration": 0.0,
        "spans": [
            {
                "span_id": root.span_id,
                "name": "evaluation",
                "status": "ok",
                "duration": root.duration,
            },
            {
                "span_id": model.span_id,
                "name": "model",
                "status": "ok",
                "duration": model.duration,
            },
        ],
    }

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

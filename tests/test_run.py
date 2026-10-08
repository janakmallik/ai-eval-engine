from aieval.result import EvaluationResult
from aieval.run import EvaluationRun
from aieval.tracing.trace import Trace
from aieval.tracing.usage import ModelUsage


def test_evaluation_run():

    results = [
        EvaluationResult(
            case_id="001",
            evaluator_name="exact_match",
            expected="4",
            actual="4",
            score=1.0,
            passed=True,
        ),
        EvaluationResult(
            case_id="002",
            evaluator_name="exact_match",
            expected="Paris",
            actual="Paris",
            score=1.0,
            passed=True,
        ),
        EvaluationResult(
            case_id="003",
            evaluator_name="exact_match",
            expected="5",
            actual="6",
            score=0.0,
            passed=False,
        ),
    ]

    run = EvaluationRun(results)

    assert run.total == 3
    assert run.passed == 2
    assert run.failed == 1
    assert run.score == 2 / 3


def test_evaluation_run_pass_rate():

    results = [
        EvaluationResult(
            case_id="001",
            evaluator_name="exact_match",
            expected="4",
            actual="4",
            score=1.0,
            passed=True,
        ),
        EvaluationResult(
            case_id="002",
            evaluator_name="exact_match",
            expected="5",
            actual="6",
            score=0.0,
            passed=False,
        ),
    ]

    run = EvaluationRun(results)

    assert run.pass_rate == 0.5


def test_evaluation_run_filters_by_evaluator():

    results = [
        EvaluationResult(
            case_id="001",
            evaluator_name="exact_match",
            expected="4",
            actual="4",
            score=1.0,
            passed=True,
        ),
        EvaluationResult(
            case_id="001",
            evaluator_name="length",
            expected="4",
            actual="4",
            score=1.0,
            passed=True,
        ),
        EvaluationResult(
            case_id="002",
            evaluator_name="exact_match",
            expected="5",
            actual="6",
            score=0.0,
            passed=False,
        ),
    ]

    run = EvaluationRun(results)

    exact_match_results = run.by_evaluator("exact_match")

    assert len(exact_match_results) == 2
    assert all(result.evaluator_name == "exact_match" for result in exact_match_results)


def test_evaluation_run_summaries():

    results = [
        EvaluationResult(
            case_id="001",
            evaluator_name="exact_match",
            expected="4",
            actual="4",
            score=1.0,
            passed=True,
        ),
        EvaluationResult(
            case_id="002",
            evaluator_name="exact_match",
            expected="5",
            actual="6",
            score=0.0,
            passed=False,
        ),
        EvaluationResult(
            case_id="003",
            evaluator_name="contains",
            expected="Paris",
            actual="Paris is a city.",
            score=1.0,
            passed=True,
        ),
    ]

    run = EvaluationRun(results)

    summaries = run.summaries()

    assert set(summaries) == {"exact_match", "contains"}

    assert summaries["exact_match"].total == 2
    assert summaries["exact_match"].passed == 1
    assert summaries["exact_match"].failed == 1

    assert summaries["contains"].total == 1
    assert summaries["contains"].passed == 1
    assert summaries["contains"].failed == 0


# Verifies that EvaluationRun.to_dict() serializes the run's aggregate stats (total, passed, failed, score, pass_rate) along with a nested per-evaluator summaries dict, where each summary is itself serialized via EvaluationSummary.to_dict().
def test_evaluation_run_to_dict():

    results = [
        EvaluationResult(
            case_id="001",
            evaluator_name="exact_match",
            expected="4",
            actual="4",
            score=1.0,
            passed=True,
        ),
        EvaluationResult(
            case_id="002",
            evaluator_name="exact_match",
            expected="5",
            actual="6",
            score=0.0,
            passed=False,
        ),
    ]

    run = EvaluationRun(results)

    assert run.to_dict() == {
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
            },
            {
                "case_id": "002",
                "evaluator_name": "exact_match",
                "expected": "5",
                "actual": "6",
                "score": 0.0,
                "passed": False,
            },
        ],
        "total": 2,
        "passed": 1,
        "failed": 1,
        "score": 0.5,
        "pass_rate": 0.5,
        "summaries": {
            "exact_match": {
                "evaluator_name": "exact_match",
                "total": 2,
                "passed": 1,
                "failed": 1,
                "score": 0.5,
                "pass_rate": 0.5,
            }
        },
        "metadata": {},
        "trace": None,
        "trace_summary": None,
    }


def test_evaluation_run_has_stable_id():
    run = EvaluationRun(
        results=[],
        metadata={
            "model_version": "v1",
            "dataset_version": "v1",
        },
    )

    assert run.run_id


def test_identical_evaluation_runs_have_same_id():
    results = []

    run1 = EvaluationRun(
        results=results,
        metadata={
            "model_version": "v1",
            "dataset_version": "v1",
        },
    )

    run2 = EvaluationRun(
        results=results,
        metadata={
            "model_version": "v1",
            "dataset_version": "v1",
        },
    )

    assert run1.run_id == run2.run_id


def test_changed_evaluation_run_has_different_id():
    run1 = EvaluationRun(
        results=[],
        metadata={
            "model_version": "v1",
            "dataset_version": "v1",
        },
    )

    run2 = EvaluationRun(
        results=[],
        metadata={
            "model_version": "v1",
            "dataset_version": "v2",
        },
    )

    assert run1.run_id != run2.run_id


def test_evaluation_run_to_dict_includes_run_id():
    run = EvaluationRun(
        results=[],
        metadata={
            "model_version": "v1",
            "dataset_version": "v1",
        },
    )

    data = run.to_dict()

    assert data["run_id"] == run.run_id


# want an EvaluationRun to be able to carry a trace, First integration test
def test_evaluation_run_can_store_trace():
    trace = Trace()

    run = EvaluationRun(
        results=[],
        trace=trace,
    )

    assert run.trace is trace


def test_evaluation_run_trace_defaults_to_none():
    run = EvaluationRun(results=[])

    assert run.trace is None


def test_evaluation_run_to_dict_includes_trace():
    trace = Trace()

    run = EvaluationRun(
        results=[],
        trace=trace,
    )

    data = run.to_dict()

    assert data["trace"] == trace.to_dict()


def test_evaluation_run_to_dict_trace_defaults_to_none():
    run = EvaluationRun(results=[])

    data = run.to_dict()

    assert data["trace"] is None


def test_evaluation_run_trace_summary():
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

    summary = run.trace_summary()

    assert summary["trace_id"] == trace.trace_id
    assert summary["span_count"] == 2
    assert summary["error_count"] == 0
    assert summary["duration"] == trace.duration


def test_evaluation_run_to_dict_includes_trace_summary():
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

    data = run.to_dict()

    assert data["trace_summary"] == trace.summary()


def test_evaluation_run_exposes_performance_metrics():
    trace = Trace()

    model_span = trace.start_model(
        model="qa-model",
        provider="test",
    )
    model_span.record_usage(
        ModelUsage(
            input_tokens=100,
            output_tokens=50,
            input_cost=0.01,
            output_cost=0.02,
        )
    )
    model_span.end()

    run = EvaluationRun(
        results=[],
        trace=trace,
    )

    metrics = run.performance_metrics()

    assert metrics is not None
    assert metrics.request_count == 1
    assert metrics.total_tokens == 150
    assert metrics.total_cost == 0.03

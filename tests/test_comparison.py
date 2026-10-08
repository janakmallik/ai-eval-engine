import time

import pytest

from aieval.comparison import compare_runs
from aieval.result import EvaluationResult
from aieval.run import EvaluationRun
from aieval.tracing.trace import Trace


def test_compare_runs():
    baseline = EvaluationRun(
        results=[
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
                actual="London",
                score=1.0,
                passed=True,
            ),
        ]
    )

    current = EvaluationRun(
        results=[
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
    )

    comparison = compare_runs(baseline, current)

    assert comparison.baseline_score == 1.0
    assert comparison.current_score == 0.5
    assert comparison.score_delta == -0.5

    assert comparison.baseline_pass_rate == 1.0
    assert comparison.current_pass_rate == 0.5
    assert comparison.pass_rate_delta == -0.5


def test_compare_runs_by_evaluator():
    baseline = EvaluationRun(
        results=[
            EvaluationResult(
                case_id="001",
                evaluator_name="exact_match",
                expected="Paris",
                actual="Paris",
                score=1.0,
                passed=True,
            ),
            EvaluationResult(
                case_id="001",
                evaluator_name="similarity",
                expected="Paris",
                actual="Paris",
                score=1.0,
                passed=True,
            ),
        ]
    )

    current = EvaluationRun(
        results=[
            EvaluationResult(
                case_id="001",
                evaluator_name="exact_match",
                expected="Paris",
                actual="London",
                score=0.0,
                passed=False,
            ),
            EvaluationResult(
                case_id="001",
                evaluator_name="similarity",
                expected="Paris",
                actual="Paris",
                score=1.0,
                passed=True,
            ),
        ]
    )

    comparison = compare_runs(baseline, current)

    assert comparison.evaluator_deltas["exact_match"] == -1.0
    assert comparison.evaluator_deltas["similarity"] == 0.0


def test_compare_runs_handles_added_evaluator():
    baseline = EvaluationRun(
        results=[
            EvaluationResult(
                case_id="001",
                evaluator_name="exact_match",
                expected="Paris",
                actual="Paris",
                score=1.0,
                passed=True,
            ),
        ]
    )

    current = EvaluationRun(
        results=[
            EvaluationResult(
                case_id="001",
                evaluator_name="exact_match",
                expected="Paris",
                actual="Paris",
                score=1.0,
                passed=True,
            ),
            EvaluationResult(
                case_id="001",
                evaluator_name="similarity",
                expected="Paris",
                actual="Paris",
                score=0.9,
                passed=True,
            ),
        ]
    )

    comparison = compare_runs(baseline, current)

    assert comparison.evaluator_deltas["exact_match"] == 0.0
    assert "similarity" in comparison.added_evaluators


def test_compare_runs_includes_latency_cost_and_error_rate():
    baseline = EvaluationRun(
        results=[
            EvaluationResult(
                case_id="1",
                evaluator_name="exact_match",
                expected="Paris",
                actual="Paris",
                score=1.0,
                passed=True,
            )
        ],
        metadata={
            "latency": 0.40,
            "cost": 0.01,
            "error_rate": 0.02,
        },
    )

    current = EvaluationRun(
        results=[
            EvaluationResult(
                case_id="1",
                evaluator_name="exact_match",
                expected="Paris",
                actual="Paris",
                score=1.0,
                passed=True,
            )
        ],
        metadata={
            "latency": 0.60,
            "cost": 0.015,
            "error_rate": 0.05,
        },
    )

    comparison = compare_runs(baseline, current)

    assert comparison.baseline_latency == 0.40
    assert comparison.current_latency == 0.60
    assert comparison.latency_delta == pytest.approx(0.20)

    assert comparison.baseline_cost == 0.01
    assert comparison.current_cost == 0.015
    assert comparison.cost_delta == pytest.approx(0.005)

    assert comparison.baseline_error_rate == 0.02
    assert comparison.current_error_rate == 0.05
    assert comparison.error_rate_delta == pytest.approx(0.03)


def test_compare_runs_uses_trace_metrics():
    from aieval.tracing.trace import Trace
    from aieval.tracing.usage import ModelUsage

    baseline_trace = Trace()
    baseline_model = baseline_trace.start_model(
        model="model",
        provider="test",
    )
    baseline_model.record_usage(
        ModelUsage(
            input_tokens=100,
            output_tokens=50,
            input_cost=0.001,
            output_cost=0.002,
        )
    )
    baseline_model.end()
    baseline_trace.end()

    current_trace = Trace()
    current_model = current_trace.start_model(
        model="model",
        provider="test",
    )
    current_model.record_usage(
        ModelUsage(
            input_tokens=200,
            output_tokens=100,
            input_cost=0.002,
            output_cost=0.004,
        )
    )
    current_model.end()
    current_trace.end()

    baseline = EvaluationRun(
        results=[
            EvaluationResult(
                case_id="1",
                evaluator_name="exact_match",
                expected="Paris",
                actual="Paris",
                score=1.0,
                passed=True,
            )
        ],
        trace=baseline_trace,
    )

    current = EvaluationRun(
        results=[
            EvaluationResult(
                case_id="1",
                evaluator_name="exact_match",
                expected="Paris",
                actual="Paris",
                score=1.0,
                passed=True,
            )
        ],
        trace=current_trace,
    )

    comparison = compare_runs(baseline, current)

    assert comparison.baseline_cost == pytest.approx(0.003)
    assert comparison.current_cost == pytest.approx(0.006)
    assert comparison.cost_delta == pytest.approx(0.003)

    assert comparison.baseline_latency == pytest.approx(baseline_trace.duration)
    assert comparison.current_latency == pytest.approx(current_trace.duration)


def test_compare_runs_uses_trace_error_rate():
    from aieval.tracing.trace import Trace

    baseline_trace = Trace()

    baseline_success = baseline_trace.start_span("success")
    baseline_success.end()

    baseline_error = baseline_trace.start_span("error")
    baseline_error.status = "error"
    baseline_error.end()

    baseline_trace.end()

    current_trace = Trace()

    current_success_1 = current_trace.start_span("success")
    current_success_1.end()

    current_success_2 = current_trace.start_span("success")
    current_success_2.end()

    current_error = current_trace.start_span("error")
    current_error.status = "error"
    current_error.end()

    current_trace.end()

    baseline = EvaluationRun(
        results=[],
        metadata={"error_rate": 0.99},
        trace=baseline_trace,
    )

    current = EvaluationRun(
        results=[],
        metadata={"error_rate": 0.99},
        trace=current_trace,
    )

    comparison = compare_runs(baseline, current)

    assert comparison.baseline_error_rate == pytest.approx(0.5)
    assert comparison.current_error_rate == pytest.approx(1 / 3)
    assert comparison.error_rate_delta == pytest.approx(1 / 3 - 0.5)


def test_compare_runs_uses_model_request_metrics():
    baseline_trace = Trace()

    with baseline_trace.start_span("evaluation"):
        baseline_model = baseline_trace.start_model(
            model="test-model",
            provider="test",
        )
        time.sleep(0.001)
        baseline_model.end()

        baseline_retrieval = baseline_trace.start_retrieval(
            query="test",
            top_k=1,
        )
        baseline_retrieval.set_status("error", "retrieval failed")
        baseline_retrieval.end()

    current_trace = Trace()

    with current_trace.start_span("evaluation"):
        current_model = current_trace.start_model(
            model="test-model",
            provider="test",
        )
        current_model.set_status("error", "model failed")
        time.sleep(0.002)
        current_model.end()

        current_retrieval = current_trace.start_retrieval(
            query="test",
            top_k=1,
        )
        current_retrieval.end()

    baseline = EvaluationRun(
        results=[],
        trace=baseline_trace,
    )

    current = EvaluationRun(
        results=[],
        trace=current_trace,
    )

    baseline_metrics = baseline_trace.performance_metrics()
    current_metrics = current_trace.performance_metrics()

    comparison = compare_runs(baseline, current)

    assert comparison.baseline_latency == pytest.approx(
        baseline_metrics.average_latency
    )
    assert comparison.current_latency == pytest.approx(current_metrics.average_latency)
    assert comparison.latency_delta == pytest.approx(
        current_metrics.average_latency - baseline_metrics.average_latency
    )

    assert comparison.baseline_error_rate == 0.0
    assert comparison.current_error_rate == 1.0

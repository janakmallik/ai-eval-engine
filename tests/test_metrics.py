from aieval.metrics import PerformanceMetrics


def test_performance_metrics_calculates_derived_metrics():
    metrics = PerformanceMetrics.from_latencies(
        request_count=4,
        successful_request_count=3,
        error_count=1,
        input_tokens=400,
        output_tokens=200,
        total_cost=0.08,
        latencies=[0.1, 0.2, 0.3, 0.4],
    )

    assert metrics.request_count == 4
    assert metrics.successful_request_count == 3
    assert metrics.error_count == 1
    assert metrics.error_rate == 0.25

    assert metrics.input_tokens == 400
    assert metrics.output_tokens == 200
    assert metrics.total_tokens == 600
    assert metrics.average_tokens == 150

    assert metrics.total_cost == 0.08
    assert metrics.cost_per_request == 0.02

    assert metrics.average_latency == 0.25
    assert metrics.p50_latency == 0.25
    assert metrics.p95_latency == 0.385


def test_empty_performance_metrics_are_zero():
    metrics = PerformanceMetrics.from_latencies(
        request_count=0,
        successful_request_count=0,
        error_count=0,
        input_tokens=0,
        output_tokens=0,
        total_cost=0.0,
        latencies=[],
    )

    assert metrics.error_rate == 0.0
    assert metrics.average_tokens == 0.0
    assert metrics.cost_per_request == 0.0
    assert metrics.average_latency == 0.0
    assert metrics.p50_latency == 0.0
    assert metrics.p95_latency == 0.0


def test_performance_metrics_serializes():
    metrics = PerformanceMetrics.from_latencies(
        request_count=1,
        successful_request_count=1,
        error_count=0,
        input_tokens=10,
        output_tokens=20,
        total_cost=0.01,
        latencies=[0.5],
        retry_count=2,
        timeout_count=1,
    )

    data = metrics.to_dict()

    assert data["total_tokens"] == 30
    assert data["average_tokens"] == 30
    assert data["cost_per_request"] == 0.01
    assert data["p50_latency"] == 0.5
    assert data["p95_latency"] == 0.5
    assert data["retry_count"] == 2
    assert data["timeout_count"] == 1

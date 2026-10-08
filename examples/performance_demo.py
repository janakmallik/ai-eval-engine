import time

from aieval import EvaluationRun, PerformanceMetrics, Trace
from aieval.tracing.usage import ModelUsage


def main() -> None:
    trace = Trace()

    first = trace.start_model(
        model="qa-model",
        provider="demo",
    )

    time.sleep(0.020)

    first.record_usage(
        ModelUsage(
            input_tokens=120,
            output_tokens=80,
            input_cost=0.0012,
            output_cost=0.0024,
        )
    )

    first.end()

    second = trace.start_model(
        model="qa-model",
        provider="demo",
    )

    time.sleep(0.040)

    second.record_usage(
        ModelUsage(
            input_tokens=200,
            output_tokens=100,
            input_cost=0.0020,
            output_cost=0.0030,
        )
    )

    second.end()

    second.record_retry()

    run = EvaluationRun(
        results=[],
        trace=trace,
    )

    metrics: PerformanceMetrics = run.performance_metrics()

    if metrics is None:
        raise RuntimeError("Performance metrics were not generated")

    print("V4 Performance Metrics")
    print("----------------------")
    print(f"Requests:          {metrics.request_count}")
    print(f"Errors:            {metrics.error_count}")
    print(f"Error rate:        {metrics.error_rate:.2%}")
    print(f"Input tokens:      {metrics.input_tokens}")
    print(f"Output tokens:     {metrics.output_tokens}")
    print(f"Total tokens:      {metrics.total_tokens}")
    print(f"Average tokens:    {metrics.average_tokens:.2f}")
    print(f"Total cost:        ${metrics.total_cost:.4f}")
    print(f"Cost/request:      ${metrics.cost_per_request:.4f}")
    print(f"Average latency:   {metrics.average_latency:.6f}s")
    print(f"P50 latency:       {metrics.p50_latency:.6f}s")
    print(f"P95 latency:       {metrics.p95_latency:.6f}s")
    print(f"Retries:           {metrics.retry_count}")
    print(f"Timeouts:          {metrics.timeout_count}")


if __name__ == "__main__":
    main()

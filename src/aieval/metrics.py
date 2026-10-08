from dataclasses import dataclass


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0

    if not 0 <= percentile <= 100:
        raise ValueError("percentile must be between 0 and 100")

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * (percentile / 100)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)

    if lower == upper:
        return values[lower]

    weight = position - lower

    return values[lower] + (values[upper] - values[lower]) * weight


@dataclass(frozen=True)
class PerformanceMetrics:
    request_count: int
    successful_request_count: int
    error_count: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_cost: float
    average_latency: float
    p50_latency: float
    p95_latency: float
    retry_count: int = 0
    timeout_count: int = 0

    @property
    def error_rate(self) -> float:
        if self.request_count == 0:
            return 0.0

        return self.error_count / self.request_count

    @property
    def average_tokens(self) -> float:
        if self.request_count == 0:
            return 0.0

        return self.total_tokens / self.request_count

    @property
    def cost_per_request(self) -> float:
        if self.request_count == 0:
            return 0.0

        return self.total_cost / self.request_count

    def to_dict(self) -> dict[str, object]:
        return {
            "request_count": self.request_count,
            "successful_request_count": self.successful_request_count,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "average_tokens": self.average_tokens,
            "total_cost": self.total_cost,
            "cost_per_request": self.cost_per_request,
            "average_latency": self.average_latency,
            "p50_latency": self.p50_latency,
            "p95_latency": self.p95_latency,
            "retry_count": self.retry_count,
            "timeout_count": self.timeout_count,
        }

    @classmethod
    def from_latencies(
        cls,
        *,
        request_count: int,
        successful_request_count: int,
        error_count: int,
        input_tokens: int,
        output_tokens: int,
        total_cost: float,
        latencies: list[float],
        retry_count: int = 0,
        timeout_count: int = 0,
    ) -> "PerformanceMetrics":
        total_tokens = input_tokens + output_tokens

        average_latency = sum(latencies) / len(latencies) if latencies else 0.0

        return cls(
            request_count=request_count,
            successful_request_count=successful_request_count,
            error_count=error_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            total_cost=total_cost,
            average_latency=average_latency,
            p50_latency=_percentile(latencies, 50),
            p95_latency=_percentile(latencies, 95),
            retry_count=retry_count,
            timeout_count=timeout_count,
        )

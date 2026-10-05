# Span is a small, self-timing, uniquely-identified unit of work that knows when it
# started, when it ended (or that it hasn't), how long it took, and which parent it
# belongs to — the building block of traces.

import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from aieval.tracing.usage import ModelUsage


@dataclass
class Span:
    name: str
    span_id: str = field(default_factory=lambda: uuid4().hex)
    trace_id: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    parent_span_id: str | None = None
    status: str = "unset"
    status_message: str | None = None
    attributes: dict[str, object] = field(default_factory=dict)
    events: list[dict[str, object]] = field(default_factory=list)
    usage: ModelUsage | None = None

    @property
    def duration(self) -> float | None:
        if self.ended_at is None:
            return None

        return (self.ended_at - self.started_at).total_seconds()

    def __enter__(self) -> "Span":
        return self

    # no exception → status stays "unset"
    # exception + "unset" → automatically "error"
    # exception + "ok" → stays "ok"
    # exception + existing "error" → preserves its existing message
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            self.record_exception(exc_value)

            if self.status == "unset":
                self.set_status("error", str(exc_value))
        elif self.status == "unset":
            self.set_status("ok")

        self.end()

    def end(self) -> None:
        if self.ended_at is not None:
            return

        if self.status == "unset":
            self.status = "ok"

        self.ended_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "trace_id": self.trace_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": (
                self.ended_at.isoformat() if self.ended_at is not None else None
            ),
            "duration": self.duration,
            "parent_span_id": self.parent_span_id,
            "status": self.status,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "usage": self.usage.to_dict() if self.usage is not None else None,
            "events": [
                {
                    "name": event["name"],
                    "timestamp": event["timestamp"].isoformat(),
                    "attributes": event["attributes"],
                }
                for event in self.events
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Span":
        started_at = datetime.fromisoformat(data["started_at"])
        ended_at = (
            datetime.fromisoformat(data["ended_at"])
            if data["ended_at"] is not None
            else None
        )

        return cls(
            name=data["name"],
            span_id=data["span_id"],
            trace_id=data.get("trace_id"),
            started_at=started_at,
            ended_at=ended_at,
            parent_span_id=data.get("parent_span_id"),
            status=data.get("status", "unset"),
            status_message=data.get("status_message"),
            attributes=dict(data.get("attributes", {})),
            events=[
                {
                    "name": event["name"],
                    "timestamp": datetime.fromisoformat(event["timestamp"]),
                    "attributes": dict(event.get("attributes", {})),
                }
                for event in data.get("events", [])
            ],
            usage=(
                ModelUsage(
                    input_tokens=data["usage"]["input_tokens"],
                    output_tokens=data["usage"]["output_tokens"],
                    input_cost=data["usage"]["input_cost"],
                    output_cost=data["usage"]["output_cost"],
                )
                if data.get("usage") is not None
                else None
            ),
        )

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def record_input(self, input_text: str) -> None:
        self.set_attribute("model.input", input_text)

    def record_output(self, output_text: str) -> None:
        self.set_attribute("model.output", output_text)

    def record_token_usage(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self.set_attribute("model.input_tokens", input_tokens)
        self.set_attribute("model.output_tokens", output_tokens)
        self.set_attribute(
            "model.total_tokens",
            input_tokens + output_tokens,
        )

    def record_usage(self, usage: ModelUsage) -> None:
        self.usage = usage

        self.record_token_usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        self.record_cost(
            input_cost=usage.input_cost,
            output_cost=usage.output_cost,
        )

    def record_finish_reason(self, finish_reason: str) -> None:
        self.set_attribute("model.finish_reason", finish_reason)

    def record_response_id(self, response_id: str) -> None:
        self.set_attribute("model.response_id", response_id)

    def record_request_id(self, request_id: str) -> None:
        self.set_attribute("model.request_id", request_id)

    def record_temperature(self, temperature: float) -> None:
        self.set_attribute("model.temperature", temperature)

    def record_max_tokens(self, max_tokens: int) -> None:
        self.set_attribute("model.max_tokens", max_tokens)

    def record_cost(
        self,
        input_cost: float,
        output_cost: float,
    ) -> None:
        self.set_attribute("model.input_cost", input_cost)
        self.set_attribute("model.output_cost", output_cost)
        self.set_attribute("model.total_cost", input_cost + output_cost)

    def record_retrieval_result(self, result_count: int) -> None:
        self.set_attribute("retrieval.result_count", result_count)

    def record_tool_result(self, result_count: int) -> None:
        self.set_attribute("tool.result_count", result_count)

    # once a span has ended, subsequent span mutations such as adding events should be ignored.
    def add_event(
        self,
        name: str,
        attributes: dict | None = None,
    ) -> None:
        if self.ended_at is not None:
            return

        self.events.append(
            {
                "name": name,
                "timestamp": datetime.now(timezone.utc),
                "attributes": dict(attributes or {}),
            }
        )

    def record_exception(self, exc: BaseException) -> None:
        self.add_event(
            "exception",
            attributes={
                "exception.type": type(exc).__name__,
                "exception.message": str(exc),
                "exception.stacktrace": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            },
        )

    def set_status(
        self,
        status: str,
        status_message: str | None = None,
    ) -> None:
        if self.ended_at is not None:
            return

        self.status = status
        self.status_message = status_message

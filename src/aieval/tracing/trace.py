# Trace is a container for Spans — it has its own ID and timing, holds a list of
# spans, and offers start_span() as a convenience to create-and-register spans in one
# step, so you can build a tree of operations all sharing one trace_id.

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Self
from uuid import uuid4

from aieval.tracing.span import Span


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    spans: list[Span] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    attributes: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata = dict(self.metadata)
        self.attributes = dict(self.attributes)

    @property
    def duration(self) -> float | None:
        if self.ended_at is None:
            return None

        return (self.ended_at - self.started_at).total_seconds()

    def summary(self) -> dict:
        durations = [span.duration for span in self.spans if span.duration is not None]

        return {
            "trace_id": self.trace_id,
            "span_count": len(self.spans),
            "error_count": sum(span.status == "error" for span in self.spans),
            "duration": self.duration,
            "completed_span_count": sum(
                span.ended_at is not None for span in self.spans
            ),
            "total_duration": sum(durations),
            "spans": [
                {
                    "span_id": span.span_id,
                    "name": span.name,
                    "status": span.status,
                    "duration": span.duration,
                }
                for span in self.spans
            ],
        }

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.end()

    def end(self) -> None:
        self.ended_at = datetime.now(UTC)

    def start_span(
        self,
        name: str,
        parent_span_id: str | None = None,
        parent: Span | None = None,
    ) -> Span:
        if parent is not None:
            parent_span_id = parent.span_id

        span = Span(
            name=name,
            parent_span_id=parent_span_id,
            trace_id=self.trace_id,
        )

        self.spans.append(span)

        return span

    def start_retrieval(
        self,
        query: str,
        top_k: int,
        parent_span_id: str | None = None,
        parent: Span | None = None,
    ) -> Span:
        span = self.start_span(
            "retrieval",
            parent_span_id=parent_span_id,
            parent=parent,
        )

        span.set_attribute("retrieval.query", query)
        span.set_attribute("retrieval.top_k", top_k)

        return span

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": (
                self.ended_at.isoformat() if self.ended_at is not None else None
            ),
            "duration": self.duration,
            "metadata": self.metadata,
            "attributes": self.attributes,
            "spans": [span.to_dict() for span in self.spans],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Trace":
        started_at = datetime.fromisoformat(data["started_at"])
        ended_at = (
            datetime.fromisoformat(data["ended_at"])
            if data["ended_at"] is not None
            else None
        )

        return cls(
            trace_id=data["trace_id"],
            started_at=started_at,
            ended_at=ended_at,
            spans=[Span.from_dict(span) for span in data.get("spans", [])],
            metadata=dict(data.get("metadata", {})),
            attributes=dict(data.get("attributes", {})),
        )

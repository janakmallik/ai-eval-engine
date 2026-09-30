# Trace is a container for Spans — it has its own ID and timing, holds a list of
# spans, and offers start_span() as a convenience to create-and-register spans in one
# step, so you can build a tree of operations all sharing one trace_id.

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from aieval.tracing.span import Span


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    ended_at: datetime | None = None
    spans: list[Span] = field(default_factory=list)

    @property
    def duration(self) -> float | None:
        if self.ended_at is None:
            return None

        return (self.ended_at - self.started_at).total_seconds()

    def end(self) -> None:
        self.ended_at = datetime.now(timezone.utc)

    def start_span(
        self,
        name: str,
        parent_span_id: str | None = None,
    ) -> Span:
        span = Span(
            name=name,
            parent_span_id=parent_span_id,
        )

        self.spans.append(span)

        return span

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": (
                self.ended_at.isoformat()
                if self.ended_at is not None
                else None
            ),
            "duration": self.duration,
            "spans": [span.to_dict() for span in self.spans],
        }
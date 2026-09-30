# Span is a small, self-timing, uniquely-identified unit of work that knows when it
# started, when it ended (or that it hasn't), how long it took, and which parent it
# belongs to — the building block of traces.

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


@dataclass
class Span:
    name: str
    span_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    ended_at: datetime | None = None
    parent_span_id: str | None = None

    @property
    def duration(self) -> float | None:
        if self.ended_at is None:
            return None

        return (self.ended_at - self.started_at).total_seconds()

    def end(self) -> None:
        self.ended_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "started_at": self.started_at.isoformat(),
            "ended_at": (
                self.ended_at.isoformat()
                if self.ended_at is not None
                else None
            ),
            "duration": self.duration,
            "parent_span_id": self.parent_span_id,
        }
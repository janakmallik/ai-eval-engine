from datetime import datetime, timezone

from aieval.tracing.trace import Trace


def test_trace_has_id():
    trace = Trace()

    assert trace.trace_id
    assert isinstance(trace.trace_id, str)


def test_trace_has_start_timestamp():
    trace = Trace()

    assert trace.started_at is not None
    assert trace.started_at.tzinfo == timezone.utc


def test_trace_is_not_ended_when_created():
    trace = Trace()

    assert trace.ended_at is None


def test_trace_can_be_ended():
    trace = Trace()

    trace.end()

    assert trace.ended_at is not None
    assert trace.ended_at.tzinfo == timezone.utc


def test_trace_duration_is_none_before_end():
    trace = Trace()

    assert trace.duration is None


def test_trace_duration_is_non_negative_after_end():
    trace = Trace()

    trace.end()

    assert trace.duration is not None
    assert trace.duration >= 0


def test_trace_ids_are_unique():
    first = Trace()
    second = Trace()

    assert first.trace_id != second.trace_id

from aieval.tracing.span import Span


def test_span_has_id():
    span = Span(name="retrieval")

    assert span.span_id
    assert isinstance(span.span_id, str)


def test_span_stores_name():
    span = Span(name="retrieval")

    assert span.name == "retrieval"


def test_span_has_start_timestamp():
    span = Span(name="retrieval")

    assert span.started_at is not None
    assert span.started_at.tzinfo == timezone.utc


def test_span_is_not_ended_when_created():
    span = Span(name="retrieval")

    assert span.ended_at is None


def test_span_can_be_ended():
    span = Span(name="retrieval")

    span.end()

    assert span.ended_at is not None
    assert span.ended_at.tzinfo == timezone.utc


def test_span_duration_is_none_before_end():
    span = Span(name="retrieval")

    assert span.duration is None


def test_span_duration_is_non_negative_after_end():
    span = Span(name="retrieval")

    span.end()

    assert span.duration is not None
    assert span.duration >= 0


def test_span_can_have_parent():
    span = Span(
        name="reranking",
        parent_span_id="parent-123",
    )

    assert span.parent_span_id == "parent-123"


def test_span_parent_is_optional():
    span = Span(name="retrieval")

    assert span.parent_span_id is None
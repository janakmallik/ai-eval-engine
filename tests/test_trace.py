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

def test_trace_starts_span():
    trace = Trace()

    span = trace.start_span("retrieval")

    assert span.name == "retrieval"
    assert span in trace.spans


def test_trace_starts_multiple_spans():
    trace = Trace()

    first = trace.start_span("retrieval")
    second = trace.start_span("llm")

    assert trace.spans == [first, second]


def test_trace_starts_span_without_parent():
    trace = Trace()

    span = trace.start_span("retrieval")

    assert span.parent_span_id is None

def test_trace_starts_child_span():
    trace = Trace()

    parent = trace.start_span("retrieval")
    child = trace.start_span(
        "vector_search",
        parent_span_id=parent.span_id,
    )

    assert child.parent_span_id == parent.span_id
    assert child in trace.spans

def test_trace_to_dict_includes_trace_identity():
    trace = Trace()

    data = trace.to_dict()

    assert data["trace_id"] == trace.trace_id


def test_trace_to_dict_includes_spans():
    trace = Trace()

    span = trace.start_span("retrieval")

    data = trace.to_dict()

    assert len(data["spans"]) == 1
    assert data["spans"][0]["span_id"] == span.span_id
    assert data["spans"][0]["name"] == "retrieval"


def test_trace_to_dict_serializes_timestamps():
    trace = Trace()
    trace.end()

    data = trace.to_dict()

    assert isinstance(data["started_at"], str)
    assert isinstance(data["ended_at"], str)


def test_trace_to_dict_includes_duration():
    trace = Trace()
    trace.end()

    data = trace.to_dict()

    assert data["duration"] is not None

def test_trace_starts_child_span_from_parent():
    trace = Trace()

    parent = trace.start_span("request")
    child = trace.start_span("retrieval", parent=parent)

    assert child.parent_span_id == parent.span_id


def test_child_span_keeps_same_trace_id():
    trace = Trace()

    parent = trace.start_span("request")
    child = trace.start_span("retrieval", parent=parent)

    assert parent.trace_id == trace.trace_id
    assert child.trace_id == trace.trace_id


def test_trace_supports_nested_span_hierarchy():
    trace = Trace()

    request = trace.start_span("request")
    retrieval = trace.start_span("retrieval", parent=request)
    reranking = trace.start_span("reranking", parent=retrieval)

    assert retrieval.parent_span_id == request.span_id
    assert reranking.parent_span_id == retrieval.span_id
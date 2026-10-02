from datetime import datetime, timezone

from aieval.tracing.trace import Trace
from aieval.dataset import EvalCase
from aieval.runner import evaluate_dataset

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

def test_span_stores_attributes():
    trace = Trace()

    span = trace.start_span(
        "retrieval",
    )

    span.set_attribute("model", "embedding-model")
    span.set_attribute("document_count", 5)

    assert span.attributes == {
        "model": "embedding-model",
        "document_count": 5,
    }

def test_span_updates_existing_attribute():
    trace = Trace()

    span = trace.start_span("generation")

    span.set_attribute("model", "model-v1")
    span.set_attribute("model", "model-v2")

    assert span.attributes["model"] == "model-v2"

# want to prove that two spans don't accidentally share the same attributes dictionary.
def test_spans_have_independent_attributes():
    trace = Trace()

    first = trace.start_span("retrieval")
    second = trace.start_span("generation")

    first.set_attribute("model", "embedding-model")

    assert first.attributes == {
        "model": "embedding-model",
    }
    assert second.attributes == {}

# confirms the Span itself now serializes its attributes correctly.
def test_span_to_dict_includes_attributes():
    trace = Trace()

    span = trace.start_span("retrieval")
    span.set_attribute("model", "embedding-model")
    span.set_attribute("document_count", 5)

    data = span.to_dict()

    assert data["attributes"] == {
        "model": "embedding-model",
        "document_count": 5,
    }

# spans carry attributes and parent/child relationships
def test_trace_to_dict_includes_span_attributes():
    trace = Trace()

    span = trace.start_span("retrieval")
    span.set_attribute("model", "embedding-model")

    data = trace.to_dict()

    assert data["spans"][0]["attributes"] == {
        "model": "embedding-model",
    }

# attributes are preserved when the span is nested.
# last test passing means the span metadata layer is now behaving correctly:
# attributes are stored independently, serialized, and preserved through parent/child
# spans.
def test_child_span_to_dict_includes_parent_and_attributes():
    trace = Trace()

    parent = trace.start_span("request")
    child = trace.start_span("retrieval", parent=parent)
    child.set_attribute("model", "embedding-model")

    data = trace.to_dict()

    child_data = next(
        span for span in data["spans"]
        if span["span_id"] == child.span_id
    )

    assert child_data["parent_span_id"] == parent.span_id
    assert child_data["attributes"] == {
        "model": "embedding-model",
    }

def test_span_adds_event():
    trace = Trace()

    span = trace.start_span("retrieval")

    span.add_event("cache_miss")

    assert len(span.events) == 1
    assert span.events[0]["name"] == "cache_miss"

def test_span_adds_event_with_attributes():
    trace = Trace()

    span = trace.start_span("retrieval")

    span.add_event(
        "cache_miss",
        attributes={
            "cache": "redis",
            "key": "user:123",
        },
    )

    assert span.events[0]["name"] == "cache_miss"
    assert span.events[0]["attributes"] == {
        "cache": "redis",
        "key": "user:123",
    }

def test_span_event_without_attributes_uses_empty_dict():
    trace = Trace()

    span = trace.start_span("retrieval")

    span.add_event("cache_miss")

    assert span.events[0]["attributes"] == {}

def test_span_to_dict_serializes_event_timestamp():
    trace = Trace()

    span = trace.start_span("retrieval")
    span.add_event("cache_miss")

    data = span.to_dict()

    assert isinstance(data["events"][0]["timestamp"], str)

def test_span_to_dict_includes_complete_event():
    trace = Trace()

    span = trace.start_span("retrieval")
    span.add_event(
        "cache_miss",
        attributes={
            "cache": "redis",
            "key": "user:123",
        },
    )

    data = span.to_dict()

    event = data["events"][0]

    assert event["name"] == "cache_miss"
    assert isinstance(event["timestamp"], str)
    assert event["attributes"] == {
        "cache": "redis",
        "key": "user:123",
    }

def test_span_preserves_event_order():
    trace = Trace()

    span = trace.start_span("retrieval")

    span.add_event("cache_miss")
    span.add_event("vector_search")
    span.add_event("documents_found")

    assert [event["name"] for event in span.events] == [
        "cache_miss",
        "vector_search",
        "documents_found",
    ]

def test_span_to_dict_preserves_event_order():
    trace = Trace()

    span = trace.start_span("retrieval")

    span.add_event("cache_miss")
    span.add_event("vector_search")
    span.add_event("documents_found")

    data = span.to_dict()

    assert [event["name"] for event in data["events"]] == [
        "cache_miss",
        "vector_search",
        "documents_found",
    ]

# make sure each event keeps its own attributes and doesn't accidentally share the same dictionary.
def test_span_events_have_independent_attributes():
    trace = Trace()

    span = trace.start_span("retrieval")

    span.add_event(
        "cache_miss",
        attributes={"cache": "redis"},
    )
    span.add_event(
        "vector_search",
        attributes={"index": "documents"},
    )

    assert span.events[0]["attributes"] == {
        "cache": "redis",
    }
    assert span.events[1]["attributes"] == {
        "index": "documents",
    }

# verify the entire list of events survives serialization correctly.
def test_span_to_dict_preserves_event_attributes():
    trace = Trace()

    span = trace.start_span("retrieval")

    span.add_event(
        "cache_miss",
        attributes={
            "cache": "redis",
            "key": "user:123",
        },
    )
    span.add_event(
        "vector_search",
        attributes={
            "index": "documents",
            "top_k": 5,
        },
    )

    data = span.to_dict()

    assert data["events"][0]["attributes"] == {
        "cache": "redis",
        "key": "user:123",
    }

    assert data["events"][1]["attributes"] == {
        "index": "documents",
        "top_k": 5,
    }

# protect span mutation after end()
# Once a span has ended, we shouldn't keep modifying it.
def test_span_cannot_add_event_after_end():
    trace = Trace()

    span = trace.start_span("retrieval")
    span.end()

    span.add_event("late_event")

    assert span.events == []

def test_span_status_defaults_to_unset():
    trace = Trace()

    span = trace.start_span("retrieval")

    assert span.status == "unset"

def test_span_status_can_be_updated():
    trace = Trace()

    span = trace.start_span("retrieval")

    span.set_status("ok")

    assert span.status == "ok"

def test_span_to_dict_includes_status():
    trace = Trace()

    span = trace.start_span("retrieval")
    span.set_status("ok")

    data = span.to_dict()

    assert data["status"] == "ok"

def test_span_error_status_stores_message():
    trace = Trace()

    span = trace.start_span("retrieval")

    span.set_status("error", "vector database unavailable")

    assert span.status == "error"
    assert span.status_message == "vector database unavailable"

def test_span_to_dict_includes_status_message():
    trace = Trace()

    span = trace.start_span("retrieval")
    span.set_status("error", "vector database unavailable")

    data = span.to_dict()

    assert data["status"] == "error"
    assert data["status_message"] == "vector database unavailable"

def test_span_cannot_change_status_after_end():
    trace = Trace()

    span = trace.start_span("retrieval")
    span.set_status("ok")

    span.end()
    span.set_status("error", "late failure")

    assert span.status == "ok"
    assert span.status_message is None

def test_span_end_preserves_status():
    trace = Trace()

    span = trace.start_span("retrieval")
    span.set_status("ok")

    span.end()

    assert span.status == "ok"
    assert span.status_message is None

def test_span_context_manager_ends_span():
    trace = Trace()

    with trace.start_span("retrieval") as span:
        assert span.ended_at is None

    assert span.ended_at is not None

# verify it also ends when an exception occurs
def test_span_context_manager_ends_span_on_exception():
    trace = Trace()

    span = trace.start_span("retrieval")

    try:
        with span:
            raise ValueError("retrieval failed")
    except ValueError:
        pass

    assert span.ended_at is not None

# automatically mark exceptions as errors
def test_span_context_manager_marks_exception_as_error():
    trace = Trace()

    span = trace.start_span("retrieval")

    try:
        with span:
            raise ValueError("retrieval failed")
    except ValueError:
        pass

    assert span.status == "error"
    assert span.status_message == "retrieval failed"

# preserve an explicitly set error status
def test_span_context_manager_preserves_existing_error_status():
    trace = Trace()

    span = trace.start_span("retrieval")

    try:
        with span:
            span.set_status("error", "database timeout")
            raise ValueError("different exception")
    except ValueError:
        pass

    assert span.status == "error"
    assert span.status_message == "database timeout"

def test_span_records_exception():
    trace = Trace()

    span = trace.start_span("retrieval")

    try:
        raise ValueError("vector database unavailable")
    except ValueError as exc:
        span.record_exception(exc)

    assert len(span.events) == 1

def test_span_records_exception_details():
    trace = Trace()

    span = trace.start_span("retrieval")

    try:
        raise ValueError("vector database unavailable")
    except ValueError as exc:
        span.record_exception(exc)

    event = span.events[0]

    assert event["name"] == "exception"
    assert event["attributes"]["exception.type"] == "ValueError"
    assert event["attributes"]["exception.message"] == "vector database unavailable"
    assert event["attributes"]["exception.stacktrace"]

# exception event timestamp
# verify that record_exception() also records when the exception happened.
def test_span_records_exception_timestamp():
    trace = Trace()

    span = trace.start_span("retrieval")

    try:
        raise ValueError("vector database unavailable")
    except ValueError as exc:
        span.record_exception(exc)

    event = span.events[0]

    assert isinstance(event["timestamp"], datetime)

# capture the stack trace
# observability feature rather than just another field. When an AI/RAG operation
# fails, the message tells us what failed, while the stack trace helps identify where
# it failed.
def test_span_records_exception_stacktrace():
    trace = Trace()

    span = trace.start_span("retrieval")

    try:
        raise ValueError("vector database unavailable")
    except ValueError as exc:
        span.record_exception(exc)

    event = span.events[0]

    assert "exception.stacktrace" in event["attributes"]
    assert "ValueError: vector database unavailable" in event["attributes"]["exception.stacktrace"]

# don't allow exceptions after span end
# because we already established that a finished span is immutable.
def test_span_cannot_record_exception_after_end():
    trace = Trace()

    span = trace.start_span("retrieval")
    span.end()

    try:
        raise ValueError("late failure")
    except ValueError as exc:
        span.record_exception(exc)

    assert span.events == []

# multiple exceptions, A real RAG/agent span could potentially encounter and record
# more than one exception event.
def test_span_can_record_multiple_exceptions():
    trace = Trace()

    span = trace.start_span("retrieval")

    try:
        raise ValueError("first failure")
    except ValueError as exc:
        span.record_exception(exc)

    try:
        raise TimeoutError("second failure")
    except TimeoutError as exc:
        span.record_exception(exc)

    assert len(span.events) == 2
    assert span.events[0]["attributes"]["exception.type"] == "ValueError"
    assert span.events[0]["attributes"]["exception.message"] == "first failure"
    assert span.events[1]["attributes"]["exception.type"] == "TimeoutError"
    assert span.events[1]["attributes"]["exception.message"] == "second failure"

# serialize exception events, We've tested the in-memory representation. Now let's
# make sure the exception information survives to_dict().
def test_span_to_dict_preserves_exception_details():
    trace = Trace()

    span = trace.start_span("retrieval")

    try:
        raise ValueError("vector database unavailable")
    except ValueError as exc:
        span.record_exception(exc)

    data = span.to_dict()

    event = data["events"][0]

    assert event["name"] == "exception"
    assert event["attributes"]["exception.type"] == "ValueError"
    assert event["attributes"]["exception.message"] == (
        "vector database unavailable"
    )
    assert "exception.stacktrace" in event["attributes"]

# automatic exception recording from the context manager
def test_span_context_manager_records_exception():
    trace = Trace()

    span = trace.start_span("retrieval")

    try:
        with span:
            raise ValueError("retrieval failed")
    except ValueError:
        pass

    assert len(span.events) == 1

    event = span.events[0]

    assert event["name"] == "exception"
    assert event["attributes"]["exception.type"] == "ValueError"
    assert event["attributes"]["exception.message"] == "retrieval failed"

# context manager preserves the exception event, We already tested that it records
# the event. Now make sure ending the span doesn't accidentally remove or mutate it.
def test_span_context_manager_preserves_exception_event_after_end():
    trace = Trace()

    span = trace.start_span("retrieval")

    try:
        with span:
            raise ValueError("retrieval failed")
    except ValueError:
        pass

    assert len(span.events) == 1

    event = span.events[0]

    assert event["name"] == "exception"
    assert event["attributes"]["exception.type"] == "ValueError"
    assert event["attributes"]["exception.message"] == "retrieval failed"

# exception event + status together, already tested both separately. Now let's lock
# down the complete behavior of the context manager.
def test_span_context_manager_records_exception_and_error_status():
    trace = Trace()

    span = trace.start_span("retrieval")

    try:
        with span:
            raise ValueError("vector database unavailable")
    except ValueError:
        pass

    assert span.status == "error"
    assert span.status_message == "vector database unavailable"

    assert len(span.events) == 1

    event = span.events[0]

    assert event["name"] == "exception"
    assert event["attributes"]["exception.type"] == "ValueError"
    assert event["attributes"]["exception.message"] == (
        "vector database unavailable"
    )
    assert "exception.stacktrace" in event["attributes"]

# handled inside span
# record_exception()
# with exits normally
# no duplicate automatic event
def test_span_does_not_record_handled_exception_twice():
    trace = Trace()

    span = trace.start_span("retrieval")

    try:
        with span:
            try:
                raise ValueError("temporary failure")
            except ValueError:
                span.record_exception(
                    ValueError("temporary failure")
                )
    except ValueError:
        pass

    assert len(span.events) == 1

def test_span_records_exception_and_marks_error():
    span = Span(name="evaluator.exploding")

    try:
        with span:
            raise RuntimeError("evaluator crashed")
    except RuntimeError:
        pass

    assert span.status == "error"
    assert span.status_message == "evaluator crashed"
    assert span.ended_at is not None
    assert len(span.events) == 1

    event = span.events[0]

    assert event["name"] == "exception"
    assert event["attributes"]["exception.type"] == "RuntimeError"
    assert event["attributes"]["exception.message"] == "evaluator crashed"

def test_evaluate_dataset_evaluator_span_records_exception():
    dataset = [
        EvalCase(
            id="1",
            input="hello",
            expected="hello",
        )
    ]

    def model(text):
        return text

    class ExplodingEvaluator:
        name = "exploding"

        def evaluate(self, context):
            raise RuntimeError("evaluator crashed")

    evaluator_span = None

    try:
        evaluate_dataset(
            model=model,
            dataset=dataset,
            evaluators=[ExplodingEvaluator()],
            enable_tracing=True,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError")